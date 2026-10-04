"""
Everything drawn on top of the camera image.

The typing arm is drawn as a glowing skeleton (shoulder - elbow - wrist,
no hand bones) with the elbow angle marked and a readout next to
the elbow. The switch hand is drawn as a 21-point hand skeleton: green
while it is a fist (typing on), grey while it is open (typing off). All other feedback lives in the panels around the image.
The HUD paints into two pygame surfaces every frame:

* `ui`   - transparent layer with text, meters and icons.
* `glow` - opaque black layer where bright shapes are added. The GPU
           blurs it and adds it on top of the image, which produces the
           soft neon light around symbols and letters.

The frosted glass panels are not drawn here. The HUD only reports where
they are (`panels`) and the GPU shader renders them, because blurring the
camera behind a panel is far cheaper on the graphics card.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import pygame

from . import draw
from .config import (ACCENT, ACTION_COLORS, DANGER, FONTS_DIR, MIN_ACCEPTABLE_FPS,
                     BONE, SUCCESS, TEXT, TEXT_FAINT, TEXT_MUTED, WARNING)
from .hand_tracker import FINGERTIPS, HAND_CONNECTIONS
from .morse import MORSE_TABLE, candidates, decode

# Characters shown in the reference chart, laid out column by column.
CHART_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
CHART_ROWS = 12
PUNCTUATION_HINT = ". , ? ! ' / ( ) : = + - @"

# The UI layer is cleared to this colour with zero alpha. Anti-aliased
# edges blend towards it, so a near-white clear colour keeps light text
# from getting dark fringes.
UI_CLEAR = (232, 236, 244, 0)

FLASH_TIME = 0.35          # how long a legend row stays lit after an action
CHIP_SIZE = (58, 26)       # elbow angle readout next to each elbow
CHIP_GAP = 26              # distance between the elbow and the readout


@dataclass
class HudState:
    """Snapshot of everything the HUD needs for one frame."""

    fps: float = 0.0
    camera_error: Optional[str] = None
    body: bool = False                   # at least one arm is being tracked
    # 0 = arm straight, 1 = fully curled
    curl: dict = field(default_factory=lambda: {"left": 0.0, "right": 0.0})
    angle: dict = field(default_factory=lambda: {"left": None, "right": None})
    visible: dict = field(default_factory=lambda: {"left": False, "right": False})
    active: Optional[str] = None         # arm driving the key while curled
    # Window pixels of shoulder, elbow, wrist, pinky, index, thumb, or None
    arms: dict = field(default_factory=dict)
    threshold: float = 0.5               # above this the arm counts as curled
    release_threshold: float = 0.4       # below this it counts as extended
    pressed: bool = False                # inside a curl
    pressed_time: float = 0.0            # length of the current curl so far
    curl_side: str = "right"             # arm that types
    gate_side: str = "left"              # hand that switches typing on / off
    armed: bool = False                  # switch hand is a fist
    hand_visible: bool = False
    hand: Optional[list] = None          # switch-hand landmarks in window pixels
    paused: bool = False                 # input ignored until P is pressed
    pause_kind: Optional[str] = None     # "letter", "space" or None
    pause_progress: float = 0.0          # 0..1 towards `pause_kind`
    pause_left: float = 0.0              # seconds until it happens
    dash_after: float = 0.75             # curl length that makes a dash
    letter_gap: float = 2.5
    word_gap: float = 3.5
    code: str = ""
    text: str = ""
    muted: bool = False
    show_chart: bool = True


@dataclass
class Panel:
    rect: pygame.Rect
    accent: float = 0.0                  # 0..1 border highlight strength
    accent_color: tuple = ACCENT


class Fonts:
    """Loads the bundled fonts, falling back to system fonts if missing."""

    FILES = {
        "regular": "Inter-Regular.ttf",
        "medium": "Inter-Medium.ttf",
        "semibold": "Inter-SemiBold.ttf",
        "display": "InterDisplay-Bold.ttf",
        "mono": "JetBrainsMono-Medium.ttf",
    }
    FALLBACK = {
        "regular": "segoeui,helveticaneue,arial",
        "medium": "segoeuisemibold,helveticaneue,arial",
        "semibold": "segoeuisemibold,helveticaneue,arial",
        "display": "segoeuiblack,helveticaneue,arial",
        "mono": "cascadiamono,consolas,menlo,couriernew",
    }

    def __init__(self):
        self._fonts = {}

    def get(self, family: str, size: int) -> pygame.font.Font:
        key = (family, size)
        if key not in self._fonts:
            path = FONTS_DIR / self.FILES[family]
            if path.exists():
                self._fonts[key] = pygame.font.Font(str(path), size)
            else:
                self._fonts[key] = pygame.font.SysFont(self.FALLBACK[family], size)
        return self._fonts[key]


class Hud:
    def __init__(self, size: tuple):
        self.size = size
        w, h = size
        self.ui = pygame.Surface(size, pygame.SRCALPHA)
        self.glow = pygame.Surface(size)
        self.fonts = Fonts()
        self._text_cache = {}
        self._chart_cache = {}

        margin = 16
        self.rect_title = pygame.Rect(margin, margin, 248, 60)
        self.rect_legend = pygame.Rect(margin, 88, 248, 132)
        self.rect_meters = pygame.Rect(margin, 232, 248, 80)
        self.rect_fps = pygame.Rect(w - margin - 100, margin, 100, 32)
        self.rect_chart = pygame.Rect(w - margin - 212, 60, 212, 362)
        self.rect_bottom = pygame.Rect(margin, h - margin - 150, w - 2 * margin, 150)
        self.rect_error = pygame.Rect(0, 0, 320, 96)
        self.rect_error.center = (w // 2, h // 2 - 40)

        # Letters pop up high between the side panels, clear of the face.
        self.pop_center = ((self.rect_legend.right + self.rect_chart.left) / 2, 140)

        # Animation state
        self._pops = []           # (text, color, start, size)
        self._shake_start = -10.0
        self._symbol_start = -10.0
        self._flash = {}          # legend key -> time it was last triggered
        self._bottom_flash = (-10.0, ACCENT)
        self.panels = []

    # ------------------------------------------------------------------------
    # Event hooks (called by the app when something happens)
    # ------------------------------------------------------------------------

    def on_symbol(self, symbol: str, now: float) -> None:
        self._symbol_start = now
        self._flash["dot" if symbol == "." else "dash"] = now

    def on_letter(self, char: str, now: float) -> None:
        self._pops.append((char, TEXT, now, 140))
        self._flash["letter"] = now
        self._bottom_flash = (now, ACTION_COLORS["letter"])


    def on_swap(self, curl_side: str, now: float) -> None:
        self._pops.append((f"{curl_side} arm types", ACCENT, now, 34))

    def on_space(self, now: float) -> None:
        self._pops.append(("space", ACTION_COLORS["word"], now, 40))
        self._flash["word"] = now

    def on_invalid(self, code: str, now: float) -> None:
        self._pops.append(("?", DANGER, now, 140))
        self._shake_start = now
        self._bottom_flash = (now, DANGER)

    def on_clear(self, now: float) -> None:
        self._pops.append(("cleared", DANGER, now, 40))
        self._bottom_flash = (now, DANGER)

    # ------------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------------

    def draw(self, state: HudState, now: float, visible: bool = True) -> None:
        self.ui.fill(UI_CLEAR)
        self.glow.fill((0, 0, 0))
        if not visible:
            # UI hidden with Z: only a plain FPS readout stays on screen.
            self.panels = []
            self._pops = []
            label = self.text("mono", 13, f"{state.fps:.0f} FPS", TEXT)
            self.blit(label, (self.size[0] - 16, 16), "topright")
            return
        self._build_panels(state, now)
        self._draw_hand(state)
        self._draw_arms(state)
        self._draw_angle_chips(state)

        self._draw_title(state)
        self._draw_legend(state, now)
        self._draw_meters(state)
        self._draw_fps(state)
        if state.show_chart:
            self._draw_chart(state)
        self._draw_bottom(state, now)
        if state.camera_error:
            self._draw_camera_error(state)
        self._draw_pops(now)

    def _build_panels(self, state: HudState, now: float) -> None:
        flash_t, flash_color = self._bottom_flash
        flash = max(0.0, 1.0 - (now - flash_t) / 0.6)
        self.panels = [
            Panel(self.rect_title),
            Panel(self.rect_legend),
            Panel(self.rect_meters),
            Panel(self.rect_fps),
            Panel(self.rect_bottom, flash, flash_color),
        ]
        if state.show_chart:
            self.panels.append(Panel(self.rect_chart))
        if state.camera_error:
            self.panels.append(Panel(self.rect_error, 0.6, DANGER))
        # Glass readouts next to the elbows; the border lights up while that
        # arm is curled, in the colour of the action it is about to trigger.
        for side, rect in self._chip_rects(state).items():
            on = self._is_pressing(state, side)
            self.panels.append(Panel(rect, 0.9 if on else 0.0, self._arm_color(state, side)))
        hand_rect = self._hand_chip_rect(state)
        if hand_rect is not None:
            self.panels.append(Panel(hand_rect, 0.8 if state.armed else 0.0, SUCCESS))

    @staticmethod
    def _is_pressing(state: HudState, side: str) -> bool:
        return state.pressed and state.active == side

    @staticmethod
    def _press_color(state: HudState) -> tuple:
        """What the curl will type if the arm extends now: dot, then dash."""
        return ACTION_COLORS["dash" if state.pressed_time >= state.dash_after else "dot"]

    def _arm_color(self, state: HudState, side: str) -> tuple:
        if self._is_pressing(state, side):
            return self._press_color(state)
        if not state.armed:
            return TEXT_FAINT
        return ACCENT if state.curl.get(side, 0.0) > state.release_threshold else BONE

    def _chip_rects(self, state: HudState) -> dict:
        """Place each angle readout beside its elbow, on the outer side."""
        if not state.body:
            return {}
        rects = {}
        w = self.size[0]
        for side, pts in state.arms.items():
            if not pts or side != state.curl_side:
                continue
            (sx, _), (ex, ey) = pts[0], pts[1]
            dx = ex - sx
            direction = (1 if dx > 0 else -1) if abs(dx) > 8 else (1 if ex > w / 2 else -1)
            rect = pygame.Rect((0, 0), CHIP_SIZE)
            rect.center = (round(ex + direction * (CHIP_GAP + CHIP_SIZE[0] / 2)), round(ey))
            rects[side] = rect
        return rects

    # ------------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------------

    def text(self, family: str, size: int, text: str, color) -> pygame.Surface:
        key = (family, size, text, tuple(color))
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 600:
                self._text_cache.clear()
            font = self.fonts.get(family, size)
            surf = font.render(text, True, color[:3])
            if len(color) == 4:
                surf.set_alpha(color[3])
            self._text_cache[key] = surf
        return surf

    def label(self, text: str, color=TEXT_MUTED, tracking: float = 1.2) -> pygame.Surface:
        """Small uppercase caption with extra letter spacing."""
        key = ("label", text, tuple(color), tracking)
        surf = self._text_cache.get(key)
        if surf is None:
            glyphs = [self.text("semibold", 10, ch, color) for ch in text.upper()]
            width = int(sum(g.get_width() + tracking for g in glyphs))
            surf = pygame.Surface((max(width, 1), glyphs[0].get_height()), pygame.SRCALPHA)
            x = 0.0
            for g in glyphs:
                surf.blit(g, (round(x), 0))
                x += g.get_width() + tracking
            self._text_cache[key] = surf
        return surf

    def blit(self, surf, pos, anchor: str = "topleft") -> pygame.Rect:
        rect = surf.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
        self.ui.blit(surf, rect)
        return rect

    # ------------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------------

    HAND_CHIP = (84, 24)

    def _hand_chip_rect(self, state: HudState):
        """Readout under the switch hand's wrist."""
        if not state.hand:
            return None
        wx, wy = state.hand[0]
        rect = pygame.Rect((0, 0), self.HAND_CHIP)
        rect.midtop = (round(wx), round(wy + 18))
        rect.clamp_ip(pygame.Rect(0, 0, *self.size))
        return rect

    def _draw_hand(self, state: HudState) -> None:
        """21-point skeleton of the switch hand: green fist = on, grey = off."""
        pts = state.hand
        if not pts:
            return
        color = SUCCESS if state.armed else BONE
        alpha = 0.95 if state.armed else 0.6
        for a, b in HAND_CONNECTIONS:
            pygame.draw.line(self.glow, draw.scale_rgb(color, 0.45 if state.armed else 0.18),
                             (round(pts[a][0]), round(pts[a][1])),
                             (round(pts[b][0]), round(pts[b][1])), 4)
            draw.bone(self.ui, draw.with_alpha(color, alpha), pts[a], pts[b], 2)
        for i, p in enumerate(pts):
            r = 4 if i in FINGERTIPS or i == 0 else 3
            draw.circle(self.ui, (12, 16, 24, 230), p, r)
            draw.ring(self.ui, draw.with_alpha(color, alpha), p, r, 1)
        rect = self._hand_chip_rect(state)
        if rect is not None:
            text = "FIST on" if state.armed else "OPEN off"
            label = self.text("mono", 12, text, SUCCESS if state.armed else TEXT_MUTED)
            self.blit(label, rect.center, "center")

    def _draw_arms(self, state: HudState) -> None:
        """
        Skeleton of the typing arm only: shoulder - elbow - wrist as glowing
        bones, joints as rings, and the elbow angle as a filled wedge.
        No hand bones on this arm (the hand skeleton belongs to the switch
        hand), and no skeleton for the other arm.
        """
        if not state.body:
            return
        arms = {side: pts for side, pts in state.arms.items()
                if pts and side == state.curl_side}

        for side, pts in arms.items():
            shoulder, elbow, wrist = pts[:3]
            color = self._arm_color(state, side)
            pressing = self._is_pressing(state, side)
            lead = pressing or state.active is None
            alpha = 0.95 if lead else 0.55

            # Glow layer: the GPU blurs these thick strokes into neon light.
            glow_k = 0.75 if pressing else 0.35
            for a, b in ((shoulder, elbow), (elbow, wrist)):
                pygame.draw.line(self.glow, draw.scale_rgb(color, glow_k),
                                 (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), 5)

            # Elbow angle wedge, between the two bones.
            ang_up = math.atan2(shoulder[1] - elbow[1], shoulder[0] - elbow[0])
            ang_fore = math.atan2(wrist[1] - elbow[1], wrist[0] - elbow[0])
            fill = 0.32 if pressing else 0.16
            draw.wedge(self.ui, draw.with_alpha(color, fill), elbow, 34, ang_up, ang_fore)

            # Bones
            draw.bone(self.ui, draw.with_alpha(color, alpha), shoulder, elbow, 3.5)
            draw.bone(self.ui, draw.with_alpha(color, alpha), elbow, wrist, 3)

            # Joints
            for point, r in ((shoulder, 7), (elbow, 8), (wrist, 6)):
                draw.circle(self.ui, (12, 16, 24, 230), point, r)
                draw.ring(self.ui, draw.with_alpha(color, 1.0 if lead else 0.7), point, r, 2)
            draw.circle(self.ui, color, elbow, 3)
            draw.glow(self.glow, color, elbow, 22 if pressing else 14,
                      0.9 if pressing else 0.45)

    def _draw_angle_chips(self, state: HudState) -> None:
        """Elbow angle in degrees beside each elbow."""
        for side, rect in self._chip_rects(state).items():
            angle = state.angle.get(side)
            if angle is None:
                continue
            on = self._is_pressing(state, side)
            color = self._arm_color(state, side)
            label = self.text("mono", 13, f"{round(angle):3d}\u00b0",
                              color if on else TEXT)
            self.blit(label, rect.center, "center")

    def _draw_title(self, state: HudState) -> None:
        r = self.rect_title
        self.blit(self.text("semibold", 17, "Curl Morse", TEXT), (r.x + 16, r.y + 10))

        if state.camera_error:
            dot, msg = DANGER, "Camera unavailable"
        elif state.paused:
            dot, msg = WARNING, "Paused, press P to listen"
        elif state.body and state.armed:
            dot, msg = SUCCESS, "Listening"
        elif not state.armed:
            dot, msg = WARNING, f"Make a {state.gate_side}-hand fist to type"
        else:
            dot, msg = TEXT_FAINT, f"Show your {state.curl_side} arm"

        cy = r.y + 43
        draw.circle(self.ui, dot, (r.x + 20, cy), 3.5)
        if dot is not TEXT_FAINT:
            draw.glow(self.glow, dot, (r.x + 20, cy), 10, 0.6)
        self.blit(self.text("regular", 12, msg, TEXT_MUTED), (r.x + 30, cy), "midleft")

    def _legend_rows(self, state: HudState) -> list:
        def sec(v):
            return f"{round(v, 2):g} s"
        return [
            ("dot", "Short curl", f"< {sec(state.dash_after)}  \u00b7"),
            ("dash", "Long curl", f"\u2265 {sec(state.dash_after)}  \u2013"),
            ("letter", f"Straight {sec(state.letter_gap)}", "end letter"),
            ("word", f"Straight {sec(state.word_gap)}", "space"),
        ]

    def _draw_legend(self, state: HudState, now: float) -> None:
        r = self.rect_legend
        for i, (key, name, hint) in enumerate(self._legend_rows(state)):
            cy = r.y + 22 + i * 29
            color = ACTION_COLORS[key]

            # A row lights up briefly when its action fires, and the pause
            # rows stay lit while the pause is counting towards them.
            lit = max(0.0, 1.0 - (now - self._flash.get(key, -10.0)) / FLASH_TIME)
            counting = {"letter": "letter", "space": "word"}.get(state.pause_kind)
            if key == counting and state.pause_progress > 0:
                lit = max(lit, 0.6 * state.pause_progress)
            if lit > 0:
                pill = pygame.Rect(r.x + 8, cy - 12, r.w - 16, 24)
                draw.rounded_rect(self.ui, draw.with_alpha(color, 0.18 * lit), pill, 8)

            draw.circle(self.ui, color, (r.x + 20, cy), 4.5)
            draw.glow(self.glow, color, (r.x + 20, cy), 9 + 8 * lit, 0.35 + 0.5 * lit)
            self.blit(self.text("medium", 13, name, TEXT), (r.x + 34, cy), "midleft")
            self.blit(self.text("regular", 12, hint, TEXT_MUTED),
                      (r.right - 14, cy), "midright")

    def _draw_meters(self, state: HudState) -> None:
        """Curl bar of the typing arm, and the state of the switch hand."""
        r = self.rect_meters
        self.blit(self.label("Arm curl"), (r.x + 16, r.y + 12))
        x0, x1 = r.x + 62, r.right - 62

        # Row 1: typing arm
        side = state.curl_side
        cy = r.y + 40
        visible = state.body and state.visible.get(side, False)
        value = state.curl.get(side, 0.0) if visible else 0.0
        on = self._is_pressing(state, side)
        color = self._press_color(state) if on else (
            ACTION_COLORS["dot"] if state.armed else TEXT_FAINT)
        self.blit(self.text("medium", 12, side.capitalize(),
                            TEXT if visible else TEXT_FAINT), (r.x + 16, cy), "midleft")
        track = pygame.Rect(x0, cy - 3, x1 - x0, 6)
        draw.rounded_rect(self.ui, (255, 255, 255, 30), track, 3)
        if visible:
            fill = track.copy()
            fill.w = max(6, round(track.w * value))
            draw.rounded_rect(self.ui, draw.with_alpha(color, 0.75), fill, 3)
            if on:
                draw.glow(self.glow, color, (fill.right - 3, cy), 16, 0.7)
        tx = x0 + track.w * state.threshold
        draw.line(self.ui, (255, 255, 255, 190), (tx, cy - 7), (tx, cy + 7), 1)
        if not visible:
            status, scol = "-", TEXT_FAINT
        elif on:
            status, scol = "curl", color
        else:
            status, scol = f"{round(value * 100)}%", TEXT_MUTED
        self.blit(self.text("regular", 11, status, scol), (r.right - 14, cy), "midright")

        # Row 2: switch hand
        cy = r.y + 64
        self.blit(self.text("medium", 12, state.gate_side.capitalize(),
                            TEXT if state.hand_visible else TEXT_FAINT),
                  (r.x + 16, cy), "midleft")
        if not state.hand_visible:
            dot, msg = TEXT_FAINT, "hand not in view, typing off"
        elif state.armed:
            dot, msg = SUCCESS, "fist, typing on"
        else:
            dot, msg = TEXT_MUTED, "open hand, typing off"
        draw.circle(self.ui, dot, (x0 + 4, cy), 4)
        if state.armed:
            draw.glow(self.glow, dot, (x0 + 4, cy), 12, 0.7)
        self.blit(self.text("regular", 12, msg, SUCCESS if state.armed else TEXT_MUTED),
                  (x0 + 16, cy), "midleft")

    def _draw_fps(self, state: HudState) -> None:
        r = self.rect_fps
        if state.fps >= 30:
            color = SUCCESS
        elif state.fps >= MIN_ACCEPTABLE_FPS:
            color = WARNING
        else:
            color = DANGER
        label = self.text("mono", 13, f"{state.fps:4.0f} FPS", TEXT)
        total = 14 + label.get_width()
        x = r.centerx - total / 2
        draw.circle(self.ui, color, (x + 3, r.centery), 3.5)
        draw.glow(self.glow, color, (x + 3, r.centery), 10, 0.6)
        self.blit(label, (x + 14, r.centery), "midleft")

    def _draw_chart(self, state: HudState) -> None:
        # The chart only changes when the typed code changes, so it is cached.
        key = state.code
        surf = self._chart_cache.get(key)
        if surf is None:
            if len(self._chart_cache) > 64:
                self._chart_cache.clear()
            surf = self._render_chart(state.code)
            self._chart_cache[key] = surf
        self.ui.blit(surf, self.rect_chart.topleft)

    def _render_chart(self, prefix: str) -> pygame.Surface:
        r = self.rect_chart
        surf = pygame.Surface(r.size, pygame.SRCALPHA)
        surf.fill(UI_CLEAR)

        header = self.label("Morse chart")
        surf.blit(header, (14, 14))
        if prefix:
            n = len(candidates(prefix))
            hint = self.text("regular", 11, f"{n} match" + ("" if n == 1 else "es"),
                             ACCENT if n else DANGER)
            surf.blit(hint, hint.get_rect(topright=(r.w - 14, 12)))

        # Digits have five symbols each, so the last column is wider.
        col_x = (14, 76, 138)
        col_w = (60, 60, 68)
        dot_color = TEXT
        for i, char in enumerate(CHART_CHARS):
            col, row = divmod(i, CHART_ROWS)
            x = col_x[col]
            cy = 46 + row * 24
            code = MORSE_TABLE[char]

            reachable = not prefix or code.startswith(prefix)
            exact = bool(prefix) and code == prefix
            alpha = 1.0 if reachable else 0.22

            if exact:
                pill = pygame.Rect(x - 6, cy - 11, col_w[col], 22)
                draw.rounded_rect(surf, draw.with_alpha(ACCENT, 0.18), pill, 7)

            char_color = ACCENT if exact else TEXT
            glyph = self.text("semibold", 13, char, draw.with_alpha(char_color, alpha))
            surf.blit(glyph, glyph.get_rect(midleft=(round(x), cy)))

            gx = x + 16
            for j, sym in enumerate(code):
                typed = prefix and j < len(prefix) and reachable
                color = ACCENT if typed else dot_color
                color = draw.with_alpha(color, alpha * (1.0 if typed or not prefix else 0.75))
                if sym == ".":
                    draw.circle(surf, color, (gx + 2, cy), 2.0)
                    gx += 7
                else:
                    draw.capsule(surf, color, (gx + 3.5, cy), 7, 3.2)
                    gx += 10

        foot = self.text("mono", 10, PUNCTUATION_HINT, TEXT_FAINT)
        surf.blit(foot, (14, r.h - 24))
        return surf

    def _draw_bottom(self, state: HudState, now: float) -> None:
        r = self.rect_bottom
        x0 = r.x + 20

        # Row 1: caption and keyboard hints
        self.blit(self.label("Current letter"), (x0, r.y + 16))
        self._draw_key_hints(r.right - 20, r.y + 21, state)

        # Row 2: the dots and dashes typed so far, and the pause countdown
        cy = r.y + 56
        shake = 0.0
        dt = now - self._shake_start
        if dt < 0.35:
            shake = math.sin(dt * 60) * 7 * (1 - dt / 0.35)

        curling = state.pressed and not state.paused
        if state.code or curling:
            end = self._draw_code(state.code, x0 + shake, cy, now)
            if curling:
                self._draw_live_curl(state, end, cy)
            if state.code:
                self._draw_prediction(state, r.right - 20, cy)
        elif state.pause_kind == "space":
            color = ACTION_COLORS["word"]
            draw.ring(self.ui, (255, 255, 255, 40), (x0 + 9, cy), 8, 2)
            draw.arc(self.ui, color, (x0 + 9, cy), 8, 0,
                     2 * math.pi * state.pause_progress, 2.5)
            self.blit(self.text("medium", 14, f"Space in {state.pause_left:.1f} s", color),
                      (x0 + 28, cy), "midleft")
            self.blit(self.text("regular", 12, "or keep typing the same word",
                                TEXT_FAINT), (x0 + 150, cy), "midleft")
        elif state.paused:
            self.blit(self.text("medium", 14, "Paused. Press P to start listening again",
                                WARNING), (x0, cy), "midleft")
        else:
            self.blit(self.text("regular", 14,
                                f"{state.gate_side.capitalize()} fist on, then "
                                f"{state.curl_side} curl: short = dot, long = dash",
                                TEXT_FAINT), (x0 + shake, cy), "midleft")

        # Divider
        pygame.draw.line(self.ui, (255, 255, 255, 22),
                         (x0, r.y + 84), (r.right - 20, r.y + 84))

        # Row 3: the decoded message with a blinking caret
        self.blit(self.label("Message"), (x0, r.y + 96))
        if state.text:
            n = len(state.text)
            count = self.text("regular", 11, f"{n} char" + ("" if n == 1 else "s"), TEXT_FAINT)
            self.blit(count, (r.right - 20, r.y + 95), "topright")

        font = self.fonts.get("medium", 26)
        max_w = r.w - 60
        shown = state.text
        # Keep the end of the message visible when it gets too long.
        while shown and font.size(shown)[0] > max_w:
            shown = shown[1:]
        if shown != state.text:
            shown = "…" + shown[1:]

        ty = r.y + 128
        if shown:
            surf = font.render(shown, True, TEXT)
            rect = self.blit(surf, (x0, ty), "midleft")
            caret_x = rect.right + 3
        else:
            self.blit(self.text("regular", 20, "Your message appears here", TEXT_FAINT),
                      (x0 + 8, ty), "midleft")
            caret_x = x0

        if (now * 1.8) % 1.0 < 0.6:
            caret = pygame.Rect(0, 0, 2, 24)
            caret.midleft = (caret_x, ty)
            pygame.draw.rect(self.ui, ACCENT, caret, border_radius=1)
            draw.glow(self.glow, ACCENT, caret.center, 10, 0.5)

    def _draw_code(self, code: str, x: float, cy: float, now: float) -> float:
        """Draw the typed symbols; returns the x where the next one would go."""
        t = now - self._symbol_start
        for i, sym in enumerate(code):
            is_last = i == len(code) - 1
            grow = 1.0
            if is_last:
                grow = draw.ease_out_back(t / 0.18)
            if sym == ".":
                color, cx, width = ACTION_COLORS["dot"], x + 8, 28
                draw.glow(self.glow, color, (cx, cy), 18, 0.55 * grow)
                draw.circle(self.ui, color, (cx, cy), 7.5 * grow)
            else:
                color, cx, width = ACTION_COLORS["dash"], x + 17, 48
                draw.glow(self.glow, color, (cx, cy), 24, 0.55 * grow)
                draw.capsule(self.ui, color, (cx, cy), 34 * grow, 13 * grow)

            # A quick ripple on the newest symbol confirms the curl landed.
            if is_last and t < 0.35:
                k = t / 0.35
                draw.ring(self.ui, draw.with_alpha(color, 1.0 - k), (cx, cy),
                          10 + 18 * draw.ease_out_cubic(k), 2)
                draw.glow(self.glow, color, (cx, cy), 30, 0.8 * (1.0 - k))
            x += width
        return x

    def _draw_live_curl(self, state: HudState, x: float, cy: float) -> None:
        """
        While the arm is curled, show the symbol the curl will produce:
        a dot outline that stretches into a dash as the curl gets longer.
        Once the dash has been typed it appears as a real symbol instead.
        """
        if state.pressed_time >= state.dash_after:
            return
        k = state.pressed_time / max(state.dash_after, 1e-3)
        color = ACTION_COLORS["dot"]
        length = 15 + 19 * k
        rect = pygame.Rect(0, 0, round(length), 15)
        rect.midleft = (round(x + 1), round(cy))
        draw.rounded_rect(self.ui, draw.with_alpha(color, 0.9), rect, 8, 2)
        draw.glow(self.glow, color, rect.center, 16, 0.4)

    def _draw_prediction(self, state: HudState, right: float, cy: float) -> None:
        """Countdown ring with the letter that the pause will commit."""
        code = state.code
        color = ACTION_COLORS["letter"]
        centre = (right - 22, cy)
        char = decode(code)
        options = candidates(code)

        progress = state.pause_progress if state.pause_kind == "letter" else 0.0
        draw.ring(self.ui, (255, 255, 255, 40), centre, 21, 3)
        draw.arc(self.ui, color, centre, 21, 0, 2 * math.pi * progress, 3.5)
        if progress > 0:
            draw.glow(self.glow, color, centre, 26, 0.35 * progress)
        if char is not None:
            glyph = self.text("display", 22, char, TEXT)
        elif options:
            glyph = self.text("display", 18, "\u2026", TEXT_MUTED)
        else:
            glyph = self.text("display", 20, "?", DANGER)
        self.blit(glyph, centre, "center")

        left = right - 56
        if char is not None:
            self.blit(self.text("regular", 12, "straighten arm to confirm", TEXT_MUTED),
                      (left, cy), "midright")
        elif options:
            preview = "  ".join(options[:6]) + ("  \u2026" if len(options) > 6 else "")
            self.blit(self.text("medium", 14, preview, TEXT_MUTED), (left, cy), "midright")
        else:
            self.blit(self.text("medium", 13, "no match", DANGER), (left, cy), "midright")

    def _draw_key_hints(self, right: float, cy: float, state: HudState) -> None:
        hints = [("S", "swap hands"), ("X", "settings"), ("Z", "hide UI"),
                 ("P", "resume" if state.paused else "pause"),
                 ("H", "chart"), ("M", "sound off" if state.muted else "sound")]
        x = right
        for key, text in reversed(hints):
            label = self.text("regular", 11, text, TEXT_MUTED)
            x -= label.get_width()
            self.blit(label, (x, cy), "midleft")
            x -= 6
            cap = pygame.Rect(0, 0, 18, 18)
            cap.midright = (round(x), round(cy))
            draw.rounded_rect(self.ui, (255, 255, 255, 28), cap, 5)
            draw.rounded_rect(self.ui, (255, 255, 255, 60), cap, 5, 1)
            self.blit(self.text("mono", 10, key, TEXT), cap.center, "center")
            x = cap.left - 14

    def _draw_camera_error(self, state: HudState) -> None:
        r = self.rect_error
        self.blit(self.text("semibold", 16, state.camera_error, TEXT),
                  (r.centerx, r.y + 32), "center")
        self.blit(self.text("regular", 12, "Press X to choose another camera",
                            TEXT_MUTED), (r.centerx, r.y + 60), "center")

    # ------------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------------

    def _draw_pops(self, now: float) -> None:
        alive = []
        cx, cy = self.pop_center
        for text, color, start, size in self._pops:
            t = (now - start) / 0.9
            if t >= 1.0:
                continue
            alive.append((text, color, start, size))
            scale = 0.6 + 0.4 * draw.ease_out_back(t / 0.28)
            fade = 1.0 if t < 0.45 else 1.0 - (t - 0.45) / 0.55
            rise = -26 * draw.ease_out_cubic(t)

            base = self.fonts.get("display", size).render(text, True, color)
            w, h = base.get_size()
            surf = pygame.transform.smoothscale(base, (max(1, int(w * scale)),
                                                       max(1, int(h * scale))))
            surf.set_alpha(int(255 * fade))
            self.blit(surf, (cx, cy + rise), "center")

            glow_color = ACCENT if color == TEXT else color
            # Text rendered with a background is 8-bit; copy it into a
            # 24-bit surface so it can be scaled smoothly.
            glyph = self.fonts.get("display", size).render(
                text, True, draw.scale_rgb(glow_color, 0.9 * fade), (0, 0, 0))
            glow_surf = pygame.Surface(glyph.get_size())
            glow_surf.blit(glyph, (0, 0))
            glow_surf = pygame.transform.smoothscale(glow_surf, surf.get_size())
            self.glow.blit(glow_surf, glow_surf.get_rect(center=(cx, cy + rise)),
                           special_flags=pygame.BLEND_RGB_ADD)
        self._pops = alive
