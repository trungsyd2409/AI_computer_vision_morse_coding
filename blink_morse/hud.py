"""
Everything drawn on top of the camera image.

The whole body is drawn as a thin, small skeleton (head circle, torso,
arms, legs) so the pose is easy to recognise without covering the image.
It lights up cyan while the body is down (a dot so far) and violet once
the push-up has become a dash. All other feedback lives in the panels.
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
from dataclasses import dataclass
from typing import Optional

import pygame

from . import draw
from .config import (ACCENT, ACTION_COLORS, DANGER, FONTS_DIR, MIN_ACCEPTABLE_FPS,
                     BONE, SUCCESS, TEXT, TEXT_FAINT, TEXT_MUTED, WARNING)
from .arm_tracker import BODY_CONNECTIONS
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


@dataclass
class HudState:
    """Snapshot of everything the HUD needs for one frame."""

    fps: float = 0.0
    camera_error: Optional[str] = None
    body: bool = False                   # a person is being tracked
    angle: Optional[float] = None        # mean elbow angle of the visible arms
    depth: float = 0.0                   # 0 = up (arms straight), 1 = chest down
    ready: bool = False                  # whole body in view and horizontal
    posture: str = "no body"             # what the posture check says
    tilt: Optional[float] = None         # body angle from horizontal, degrees
    # 33 pose landmarks as (x, y, visibility) in window pixels, or None
    skeleton: Optional[list] = None
    threshold: float = 0.5               # above this the body counts as down
    release_threshold: float = 0.4       # below this it counts as up again
    pressed: bool = False                # body is down
    pressed_time: float = 0.0            # how long it has been down so far
    paused: bool = False                 # input ignored until P is pressed
    pause_kind: Optional[str] = None     # "letter", "space" or None
    pause_progress: float = 0.0          # 0..1 towards `pause_kind`
    pause_left: float = 0.0              # seconds until it happens
    dash_after: float = 0.9              # time down that makes a dash
    letter_gap: float = 1.8
    word_gap: float = 3.0
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

        # Compact layout: small panels hugging the edges, so most of the
        # window shows the camera.
        margin = 10
        self.rect_title = pygame.Rect(margin, margin, 196, 44)
        self.rect_legend = pygame.Rect(margin, 60, 196, 96)
        self.rect_meters = pygame.Rect(margin, 162, 196, 58)
        self.rect_fps = pygame.Rect(w - margin - 76, margin, 76, 24)
        self.rect_chart = pygame.Rect(w - margin - 182, 40, 182, 280)
        self.rect_bottom = pygame.Rect(margin, h - margin - 100, w - 2 * margin, 100)
        self.rect_error = pygame.Rect(0, 0, 320, 96)
        self.rect_error.center = (w // 2, h // 2 - 40)

        # Letters pop up high between the side panels, clear of the face.
        self.pop_center = ((self.rect_legend.right + self.rect_chart.left) / 2, 110)

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
        self._draw_skeleton(state)

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

    @staticmethod
    def _press_color(state: HudState) -> tuple:
        """What the push-up will type if the body comes up now: dot, then dash."""
        return ACTION_COLORS["dash" if state.pressed_time >= state.dash_after else "dot"]

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

    def _draw_skeleton(self, state: HudState) -> None:
        """
        Thin whole-body skeleton: small head circle, torso, arms, legs.
        Kept deliberately light (1.5 px bones, 2.5 px joints) so it is easy
        to recognise without hiding the camera image. Joints MediaPipe is
        unsure about are skipped.
        """
        pts = state.skeleton
        if not pts:
            return
        if state.pressed and not state.paused:
            color, alpha, glow_k = self._press_color(state), 0.95, 0.5
        elif state.ready:
            color, alpha, glow_k = SUCCESS, 0.85, 0.2      # in position, counting
        else:
            color, alpha, glow_k = BONE, 0.45, 0.0         # not counting

        def ok(i):
            return pts[i][2] >= 0.5

        for a, b in BODY_CONNECTIONS:
            if not (ok(a) and ok(b)):
                continue
            pa, pb = pts[a][:2], pts[b][:2]
            pygame.draw.line(self.glow, draw.scale_rgb(color, glow_k),
                             (round(pa[0]), round(pa[1])), (round(pb[0]), round(pb[1])), 3)
            draw.bone(self.ui, draw.with_alpha(color, alpha), pa, pb, 1.5)
        joints = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
        for i in joints:
            if ok(i):
                draw.circle(self.ui, draw.with_alpha(color, alpha), pts[i][:2], 2.5)

        # Head: a small ring around the nose, sized from the ear distance.
        if ok(0):
            nose = pts[0][:2]
            ears = [pts[i][:2] for i in (7, 8) if ok(i)]
            if len(ears) == 2:
                r = max(6.0, math.dist(ears[0], ears[1]) * 0.6)
            elif ears:
                r = max(6.0, math.dist(ears[0], nose) * 1.1)
            else:
                r = 10.0
            draw.ring(self.ui, draw.with_alpha(color, alpha), nose, r, 1)
            # Neck: nose ring to the middle of the shoulders.
            if ok(11) and ok(12):
                mid = ((pts[11][0] + pts[12][0]) / 2, (pts[11][1] + pts[12][1]) / 2)
                d = math.dist(nose, mid)
                if d > r:
                    k = r / d
                    start = (nose[0] + (mid[0] - nose[0]) * k, nose[1] + (mid[1] - nose[1]) * k)
                    draw.bone(self.ui, draw.with_alpha(color, alpha), start, mid, 1.5)

    def _draw_title(self, state: HudState) -> None:
        r = self.rect_title
        self.blit(self.text("semibold", 14, "Push-up Morse", TEXT), (r.x + 12, r.y + 6))

        if state.camera_error:
            dot, msg = DANGER, "Camera unavailable"
        elif state.paused:
            dot, msg = WARNING, "Paused, press P to listen"
        elif state.ready:
            dot, msg = SUCCESS, "Listening"
        elif state.body:
            dot, msg = WARNING, "Get into push-up position"
        else:
            dot, msg = TEXT_FAINT, "Show your whole body"

        cy = r.y + 32
        draw.circle(self.ui, dot, (r.x + 15, cy), 3)
        if dot is not TEXT_FAINT:
            draw.glow(self.glow, dot, (r.x + 15, cy), 8, 0.6)
        self.blit(self.text("regular", 11, msg, TEXT_MUTED), (r.x + 24, cy), "midleft")

    def _legend_rows(self, state: HudState) -> list:
        def sec(v):
            return f"{round(v, 2):g} s"
        return [
            ("dot", "Down short", f"< {sec(state.dash_after)}  \u00b7"),
            ("dash", "Down long", f"\u2265 {sec(state.dash_after)}  \u2013"),
            ("letter", f"Up {sec(state.letter_gap)}", "end letter"),
            ("word", f"Up {sec(state.word_gap)}", "space"),
        ]

    def _draw_legend(self, state: HudState, now: float) -> None:
        r = self.rect_legend
        for i, (key, name, hint) in enumerate(self._legend_rows(state)):
            cy = r.y + 15 + i * 22
            color = ACTION_COLORS[key]

            # A row lights up briefly when its action fires, and the pause
            # rows stay lit while the pause is counting towards them.
            lit = max(0.0, 1.0 - (now - self._flash.get(key, -10.0)) / FLASH_TIME)
            counting = {"letter": "letter", "space": "word"}.get(state.pause_kind)
            if key == counting and state.pause_progress > 0:
                lit = max(lit, 0.6 * state.pause_progress)
            if lit > 0:
                pill = pygame.Rect(r.x + 6, cy - 10, r.w - 12, 20)
                draw.rounded_rect(self.ui, draw.with_alpha(color, 0.18 * lit), pill, 8)

            draw.circle(self.ui, color, (r.x + 15, cy), 3.5)
            draw.glow(self.glow, color, (r.x + 15, cy), 7 + 6 * lit, 0.35 + 0.5 * lit)
            self.blit(self.text("medium", 12, name, TEXT), (r.x + 26, cy), "midleft")
            self.blit(self.text("regular", 11, hint, TEXT_MUTED),
                      (r.right - 10, cy), "midright")

    def _draw_meters(self, state: HudState) -> None:
        """Push-up depth bar with the threshold marked, and the elbow angle."""
        r = self.rect_meters
        self.blit(self.label("Push-up depth"), (r.x + 12, r.y + 7))
        x0, x1 = r.x + 52, r.right - 44

        cy = r.y + 29
        visible = state.body and state.angle is not None
        value = state.depth if visible else 0.0
        on = state.pressed and visible
        color = self._press_color(state) if on else (
            ACTION_COLORS["dot"] if state.ready else TEXT_FAINT)
        self.blit(self.text("medium", 11, "Depth", TEXT if visible else TEXT_FAINT),
                  (r.x + 12, cy), "midleft")
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
            status, scol = "down", color
        else:
            status, scol = "up", TEXT_MUTED
        self.blit(self.text("regular", 11, status, scol), (r.right - 10, cy), "midright")

        # Row 2: posture check, and the elbow angle on the right
        cy = r.y + 46
        self.blit(self.text("medium", 11, "Pose", TEXT if state.body else TEXT_FAINT),
                  (r.x + 12, cy), "midleft")
        dot = SUCCESS if state.ready else (WARNING if state.body else TEXT_FAINT)
        draw.circle(self.ui, dot, (x0 + 3, cy), 3)
        if state.ready:
            draw.glow(self.glow, dot, (x0 + 4, cy), 12, 0.7)
        text = "ready" if state.ready else state.posture
        self.blit(self.text("regular", 10, text, SUCCESS if state.ready else TEXT_MUTED),
                  (x0 + 11, cy), "midleft")
        if visible and state.ready:
            self.blit(self.text("mono", 11, f"{round(state.angle)}\u00b0", TEXT_MUTED),
                      (r.right - 10, cy), "midright")

    def _draw_fps(self, state: HudState) -> None:
        r = self.rect_fps
        if state.fps >= 30:
            color = SUCCESS
        elif state.fps >= MIN_ACCEPTABLE_FPS:
            color = WARNING
        else:
            color = DANGER
        label = self.text("mono", 11, f"{state.fps:3.0f} FPS", TEXT)
        total = 11 + label.get_width()
        x = r.centerx - total / 2
        draw.circle(self.ui, color, (x + 3, r.centery), 3)
        draw.glow(self.glow, color, (x + 3, r.centery), 8, 0.6)
        self.blit(label, (x + 11, r.centery), "midleft")

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
        surf.blit(header, (10, 9))
        if prefix:
            n = len(candidates(prefix))
            hint = self.text("regular", 11, f"{n} match" + ("" if n == 1 else "es"),
                             ACCENT if n else DANGER)
            surf.blit(hint, hint.get_rect(topright=(r.w - 10, 7)))

        # Digits have five symbols each, so the last column is wider.
        col_x = (10, 62, 114)
        col_w = (50, 50, 64)
        dot_color = TEXT
        for i, char in enumerate(CHART_CHARS):
            col, row = divmod(i, CHART_ROWS)
            x = col_x[col]
            cy = 34 + row * 19
            code = MORSE_TABLE[char]

            reachable = not prefix or code.startswith(prefix)
            exact = bool(prefix) and code == prefix
            alpha = 1.0 if reachable else 0.22

            if exact:
                pill = pygame.Rect(x - 5, cy - 9, col_w[col], 18)
                draw.rounded_rect(surf, draw.with_alpha(ACCENT, 0.18), pill, 6)

            char_color = ACCENT if exact else TEXT
            glyph = self.text("semibold", 11, char, draw.with_alpha(char_color, alpha))
            surf.blit(glyph, glyph.get_rect(midleft=(round(x), cy)))

            gx = x + 13
            for j, sym in enumerate(code):
                typed = prefix and j < len(prefix) and reachable
                color = ACCENT if typed else dot_color
                color = draw.with_alpha(color, alpha * (1.0 if typed or not prefix else 0.75))
                if sym == ".":
                    draw.circle(surf, color, (gx + 2, cy), 1.7)
                    gx += 6
                else:
                    draw.capsule(surf, color, (gx + 3, cy), 6, 2.8)
                    gx += 8

        foot = self.text("mono", 9, PUNCTUATION_HINT, TEXT_FAINT)
        surf.blit(foot, (10, r.h - 18))
        return surf

    def _draw_bottom(self, state: HudState, now: float) -> None:
        r = self.rect_bottom
        x0 = r.x + 14

        # Row 1: caption and keyboard hints
        self.blit(self.label("Current letter"), (x0, r.y + 9))
        self._draw_key_hints(r.right - 14, r.y + 14, state)

        # Row 2: the dots and dashes typed so far, and the pause countdown
        cy = r.y + 36
        shake = 0.0
        dt = now - self._shake_start
        if dt < 0.35:
            shake = math.sin(dt * 60) * 7 * (1 - dt / 0.35)

        down = state.pressed and not state.paused
        if state.code or down:
            end = self._draw_code(state.code, x0 + shake, cy, now)
            if down:
                self._draw_live_press(state, end, cy)
            if state.code:
                self._draw_prediction(state, r.right - 14, cy)
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
                                ("Go down quickly for a dot, hold down for a dash"
                                 if state.ready else
                                 "Get into push-up position, whole body in view"),
                                TEXT_FAINT), (x0 + shake, cy), "midleft")

        # Divider
        pygame.draw.line(self.ui, (255, 255, 255, 22),
                         (x0, r.y + 55), (r.right - 14, r.y + 55))

        # Row 3: the decoded message with a blinking caret
        self.blit(self.label("Message"), (x0, r.y + 61))
        if state.text:
            n = len(state.text)
            count = self.text("regular", 11, f"{n} char" + ("" if n == 1 else "s"), TEXT_FAINT)
            self.blit(count, (r.right - 14, r.y + 60), "topright")

        font = self.fonts.get("medium", 20)
        max_w = r.w - 60
        shown = state.text
        # Keep the end of the message visible when it gets too long.
        while shown and font.size(shown)[0] > max_w:
            shown = shown[1:]
        if shown != state.text:
            shown = "…" + shown[1:]

        ty = r.y + 84
        if shown:
            surf = font.render(shown, True, TEXT)
            rect = self.blit(surf, (x0, ty), "midleft")
            caret_x = rect.right + 3
        else:
            self.blit(self.text("regular", 15, "Your message appears here", TEXT_FAINT),
                      (x0 + 8, ty), "midleft")
            caret_x = x0

        if (now * 1.8) % 1.0 < 0.6:
            caret = pygame.Rect(0, 0, 2, 19)
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
                color, cx, width = ACTION_COLORS["dot"], x + 6, 21
                draw.glow(self.glow, color, (cx, cy), 14, 0.55 * grow)
                draw.circle(self.ui, color, (cx, cy), 5.5 * grow)
            else:
                color, cx, width = ACTION_COLORS["dash"], x + 13, 36
                draw.glow(self.glow, color, (cx, cy), 18, 0.55 * grow)
                draw.capsule(self.ui, color, (cx, cy), 26 * grow, 10 * grow)

            # A quick ripple on the newest symbol confirms the push-up landed.
            if is_last and t < 0.35:
                k = t / 0.35
                draw.ring(self.ui, draw.with_alpha(color, 1.0 - k), (cx, cy),
                          8 + 12 * draw.ease_out_cubic(k), 2)
                draw.glow(self.glow, color, (cx, cy), 30, 0.8 * (1.0 - k))
            x += width
        return x

    def _draw_live_press(self, state: HudState, x: float, cy: float) -> None:
        """
        While the body is down, show the symbol the push-up will produce:
        a dot outline that stretches into a dash the longer it stays down.
        Once the dash has been typed it appears as a real symbol instead.
        """
        if state.pressed_time >= state.dash_after:
            return
        k = state.pressed_time / max(state.dash_after, 1e-3)
        color = ACTION_COLORS["dot"]
        length = 11 + 15 * k
        rect = pygame.Rect(0, 0, round(length), 11)
        rect.midleft = (round(x + 1), round(cy))
        draw.rounded_rect(self.ui, draw.with_alpha(color, 0.9), rect, 8, 2)
        draw.glow(self.glow, color, rect.center, 16, 0.4)

    def _draw_prediction(self, state: HudState, right: float, cy: float) -> None:
        """Countdown ring with the letter that the pause will commit."""
        code = state.code
        color = ACTION_COLORS["letter"]
        centre = (right - 16, cy)
        char = decode(code)
        options = candidates(code)

        progress = state.pause_progress if state.pause_kind == "letter" else 0.0
        draw.ring(self.ui, (255, 255, 255, 40), centre, 15, 2)
        draw.arc(self.ui, color, centre, 15, 0, 2 * math.pi * progress, 3)
        if progress > 0:
            draw.glow(self.glow, color, centre, 26, 0.35 * progress)
        if char is not None:
            glyph = self.text("display", 17, char, TEXT)
        elif options:
            glyph = self.text("display", 18, "\u2026", TEXT_MUTED)
        else:
            glyph = self.text("display", 20, "?", DANGER)
        self.blit(glyph, centre, "center")

        left = right - 40
        if char is not None:
            self.blit(self.text("regular", 12, "stay up to confirm", TEXT_MUTED),
                      (left, cy), "midright")
        elif options:
            preview = "  ".join(options[:6]) + ("  \u2026" if len(options) > 6 else "")
            self.blit(self.text("medium", 14, preview, TEXT_MUTED), (left, cy), "midright")
        else:
            self.blit(self.text("medium", 13, "no match", DANGER), (left, cy), "midright")

    def _draw_key_hints(self, right: float, cy: float, state: HudState) -> None:
        hints = [("X", "settings"), ("Z", "hide UI"),
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
