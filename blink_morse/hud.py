"""
Everything drawn on top of the camera image.

Kept small and flat on purpose: thin bones with a dark outline so they
read on any background, one amber accent for "live", plain text on a
plain dark backdrop. No glow, no glass, no animated pop-ups.

On screen:
* Body bones: shoulders, torso, hips, legs and feet, plus the arms.
* A 21-point skeleton for both hands. Both turn amber while they are held
  together (the gesture that switches typing on and off).
* The hip line turns amber while a sway is held after it fired, and flashes
  thicker for a moment when a symbol, letter end or space is typed.
* H: the hip angle, the +-A lines and the waiting timer.
* "on" / "off" at the right end of the bottom strip.
* Morse chart in the top-right corner.
* Code being typed and the message, in a strip at the bottom.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import pygame

from . import draw
from .body_tracker import BONES
from .config import ACCENT, BONE, DANGER, FONTS_DIR, PANEL, TEXT, TEXT_MUTED
from .hand_tracker import FINGERTIPS, HAND_CONNECTIONS
from .morse import MORSE_TABLE, candidates, decode

CHART_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
CHART_ROWS = 12

# The UI layer is cleared to this colour with zero alpha. Anti-aliased
# edges blend towards it, so a near-white clear colour keeps light text
# from getting dark fringes.
UI_CLEAR = (232, 232, 228, 0)
OUTLINE = (0, 0, 0, 150)          # thin dark edge under bones and text
BACKDROP = PANEL + (165,)

MARGIN = 12
FLASH_TIME = 0.25                 # how long a forearm stays thick after a gesture
INVALID_TIME = 0.8                # how long "no match" stays on screen


@dataclass
class HudState:
    """Snapshot of everything the HUD needs for one frame."""

    camera_error: Optional[str] = None
    body: bool = False
    armed: bool = False                  # typing switched on and not paused
    together: bool = False               # hands held together right now
    paused: bool = False
    right_symbol: str = "dot"            # what a sway to the user's right types
    left_symbol: str = "dash"
    trigger_angle: float = 5.0           # A
    unit_time: float = 0.6               # t, seconds
    # Window pixels of every visible joint, named "<side>_<joint>"
    points: dict = field(default_factory=dict)
    # Window pixels of the 21 hand landmarks per side, or None
    hands: dict = field(default_factory=lambda: {"left": None, "right": None})
    hip_angle: Optional[float] = None
    hip_zone: Optional[str] = None       # "waiting", or "right" / "left" once typed
    hip_latched: Optional[str] = None    # sway already typed, waiting for the return
    wait_for: float = 0.0                # seconds spent in waiting
    code: str = ""
    text: str = ""
    show_chart: bool = True
    show_angles: bool = True


def _mid(a, b) -> tuple:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


class Fonts:
    """Loads the bundled fonts, falling back to system fonts if missing."""

    FILES = {
        "regular": "Inter-Regular.ttf",
        "medium": "Inter-Medium.ttf",
        "semibold": "Inter-SemiBold.ttf",
        "mono": "JetBrainsMono-Medium.ttf",
    }
    FALLBACK = {
        "regular": "segoeui,helveticaneue,arial",
        "medium": "segoeuisemibold,helveticaneue,arial",
        "semibold": "segoeuisemibold,helveticaneue,arial",
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
        self.fonts = Fonts()
        self._text_cache = {}
        self._chart_cache = {}

        self.rect_chart = pygame.Rect(0, 0, 172, 12 + CHART_ROWS * 15 + 6)
        self.rect_chart.topright = (w - MARGIN, MARGIN)
        self.rect_bottom = pygame.Rect(0, 0, min(w - 2 * MARGIN, 460), 60)
        self.rect_bottom.midbottom = (w // 2, h - MARGIN)

        self._flash = -10.0
        self._invalid = (-10.0, "")
        self._switch = -10.0
        self._now = 0.0

    # ------------------------------------------------------------------------
    # Event hooks (called by the app)
    # ------------------------------------------------------------------------

    def on_gesture(self, now: float) -> None:
        self._flash = now

    def on_switch(self, on: bool, now: float) -> None:
        self._switch = now

    def on_invalid(self, code: str, now: float) -> None:
        self._invalid = (now, code)

    # ------------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------------

    def draw(self, state: HudState, now: float, visible: bool = True) -> None:
        self.ui.fill(UI_CLEAR)
        self._now = now
        if not visible:
            return
        segments, joints = [], []
        self._body_shapes(state, now, segments, joints)
        for side in ("left", "right"):
            self._hand_shapes(state, side, segments, joints)
        self._draw_shapes(segments, joints)
        if state.show_angles:
            self._draw_hip(state)
        if state.show_chart:
            self._draw_chart(state)
        self._draw_bottom(state, now)
        if state.camera_error:
            self._draw_camera_error(state)

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------

    def text(self, family: str, size: int, text: str, color) -> pygame.Surface:
        key = (family, size, text, tuple(color))
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 600:
                self._text_cache.clear()
            surf = self.fonts.get(family, size).render(text, True, color[:3])
            if len(color) == 4:
                surf.set_alpha(color[3])
            self._text_cache[key] = surf
        return surf

    def blit(self, surf, pos, anchor: str = "topleft") -> pygame.Rect:
        rect = surf.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
        self.ui.blit(surf, rect)
        return rect

    def _draw_shapes(self, segments: list, joints: list) -> None:
        """
        Bones and joints with a thin dark outline, so they read on light and
        dark walls. All outlines go first, so one bone's outline never cuts
        across another bone where they meet.
        """
        for _, a, b, width in segments:
            draw.bone(self.ui, OUTLINE, a, b, width + 2)
        for _, p, r in joints:
            draw.circle(self.ui, OUTLINE, p, r + 1)
        for color, a, b, width in segments:
            draw.bone(self.ui, color, a, b, width)
        for color, p, r in joints:
            draw.circle(self.ui, color, p, r)

    # ------------------------------------------------------------------------
    # Skeleton
    # ------------------------------------------------------------------------

    def _body_shapes(self, state: HudState, now: float,
                     segments: list, joints: list) -> None:
        pts = state.points
        if not pts:
            return
        base = draw.with_alpha(BONE, 0.85 if state.armed else 0.55)
        flash = now - self._flash < FLASH_TIME
        hip = ACCENT if (flash or state.hip_latched is not None) else base

        for a, b in BONES:
            if a in pts and b in pts:
                is_hip = (a, b) == ("left_hip", "right_hip")
                segments.append((hip if is_hip else base, pts[a], pts[b],
                                 4 if (is_hip and flash) else 2))
        # Spine: middle of the shoulders to the middle of the hips.
        needed = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if all(n in pts for n in needed):
            top = _mid(pts["left_shoulder"], pts["right_shoulder"])
            bottom = _mid(pts["left_hip"], pts["right_hip"])
            segments.append((draw.with_alpha(BONE, 0.4), top, bottom, 1.5))
        for name, p in pts.items():
            small = name.endswith(("heel", "toe"))
            joints.append((hip if name.endswith("hip") else base, p, 2 if small else 3))

    def _hand_shapes(self, state: HudState, side: str,
                     segments: list, joints: list) -> None:
        pts = state.hands.get(side)
        if not pts:
            return
        if state.together or self._now - self._switch < 0.4:
            color = ACCENT
        else:
            color = draw.with_alpha(BONE, 0.8 if state.armed else 0.5)
        for a, b in HAND_CONNECTIONS:
            segments.append((color, pts[a], pts[b], 1.5))
        for i in FINGERTIPS + (0,):
            joints.append((color, pts[i], 2))

    def _draw_hip(self, state: HudState) -> None:
        """Hip angle, the +-A lines and the waiting timer (H key)."""
        pts = state.points
        angle = state.hip_angle
        if angle is not None and abs(angle) < 0.05:
            angle = 0.0                               # no "-0.0"
        if angle is None or "left_hip" not in pts or "right_hip" not in pts:
            return
        lh, rh = pts["left_hip"], pts["right_hip"]
        cx, cy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
        # +1 when the user's right hip is on the right of the screen (mirrored view)
        flip = 1.0 if rh[0] >= lh[0] else -1.0
        half = max(40.0, abs(rh[0] - lh[0]) / 2 + 26)

        # Ghost lines at +-A: when the hip line reaches one, the sway fires.
        for sign, symbol in ((1, state.right_symbol), (-1, state.left_symbol)):
            theta = math.radians(state.trigger_angle * sign)
            dx, dy = flip * half * math.cos(theta), -flip * half * math.sin(theta)
            a, b = (cx - dx, cy - dy), (cx + dx, cy + dy)
            draw.bone(self.ui, draw.with_alpha(TEXT, 0.35), a, b, 1)
            high = b if sign > 0 else a              # the end that is raised
            tag = self.text("regular", 11, symbol, TEXT_MUTED)
            box = tag.get_rect(center=(round(high[0]), round(high[1] - 11)))
            draw.rounded_rect(self.ui, BACKDROP, box.inflate(8, 2), 3)
            self.blit(tag, box.center, "center")

        names = {"right": state.right_symbol, "left": state.left_symbol,
                 "waiting": "waiting"}
        name = names.get(state.hip_zone, "-")
        done = state.hip_latched is not None
        if done:
            name += " ok"
        label = self.text("mono", 14, f"{angle:+5.1f}\u00b0 {name}", ACCENT if done else TEXT)

        # Waiting timer: a bar to 3t with ticks at t (end) and 3t (space).
        unit = max(state.unit_time, 0.05)
        held = state.wait_for
        bar_w = 118
        box = pygame.Rect(0, 0, max(label.get_width(), bar_w) + 16, 50)
        box.midleft = (round(max(lh[0], rh[0]) + half + 14), round(cy))
        if box.right > self.size[0] - MARGIN:
            box.midright = (round(min(lh[0], rh[0]) - half - 14), round(cy))
        box.clamp_ip(pygame.Rect(0, 0, *self.size))
        draw.rounded_rect(self.ui, BACKDROP, box, 3)
        self.blit(label, (box.x + 8, box.y + 6))

        bar = pygame.Rect(box.x + 8, box.y + 30, bar_w, 4)
        pygame.draw.rect(self.ui, (255, 255, 255, 40), bar)
        fill = min(1.0, held / (3 * unit))
        color = ACCENT if held >= unit else TEXT_MUTED
        pygame.draw.rect(self.ui, color, pygame.Rect(bar.x, bar.y, round(bar.w * fill), bar.h))
        for k, tick in ((1, "end"), (3, "space")):
            x = bar.x + round(bar.w * k / 3) - (1 if k == 3 else 0)
            pygame.draw.rect(self.ui, TEXT, pygame.Rect(x, bar.y - 2, 1, bar.h + 4))
            reached = held >= k * unit
            img = self.text("regular", 10, tick, ACCENT if reached else TEXT_MUTED)
            self.blit(img, (x, bar.bottom + 3), "topright" if k == 3 else "midtop")

    # ------------------------------------------------------------------------
    # Morse chart
    # ------------------------------------------------------------------------

    def _draw_chart(self, state: HudState) -> None:
        # The chart only changes when the typed code changes, so it is cached.
        surf = self._chart_cache.get(state.code)
        if surf is None:
            if len(self._chart_cache) > 64:
                self._chart_cache.clear()
            surf = self._render_chart(state.code)
            self._chart_cache[state.code] = surf
        self.ui.blit(surf, self.rect_chart.topleft)

    def _render_chart(self, prefix: str) -> pygame.Surface:
        r = self.rect_chart
        surf = pygame.Surface(r.size, pygame.SRCALPHA)
        surf.fill(UI_CLEAR)
        draw.rounded_rect(surf, BACKDROP, surf.get_rect(), 4)

        col_x = (10, 64, 118)
        for i, char in enumerate(CHART_CHARS):
            col, row = divmod(i, CHART_ROWS)
            x = col_x[col]
            cy = 14 + row * 15
            code = MORSE_TABLE[char]
            reachable = not prefix or code.startswith(prefix)
            exact = bool(prefix) and code == prefix

            if exact:
                color = ACCENT
            elif reachable:
                color = TEXT
            else:
                color = draw.with_alpha(TEXT, 0.28)
            glyph = self.text("semibold", 11, char, color)
            surf.blit(glyph, glyph.get_rect(midleft=(x, cy)))

            gx = x + 12
            for j, sym in enumerate(code):
                typed = bool(prefix) and reachable and j < len(prefix)
                c = ACCENT if typed else color
                if sym == ".":
                    draw.circle(surf, c, (gx + 1.5, cy), 1.6)
                    gx += 5
                else:
                    pygame.draw.rect(surf, c, pygame.Rect(round(gx), round(cy) - 1, 6, 2))
                    gx += 8
        return surf

    # ------------------------------------------------------------------------
    # Bottom strip: code in progress and message
    # ------------------------------------------------------------------------

    def _draw_bottom(self, state: HudState, now: float) -> None:
        r = self.rect_bottom
        draw.rounded_rect(self.ui, BACKDROP, r, 4)
        x0 = r.x + 12
        cy = r.y + 18

        # Line 1: symbols typed so far and what they decode to.
        x = x0
        for sym in state.code:
            if sym == ".":
                draw.circle(self.ui, TEXT, (x + 4, cy), 4)
                x += 14
            else:
                pygame.draw.rect(self.ui, TEXT, pygame.Rect(x, cy - 2, 18, 5))
                x += 26

        t_invalid, bad = self._invalid
        if now - t_invalid < INVALID_TIME:
            self.blit(self.text("regular", 12, f"{bad}  no match", DANGER), (x0, cy), "midleft")
        elif state.code:
            char = decode(state.code)
            options = candidates(state.code)
            if char is not None:
                hint, color = f"= {char}", ACCENT
            elif options:
                hint = "  ".join(options[:6]) + ("  …" if len(options) > 6 else "")
                color = TEXT_MUTED
            else:
                hint, color = "no match", DANGER
            self.blit(self.text("medium", 13, hint, color), (x + 6, cy), "midleft")
        elif not state.armed:
            self.blit(self.text("regular", 12, "put both hands together to start",
                                TEXT_MUTED), (x0, cy), "midleft")
        elif not state.text:
            hint = (f"hips right = {state.right_symbol}   left = {state.left_symbol}"
                    f"   wait = end / space")
            self.blit(self.text("regular", 12, hint, TEXT_MUTED), (x0, cy), "midleft")

        # Typing switch state, right end of line 1.
        if state.paused:
            status, scol = "paused", TEXT_MUTED
        elif state.armed:
            status, scol = "on", ACCENT
        else:
            status, scol = "off", TEXT_MUTED
        self.blit(self.text("medium", 12, status, scol), (r.right - 12, cy), "midright")

        # Line 2: the message, keeping its end visible when it gets long.
        ty = r.y + 42
        font = self.fonts.get("medium", 20)
        max_w = r.w - 30
        shown = state.text
        while shown and font.size(shown)[0] > max_w:
            shown = shown[1:]
        if shown != state.text:
            shown = "…" + shown[1:]
        caret_x = x0
        if shown:
            rect = self.blit(font.render(shown, True, TEXT), (x0, ty), "midleft")
            caret_x = rect.right + 2
        if (now * 1.6) % 1.0 < 0.6:
            pygame.draw.rect(self.ui, TEXT_MUTED, pygame.Rect(caret_x, ty - 10, 2, 20))

    def _draw_camera_error(self, state: HudState) -> None:
        w, h = self.size
        title = self.text("medium", 14, state.camera_error, TEXT)
        box = pygame.Rect(0, 0, min(w - 2 * MARGIN, max(300, title.get_width() + 32)), 54)
        box.center = (w // 2, h // 2 - 40)
        draw.rounded_rect(self.ui, BACKDROP, box, 4)
        self.blit(title, (box.centerx, box.y + 18), "center")
        self.blit(self.text("regular", 11, "Press X to choose another camera", TEXT_MUTED),
                  (box.centerx, box.y + 38), "center")
