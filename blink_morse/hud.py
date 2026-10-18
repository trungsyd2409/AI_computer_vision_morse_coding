"""
Everything drawn on top of the camera image.

Kept small and flat on purpose: thin bones with a dark outline so they
read on any background, one amber accent for "live", plain text on a
plain dark backdrop. No glow, no glass, no animated pop-ups.

On screen:
* Arm bones (shoulder - elbow - wrist) for both arms.
* A 21-point skeleton for both hands. The control hand turns amber while
  it holds the fist that allows typing.
* A forearm turns amber while it sits on a diagonal that will type, and
  flashes thicker for a moment when its gesture fires.
* Morse chart in the top-right corner.
* Code being typed and the message, in a strip at the bottom.
"""

from dataclasses import dataclass, field
from typing import Optional

import pygame

from . import draw
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
    armed: bool = False                  # control fist held and not paused
    fist: bool = False                   # control hand is a fist
    control_visible: bool = False
    paused: bool = False
    symbol_side: str = "left"
    control_side: str = "right"
    # Window pixels of shoulder, elbow, wrist per side, or None
    arms: dict = field(default_factory=lambda: {"left": None, "right": None})
    # Window pixels of the 21 hand landmarks per side, or None
    hands: dict = field(default_factory=lambda: {"left": None, "right": None})
    angle: dict = field(default_factory=lambda: {"left": None, "right": None})
    zone: dict = field(default_factory=lambda: {"left": None, "right": None})
    ready: dict = field(default_factory=lambda: {"left": False, "right": False})
    latched: dict = field(default_factory=lambda: {"left": None, "right": None})
    code: str = ""
    text: str = ""
    show_chart: bool = True
    show_angles: bool = True


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

        self._flash = {"left": -10.0, "right": -10.0}
        self._invalid = (-10.0, "")

    # ------------------------------------------------------------------------
    # Event hooks (called by the app)
    # ------------------------------------------------------------------------

    def on_gesture(self, side: str, now: float) -> None:
        if side in self._flash:
            self._flash[side] = now

    def on_invalid(self, code: str, now: float) -> None:
        self._invalid = (now, code)

    # ------------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------------

    def draw(self, state: HudState, now: float, visible: bool = True) -> None:
        self.ui.fill(UI_CLEAR)
        if not visible:
            return
        segments, joints = [], []
        for side in ("left", "right"):
            self._arm_shapes(state, side, now, segments, joints)
            self._hand_shapes(state, side, segments, joints)
        self._draw_shapes(segments, joints)
        if state.show_angles:
            for side in ("left", "right"):
                self._draw_angle(state, side)
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

    def _live_zone(self, state: HudState, side: str) -> bool:
        """
        Control arm: on a diagonal that will type once it is held.
        Dot / dash arm: inside a zone that has already been typed.
        """
        if side == state.symbol_side:
            return state.latched.get(side) is not None
        return state.armed and state.ready.get(side) and state.zone.get(side) in ("up", "down")

    def _arm_shapes(self, state: HudState, side: str, now: float,
                    segments: list, joints: list) -> None:
        pts = state.arms.get(side)
        if not pts:
            return
        shoulder, elbow, wrist = pts[:3]
        base = draw.with_alpha(BONE, 0.85 if state.armed else 0.55)
        flash = now - self._flash.get(side, -10.0) < FLASH_TIME
        fore = ACCENT if (flash or self._live_zone(state, side)) else base

        segments.append((base, shoulder, elbow, 2))
        segments.append((fore, elbow, wrist, 4 if flash else 2))
        joints.extend([(base, shoulder, 3), (base, elbow, 3), (fore, wrist, 3)])

    def _hand_shapes(self, state: HudState, side: str,
                     segments: list, joints: list) -> None:
        pts = state.hands.get(side)
        if not pts:
            return
        if side == state.control_side and state.fist:
            color = ACCENT
        else:
            color = draw.with_alpha(BONE, 0.8 if state.armed else 0.5)
        for a, b in HAND_CONNECTIONS:
            segments.append((color, pts[a], pts[b], 1.5))
        for i in FINGERTIPS + (0,):
            joints.append((color, pts[i], 2))

    ZONE_NAMES = {
        "symbol": {"up": "dash", "down": "dot", "level": "level"},
        "control": {"up": "end", "down": "space", "level": "level"},
    }

    def _draw_angle(self, state: HudState, side: str) -> None:
        """Forearm angle and the zone it is in, next to the elbow (H key)."""
        pts = state.arms.get(side)
        angle = state.angle.get(side)
        if not pts or angle is None:
            return
        role = "symbol" if side == state.symbol_side else "control"
        zone = state.zone.get(side)
        name = self.ZONE_NAMES[role].get(zone, "-")
        if role == "symbol":
            # Amber + "ok" once the symbol in this zone has been typed.
            done = state.latched.get(side) is not None
            name += " ok" if done else ""
            color = ACCENT if done else TEXT
        else:
            color = ACCENT if zone in ("up", "down") else TEXT
        label = self.text("mono", 14, f"{angle:+4.0f}\u00b0 {name}", color)

        # Beside the elbow, on the side away from the body.
        (sx, _), (ex, ey) = pts[0], pts[1]
        out = -1 if side == "left" else 1
        if abs(ex - sx) > 8:
            out = 1 if ex > sx else -1
        box = label.get_rect()
        box.inflate_ip(12, 6)
        if out > 0:
            box.midleft = (round(ex + 12), round(ey))
        else:
            box.midright = (round(ex - 12), round(ey))
        box.clamp_ip(pygame.Rect(0, 0, *self.size))
        draw.rounded_rect(self.ui, BACKDROP, box, 3)
        self.blit(label, box.center, "center")

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
        elif state.paused:
            self.blit(self.text("regular", 12, "paused (P)", TEXT_MUTED), (x0, cy), "midleft")
        elif not state.armed:
            self.blit(self.text("regular", 12, f"hold a {state.control_side} fist to type",
                                TEXT_MUTED), (x0, cy), "midleft")

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
        box = pygame.Rect(0, 0, 300, 54)
        box.center = (w // 2, h // 2 - 40)
        draw.rounded_rect(self.ui, BACKDROP, box, 4)
        self.blit(self.text("medium", 14, state.camera_error, TEXT),
                  (box.centerx, box.y + 18), "center")
        self.blit(self.text("regular", 11, "Press X to choose another camera", TEXT_MUTED),
                  (box.centerx, box.y + 38), "center")
