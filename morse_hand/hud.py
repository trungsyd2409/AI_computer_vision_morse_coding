"""
Everything drawn on top of the camera image.

The HUD paints into two pygame surfaces every frame:

* `ui`   - transparent layer with text, icons and the hand skeleton.
* `glow` - opaque black layer where bright shapes are added. The GPU
           blurs it and adds it on top of the image, which produces the
           soft light around fingertips and letters.

The frosted glass panels are not drawn here. The HUD only reports where
they are (`panels`) and the GPU shader renders them, because blurring the
camera behind a panel is far cheaper on the graphics card.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pygame

from . import draw
from .config import (ACCENT, DANGER, FINGER_COLORS, FONTS_DIR, MIN_ACCEPTABLE_FPS,
                     SUCCESS, TEXT, TEXT_FAINT, TEXT_MUTED, WARNING)
from .gestures import FINGER_ORDER, FINGER_TIPS, THUMB_TIP
from .morse import MORSE_TABLE, candidates, decode

# Bones of the hand as pairs of landmark indices.
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

LEGEND_ROWS = [
    ("index", "Index", "dot"),
    ("middle", "Middle", "dash"),
    ("ring", "Ring", "end  ·  ×2 space"),
    ("pinky", "Pinky", "delete  ·  hold clear"),
]

# Characters shown in the reference chart, laid out column by column.
CHART_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
CHART_ROWS = 12
PUNCTUATION_HINT = ". , ? ! ' / ( ) : = + - @"

# The UI layer is cleared to this colour with zero alpha. Anti-aliased
# edges blend towards it, so a near-white clear colour keeps light text
# from getting dark fringes.
UI_CLEAR = (232, 236, 244, 0)


@dataclass
class HudState:
    """Snapshot of everything the HUD needs for one frame."""

    fps: float = 0.0
    camera_error: Optional[str] = None
    hand: str = "none"                   # "right", "left_only" or "none"
    landmarks: Optional[np.ndarray] = None   # (21, 2+) screen pixels, smoothed
    closeness: dict = field(default_factory=dict)
    active_finger: Optional[str] = None
    contact_point: tuple = (0, 0)
    hold_progress: float = -1.0          # 0..1 while the pinky is held
    code: str = ""
    text: str = ""
    space_progress: float = -1.0         # double-tap countdown, -1 if idle
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
        self.rect_title = pygame.Rect(margin, margin, 236, 60)
        self.rect_legend = pygame.Rect(margin, 88, 236, 132)
        self.rect_fps = pygame.Rect(w - margin - 100, margin, 100, 32)
        self.rect_chart = pygame.Rect(w - margin - 212, 60, 212, 362)
        self.rect_bottom = pygame.Rect(margin, h - margin - 150, w - 2 * margin, 150)
        self.rect_error = pygame.Rect(0, 0, 320, 96)
        self.rect_error.center = (w // 2, h // 2 - 40)

        # Point between the two side panels where letters pop up.
        self.pop_center = ((self.rect_legend.right + self.rect_chart.left) / 2, 250)

        # Animation state
        self._pops = []           # (text, color, start, size)
        self._bursts = []         # (x, y, color, start)
        self._shake_start = -10.0
        self._symbol_start = -10.0
        self._bottom_flash = (-10.0, ACCENT)
        self.panels = []

    # ------------------------------------------------------------------------
    # Event hooks (called by the app when something happens)
    # ------------------------------------------------------------------------

    def on_touch(self, finger: str, point: tuple, now: float) -> None:
        self._bursts.append((point[0], point[1], FINGER_COLORS[finger], now))

    def on_symbol(self, now: float) -> None:
        self._symbol_start = now

    def on_letter(self, char: str, now: float) -> None:
        self._pops.append((char, TEXT, now, 140))
        self._bottom_flash = (now, ACCENT)

    def on_space(self, now: float) -> None:
        self._pops.append(("space", FINGER_COLORS["ring"], now, 40))

    def on_invalid(self, code: str, now: float) -> None:
        self._pops.append(("?", DANGER, now, 140))
        self._shake_start = now
        self._bottom_flash = (now, DANGER)

    def on_clear(self, now: float) -> None:
        self._pops.append(("cleared", FINGER_COLORS["pinky"], now, 40))
        self._bottom_flash = (now, FINGER_COLORS["pinky"])

    # ------------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------------

    def draw(self, state: HudState, now: float) -> None:
        self.ui.fill(UI_CLEAR)
        self.glow.fill((0, 0, 0))
        self._build_panels(state, now)

        if state.landmarks is not None:
            self._draw_hand(state, now)

        self._draw_title(state)
        self._draw_legend(state)
        self._draw_fps(state)
        if state.show_chart:
            self._draw_chart(state)
        self._draw_bottom(state, now)
        if state.camera_error:
            self._draw_camera_error(state)

        self._draw_bursts(now)
        self._draw_hold(state)
        self._draw_pops(now)

    def _build_panels(self, state: HudState, now: float) -> None:
        flash_t, flash_color = self._bottom_flash
        flash = max(0.0, 1.0 - (now - flash_t) / 0.6)
        self.panels = [
            Panel(self.rect_title),
            Panel(self.rect_legend),
            Panel(self.rect_fps),
            Panel(self.rect_bottom, flash, flash_color),
        ]
        if state.show_chart:
            self.panels.append(Panel(self.rect_chart))
        if state.camera_error:
            self.panels.append(Panel(self.rect_error, 0.6, DANGER))

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
    # Hand
    # ------------------------------------------------------------------------

    def _draw_hand(self, state: HudState, now: float) -> None:
        pts = state.landmarks[:, :2]
        bone_color = (236, 241, 255, 120)
        for a, b in HAND_BONES:
            draw.line(self.ui, bone_color, pts[a], pts[b], 2)
        for i, p in enumerate(pts):
            if i in FINGER_TIPS.values() or i == THUMB_TIP:
                continue
            draw.circle(self.ui, (236, 241, 255, 200), p, 2.5)

        thumb = pts[THUMB_TIP]

        # Thin "tether" from the thumb to whichever finger is closest, so
        # the user can see which gesture is about to fire.
        if state.closeness:
            nearest = max(state.closeness, key=state.closeness.get)
            c = state.closeness[nearest]
            if c > 0.35 and state.active_finger is None:
                tip = pts[FINGER_TIPS[nearest]]
                color = FINGER_COLORS[nearest]
                draw.line(self.ui, draw.with_alpha(color, (c - 0.35) * 1.4), thumb, tip, 2)

        for finger in FINGER_ORDER:
            tip = pts[FINGER_TIPS[finger]]
            color = FINGER_COLORS[finger]
            c = state.closeness.get(finger, 0.0)
            active = state.active_finger == finger
            radius = 4.5 + 3.0 * c + (2.0 if active else 0.0)
            draw.glow(self.glow, color, tip, 14 + 18 * c, 0.25 + 0.6 * c)
            draw.circle(self.ui, color, tip, radius)
            draw.circle(self.ui, (255, 255, 255), tip, max(1.5, radius * 0.38))

        draw.glow(self.glow, (255, 255, 255), thumb, 16, 0.3)
        draw.circle(self.ui, (255, 255, 255), thumb, 5.5)

        if state.active_finger is not None:
            color = FINGER_COLORS[state.active_finger]
            pulse = 0.5 + 0.5 * math.sin(now * 10.0)
            draw.glow(self.glow, color, state.contact_point, 28 + 5 * pulse, 0.55)
            draw.ring(self.ui, draw.with_alpha(color, 0.9), state.contact_point, 13, 2)

    # ------------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------------

    def _draw_title(self, state: HudState) -> None:
        r = self.rect_title
        self.blit(self.text("semibold", 17, "Morse Hand", TEXT), (r.x + 16, r.y + 10))

        if state.camera_error:
            dot, msg = DANGER, "Camera unavailable"
        elif state.hand == "right":
            dot, msg = SUCCESS, "Right hand tracked"
        elif state.hand == "left_only":
            dot, msg = WARNING, "Left hand ignored, use right"
        else:
            dot, msg = TEXT_FAINT, "Show your right hand"

        cy = r.y + 43
        draw.circle(self.ui, dot, (r.x + 20, cy), 3.5)
        if dot is not TEXT_FAINT:
            draw.glow(self.glow, dot, (r.x + 20, cy), 10, 0.6)
        self.blit(self.text("regular", 12, msg, TEXT_MUTED), (r.x + 30, cy), "midleft")

    def _draw_legend(self, state: HudState) -> None:
        r = self.rect_legend
        for i, (finger, name, action) in enumerate(LEGEND_ROWS):
            cy = r.y + 22 + i * 29
            color = FINGER_COLORS[finger]
            if state.active_finger == finger:
                pill = pygame.Rect(r.x + 8, cy - 12, r.w - 16, 24)
                draw.rounded_rect(self.ui, draw.with_alpha(color, 0.16), pill, 8)
            draw.circle(self.ui, color, (r.x + 20, cy), 4.5)
            draw.glow(self.glow, color, (r.x + 20, cy), 9, 0.35)
            self.blit(self.text("medium", 13, name, TEXT), (r.x + 34, cy), "midleft")
            self.blit(self.text("regular", 12, action, TEXT_MUTED),
                      (r.right - 14, cy), "midright")

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

        # Row 2: the dots and dashes typed so far
        cy = r.y + 56
        shake = 0.0
        dt = now - self._shake_start
        if dt < 0.35:
            shake = math.sin(dt * 60) * 7 * (1 - dt / 0.35)

        if state.code:
            self._draw_code(state.code, x0 + shake, cy, now)
            self._draw_prediction(state.code, r.right - 20, cy)
        elif state.space_progress >= 0:
            color = FINGER_COLORS["ring"]
            draw.arc(self.ui, color, (x0 + 9, cy), 8, 0,
                     2 * math.pi * (1 - state.space_progress), 2.5)
            self.blit(self.text("medium", 14, "Tap ring again to add a space", color),
                      (x0 + 28, cy), "midleft")
        else:
            self.blit(self.text("regular", 14,
                                "Touch your thumb to a finger to start a letter",
                                TEXT_FAINT), (x0 + shake, cy), "midleft")

        # Divider
        pygame.draw.line(self.ui, (255, 255, 255, 22),
                         (x0, r.y + 84), (r.right - 20, r.y + 84))

        # Row 3: the decoded message with a blinking caret
        self.blit(self.label("Message"), (x0, r.y + 96))
        if state.text:
            count = self.text("regular", 11, f"{len(state.text)} chars", TEXT_FAINT)
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

    def _draw_code(self, code: str, x: float, cy: float, now: float) -> None:
        for i, sym in enumerate(code):
            is_last = i == len(code) - 1
            grow = 1.0
            if is_last:
                grow = draw.ease_out_back((now - self._symbol_start) / 0.22)
            if sym == ".":
                color = FINGER_COLORS["index"]
                draw.glow(self.glow, color, (x + 8, cy), 18, 0.55 * grow)
                draw.circle(self.ui, color, (x + 8, cy), 7.5 * grow)
                x += 28
            else:
                color = FINGER_COLORS["middle"]
                draw.glow(self.glow, color, (x + 17, cy), 24, 0.55 * grow)
                draw.capsule(self.ui, color, (x + 17, cy), 34 * grow, 13 * grow)
                x += 48

    def _draw_prediction(self, code: str, right: float, cy: float) -> None:
        char = decode(code)
        if char is not None:
            glyph = self.text("display", 34, char, TEXT)
            rect = self.blit(glyph, (right, cy), "midright")
            draw.glow(self.glow, ACCENT, rect.center, 30, 0.45)
            self.blit(self.text("regular", 12, "ring to confirm", TEXT_MUTED),
                      (rect.left - 14, cy), "midright")
            return

        options = candidates(code)
        if options:
            preview = "  ".join(options[:6]) + ("  …" if len(options) > 6 else "")
            self.blit(self.text("medium", 15, preview, TEXT_MUTED), (right, cy), "midright")
        else:
            self.blit(self.text("medium", 14, "no match", DANGER), (right, cy), "midright")

    def _draw_key_hints(self, right: float, cy: float, state: HudState) -> None:
        hints = [("X", "settings"), ("H", "chart"),
                 ("M", "sound off" if state.muted else "sound")]
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

    def _draw_bursts(self, now: float) -> None:
        alive = []
        for x, y, color, start in self._bursts:
            t = (now - start) / 0.45
            if t >= 1.0:
                continue
            alive.append((x, y, color, start))
            e = draw.ease_out_cubic(t)
            draw.ring(self.ui, draw.with_alpha(color, 1.0 - t), (x, y), 10 + 34 * e, 2)
            draw.glow(self.glow, color, (x, y), 26 + 26 * e, 0.6 * (1.0 - t))
        self._bursts = alive

    def _draw_hold(self, state: HudState) -> None:
        if state.hold_progress < 0 or state.active_finger != "pinky":
            return
        color = FINGER_COLORS["pinky"]
        center = state.contact_point
        draw.ring(self.ui, (255, 255, 255, 40), center, 24, 3)
        draw.arc(self.ui, color, center, 24, 0, 2 * math.pi * state.hold_progress, 3.5)
        draw.glow(self.glow, color, center, 40, 0.6 * state.hold_progress)
        self.blit(self.text("medium", 11, "hold to clear", color),
                  (center[0], center[1] + 40), "center")

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
