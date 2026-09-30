"""
Everything drawn on top of the camera image.

Layout of the 480 x 800 portrait window (3:5):

    +---------------------------+
    |  Morse chart              |   black strip above the camera
    +---------------------------+
    |                           |
    |   camera, code in centre  |   camera keeps its own aspect ratio
    |                           |
    +---------------------------+
    |  sent messages (bubbles)  |   black strip below the camera
    |  [ message being typed ]  |
    +---------------------------+

The HUD paints into two pygame surfaces every frame:

* `ui`   - transparent layer with text, icons and the hand skeleton.
* `glow` - opaque black layer where bright shapes are added. The GPU
           blurs it and adds it on top of the image, which produces the
           soft light around fingertips, symbols and letters.

The glass panels are not drawn here. The HUD only reports where they are
(`panels`) and the GPU shader renders them.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pygame

from . import draw
from .config import (ACCENT, CAMERA_RECT, DANGER, FINGER_COLORS, FONTS_DIR,
                     TEXT, TEXT_FAINT, TEXT_MUTED)
from .gestures import FINGER_ORDER, FINGER_TIPS, THUMB_TIP
from .morse import MORSE_TABLE, candidates

# Bones of the hand as pairs of landmark indices.
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Characters shown in the reference chart, laid out column by column.
CHART_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
CHART_ROWS = 6
CHART_COLUMNS = 6
PUNCTUATION_HINT = ". , ? ! ' / ( ) : = + - @"

# The UI layer is cleared to this colour with zero alpha. Anti-aliased
# edges blend towards it, so a near-white clear colour keeps light text
# from getting dark fringes.
UI_CLEAR = (232, 236, 244, 0)

BUBBLE_COLOR = (34, 86, 104, 235)       # sent message background
BUBBLE_MAX_WIDTH = 330


@dataclass
class ChatMessage:
    text: str
    time: str                            # "14:05", shown next to the bubble


@dataclass
class HudState:
    """Snapshot of everything the HUD needs for one frame."""

    camera_error: Optional[str] = None
    landmarks: Optional[np.ndarray] = None   # (21, 2+) screen pixels, smoothed
    closeness: dict = field(default_factory=dict)
    active_finger: Optional[str] = None
    contact_point: tuple = (0, 0)
    hold_progress: float = -1.0          # 0..1 while the pinky is held
    code: str = ""
    text: str = ""
    chat: list = field(default_factory=list)   # ChatMessage, oldest first
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
        self._bubble_cache = {}

        cam = pygame.Rect(CAMERA_RECT)
        margin = 12
        self.rect_camera = cam
        # Chart fills the black strip above the camera.
        self.rect_chart = pygame.Rect(margin, margin, w - 2 * margin, cam.top - 2 * margin)
        # Message box sits at the bottom, sent messages stack above it.
        self.rect_message = pygame.Rect(margin, h - margin - 72, w - 2 * margin, 72)
        self.rect_chat = pygame.Rect(margin, cam.bottom + 8, w - 2 * margin,
                                     self.rect_message.top - cam.bottom - 16)
        self.rect_error = pygame.Rect(0, 0, 320, 96)
        self.rect_error.center = cam.center

        # The Morse code in progress, and the letter it becomes, appear in
        # the middle of the camera image.
        self.code_center = cam.center

        # Animation state
        self._pops = []           # (text, color, start, size)
        self._bursts = []         # (x, y, color, start)
        self._shake_start = -10.0
        self._symbol_start = -10.0
        self._message_flash = (-10.0, ACCENT)
        self._sent_start = -10.0
        self.panels = []

    # ------------------------------------------------------------------------
    # Event hooks (called by the app when something happens)
    # ------------------------------------------------------------------------

    def on_touch(self, finger: str, point: tuple, now: float) -> None:
        self._bursts.append((point[0], point[1], FINGER_COLORS[finger], now))

    def on_symbol(self, now: float) -> None:
        self._symbol_start = now

    def on_letter(self, char: str, now: float) -> None:
        self._pops.append((char, TEXT, now, 120))
        self._message_flash = (now, ACCENT)

    def on_space(self, now: float) -> None:
        self._pops.append(("space", FINGER_COLORS["ring"], now, 36))

    def on_invalid(self, code: str, now: float) -> None:
        self._pops.append(("?", DANGER, now, 120))
        self._shake_start = now
        self._message_flash = (now, DANGER)

    def on_clear(self, now: float) -> None:
        self._pops.append(("cleared", FINGER_COLORS["pinky"], now, 36))
        self._message_flash = (now, FINGER_COLORS["pinky"])

    def on_sent(self, now: float) -> None:
        self._sent_start = now
        self._message_flash = (now, ACCENT)

    # ------------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------------

    def draw(self, state: HudState, now: float) -> None:
        self.ui.fill(UI_CLEAR)
        self.glow.fill((0, 0, 0))
        self._build_panels(state, now)

        if state.landmarks is not None:
            self._draw_hand(state, now)
        if state.show_chart:
            self._draw_chart(state)
        self._draw_code(state, now)
        self._draw_chat(state, now)
        self._draw_message(state, now)
        if state.camera_error:
            self._draw_camera_error(state)

        self._draw_bursts(now)
        self._draw_hold(state)
        self._draw_pops(now)

    def _build_panels(self, state: HudState, now: float) -> None:
        flash_t, flash_color = self._message_flash
        flash = max(0.0, 1.0 - (now - flash_t) / 0.6)
        self.panels = [Panel(self.rect_message, flash, flash_color)]
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
            surf.blit(hint, (header.get_width() + 26, 13))

        # Six columns of six characters fill the wide, short strip above the
        # camera. Each column is wide enough for a five-symbol digit code.
        col_w = (r.w - 28) / CHART_COLUMNS
        dot_color = TEXT
        for i, char in enumerate(CHART_CHARS):
            col, row = divmod(i, CHART_ROWS)
            x = round(14 + col * col_w)
            cy = 48 + row * 26
            code = MORSE_TABLE[char]

            reachable = not prefix or code.startswith(prefix)
            exact = bool(prefix) and code == prefix
            alpha = 1.0 if reachable else 0.22

            if exact:
                pill = pygame.Rect(x - 6, cy - 10, round(col_w) - 2, 20)
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
        surf.blit(foot, foot.get_rect(topright=(r.w - 14, 16)))
        return surf

    # ------------------------------------------------------------------------
    # Morse code in the middle of the camera
    # ------------------------------------------------------------------------

    def _draw_code(self, state: HudState, now: float) -> None:
        """
        The dots and dashes of the letter in progress, centred on the camera
        image with no box around them. They stay until the letter ends.
        """
        code = state.code
        if not code:
            return
        widths = [22 if sym == "." else 48 for sym in code]
        gap = 14
        total = sum(widths) + gap * (len(code) - 1)
        cx, cy = self.code_center
        x = cx - total / 2

        dt = now - self._shake_start
        if dt < 0.35:
            x += math.sin(dt * 60) * 8 * (1 - dt / 0.35)

        t = now - self._symbol_start
        for i, (sym, width) in enumerate(zip(code, widths)):
            grow = draw.ease_out_back(t / 0.2) if i == len(code) - 1 else 1.0
            mid = (x + width / 2, cy)
            if sym == ".":
                color = FINGER_COLORS["index"]
                draw.glow(self.glow, color, mid, 26, 0.7 * grow)
                draw.circle(self.ui, color, mid, 11 * grow)
            else:
                color = FINGER_COLORS["middle"]
                draw.glow(self.glow, color, mid, 32, 0.7 * grow)
                draw.capsule(self.ui, color, mid, 48 * grow, 18 * grow)
            x += width + gap

    # ------------------------------------------------------------------------
    # Message box and sent messages
    # ------------------------------------------------------------------------

    def _draw_message(self, state: HudState, now: float) -> None:
        r = self.rect_message
        x0 = r.x + 16
        self.blit(self.label("Message"), (x0, r.y + 12))
        hint = "Enter to send" if state.text else ""
        if hint:
            self.blit(self.text("regular", 11, hint, TEXT_FAINT), (r.right - 16, r.y + 11),
                      "topright")

        font = self.fonts.get("medium", 22)
        max_w = r.w - 44
        shown = state.text
        # Keep the end of the message visible when it gets too long.
        while shown and font.size(shown)[0] > max_w:
            shown = shown[1:]
        if shown != state.text:
            shown = "\u2026" + shown[1:]

        ty = r.y + 47
        if shown:
            surf = font.render(shown, True, TEXT)
            rect = self.blit(surf, (x0, ty), "midleft")
            caret_x = rect.right + 3
        else:
            self.blit(self.text("regular", 18, "Your message appears here", TEXT_FAINT),
                      (x0 + 6, ty), "midleft")
            caret_x = x0

        if (now * 1.8) % 1.0 < 0.6:
            caret = pygame.Rect(0, 0, 2, 22)
            caret.midleft = (caret_x, ty)
            pygame.draw.rect(self.ui, ACCENT, caret, border_radius=1)
            draw.glow(self.glow, ACCENT, caret.center, 10, 0.5)

    def _bubble(self, message: ChatMessage) -> pygame.Surface:
        """Render one sent message as a rounded bubble, word-wrapped."""
        key = (message.text, message.time)
        surf = self._bubble_cache.get(key)
        if surf is not None:
            return surf
        if len(self._bubble_cache) > 200:
            self._bubble_cache.clear()

        font = self.fonts.get("regular", 15)
        lines = _wrap(message.text, font, BUBBLE_MAX_WIDTH - 28)
        line_h = font.get_linesize()
        text_w = max(font.size(line)[0] for line in lines)
        w, h = text_w + 28, line_h * len(lines) + 18
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill(UI_CLEAR)
        # Rounded on all corners except the bottom right, like a sent message.
        pygame.draw.rect(surf, BUBBLE_COLOR, surf.get_rect(), border_radius=14,
                         border_bottom_right_radius=4)
        for i, line in enumerate(lines):
            surf.blit(font.render(line, True, TEXT), (14, 9 + i * line_h))
        self._bubble_cache[key] = surf
        return surf

    def _draw_chat(self, state: HudState, now: float) -> None:
        """Sent messages, newest at the bottom, older ones pushed upwards."""
        area = self.rect_chat
        if not state.chat:
            return
        self.ui.set_clip(area)
        y = area.bottom
        slide = 0.0
        t = now - self._sent_start
        if t < 0.25:
            # The newest bubble slides up into place.
            slide = 18 * (1 - draw.ease_out_cubic(t / 0.25))

        for i, message in enumerate(reversed(state.chat)):
            bubble = self._bubble(message)
            rect = bubble.get_rect(bottomright=(area.right, round(y + slide)))
            if i == 0 and t < 0.25:
                bubble.set_alpha(int(255 * draw.ease_out_cubic(t / 0.25)))
            else:
                bubble.set_alpha(255)
            self.ui.blit(bubble, rect)
            stamp = self.text("regular", 10, message.time, TEXT_FAINT)
            self.ui.blit(stamp, stamp.get_rect(bottomright=(rect.left - 8, rect.bottom - 2)))
            y = rect.top - 8
            if y < area.top:
                break
        self.ui.set_clip(None)

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
        cx, cy = self.code_center
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


def _wrap(text: str, font: pygame.font.Font, width: int) -> list:
    """Split `text` into lines no wider than `width` pixels."""
    lines, line = [], ""
    for word in text.split(" "):
        candidate = f"{line} {word}" if line else word
        if font.size(candidate)[0] <= width:
            line = candidate
            continue
        if line:
            lines.append(line)
        # A single word longer than the line is cut into pieces.
        while font.size(word)[0] > width:
            cut = len(word)
            while cut > 1 and font.size(word[:cut])[0] > width:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        line = word
    lines.append(line)
    return lines
