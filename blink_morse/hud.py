"""
Everything drawn on top of the camera image.

Apart from one small readout next to the mouth, nothing is drawn on the
face; all other feedback lives in the panels around it. The top-left panel
shows a redrawn face mesh with the head angles for debugging (G switches it
to the timing legend). The HUD paints
into two pygame surfaces every frame:

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

import numpy as np
import pygame

from . import draw
from .config import (ACCENT, ACTION_COLORS, DANGER, FONTS_DIR, MIN_ACCEPTABLE_FPS,
                     SUCCESS, TEXT, TEXT_FAINT, TEXT_MUTED, WARNING)
from .face_view import FaceMeshView
from .morse import MORSE_TABLE, candidates, decode
from .fusion import quality_label
from .smile import FEATURES

# Characters shown in the reference chart, laid out column by column.
CHART_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
CHART_ROWS = 12
PUNCTUATION_HINT = ". , ? ! ' / ( ) : = + - @"

# The UI layer is cleared to this colour with zero alpha. Anti-aliased
# edges blend towards it, so a near-white clear colour keeps light text
# from getting dark fringes.
UI_CLEAR = (232, 236, 244, 0)

FLASH_TIME = 0.35          # how long a legend row stays lit after an action
CHIP_SIZE = (66, 28)       # smile readout next to the mouth
CHIP_GAP = 14              # distance between the mouth corner and the readout
SCORE_MAX = 1.5            # smile meters run from 0 to 150 % of a typical smile

# Colours of the calibration prompts.
PROMPT_COLORS = {"relax": ACTION_COLORS["word"], "smile": ACTION_COLORS["dash"]}


@dataclass
class HudState:
    """Snapshot of everything the HUD needs for one frame."""

    fps: float = 0.0
    camera_error: Optional[str] = None
    face: bool = False                   # a face is being tracked
    signal: object = None                # SignalResult of the last frame
    calibration: object = None           # Calibration in use, or None
    chip: Optional[tuple] = None         # screen position of the smile readout
    threshold: float = 0.5               # smile score that counts as pressed
    pressed: bool = False                # inside a smile
    pressed_time: float = 0.0            # length of the current smile so far
    paused: bool = False                 # input ignored until P is pressed
    pause_kind: Optional[str] = None     # "letter", "space" or None
    pause_progress: float = 0.0          # 0..1 towards `pause_kind`
    pause_left: float = 0.0              # seconds until it happens
    dash_after: float = 0.375            # smile length that makes a dash
    letter_gap: float = 0.75
    word_gap: float = 1.75
    code: str = ""
    text: str = ""
    muted: bool = False
    show_chart: bool = True
    show_debug: bool = False             # feature weights instead of the chart
    logging: bool = False                # features are being written to CSV
    show_legend: bool = False            # timing legend instead of the face mesh
    face_points: Optional[np.ndarray] = None   # landmarks in camera pixels
    yaw: float = 0.0                     # head turn, + = towards screen right
    pitch: float = 0.0                   # head nod, + = looking down
    yaw_ref: float = 0.0                 # calibrated pose the gate compares with
    pitch_ref: float = 0.0
    max_yaw: float = 20.0                # gate limits around the calibrated pose
    max_pitch: float = 12.0

    @property
    def calibrating(self) -> bool:
        return self.signal is not None and self.signal.calibrating

    @property
    def level(self) -> float:
        return self.signal.score if (self.face and self.signal is not None) else 0.0


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
        self.rect_face = pygame.Rect(margin, 88, 248, 232)
        self.rect_meters = pygame.Rect(margin, self.rect_face.bottom + 12, 248, 80)
        self.face_view = FaceMeshView()
        self.rect_fps = pygame.Rect(w - margin - 100, margin, 100, 32)
        self.rect_chart = pygame.Rect(w - margin - 212, 60, 212, 362)
        self.rect_bottom = pygame.Rect(margin, h - margin - 150, w - 2 * margin, 150)
        self.rect_error = pygame.Rect(0, 0, 320, 96)
        self.rect_error.center = (w // 2, h // 2 - 40)
        mid = (self.rect_legend.right + self.rect_chart.left) // 2
        self.rect_calib = pygame.Rect(0, 0, 280, 84)
        self.rect_calib.midtop = (mid, margin)

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

    def on_calibrated(self, label: str, now: float) -> None:
        color = SUCCESS if label == "good" else WARNING
        self._pops.append((f"ready, {label}", color, now, 40))
        self._bottom_flash = (now, color)

    def on_recentred(self, now: float) -> None:
        self._pops.append(("face re-centred", WARNING, now, 32))

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
        self._draw_chip(state)

        self._draw_title(state)
        if state.show_legend:
            self._draw_legend(state, now)
        else:
            self._draw_face(state)
        self._draw_meters(state)
        self._draw_fps(state)
        if state.show_debug:
            self._draw_debug(state)
        elif state.show_chart:
            self._draw_chart(state)
        if state.calibrating:
            self._draw_calibration(state)
        self._draw_bottom(state, now)
        if state.camera_error:
            self._draw_camera_error(state)
        self._draw_pops(now)

    def _build_panels(self, state: HudState, now: float) -> None:
        top = self.rect_legend if state.show_legend else self.rect_face
        self.rect_meters.y = top.bottom + 12
        flash_t, flash_color = self._bottom_flash
        flash = max(0.0, 1.0 - (now - flash_t) / 0.6)
        self.panels = [
            Panel(self.rect_title),
            Panel(self.rect_legend if state.show_legend else self.rect_face),
            Panel(self.rect_meters),
            Panel(self.rect_fps),
            Panel(self.rect_bottom, flash, flash_color),
        ]
        if state.show_chart or state.show_debug:
            self.panels.append(Panel(self.rect_chart))
        if state.calibrating:
            color = PROMPT_COLORS.get(state.signal.step, ACCENT)
            self.panels.append(Panel(self.rect_calib, 0.8, color))
        if state.camera_error:
            self.panels.append(Panel(self.rect_error, 0.6, DANGER))
        # Glass readout next to the mouth; the border lights up during a
        # smile, in the colour of the symbol it is about to type.
        rect = self._chip_rect(state)
        if rect is not None:
            self.panels.append(Panel(rect, 0.9 if state.pressed else 0.0,
                                     self._press_color(state)))

    @staticmethod
    def _press_color(state: HudState) -> tuple:
        """
        Colour of a smile in progress: what it will type if the face relaxes
        now (dot, then dash once it is long enough).
        """
        if state.pressed:
            return ACTION_COLORS["dash" if state.pressed_time >= state.dash_after else "dot"]
        return ACTION_COLORS["dot"]

    def _chip_rect(self, state: HudState):
        if not state.face or state.chip is None:
            return None
        rect = pygame.Rect((0, 0), CHIP_SIZE)
        rect.center = (round(state.chip[0]), round(state.chip[1]))
        return rect

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

    def _draw_chip(self, state: HudState) -> None:
        """Smile as a percentage of a typical smile, beside the mouth."""
        rect = self._chip_rect(state)
        if rect is None:
            return
        color = self._press_color(state)
        dot = (rect.x + 13, rect.centery)
        draw.circle(self.ui, color if state.pressed else TEXT_MUTED, dot, 3.5)
        if state.pressed:
            draw.glow(self.glow, color, dot, 14, 0.8)
        if state.calibrating:
            text, text_color = "cal", TEXT_MUTED
        elif state.signal is not None and state.signal.gated:
            text, text_color = "off", TEXT_FAINT
        else:
            text = f"{min(999, round(state.level * 100)):3d}%"
            text_color = color if state.pressed else TEXT
        self.blit(self.text("mono", 13, text, text_color),
                  (rect.right - 10, rect.centery), "midright")

    def _draw_title(self, state: HudState) -> None:
        r = self.rect_title
        self.blit(self.text("semibold", 17, "Smile Morse", TEXT), (r.x + 16, r.y + 10))
        if state.logging:
            draw.circle(self.ui, DANGER, (r.right - 46, r.y + 21), 3.5)
            draw.glow(self.glow, DANGER, (r.right - 46, r.y + 21), 10, 0.6)
            self.blit(self.text("semibold", 11, "REC", DANGER),
                      (r.right - 16, r.y + 21), "midright")

        gated = state.signal.gated if (state.face and state.signal is not None) else None
        if state.camera_error:
            dot, msg = DANGER, "Camera unavailable"
        elif not state.face:
            dot, msg = TEXT_FAINT, "Look at the camera"
        elif state.calibrating:
            dot, msg = ACCENT, "Calibrating, follow the prompt"
        elif state.paused:
            dot, msg = WARNING, "Paused, press P to listen"
        elif gated:
            dot, msg = WARNING, {"head": "Face the camera"}.get(gated, "Input ignored")
        else:
            dot, msg = SUCCESS, "Listening"

        cy = r.y + 43
        draw.circle(self.ui, dot, (r.x + 20, cy), 3.5)
        if dot is not TEXT_FAINT:
            draw.glow(self.glow, dot, (r.x + 20, cy), 10, 0.6)
        self.blit(self.text("regular", 12, msg, TEXT_MUTED), (r.x + 30, cy), "midleft")

    def _legend_rows(self, state: HudState) -> list:
        def sec(v):
            return f"{round(v, 2):g} s"
        return [
            ("dot", "Short smile", f"< {sec(state.dash_after)}  \u00b7"),
            ("dash", "Long smile", f"\u2265 {sec(state.dash_after)}  \u2013"),
            ("letter", f"Relax {sec(state.letter_gap)}", "end letter"),
            ("word", f"Relax {sec(state.word_gap)}", "space"),
        ]

    def _draw_face(self, state: HudState) -> None:
        """
        The tracked face redrawn as a mesh on the left, head angles on the
        right. Debug view: shows how steady the landmarks really are.
        """
        r = self.rect_face
        self.blit(self.label("Face mesh"), (r.x + 16, r.y + 12))
        area = pygame.Rect(r.x + 8, r.y + 28, 136, r.h - 36)
        pts = state.face_points if state.face else None
        if pts is None:
            self.blit(self.text("regular", 12, "No face", TEXT_FAINT), area.center, "center")
        else:
            color = self._press_color(state) if state.pressed else ACCENT
            fill = 0.15 + 0.55 * min(1.0, state.level)
            surf = self.face_view.render(pts, area.size, color, fill)
            self.ui.blit(surf, area.topleft)
            if state.pressed:
                for cx, cy in self.face_view.highlight_centres:
                    draw.glow(self.glow, color, (area.x + cx, area.y + cy), 12, 0.7)

        x = r.x + 150
        rows = (
            ("Yaw", state.yaw, state.yaw_ref, state.max_yaw, ("left", "right")),
            ("Pitch", state.pitch, state.pitch_ref, state.max_pitch, ("up", "down")),
        )
        for i, (name, value, ref, limit, words) in enumerate(rows):
            y = r.y + 36 + i * 92
            self.blit(self.label(name), (x, y))
            self.blit(self.text("regular", 10, f"\u00b1{limit:.0f}\u00b0", TEXT_MUTED),
                      (r.right - 12, y), "topright")
            if pts is None:
                self.blit(self.text("semibold", 20, "-", TEXT_FAINT), (x, y + 14))
                continue
            calibrated = state.calibration is not None
            delta = value - ref
            over = calibrated and abs(delta) > limit
            color = WARNING if over else TEXT
            self.blit(self.text("semibold", 20, f"{abs(value):.1f}\u00b0", color), (x, y + 14))
            direction = words[1] if value > 0.5 else words[0] if value < -0.5 else "centre"
            self.blit(self.text("regular", 12, direction, TEXT_MUTED), (x, y + 42))
            note = f"\u0394 calib {delta:+.1f}\u00b0" if calibrated else "not calibrated"
            self.blit(self.text("regular", 11, note, WARNING if over else TEXT_FAINT),
                      (x, y + 60))

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
        """Smile bar with the threshold marked, and the calibration status."""
        r = self.rect_meters
        self.blit(self.label("Smile"), (r.x + 16, r.y + 12))
        x0, x1 = r.x + 16, r.right - 62
        cy = r.y + 40
        value = state.level
        color = self._press_color(state)

        track = pygame.Rect(x0, cy - 3, x1 - x0, 6)
        draw.rounded_rect(self.ui, (255, 255, 255, 30), track, 3)
        fill = track.copy()
        fill.w = max(6, round(track.w * min(1.0, value / SCORE_MAX)))
        draw.rounded_rect(self.ui, draw.with_alpha(color, 0.75), fill, 3)
        if state.pressed:
            draw.glow(self.glow, color, (fill.right - 4, cy), 16, 0.7)
        tx = x0 + track.w * min(1.0, state.threshold / SCORE_MAX)
        draw.line(self.ui, (255, 255, 255, 190), (tx, cy - 7), (tx, cy + 7), 1)
        # A faint tick where an average calibration smile lands.
        ax = x0 + track.w / SCORE_MAX
        draw.line(self.ui, (255, 255, 255, 70), (ax, cy - 5), (ax, cy + 5), 1)

        if not state.face:
            status = "-"
        elif state.calibrating:
            status = "cal"
        else:
            status = f"{round(value * 100)}%"
        self.blit(self.text("regular", 11, status, color if state.pressed else TEXT_MUTED),
                  (r.right - 14, cy), "midright")

        cal = state.calibration
        label = quality_label(cal)
        if cal is None:
            line, line_color = "Not calibrated yet", TEXT_FAINT
        elif cal.fallback:
            line, line_color = "Calibration: default weights, press C", WARNING
        else:
            line_color = SUCCESS if label == "good" else WARNING
            line = f"Calibration: {label}, d\u2032 {cal.quality:.1f}"
        self.blit(self.text("regular", 11, line, line_color), (r.x + 16, r.y + 62), "midleft")

    def _draw_calibration(self, state: HudState) -> None:
        """Big prompt telling the user to relax or smile, with a progress bar."""
        r = self.rect_calib
        n = state.signal
        step = n.step or "relax"
        color = PROMPT_COLORS.get(step, ACCENT)
        title = "Smile, lips closed" if step == "smile" else "Relax your face"
        self.blit(self.text("semibold", 20, title, color), (r.centerx, r.y + 28), "center")
        draw.glow(self.glow, color, (r.centerx, r.y + 28), 40, 0.25)
        sub = f"Calibration {n.step_index + 1} / {n.step_count}  \u00b7  keep your head still"
        self.blit(self.text("regular", 11, sub, TEXT_MUTED), (r.centerx, r.y + 52), "center")
        track = pygame.Rect(r.x + 20, r.y + 66, r.w - 40, 5)
        draw.rounded_rect(self.ui, (255, 255, 255, 30), track, 2)
        fill = track.copy()
        fill.w = max(5, round(track.w * n.step_progress))
        draw.rounded_rect(self.ui, draw.with_alpha(color, 0.85), fill, 2)

    def _draw_debug(self, state: HudState) -> None:
        """Each feature's weight, current z-score and share of the score (F key)."""
        r = self.rect_chart
        x0 = r.x + 14
        self.blit(self.label("Features"), (x0, r.y + 14))
        self.blit(self.text("regular", 11, "share   now z", TEXT_FAINT),
                  (r.right - 14, r.y + 12), "topright")
        cal = state.calibration
        sig = state.signal
        for i, name in enumerate(FEATURES):
            cy = r.y + 50 + i * 40
            w = cal.weights[i] if cal is not None else 0.0
            # Share of an average calibration smile that this feature
            # provides (the shares add up to 100 %).
            share = w * cal.effect[i] * 100 if cal is not None else 0.0
            live = sig is not None and state.face and len(sig.z) == len(FEATURES)
            z = sig.z[i] if live else 0.0
            part = sig.contributions[i] if live else 0.0
            dim = abs(w) < 1e-4
            self.blit(self.text("medium", 13, name, TEXT_FAINT if dim else TEXT),
                      (x0, cy), "midleft")
            self.blit(self.text("mono", 11, f"{share:4.0f}% {z:6.1f}",
                                TEXT_FAINT if dim else TEXT_MUTED),
                      (r.right - 14, cy), "midright")
            # Contribution to the score: centre line = 0, full half = 1.0.
            bar = pygame.Rect(x0, cy + 12, r.w - 28, 4)
            draw.rounded_rect(self.ui, (255, 255, 255, 22), bar, 2)
            half = bar.w / 2
            length = max(-1.0, min(1.0, part)) * half
            seg = pygame.Rect(0, bar.y, max(2, round(abs(length))), 4)
            if length >= 0:
                seg.x = round(bar.x + half)
            else:
                seg.right = round(bar.x + half)
            col = ACTION_COLORS["dot"] if length >= 0 else DANGER
            draw.rounded_rect(self.ui, draw.with_alpha(col, 0.85), seg, 2)

        y = r.y + 50 + len(FEATURES) * 40
        raw = sig.raw if (sig is not None and state.face) else 0.0
        self.blit(self.text("regular", 12, f"sum {raw:+.2f}   smoothed {state.level:.2f}",
                            TEXT_MUTED), (x0, y), "midleft")
        quality = "-" if cal is None else f"{cal.quality:.1f}"
        self.blit(self.text("regular", 12, f"d\u2032 {quality}  \u00b7  F to close",
                            TEXT_FAINT), (x0, y + 22), "midleft")

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

        flaring = state.pressed and not state.paused
        if state.code or flaring:
            end = self._draw_code(state.code, x0 + shake, cy, now)
            if flaring:
                self._draw_live_smile(state, end, cy)
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
        elif state.calibrating:
            self.blit(self.text("regular", 14,
                                "Calibrating: follow the prompt at the top of the screen",
                                TEXT_FAINT), (x0, cy), "midleft")
        else:
            self.blit(self.text("regular", 14,
                                "Short smile for a dot, long smile for a dash",
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

            # A quick ripple on the newest symbol confirms the smile landed.
            if is_last and t < 0.35:
                k = t / 0.35
                draw.ring(self.ui, draw.with_alpha(color, 1.0 - k), (cx, cy),
                          10 + 18 * draw.ease_out_cubic(k), 2)
                draw.glow(self.glow, color, (cx, cy), 30, 0.8 * (1.0 - k))
            x += width
        return x

    def _draw_live_smile(self, state: HudState, x: float, cy: float) -> None:
        """
        While smiling, show the symbol the smile will produce: a dot
        outline that stretches into a dash as the smile gets longer.
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
            self.blit(self.text("regular", 12, "relax your face to confirm", TEXT_MUTED),
                      (left, cy), "midright")
        elif options:
            preview = "  ".join(options[:6]) + ("  \u2026" if len(options) > 6 else "")
            self.blit(self.text("medium", 14, preview, TEXT_MUTED), (left, cy), "midright")
        else:
            self.blit(self.text("medium", 13, "no match", DANGER), (left, cy), "midright")

    def _draw_key_hints(self, right: float, cy: float, state: HudState) -> None:
        hints = [("C", "calibrate"), ("X", "settings"), ("Z", "hide UI"),
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
