"""
Settings window with two tabs: Camera and Tongue.

It runs in a separate process with its own pygame window. The main window
uses an OpenGL context, and a second window in the same process would
fight with it over that context. A separate process sidesteps the problem
and also keeps the main render loop from stalling while the user drags a
slider.

The two processes talk over a multiprocessing Pipe:

    main -> settings   ("stats", {...})        live FPS and resolution
                       ("props", {...})        driver values for the sliders
    settings -> main   ("set", section, key, value)
                       ("action", name)
                       ("closed",)
"""

import math
import multiprocessing as mp
import platform

from .config import (ACCENT, ACTION_COLORS, DANGER, FPS_CHOICES, MIN_ACCEPTABLE_FPS,
                     RESOLUTION_CHOICES, SETTINGS_SIZE, SETTINGS_TITLE, SUCCESS,
                     TEXT, TEXT_FAINT, TEXT_MUTED, WARNING)

ROW_H = 32
PAD = 24
TABS = [("Camera", "camera"), ("Tongue", "eyes")]


def list_camera_names() -> list:
    """
    Names of the video devices, in the order OpenCV's DirectShow backend
    numbers them. Windows only, and only if the optional `pygrabber`
    package is installed; otherwise an empty list and the UI falls back to
    plain numbers.
    """
    if platform.system() != "Windows":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
        return list(FilterGraph().get_input_devices())
    except Exception:          # missing package, COM error, no devices...
        return []


# ===========================================================================
# Main-process side
# ===========================================================================

class SettingsLink:
    """Owns the settings process and the pipe to it."""

    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None

    @property
    def is_open(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def toggle(self, initial: dict) -> None:
        if self.is_open:
            self.close()
            return
        parent, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=run_settings_window,
                                       args=(child, initial), daemon=True)
        self._proc.start()
        self._conn = parent

    def poll(self) -> list:
        """Messages sent by the settings window since the last call."""
        messages = []
        if self._conn is None:
            return messages
        try:
            while self._conn.poll():
                messages.append(self._conn.recv())
        except (EOFError, OSError):
            messages.append(("closed",))
        if any(m[0] == "closed" for m in messages):
            self._cleanup()
        return messages

    def send(self, *message) -> None:
        if self._conn is None:
            return
        try:
            self._conn.send(message)
        except (BrokenPipeError, OSError):
            self._cleanup()

    def close(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)
        self._cleanup()

    def _cleanup(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        self._proc = None


# ===========================================================================
# Settings-process side
# ===========================================================================

def run_settings_window(conn, initial: dict) -> None:
    """Entry point of the child process."""
    # Imported here so the main process does not pay for it twice.
    import pygame

    pygame.init()
    screen = pygame.display.set_mode(SETTINGS_SIZE)
    pygame.display.set_caption(SETTINGS_TITLE)
    window = SettingsWindow(screen, conn, initial)
    try:
        window.run()
    finally:
        try:
            conn.send(("closed",))
        except (BrokenPipeError, OSError):
            pass
        pygame.quit()


class SettingsWindow:
    def __init__(self, screen, conn, initial: dict):
        import pygame
        from .hud import Fonts

        self.pg = pygame
        self.screen = screen
        self.conn = conn
        self.fonts = Fonts()
        self.values = {"camera": dict(initial["camera"]),
                       "eyes": dict(initial["eyes"])}
        self.stats = {}
        self.camera_names = list_camera_names()
        self.tab = "camera"
        self.background = self._make_background()
        self._build_widgets()
        self.hover = None

    @property
    def widgets(self) -> list:
        """Widgets that are visible and clickable right now."""
        return [self.tab_bar] + self.pages[self.tab] + self.footer

    # -- layout -----------------------------------------------------------------

    def _build_widgets(self) -> None:
        cam = self.values["camera"]
        eyes = self.values["eyes"]
        w = SETTINGS_SIZE[0] - 2 * PAD
        self.tab_bar = TabBar(TABS, self.tab)
        self.tab_bar.rect = self.pg.Rect(PAD, 150, w, 34)

        def page(widgets):
            y = 198
            for widget in widgets:
                height = getattr(widget, "HEIGHT", ROW_H)
                widget.rect = self.pg.Rect(PAD, y, w, height)
                y += height + 2
            return widgets

        max_index = max(9, len(self.camera_names) - 1)
        self.pages = {
            "camera": page([
                Stepper("Device", "camera", "index", cam["index"], 0, max_index,
                        self.camera_names),
                Segmented("Resolution", "camera", "resolution",
                          [(f"{a}\u00d7{b}", f"{a}x{b}") for a, b in RESOLUTION_CHOICES],
                          f"{cam['width']}x{cam['height']}"),
                Segmented("Target FPS", "camera", "fps",
                          [(str(f), f) for f in FPS_CHOICES], cam["fps"]),
                Toggle("Mirror view", "camera", "mirror", cam["mirror"]),
                Toggle("Auto exposure", "camera", "auto_exposure", cam["auto_exposure"]),
                Slider("Exposure", "camera", "exposure", cam["exposure"], -13, -1, 1, "{:.0f}"),
                Slider("Brightness", "camera", "brightness", cam["brightness"], 0, 255, 1, "{:.0f}"),
                Slider("Contrast", "camera", "contrast", cam["contrast"], 0, 255, 1, "{:.0f}"),
                Slider("Gain", "camera", "gain", cam["gain"], 0, 255, 1, "{:.0f}"),
            ]),
            "eyes": page([
                Slider("Tongue length", "eyes", "tongue_cm",
                       eyes["tongue_cm"], 0.5, 6.0, 0.1, "{:.1f} cm"),
                Slider("Time unit t", "eyes", "time_unit",
                       eyes["time_unit"], 0.05, 0.50, 0.01, "{:.2f} s"),
            ]),
        }

        footer_y = SETTINGS_SIZE[1] - PAD - 36
        half = (w - 10) // 2
        if platform.system() == "Windows":
            self.footer = [
                Button("Driver settings", "driver_dialog",
                       self.pg.Rect(PAD, footer_y, half, 36)),
                Button("Reset defaults", "reset",
                       self.pg.Rect(PAD + half + 10, footer_y, half, 36)),
            ]
        else:
            self.footer = [Button("Reset defaults", "reset",
                                  self.pg.Rect(PAD, footer_y, w, 36))]

    def _make_background(self):
        pg = self.pg
        w, h = SETTINGS_SIZE
        surf = pg.Surface((w, h))
        top, bottom = (12, 15, 22), (17, 21, 33)
        for y in range(h):
            k = y / h
            pg.draw.line(surf, [round(top[i] + (bottom[i] - top[i]) * k) for i in range(3)],
                         (0, y), (w, y))
        # A faint coloured light in the corner gives the dark UI some depth.
        from .draw import glow
        glow(surf, (40, 120, 150), (w - 40, 30), 220, 0.35)
        glow(surf, (70, 50, 140), (30, h - 60), 200, 0.25)
        return surf

    # -- loop -------------------------------------------------------------------

    def run(self) -> None:
        pg = self.pg
        clock = pg.time.Clock()
        dragging = None
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
                if event.type == pg.KEYDOWN and event.key in (pg.K_ESCAPE, pg.K_x):
                    return
                if event.type == pg.MOUSEMOTION:
                    self.hover = next((w for w in self.widgets
                                       if w.rect.collidepoint(event.pos)), None)
                    if dragging is not None:
                        self._emit(dragging.drag(event.pos[0]))
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    for w in self.widgets:
                        if w.rect.collidepoint(event.pos) and self._enabled(w):
                            self._emit(w.click(event.pos))
                            if isinstance(w, Slider):
                                dragging = w
                            break
                if event.type == pg.MOUSEBUTTONUP and event.button == 1:
                    dragging = None

            if not self._read_pipe():
                return
            self._draw()
            pg.display.flip()
            clock.tick(60)

    def _enabled(self, widget) -> bool:
        if getattr(widget, "key", None) == "exposure":
            return not self.values["camera"]["auto_exposure"]
        return True

    def _emit(self, change) -> None:
        if not change:
            return
        if change[0] == "tab":
            self.tab = change[1]
            self.hover = None
            return
        if change[0] == "action":
            if change[1] == "reset":
                self._reset()
            self.conn.send(change)
            return
        _, section, key, value = change
        self.values[section][key] = value
        self.conn.send(change)

    def _reset(self) -> None:
        from .config import CameraSettings, EyeSettings
        from dataclasses import asdict
        cam, eyes = asdict(CameraSettings()), asdict(EyeSettings())
        cam["index"] = self.values["camera"]["index"]
        self.values = {"camera": cam, "eyes": eyes}
        self._build_widgets()

    def _read_pipe(self) -> bool:
        try:
            while self.conn.poll():
                msg = self.conn.recv()
                if msg[0] == "stats":
                    self.stats = msg[1]
                elif msg[0] == "props":
                    for w in self.pages["camera"]:
                        if isinstance(w, Slider) and w.key in msg[1]:
                            w.adopt(msg[1][w.key])
        except (EOFError, OSError):
            return False
        return True

    # -- drawing ----------------------------------------------------------------

    def _text(self, family, size, text, color):
        return self.fonts.get(family, size).render(text, True, color)

    def _draw(self) -> None:
        pg = self.pg
        from . import draw
        s = self.screen
        s.blit(self.background, (0, 0))
        w = SETTINGS_SIZE[0]

        s.blit(self._text("semibold", 19, SETTINGS_TITLE, TEXT), (PAD, 22))
        # The subtitle doubles as the place for camera errors.
        error = self.stats.get("error")
        subtitle = (error, DANGER) if error else ("Changes apply instantly", TEXT_MUTED)
        s.blit(self._text("regular", 12, *subtitle), (PAD, 48))

        # Live stats card
        card = pg.Rect(PAD, 76, w - 2 * PAD, 62)
        layer = pg.Surface(card.size, pg.SRCALPHA)
        draw.rounded_rect(layer, (255, 255, 255, 12), layer.get_rect(), 12)
        draw.rounded_rect(layer, (255, 255, 255, 26), layer.get_rect(), 12, 1)
        s.blit(layer, card)

        size = self.stats.get("size", (0, 0))
        cam_fps = self.stats.get("camera_fps", 0.0)
        app_fps = self.stats.get("app_fps", 0.0)
        cols = [
            ("Resolution", f"{size[0]}×{size[1]}" if size[0] else "-", TEXT),
            ("Camera", f"{cam_fps:.0f} fps", _fps_color(cam_fps)),
            ("App", f"{app_fps:.0f} fps", _fps_color(app_fps)),
        ]
        col_w = card.w / 3
        for i, (label, value, color) in enumerate(cols):
            x = card.x + 16 + i * col_w
            s.blit(self._text("regular", 11, label, TEXT_FAINT), (x, card.y + 12))
            s.blit(self._text("semibold", 16, value, color), (x, card.y + 30))
        backend = self.stats.get("backend", "")
        if backend and backend != "-":
            img = self._text("regular", 11, backend, TEXT_FAINT)
            s.blit(img, img.get_rect(topright=(card.right - 14, card.y + 12)))

        for widget in self.widgets:
            enabled = self._enabled(widget)
            widget.draw(s, self, hover=(widget is self.hover), enabled=enabled)

        if self.tab == "eyes":
            self._draw_timing_card()

    def _draw_timing_card(self) -> None:
        """What the current time unit means, spelled out in seconds."""
        pg = self.pg
        from . import draw
        s = self.screen
        t = self.values["eyes"]["time_unit"]
        last = self.pages["eyes"][-1].rect
        card = pg.Rect(PAD, last.bottom + 18, SETTINGS_SIZE[0] - 2 * PAD, 138)
        layer = pg.Surface(card.size, pg.SRCALPHA)
        draw.rounded_rect(layer, (255, 255, 255, 12), layer.get_rect(), 12)
        draw.rounded_rect(layer, (255, 255, 255, 26), layer.get_rect(), 12, 1)
        s.blit(layer, card)
        s.blit(self._label("Timing"), (card.x + 16, card.y + 14))

        rows = [
            (ACTION_COLORS["dot"], "Dot", f"tongue out shorter than {1.5 * t:.2f} s"),
            (ACTION_COLORS["dash"], "Dash", f"tongue out {1.5 * t:.2f} s or longer"),
            (ACTION_COLORS["letter"], "End letter", f"tongue in {3 * t:.2f} s"),
            (ACTION_COLORS["word"], "Space", f"tongue in {7 * t:.2f} s"),
        ]
        for i, (color, name, rule) in enumerate(rows):
            cy = card.y + 46 + i * 24
            draw.circle(s, color, (card.x + 22, cy), 4)
            img = self._text("medium", 13, name, TEXT)
            s.blit(img, img.get_rect(midleft=(card.x + 34, cy)))
            img = self._text("regular", 12, rule, TEXT_MUTED)
            s.blit(img, img.get_rect(midright=(card.right - 16, cy)))

    def _label(self, text):
        pg = self.pg
        glyphs = [self._text("semibold", 10, c, TEXT_FAINT) for c in text.upper()]
        width = sum(g.get_width() + 1 for g in glyphs)
        surf = pg.Surface((width, glyphs[0].get_height()), pg.SRCALPHA)
        x = 0
        for g in glyphs:
            surf.blit(g, (x, 0))
            x += g.get_width() + 1
        return surf


def _fps_color(fps: float):
    if fps >= 30:
        return SUCCESS
    if fps >= MIN_ACCEPTABLE_FPS:
        return WARNING
    return DANGER if fps > 0 else TEXT_MUTED


# ===========================================================================
# Widgets
# ===========================================================================

class Widget:
    rect = None

    def __init__(self, label, section, key):
        self.label = label
        self.section = section
        self.key = key

    def click(self, pos):
        return None

    def drag(self, x):
        return None

    def draw_label(self, surface, win, enabled):
        color = TEXT if enabled else TEXT_FAINT
        img = win._text("regular", 13, self.label, color)
        surface.blit(img, img.get_rect(midleft=(self.rect.x, self.rect.centery)))


class Slider(Widget):
    TRACK_X = 130       # where the track starts, relative to the row
    VALUE_W = 54        # space reserved for the value on the right

    def __init__(self, label, section, key, value, lo, hi, step, fmt):
        super().__init__(label, section, key)
        self.lo, self.hi, self.step, self.fmt = lo, hi, step, fmt
        # None means the driver value is not known yet; the "props" message
        # from the main window fills it in a moment later.
        self.value = float(value) if value is not None else (lo + hi) / 2

    def _track(self):
        x0 = self.rect.x + self.TRACK_X
        x1 = self.rect.right - self.VALUE_W
        return x0, x1

    def adopt(self, driver_value: float) -> None:
        """Take the value the camera driver reports, widening the range if needed."""
        if driver_value < self.lo:
            self.lo = math.floor(driver_value)
        if driver_value > self.hi:
            self.hi = math.ceil(driver_value)
        self.value = driver_value

    def click(self, pos):
        return self.drag(pos[0])

    def drag(self, x):
        x0, x1 = self._track()
        k = min(1.0, max(0.0, (x - x0) / (x1 - x0)))
        raw = self.lo + k * (self.hi - self.lo)
        value = round(round(raw / self.step) * self.step, 4)
        if value == self.value:
            return None
        self.value = value
        return ("set", self.section, self.key, value)

    def draw(self, surface, win, hover, enabled):
        from . import draw
        self.draw_label(surface, win, enabled)
        x0, x1 = self._track()
        cy = self.rect.centery
        k = (self.value - self.lo) / max(self.hi - self.lo, 1e-6)
        kx = x0 + k * (x1 - x0)
        accent = ACCENT if enabled else TEXT_FAINT
        win.pg.draw.rect(surface, (48, 54, 70), (x0, cy - 2, x1 - x0, 4), border_radius=2)
        win.pg.draw.rect(surface, accent, (x0, cy - 2, max(kx - x0, 0), 4), border_radius=2)
        if enabled and hover:
            draw.glow(surface, ACCENT, (kx, cy), 22, 0.35)
        draw.circle(surface, (245, 247, 252) if enabled else TEXT_FAINT, (kx, cy), 7)
        draw.ring(surface, accent, (kx, cy), 7, 2)
        img = win._text("medium", 12, self.fmt.format(self.value),
                        TEXT_MUTED if enabled else TEXT_FAINT)
        surface.blit(img, img.get_rect(midright=(self.rect.right, cy)))


class Toggle(Widget):
    def __init__(self, label, section, key, value):
        super().__init__(label, section, key)
        self.value = bool(value)
        self._anim = 1.0 if self.value else 0.0

    def click(self, pos):
        self.value = not self.value
        return ("set", self.section, self.key, self.value)

    def draw(self, surface, win, hover, enabled):
        from . import draw
        self.draw_label(surface, win, enabled)
        target = 1.0 if self.value else 0.0
        self._anim += (target - self._anim) * 0.35
        pill = win.pg.Rect(0, 0, 38, 22)
        pill.midright = (self.rect.right, self.rect.centery)
        off, on = (58, 64, 82), (40, 150, 170)
        color = [round(off[i] + (on[i] - off[i]) * self._anim) for i in range(3)]
        win.pg.draw.rect(surface, color, pill, border_radius=11)
        kx = pill.x + 11 + self._anim * 16
        if self.value:
            draw.glow(surface, ACCENT, (kx, pill.centery), 18, 0.3)
        draw.circle(surface, (245, 247, 252), (kx, pill.centery), 8)


class Segmented(Widget):
    def __init__(self, label, section, key, options, value):
        super().__init__(label, section, key)
        self.options = options
        self.value = value

    def _segments(self):
        x0 = self.rect.x + 130
        width = self.rect.right - x0
        seg_w = width / len(self.options)
        for i, (text, value) in enumerate(self.options):
            yield self.rect.__class__(round(x0 + i * seg_w), self.rect.y + 3,
                                      round(seg_w), self.rect.h - 6), text, value

    def click(self, pos):
        for rect, _, value in self._segments():
            if rect.collidepoint(pos) and value != self.value:
                self.value = value
                return ("set", self.section, self.key, value)
        return None

    def draw(self, surface, win, hover, enabled):
        from . import draw
        self.draw_label(surface, win, enabled)
        segs = list(self._segments())
        outer = segs[0][0].union(segs[-1][0])
        win.pg.draw.rect(surface, (30, 35, 48), outer, border_radius=9)
        for rect, text, value in segs:
            selected = value == self.value
            if selected:
                inner = rect.inflate(-4, -4)
                win.pg.draw.rect(surface, (34, 72, 86), inner, border_radius=7)
                draw.rounded_rect(surface, (80, 190, 210), inner, 7, 1)
            img = win._text("medium" if selected else "regular", 12, text,
                            TEXT if selected else TEXT_MUTED)
            surface.blit(img, img.get_rect(center=rect.center))


class Stepper(Widget):
    HEIGHT = 40          # a little taller to fit the device name

    def __init__(self, label, section, key, value, lo, hi, names=()):
        super().__init__(label, section, key)
        self.value, self.lo, self.hi = int(value), lo, hi
        self.names = list(names)

    def _buttons(self):
        size = 26
        plus = self.rect.__class__(0, 0, size, size)
        plus.midright = (self.rect.right, self.rect.centery)
        minus = plus.copy()
        minus.right = plus.left - 86
        return minus, plus

    def click(self, pos):
        minus, plus = self._buttons()
        step = -1 if minus.collidepoint(pos) else 1 if plus.collidepoint(pos) else 0
        new = min(self.hi, max(self.lo, self.value + step))
        if step == 0 or new == self.value:
            return None
        self.value = new
        return ("set", self.section, self.key, new)

    def draw(self, surface, win, hover, enabled):
        self.draw_label(surface, win, enabled)
        minus, plus = self._buttons()
        for rect, sign in ((minus, "−"), (plus, "+")):
            win.pg.draw.rect(surface, (34, 40, 55), rect, border_radius=8)
            win.pg.draw.rect(surface, (70, 78, 98), rect, width=1, border_radius=8)
            img = win._text("medium", 15, sign, TEXT)
            surface.blit(img, img.get_rect(center=rect.center))
        img = win._text("semibold", 14, f"Camera {self.value}", TEXT)
        surface.blit(img, img.get_rect(center=((minus.right + plus.left) // 2,
                                               self.rect.centery)))

    def draw_label(self, surface, win, enabled):
        # With device names available, show the name under the label so the
        # user can tell the laptop camera from a phone or virtual camera.
        if not self.names:
            super().draw_label(surface, win, enabled)
            return
        name = (self.names[self.value] if self.value < len(self.names)
                else "Not connected")
        if len(name) > 22:
            name = name[:21] + "\u2026"
        img = win._text("regular", 13, self.label, TEXT)
        surface.blit(img, img.get_rect(bottomleft=(self.rect.x, self.rect.centery + 1)))
        img = win._text("regular", 11, name, ACCENT)
        surface.blit(img, img.get_rect(topleft=(self.rect.x, self.rect.centery + 2)))


class TabBar(Widget):
    """Two or more tabs drawn as a segmented control."""

    def __init__(self, tabs, value):
        super().__init__("", None, None)
        self.tabs = tabs
        self.value = value

    def _segments(self):
        seg_w = self.rect.w / len(self.tabs)
        for i, (text, value) in enumerate(self.tabs):
            yield (self.rect.__class__(round(self.rect.x + i * seg_w), self.rect.y,
                                       round(seg_w), self.rect.h), text, value)

    def click(self, pos):
        for rect, _, value in self._segments():
            if rect.collidepoint(pos) and value != self.value:
                self.value = value
                return ("tab", value)
        return None

    def draw(self, surface, win, hover, enabled):
        from . import draw
        win.pg.draw.rect(surface, (26, 31, 43), self.rect, border_radius=10)
        for rect, text, value in self._segments():
            selected = value == self.value
            if selected:
                inner = rect.inflate(-6, -6)
                win.pg.draw.rect(surface, (44, 52, 70), inner, border_radius=8)
                draw.rounded_rect(surface, (86, 98, 124), inner, 8, 1)
            img = win._text("semibold" if selected else "medium", 13, text,
                            TEXT if selected else TEXT_MUTED)
            surface.blit(img, img.get_rect(center=rect.center))


class Button(Widget):
    def __init__(self, label, action, rect):
        super().__init__(label, None, None)
        self.action = action
        self.rect = rect

    def click(self, pos):
        return ("action", self.action)

    def draw(self, surface, win, hover, enabled):
        fill = (38, 46, 62) if hover else (28, 34, 47)
        win.pg.draw.rect(surface, fill, self.rect, border_radius=10)
        win.pg.draw.rect(surface, (74, 84, 106), self.rect, width=1, border_radius=10)
        img = win._text("medium", 13, self.label, TEXT)
        surface.blit(img, img.get_rect(center=self.rect.center))
