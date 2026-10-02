"""
Application loop: camera -> face tracking -> blinks -> Morse -> screen.

Per frame:
  1. Take the newest camera frame (the camera runs in its own thread).
  2. Mirror it, run MediaPipe for the face landmarks and head pose, and
     measure the dark area of the nostrils (nostril.py).
  3. Turn the flare into a score (the threshold from the settings is the
     "pressed" point) and feed it to the press detector, which times each
     press: short flare = dot, long flare = dash.
  4. Let the Morse composer turn pauses into letters and spaces.

Pressing Z hides the whole interface, shows the untouched camera image and
switches input off until Z is pressed again.
  5. Draw the HUD and let the GPU compose the final image.

Latency matters more than anything else here, so the symbol is typed and
its sound played in the same frame the blink is recognised, before any
drawing happens.
"""

import csv
import time

import cv2
import moderngl
import numpy as np
import pygame

from .audio import SoundBank
from .blinks import BlinkDetector, BlinkKind
from .camera import CameraStream
from .compositor import Compositor, cover_crop
from .config import APP_TITLE, ROOT_DIR, WINDOW_SIZE, AppSettings, CameraSettings, EyeSettings
from .face_tracker import FaceTracker
from .hud import Hud, HudState
from .morse import EventKind, MorseComposer
from .settings_window import SettingsLink

# Score (0..1) at which the nostrils count as "pressed": the flare threshold
# from the settings is mapped onto this value.
PRESS_SCORE = 0.5
LOG_DIR = ROOT_DIR / "logs"


class BlinkMorseApp:
    def __init__(self):
        self.settings = AppSettings.load()

        # The model is loaded before the window opens so a first-run
        # download does not leave a frozen black window on screen.
        self.tracker = FaceTracker()

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                        pygame.GL_CONTEXT_PROFILE_CORE)
        # Required on macOS for a core profile, harmless elsewhere.
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FLAGS,
                                        pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
        # VSync off: the camera already paces the loop, and vsync would add
        # up to one extra frame of latency between a blink and its feedback.
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)
        pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(APP_TITLE)

        self.ctx = moderngl.create_context()
        self.compositor = Compositor(self.ctx, WINDOW_SIZE)
        self.hud = Hud(WINDOW_SIZE)
        self.sounds = SoundBank()

        self.camera = CameraStream(self.settings.camera)
        self.camera.start()

        e = self.settings.eyes
        # The nostril flare is turned into a 0..1 score where the threshold
        # from the settings sits exactly at 0.5 (see _flare_score).
        self.detector = BlinkDetector(PRESS_SCORE, e.time_unit)
        for signal in (self.detector.left, self.detector.right):
            # The nostril detector already removes the resting level.
            signal.baseline = 0.0
            signal.MAX_BASELINE = 0.0
        self.composer = MorseComposer(e.letter_gap, e.word_gap)
        self.settings_link = SettingsLink()

        self.paused = False               # P stops typing
        self.ui_visible = True            # Z hides the UI and stops input
        self._filter = 1.0                # current strength of the styled look
        self.show_chart = True
        self.fps = 0.0
        self._frame_id = 0
        self._start = time.perf_counter()
        self._last_stats = 0.0
        self._props_sent = False
        self._state = HudState()
        self._nostril = None
        self._log_file = None
        self._log_writer = None

    # ------------------------------------------------------------------------

    def run(self) -> None:
        clock = pygame.time.Clock()
        last = time.perf_counter()
        running = True
        try:
            while running:
                running = self._handle_events()
                self._handle_settings_messages()

                frame_id, frame = self.camera.wait_frame(self._frame_id, timeout=0.05)
                now = time.perf_counter()
                if frame is not None:
                    self._frame_id = frame_id
                    self._process_frame(frame, now)

                self._render(now)

                dt = now - last
                last = now
                if dt > 0:
                    # Exponential moving average keeps the number readable.
                    self.fps = self.fps * 0.9 + (1.0 / dt) * 0.1
                self._push_stats(now)
                clock.tick(120)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
        self.settings_link.close()
        self.camera.stop()
        self.tracker.close()
        try:
            self.settings.save()
        except OSError:
            pass
        pygame.quit()

    # ------------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------------

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if event.key == pygame.K_x:
                self.settings_link.toggle(self.settings.to_dict())
                self._last_stats = 0.0
                self._props_sent = False
            elif event.key == pygame.K_z:
                self._toggle_ui()
            elif event.key == pygame.K_p:
                self.paused = not self.paused
                self.sounds.play("pause" if self.paused else "resume")
            elif event.key == pygame.K_h:
                self.show_chart = not self.show_chart
            elif event.key == pygame.K_m:
                self.sounds.toggle()
            elif event.key == pygame.K_BACKSPACE:
                self._apply_composer(self.composer.delete(), time.perf_counter())
            elif event.key == pygame.K_DELETE:
                self._apply_composer(self.composer.clear(), time.perf_counter())
            elif event.key == pygame.K_c:
                self.tracker.recalibrate()
                self.detector.reset(time.perf_counter())
            elif event.key == pygame.K_l:
                self._toggle_logging()
        return True

    def _toggle_ui(self) -> None:
        self.ui_visible = not self.ui_visible
        state = self._state
        state.face = False
        state.pose = "open"
        self._nostril = None
        # Start timing afresh, so time spent with the UI hidden is not
        # counted as a pause that ends the letter.
        self.detector.reset(time.perf_counter())

    def _process_frame(self, frame_bgr: np.ndarray, now: float) -> None:
        cam = self.settings.camera
        if cam.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if not self.ui_visible:
            # UI hidden: plain camera only, no face tracking, no typing.
            self.compositor.update_camera(frame_rgb)
            return

        # Detection first, so a blink is typed before any time is spent on
        # uploading textures or drawing.
        timestamp_ms = int((now - self._start) * 1000)
        face = self.tracker.detect(frame_rgb, timestamp_ms, cam.mirror,
                                   self.settings.eyes.swap_eyes)

        state = self._state
        if face is None:
            self.detector.reset()
            state.face = False
            state.pose = "open"
            self._nostril = None
        else:
            score = self._flare_score(face.nostril)
            self._log(face.nostril, now)
            for event in self.detector.update(score, score, now):
                if not self.paused:
                    self._handle_blink(event, now)
            state.face = True
            state.openness = {"left": 1.0 - self.detector.left.value,
                              "right": 1.0 - self.detector.right.value}
            state.pose = self.detector.pose
            self._update_nostril_overlay(face, frame_rgb.shape)

        # Pauses end letters and words. Without a face the eyes cannot be
        # blinking, so the time since the last blink keeps counting.
        for event in self.composer.update(self.detector.pause_time(now)):
            self._apply_composer(event, now)
        self.compositor.update_camera(frame_rgb)

    def _handle_blink(self, event, now: float) -> None:
        if event.kind == BlinkKind.DOT:
            self._apply_composer(self.composer.add_symbol("."), now)
        elif event.kind == BlinkKind.DASH:
            self._apply_composer(self.composer.add_symbol("-"), now)

    def _flare_score(self, nostril) -> float:
        """
        Nostril flare as a 0..1 score for the press detector: the threshold
        from the settings maps to 0.5, twice the threshold to 1.
        """
        if nostril is None or not nostril.ok or nostril.calibrating or nostril.tilted:
            return 0.0
        threshold = max(self.settings.eyes.flare_pct / 100.0, 0.01)
        return min(1.0, max(0.0, PRESS_SCORE * nostril.flare / threshold))

    def _update_nostril_overlay(self, face, shape) -> None:
        """Nostril outlines and the measured patch in window pixels."""
        h, w = shape[:2]
        (sx, sy), (ox, oy) = cover_crop((w, h), WINDOW_SIZE)
        W, H = WINDOW_SIZE

        def to_screen(p):
            return (float((p[0] - ox) / sx * W), float((p[1] - oy) / sy * H))

        n = face.nostril
        if n is None or not n.ok:
            self._nostril = None
            return
        self._nostril = {
            "contours": [[to_screen(p) for p in c] for c in n.contours],
            "box": [to_screen(p) for p in n.box],
            "flare": n.flare,
            "calibrating": n.calibrating,
            "tilted": n.tilted,
        }

    # ------------------------------------------------------------------------
    # Data logging (L key): one CSV row per frame, for analysis later
    # ------------------------------------------------------------------------

    def _toggle_logging(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = self._log_writer = None
            print("Logging stopped")
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / time.strftime("nostril_%Y%m%d_%H%M%S.csv")
        self._log_file = open(path, "w", newline="")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(["time", "ratio_left", "ratio_right", "rest_left",
                                   "rest_right", "flare", "pitch", "calibrating",
                                   "tilted", "pressed"])
        print(f"Logging to {path}")

    def _log(self, n, now: float) -> None:
        if self._log_writer is None or n is None:
            return
        self._log_writer.writerow([
            f"{now - self._start:.3f}", f"{n.ratio[0]:.4f}", f"{n.ratio[1]:.4f}",
            f"{n.rest[0]:.4f}", f"{n.rest[1]:.4f}", f"{n.flare:.4f}", f"{n.pitch:.2f}",
            int(n.calibrating), int(n.tilted), int(self.detector.pose == "both")])

    def _apply_composer(self, event, now: float) -> None:
        if event is None:
            return
        kind = event.kind
        if kind == EventKind.SYMBOL:
            self.sounds.play("dot" if event.value == "." else "dash")
            self.hud.on_symbol(event.value, now)
        elif kind == EventKind.LETTER:
            self.sounds.play("letter")
            self.hud.on_letter(event.value, now)
        elif kind == EventKind.SPACE:
            self.sounds.play("space")
            self.hud.on_space(now)
        elif kind == EventKind.INVALID:
            self.sounds.play("invalid")
            self.hud.on_invalid(event.value, now)
        elif kind == EventKind.DELETE:
            self.sounds.play("delete")
        elif kind == EventKind.CLEAR:
            self.sounds.play("clear")
            self.hud.on_clear(now)

    # ------------------------------------------------------------------------
    # Settings window
    # ------------------------------------------------------------------------

    def _handle_settings_messages(self) -> None:
        for msg in self.settings_link.poll():
            if msg[0] == "set":
                _, section, key, value = msg
                if section == "camera":
                    self._apply_camera_setting(key, value)
                else:
                    self._apply_eye_setting(key, value)
            elif msg[0] == "action":
                if msg[1] == "driver_dialog":
                    self.camera.open_driver_dialog()
                elif msg[1] == "reset":
                    self._reset_settings()

    def _apply_camera_setting(self, key: str, value) -> None:
        cam = self.settings.camera
        if key == "resolution":
            cam.width, cam.height = (int(v) for v in value.split("x"))
            self.camera.reopen()
        elif key in ("index", "fps"):
            setattr(cam, key, int(value))
            self.camera.reopen()
            if key == "index":
                self._props_sent = False    # a new device has its own values
        elif key == "mirror":
            cam.mirror = bool(value)
        else:
            setattr(cam, key, value)
            self.camera.set_property(key, value)

    def _apply_eye_setting(self, key: str, value) -> None:
        e = self.settings.eyes
        setattr(e, key, value)
        self.detector.close_threshold = PRESS_SCORE
        self.detector.time_unit = e.time_unit
        self.composer.letter_gap = e.letter_gap
        self.composer.word_gap = e.word_gap

    def _reset_settings(self) -> None:
        index = self.settings.camera.index
        fresh_cam = CameraSettings(index=index)
        for field_name, value in vars(fresh_cam).items():
            setattr(self.settings.camera, field_name, value)
        for field_name, value in vars(EyeSettings()).items():
            self._apply_eye_setting(field_name, value)
        self.camera.restore_driver_defaults()
        self.camera.reopen()
        # Send the driver values again so the rebuilt sliders match them.
        self._props_sent = False

    def _push_stats(self, now: float) -> None:
        if not self.settings_link.is_open or now - self._last_stats < 0.25:
            return
        self._last_stats = now
        self.settings_link.send("stats", {
            "size": self.camera.actual_size,
            "camera_fps": self.camera.actual_fps,
            "app_fps": self.fps,
            "backend": self.camera.backend_name,
            "error": self.camera.error,
        })
        if not self._props_sent and self.camera.properties:
            # Seed the sliders once with what the driver actually reports.
            self.settings_link.send("props", self.camera.properties)
            self._props_sent = True

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    def _render(self, now: float) -> None:
        state = self._state
        e = self.settings.eyes
        state.fps = self.fps
        state.camera_error = self.camera.error
        state.open_threshold = 1.0 - PRESS_SCORE
        state.flare_threshold = e.flare_pct / 100.0
        state.dash_after = e.dash_after
        state.letter_gap = e.letter_gap
        state.word_gap = e.word_gap
        state.closed_time = self.detector.closed_time(now)
        state.code = self.composer.code
        state.text = self.composer.text
        state.paused = self.paused
        state.chips = {}
        state.nostril = self._nostril

        pause = self.detector.pause_time(now)
        kind, progress = self.composer.pause_state(pause)
        state.pause_kind = kind
        state.pause_progress = progress
        target = e.letter_gap if kind == "letter" else e.word_gap
        state.pause_left = max(0.0, target - pause)
        state.muted = not self.sounds.enabled
        state.show_chart = self.show_chart

        # Fade the colour filter in or out over about a fifth of a second.
        goal = 1.0 if self.ui_visible else 0.0
        self._filter += (goal - self._filter) * 0.25
        if abs(goal - self._filter) < 0.01:
            self._filter = goal

        self.hud.draw(state, now, self.ui_visible)
        self.compositor.set_panels(self.hud.panels)
        self.compositor.update_layers(self.hud.ui, self.hud.glow)
        self.compositor.render(now - self._start, self._filter)
        pygame.display.flip()
