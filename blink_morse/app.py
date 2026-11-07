"""
Application loop: camera -> face tracking -> ears -> Morse -> screen.

Per frame:
  1. Take the newest camera frame (the camera runs in its own thread).
  2. Mirror it and run MediaPipe to get landmarks, blendshapes and head pose.
  3. Measure both ears in the image relative to the head (ear.py) and
     fuse the measurements into one ear score (fusion.py). A short guided
     calibration at start-up (and on C) learns how much each measurement
     counts and how much of it just follows the head angle.
  4. Feed the score to the press detector. Hold mode times each pull
     (short = dot, long = dash); twitch mode counts wiggles (one = dot,
     two in a row = dash).
  5. Let the Morse composer turn pauses into letters and spaces.
  6. Draw the HUD and let the GPU compose the final image.

Pressing Z hides the whole interface, shows the untouched camera image and
switches input off until Z is pressed again.

Latency matters more than anything else here, so the symbol is typed and
its sound played in the same frame the ear movement is recognised, before any
drawing happens.
"""

import time

import cv2
import moderngl
import numpy as np
import pygame

from .audio import SoundBank
from .camera import CameraStream
from .compositor import Compositor, cover_crop
from .config import (APP_TITLE, LOGS_DIR, WINDOW_SIZE, AppSettings, CameraSettings,
                     EarSettings)
from .face_tracker import FaceTracker
from .hud import CHIP_GAP, CHIP_SIZE, Hud, HudState
from .morse import EventKind, MorseComposer
from .fusion import FeatureLog, quality_label
from .ear import FEATURES, EarSignal, EarTracker
from .press import PressDetector, PressKind, TwitchDetector
from .settings_window import SettingsLink


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
        # up to one extra frame of latency between a wiggle and its feedback.
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)
        pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(APP_TITLE)

        self.ctx = moderngl.create_context()
        self.compositor = Compositor(self.ctx, WINDOW_SIZE)
        self.hud = Hud(WINDOW_SIZE)
        self.sounds = SoundBank()

        self.camera = CameraStream(self.settings.camera)
        self.camera.start()

        n = self.settings.ear
        self.ears = EarTracker()
        self.signal = EarSignal(n.threshold)
        self.detector = self._make_detector()
        self.composer = MorseComposer(n.letter_gap, n.word_gap)
        self.settings_link = SettingsLink()

        self.paused = False               # P stops typing, e.g. to rest
        self.ui_visible = True            # Z hides the UI and stops input
        self.show_chart = True
        self.show_debug = False           # F shows the feature weights
        self.show_legend = False          # G swaps the ear view for the legend
        self.log: FeatureLog = None       # L writes features to a CSV file
        self._filter = 1.0                # current strength of the styled look
        self.fps = 0.0
        self._frame_id = 0
        self._start = time.perf_counter()
        self._last_stats = 0.0
        self._props_sent = False
        self._state = HudState()
        self._chip_pos = None

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
        self._stop_log()
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
            elif event.key == pygame.K_c:
                self._recalibrate()
            elif event.key == pygame.K_t:
                self._apply_ear_setting("twitch_mode", not self.settings.ear.twitch_mode)
            elif event.key == pygame.K_f:
                self.show_debug = not self.show_debug
            elif event.key == pygame.K_g:
                self.show_legend = not self.show_legend
            elif event.key == pygame.K_l:
                self._toggle_log()
            elif event.key == pygame.K_h:
                self.show_chart = not self.show_chart
            elif event.key == pygame.K_m:
                self.sounds.toggle()
            elif event.key == pygame.K_BACKSPACE:
                self._apply_composer(self.composer.delete(), time.perf_counter())
            elif event.key == pygame.K_DELETE:
                self._apply_composer(self.composer.clear(), time.perf_counter())
        return True

    def _recalibrate(self) -> None:
        self.signal.start_calibration()
        self.detector.reset(time.perf_counter())

    def _toggle_log(self) -> None:
        if self.log is not None:
            self._stop_log()
            self.sounds.play("pause")
            return
        try:
            self.log = FeatureLog(LOGS_DIR, "ear", FEATURES)
            print(f"Logging ear features to {self.log.path}")
            self.sounds.play("resume")
        except OSError as exc:
            print(f"Cannot write the log: {exc}")
            self.log = None

    def _stop_log(self) -> None:
        if self.log is not None:
            self.log.close()
            print(f"Saved {self.log.path}")
            self.log = None

    def _toggle_ui(self) -> None:
        self.ui_visible = not self.ui_visible
        state = self._state
        state.face = False
        state.pressed = False
        self._chip_pos = None
        self.signal.lost()
        self.ears.lost()
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

        # Detection first, so a symbol is typed before any time is spent on
        # uploading textures or drawing.
        timestamp_ms = int((now - self._start) * 1000)
        face = self.tracker.detect(frame_rgb, timestamp_ms, cam.mirror)

        state = self._state
        if face is None or face.points is None:
            self.signal.lost()
            self.ears.lost()
            self.detector.reset()
            state.face = False
            state.pressed = False
            self._chip_pos = None
        else:
            ears = self.ears.update(frame_rgb, face.points, face.yaw, face.pitch, now)
            features = ears.features
            result = self.signal.update(features, now, face.pitch, face.yaw)
            self._handle_signal_events(result.events, now)

            if result.calibrating:
                # No typing while the user follows the prompts, and the
                # pause clock restarts so no letter ends meanwhile.
                self.detector.reset(now)
            else:
                for event in self.detector.update(result.score, now):
                    if not self.paused:
                        self._handle_press(event, now)
            if result.recentred:
                self.hud.on_recentred(now)

            if self.log is not None:
                label = pygame.key.get_pressed()[pygame.K_n]
                self.log.write(now, label, result, self.detector.pressed, features,
                               face.pitch, face.yaw)

            state.face = True
            state.signal = result
            state.pressed = self.detector.pressed
            state.ear_views = ears.views
            state.yaw, state.pitch = face.yaw, face.pitch
            self._update_chip_position(ears.anchors, frame_rgb.shape)

        # Pauses end letters and words. Without a face the ears cannot be
        # read, so the time since the last symbol keeps counting.
        for event in self.composer.update(self.detector.pause_time(now)):
            self._apply_composer(event, now)
        self.compositor.update_camera(frame_rgb)

    def _handle_signal_events(self, events: list, now: float) -> None:
        for name in events:
            if name == "ear":
                self.sounds.play("resume")
            elif name in ("relax", "turn"):
                self.sounds.play("pause")
            elif name == "done":
                self.sounds.play("letter")
                cal = self.signal.calibration
                self.hud.on_calibrated(quality_label(cal), now)
                if cal is not None:
                    shares = ", ".join(f"{n} {w * d * 100:.0f}%" for n, w, d
                                       in zip(FEATURES, cal.weights, cal.effect))
                    print(f"Calibration: {quality_label(cal)}, d' = {cal.quality:.2f}, "
                          f"share of the ear score: {shares}")

    def _handle_press(self, event, now: float) -> None:
        symbol = "." if event.kind == PressKind.DOT else "-"
        self._apply_composer(self.composer.add_symbol(symbol), now)

    def _make_detector(self):
        n = self.settings.ear
        cls = TwitchDetector if n.twitch_mode else PressDetector
        detector = cls(n.threshold, n.time_unit)
        detector.reset(time.perf_counter())
        return detector

    def _update_chip_position(self, anchors: dict, shape) -> None:
        """
        Place the score readout just outside a visible ear (the one on the
        screen's right if both are visible), in window pixels. Lightly
        smoothed so it does not jitter.
        """
        side = "right" if "right" in anchors else "left" if "left" in anchors else None
        if side is None:
            self._chip_pos = None
            return
        h, w = shape[:2]
        (sx, sy), (ox, oy) = cover_crop((w, h), WINDOW_SIZE)
        W, H = WINDOW_SIZE

        def to_screen(p):
            return np.array([(p[0] / w - ox) / sx * W, (p[1] / h - oy) / sy * H])

        ear = to_screen(anchors[side])
        direction = 1.0 if side == "right" else -1.0
        target = ear + np.array([direction * (CHIP_GAP + CHIP_SIZE[0] / 2), 0.0])
        if self._chip_pos is not None:
            target = self._chip_pos * 0.5 + target * 0.5
        self._chip_pos = target

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
                    self._apply_ear_setting(key, value)
            elif msg[0] == "action":
                if msg[1] == "driver_dialog":
                    self.camera.open_driver_dialog()
                elif msg[1] == "reset":
                    self._reset_settings()
                elif msg[1] == "calibrate":
                    self._recalibrate()

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

    def _apply_ear_setting(self, key: str, value) -> None:
        n = self.settings.ear
        setattr(n, key, value)
        if key == "twitch_mode":
            self.detector = self._make_detector()
        self.detector.threshold = n.threshold
        self.detector.time_unit = n.time_unit
        self.signal.threshold = n.threshold
        self.composer.letter_gap = n.letter_gap
        self.composer.word_gap = n.word_gap

    def _reset_settings(self) -> None:
        index = self.settings.camera.index
        fresh_cam = CameraSettings(index=index)
        for field_name, value in vars(fresh_cam).items():
            setattr(self.settings.camera, field_name, value)
        for field_name, value in vars(EarSettings()).items():
            self._apply_ear_setting(field_name, value)
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
        n = self.settings.ear
        state.fps = self.fps
        state.camera_error = self.camera.error
        state.threshold = n.threshold
        state.dash_after = n.dash_after
        state.letter_gap = n.letter_gap
        state.word_gap = n.word_gap
        # In twitch mode a long pull is still one wiggle: no dash preview.
        state.pressed_time = 0.0 if n.twitch_mode else self.detector.pressed_time(now)
        state.twitch_mode = n.twitch_mode
        state.waiting = getattr(self.detector, "waiting", False)
        state.code = self.composer.code
        state.text = self.composer.text
        state.paused = self.paused
        state.calibration = self.signal.calibration
        state.chip = None if self._chip_pos is None else (float(self._chip_pos[0]),
                                                         float(self._chip_pos[1]))

        pause = self.detector.pause_time(now)
        kind, progress = self.composer.pause_state(pause)
        state.pause_kind = kind
        state.pause_progress = progress
        target = n.letter_gap if kind == "letter" else n.word_gap
        state.pause_left = max(0.0, target - pause)
        state.muted = not self.sounds.enabled
        state.show_chart = self.show_chart
        state.show_debug = self.show_debug
        state.show_legend = self.show_legend
        state.yaw_ref, state.pitch_ref = self.signal.yaw_ref, self.signal.pitch_ref
        state.max_yaw, state.max_pitch = self.signal.MAX_YAW, self.signal.MAX_PITCH
        state.logging = self.log is not None

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
