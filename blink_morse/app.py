"""
Application loop: camera -> pose tracking -> curls -> Morse -> screen.

Per frame:
  1. Take the newest camera frame (the camera runs in its own thread).
  2. Mirror it and run MediaPipe Pose (elbow angle of the typing arm) and
     MediaPipe Hands (fist or open hand on the other side).
  3. Only while the switch hand is a fist, feed the curl score to the curl
     detector, which times each curl (like lifting a dumbbell):
     short = dot, long = dash. Open hand = nothing is typed.
     S swaps the roles of the two hands.
  4. Let the Morse composer turn pauses (arm extended) into letters and
     spaces.
  5. Draw the HUD, including the arm skeleton, and let the GPU compose
     the final image.

Pressing Z hides the whole interface, shows the untouched camera image and
switches input off until Z is pressed again.

Latency matters more than anything else here, so the symbol is typed and
its sound played in the same frame the curl is recognised, before any
drawing happens.
"""

import time

import cv2
import moderngl
import numpy as np
import pygame

from .audio import SoundBank
from .arm_tracker import ArmTracker
from .hand_tracker import FistGate, HandTracker, pick_switch_hand, to_pixels
from .camera import CameraStream
from .compositor import Compositor, cover_crop
from .config import APP_TITLE, WINDOW_SIZE, AppSettings, ArmSettings, CameraSettings
from .curls import CurlDetector, CurlKind
from .hud import Hud, HudState
from .morse import EventKind, MorseComposer
from .settings_window import SettingsLink


class BlinkMorseApp:
    def __init__(self):
        self.settings = AppSettings.load()

        # The model is loaded before the window opens so a first-run
        # download does not leave a frozen black window on screen.
        self.tracker = ArmTracker()
        self.hands = HandTracker()
        self.gate = FistGate()

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                        pygame.GL_CONTEXT_PROFILE_CORE)
        # Required on macOS for a core profile, harmless elsewhere.
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FLAGS,
                                        pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
        # VSync off: the camera already paces the loop, and vsync would add
        # up to one extra frame of latency between a curl and its feedback.
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)
        pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(APP_TITLE)

        self.ctx = moderngl.create_context()
        self.compositor = Compositor(self.ctx, WINDOW_SIZE)
        self.hud = Hud(WINDOW_SIZE)
        self.sounds = SoundBank()

        self.camera = CameraStream(self.settings.camera)
        self.camera.start()

        e = self.settings.arm
        self.detector = CurlDetector(e.curl_threshold, e.time_unit)
        self.composer = MorseComposer(e.letter_gap, e.word_gap)
        self.settings_link = SettingsLink()

        self.paused = False               # P stops typing, e.g. to rest the arm
        self.ui_visible = True            # Z hides the UI and stops input
        self._filter = 1.0                # current strength of the styled look
        self.show_chart = True
        self.fps = 0.0
        self._frame_id = 0
        self._start = time.perf_counter()
        self._last_stats = 0.0
        self._props_sent = False
        self._state = HudState()
        self._arm_pts = {"left": None, "right": None}
        self._hand_pts = None

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
        self.settings_link.close()
        self.camera.stop()
        self.tracker.close()
        self.hands.close()
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
            elif event.key == pygame.K_s:
                e = self.settings.arm
                self._apply_arm_setting("curl_side", e.gate_side)
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
        return True

    def _toggle_ui(self) -> None:
        self.ui_visible = not self.ui_visible
        state = self._state
        state.body = False
        self._arm_pts = {"left": None, "right": None}
        self._hand_pts = None
        self.gate.reset()
        # Start timing afresh, so time spent with the UI hidden is not
        # counted as a pause that ends the letter.
        self.detector.reset(time.perf_counter())

    def _process_frame(self, frame_bgr: np.ndarray, now: float) -> None:
        cam = self.settings.camera
        if cam.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if not self.ui_visible:
            # UI hidden: plain camera only, no pose tracking, no typing.
            self.compositor.update_camera(frame_rgb)
            return

        # Detection first, so a curl is typed before any time is spent on
        # uploading textures or drawing.
        timestamp_ms = int((now - self._start) * 1000)
        arm_cfg = self.settings.arm
        curl_side, gate_side = arm_cfg.curl_side, arm_cfg.gate_side
        body = self.tracker.detect(frame_rgb, timestamp_ms, cam.mirror, arm_cfg.swap_arms)
        hands = self.hands.detect_all(frame_rgb, timestamp_ms)

        # The switch hand: fist = typing on, open = off. It is matched to
        # the body by wrist position, so it is always on the opposite side
        # to the typing arm.
        h, w = frame_rgb.shape[:2]
        wrists = body.wrists if body is not None else {"left": None, "right": None}
        hand = pick_switch_hand(hands, wrists[gate_side], wrists[curl_side])
        was_armed = self.gate.fist
        armed = self.gate.update(None if hand is None else to_pixels(hand, w, h))
        if armed != was_armed:
            self.sounds.play("resume" if armed else "pause")
        self._update_hand_points(hand, frame_rgb.shape)

        state = self._state
        arm = None if body is None else getattr(body, curl_side)
        if arm is None:
            self.detector.reset()
            state.body = False
            self._arm_pts = {"left": None, "right": None}
        else:
            if armed:
                curls = {"left": None, "right": None, curl_side: arm.curl}
                for event in self.detector.update(curls["left"], curls["right"], now):
                    if not self.paused:
                        self._handle_curl(event, now)
            else:
                # Open hand: still show the arm, but type nothing.
                self.detector.cancel(now)
                self.detector.curl = {"left": 0.0, "right": 0.0, curl_side: arm.curl}
                self.detector.visible = {"left": False, "right": False, curl_side: True}
            state.body = True
            state.angle = {"left": None, "right": None, curl_side: arm.angle}
            self._update_arm_points(body, frame_rgb.shape, curl_side)
        state.curl = dict(self.detector.curl)
        state.visible = dict(self.detector.visible)
        state.active = self.detector.active
        state.armed = armed
        state.hand_visible = self.gate.visible
        state.curl_side, state.gate_side = curl_side, gate_side

        # Pauses end letters and words. Without a body the arm cannot be
        # curling, so the time since the last curl keeps counting.
        for event in self.composer.update(self.detector.pause_time(now)):
            self._apply_composer(event, now)
        self.compositor.update_camera(frame_rgb)

    def _handle_curl(self, event, now: float) -> None:
        if event.kind == CurlKind.DOT:
            self._apply_composer(self.composer.add_symbol("."), now)
        elif event.kind == CurlKind.DASH:
            self._apply_composer(self.composer.add_symbol("-"), now)

    def _update_hand_points(self, hand, shape) -> None:
        """Switch-hand landmarks in window pixels, lightly smoothed for drawing."""
        if hand is None:
            self._hand_pts = None
            return
        h, w = shape[:2]
        (sx, sy), (ox, oy) = cover_crop((w, h), WINDOW_SIZE)
        W, H = WINDOW_SIZE
        pts = np.column_stack(((hand[:, 0] - ox) / sx * W, (hand[:, 1] - oy) / sy * H))
        if self._hand_pts is not None and self._hand_pts.shape == pts.shape:
            pts = self._hand_pts * 0.4 + pts * 0.6
        self._hand_pts = pts

    def _update_arm_points(self, body, shape, only_side: str) -> None:
        """
        Convert each arm's landmarks (shoulder, elbow, wrist, pinky, index,
        thumb) to window pixels for the skeleton drawing. Positions are
        lightly smoothed so the bones do not jitter; this has no effect on
        input, which always uses the raw angle.
        """
        h, w = shape[:2]
        (sx, sy), (ox, oy) = cover_crop((w, h), WINDOW_SIZE)
        W, H = WINDOW_SIZE
        for side, arm in (("left", body.left), ("right", body.right)):
            if arm is None or side != only_side:
                self._arm_pts[side] = None
                continue
            pts = np.array([((x - ox) / sx * W, (y - oy) / sy * H)
                            for x, y in arm.points], dtype=np.float64)
            previous = self._arm_pts[side]
            if previous is not None and previous.shape == pts.shape:
                pts = previous * 0.4 + pts * 0.6
            self._arm_pts[side] = pts

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
                    self._apply_arm_setting(key, value)
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

    def _apply_arm_setting(self, key: str, value) -> None:
        e = self.settings.arm
        if key == "curl_side" and value != e.curl_side:
            # Hands swapped: drop any curl in progress and start clean.
            setattr(e, key, value)
            self.detector.cancel(time.perf_counter())
            self.gate.reset()
            self._arm_pts = {"left": None, "right": None}
            self._hand_pts = None
            self.sounds.play("resume")
            self.hud.on_swap(e.curl_side, time.perf_counter())
        setattr(e, key, value)
        self.detector.curl_threshold = e.curl_threshold
        self.detector.time_unit = e.time_unit
        self.composer.letter_gap = e.letter_gap
        self.composer.word_gap = e.word_gap

    def _reset_settings(self) -> None:
        index = self.settings.camera.index
        fresh_cam = CameraSettings(index=index)
        for field_name, value in vars(fresh_cam).items():
            setattr(self.settings.camera, field_name, value)
        for field_name, value in vars(ArmSettings()).items():
            self._apply_arm_setting(field_name, value)
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
        e = self.settings.arm
        state.fps = self.fps
        state.camera_error = self.camera.error
        state.threshold = e.curl_threshold
        state.release_threshold = self.detector.release_threshold
        state.dash_after = e.dash_after
        state.letter_gap = e.letter_gap
        state.word_gap = e.word_gap
        state.pressed = self.detector.pressed
        state.pressed_time = self.detector.pressed_time(now)
        state.code = self.composer.code
        state.text = self.composer.text
        state.paused = self.paused
        state.arms = {side: (None if p is None else [(float(x), float(y)) for x, y in p])
                      for side, p in self._arm_pts.items()}
        state.hand = (None if self._hand_pts is None
                      else [(float(x), float(y)) for x, y in self._hand_pts])
        state.curl_side, state.gate_side = e.curl_side, e.gate_side

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
