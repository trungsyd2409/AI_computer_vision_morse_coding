"""
Application loop: camera -> hand tracking -> gestures -> Morse -> screen.

Per frame:
  1. Take the newest camera frame (the camera runs in its own thread).
  2. Mirror it and run MediaPipe to find the right hand.
  3. Convert the landmarks to screen pixels and feed the pinch detector.
  4. Translate touch events into Morse input (dot, dash, commit, delete).
  5. Draw the HUD and let the GPU compose the final image.

Enter sends the typed message: it moves into the chat history above the
message box and the box is emptied for the next one.
"""

import time
from datetime import datetime

import cv2
import moderngl
import numpy as np
import pygame

from .audio import SoundBank
from .camera import CameraStream
from .compositor import Compositor, cover_crop
from .config import (APP_TITLE, CAMERA_RECT, WINDOW_SIZE, AppSettings, CameraSettings,
                     GestureSettings)
from .filters import OneEuroFilter
from .gestures import PinchDetector, TouchPhase
from .hand_tracker import HandTracker
from .hud import ChatMessage, Hud, HudState
from .morse import EventKind, MorseComposer
from .settings_window import SettingsLink


class MorseHandApp:
    def __init__(self):
        self.settings = AppSettings.load()

        # The model is loaded before the window opens so a first-run
        # download does not leave a frozen black window on screen.
        self.tracker = HandTracker()

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                        pygame.GL_CONTEXT_PROFILE_CORE)
        # Required on macOS for a core profile, harmless elsewhere.
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FLAGS,
                                        pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
        # VSync off: the camera already paces the loop, and vsync would add
        # up to one extra frame of latency between a tap and its feedback.
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)
        pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(APP_TITLE)

        self.ctx = moderngl.create_context()
        self.compositor = Compositor(self.ctx, WINDOW_SIZE, CAMERA_RECT)
        self.hud = Hud(WINDOW_SIZE)
        self.sounds = SoundBank()

        self.camera = CameraStream(self.settings.camera)
        self.camera.start()

        g = self.settings.gesture
        self.detector = PinchDetector(g.touch_ratio, g.release_gap, g.hold_to_clear)
        self.composer = MorseComposer(g.double_tap_window)
        self.smoother = OneEuroFilter()
        self.settings_link = SettingsLink()

        self.show_chart = True
        self.fps = 0.0
        self._frame_id = 0
        self._start = time.perf_counter()
        self._last_stats = 0.0
        self._props_sent = False
        self._state = HudState()

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
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._send_message()
            elif event.key == pygame.K_h:
                self.show_chart = not self.show_chart
            elif event.key == pygame.K_m:
                self.sounds.toggle()
            elif event.key == pygame.K_BACKSPACE:
                self._apply_composer(self.composer.delete(), time.perf_counter())
            elif event.key == pygame.K_DELETE:
                self._apply_composer(self.composer.clear(), time.perf_counter())
        return True

    def _send_message(self) -> None:
        """Move the typed text into the chat history (this session only)."""
        text = self.composer.take_text()
        if not text:
            return
        self._state.chat.append(ChatMessage(text, datetime.now().strftime("%H:%M")))
        del self._state.chat[:-50]           # keep the history short
        self.sounds.play("letter")
        self.hud.on_sent(time.perf_counter())

    def _process_frame(self, frame_bgr: np.ndarray, now: float) -> None:
        cam = self.settings.camera
        if cam.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.compositor.update_camera(frame_rgb)

        timestamp_ms = int((now - self._start) * 1000)
        result = self.tracker.detect(frame_rgb, timestamp_ms, cam.mirror,
                                     self.settings.gesture.swap_hands)

        state = self._state
        if result.right is None:
            self.detector.reset()
            self.smoother.reset()
            state.landmarks = None
            state.active_finger = None
            state.hold_progress = -1.0
            return

        points = self._to_screen(result.right, frame_rgb.shape)
        state.landmarks = self.smoother(points, now)

        for event in self.detector.update(points, now):
            self._handle_touch(event, now)

        state.closeness = dict(self.detector.closeness)
        state.active_finger = self.detector.active
        state.contact_point = self.detector.contact_point
        if self.detector.active == "pinky":
            held = now - self.detector.active_since
            state.hold_progress = min(1.0, held / self.detector.hold_time) if held > 0.2 else -1.0
        else:
            state.hold_progress = -1.0

    def _to_screen(self, landmarks: np.ndarray, shape) -> np.ndarray:
        """Normalised camera coordinates -> window pixels (same crop as the GPU)."""
        h, w = shape[:2]
        x0, y0, W, H = CAMERA_RECT
        (sx, sy), (ox, oy) = cover_crop((w, h), (W, H))
        out = np.empty_like(landmarks)
        out[:, 0] = x0 + (landmarks[:, 0] - ox) / sx * W
        out[:, 1] = y0 + (landmarks[:, 1] - oy) / sy * H
        out[:, 2] = landmarks[:, 2] * W / sx     # depth, roughly in x-pixel units
        return out

    def _handle_touch(self, event, now: float) -> None:
        finger = event.finger
        if event.phase == TouchPhase.DOWN:
            self.hud.on_touch(finger, event.point, now)
            if finger == "index":
                self._apply_composer(self.composer.add_symbol("."), now)
            elif finger == "middle":
                self._apply_composer(self.composer.add_symbol("-"), now)
            elif finger == "ring":
                self._apply_composer(self.composer.ring_tap(now), now)
        elif finger == "pinky":
            # Delete fires on release so a long hold can become "clear all"
            # without deleting a character first.
            if event.phase == TouchPhase.HOLD:
                self._apply_composer(self.composer.clear(), now)
            elif event.phase == TouchPhase.UP and event.duration < self.detector.hold_time:
                self._apply_composer(self.composer.delete(), now)

    def _apply_composer(self, event, now: float) -> None:
        if event is None:
            return
        kind = event.kind
        if kind == EventKind.SYMBOL:
            self.sounds.play("dot" if event.value == "." else "dash")
            self.hud.on_symbol(now)
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
                    self._apply_gesture_setting(key, value)
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
        elif key == "mirror":
            cam.mirror = bool(value)
        else:
            setattr(cam, key, value)
            self.camera.set_property(key, value)

    def _apply_gesture_setting(self, key: str, value) -> None:
        g = self.settings.gesture
        setattr(g, key, value)
        self.detector.touch_ratio = g.touch_ratio
        self.detector.release_gap = g.release_gap
        self.detector.hold_time = g.hold_to_clear
        self.composer.double_tap_window = g.double_tap_window

    def _reset_settings(self) -> None:
        index = self.settings.camera.index
        fresh_cam = CameraSettings(index=index)
        for field_name, value in vars(fresh_cam).items():
            setattr(self.settings.camera, field_name, value)
        for field_name, value in vars(GestureSettings()).items():
            self._apply_gesture_setting(field_name, value)
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
        state.camera_error = self.camera.error
        state.code = self.composer.code
        state.text = self.composer.text
        state.show_chart = self.show_chart

        self.hud.draw(state, now)
        self.compositor.set_panels(self.hud.panels)
        self.compositor.update_layers(self.hud.ui, self.hud.glow)
        self.compositor.render()
        pygame.display.flip()
