"""
Application loop: camera -> pose + hands -> arm gestures -> Morse -> screen.

Per frame:
  1. Take the newest camera frame (the camera runs in its own thread).
  2. Mirror it and run MediaPipe Pose (both forearms) and MediaPipe Hands
     (both hands, matched to the body by wrist position; drawn only).
  3. Bring both hands together to switch typing on or off.
  4. While typing is on, each forearm's angle against the X axis (the
     right arm's mirrored, so both read +45 for "raised towards the body")
     fires a gesture the moment it enters a zone:
       symbol arm  (left):  35..145 = dash,          -135..-35 = dot
       control arm (right): 35..145 = end of letter, -135..-35 = space
     An arm must leave its zone before it can fire again. S swaps roles.
  5. Draw the HUD (arm bones, hand skeletons, Morse chart, typed code and
     message) and let the GPU put it over the camera image.

Pressing Z hides the whole interface, shows the plain camera image and
switches input off until Z is pressed again.
"""

import time

import cv2
import moderngl
import numpy as np
import pygame

from .arm_gestures import TogetherSwitch, ZoneKey, mirror_angle, together_ratio
from .arm_tracker import ArmTracker
from .audio import SoundBank
from .camera import CameraStream
from .compositor import Compositor, cover_crop
from .config import APP_TITLE, WINDOW_SIZE, AppSettings, ArmSettings, CameraSettings
from .hand_tracker import HandTracker, assign_hands
from .hud import Hud, HudState
from .morse import EventKind, MorseComposer
from .settings_window import SettingsLink

SIDES = ("left", "right")


class BlinkMorseApp:
    def __init__(self):
        self.settings = AppSettings.load()

        # The models are loaded before the window opens so a first-run
        # download does not leave a frozen black window on screen.
        self.tracker = ArmTracker()
        self.hands = HandTracker()
        self.switch = TogetherSwitch()   # both hands together = typing on / off

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                        pygame.GL_CONTEXT_PROFILE_CORE)
        # Required on macOS for a core profile, harmless elsewhere.
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FLAGS,
                                        pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
        # VSync off: the camera already paces the loop.
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)
        pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(APP_TITLE)

        self.ctx = moderngl.create_context()
        self.compositor = Compositor(self.ctx, WINDOW_SIZE)
        self.hud = Hud(WINDOW_SIZE)
        self.sounds = SoundBank()

        self.camera = CameraStream(self.settings.camera)
        self.camera.start()

        # One instant zone key per arm; what a zone means depends on the role.
        self.arm_keys = {side: ZoneKey() for side in SIDES}
        self._apply_key_settings()
        self.composer = MorseComposer()
        self.settings_link = SettingsLink()

        self.paused = False               # P stops typing, e.g. to rest the arms
        self.ui_visible = True            # Z hides the UI and stops input
        self.show_chart = True
        self.show_angles = True           # H: forearm angle readouts for debugging
        self.fps = 0.0
        self._frame_id = 0
        self._start = time.perf_counter()
        self._last_stats = 0.0
        self._props_sent = -1
        self._state = HudState()
        self._arm_pts = {"left": None, "right": None}
        self._hand_pts = {"left": None, "right": None}

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
                self._props_sent = -1
            elif event.key == pygame.K_s:
                self._apply_arm_setting("symbol_side", self.settings.arm.control_side)
            elif event.key == pygame.K_z:
                self._toggle_ui()
            elif event.key == pygame.K_p:
                self.paused = not self.paused
                self._reset_input()
                self.sounds.play("pause" if self.paused else "resume")
            elif event.key == pygame.K_h:
                self.show_angles = not self.show_angles
            elif event.key == pygame.K_c:
                self.show_chart = not self.show_chart
            elif event.key == pygame.K_m:
                self.sounds.toggle()
            elif event.key == pygame.K_BACKSPACE:
                self._apply_composer(self.composer.delete())
            elif event.key == pygame.K_DELETE:
                self._apply_composer(self.composer.clear())
        return True

    def _reset_input(self, switch_off: bool = False) -> None:
        for key in self.arm_keys.values():
            key.reset()
        if switch_off:
            self.switch.reset()

    def _toggle_ui(self) -> None:
        self.ui_visible = not self.ui_visible
        self._state.body = False
        self._arm_pts = {"left": None, "right": None}
        self._hand_pts = {"left": None, "right": None}
        self._reset_input(switch_off=True)

    def _process_frame(self, frame_bgr: np.ndarray, now: float) -> None:
        cam = self.settings.camera
        if cam.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if not self.ui_visible:
            # UI hidden: plain camera only, no tracking, no typing.
            self.compositor.update_camera(frame_rgb)
            return

        timestamp_ms = int((now - self._start) * 1000)
        arm_cfg = self.settings.arm
        symbol_side = arm_cfg.symbol_side
        body = self.tracker.detect(frame_rgb, timestamp_ms, cam.mirror, arm_cfg.swap_arms)
        detected = self.hands.detect_all(frame_rgb, timestamp_ms)

        h, w = frame_rgb.shape[:2]
        wrists = body.wrists if body is not None else {"left": None, "right": None}
        hands = assign_hands(detected, wrists)

        # Both hands together flips typing on / off.
        ratio = None if body is None else together_ratio(body.wrists, body.shoulders, (w, h))
        if self.switch.update(ratio, now):
            self._reset_input()
            self.sounds.play("resume" if self.switch.on else "pause")
            self.hud.on_switch(self.switch.on, now)
        armed = self.switch.on and not self.paused
        # Nothing is typed while the hands are together: the arms are busy
        # with the switch gesture.
        listening = armed and not self.switch.together

        arms = {side: (None if body is None else getattr(body, side)) for side in SIDES}
        for side in SIDES:
            arm = arms[side]
            angle = None
            if arm is not None:
                angle = mirror_angle(arm.forearm_x) if side == "right" else arm.forearm_x
            key = self.arm_keys[side]
            if listening:
                gesture = key.update(angle, now)
                if gesture is not None:
                    self._fire(side, gesture, symbol_side)
            else:
                # Not typing: show the angle but forget any state, so an arm
                # that is already in a zone when typing starts types nothing.
                key.reset()
                key.angle = angle
                key.zone = None if angle is None else key.zone_of(angle)

        self._update_points(arms, hands, frame_rgb.shape)

        state = self._state
        state.body = body is not None
        state.armed = armed
        state.together = self.switch.together
        keys = self.arm_keys
        state.angle = {side: keys[side].angle for side in SIDES}
        state.zone = {side: keys[side].zone for side in SIDES}
        state.ready = {side: keys[side].ready for side in SIDES}
        # Zone already fired, waiting for the arm to leave it.
        state.latched = {side: (keys[side].latched if listening else None) for side in SIDES}
        self.compositor.update_camera(frame_rgb)

    def _fire(self, side: str, gesture: str, symbol_side: str) -> None:
        if side == symbol_side:
            symbol = "-" if gesture == ZoneKey.UP else "."
            self._apply_composer(self.composer.add_symbol(symbol), side)
        elif gesture == ZoneKey.UP:
            self._apply_composer(self.composer.end_letter(), side)
        else:
            for event in self.composer.add_space():
                self._apply_composer(event, side)

    def _update_points(self, arms: dict, hands: dict, shape) -> None:
        """
        Arm and hand landmarks in window pixels, for drawing only. They are
        lightly smoothed so the bones do not jitter; input always uses the
        raw angles.
        """
        h, w = shape[:2]
        (sx, sy), (ox, oy) = cover_crop((w, h), WINDOW_SIZE)
        W, H = WINDOW_SIZE

        def to_window(points):
            pts = np.asarray(points, dtype=np.float64)[:, :2]
            return np.column_stack(((pts[:, 0] - ox) / sx * W, (pts[:, 1] - oy) / sy * H))

        def smooth(previous, pts):
            if previous is not None and previous.shape == pts.shape:
                return previous * 0.4 + pts * 0.6
            return pts

        for side in SIDES:
            arm = arms[side]
            # Shoulder, elbow, wrist only: the hand comes from the hand model.
            self._arm_pts[side] = (None if arm is None else
                                   smooth(self._arm_pts[side], to_window(arm.points[:3])))
            hand = hands[side]
            self._hand_pts[side] = (None if hand is None else
                                    smooth(self._hand_pts[side], to_window(hand)))

    def _apply_composer(self, event, side: str = "") -> None:
        if event is None:
            return
        now = time.perf_counter()
        kind = event.kind
        if kind == EventKind.SYMBOL:
            self.sounds.play("dot" if event.value == "." else "dash")
            self.hud.on_gesture(side, now)
        elif kind == EventKind.LETTER:
            self.sounds.play("letter")
            self.hud.on_gesture(side, now)
        elif kind == EventKind.SPACE:
            self.sounds.play("space")
            self.hud.on_gesture(side, now)
        elif kind == EventKind.INVALID:
            self.sounds.play("invalid")
            self.hud.on_invalid(event.value, now)
        elif kind == EventKind.DELETE:
            self.sounds.play("delete")
        elif kind == EventKind.CLEAR:
            self.sounds.play("clear")

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
                self._props_sent = -1
        elif key == "mirror":
            cam.mirror = bool(value)
        else:
            setattr(cam, key, value)
            self.camera.set_property(key, value)

    def _apply_arm_setting(self, key: str, value) -> None:
        e = self.settings.arm
        if not hasattr(e, key):
            return
        changed_roles = key == "symbol_side" and value != e.symbol_side
        setattr(e, key, value)
        if changed_roles or key == "swap_arms":
            self._reset_input()
            self._arm_pts = {"left": None, "right": None}
            self._hand_pts = {"left": None, "right": None}
        if changed_roles:
            self.sounds.play("resume")
        self._apply_key_settings()

    def _apply_key_settings(self) -> None:
        e = self.settings.arm
        for k in self.arm_keys.values():
            k.dash = (min(e.dash_min, e.dash_max), max(e.dash_min, e.dash_max))
            k.dot = (min(e.dot_min, e.dot_max), max(e.dot_min, e.dot_max))
            k.margin = e.exit_margin
        self.switch.ratio = e.together_ratio
        self.switch.release = max(0.6, e.together_ratio + 0.2)

    def _reset_settings(self) -> None:
        index = self.settings.camera.index
        fresh_cam = CameraSettings(index=index)
        for field_name, value in vars(fresh_cam).items():
            setattr(self.settings.camera, field_name, value)
        for field_name, value in vars(ArmSettings()).items():
            self._apply_arm_setting(field_name, value)
        self.camera.restore_driver_defaults()
        self.camera.reopen()
        self._props_sent = -1

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
        # Re-seed the sliders whenever the camera reports new driver values
        # (first open, reopen, another device). None = not supported.
        version = self.camera.properties_version
        if self._props_sent != version and self.camera.properties:
            self.settings_link.send("props", self.camera.properties)
            self._props_sent = version

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    def _render(self, now: float) -> None:
        state = self._state
        e = self.settings.arm
        state.camera_error = self.camera.error
        state.code = self.composer.code
        state.text = self.composer.text
        state.paused = self.paused
        state.symbol_side, state.control_side = e.symbol_side, e.control_side
        state.arms = {side: (None if p is None else [(float(x), float(y)) for x, y in p])
                      for side, p in self._arm_pts.items()}
        state.hands = {side: (None if p is None else [(float(x), float(y)) for x, y in p])
                       for side, p in self._hand_pts.items()}
        state.show_chart = self.show_chart
        state.show_angles = self.show_angles

        self.hud.draw(state, now, self.ui_visible)
        self.compositor.update_ui(self.hud.ui)
        self.compositor.render()
        pygame.display.flip()
