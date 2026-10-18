"""
Threaded webcam reader that tries hard to get a working picture.

`cv2.VideoCapture.read()` blocks until the next frame arrives. Running it in
its own thread means the render loop never waits on the camera, and it
always gets the most recent frame instead of an old buffered one.

Why this is more than a plain VideoCapture
-----------------------------------------
Virtual cameras (Camo, OBS, phone webcam apps) and some laptop drivers do
not behave like normal webcams:

* asking for MJPG, a size or a frame rate they do not offer can make them
  hang inside `read()` or send nothing but black frames;
* they often do not support brightness / contrast / gain / exposure, and
  the driver then reports -1 for those values.

So the camera is opened with a list of "modes" (backend + format + size +
fps), best first. A mode is kept only once it delivers a frame that is not
completely black. If `read()` hangs, a watchdog thread starts a fresh
capture thread with the next mode. Properties the driver does not support
are reported as None, so the settings window can grey them out instead of
showing -1.
"""

import platform
import threading
import time
from typing import Optional

import cv2
import numpy as np

from .config import CameraSettings

IS_WINDOWS = platform.system() == "Windows"

# Map setting names to OpenCV property ids for the sliders in the settings UI.
PROPERTY_IDS = {
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "gain": cv2.CAP_PROP_GAIN,
}

PROBE_SECONDS = 2.5      # how long a new mode may send only black frames
STALL_READ = 3.0         # read() blocked this long = camera hung
STALL_OPEN = 10.0        # opening may take a while (Media Foundation is slow)
BLACK_LEVEL = 12         # brightest pixel below this = black frame


def camera_modes(s: CameraSettings) -> list:
    """
    (backend, fourcc, width, height, fps) to try, best first. None means
    "do not ask, use whatever the driver gives".
    """
    w, h, fps = s.width, s.height, s.fps
    if IS_WINDOWS:
        return [
            (cv2.CAP_DSHOW, "MJPG", w, h, fps),
            (cv2.CAP_DSHOW, None, w, h, fps),
            (cv2.CAP_DSHOW, None, w, h, 30),
            (cv2.CAP_DSHOW, None, None, None, None),
            (cv2.CAP_MSMF, None, w, h, fps),
            (cv2.CAP_MSMF, None, None, None, None),
        ]
    if platform.system() == "Darwin":
        return [
            (cv2.CAP_AVFOUNDATION, None, w, h, fps),
            (cv2.CAP_AVFOUNDATION, None, None, None, None),
        ]
    return [
        (cv2.CAP_V4L2, "MJPG", w, h, fps),
        (cv2.CAP_V4L2, None, w, h, fps),
        (cv2.CAP_V4L2, None, None, None, None),
        (cv2.CAP_ANY, None, None, None, None),
    ]


def is_black(frame: np.ndarray) -> bool:
    """True when even the brightest pixel is near black (sampled for speed)."""
    return int(frame[::8, ::8].max()) < BLACK_LEVEL


def mode_name(mode, backend_name: str) -> str:
    _, fourcc, w, h, fps = mode
    parts = [backend_name]
    if fourcc:
        parts.append(fourcc)
    if w is None:
        parts.append("driver default")
    elif fps:
        parts.append(f"{fps} fps")
    return " ".join(parts)


class CameraStream:
    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._watchdog: Optional[threading.Thread] = None
        self._pending: list = []            # property changes queued for the thread

        self._gen = 0                       # id of the capture thread that is current
        # Index into camera_modes(): start with the mode that worked last time.
        self._mode = settings.mode_hint % len(camera_modes(settings))
        self._failures = 0                  # modes tried in a row without a picture
        self._phase = "idle"                # "open", "read" or "idle", for the watchdog
        self._progress = 0.0                # last time the capture thread made progress

        # Public, read-only stats for the HUD and settings window.
        self.backend_name = "-"
        self.actual_size = (0, 0)
        self.actual_fps = 0.0
        self.error: Optional[str] = None
        # Driver values read after opening; None = not supported by this camera.
        self.properties: dict = {}
        self.properties_version = 0         # bumped whenever `properties` changes
        self.driver_defaults: dict = {}     # values found before any change

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._spawn()
        self._watchdog = threading.Thread(target=self._watch, name="camera-watchdog",
                                          daemon=True)
        self._watchdog.start()

    def stop(self) -> None:
        self._running = False
        with self._lock:
            self._gen += 1
            self._new_frame.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def reopen(self) -> None:
        """Reopen the device with new settings, starting from the best mode."""
        with self._lock:
            self._pending.append(("__reopen__", None))

    def set_property(self, name: str, value) -> None:
        """Queue a property change; it is applied inside the capture thread."""
        with self._lock:
            self._pending.append((name, value))

    def restore_driver_defaults(self) -> None:
        """Put brightness, contrast, gain and exposure back to their original values."""
        with self._lock:
            for name, value in self.driver_defaults.items():
                self._pending.append((name, value))

    def open_driver_dialog(self) -> None:
        """Show the native driver dialog (DirectShow on Windows only)."""
        with self._lock:
            self._pending.append(("__dialog__", None))

    # -- frame access ---------------------------------------------------------

    def wait_frame(self, last_id: int, timeout: float = 0.1):
        """
        Block until a frame newer than `last_id` is available.
        Returns (frame_id, frame) or (last_id, None) on timeout.
        """
        with self._new_frame:
            if self._frame_id == last_id:
                self._new_frame.wait(timeout)
            if self._frame_id == last_id or self._frame is None:
                return last_id, None
            return self._frame_id, self._frame

    # -- threads --------------------------------------------------------------

    def _spawn(self) -> None:
        """Start a new capture thread; any older one quits when it wakes up."""
        with self._lock:
            self._gen += 1
            gen = self._gen
            self._progress = time.perf_counter()
            self._phase = "idle"
        self._cap = None
        self._thread = threading.Thread(target=self._run, args=(gen,),
                                        name=f"camera-{gen}", daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        """Restart capture with the next mode if open() or read() hangs."""
        while self._running:
            time.sleep(0.5)
            limit = {"open": STALL_OPEN, "read": STALL_READ}.get(self._phase)
            if limit is None or time.perf_counter() - self._progress < limit:
                continue
            what = "opening" if self._phase == "open" else "sending frames"
            self.error = f"Camera {self.settings.index} stopped {what}, trying another mode"
            self.actual_fps = 0.0
            self._next_mode()
            # The hung thread keeps its device handle until read() returns;
            # it then sees it is no longer current and releases it.
            self._spawn()

    def _current(self, gen: int) -> bool:
        return self._running and gen == self._gen

    def _next_mode(self) -> None:
        self._mode = (self._mode + 1) % len(camera_modes(self.settings))
        self._failures += 1

    def _touch(self, phase: str) -> None:
        self._phase = phase
        self._progress = time.perf_counter()

    # -- capture thread -------------------------------------------------------

    def _open_mode(self, gen: int) -> Optional[cv2.VideoCapture]:
        s = self.settings
        mode = camera_modes(s)[self._mode]
        backend, fourcc, w, h, fps = mode
        self._touch("open")
        cap = cv2.VideoCapture(s.index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        if fourcc:
            # MJPG must be requested before the size, otherwise many webcams
            # fall back to raw YUY2 which caps 720p at around 10 FPS.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if w:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._current(gen):
            cap.release()
            return None
        self.backend_name = mode_name(mode, cap.getBackendName())
        return cap

    def _run(self, gen: int) -> None:
        cap = None
        probe_start = None              # set while a new mode is on trial
        fps_clock, fps_frames = time.perf_counter(), 0
        try:
            while self._current(gen):
                self._handle_pending(gen)
                if self._cap is None and cap is not None:
                    cap.release()       # a reopen was requested
                    cap = None

                if cap is None:
                    cap = self._open_mode(gen)
                    if cap is None:
                        if not self._current(gen):
                            break
                        self._mode_failed(f"Camera {self.settings.index} could not be opened")
                        self._touch("idle")
                        time.sleep(0.3 if self._failures < len(camera_modes(self.settings)) else 1.0)
                        continue
                    self._cap = cap
                    self._apply_image_settings(cap)
                    probe_start = time.perf_counter()
                    fps_clock, fps_frames = probe_start, 0

                self._touch("read")
                ok, frame = cap.read()
                self._touch("read")
                if not self._current(gen):
                    break
                now = time.perf_counter()

                if not ok or frame is None:
                    cap.release()
                    cap = self._cap = None
                    self.actual_fps = 0.0
                    if probe_start is not None:
                        self._mode_failed(f"Camera {self.settings.index} sends no frames "
                                          "in this mode, trying another")
                    else:
                        # It worked before: unplugged or taken by another app.
                        self.error = "Camera stopped sending frames, reconnecting"
                        self._touch("idle")
                        time.sleep(0.5)
                    continue

                if probe_start is not None:
                    if not is_black(frame):
                        probe_start = None          # this mode works: keep it
                        self._failures = 0
                        self.settings.mode_hint = self._mode
                        self.error = None
                    elif now - probe_start > PROBE_SECONDS:
                        if self._failures + 1 < len(camera_modes(self.settings)):
                            cap.release()
                            cap = self._cap = None
                            self._mode_failed("Camera sends black frames, trying another mode")
                            continue
                        # Every mode is black: keep this one and say so.
                        probe_start = None
                        self.error = "Only black frames. Lens covered, or Camo phone not connected?"

                with self._new_frame:
                    self._frame = frame
                    self._frame_id += 1
                    self._new_frame.notify_all()

                fps_frames += 1
                if now - fps_clock >= 1.0:
                    self.actual_fps = fps_frames / (now - fps_clock)
                    self.actual_size = (frame.shape[1], frame.shape[0])
                    fps_clock, fps_frames = now, 0
        finally:
            if cap is not None:
                cap.release()
            if self._cap is cap:
                self._cap = None

    def _mode_failed(self, message: str) -> None:
        self.error = message
        self._next_mode()

    # -- properties -----------------------------------------------------------

    def _supported(self, cap, name: str, pid: int) -> Optional[float]:
        """Current driver value, or None if the camera does not support it."""
        try:
            value = float(cap.get(pid))
        except cv2.error:
            return None
        # DirectShow and Media Foundation report -1 for controls the driver
        # does not have (virtual cameras such as Camo have none of them).
        if value == -1.0 or np.isnan(value):
            return None
        return value

    def _apply_image_settings(self, cap) -> None:
        s = self.settings
        found = {name: self._supported(cap, name, pid) for name, pid in PROPERTY_IDS.items()}

        # Remember the untouched driver values once, for "Reset defaults".
        if not self.driver_defaults:
            self.driver_defaults = {n: v for n, v in found.items() if v is not None}

        if found["exposure"] is not None:
            self._set_auto_exposure(cap, s.auto_exposure)
        for name, pid in PROPERTY_IDS.items():
            value = getattr(s, name)
            if value is None or found[name] is None:
                continue            # never changed, or the camera cannot do it
            if name == "exposure" and s.auto_exposure:
                continue            # manual exposure is ignored while auto is on
            cap.set(pid, float(value))

        self.properties = {name: (None if found[name] is None else self._supported(cap, name, pid))
                           for name, pid in PROPERTY_IDS.items()}
        self.properties_version += 1

    def _set_auto_exposure(self, cap, enabled: bool) -> None:
        # Backends disagree on the values: V4L2 and DirectShow through OpenCV
        # use 0.75 / 0.25, Media Foundation uses 1 / 0.
        if self.backend_name.startswith("MSMF"):
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if enabled else 0)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if enabled else 0.25)

    def _handle_pending(self, gen: int) -> None:
        with self._lock:
            pending, self._pending = self._pending, []
        cap = self._cap
        for name, value in pending:
            if name == "__reopen__":
                # New size, fps or device: start again from the best mode.
                self._mode = 0
                self.settings.mode_hint = 0
                self._failures = 0
                self._cap = None
                self.driver_defaults = {}       # the new device has its own
                cap = None
            elif cap is None:
                continue
            elif name == "__dialog__":
                cap.set(cv2.CAP_PROP_SETTINGS, 1)
            elif name == "auto_exposure":
                if self.properties.get("exposure") is None:
                    continue
                self._set_auto_exposure(cap, bool(value))
                if not value and self.settings.exposure is not None:
                    cap.set(cv2.CAP_PROP_EXPOSURE, self.settings.exposure)
            elif name in PROPERTY_IDS:
                if self.properties.get(name) is None:
                    continue      # this camera does not have the control
                if name == "exposure" and self.settings.auto_exposure:
                    continue      # manual exposure is ignored while auto is on
                cap.set(PROPERTY_IDS[name], float(value))
