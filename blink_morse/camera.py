"""
Threaded webcam reader.

`cv2.VideoCapture.read()` blocks until the next frame arrives. Running it in
its own thread means the render loop never waits on the camera, and it
always gets the most recent frame instead of an old buffered one.
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


def _backends() -> list:
    """Capture backends to try, best first, for the current OS."""
    if IS_WINDOWS:
        # DirectShow opens fast and supports MJPG at high frame rates.
        # Media Foundation is the fallback for cameras DirectShow rejects.
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if platform.system() == "Darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


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
        self._pending: list = []            # property changes queued for the thread

        # Public, read-only stats for the HUD and settings window.
        self.backend_name = "-"
        self.actual_size = (0, 0)
        self.actual_fps = 0.0
        self.error: Optional[str] = None
        self.properties: dict = {}          # driver values read after opening
        self.driver_defaults: dict = {}     # values found before any change

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._release()

    def reopen(self) -> None:
        """Ask the capture thread to reopen the device with new settings."""
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

    # -- capture thread -------------------------------------------------------

    def _open(self) -> bool:
        s = self.settings
        self._release()
        for backend in _backends():
            cap = cv2.VideoCapture(s.index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # MJPG must be requested before the size, otherwise many webcams
            # fall back to raw YUY2 which caps 720p at around 10 FPS.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
            cap.set(cv2.CAP_PROP_FPS, s.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue

            self._cap = cap
            self.backend_name = cap.getBackendName()
            self.actual_size = (frame.shape[1], frame.shape[0])
            self.error = None
            self._apply_image_settings()
            return True

        self.error = f"Camera {s.index} could not be opened"
        return False

    def _apply_image_settings(self) -> None:
        s = self.settings
        # Remember the untouched driver values once, for "Reset defaults".
        if not self.driver_defaults:
            self.driver_defaults = {
                name: float(self._cap.get(pid)) for name, pid in PROPERTY_IDS.items()
            }

        self._set_auto_exposure(s.auto_exposure)
        for name, pid in PROPERTY_IDS.items():
            value = getattr(s, name)
            if value is None:
                continue            # user never changed it, keep the driver value
            if name == "exposure" and s.auto_exposure:
                continue            # manual exposure is ignored while auto is on
            self._cap.set(pid, float(value))

        self.properties = {
            name: float(self._cap.get(pid)) for name, pid in PROPERTY_IDS.items()
        }

    def _set_auto_exposure(self, enabled: bool) -> None:
        # Backends disagree on the values: V4L2 and DirectShow through OpenCV
        # use 0.75 / 0.25, Media Foundation uses 1 / 0.
        if self.backend_name == "MSMF":
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if enabled else 0)
        else:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if enabled else 0.25)

    def _handle_pending(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, []
        for name, value in pending:
            if name == "__reopen__":
                self._open()
            elif self._cap is None:
                continue
            elif name == "__dialog__":
                self._cap.set(cv2.CAP_PROP_SETTINGS, 1)
            elif name == "auto_exposure":
                self._set_auto_exposure(bool(value))
                if not value and self.settings.exposure is not None:
                    self._cap.set(cv2.CAP_PROP_EXPOSURE, self.settings.exposure)
            elif name in PROPERTY_IDS:
                if name == "exposure" and self.settings.auto_exposure:
                    continue      # manual exposure is ignored while auto is on
                self._cap.set(PROPERTY_IDS[name], float(value))

    def _run(self) -> None:
        self._open()
        fps_clock = time.perf_counter()
        fps_frames = 0

        while self._running:
            self._handle_pending()

            if self._cap is None:
                # No device yet: retry once a second without spinning the CPU.
                time.sleep(1.0)
                self._open()
                continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                # Unplugged or grabbed by another app. Try again shortly.
                self.error = "Camera stopped sending frames"
                self._release()
                time.sleep(0.5)
                continue

            with self._new_frame:
                self._frame = frame
                self._frame_id += 1
                self._new_frame.notify_all()

            fps_frames += 1
            now = time.perf_counter()
            if now - fps_clock >= 1.0:
                self.actual_fps = fps_frames / (now - fps_clock)
                self.actual_size = (frame.shape[1], frame.shape[0])
                fps_clock, fps_frames = now, 0

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
