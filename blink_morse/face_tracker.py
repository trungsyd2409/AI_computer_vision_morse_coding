"""
Wrapper around MediaPipe's Face Landmarker (Tasks API).

For every frame MediaPipe returns 478 face landmarks and a 4x4 matrix
with the head pose. MediaPipe has no blendshape for nostril flaring, so
`NostrilDetector` (see nostril.py) measures the dark area of the nostril
openings under the nose, using the landmarks to know where to look and the
head pitch to ignore tilting the head back. The result is how much bigger
the nostrils are than at rest; the app decides from the threshold in the
settings whether that counts as a Morse "press".

Which side is "left"
--------------------
MediaPipe names landmarks after the face shown in the image, not after the
viewer. When the frame is mirrored before detection (the usual selfie
view), the face in the image is a mirror copy, so its "left" is the user's
real right. `build_result` does that swap once, so the rest of the app only
ever deals with the user's own left and right.
"""

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import FACE_MODEL_PATH, FACE_MODEL_URL
from .nostril import NostrilDetector, NostrilResult, pitch_from_matrix


def ensure_model(path: Path = FACE_MODEL_PATH, url: str = FACE_MODEL_URL) -> Path:
    """Download the face landmark model (about 3.7 MB) if it is missing."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face model to {path} ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


@dataclass
class FaceResult:
    """One detected face, from the user's point of view."""

    left_score: float = 0.0
    right_score: float = 0.0
    left_corners: Optional[tuple] = None
    right_corners: Optional[tuple] = None
    # Nostril dark-area measurement and flare (see nostril.py)
    nostril: NostrilResult = None


class FaceTracker:
    def __init__(self, model_path: Path = FACE_MODEL_PATH):
        # Imported here so that the rest of the package (and the tests) can
        # be used without MediaPipe installed.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1
        self._nostril = NostrilDetector()

    def recalibrate(self) -> None:
        """Learn the resting nostril size again (keep the nose relaxed)."""
        self._nostril.recalibrate()

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int,
               mirrored: bool, swap_eyes: bool = False) -> Optional[FaceResult]:
        """
        frame_rgb:    RGB uint8 image (already mirrored if `mirrored` is True)
        timestamp_ms: capture time; VIDEO mode requires it to keep increasing
        mirrored:     whether the frame was flipped horizontally
        swap_eyes:    kept for compatibility, not used by the nostril input
        """
        # Guard against two frames sharing the same millisecond.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            self._nostril.reset()
            return None

        lm = result.face_landmarks[0]
        h, w = frame_rgb.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)
        pitch = 0.0
        if result.facial_transformation_matrixes:
            pitch = pitch_from_matrix(result.facial_transformation_matrixes[0])
        nostril = self._nostril.update(frame_rgb, pts, timestamp_ms / 1000.0, pitch)
        return FaceResult(nostril=nostril)

    def close(self) -> None:
        self._landmarker.close()


def build_result(mp_left: float, mp_right: float, mirrored: bool,
                 swap_eyes: bool = False, corners: Optional[tuple] = None) -> FaceResult:
    """
    Map MediaPipe's image-face naming onto the user's own left and right.
    `corners` is (MediaPipe left points, MediaPipe right points).
    Kept separate from the tracker so it can be unit tested.
    """
    mp_left_corners, mp_right_corners = corners if corners else (None, None)
    # In a mirrored frame the depicted face is flipped, so MediaPipe's
    # "left" is the user's right.
    user_left_is_mp_left = not mirrored
    if swap_eyes:
        user_left_is_mp_left = not user_left_is_mp_left

    if user_left_is_mp_left:
        return FaceResult(mp_left, mp_right, mp_left_corners, mp_right_corners)
    return FaceResult(mp_right, mp_left, mp_right_corners, mp_left_corners)
