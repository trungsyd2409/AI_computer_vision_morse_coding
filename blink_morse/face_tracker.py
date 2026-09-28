"""
Wrapper around MediaPipe's Face Landmarker (Tasks API).

For every frame MediaPipe returns 478 face landmarks and 52 "blendshape"
scores that describe the expression. Two of those scores, `eyeBlinkLeft`
and `eyeBlinkRight`, go from about 0 (eye open) to about 1 (eye closed)
and are the only signal this app needs.

Which eye is "left"
-------------------
MediaPipe names both landmarks and blendshapes after the face shown in the
image, not after the viewer. When the frame is mirrored before detection
(the usual selfie view), the face in the image is a mirror copy, so its
"left" eye is the user's real right eye. This module does that swap once,
so the rest of the app only ever deals with the user's own left and right.
"""

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import FACE_MODEL_PATH, FACE_MODEL_URL


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
    """Blink scores of one detected face, from the user's point of view."""

    left_score: float            # raw blink score of the user's left eye
    right_score: float           # raw blink score of the user's right eye


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
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int,
               mirrored: bool, swap_eyes: bool = False) -> Optional[FaceResult]:
        """
        frame_rgb:    RGB uint8 image (already mirrored if `mirrored` is True)
        timestamp_ms: capture time; VIDEO mode requires it to keep increasing
        mirrored:     whether the frame was flipped horizontally
        swap_eyes:    manual override for cameras that flip the image themselves
        """
        # Guard against two frames sharing the same millisecond.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_blendshapes:
            return None

        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        return build_result(scores.get("eyeBlinkLeft", 0.0),
                            scores.get("eyeBlinkRight", 0.0), mirrored, swap_eyes)

    def close(self) -> None:
        self._landmarker.close()


def build_result(mp_left: float, mp_right: float, mirrored: bool,
                 swap_eyes: bool = False) -> FaceResult:
    """
    Map MediaPipe's image-face naming onto the user's own eyes.
    Kept separate from the tracker so it can be unit tested.
    """
    # In a mirrored frame the depicted face is flipped, so MediaPipe's
    # "left" is the user's right.
    user_left_is_mp_left = not mirrored
    if swap_eyes:
        user_left_is_mp_left = not user_left_is_mp_left

    if user_left_is_mp_left:
        return FaceResult(mp_left, mp_right)
    return FaceResult(mp_right, mp_left)
