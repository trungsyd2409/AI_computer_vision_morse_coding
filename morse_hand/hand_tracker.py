"""
Wrapper around MediaPipe's Hand Landmarker (Tasks API).

MediaPipe returns up to two hands per frame, each with 21 landmarks and a
"Left" / "Right" label. This module downloads the model on first run,
runs the detector, and hands back only what the app needs: the user's
right hand, and whether a left hand is in view (so the HUD can explain why
it is being ignored).
"""

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import HAND_MODEL_PATH, HAND_MODEL_URL


def ensure_model(path: Path = HAND_MODEL_PATH, url: str = HAND_MODEL_URL) -> Path:
    """Download the hand landmark model (about 7.5 MB) if it is missing."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {path} ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


@dataclass
class HandResult:
    right: Optional[np.ndarray] = None   # (21, 3) normalised x, y, z
    left_visible: bool = False


class HandTracker:
    def __init__(self, model_path: Path = HAND_MODEL_PATH):
        # Imported here so that the rest of the package (and the tests) can
        # be used without MediaPipe installed.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int,
               mirrored: bool, swap_hands: bool = False) -> HandResult:
        """
        frame_rgb:    RGB uint8 image (already mirrored if `mirrored` is True)
        timestamp_ms: capture time; VIDEO mode requires it to keep increasing
        mirrored:     MediaPipe labels hands assuming a mirrored selfie image,
                      so the labels are flipped back for a non-mirrored feed
        """
        # Guard against two frames sharing the same millisecond.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        out = HandResult()
        best_score = 0.0
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name
            score = handedness[0].score
            is_right = (label == "Right")
            if not mirrored:
                is_right = not is_right
            if swap_hands:
                is_right = not is_right

            if is_right and score > best_score:
                best_score = score
                out.right = np.array([[p.x, p.y, p.z] for p in landmarks],
                                     dtype=np.float32)
            elif not is_right:
                out.left_visible = True
        return out

    def close(self) -> None:
        self._landmarker.close()
