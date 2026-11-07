"""
Wrapper around MediaPipe's Hand Landmarker (Tasks API).

For every hand MediaPipe returns 21 landmarks twice:

* image landmarks (x, y normalised to the frame), used for drawing,
* world landmarks in metres around the hand's own centre, used for
  measuring. Angles and distances in world space do not change with the
  distance to the camera or with how the hand is turned towards it.

Up to two hands are tracked so the gesture detector can pick the one that
makes the gesture, whichever hand that is.
"""

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import HAND_MODEL_PATH, HAND_MODEL_URL

# Bones of the 21-point hand skeleton (wrist, palm and the five fingers).
HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),  # palm
    (5, 6), (6, 7), (7, 8),                    # index
    (9, 10), (10, 11), (11, 12),               # middle
    (13, 14), (14, 15), (15, 16),              # ring
    (17, 18), (18, 19), (19, 20),              # pinky
)


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
class Hand:
    points: np.ndarray     # (21, 2) landmarks in image pixels
    world: np.ndarray      # (21, 3) landmarks in metres
    label: str             # "Left" / "Right" as MediaPipe reports it
    score: float


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
            # Lower than MediaPipe's default (0.5): with three fingers folded
            # and two pressed together, the palm detector is less sure of
            # itself (one reference photo scored between 0.3 and 0.4). The
            # gesture rules are strict, so a false hand cannot type.
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list:
        """All hands in the frame, as a list of Hand."""
        # Guard against two frames sharing the same millisecond.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        h, w = frame_rgb.shape[:2]
        hands = []
        for lms, world, handedness in zip(result.hand_landmarks,
                                          result.hand_world_landmarks,
                                          result.handedness):
            hands.append(Hand(
                points=np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float64),
                world=np.array([[p.x, p.y, p.z] for p in world], dtype=np.float64),
                label=handedness[0].category_name,
                score=float(handedness[0].score),
            ))
        return hands

    def close(self) -> None:
        self._landmarker.close()
