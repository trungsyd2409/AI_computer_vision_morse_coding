"""
Wrapper around MediaPipe's Hand Landmarker, plus the fist / open-hand
switch.

One hand (the left one by default) does not type. It works as a safety
switch: make a fist and the curling arm types Morse, open the hand and
nothing is typed. The hand is drawn as a 21-point skeleton.

Fist test
---------
For the index, middle, ring and pinky fingers the distance from the wrist
to the fingertip is compared with the distance from the wrist to the
knuckle (MCP joint). On an open hand the tip is about twice as far away as
the knuckle (ratio ~1.9); in a fist the tip folds back to the palm (ratio
~1.0). The average ratio of the four fingers is used with hysteresis, so
the switch does not flicker while the hand is half closed.
"""

import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from .config import HAND_MODEL_PATH, HAND_MODEL_URL

# The 21-point hand skeleton, as pairs of landmark indices.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),   # pinky + palm
)
FINGERS = ((5, 8), (9, 12), (13, 16), (17, 20))   # (knuckle, tip)
FINGERTIPS = (4, 8, 12, 16, 20)


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


def openness_ratio(points: np.ndarray) -> float:
    """Average wrist->tip / wrist->knuckle distance of the four fingers."""
    pts = np.asarray(points, dtype=np.float64)
    wrist = pts[0]
    ratios = []
    for knuckle, tip in FINGERS:
        base = np.linalg.norm(pts[knuckle] - wrist)
        if base < 1e-6:
            continue
        ratios.append(np.linalg.norm(pts[tip] - wrist) / base)
    return float(np.mean(ratios)) if ratios else 2.0


class FistGate:
    """Fist = on, open hand = off, hand not in view = off."""

    FIST_BELOW = 1.30      # ratio under which the hand becomes a fist
    OPEN_ABOVE = 1.50      # ratio over which it counts as open again

    def __init__(self):
        self.fist = False
        self.visible = False
        self.ratio = 2.0

    def reset(self) -> None:
        self.fist = False
        self.visible = False

    def update(self, points: Optional[np.ndarray]) -> bool:
        if points is None:
            self.reset()
            return False
        self.visible = True
        self.ratio = openness_ratio(points)
        if self.fist:
            self.fist = self.ratio < self.OPEN_ABOVE
        else:
            self.fist = self.ratio < self.FIST_BELOW
        return self.fist


class HandTracker:
    def __init__(self, model_path: Path = HAND_MODEL_PATH):
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
               mirrored: bool, swap: bool = False) -> dict:
        """
        Returns {"left": pts or None, "right": pts or None} for the user's
        own hands. Each pts is a (21, 3) array: x, y normalised to the
        image, z in roughly the same scale as x.
        """
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        out = {"left": None, "right": None}
        best = {"left": 0.0, "right": 0.0}
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            # MediaPipe labels hands assuming a mirrored selfie image.
            is_right = handedness[0].category_name == "Right"
            if not mirrored:
                is_right = not is_right
            if swap:
                is_right = not is_right
            side = "right" if is_right else "left"
            score = handedness[0].score
            if score > best[side]:
                best[side] = score
                out[side] = np.array([[p.x, p.y, p.z] for p in landmarks],
                                     dtype=np.float32)
        return out

    def detect_all(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list:
        """Every detected hand as a (21, 3) array, without trusting the labels."""
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return [np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
                for landmarks in result.hand_landmarks]

    def close(self) -> None:
        self._landmarker.close()


def to_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised landmarks -> pixel units, so the fist test ignores aspect ratio."""
    pts = np.asarray(points, dtype=np.float64).copy()
    pts[:, 0] *= width
    pts[:, 1] *= height
    pts[:, 2] *= width
    return pts


def pick_switch_hand(hands: list, gate_wrist, curl_wrist,
                     max_dist: float = 0.15) -> Optional[np.ndarray]:
    """
    Choose which detected hand is the switch hand, using the body pose
    instead of MediaPipe's Left/Right hand labels (those labels can come
    out flipped, which put the hand skeleton on the same arm as the curl).

    * The switch hand is the hand whose wrist is next to the pose's wrist
      on the switch side (normalised distance < max_dist).
    * A hand that is closer to the typing arm's wrist is never chosen, so
      the hand skeleton and the arm skeleton can never be on the same side.
    * If the switch-side wrist is not visible, any hand far enough from the
      typing arm's wrist is used.
    """
    best, best_d = None, None
    for pts in hands:
        w = np.asarray(pts[0][:2], dtype=np.float64)
        d_curl = (np.linalg.norm(w - np.asarray(curl_wrist))
                  if curl_wrist is not None else None)
        if gate_wrist is not None:
            d = float(np.linalg.norm(w - np.asarray(gate_wrist)))
            if d > max_dist or (d_curl is not None and d_curl <= d):
                continue
        elif d_curl is not None:
            if d_curl < max_dist:
                continue          # this is the typing arm's own hand
            d = -float(d_curl)    # prefer the hand farthest from it
        else:
            continue              # no body to anchor to: stay off
        if best_d is None or d < best_d:
            best, best_d = pts, d
    return best
