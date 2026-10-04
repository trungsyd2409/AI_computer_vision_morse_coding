"""
Wrapper around MediaPipe's Pose Landmarker (Tasks API).

For every frame MediaPipe returns 33 body landmarks. Only the arms matter
here: shoulder, elbow, wrist and three hand points per side. From them
this module measures the elbow angle of each arm, the way a coach would
watch a dumbbell curl:

    shoulder
       \\        upper arm (humerus)
        \\
       elbow  <- angle between the two bones
        /
       /        forearm
    wrist

* arm straight      ~ 170-180 deg  -> curl 0
* fully curled      ~  40- 50 deg  -> curl 1

The angle is taken from MediaPipe's 3D "world" landmarks (metres, centred
on the hips), so it stays right even when the forearm points partly
towards the camera. If those are missing it falls back to the 2D image.

Which arm is "left"
-------------------
MediaPipe names the landmarks after the body as shown in the image. When
the frame is mirrored before detection (the usual selfie view), that body
is a mirror copy, so its "left" arm is the user's real right arm. This
module does the swap once, so the rest of the app only deals with the
user's own left and right.
"""

import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import POSE_MODEL_PATH, POSE_MODEL_URL

# MediaPipe pose indices: (shoulder, elbow, wrist, pinky, index, thumb)
MP_LEFT_ARM = (11, 13, 15, 17, 19, 21)
MP_RIGHT_ARM = (12, 14, 16, 18, 20, 22)

STRAIGHT_ANGLE = 170.0     # elbow angle that counts as curl 0
CURLED_ANGLE = 45.0        # elbow angle that counts as curl 1
MIN_VISIBILITY = 0.5       # shoulder, elbow and wrist must all be this visible


def ensure_model(path: Path = POSE_MODEL_PATH, url: str = POSE_MODEL_URL) -> Path:
    """Download the pose landmark model (about 5.5 MB) if it is missing."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {path} ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


def elbow_angle(shoulder, elbow, wrist) -> float:
    """Angle at the elbow in degrees, from any 2D or 3D points."""
    a = np.asarray(shoulder, dtype=np.float64) - np.asarray(elbow, dtype=np.float64)
    b = np.asarray(wrist, dtype=np.float64) - np.asarray(elbow, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return STRAIGHT_ANGLE
    cos = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(min(1.0, max(-1.0, cos))))


def curl_from_angle(angle: float) -> float:
    """Elbow angle -> curl score, 0 = straight arm, 1 = fully curled."""
    k = (STRAIGHT_ANGLE - angle) / (STRAIGHT_ANGLE - CURLED_ANGLE)
    return min(1.0, max(0.0, k))


@dataclass
class ArmPose:
    """One arm, from the user's point of view."""

    angle: float                 # elbow angle in degrees
    curl: float                  # 0 (straight) .. 1 (fully curled)
    # Normalised image (x, y) of shoulder, elbow, wrist, pinky, index, thumb
    points: list = field(default_factory=list)


@dataclass
class BodyResult:
    left: Optional[ArmPose] = None      # None when the arm is not visible
    right: Optional[ArmPose] = None
    # Normalised image (x, y) of each wrist, or None if not visible. Kept
    # even when the rest of the arm is hidden: the app uses it to decide
    # which detected hand belongs to which side of the body.
    wrists: dict = field(default_factory=lambda: {"left": None, "right": None})


class ArmTracker:
    def __init__(self, model_path: Path = POSE_MODEL_PATH):
        # Imported here so that the rest of the package (and the tests) can
        # be used without MediaPipe installed.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int,
               mirrored: bool, swap_arms: bool = False) -> Optional[BodyResult]:
        """
        frame_rgb:    RGB uint8 image (already mirrored if `mirrored` is True)
        timestamp_ms: capture time; VIDEO mode requires it to keep increasing
        Returns None when no person is in view.
        """
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks[0]
        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
        h, w = frame_rgb.shape[:2]

        def arm(ids) -> Optional[ArmPose]:
            s, e, wr = ids[:3]
            if min(_visibility(lm[i]) for i in (s, e, wr)) < MIN_VISIBILITY:
                return None
            if world is not None:
                pts = [(world[i].x, world[i].y, world[i].z) for i in (s, e, wr)]
            else:
                # Pixel units, so the image aspect ratio does not bend the angle.
                pts = [(lm[i].x * w, lm[i].y * h) for i in (s, e, wr)]
            angle = elbow_angle(*pts)
            return ArmPose(angle, curl_from_angle(angle),
                           [(lm[i].x, lm[i].y) for i in ids])

        def wrist(i):
            return (lm[i].x, lm[i].y) if _visibility(lm[i]) >= 0.3 else None

        result = build_result(arm(MP_LEFT_ARM), arm(MP_RIGHT_ARM), mirrored, swap_arms)
        mp_wrists = (wrist(MP_LEFT_ARM[2]), wrist(MP_RIGHT_ARM[2]))
        # Apply the same left/right swap to the wrists as to the arms.
        user_left_is_mp_left = (not mirrored) != swap_arms
        if user_left_is_mp_left:
            result.wrists = {"left": mp_wrists[0], "right": mp_wrists[1]}
        else:
            result.wrists = {"left": mp_wrists[1], "right": mp_wrists[0]}
        return result

    def close(self) -> None:
        self._landmarker.close()


def _visibility(landmark) -> float:
    v = getattr(landmark, "visibility", None)
    return 1.0 if v is None else float(v)


def build_result(mp_left: Optional[ArmPose], mp_right: Optional[ArmPose],
                 mirrored: bool, swap_arms: bool = False) -> BodyResult:
    """
    Map MediaPipe's image-body naming onto the user's own arms.
    Kept separate from the tracker so it can be unit tested.
    """
    user_left_is_mp_left = not mirrored
    if swap_arms:
        user_left_is_mp_left = not user_left_is_mp_left
    if user_left_is_mp_left:
        return BodyResult(mp_left, mp_right)
    return BodyResult(mp_right, mp_left)
