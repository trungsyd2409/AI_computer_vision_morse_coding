"""
Wrapper around MediaPipe's Pose Landmarker (Tasks API).

MediaPipe returns 33 body landmarks per frame. This module keeps the ones
the app uses and gives them the user's own left and right:

    shoulders, elbows, wrists   drawn, and the wrists drive the on/off switch
    hips                        their tilt is the Morse input (`hip_gestures`)
    knees, ankles, heels, toes  drawn only

Which side is "left"
--------------------
MediaPipe names the landmarks after the body as shown in the image. When
the frame is mirrored before detection (the usual selfie view), that body
is a mirror copy, so its "left" is the user's real right side. This module
does the swap once, so the rest of the app only deals with the user's own
left and right.
"""

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import POSE_MODEL_PATH, POSE_MODEL_URL
from .hip_gestures import hip_tilt

# MediaPipe pose index of each joint on the body's own left side; the right
# side is always the next index.
JOINTS = {
    "shoulder": 11, "elbow": 13, "wrist": 15,
    "hip": 23, "knee": 25, "ankle": 27, "heel": 29, "toe": 31,
}

# Bones as pairs of point names. `left_` / `right_` are the user's sides.
BONES = (
    [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")]
    + [(f"{s}_shoulder", f"{s}_hip") for s in ("left", "right")]
    + [(f"{s}_{a}", f"{s}_{b}") for s in ("left", "right") for a, b in (
        ("shoulder", "elbow"), ("elbow", "wrist"),
        ("hip", "knee"), ("knee", "ankle"),
        ("ankle", "heel"), ("heel", "toe"), ("ankle", "toe"))]
)

MIN_VISIBILITY = 0.5          # a joint must be this visible to be drawn / used
MIN_VISIBILITY_WRIST = 0.3    # the on/off switch can live with a weaker wrist
MIN_HIP_WIDTH = 0.02          # of the image width; closer than this = turned sideways


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


@dataclass
class BodyResult:
    # Normalised image (x, y) of every visible joint, named "<side>_<joint>"
    # with the user's own side, e.g. "right_knee".
    points: dict = field(default_factory=dict)
    # Hip line tilt in degrees (see `hip_gestures.hip_tilt`), or None when
    # a hip is hidden or the body is turned sideways.
    hip_angle: Optional[float] = None
    # Wrists and shoulders per side, or None. Shoulder width is the yardstick
    # for "hands together", so it works at any distance.
    wrists: dict = field(default_factory=lambda: {"left": None, "right": None})
    shoulders: dict = field(default_factory=lambda: {"left": None, "right": None})


def _visibility(landmark) -> float:
    v = getattr(landmark, "visibility", None)
    return 1.0 if v is None else float(v)


def build_body(landmarks, size, mirrored: bool, swap_sides: bool = False) -> BodyResult:
    """
    Turn MediaPipe's 33 landmarks into a `BodyResult`.
    Kept separate from the tracker so it can be unit tested.

    landmarks: sequence of objects with x, y (normalised) and visibility
    size:      (width, height) of the image in pixels
    """
    w, h = size
    # In a mirrored image MediaPipe's "left" body is the user's right.
    user_left_is_mp_left = (not mirrored) != swap_sides
    mp_side = {"left": 0, "right": 1} if user_left_is_mp_left else {"left": 1, "right": 0}

    points, weak = {}, {}
    for side, offset in mp_side.items():
        for joint, base in JOINTS.items():
            lm = landmarks[base + offset]
            vis = _visibility(lm)
            if vis >= MIN_VISIBILITY:
                points[f"{side}_{joint}"] = (float(lm.x), float(lm.y))
            if joint in ("wrist", "shoulder") and vis >= MIN_VISIBILITY_WRIST:
                weak[f"{side}_{joint}"] = (float(lm.x), float(lm.y))

    hip_angle = None
    if "left_hip" in points and "right_hip" in points:
        lh, rh = points["left_hip"], points["right_hip"]
        hip_angle = hip_tilt((lh[0] * w, lh[1] * h), (rh[0] * w, rh[1] * h),
                             min_width=MIN_HIP_WIDTH * w)

    return BodyResult(
        points=points,
        hip_angle=hip_angle,
        wrists={s: weak.get(f"{s}_wrist") for s in ("left", "right")},
        shoulders={s: weak.get(f"{s}_shoulder") for s in ("left", "right")},
    )


class BodyTracker:
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
               mirrored: bool, swap_sides: bool = False) -> Optional[BodyResult]:
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
        h, w = frame_rgb.shape[:2]
        return build_body(result.pose_landmarks[0], (w, h), mirrored, swap_sides)

    def close(self) -> None:
        self._landmarker.close()
