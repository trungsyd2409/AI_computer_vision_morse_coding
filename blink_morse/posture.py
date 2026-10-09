"""
Squat measurement and the "hands touching" switch.

Squat depth
-----------
The knee angle (hip -> knee -> ankle) tells how deep a squat is: about
165 deg or more standing, about 95 deg in a deep squat. It is averaged
over the legs MediaPipe can see, and measured on MediaPipe's 3D world
landmarks when they are available, so it works from the front, the side
or in between. `squat_depth` maps it to 0 (standing) .. 1 (deep).

Hands touching
--------------
Typing is only on while the two hands touch each other, for example
clasped in front of the chest during a squat. The distance between the
centres of the two hands (the mean of wrist, pinky, index and thumb) is
divided by the torso length (mid-shoulders to mid-hips, in pixels), so it
does not depend on how far you stand from the camera. The hands count as
touching under `touch_ratio` and stay touching until the distance grows
past 1.5 times that, with a short grace period against flicker.
"""

import math
from typing import Optional

# MediaPipe pose indices: (hip, knee, ankle)
LEFT_LEG = (23, 25, 27)
RIGHT_LEG = (24, 26, 28)
# (wrist, pinky, index, thumb) of each hand
LEFT_HAND = (15, 17, 19, 21)
RIGHT_HAND = (16, 18, 20, 22)
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 11, 12, 23, 24

MIN_VIS = 0.5          # joints of a leg must be at least this visible
HAND_VIS = 0.3         # hand points are often blurry, so a lower bar

STAND_ANGLE = 165.0    # knee angle that counts as depth 0
DEEP_ANGLE = 95.0      # knee angle that counts as depth 1


def angle_at(a, b, c) -> float:
    """Angle at b in degrees, from any 2D or 3D points."""
    v1 = [p - q for p, q in zip(a, b)]
    v2 = [p - q for p, q in zip(c, b)]
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return STAND_ANGLE
    cos = sum(x * y for x, y in zip(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(min(1.0, max(-1.0, cos))))


def knee_angle(landmarks, world, width: int, height: int) -> Optional[float]:
    """
    Mean knee angle of the visible legs, or None if no leg is fully visible.
    landmarks: 33 (x, y, visibility), normalised. world: 33 (x, y, z) in
    metres, or None / empty (then pixels are used).
    """
    if not landmarks or len(landmarks) < 33:
        return None
    angles = []
    for leg in (LEFT_LEG, RIGHT_LEG):
        if min(landmarks[i][2] for i in leg) < MIN_VIS:
            continue
        if world and len(world) >= 33:
            pts = [tuple(world[i][:3]) for i in leg]
        else:
            pts = [(landmarks[i][0] * width, landmarks[i][1] * height) for i in leg]
        angles.append(angle_at(*pts))
    return sum(angles) / len(angles) if angles else None


def squat_depth(angle: float) -> float:
    """Knee angle -> 0 (standing) .. 1 (deep squat)."""
    k = (STAND_ANGLE - angle) / (STAND_ANGLE - DEEP_ANGLE)
    return min(1.0, max(0.0, k))


def _hand_center(landmarks, ids, width: int, height: int):
    pts = [landmarks[i] for i in ids if landmarks[i][2] >= HAND_VIS]
    if landmarks[ids[0]][2] < HAND_VIS or not pts:     # the wrist must be seen
        return None
    return (sum(p[0] for p in pts) / len(pts) * width,
            sum(p[1] for p in pts) / len(pts) * height)


def _torso_length(landmarks, width: int, height: int) -> Optional[float]:
    """Mid-shoulders to mid-hips in pixels, with a fallback from the shoulder width."""
    def mid(a, b):
        if landmarks[a][2] < HAND_VIS or landmarks[b][2] < HAND_VIS:
            return None
        return ((landmarks[a][0] + landmarks[b][0]) / 2 * width,
                (landmarks[a][1] + landmarks[b][1]) / 2 * height)

    sh, hip = mid(LEFT_SHOULDER, RIGHT_SHOULDER), mid(LEFT_HIP, RIGHT_HIP)
    if sh is not None and hip is not None:
        length = math.dist(sh, hip)
        if length > 1e-6:
            return length
    if landmarks[LEFT_SHOULDER][2] >= HAND_VIS and landmarks[RIGHT_SHOULDER][2] >= HAND_VIS:
        width_px = math.dist((landmarks[LEFT_SHOULDER][0] * width, landmarks[LEFT_SHOULDER][1] * height),
                             (landmarks[RIGHT_SHOULDER][0] * width, landmarks[RIGHT_SHOULDER][1] * height))
        if width_px > 1e-6:
            return width_px * 1.4
    return None


def hands_distance(landmarks, width: int, height: int) -> Optional[float]:
    """Distance between the two hand centres as a fraction of the torso length."""
    if not landmarks or len(landmarks) < 33:
        return None
    left = _hand_center(landmarks, LEFT_HAND, width, height)
    right = _hand_center(landmarks, RIGHT_HAND, width, height)
    scale = _torso_length(landmarks, width, height)
    if left is None or right is None or scale is None:
        return None
    return math.dist(left, right) / scale


class HandsGate:
    """On while the two hands touch. Hold to type, apart = off."""

    RELEASE_FACTOR = 1.5      # hands stay "touching" up to this many times touch_ratio
    GRACE = 0.2               # seconds a lost touch is tolerated

    def __init__(self, touch_ratio: float = 0.30):
        self.touch_ratio = touch_ratio
        self.touching = False
        self.ratio: Optional[float] = None
        self.reason = "hands not in view"
        self._lost_since: Optional[float] = None

    def reset(self) -> None:
        self.touching = False
        self.ratio = None
        self.reason = "hands not in view"
        self._lost_since = None

    def update(self, landmarks, width: int, height: int, now: float) -> bool:
        self.ratio = hands_distance(landmarks, width, height)
        if self.ratio is None:
            good = False
            self.reason = "hands not in view"
        else:
            limit = self.touch_ratio * (self.RELEASE_FACTOR if self.touching else 1.0)
            good = self.ratio <= limit
            self.reason = "hands touching" if good else "touch your hands together"

        if good:
            self.touching = True
            self._lost_since = None
        elif self.touching:
            if self._lost_since is None:
                self._lost_since = now
            if now - self._lost_since >= self.GRACE:
                self.touching = False
                self._lost_since = None
        return self.touching
