"""
Sit-up posture check and measurement.

Sit-ups only count while the whole body is in view and lying on the floor
(side-on to the camera). Standing, sitting on a chair, or only the upper
body in frame types nothing.

Rules
-----
* Whole body visible: on at least one side of the body the shoulder, hip,
  knee and ankle are all visible.
* Lying on the floor: the line from the hip to the ankle (the legs, which
  stay on the floor during a sit-up) is within `MAX_TILT` degrees of
  horizontal. Pixels are used so the image aspect ratio does not bend it.
  The torso is not part of this test, because it rises during the sit-up.

Measurement
-----------
`torso_elevation` is the angle of the hip -> shoulder line above the
floor: about 0-15 deg lying flat, 60-90 deg sitting up.

Hysteresis and a short grace period keep the check from flickering: the
posture becomes "ready" under `MAX_TILT` degrees and only stops being
ready above `RELEASE_TILT` degrees, or after it has looked wrong for
`GRACE` seconds in a row (one bad frame does not cancel a sit-up).
"""

import math
from typing import Optional

# MediaPipe pose indices per side: (shoulder, hip, knee, ankle)
LEFT_BODY = (11, 23, 25, 27)
RIGHT_BODY = (12, 24, 26, 28)
MIN_VIS = 0.5

LYING_ANGLE = 15.0        # torso elevation that counts as fully down
UP_ANGLE = 60.0           # torso elevation that counts as fully up


def _vis(lm, i) -> bool:
    return lm[i][2] >= MIN_VIS


def _sides(landmarks) -> list:
    if not landmarks or len(landmarks) < 33:
        return []
    return [s for s in (LEFT_BODY, RIGHT_BODY) if all(_vis(landmarks, i) for i in s)]


def _mid(landmarks, sides, index, width, height):
    """Average pixel position of one joint (0 shoulder .. 3 ankle) over the visible sides."""
    pts = [landmarks[s[index]] for s in sides]
    return (sum(p[0] for p in pts) / len(pts) * width,
            sum(p[1] for p in pts) / len(pts) * height)


def legs_tilt(landmarks, width: int, height: int) -> tuple:
    """
    landmarks: 33 (x, y, visibility) in normalised image coordinates.
    Returns (tilt of the hip -> ankle line in degrees, or None, reason).
    """
    if not landmarks:
        return None, "no body"
    sides = _sides(landmarks)
    if not sides:
        return None, "whole body not in view"
    (hx, hy), (ax, ay) = _mid(landmarks, sides, 1, width, height), \
        _mid(landmarks, sides, 3, width, height)
    dx, dy = abs(ax - hx), abs(ay - hy)
    if dx < 1e-6 and dy < 1e-6:
        return None, "whole body not in view"
    return math.degrees(math.atan2(dy, dx)), ""


def torso_elevation(landmarks, width: int, height: int) -> Optional[float]:
    """Angle of the hip -> shoulder line above horizontal, in degrees (0..90)."""
    sides = _sides(landmarks)
    if not sides:
        return None
    (sx, sy), (hx, hy) = _mid(landmarks, sides, 0, width, height), \
        _mid(landmarks, sides, 1, width, height)
    rise = hy - sy                      # image y grows downwards
    run = abs(sx - hx)
    if rise <= 0:
        return 0.0                      # shoulders at or below the hips: lying
    return math.degrees(math.atan2(rise, max(run, 1e-6)))


def situp_height(elevation: float) -> float:
    """Torso elevation -> 0 (lying flat) .. 1 (sat up)."""
    k = (elevation - LYING_ANGLE) / (UP_ANGLE - LYING_ANGLE)
    return min(1.0, max(0.0, k))


class PostureGate:
    MAX_TILT = 35.0       # legs within this many degrees of horizontal = lying
    RELEASE_TILT = 45.0   # stop being ready above this
    GRACE = 0.3           # seconds a bad posture is tolerated while ready

    def __init__(self):
        self.ready = False
        self.tilt: Optional[float] = None
        self.reason = "no body"
        self._bad_since: Optional[float] = None

    def reset(self) -> None:
        self.ready = False
        self.tilt = None
        self.reason = "no body"
        self._bad_since = None

    def update(self, landmarks, width: int, height: int, now: float) -> bool:
        self.tilt, reason = legs_tilt(landmarks, width, height)
        if self.tilt is None:
            good = False
        elif self.ready:
            good = self.tilt <= self.RELEASE_TILT
        else:
            good = self.tilt <= self.MAX_TILT
        if not good and not reason:
            reason = f"not lying down ({round(self.tilt)}°)"
        self.reason = reason

        if good:
            self.ready = True
            self._bad_since = None
        elif self.ready:
            if self._bad_since is None:
                self._bad_since = now
            if now - self._bad_since >= self.GRACE:
                self.ready = False
                self._bad_since = None
        return self.ready
