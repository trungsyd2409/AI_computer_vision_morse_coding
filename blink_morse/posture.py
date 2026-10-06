"""
Push-up posture check: push-ups only count while the whole body is in
view and lying horizontally (the plank / push-up position). Standing up
and bending the elbows, or only the upper body in frame, types nothing.

Rules
-----
* Whole body visible: on at least one side of the body the shoulder, hip,
  knee and ankle are all visible, plus at least one full arm
  (shoulder, elbow, wrist).
* Horizontal: the line from the middle of the shoulders to the middle of
  the ankles is within `MAX_TILT` degrees of horizontal, measured in
  pixels so the image aspect ratio does not bend it.

Hysteresis and a short grace period keep the check from flickering: the
posture becomes "ready" under `MAX_TILT` degrees and only stops being
ready above `RELEASE_TILT` degrees, or after it has looked wrong for
`GRACE` seconds in a row (one bad frame does not cancel a push-up).
"""

import math
from typing import Optional

# MediaPipe pose indices per side: (shoulder, hip, knee, ankle)
LEFT_BODY = (11, 23, 25, 27)
RIGHT_BODY = (12, 24, 26, 28)
LEFT_ARM = (11, 13, 15)
RIGHT_ARM = (12, 14, 16)
MIN_VIS = 0.5


def _vis(lm, i) -> bool:
    return lm[i][2] >= MIN_VIS


def body_tilt(landmarks, width: int, height: int) -> tuple:
    """
    landmarks: 33 (x, y, visibility) in normalised image coordinates.
    Returns (tilt in degrees or None, reason). tilt is None when the
    whole body is not visible; reason says what is missing.
    """
    if not landmarks or len(landmarks) < 33:
        return None, "no body"
    lm = landmarks
    sides = [s for s in (LEFT_BODY, RIGHT_BODY) if all(_vis(lm, i) for i in s)]
    if not sides:
        return None, "whole body not in view"
    if not any(all(_vis(lm, i) for i in arm) for arm in (LEFT_ARM, RIGHT_ARM)):
        return None, "arms not in view"

    def mid(index):           # index into (shoulder, hip, knee, ankle)
        pts = [lm[s[index]] for s in sides]
        return (sum(p[0] for p in pts) / len(pts) * width,
                sum(p[1] for p in pts) / len(pts) * height)

    (sx, sy), (ax, ay) = mid(0), mid(3)
    dx, dy = abs(ax - sx), abs(ay - sy)
    if dx < 1e-6 and dy < 1e-6:
        return None, "whole body not in view"
    return math.degrees(math.atan2(dy, dx)), ""


class PostureGate:
    MAX_TILT = 35.0       # degrees from horizontal to become ready
    RELEASE_TILT = 45.0   # degrees from horizontal to stop being ready
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
        self.tilt, reason = body_tilt(landmarks, width, height)
        if self.tilt is None:
            good = False
        elif self.ready:
            good = self.tilt <= self.RELEASE_TILT
        else:
            good = self.tilt <= self.MAX_TILT
        if not good and not reason:
            reason = f"body not horizontal ({round(self.tilt)}°)"
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
