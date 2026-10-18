"""
Reads diagonal forearm poses as Morse keys.

Each arm is watched by one `TiltKey`. It looks at the forearm angle
(elbow -> wrist, measured on the image against the horizontal) and
reports a gesture when the forearm is held on a diagonal:

         +45  "up"     hand raised diagonally
    ----- 0 ------     forearm level (neutral)
         -45  "down"   hand lowered diagonally

Rules that keep it from typing by accident:

* The forearm has to be level before the first gesture, and has to come
  back to level between two gestures. Two dots in a row therefore means
  down, level, down.
* A pose has to be held for `hold_time` seconds, so sweeping the arm
  through 45 degrees on the way somewhere else does not type anything.
* Angles outside every zone (an arm hanging straight down is about -90)
  do nothing and do not count as level either.

What each gesture means is decided by the app:
    symbol arm  : up = dash, down = dot
    control arm : up = end of letter, down = space
"""

import math
from typing import Optional


def forearm_angle(elbow, wrist) -> float:
    """
    Forearm angle in degrees from image points (pixels, y pointing down).
    0 = level, positive = hand above the elbow. Works the same for the
    left and the right arm, whichever way the forearm points.
    """
    dx = float(wrist[0]) - float(elbow[0])
    dy = float(elbow[1]) - float(wrist[1])      # flip: up is positive
    return math.degrees(math.atan2(dy, abs(dx)))


class TiltKey:
    UP = "up"
    DOWN = "down"
    LEVEL = "level"

    def __init__(self, target: float = 45.0, tolerance: float = 20.0,
                 neutral: float = 15.0, hold_time: float = 0.10):
        self.target = target
        self.tolerance = tolerance
        self.neutral = neutral
        self.hold_time = hold_time

        # Public state for the HUD
        self.angle: Optional[float] = None
        self.zone: Optional[str] = None      # zone the forearm is in right now
        self.ready = False                   # level seen, next gesture can fire

        self._candidate: Optional[str] = None
        self._since = 0.0

    def reset(self) -> None:
        """Forget everything, e.g. when the arm leaves the frame or typing stops."""
        self.angle = None
        self.zone = None
        self.ready = False
        self._candidate = None

    def zone_of(self, angle: float) -> Optional[str]:
        if abs(angle) <= self.neutral:
            return self.LEVEL
        if abs(angle - self.target) <= self.tolerance:
            return self.UP
        if abs(angle + self.target) <= self.tolerance:
            return self.DOWN
        return None

    def update(self, angle: Optional[float], now: float) -> Optional[str]:
        """Feed one frame (None = arm not visible). Returns "up", "down" or None."""
        if angle is None:
            self.reset()
            return None
        self.angle = angle
        zone = self.zone_of(angle)
        self.zone = zone

        if zone != self._candidate:
            self._candidate = zone
            self._since = now
        if zone is None or now - self._since < self.hold_time:
            return None

        if zone == self.LEVEL:
            self.ready = True
            return None
        if self.ready:
            self.ready = False
            return zone
        return None
