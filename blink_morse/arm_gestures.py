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

`TiltKey` is used for the control arm (up = end of letter, down = space).
The dot / dash arm uses `ZoneKey` below, which reacts instantly.
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


def x_axis_angle(elbow, wrist) -> float:
    """
    Angle of the forearm vector (elbow -> wrist) against the screen's X axis,
    in degrees from -180 to 180, counter-clockwise positive (y flipped so up
    is positive). Pointing right = 0, up = 90, left = 180, down = -90.
    """
    dx = float(wrist[0]) - float(elbow[0])
    dy = float(elbow[1]) - float(wrist[1])
    return math.degrees(math.atan2(dy, dx))


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


class ZoneKey:
    """
    Instant key for the dot / dash arm, driven by the X-axis angle.

        dash zone: 35 .. 145 deg   (forearm pointing up, either way)
        dot zone : -135 .. -35 deg (forearm pointing down, either way)

    * The first frame inside a zone types the symbol straight away, with no
      hold time, so fast movements are caught.
    * The key is then "latched" and types nothing more until the forearm
      leaves that zone. Leaving it (by more than `margin` degrees, so tracking
      noise at the edge does not type twice) makes the key ready again.
    * If the forearm is already inside a zone when the key starts (arm just
      came into view, or the fist just closed), it is latched without typing.
    """

    UP = "up"
    DOWN = "down"

    def __init__(self, dash=(35.0, 145.0), dot=(-135.0, -35.0), margin: float = 5.0):
        self.dash = tuple(dash)
        self.dot = tuple(dot)
        self.margin = margin

        # Public state for the HUD
        self.angle: Optional[float] = None
        self.zone: Optional[str] = None
        self.latched: Optional[str] = None
        self._primed = False

    @property
    def ready(self) -> bool:
        return self._primed and self.latched is None

    def reset(self) -> None:
        self.angle = None
        self.zone = None
        self.latched = None
        self._primed = False

    def _range(self, zone: str):
        return self.dash if zone == self.UP else self.dot

    def zone_of(self, angle: float, margin: float = 0.0) -> Optional[str]:
        for zone in (self.UP, self.DOWN):
            lo, hi = self._range(zone)
            if lo - margin <= angle <= hi + margin:
                return zone
        return None

    def update(self, angle: Optional[float], now: float = 0.0) -> Optional[str]:
        """Feed one frame (None = arm not visible). Returns "up", "down" or None."""
        if angle is None:
            self.reset()
            return None
        self.angle = angle

        if self.latched is not None:
            lo, hi = self._range(self.latched)
            if lo - self.margin <= angle <= hi + self.margin:
                self.zone = self.latched
                return None
            self.latched = None            # left the zone: ready again

        zone = self.zone_of(angle)
        self.zone = zone
        if zone is None:
            self._primed = True
            return None
        self.latched = zone
        if not self._primed:
            # Was already in the zone when tracking started: do not type.
            self._primed = True
            return None
        return zone
