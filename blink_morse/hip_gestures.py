"""
Reads hip sway as Morse input.

The hip angle is the tilt of the line through the two hips, measured on the
image in degrees:

        0 = hips level
      > 0 = the user's right hip is higher
      < 0 = the user's left hip is higher

With one threshold A (`trigger`) there are two states:

    waiting   -A < angle < A
    recorded  a sway was typed, the hips have not come back yet

* In waiting, a timer runs. Longer than t it fires "end" (end of letter),
  longer than 3t it fires "space". Each fires once per waiting period.
* In waiting, the first frame with angle >= A fires "right" and angle <= -A
  fires "left" (what they type is up to the app). No hold time, no delay.
  The key is then "recorded".
* From recorded, nothing fires until the hips are back in waiting. Leaving
  recorded needs `margin` degrees below A, so noise at the edge cannot type
  twice.
"""

import math
from typing import Optional


def hip_tilt(left_hip, right_hip, min_width: float = 1.0) -> Optional[float]:
    """
    Tilt of the hip line in degrees from two image points (pixels, y down).
    Positive when the user's right hip is higher on the image. Works with
    or without a mirrored image, because only the horizontal distance is
    used, not its sign. Returns None when the hips are on top of each other
    (person turned sideways), where the angle would only be noise.
    """
    dx = abs(float(right_hip[0]) - float(left_hip[0]))
    if dx < min_width:
        return None
    dy = float(left_hip[1]) - float(right_hip[1])      # up is positive
    return math.degrees(math.atan2(dy, dx))


class HipKey:
    RIGHT = "right"
    LEFT = "left"
    WAITING = "waiting"

    def __init__(self, trigger: float = 5.0, margin: float = 1.0,
                 unit: float = 0.6, smooth: float = 1.0):
        self.trigger = trigger        # A, degrees
        self.margin = margin          # how far below A the hips must come back
        self.unit = unit              # t, seconds
        self.smooth = smooth          # weight of the newest frame, 1 = no smoothing

        # Public state for the HUD
        self.angle: Optional[float] = None       # angle the logic sees
        self.zone: Optional[str] = None          # RIGHT, LEFT (recorded) or WAITING
        self.latched: Optional[str] = None       # sway typed, waiting for the return
        self.wait_for = 0.0                      # seconds spent in waiting

        self._smoothed: Optional[float] = None
        self._primed = False
        self._waiting = False
        self._since = 0.0
        self._end_sent = False
        self._space_sent = False

    # ------------------------------------------------------------------------

    def _forget_input(self) -> None:
        """Drop latch and timers, keep the smoothing."""
        self.latched = None
        self.wait_for = 0.0
        self._primed = False
        self._waiting = False
        self._end_sent = self._space_sent = False

    def reset(self) -> None:
        self._forget_input()
        self.angle = None
        self.zone = None
        self._smoothed = None

    def update(self, angle: Optional[float], now: float, active: bool = True) -> list:
        """
        Feed one frame (None = hips not visible). Returns the events of this
        frame, in order: any of "right", "left", "end", "space".

        With active=False the angle and zone are still tracked for the HUD,
        but nothing fires and nothing is remembered, so a pose held while
        typing is switched on does not type.
        """
        if angle is None:
            self.reset()
            return []

        if self._smoothed is None or self.smooth >= 1.0:
            self._smoothed = angle
        else:
            self._smoothed += self.smooth * (angle - self._smoothed)
        a = self._smoothed
        mag = abs(a)
        side = self.RIGHT if a > 0 else self.LEFT
        self.angle = a

        if not active:
            self._forget_input()
            self.zone = side if mag >= self.trigger else self.WAITING
            return []

        events = []
        if self.latched is not None:
            if mag < self.trigger - self.margin:
                self.latched = None                      # back in waiting
            elif mag >= self.trigger and side != self.latched:
                self.latched = None                      # swung across in one go

        if self.latched is None and mag >= self.trigger:
            self.latched = side
            self._waiting = False
            if self._primed:                             # not when typing just started
                events.append(side)

        if self.latched is None:
            if not self._waiting:
                self._waiting = True
                self._since = now
                self._end_sent = self._space_sent = False
            self.wait_for = max(0.0, now - self._since)
            if self.wait_for >= self.unit and not self._end_sent:
                self._end_sent = True
                events.append("end")
            if self.wait_for >= 3.0 * self.unit and not self._space_sent:
                self._space_sent = True
                events.append("space")
            self.zone = self.WAITING
        else:
            self.wait_for = 0.0
            self.zone = self.latched
        self._primed = True
        return events
