"""
Turns 21 hand landmarks into Morse presses: the tips of the index and
middle fingers touching each other work like a telegraph key.

How a touch is detected
-----------------------
We measure the distance between the index fingertip and the middle
fingertip, then divide it by the palm length (wrist to the base of the
middle finger). Dividing by the palm length makes the value independent of
how far the hand is from the camera.

A touch starts when this ratio drops below `touch_ratio` for a couple of
consecutive frames, and ends only when it rises above
`touch_ratio + release_gap`. Using two thresholds instead of one
(hysteresis) prevents a noisy frame from ending and restarting a touch.

Timing
------
Everything is expressed in one time unit `t`:

* fingers together for less than 1.5 t  -> DOT   (typed when they separate)
* fingers together for 1.5 t or longer  -> DASH  (typed the moment 1.5 t
                                                  is reached, still together)
* fingers apart for 3 t                 -> end of letter  } handled by the
* fingers apart for 7 t                 -> end of word    } Morse composer

The dash does not wait for the fingers to separate, so it lands as soon as
it can be told apart from a dot. A dot can only be known once they separate.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

# MediaPipe landmark indices
WRIST = 0
THUMB_TIP = 4
MIDDLE_MCP = 9
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_ORDER = ("index", "middle", "ring", "pinky")
KEY_FINGERS = ("index", "middle")     # the two fingers that form the key

DASH_SPLIT = 1.5      # presses shorter than 1.5 t are dots
LETTER_GAP = 3.0      # fingers apart this many t -> end of letter
WORD_GAP = 7.0        # fingers apart this many t -> space


class PressKind(Enum):
    DOWN = auto()      # fingertips just touched
    DOT = auto()       # short press, typed when the fingers separate
    DASH = auto()      # long press, typed as soon as it reaches 1.5 t
    UP = auto()        # fingertips separated


@dataclass
class PressEvent:
    kind: PressKind
    point: tuple            # contact point in screen pixels
    duration: float = 0.0   # seconds the fingers have been together


class FingerKey:
    """Stateful detector, feed it one frame of landmarks at a time."""

    CONFIRM_FRAMES = 2     # frames below the threshold before a touch counts
    RELEASE_FRAMES = 2     # frames above the release threshold before it ends

    def __init__(self, touch_ratio: float = 0.22, release_gap: float = 0.10,
                 time_unit: float = 0.20):
        self.touch_ratio = touch_ratio
        self.release_gap = release_gap
        self.time_unit = time_unit

        self.touching = False
        self.touch_since = 0.0
        self.dash_fired = False
        self.open_since: Optional[float] = None   # when the fingers last separated
        self.contact_point = (0.0, 0.0)
        self.ratio = 1.0                 # current tip distance / palm length
        self.closeness = 0.0             # 0 = far apart, 1 = touching

        self._below = 0
        self._above = 0

    @property
    def dash_after(self) -> float:
        return DASH_SPLIT * self.time_unit

    @property
    def letter_gap(self) -> float:
        return LETTER_GAP * self.time_unit

    @property
    def word_gap(self) -> float:
        return WORD_GAP * self.time_unit

    def press_time(self, now: float) -> float:
        """How long the fingers have been together (0 while apart)."""
        return now - self.touch_since if self.touching else 0.0

    def pause_time(self, now: float) -> float:
        """Seconds since the fingers separated (0 while together)."""
        if self.touching or self.open_since is None:
            return 0.0
        return max(0.0, now - self.open_since)

    def reset(self, now: Optional[float] = None) -> list:
        """
        The hand left the frame. A press in progress ends without typing a
        dot (we cannot know how it ended); the pause starts counting now.
        """
        events = []
        if self.touching and now is not None:
            events.append(PressEvent(PressKind.UP, self.contact_point,
                                     now - self.touch_since))
            self.open_since = now
        self.touching = False
        self.dash_fired = False
        self._below = self._above = 0
        self.closeness = 0.0
        self.ratio = 1.0
        return events

    def update(self, points: np.ndarray, now: float) -> list:
        """
        points: (21, 3) array in screen pixels (x, y) plus a depth value z
                expressed in the same pixel scale.
        now:    current time in seconds.
        Returns a list of PressEvent produced by this frame.
        """
        if self.open_since is None:
            self.open_since = now
        palm = float(np.linalg.norm(points[MIDDLE_MCP, :2] - points[WRIST, :2]))
        if palm < 1e-3:
            return []

        a = points[FINGER_TIPS["index"]]
        b = points[FINGER_TIPS["middle"]]
        delta = b - a
        # Depth from a single camera is noisy, so it only gets half weight.
        dist = float(np.sqrt(delta[0] ** 2 + delta[1] ** 2 + (0.5 * delta[2]) ** 2))
        self.ratio = dist / palm
        # Map "fingers spread" (about 0.6 palm) .. "touching" to 0 .. 1
        span = max(0.6 - self.touch_ratio, 1e-3)
        self.closeness = float(np.clip((0.6 - self.ratio) / span, 0.0, 1.0))
        self.contact_point = (float((a[0] + b[0]) * 0.5), float((a[1] + b[1]) * 0.5))

        if self.touching:
            return self._update_touching(now)
        return self._update_apart(now)

    # -- state handlers ------------------------------------------------------

    def _update_apart(self, now: float) -> list:
        if self.ratio < self.touch_ratio:
            self._below += 1
        else:
            self._below = 0
        if self._below < self.CONFIRM_FRAMES:
            return []
        self._below = 0
        self.touching = True
        self.touch_since = now
        self.dash_fired = False
        return [PressEvent(PressKind.DOWN, self.contact_point)]

    def _update_touching(self, now: float) -> list:
        events = []
        duration = now - self.touch_since
        if not self.dash_fired and duration >= self.dash_after:
            self.dash_fired = True
            events.append(PressEvent(PressKind.DASH, self.contact_point, duration))

        if self.ratio > self.touch_ratio + self.release_gap:
            self._above += 1
        else:
            self._above = 0
        if self._above < self.RELEASE_FRAMES:
            return events

        self._above = 0
        self.touching = False
        self.open_since = now
        if not self.dash_fired:
            events.append(PressEvent(PressKind.DOT, self.contact_point, duration))
        events.append(PressEvent(PressKind.UP, self.contact_point, duration))
        self.dash_fired = False
        return events
