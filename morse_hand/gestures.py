"""
Turns 21 hand landmarks into discrete "thumb touched finger X" events.

How a touch is detected
-----------------------
For each of the four fingers we measure the distance between the thumb tip
and that fingertip, then divide it by the palm length (wrist to the base of
the middle finger). Dividing by the palm length makes the value independent
of how far the hand is from the camera.

A touch starts when this ratio drops below `touch_ratio` for a couple of
consecutive frames, and ends only when it rises above
`touch_ratio + release_gap`. Using two thresholds instead of one
(hysteresis) prevents a noisy frame from ending and restarting a touch.

Only one finger can be "down" at a time. When several fingers are close to
the thumb, the closest one wins.
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


class TouchPhase(Enum):
    DOWN = auto()      # thumb just touched the finger
    HOLD = auto()      # contact kept for longer than the hold time (once)
    UP = auto()        # contact released


@dataclass
class TouchEvent:
    phase: TouchPhase
    finger: str
    point: tuple          # contact point in screen pixels
    duration: float = 0.0  # seconds the contact lasted (UP / HOLD only)


class PinchDetector:
    """Stateful detector, feed it one frame of landmarks at a time."""

    CONFIRM_FRAMES = 2     # frames below the threshold before a touch counts
    RELEASE_FRAMES = 2     # frames above the release threshold before it ends
    REFRACTORY = 0.08      # seconds after a release where no new touch starts

    def __init__(self, touch_ratio: float = 0.30, release_gap: float = 0.12,
                 hold_time: float = 1.0):
        self.touch_ratio = touch_ratio
        self.release_gap = release_gap
        self.hold_time = hold_time

        self.active: Optional[str] = None      # finger currently touching
        self.active_since = 0.0
        self.hold_fired = False
        self.contact_point = (0.0, 0.0)

        # Per-finger closeness in [0, 1]; 1 means touching. Drives visuals.
        self.closeness = {f: 0.0 for f in FINGER_ORDER}

        self._candidate: Optional[str] = None
        self._candidate_frames = 0
        self._release_frames = 0
        self._last_release = -1.0

    def reset(self) -> None:
        """Forget the current touch, e.g. when the hand leaves the frame."""
        self.active = None
        self.hold_fired = False
        self._candidate = None
        self._candidate_frames = 0
        self._release_frames = 0
        self.closeness = {f: 0.0 for f in FINGER_ORDER}

    def update(self, points: np.ndarray, now: float) -> list:
        """
        points: (21, 3) array in screen pixels (x, y) plus a depth value z
                expressed in the same pixel scale.
        now:    current time in seconds.
        Returns a list of TouchEvent produced by this frame.
        """
        events = []
        palm = float(np.linalg.norm(points[MIDDLE_MCP, :2] - points[WRIST, :2]))
        if palm < 1e-3:
            return events

        thumb = points[THUMB_TIP]
        ratios = {}
        for finger in FINGER_ORDER:
            tip = points[FINGER_TIPS[finger]]
            delta = tip - thumb
            # Depth from a single camera is noisy, so it only gets half weight.
            dist = float(np.sqrt(delta[0] ** 2 + delta[1] ** 2 + (0.5 * delta[2]) ** 2))
            ratios[finger] = dist / palm

        release_ratio = self.touch_ratio + self.release_gap
        for finger, ratio in ratios.items():
            # Map "fully open" (about 1.0) .. "touching" to 0 .. 1
            span = max(1.0 - self.touch_ratio, 1e-3)
            self.closeness[finger] = float(np.clip((1.0 - ratio) / span, 0.0, 1.0))

        if self.active is not None:
            events.extend(self._update_active(points, ratios[self.active],
                                              release_ratio, now))
        else:
            events.extend(self._update_idle(points, ratios, now))
        return events

    # -- state handlers ------------------------------------------------------

    def _update_active(self, points, ratio, release_ratio, now):
        events = []
        finger = self.active
        self.contact_point = self._midpoint(points, finger)

        if ratio > release_ratio:
            self._release_frames += 1
        else:
            self._release_frames = 0

        duration = now - self.active_since
        if not self.hold_fired and duration >= self.hold_time:
            self.hold_fired = True
            events.append(TouchEvent(TouchPhase.HOLD, finger,
                                     self.contact_point, duration))

        if self._release_frames >= self.RELEASE_FRAMES:
            events.append(TouchEvent(TouchPhase.UP, finger,
                                     self.contact_point, duration))
            self.active = None
            self.hold_fired = False
            self._release_frames = 0
            self._last_release = now
        return events

    def _update_idle(self, points, ratios, now):
        if now - self._last_release < self.REFRACTORY:
            return []

        closest = min(ratios, key=ratios.get)
        if ratios[closest] >= self.touch_ratio:
            self._candidate = None
            self._candidate_frames = 0
            return []

        # The same finger has to stay closest for a few frames in a row.
        if closest == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = closest
            self._candidate_frames = 1

        if self._candidate_frames < self.CONFIRM_FRAMES:
            return []

        self.active = closest
        self.active_since = now
        self.hold_fired = False
        self._candidate = None
        self._candidate_frames = 0
        self.contact_point = self._midpoint(points, closest)
        return [TouchEvent(TouchPhase.DOWN, closest, self.contact_point)]

    @staticmethod
    def _midpoint(points, finger):
        tip = points[FINGER_TIPS[finger]]
        thumb = points[THUMB_TIP]
        return (float((tip[0] + thumb[0]) * 0.5), float((tip[1] + thumb[1]) * 0.5))
