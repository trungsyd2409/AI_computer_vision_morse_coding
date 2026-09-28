"""
Turns two eye-closure scores per frame into Morse symbols, as fast as the
camera allows.

Signals
-------
MediaPipe gives each eye a blink score between 0 (open) and 1 (closed).
The resting value is not zero for everyone: people with narrower eyes, or
who look down at a laptop screen, sit around 0.2-0.4 with their eyes open.
`EyeSignal` learns that resting level and rescales the score so that 0
means "this person's open eye" again.

Episodes
--------
Everything between the moment any eye closes and the moment both eyes are
open again is one "episode", and each episode types at most one symbol:

* Both eyes closed  -> DASH, on the very first frame they are both shut
                       (optionally after `blink_filter` seconds).
* Right eye closed, left eye clearly open for `wink_confirm` seconds
                    -> DOT.

The short confirmation for the dot is needed because a normal blink does
not close both eyes on exactly the same frame. For a frame or two one eye
can look closed on its own. Waiting ~50 ms (two to three frames) is enough
to tell "one eye leading a blink" from "a real wink", and is too short to
feel like a delay. A left wink does nothing.

The detector does not decide when a letter or word ends. It only reports
how long the eyes have been open (`pause_time`), and the Morse composer
turns long enough pauses into "end of letter" and "space".
"""

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class BlinkKind(Enum):
    DOT = auto()
    DASH = auto()


@dataclass
class BlinkEvent:
    kind: BlinkKind
    started: float = 0.0  # when the eye(s) closed
    fired: float = 0.0    # when the symbol was typed


class EyeSignal:
    """Removes a person's resting eye-closure level from the raw score."""

    MAX_BASELINE = 0.40   # never treat a mostly closed eye as "resting"
    RISE_TIME = 2.0       # seconds for the baseline to follow a slow rise
    FALL_TIME = 0.25      # it drops much faster, the open level is a floor

    def __init__(self, initial: float = 0.10):
        self.baseline = initial
        self.value = 0.0          # normalised closure, 0 = open, 1 = shut
        self._t: Optional[float] = None

    def reset(self) -> None:
        self._t = None

    def update(self, raw: float, now: float, is_closed: bool) -> float:
        dt = 0.0 if self._t is None else max(0.0, now - self._t)
        self._t = now

        # Only learn from frames where the eye is open. The baseline follows
        # the lower envelope of the signal: quick to drop, slow to rise.
        if not is_closed and raw < self.baseline + 0.30:
            tau = self.FALL_TIME if raw < self.baseline else self.RISE_TIME
            k = 1.0 - math.exp(-dt / tau) if dt > 0 else 0.0
            self.baseline += (raw - self.baseline) * k
            self.baseline = min(max(self.baseline, 0.0), self.MAX_BASELINE)

        span = max(1.0 - self.baseline, 0.2)
        self.value = min(max((raw - self.baseline) / span, 0.0), 1.0)
        return self.value


class BlinkDetector:
    # When both eyes are above the threshold but one is much more closed,
    # the user is winking and the other eye is only squinting along with it.
    WINK_DIFFERENCE = 0.28
    RELEASE_GAP = 0.15    # hysteresis between the close and open thresholds
    # For a dot the left eye must be below this fraction of the threshold,
    # so a left eye that is already on its way down is not mistaken for open.
    OPEN_FRACTION = 0.6
    # A wink with a squinting left eye still counts once it lasts this many
    # times the normal confirmation time.
    SQUINT_FACTOR = 3.0
    # Both eyes must look open on this many frames in a row before an
    # episode ends, so one noisy frame cannot split a blink into two.
    OPEN_FRAMES = 2

    def __init__(self, close_threshold: float = 0.45, wink_confirm: float = 0.05,
                 blink_filter: float = 0.0):
        self.close_threshold = close_threshold
        self.wink_confirm = wink_confirm
        self.blink_filter = blink_filter

        self.left = EyeSignal()
        self.right = EyeSignal()
        self.left_closed = False
        self.right_closed = False

        # Public state for the HUD
        self.pose = "open"               # open / left / right / both
        self.in_episode = False
        self.open_since: Optional[float] = None

        self._reset_episode()

    def reset(self) -> None:
        """Forget everything, e.g. when the face leaves the frame."""
        self.left.reset()
        self.right.reset()
        self.left_closed = self.right_closed = False
        self.pose = "open"
        self._reset_episode()

    def pause_time(self, now: float) -> float:
        """Seconds since both eyes opened after the last blink (0 while closed)."""
        if self.in_episode or self.open_since is None:
            return 0.0
        return max(0.0, now - self.open_since)

    def _reset_episode(self) -> None:
        self.in_episode = False
        self._consumed = False           # this episode already typed a symbol
        self._run_pose = "open"
        self._run_start = 0.0
        self._run_frames = 0
        self._open_frames = 0
        self._open_start = 0.0

    # ------------------------------------------------------------------------

    def update(self, left_raw: float, right_raw: float, now: float) -> list:
        """Feed one frame of raw blink scores; returns the symbols it typed."""
        nl = self.left.update(left_raw, now, self.left_closed)
        nr = self.right.update(right_raw, now, self.right_closed)
        self.left_closed = self._hysteresis(nl, self.left_closed)
        self.right_closed = self._hysteresis(nr, self.right_closed)
        pose = self._classify(nl, nr)
        self.pose = pose

        if self.open_since is None:
            self.open_since = now

        if not self.in_episode:
            if pose == "open":
                return []
            self.in_episode = True
            self._run_pose = None        # forces a new run below

        if pose == "open":
            return self._update_open(now)
        self._open_frames = 0

        if pose != self._run_pose:
            self._run_pose = pose
            self._run_start = now
            self._run_frames = 0
        self._run_frames += 1

        if self._consumed:
            return []
        return self._decide(pose, nl, now)

    # ------------------------------------------------------------------------

    def _decide(self, pose: str, nl: float, now: float) -> list:
        run = now - self._run_start

        if pose == "both":
            if run >= self.blink_filter:
                self._consumed = True
                return [BlinkEvent(BlinkKind.DASH, self._run_start, now)]
            return []

        confirmed = run >= self.wink_confirm and self._run_frames >= 2
        if pose == "right" and confirmed:
            left_open = nl < self.close_threshold * self.OPEN_FRACTION
            long_enough = run >= self.wink_confirm * self.SQUINT_FACTOR
            if left_open or long_enough:
                self._consumed = True
                return [BlinkEvent(BlinkKind.DOT, self._run_start, now)]
        elif pose == "left" and confirmed:
            # A left wink types nothing, but it must not turn into a dash if
            # the right eye closes a moment later.
            self._consumed = True
        return []

    def _update_open(self, now: float) -> list:
        if self._open_frames == 0:
            self._open_start = now
        self._open_frames += 1
        if self._open_frames >= self.OPEN_FRAMES:
            self.open_since = self._open_start
            self._reset_episode()
        return []

    def _hysteresis(self, value: float, was_closed: bool) -> bool:
        if was_closed:
            return value > self.close_threshold - self.RELEASE_GAP
        return value > self.close_threshold

    def _classify(self, nl: float, nr: float) -> str:
        if self.left_closed and self.right_closed:
            if nl - nr > self.WINK_DIFFERENCE:
                return "left"
            if nr - nl > self.WINK_DIFFERENCE:
                return "right"
            return "both"
        if self.left_closed:
            return "left"
        if self.right_closed:
            return "right"
        return "open"
