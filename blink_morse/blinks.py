"""
Turns two scores per frame into Morse dots and dashes, using the length of
each "press", the same way a telegraph key uses the length of each press.

The app now feeds the frown score (see frown.py) as both scores, so a
"blink" in this module means "frowning past the threshold" and "open"
means "face relaxed". The timing logic is the same as for blinks.

Signals
-------
MediaPipe gives each eye a blink score between 0 (open) and 1 (closed).
The resting value is not zero for everyone: people with narrower eyes, or
who look down at a laptop screen, sit around 0.2-0.4 with their eyes open.
`EyeSignal` learns that resting level and rescales the score so that 0
means "this person's open eye" again.

Timing
------
Everything is expressed in one time unit `t` (0.1 s by default):

* both eyes closed for less than 1.5 t  -> DOT   (typed when they open)
* both eyes closed for 1.5 t or longer  -> DASH  (typed the moment 1.5 t
                                                  is reached, eyes still shut)
* eyes open for 3 t                     -> end of letter  } handled by the
* eyes open for 7 t                     -> end of word    } Morse composer

The dash does not wait for the eyes to open, so it lands as soon as it can
be told apart from a dot. A dot can only be known once the eyes open.
Winks (one eye only) are ignored.
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
    started: float = 0.0  # when both eyes closed
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
    RELEASE_GAP = 0.15    # hysteresis between the close and open thresholds
    DASH_SPLIT = 1.5      # closures shorter than 1.5 t are dots
    # Both eyes must look open on this many frames in a row before a blink
    # counts as finished, so one noisy frame cannot split it in two.
    OPEN_FRAMES = 2

    def __init__(self, close_threshold: float = 0.45, time_unit: float = 0.1):
        self.close_threshold = close_threshold
        self.time_unit = time_unit

        self.left = EyeSignal()
        self.right = EyeSignal()
        self.left_closed = False
        self.right_closed = False

        # Public state for the HUD
        self.pose = "open"               # open / left / right / both
        self.closed = False              # inside a both-eye blink
        self.closed_since = 0.0
        self.open_since: Optional[float] = None

        self._dash_fired = False
        self._open_frames = 0
        self._open_start = 0.0

    @property
    def dash_after(self) -> float:
        """Seconds of closure after which a blink becomes a dash."""
        return self.DASH_SPLIT * self.time_unit

    def reset(self, now: Optional[float] = None) -> None:
        """Forget everything, e.g. when the face is lost or input is switched off."""
        self.left.reset()
        self.right.reset()
        self.left_closed = self.right_closed = False
        self.pose = "open"
        self.closed = False
        self._dash_fired = False
        self._open_frames = 0
        if now is not None:
            self.open_since = now

    def closed_time(self, now: float) -> float:
        """How long both eyes have been shut in the current blink (0 if open)."""
        return now - self.closed_since if self.closed else 0.0

    def pause_time(self, now: float) -> float:
        """Seconds since the eyes opened after the last blink (0 while shut)."""
        if self.closed or self.open_since is None:
            return 0.0
        return max(0.0, now - self.open_since)

    # ------------------------------------------------------------------------

    def update(self, left_raw: float, right_raw: float, now: float) -> list:
        """Feed one frame of raw blink scores; returns the symbols it typed."""
        nl = self.left.update(left_raw, now, self.left_closed)
        nr = self.right.update(right_raw, now, self.right_closed)
        self.left_closed = self._hysteresis(nl, self.left_closed)
        self.right_closed = self._hysteresis(nr, self.right_closed)
        self.pose = self._classify()
        if self.open_since is None:
            self.open_since = now

        both = self.pose == "both"
        if not self.closed:
            if both:
                self.closed = True
                self.closed_since = now
                self._dash_fired = False
                self._open_frames = 0
            return []

        # Inside a blink.
        if both:
            self._open_frames = 0
            if not self._dash_fired and now - self.closed_since >= self.dash_after:
                self._dash_fired = True
                return [BlinkEvent(BlinkKind.DASH, self.closed_since, now)]
            return []

        # The eyes are opening. Remember the first open frame, and only end
        # the blink once they have stayed open for a couple of frames.
        if self._open_frames == 0:
            self._open_start = now
        self._open_frames += 1
        if self._open_frames < self.OPEN_FRAMES:
            return []

        self.closed = False
        self.open_since = self._open_start
        if self._dash_fired:
            return []
        return [BlinkEvent(BlinkKind.DOT, self.closed_since, self._open_start)]

    # ------------------------------------------------------------------------

    def _hysteresis(self, value: float, was_closed: bool) -> bool:
        if was_closed:
            # With a low threshold a fixed gap would put the release level
            # at or below zero and a press could never end, so the gap is
            # capped at half the threshold.
            gap = min(self.RELEASE_GAP, self.close_threshold * 0.5)
            return value > self.close_threshold - gap
        return value > self.close_threshold

    def _classify(self) -> str:
        if self.left_closed and self.right_closed:
            return "both"
        if self.left_closed:
            return "left"
        if self.right_closed:
            return "right"
        return "open"
