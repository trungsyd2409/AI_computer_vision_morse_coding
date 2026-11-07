"""
Turns one "press" score per frame into Morse dots and dashes, the way a
telegraph key uses the length of each press.

The score comes from the hand (see hand.py): 0.5 at the loop angle, lower
while the fingers are straight. It counts as pressed above `threshold`
and as released below a slightly lower level, so noise around the
threshold cannot type a burst of symbols.

Timing, all in one time unit `t`:

* pressed for less than 1.5 t  -> DOT   (typed when the fingers straighten)
* pressed for 1.5 t or longer  -> DASH  (typed the moment 1.5 t is reached,
                                         if the score is still above the
                                         threshold at that moment)
* relaxed for 3 t / 7 t        -> end of letter / word (Morse composer)
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class PressKind(Enum):
    DOT = auto()
    DASH = auto()


@dataclass
class PressEvent:
    kind: PressKind
    started: float = 0.0   # when the press began
    fired: float = 0.0     # when the symbol was typed


class PressDetector:
    RELEASE_GAP = 0.15     # hysteresis between the press and release levels
    DASH_SPLIT = 1.5       # presses shorter than 1.5 t are dots
    # The score must stay below the release level this many frames in a
    # row before a press ends, so one noisy frame cannot split it in two.
    RELEASE_FRAMES = 2

    def __init__(self, threshold: float = 0.5, time_unit: float = 0.25):
        self.threshold = threshold
        self.time_unit = time_unit
        self.pressed = False
        self.pressed_since = 0.0
        self.open_since: Optional[float] = None
        self._dash_fired = False
        self._release_frames = 0
        self._release_start = 0.0

    @property
    def release_level(self) -> float:
        return max(self.threshold - self.RELEASE_GAP, self.threshold * 0.6)

    @property
    def dash_after(self) -> float:
        return self.DASH_SPLIT * self.time_unit

    def reset(self, now: Optional[float] = None) -> None:
        """Forget the current press without typing it, e.g. hand lost or recording off."""
        self.pressed = False
        self._dash_fired = False
        self._release_frames = 0
        if now is not None:
            self.open_since = now

    def pressed_time(self, now: float) -> float:
        return now - self.pressed_since if self.pressed else 0.0

    def pause_time(self, now: float) -> float:
        """Seconds since the last press ended (0 while pressed)."""
        if self.pressed or self.open_since is None:
            return 0.0
        return max(0.0, now - self.open_since)

    def update(self, score: float, now: float) -> list:
        """Feed one frame; returns the symbols it typed."""
        if self.open_since is None:
            self.open_since = now

        if not self.pressed:
            if score > self.threshold:
                self.pressed = True
                self.pressed_since = now
                self._dash_fired = False
                self._release_frames = 0
            return []

        if score >= self.release_level:
            self._release_frames = 0
            # The dash needs the score above the threshold itself, not just
            # above the release level: the tracker and the filter make the
            # score fade out slowly after a short loop, and that tail must
            # not stretch a dot into a dash.
            if (not self._dash_fired and score >= self.threshold
                    and now - self.pressed_since >= self.dash_after):
                self._dash_fired = True
                return [PressEvent(PressKind.DASH, self.pressed_since, now)]
            return []

        # Relaxing: the press ends at the first relaxed frame, once the
        # next frames confirm it.
        if self._release_frames == 0:
            self._release_start = now
        self._release_frames += 1
        if self._release_frames < self.RELEASE_FRAMES:
            return []

        self.pressed = False
        self.open_since = self._release_start
        if self._dash_fired:
            return []
        return [PressEvent(PressKind.DOT, self.pressed_since, self._release_start)]
