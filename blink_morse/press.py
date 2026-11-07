"""
Turns one "press" score per frame into Morse dots and dashes, the way a
telegraph key uses the length of each press.

The score comes from the ears (see ear.py and fusion.py): 0 = relaxed,
1 = a typical wiggle from calibration. It counts as pressed above
`threshold` and as released below a slightly lower level, so noise around
the threshold cannot type a burst of symbols.

Timing, all in one time unit `t`:

* pressed for less than 1.5 t  -> DOT   (typed when the ears relax)
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
        """Forget the current press, e.g. face lost or calibration running."""
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
            # score fade out slowly after a short pull, and that tail must
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


class TwitchDetector:
    """
    For a body part that can only twitch, not hold a position (ears, for
    most people). The same hysteresis as PressDetector finds each twitch;
    what counts is how many come in a row:

    * one twitch, then nothing for 2 t   -> DOT  (typed after the 2 t wait)
    * a second twitch within 2 t         -> DASH (typed at once)
    * relaxed for 3 t / 7 t after the symbol -> end of letter / word

    It has the same interface as PressDetector, so the app can switch.
    """

    WINDOW = 2.0           # time units to wait for a second twitch

    def __init__(self, threshold: float = 0.5, time_unit: float = 0.25):
        self._press = PressDetector(threshold, time_unit)
        self._pending: Optional[float] = None    # release time of a lone twitch
        self._pending_start = 0.0
        self.open_since: Optional[float] = None  # when the last symbol was typed

    # Settings are kept in the inner detector.
    @property
    def threshold(self) -> float:
        return self._press.threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._press.threshold = value

    @property
    def time_unit(self) -> float:
        return self._press.time_unit

    @time_unit.setter
    def time_unit(self, value: float) -> None:
        self._press.time_unit = value

    @property
    def window(self) -> float:
        return self.WINDOW * self.time_unit

    @property
    def pressed(self) -> bool:
        return self._press.pressed

    @property
    def waiting(self) -> bool:
        """One twitch seen, waiting to know whether a second one follows."""
        return self._pending is not None

    def pressed_time(self, now: float) -> float:
        return self._press.pressed_time(now)

    def pause_time(self, now: float) -> float:
        """Seconds since the last symbol was typed (0 while a twitch is open)."""
        if self.pressed or self.waiting or self.open_since is None:
            return 0.0
        return max(0.0, now - self.open_since)

    def reset(self, now: Optional[float] = None) -> None:
        self._press.reset(now)
        self._pending = None
        if now is not None:
            self.open_since = now

    def update(self, score: float, now: float) -> list:
        if self.open_since is None:
            self.open_since = now
        events = []
        was_pressed = self._press.pressed
        started = self._press.pressed_since
        self._press.update(score, now)
        # Long holds are not dashes here; only the release matters.
        self._press._dash_fired = True
        if was_pressed and not self._press.pressed:
            if self._pending is None:
                self._pending, self._pending_start = now, started
            else:
                events.append(PressEvent(PressKind.DASH, self._pending_start, now))
                self._pending = None
                self.open_since = now
        if self._pending is not None and not self._press.pressed \
                and now - self._pending >= self.window:
            events.append(PressEvent(PressKind.DOT, self._pending_start, now))
            self._pending = None
            self.open_since = now
        return events
