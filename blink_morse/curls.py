"""
Turns the curl of the arms into Morse dots and dashes, using how long each
curl is held, the same way a telegraph key uses the length of each press.

Signal
------
Each frame brings a curl score per arm (0 = arm straight, 1 = fully
curled, see `arm_tracker`), or None when that arm is out of view. Either
arm can be used; the more curled one drives the key. The arm counts as
"curled" (key down) above the threshold, and as "extended" (key up) once
it drops a little below it again (hysteresis, so a shaky arm near the
threshold does not chatter).

Timing
------
Everything is expressed in one time unit `t` (0.5 s by default):

* curled for less than 1.5 t  -> DOT   (typed when the arm extends)
* curled for 1.5 t or longer  -> DASH  (typed the moment 1.5 t is reached,
                                        arm still curled)
* arm extended for 5 t        -> end of letter  } handled by the
* arm extended for 7 t        -> end of word    } Morse composer
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class CurlKind(Enum):
    DOT = auto()
    DASH = auto()


@dataclass
class CurlEvent:
    kind: CurlKind
    started: float = 0.0  # when the arm curled past the threshold
    fired: float = 0.0    # when the symbol was typed


class CurlDetector:
    RELEASE_GAP = 0.10    # hysteresis between the curl and extend thresholds
    DASH_SPLIT = 1.5      # curls shorter than 1.5 t are dots
    # The arm must look extended on this many frames in a row before a curl
    # counts as finished, so one noisy frame cannot split it in two.
    RELEASE_FRAMES = 2

    def __init__(self, curl_threshold: float = 0.5, time_unit: float = 0.5):
        self.curl_threshold = curl_threshold
        self.time_unit = time_unit

        # Public state for the HUD
        self.curl = {"left": 0.0, "right": 0.0}
        self.visible = {"left": False, "right": False}
        self.active = None               # "left" / "right": arm driving the key
        self.pressed = False             # inside a curl
        self.pressed_since = 0.0
        self.open_since: Optional[float] = None

        self._dash_fired = False
        self._release_frames = 0
        self._release_start = 0.0
        self._need_release = False

    @property
    def dash_after(self) -> float:
        """Seconds of curl after which it becomes a dash."""
        return self.DASH_SPLIT * self.time_unit

    @property
    def release_threshold(self) -> float:
        # With very low thresholds (down to 5%) a fixed gap would put the
        # release level below zero and the curl would never end.
        return self.curl_threshold - min(self.RELEASE_GAP, self.curl_threshold * 0.5)

    def reset(self, now: Optional[float] = None) -> None:
        """Forget the current curl, e.g. when the body is lost or input is off."""
        self.pressed = False
        self.active = None
        self.visible = {"left": False, "right": False}
        self.curl = {"left": 0.0, "right": 0.0}
        self._dash_fired = False
        self._release_frames = 0
        if now is not None:
            self.open_since = now

    def cancel(self, now: float) -> None:
        """
        Drop the current curl without typing anything (the switch hand was
        opened). The arm must then go back down before the next curl
        counts, so re-closing the fist with the arm already up types nothing.
        """
        if self.pressed:
            self.open_since = now
        self.pressed = False
        self.active = None
        self._dash_fired = False
        self._release_frames = 0
        self._need_release = True

    def pressed_time(self, now: float) -> float:
        """How long the arm has been curled in the current curl (0 if extended)."""
        return now - self.pressed_since if self.pressed else 0.0

    def pause_time(self, now: float) -> float:
        """Seconds since the arm extended after the last curl (0 while curled)."""
        if self.pressed or self.open_since is None:
            return 0.0
        return max(0.0, now - self.open_since)

    # ------------------------------------------------------------------------

    def update(self, left: Optional[float], right: Optional[float], now: float) -> list:
        """Feed one frame of curl scores (None = arm not visible); returns symbols."""
        for side, value in (("left", left), ("right", right)):
            self.visible[side] = value is not None
            self.curl[side] = 0.0 if value is None else float(value)
        if self.open_since is None:
            self.open_since = now

        # While curling, stay with the arm that started it, so a stray
        # movement of the other arm cannot end or extend the curl.
        if self.pressed and self.active and self.visible[self.active]:
            side = self.active
        else:
            side = self._most_curled()
        value = self.curl[side] if side else 0.0

        if not self.pressed:
            if self._need_release:
                if side is not None and value <= self.release_threshold:
                    self._need_release = False
                return []
            self.active = side if value > self.curl_threshold else None
            if self.active:
                self.pressed = True
                self.pressed_since = now
                self._dash_fired = False
                self._release_frames = 0
            return []

        # Inside a curl.
        if value > self.release_threshold:
            self._release_frames = 0
            if not self._dash_fired and now - self.pressed_since >= self.dash_after:
                self._dash_fired = True
                return [CurlEvent(CurlKind.DASH, self.pressed_since, now)]
            return []

        # The arm is extending. Remember the first extended frame, and only
        # end the curl once it has stayed extended for a couple of frames.
        if self._release_frames == 0:
            self._release_start = now
        self._release_frames += 1
        if self._release_frames < self.RELEASE_FRAMES:
            return []

        self.pressed = False
        self.active = None
        self.open_since = self._release_start
        if self._dash_fired:
            return []
        return [CurlEvent(CurlKind.DOT, self.pressed_since, self._release_start)]

    def _most_curled(self) -> Optional[str]:
        sides = [s for s in ("right", "left") if self.visible[s]]
        if not sides:
            return None
        return max(sides, key=lambda s: self.curl[s])
