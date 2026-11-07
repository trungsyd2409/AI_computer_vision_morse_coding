"""
Index + middle finger gestures from MediaPipe's 21 hand landmarks.

Two poses, both with the index and middle fingers raised and touching at
the tips, and the ring and pinky fingers folded:

* READY: both fingers straight, side by side. Holding it for ARM_TIME
  switches recording on. Recording stays on while the hand keeps one of
  the two poses.
* LOOP: the fingers bend so that their tips meet above an open gap. This
  is the Morse key: a short loop is a dot, a long loop a dash (press.py),
  and going back to READY releases it.

Any other hand shape (open hand, fist, V sign) or no hand switches
recording off, and a loop made while recording is off types nothing.

Measurements use the world landmarks (metres, centred on the hand), so
they do not change with the distance to the camera or the hand's angle:

* reach: wrist-to-fingertip distance divided by the wrist-to-middle-knuckle
  distance (landmarks 0 and 9). A raised finger reaches far, a folded
  finger stays close to the palm.
* bend: how far a finger bends at its two middle joints, in degrees
  (0 = perfectly straight).
* curl = bend of the index + bend of the middle finger.
* tip gap: distance between the index and middle fingertips divided by
  the palm width (landmarks 5 and 17).

On the two reference photos, fed through the tracker like camera frames,
the curl was 66 to 74 degrees for READY and 95 to 99 degrees for LOOP. The
tip gap was 0.41 to 0.52 in both, the reach about 1.7 to 1.9 for the
raised fingers and about 0.9 for the folded ones. The default loop angle
(85 degrees) sits between the two curls, and the release level (6 degrees
lower) stays above the straight pose.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17
FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
INDEX_TIP, MIDDLE_TIP = 8, 12
INDEX_PIP, MIDDLE_PIP = 6, 10


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b between b->a and b->c, in degrees."""
    u, v = a - b, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 180.0
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))


@dataclass
class HandMeasure:
    reach: dict            # finger -> reach
    bend: dict             # finger -> bend in degrees
    curl: float            # bend(index) + bend(middle)
    tip_gap: float         # index-middle tip distance / palm width
    pip_gap: float         # index-middle middle-joint distance / palm width


def measure(world: np.ndarray) -> Optional[HandMeasure]:
    """Everything the gesture rules need, from (21, 3) world landmarks."""
    palm_len = float(np.linalg.norm(world[MIDDLE_MCP] - world[WRIST]))
    palm_w = float(np.linalg.norm(world[PINKY_MCP] - world[INDEX_MCP]))
    if palm_len < 1e-6 or palm_w < 1e-6:
        return None
    reach, bend = {}, {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        reach[name] = float(np.linalg.norm(world[tip] - world[WRIST])) / palm_len
        bend[name] = ((180.0 - _angle(world[mcp], world[pip], world[dip]))
                      + (180.0 - _angle(world[pip], world[dip], world[tip])))
    return HandMeasure(
        reach=reach, bend=bend,
        curl=bend["index"] + bend["middle"],
        tip_gap=float(np.linalg.norm(world[INDEX_TIP] - world[MIDDLE_TIP])) / palm_w,
        pip_gap=float(np.linalg.norm(world[INDEX_PIP] - world[MIDDLE_PIP])) / palm_w,
    )


@dataclass
class GestureResult:
    state: str = "none"          # "none" (no hand), "off", "ready" or "loop"
    armed: bool = False          # recording is on
    holding: bool = False        # pose lost for a moment: keep the press as it is
    score: float = 0.0           # for the press detector, 0.5 at the loop angle
    measure: Optional[HandMeasure] = None
    checks: dict = field(default_factory=dict)   # rule name -> passed
    events: list = field(default_factory=list)   # "armed", "disarmed"


class GestureDetector:
    RAISED = 1.3       # reach above this: finger raised
    FOLDED = 1.25      # reach below this: finger folded
    TOGETHER = 0.6     # tip gap below this: the fingertips touch...
    APART = 0.75       # ... and they count as apart again only above this
    ARM_TIME = 0.15    # READY held this long switches recording on
    GRACE = 0.25       # the pose may be lost this long before recording stops
    SCORE_SPAN = 40.0  # degrees of curl between score 0.0 and 1.0, so the
                       # press detector's release gap (0.15) is 6 degrees
    READY_SCORE = 0.35  # below this score the fingers count as straight

    def __init__(self, loop_angle: float = 85.0, need_folded: bool = True):
        self.loop_angle = loop_angle
        self.need_folded = need_folded
        self.armed = False
        self._together = False
        self._ready_since: Optional[float] = None
        self._lost_since: Optional[float] = None
        self._last_score = 0.0

    def score_for(self, curl: float) -> float:
        """Curl as a press score: 0.5 exactly at the loop angle."""
        return float(np.clip(0.5 + (curl - self.loop_angle) / self.SCORE_SPAN, 0.0, 1.5))

    def checks(self, m: HandMeasure, together_before: bool = False) -> dict:
        limit = self.APART if together_before else self.TOGETHER
        out = {
            "index up": m.reach["index"] > self.RAISED,
            "middle up": m.reach["middle"] > self.RAISED,
            "tips together": m.tip_gap < limit,
        }
        if self.need_folded:
            out["ring folded"] = m.reach["ring"] < self.FOLDED
            out["pinky folded"] = m.reach["pinky"] < self.FOLDED
        return out

    def pose_ok(self, world: np.ndarray) -> bool:
        """Stateless check, used to pick the right hand when two are visible."""
        m = measure(world)
        return m is not None and all(self.checks(m).values())

    def reset(self) -> None:
        self.armed = False
        self._together = False
        self._ready_since = self._lost_since = None
        self._last_score = 0.0

    def update(self, world: Optional[np.ndarray], now: float) -> GestureResult:
        result = GestureResult()
        m = measure(world) if world is not None else None
        result.measure = m
        ok = False
        if m is not None:
            result.checks = self.checks(m, self._together)
            self._together = result.checks["tips together"]
            ok = all(result.checks.values())
        else:
            self._together = False

        score = self.score_for(m.curl) if m is not None else 0.0
        if ok:
            self._lost_since = None
            self._last_score = score
            if not self.armed:
                if score < self.READY_SCORE:
                    if self._ready_since is None:
                        self._ready_since = now
                    if now - self._ready_since >= self.ARM_TIME:
                        self.armed = True
                        result.events.append("armed")
                else:
                    # A loop without READY first does not switch recording on.
                    self._ready_since = None
        else:
            self._ready_since = None
            if self.armed:
                if self._lost_since is None:
                    self._lost_since = now
                if now - self._lost_since >= self.GRACE:
                    self.armed = False
                    self._lost_since = None
                    result.events.append("disarmed")

        result.armed = self.armed
        if self.armed:
            # During the short grace period the caller should leave the
            # press detector alone, so one bad frame neither releases a
            # loop nor lets it grow into a dash without a hand.
            result.holding = not ok
            result.score = score if ok else self._last_score
        if m is None:
            result.state = "none"
        elif not ok:
            result.state = "off"
        else:
            result.state = "loop" if score >= 0.5 else "ready"
        return result


class GestureLog:
    """
    Writes one CSV row per frame with every measurement and a manual label
    (N key held = "I am making the loop now"), for later analysis.
    """

    def __init__(self, folder):
        import csv
        import time

        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / time.strftime("hand_%Y%m%d_%H%M%S.csv")
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        fingers = list(FINGERS)
        self._writer.writerow(
            ["t", "label", "state", "armed", "pressed", "score", "curl", "tip_gap",
             "pip_gap"] + [f"reach_{f}" for f in fingers] + [f"bend_{f}" for f in fingers])
        self._t0 = None

    def write(self, now: float, label: bool, result: GestureResult, pressed: bool) -> None:
        if self._t0 is None:
            self._t0 = now
        m = result.measure
        row = [f"{now - self._t0:.3f}", int(label), result.state, int(result.armed),
               int(pressed), f"{result.score:.3f}"]
        if m is None:
            row += [""] * (3 + 2 * len(FINGERS))
        else:
            row += [f"{m.curl:.1f}", f"{m.tip_gap:.3f}", f"{m.pip_gap:.3f}"]
            row += [f"{m.reach[f]:.3f}" for f in FINGERS]
            row += [f"{m.bend[f]:.1f}" for f in FINGERS]
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()
