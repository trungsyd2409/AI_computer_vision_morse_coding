"""
Turns several weak face measurements into one "press" score with a short
guided calibration. Nothing here knows which body part is measured: a
subclass (see ear.py) lists its features and the word on the prompt.

Calibration
-----------
The user follows on-screen prompts: relax for 2 s, then three rounds of
the action (1.5 s) and relax (1.5 s). The first SKIP seconds of every step
are thrown away to allow for reaction time. From the remaining frames:

1. Every feature is standardised with its resting mean and noise:
   z = (x - mean_rest) / std_rest.
2. d = mean z during the action, i.e. how many noise units each feature moves.
3. The noise covariance C of z at rest has its correlations shrunk
   (features from the same face are correlated, and a short calibration
   overestimates that) and the weights are the Fisher linear
   discriminant: w = C^-1 d. Nuisance features (head angles) are first
   regressed out of the others, see fit_calibration.
4. The weights are scaled so that the average action scores 1.0 and rest
   scores 0.0. The separation d' = sqrt(d . C^-1 d) is shown as the
   calibration quality.

A feature that does not move for this person gets a weight close to zero,
so nothing has to be tuned by hand. If calibration fails (no action seen),
fixed fallback weights are used.

Running
-------
Each frame: score = sum(w * z), smoothed with a One Euro filter. The score
is forced to 0 while the head is turned away from the calibrated pose.
While the face is relaxed the resting means follow slow changes (light,
posture). An action held for STUCK_TIME seconds is not real typing, so the
resting means are re-centred on the current values.
"""

import csv
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .filters import OneEuroFilter


# ---------------------------------------------------------------------------
# Calibration maths
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    mean: np.ndarray          # resting value of each feature
    std: np.ndarray           # resting noise of each feature
    weights: np.ndarray       # applied to z; the average action scores 1.0
    quality: float            # d' between rest and action (0 for the fallback)
    effect: np.ndarray        # d: action - rest in noise units, per feature
    fallback: bool = False


def fit_calibration(rest: np.ndarray, active: Optional[np.ndarray],
                    min_std: np.ndarray, fallback_weights: np.ndarray,
                    enabled: Optional[np.ndarray] = None,
                    shrink: float = 0.3, min_quality: float = 0.8,
                    nuisance: Optional[np.ndarray] = None) -> Calibration:
    """
    rest, active:     arrays of shape (frames, n_features), NaN allowed
    min_std:          smallest resting noise per feature. Without a floor, a
                      feature that sat perfectly still during calibration
                      would get a huge weight from a tiny change later.
    fallback_weights: relative weights used when calibration fails
    enabled:          boolean mask; disabled features get weight 0
    nuisance:         boolean mask of features that must never count as
                      the action themselves (head angles, for example).
                      At rest, every other feature is regressed on them
                      (ridge, in z units) and only what they cannot
                      explain goes into the discriminant. Their weights
                      then subtract the explained part at run time.
    """
    rest = np.atleast_2d(np.asarray(rest, dtype=np.float64))
    n = rest.shape[1]
    if enabled is None:
        enabled = np.ones(n, dtype=bool)
    enabled = np.asarray(enabled, dtype=bool).copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN columns
        mean = np.nanmean(rest, axis=0)
        std = np.nanstd(rest, axis=0)
    # A feature that was never measured is unusable.
    missing = ~np.isfinite(mean)
    enabled &= ~missing
    mean = np.where(missing, 0.0, mean)
    std = np.maximum(np.where(np.isfinite(std), std, 0.0), min_std)

    def standardise(x):
        z = (x - mean) / std
        return np.where(np.isfinite(z), z, 0.0)

    fb = np.asarray(fallback_weights, dtype=np.float64) * enabled
    # Mean z of the fallback features (a negative weight means the feature
    # drops during the action), with a typical action at 3 sigma.
    fb = fb / (max(1.0, float(np.abs(fb).sum())) * 3.0)
    fallback = Calibration(mean, std, fb, 0.0, np.zeros(n), fallback=True)
    if active is None or len(active) < 5 or len(rest) < 10 or not enabled.any():
        return fallback

    zr, za = standardise(rest), standardise(np.asarray(active, dtype=np.float64))
    nuis = np.zeros(n, dtype=bool) if nuisance is None else np.asarray(nuisance, dtype=bool)
    idx = np.flatnonzero(enabled & ~nuis)            # features that may signal
    nidx = np.flatnonzero(enabled & nuis)            # features that only cancel
    if len(idx) == 0:
        return fallback

    # Remove from each signal feature the part that follows the nuisance
    # features at rest: B maps nuisance z -> expected signal z.
    B = np.zeros((len(nidx), len(idx)))
    if len(nidx):
        zn = zr[:, nidx]
        ridge = 0.05 * len(zr) * np.eye(len(nidx))
        try:
            B = np.linalg.solve(zn.T @ zn + ridge, zn.T @ zr[:, idx])
        except np.linalg.LinAlgError:
            B = np.zeros((len(nidx), len(idx)))
    res_r = zr[:, idx] - zr[:, nidx] @ B
    res_a = za[:, idx] - za[:, nidx] @ B

    d = np.zeros(n)
    d[idx] = res_a.mean(axis=0)
    cov = np.atleast_2d(np.cov(res_r, rowvar=False))
    # Shrink the correlations, keep each feature's own variance. After the
    # head angles are regressed out, a feature's variance can be far below
    # 1, and shrinking towards the identity would add noise it never had.
    var = np.maximum(np.diag(cov), 1e-6)
    cov = (1.0 - shrink) * cov + shrink * np.diag(var)
    try:
        w_sub = np.linalg.solve(cov, d[idx])
    except np.linalg.LinAlgError:
        return fallback
    separation = float(np.dot(d[idx], w_sub))       # d . C^-1 d
    if not math.isfinite(separation) or separation <= 0:
        return fallback
    quality = math.sqrt(separation)
    if quality < min_quality:
        fallback.effect = d
        return fallback

    weights = np.zeros(n)
    weights[idx] = w_sub / separation               # average action -> 1.0
    # score = w . (z_signal - z_nuisance B), so the nuisance weights are -B w.
    weights[nidx] = -B @ weights[idx]
    return Calibration(mean, std, weights, quality, d)


def quality_label(cal: Optional[Calibration]) -> str:
    if cal is None:
        return "not calibrated"
    if cal.fallback:
        return "default weights"
    if cal.quality >= 3.0:
        return "good"
    if cal.quality >= 1.5:
        return "weak"
    return "poor"


# ---------------------------------------------------------------------------
# Runtime signal
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    score: float = 0.0                 # smoothed, 0 = rest, 1 = typical action
    raw: float = 0.0                   # before smoothing and gating
    calibrating: bool = False
    step: Optional[str] = None         # "relax" or the action while calibrating
    step_index: int = 0
    step_count: int = 0
    step_progress: float = 0.0         # 0..1 inside the current step
    step_left: float = 0.0
    gated: Optional[str] = None        # why the input is ignored, e.g. "head"
    recentred: bool = False            # resting level was just re-learnt
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    contributions: np.ndarray = field(default_factory=lambda: np.zeros(0))
    events: list = field(default_factory=list)   # "relax", the action, "done"


class FusedSignal:
    FEATURES: tuple = ()               # feature names, set by the subclass
    MIN_STD: tuple = ()                # noise floor per feature
    FALLBACK: dict = {}                # feature name -> relative fallback weight
    ACTION = "active"                  # name of the action step in calibration
    REST_STEPS = ("relax",)            # calibration steps that count as rest
    NUISANCE: tuple = ()               # feature names used only to cancel noise
    SHRINK = 0.3                       # covariance shrinkage towards identity

    FIRST_REST = 2.0       # seconds of the first relax step
    ACTION_TIME = 1.5
    RELAX_TIME = 1.5
    ROUNDS = 3
    SKIP = 0.45            # reaction time ignored at the start of each step

    ADAPT_TIME = 6.0       # resting means follow slow changes with this time constant
    LEARN_BELOW = 0.2      # ... while the score is below this,
    CREEP_TIME = 20.0      # and much slower while it is between that and the
                           # threshold, so a small drift cannot sit there forever
    STUCK_TIME = 4.0       # an action this long means the resting level is wrong
    MAX_PITCH = 12.0       # degrees away from the calibrated head pose
    MAX_YAW = 20.0

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold         # only used for the stuck check
        self.n = len(self.FEATURES)
        self.calibration: Optional[Calibration] = None
        self._filter = OneEuroFilter(min_cutoff=2.5, beta=0.5, d_cutoff=1.0)
        self.pitch_ref = 0.0
        self.yaw_ref = 0.0
        self.start_calibration()

    # -- calibration flow -----------------------------------------------------

    @property
    def steps(self) -> list:
        steps = [("relax", self.FIRST_REST)]
        for _ in range(self.ROUNDS):
            steps += [(self.ACTION, self.ACTION_TIME), ("relax", self.RELAX_TIME)]
        return steps

    @property
    def calibrating(self) -> bool:
        return self._step is not None

    @property
    def step(self) -> Optional[str]:
        """Current calibration prompt, "relax" or the action, or None."""
        return None if self._step is None else self.steps[self._step][0]

    @property
    def enabled(self) -> np.ndarray:
        return np.ones(self.n, dtype=bool)

    def _fit(self, rest, active) -> Calibration:
        fallback = np.array([self.FALLBACK.get(name, 0.0) for name in self.FEATURES])
        nuisance = np.array([name in self.NUISANCE for name in self.FEATURES])
        return fit_calibration(rest, active, np.asarray(self.MIN_STD, dtype=np.float64),
                               fallback, self.enabled, shrink=self.SHRINK,
                               nuisance=nuisance)

    def start_calibration(self) -> None:
        """Begin the guided relax / action sequence (C key, or at start)."""
        self._step = 0
        self._step_time = 0.0
        self._rest, self._active, self._poses = [], [], []
        self._t: Optional[float] = None
        self._announced = -1
        self._stuck_since: Optional[float] = None
        self._filter.reset()

    def lost(self) -> None:
        """Face lost: keep the calibration, forget the timing."""
        self._t = None
        self._filter.reset()
        self._stuck_since = None

    # -- per frame ------------------------------------------------------------

    def update(self, features: np.ndarray, now: float, pitch: float = 0.0,
               yaw: float = 0.0, gate: Optional[str] = None) -> SignalResult:
        """`gate` lets the caller ignore the input for its own reasons."""
        dt = 0.0 if self._t is None else max(0.0, now - self._t)
        self._t = now
        features = np.asarray(features, dtype=np.float64)

        if self.calibrating:
            return self._calibration_frame(features, min(dt, 0.1), pitch, yaw)

        cal = self.calibration
        z = (features - cal.mean) / cal.std
        z = np.where(np.isfinite(z), z, 0.0)
        contributions = cal.weights * z
        raw = float(contributions.sum())

        gated = gate
        if gated is None and (abs(pitch - self.pitch_ref) > self.MAX_PITCH
                              or abs(yaw - self.yaw_ref) > self.MAX_YAW):
            gated = "head"

        value = 0.0 if gated else raw
        score = float(self._filter(np.array([value]), now)[0])
        score = min(max(score, 0.0), 2.0)
        result = SignalResult(score=score, raw=raw, gated=gated, z=z,
                              contributions=contributions)

        # Action held far too long: the resting level is off (camera moved,
        # light changed). Re-centre on what the face looks like right now.
        if score >= self.threshold and not gated:
            if self._stuck_since is None:
                self._stuck_since = now
            elif now - self._stuck_since >= self.STUCK_TIME:
                valid = np.isfinite(features)
                cal.mean[valid] = features[valid]
                self._filter.reset()
                self._stuck_since = None
                result.recentred = True
                result.score = 0.0
        else:
            self._stuck_since = None

        # Follow slow drift of the resting level while the face is relaxed.
        if dt > 0 and not gated and score < self.threshold * 0.9:
            tau = self.ADAPT_TIME if score < self.LEARN_BELOW else self.CREEP_TIME
            k = 1.0 - math.exp(-dt / tau)
            valid = np.isfinite(features)
            cal.mean[valid] += (features[valid] - cal.mean[valid]) * k
            self.pitch_ref += (pitch - self.pitch_ref) * k
            self.yaw_ref += (yaw - self.yaw_ref) * k
        return result

    def _calibration_frame(self, features, dt, pitch, yaw) -> SignalResult:
        steps = self.steps
        result = SignalResult(calibrating=True, step_count=len(steps),
                              z=np.zeros(self.n), contributions=np.zeros(self.n))
        if self._announced != self._step:
            self._announced = self._step
            result.events.append(steps[self._step][0])

        kind, length = steps[self._step]
        self._step_time += dt
        if self._step_time >= self.SKIP:
            if kind in self.REST_STEPS:
                self._rest.append(features)
                self._poses.append((pitch, yaw))
            else:
                self._active.append(features)

        if self._step_time >= length:
            self._step += 1
            self._step_time = 0.0
            if self._step >= len(steps):
                self._finish()
                result.events.append("done")
                result.calibrating = False
                return result
            kind, length = steps[self._step]
            self._announced = self._step
            result.events.append(kind)

        result.step = kind
        result.step_index = self._step
        result.step_progress = min(1.0, self._step_time / length)
        result.step_left = max(0.0, length - self._step_time)
        return result

    def _finish(self) -> None:
        self._step = None
        rest = np.array(self._rest) if self._rest else np.full((1, self.n), np.nan)
        active = np.array(self._active) if self._active else None
        self.calibration = self._fit(rest, active)
        if self._poses:
            poses = np.array(self._poses)
            self.pitch_ref = float(np.median(poses[:, 0]))
            self.yaw_ref = float(np.median(poses[:, 1]))
        self._filter.reset()


# ---------------------------------------------------------------------------
# Feature log for offline analysis (L key)
# ---------------------------------------------------------------------------

class FeatureLog:
    """
    Writes one CSV row per frame: every feature, its z-score, the fused
    score and a manual label (N key held = "I am doing the action now"),
    so the signals can be studied later in a notebook.
    """

    def __init__(self, folder: Path, prefix: str, feature_names: tuple):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / time.strftime(f"{prefix}_%Y%m%d_%H%M%S.csv")
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["t", "label", "score", "raw", "pressed", "gated", "calibrating",
             "step", "pitch", "yaw"]
            + list(feature_names) + [f"z_{n}" for n in feature_names])
        self._t0 = None

    def write(self, now: float, label: bool, result: SignalResult, pressed: bool,
              features: np.ndarray, pitch: float, yaw: float) -> None:
        if self._t0 is None:
            self._t0 = now

        def fmt(v):
            return "" if not np.isfinite(v) else f"{v:.5f}"

        self._writer.writerow(
            [f"{now - self._t0:.3f}", int(label), f"{result.score:.4f}",
             f"{result.raw:.4f}", int(pressed), result.gated or "",
             int(result.calibrating), result.step or "",
             f"{pitch:.2f}", f"{yaw:.2f}"]
            + [fmt(v) for v in features] + [fmt(v) for v in result.z])

    def close(self) -> None:
        self._file.close()
