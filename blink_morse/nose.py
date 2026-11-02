"""
Nostril flare from several weak signals, combined into one score.

No single measurement of a nostril flare is reliable from a webcam:

* the face mesh moves the nose wings only a little, close to its own jitter,
* MediaPipe's `noseSneer` blendshape reacts to wrinkling the nose, which
  often comes with a flare but not always,
* the dark area of the nostril openings depends on the light and on how
  much of the nostrils the camera can see.

So the app measures all of them (FEATURES below) and lets a short guided
calibration decide how much each one is worth for this face, this camera
and this light.

Calibration
-----------
The user follows on-screen prompts: relax for 2 s, then three rounds of
"flare" (1.5 s) and "relax" (1.5 s). The first SKIP seconds of every step
are thrown away to allow for reaction time. From the remaining frames:

1. Every feature is standardised with its resting mean and noise:
   z = (x - mean_rest) / std_rest.
2. d = mean z while flaring, i.e. how many noise units each feature moves.
3. The noise covariance C of z at rest is shrunk towards the identity
   (features from the same mesh are correlated) and the weights are the
   Fisher linear discriminant: w = C^-1 d.
4. The weights are scaled so that the average flare scores 1.0 and rest
   scores 0.0. The separation d' = sqrt(d . C^-1 d) is shown as the
   calibration quality.

A feature that does not move for this person gets a weight close to zero,
so nothing has to be tuned by hand. If calibration fails (no flare seen),
a fixed fallback (mean z of the geometry and sneer features / 3) is used.

Running
-------
Each frame: score = sum(w * z), smoothed with a One Euro filter. The score
is forced to 0 while the user smiles, opens the mouth or turns the head
away from the calibrated pose, because all of these widen the nose too.
While the nose is relaxed the resting means follow slow changes (light,
posture). A "flare" held for STUCK_TIME seconds is not real typing, so the
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
# Features
# ---------------------------------------------------------------------------

# Nose width measured between several landmark pairs, each divided by the
# distance between the inner eye corners so the camera distance cancels out.
WIDTH_PAIRS = {
    "wing": (129, 358),       # outer edge of the nose wings
    "alar": (48, 278),        # middle of the nose wings
    "base": (64, 294),        # alar base, where the wing meets the cheek
    "lower": (98, 327),       # lower nostril edge
}
EYE_INNER = (133, 362)
NOSE_TIP, SUBNASALE = 1, 2
NOSE_BRIDGE_SKIN = (5, 195)   # skin used as the brightness reference
FACE_DOWN = (168, 152)        # nose bridge -> chin, gives the face's "down"

FEATURES = ("wing", "alar", "base", "lower", "sneer", "dark")
GEOMETRY = ("wing", "alar", "base", "lower")
N_FEATURES = len(FEATURES)
DARK_INDEX = FEATURES.index("dark")

# Smallest resting noise per feature. Without a floor, a feature that
# happens to sit perfectly still during calibration (sneer at exactly 0)
# would get a huge weight from a tiny change later.
MIN_STD = np.array([0.002, 0.002, 0.002, 0.002, 0.01, 0.005])


def _dist(pts: np.ndarray, pair: tuple) -> float:
    a, b = pts[pair[0]], pts[pair[1]]
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def extract_features(pts: np.ndarray, blend: dict,
                     frame_rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """
    One value per name in FEATURES; NaN when it cannot be measured.

    pts:       landmarks in image pixels, shape (478, 2)
    blend:     blendshape name -> score
    frame_rgb: the camera frame, needed for "dark" only (None skips it)
    """
    out = np.full(N_FEATURES, np.nan)
    ref = _dist(pts, EYE_INNER)
    if ref < 8.0:                       # face too small to measure
        return out
    for i, name in enumerate(FEATURES):
        if name in WIDTH_PAIRS:
            out[i] = _dist(pts, WIDTH_PAIRS[name]) / ref
    out[FEATURES.index("sneer")] = 0.5 * (blend.get("noseSneerLeft", 0.0)
                                          + blend.get("noseSneerRight", 0.0))
    if frame_rgb is not None:
        out[DARK_INDEX] = dark_fraction(frame_rgb, pts)
    return out


def dark_fraction(frame_rgb: np.ndarray, pts: np.ndarray, patch_w: int = 64,
                  return_mask: bool = False):
    """
    How dark the two nostril openings are, as a weighted mean darkness.

    A face-aligned patch is cut out under the nose tip, scaled by the nose
    width so it covers the same part of the face at any distance. Each
    pixel gets a darkness value: 0 when it is as bright as the nose bridge
    skin in the same frame, 1 when it is black. The value is averaged with
    two soft round weights placed where the nostrils sit (halfway between
    the subnasale and each alar base, a little above the subnasale). When
    the nostrils flare, the dark openings grow into the weighted area.

    A continuous value is used instead of counting pixels under a fixed
    threshold, because a count jumps from frame to frame when the nostrils
    are close to the threshold.
    """
    import cv2

    nose_w = _dist(pts, WIDTH_PAIRS["wing"])
    if nose_w < 14.0:
        return float("nan")

    down = pts[FACE_DOWN[1]] - pts[FACE_DOWN[0]]
    norm = float(np.linalg.norm(down))
    if norm < 1e-6:
        return float("nan")
    down = down / norm
    across = np.array([-down[1], down[0]])
    base_l, base_r = pts[WIDTH_PAIRS["base"][0]], pts[WIDTH_PAIRS["base"][1]]
    if np.dot(across, base_r - base_l) < 0:
        across = -across
    centre = pts[SUBNASALE]

    def proj(p):
        d = p - centre
        return float(np.dot(d, across)), float(np.dot(d, down))

    ul, ur = sorted((proj(base_l)[0], proj(base_r)[0]))
    u0, u1 = ul - 0.05 * nose_w, ur + 0.05 * nose_w
    v0 = proj(pts[NOSE_TIP])[1] - 0.12 * nose_w
    v1 = 0.06 * nose_w
    if u1 - u0 < 4 or v1 - v0 < 2:
        return float("nan")

    s = patch_w / (u1 - u0)                    # patch px per image px
    ph = max(6, int(round((v1 - v0) * s)))
    origin = centre + across * u0 + down * v0
    to_img = np.array([[across[0] / s, down[0] / s, origin[0]],
                       [across[1] / s, down[1] / s, origin[1]]], dtype=np.float32)
    patch = cv2.warpAffine(frame_rgb, to_img, (patch_w, ph),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE)
    light = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)

    h, w = frame_rgb.shape[:2]
    r = max(2, int(nose_w * 0.06))
    values = []
    for i in NOSE_BRIDGE_SKIN:
        x, y = int(pts[i][0]), int(pts[i][1])
        x0, x1, y0, y1 = max(0, x - r), min(w, x + r), max(0, y - r), min(h, y + r)
        if x1 > x0 and y1 > y0:
            lab = cv2.cvtColor(np.ascontiguousarray(frame_rgb[y0:y1, x0:x1]),
                               cv2.COLOR_RGB2LAB)
            values.append(float(np.median(lab[:, :, 0])))
    if not values:
        return float("nan")
    ref = max(float(np.mean(values)), 1.0)
    darkness = np.clip((ref - light) / ref, 0.0, 1.0)

    # Soft weights centred on the usual nostril positions, in patch pixels.
    gu = (np.arange(patch_w) + 0.5) / s + u0          # back to image px, across
    gv = (np.arange(ph) + 0.5) / s + v0               # ... and down
    uu, vv = np.meshgrid(gu, gv)
    weight = np.zeros_like(darkness)
    for u_base in (ul, ur):
        cu, cv_ = 0.55 * u_base, -0.08 * nose_w
        weight += np.exp(-0.5 * (((uu - cu) / (0.13 * nose_w)) ** 2
                                 + ((vv - cv_) / (0.07 * nose_w)) ** 2))
    value = float((darkness * weight).sum() / max(weight.sum(), 1e-6))
    if return_mask:
        return value, patch, darkness, weight
    return value


# ---------------------------------------------------------------------------
# Calibration maths
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    mean: np.ndarray          # resting value of each feature
    std: np.ndarray           # resting noise of each feature
    weights: np.ndarray       # applied to z; the average flare scores 1.0
    quality: float            # d' between rest and flare (0 for the fallback)
    effect: np.ndarray        # d: flare - rest in noise units, per feature
    fallback: bool = False


def _fallback_weights(enabled: np.ndarray) -> np.ndarray:
    w = np.zeros(N_FEATURES)
    for name in GEOMETRY + ("sneer",):
        w[FEATURES.index(name)] = 1.0
    w *= enabled
    n = max(1.0, float(w.sum()))
    # Mean z of the used features, with a typical flare assumed at 3 sigma.
    return w / (n * 3.0)


def fit_calibration(rest: np.ndarray, flare: Optional[np.ndarray],
                    enabled: Optional[np.ndarray] = None,
                    shrink: float = 0.3, min_quality: float = 0.8) -> Calibration:
    """
    rest, flare: arrays of shape (frames, N_FEATURES), NaN allowed.
    enabled:     boolean mask; disabled features get weight 0.
    """
    if enabled is None:
        enabled = np.ones(N_FEATURES, dtype=bool)
    enabled = np.asarray(enabled, dtype=bool).copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN columns
        mean = np.nanmean(rest, axis=0)
        std = np.nanstd(rest, axis=0)
    # A feature that was never measured (no frame for "dark") is unusable.
    missing = ~np.isfinite(mean)
    enabled &= ~missing
    mean = np.where(missing, 0.0, mean)
    std = np.maximum(np.where(np.isfinite(std), std, 0.0), MIN_STD)

    def standardise(x):
        z = (x - mean) / std
        return np.where(np.isfinite(z), z, 0.0)

    fallback = Calibration(mean, std, _fallback_weights(enabled), 0.0,
                           np.zeros(N_FEATURES), fallback=True)
    if flare is None or len(flare) < 5 or len(rest) < 10 or not enabled.any():
        return fallback

    zr, zf = standardise(rest), standardise(flare)
    d = zf.mean(axis=0) * enabled
    idx = np.flatnonzero(enabled)
    cov = np.atleast_2d(np.cov(zr[:, idx], rowvar=False))
    cov = (1.0 - shrink) * cov + shrink * np.eye(len(idx))
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

    weights = np.zeros(N_FEATURES)
    weights[idx] = w_sub / separation               # average flare -> 1.0
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
class NoseResult:
    score: float = 0.0                 # smoothed flare, 0 = rest, 1 = typical flare
    raw: float = 0.0                   # before smoothing and gating
    calibrating: bool = False
    step: Optional[str] = None         # "relax" / "flare" while calibrating
    step_index: int = 0
    step_count: int = 0
    step_progress: float = 0.0         # 0..1 inside the current step
    step_left: float = 0.0
    gated: Optional[str] = None        # "smile", "mouth" or "head"
    recentred: bool = False            # resting level was just re-learnt
    z: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES))
    contributions: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES))
    events: list = field(default_factory=list)   # "relax", "flare", "done"


class NoseSignal:
    FIRST_REST = 2.0       # seconds of the first relax step
    FLARE_TIME = 1.5
    RELAX_TIME = 1.5
    ROUNDS = 3
    SKIP = 0.45            # reaction time ignored at the start of each step

    ADAPT_TIME = 6.0       # resting means follow slow changes with this time constant
    LEARN_BELOW = 0.2      # ... while the score is below this,
    CREEP_TIME = 20.0      # and much slower while it is between that and the
                           # threshold, so a small drift cannot sit there forever
    STUCK_TIME = 4.0       # a "flare" this long means the resting level is wrong
    MAX_PITCH = 12.0       # degrees away from the calibrated head pose
    MAX_YAW = 20.0
    MOUTH_GATE = 0.45      # jawOpen above this ignores the nose

    def __init__(self, smile_gate: float = 0.35, threshold: float = 0.5,
                 use_dark: bool = True):
        self.smile_gate = smile_gate
        self.threshold = threshold         # only used for the stuck check
        self.use_dark = use_dark
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
            steps += [("flare", self.FLARE_TIME), ("relax", self.RELAX_TIME)]
        return steps

    @property
    def calibrating(self) -> bool:
        return self._step is not None

    @property
    def step(self) -> Optional[str]:
        """Current calibration prompt, "relax" or "flare", or None."""
        return None if self._step is None else self.steps[self._step][0]

    @property
    def enabled(self) -> np.ndarray:
        mask = np.ones(N_FEATURES, dtype=bool)
        mask[DARK_INDEX] = self.use_dark
        return mask

    def start_calibration(self) -> None:
        """Begin the guided relax / flare sequence (C key, or at start)."""
        self._step = 0
        self._step_time = 0.0
        self._rest, self._flare, self._poses = [], [], []
        self._t: Optional[float] = None
        self._announced = -1
        self._stuck_since: Optional[float] = None
        self._filter.reset()

    def set_use_dark(self, value: bool) -> None:
        """Turn the dark-area feature on or off and refit from saved frames."""
        self.use_dark = bool(value)
        if self.calibration is not None and self._rest:
            self.calibration = fit_calibration(
                np.array(self._rest),
                np.array(self._flare) if self._flare else None, self.enabled)

    def lost(self) -> None:
        """Face lost: keep the calibration, forget the timing."""
        self._t = None
        self._filter.reset()
        self._stuck_since = None

    # -- per frame ------------------------------------------------------------

    def update(self, features: np.ndarray, now: float, smile: float = 0.0,
               jaw: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> NoseResult:
        dt = 0.0 if self._t is None else max(0.0, now - self._t)
        self._t = now
        features = np.asarray(features, dtype=np.float64)
        if not self.use_dark:
            features = features.copy()
            features[DARK_INDEX] = np.nan

        if self.calibrating:
            return self._calibration_frame(features, min(dt, 0.1), pitch, yaw)

        cal = self.calibration
        z = (features - cal.mean) / cal.std
        z = np.where(np.isfinite(z), z, 0.0)
        contributions = cal.weights * z
        raw = float(contributions.sum())

        gated = None
        if smile > self.smile_gate:
            gated = "smile"
        elif jaw > self.MOUTH_GATE:
            gated = "mouth"
        elif (abs(pitch - self.pitch_ref) > self.MAX_PITCH
              or abs(yaw - self.yaw_ref) > self.MAX_YAW):
            gated = "head"

        value = 0.0 if gated else raw
        score = float(self._filter(np.array([value]), now)[0])
        score = min(max(score, 0.0), 2.0)
        result = NoseResult(score=score, raw=raw, gated=gated, z=z,
                            contributions=contributions)

        # Held "flare" far too long: the resting level is off (camera moved,
        # light changed). Re-centre on what the nose looks like right now.
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

        # Follow slow drift of the resting level while the nose is relaxed.
        if dt > 0 and not gated and score < self.threshold * 0.9:
            tau = self.ADAPT_TIME if score < self.LEARN_BELOW else self.CREEP_TIME
            k = 1.0 - math.exp(-dt / tau)
            valid = np.isfinite(features)
            cal.mean[valid] += (features[valid] - cal.mean[valid]) * k
            self.pitch_ref += (pitch - self.pitch_ref) * k
            self.yaw_ref += (yaw - self.yaw_ref) * k
        return result

    def _calibration_frame(self, features, dt, pitch, yaw) -> NoseResult:
        steps = self.steps
        result = NoseResult(calibrating=True, step_count=len(steps))
        if self._announced != self._step:
            self._announced = self._step
            result.events.append(steps[self._step][0])

        kind, length = steps[self._step]
        self._step_time += dt
        if self._step_time >= self.SKIP:
            if kind == "relax":
                self._rest.append(features)
                self._poses.append((pitch, yaw))
            else:
                self._flare.append(features)

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
        rest = np.array(self._rest) if self._rest else np.full((1, N_FEATURES), np.nan)
        flare = np.array(self._flare) if self._flare else None
        self.calibration = fit_calibration(rest, flare, self.enabled)
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
    score and a manual label (N key held = "I am flaring"), so the signals
    can be studied later in a notebook.
    """

    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / time.strftime("nose_%Y%m%d_%H%M%S.csv")
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["t", "label", "score", "raw", "pressed", "gated", "calibrating",
             "step", "smile", "jaw", "pitch", "yaw"]
            + list(FEATURES) + [f"z_{n}" for n in FEATURES])
        self._t0 = None

    def write(self, now: float, label: bool, result: NoseResult, pressed: bool,
              features: np.ndarray, smile: float, jaw: float,
              pitch: float, yaw: float) -> None:
        if self._t0 is None:
            self._t0 = now

        def fmt(v):
            return "" if not np.isfinite(v) else f"{v:.5f}"

        self._writer.writerow(
            [f"{now - self._t0:.3f}", int(label), f"{result.score:.4f}",
             f"{result.raw:.4f}", int(pressed), result.gated or "",
             int(result.calibrating), result.step or "", f"{smile:.3f}",
             f"{jaw:.3f}", f"{pitch:.2f}", f"{yaw:.2f}"]
            + [fmt(v) for v in features] + [fmt(v) for v in result.z])

    def close(self) -> None:
        self._file.close()
