"""
Nostril flaring measured from the dark area inside two nostril ovals.

MediaPipe has no blendshape for nostril flare, but the nostril openings
are the darkest thing under the nose. When the nostrils flare, the dark
openings get bigger. Counting every dark pixel under the nose is noisy
(shadows at the edge of the nose, the lip line, ...), so this module only
counts dark pixels inside one oval per nostril.

How it works
------------
1. A small face-aligned patch is cut out under the nose tip, from one alar
   base (landmark 64) to the other (294). It is scaled by the nose width,
   so moving closer or further away does not change what it measures.
   All positions below are stored in "nose widths" relative to the
   subnasale (landmark 2), so they follow the face when it moves.
2. A pixel is "dark" when it is darker than DARK_RATIO x the brightness of
   the nose bridge skin (landmarks 5 and 195) in the same frame.
3. Calibration, phase 1 (C key, or the start): with the nose relaxed, the
   largest dark blob on each side of the nose is found every frame and an
   oval is fitted to it from its pixel spread (centre, two axes, angle).
   The median over the frames becomes the nostril oval, enlarged by
   OVAL_SCALE so a flared nostril still fits inside. If nothing is found
   a default oval at the usual nostril position is used.
4. Calibration, phase 2: ratio = dark pixels inside the oval / oval area,
   averaged over a short time -> the resting ratio r0 per side.
5. After that: flare = (ratio - r0) / r0, averaged over both sides,
   e.g. 0.4 = 40 % more dark area than at rest. r0 keeps following slow
   changes (light) while the nose is relaxed.

Tilting the head back shows more of the nostrils and would look like a
flare, so the head pitch from MediaPipe's transformation matrix is
recorded at calibration, and no flare is reported while the head is tilted
more than MAX_PITCH degrees away from it.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

ALAR_BASE = (64, 294)          # outer ends of the nostrils (image left, right)
NOSE_TIP_LOW, SUBNASALE = 1, 2
NOSE_WIDTH = (129, 358)        # outer edges of the nose wings
NOSE_BRIDGE_SKIN = (5, 195)    # skin used as the brightness reference
FACE_DOWN = (168, 152)         # nose bridge -> chin, gives the face's "down"

# Default ovals when calibration finds no nostril, in nose widths from the
# subnasale: (centre u, centre v, full length, full height, angle in degrees)
DEFAULT_OVALS = ((-0.27, -0.14, 0.22, 0.10, 0.0), (0.27, -0.14, 0.22, 0.10, 0.0))


@dataclass
class NostrilResult:
    ok: bool = False                     # nostrils could be measured
    calibrating: bool = False            # still learning ovals / resting level
    tilted: bool = False                 # head tilted too far, no flare reported
    ratio: tuple = (0.0, 0.0)            # dark area / oval area (image left, right)
    rest: tuple = (0.0, 0.0)             # resting ratio r0 per side
    flare: float = 0.0                   # mean (ratio - r0) / r0 of both sides
    flare_sides: tuple = (0.0, 0.0)
    pitch: float = 0.0                   # head pitch in degrees
    contours: list = field(default_factory=list)  # dark area inside ovals, normalised (x, y)
    ovals: list = field(default_factory=list)     # oval outlines, normalised (x, y)
    box: tuple = ()                      # kept for compatibility, unused


class NostrilDetector:
    PATCH_W = 96            # patch pixels across the nostril area
    TOP = 0.12              # patch starts this far above landmark 1 (nose widths)
    BOTTOM = 0.06           # ... and ends this far below landmark 2
    SIDE = 0.05             # extra width outside the alar bases
    DARK_RATIO = 0.65       # darker than 65 % of the nose skin = nostril
    OVAL_SCALE = 1.5        # calibrated oval is enlarged by this much
    MIN_BLOB = 8            # smallest dark blob (patch px) accepted as a nostril
    MIN_REST = 0.12         # floor for r0, so a nearly empty oval at rest
                            # cannot turn a small change into +2000 %
    MIN_VISIBLE = 0.06      # resting dark fraction below this = nostrils not
                            # seen during calibration -> calibrate again
    STUCK_TIME = 3.0        # "flared" this long without a break is not a
                            # real flare (a dash is well under a second):
                            # the resting level is re-learnt
    FIT_TIME = 1.2          # seconds to fit the ovals (calibration phase 1)
    REST_TIME = 0.8         # seconds to learn r0 (calibration phase 2)
    RISE_TIME = 6.0         # r0 follows a slowly growing resting level
    FALL_TIME = 0.8         # ... and a shrinking one faster
    LEARN_BELOW = 0.10      # only adapt r0 while flare is below this
    MAX_PITCH = 10.0        # degrees away from the calibrated pitch
    SMOOTH = 0.4            # flare smoothing (0 = none)

    def __init__(self):
        self.recalibrate()

    def recalibrate(self) -> None:
        """Fit the ovals and learn the resting level again (C key)."""
        self._cal_start: Optional[float] = None
        self._fits = ([], [])             # oval fits per side during phase 1
        self._rest_samples = []
        self._cal_pitch = []
        self.ovals = None                 # [(cu, cv, length, height, angle)] x 2
        self.rest = None                  # [left, right]
        self.pitch_ref = None
        self._t: Optional[float] = None
        self._flare = 0.0
        self._flared_since: Optional[float] = None

    def reset(self) -> None:
        """Face lost: keep the calibration, just forget the timing."""
        self._t = None
        self._flare = 0.0

    # ------------------------------------------------------------------

    def update(self, frame_rgb: np.ndarray, pts: np.ndarray, now: float,
               pitch: float = 0.0) -> NostrilResult:
        h, w = frame_rgb.shape[:2]
        geo = self._patch(frame_rgb, pts)
        if geo is None:
            return NostrilResult(pitch=pitch)

        dt = 0.0 if self._t is None else max(0.0, now - self._t)
        self._t = now
        if self._cal_start is None:
            self._cal_start = now

        # Calibration phase 1: fit an oval to each nostril.
        if self.ovals is None:
            for side, fit in enumerate(self._fit_blobs(geo)):
                if fit is not None:
                    self._fits[side].append(fit)
            self._cal_pitch.append(pitch)
            if now - self._cal_start >= self.FIT_TIME:
                self.ovals = [self._median_oval(self._fits[s], DEFAULT_OVALS[s])
                              for s in (0, 1)]
            return NostrilResult(ok=True, calibrating=True, pitch=pitch,
                                 ovals=self._oval_outlines(geo, DEFAULT_OVALS, w, h))

        ratios, contours = self._measure(geo)
        result = NostrilResult(
            ok=True, ratio=tuple(ratios), pitch=pitch,
            contours=[[(float(x / w), float(y / h)) for x, y in c] for c in contours],
            ovals=self._oval_outlines(geo, self.ovals, w, h))

        # Calibration phase 2: resting ratio inside the ovals.
        if self.rest is None:
            self._rest_samples.append(ratios)
            self._cal_pitch.append(pitch)
            if now - self._cal_start >= self.FIT_TIME + self.REST_TIME:
                rest = np.median(np.array(self._rest_samples), axis=0)
                if float(np.mean(rest)) < self.MIN_VISIBLE:
                    # Almost no dark area in the ovals: the nostrils were not
                    # visible (camera still adjusting its exposure, head
                    # turned, ...). Start the whole calibration again.
                    self.recalibrate()
                    result.calibrating = True
                    return result
                self.rest = [max(float(v), self.MIN_REST) for v in rest]
                self.pitch_ref = float(np.median(self._cal_pitch))
                self._flared_since = None
            result.calibrating = True
            return result

        # Head tilted too far: the nostrils look bigger, do not trust it.
        if abs(pitch - self.pitch_ref) > self.MAX_PITCH:
            self._flare = 0.0
            result.tilted = True
            result.rest = tuple(self.rest)
            return result

        sides = [(r - r0) / max(r0, self.MIN_REST) for r, r0 in zip(ratios, self.rest)]
        flare = float(np.mean(sides))
        self._flare = self._flare * self.SMOOTH + flare * (1 - self.SMOOTH)

        # Flared for far too long: the resting level is wrong (light changed,
        # bad calibration). Learn it again instead of staying stuck.
        if self._flare >= self.LEARN_BELOW:
            if self._flared_since is None:
                self._flared_since = now
            elif now - self._flared_since >= self.STUCK_TIME:
                self.rest = None
                self._rest_samples = []
                self._cal_start = now - self.FIT_TIME     # skip refitting the ovals
                self._flare = 0.0
                self._flared_since = None
                result.calibrating = True
                return result
        else:
            self._flared_since = None

        # Let the resting level follow slow changes, only while relaxed.
        if dt > 0 and self._flare < self.LEARN_BELOW:
            for i, r in enumerate(ratios):
                tau = self.FALL_TIME if r < self.rest[i] else self.RISE_TIME
                k = 1.0 - math.exp(-dt / tau)
                self.rest[i] = max(self.MIN_REST, self.rest[i] + (r - self.rest[i]) * k)
            self.pitch_ref += (pitch - self.pitch_ref) * (1.0 - math.exp(-dt / self.RISE_TIME))

        result.rest = tuple(self.rest)
        result.flare = self._flare
        result.flare_sides = tuple(float(s) for s in sides)
        return result

    # ------------------------------------------------------------------
    # Patch geometry
    # ------------------------------------------------------------------

    def _patch(self, frame_rgb, pts):
        """Face-aligned patch under the nose, its dark mask and the mapping back."""
        nose_w = float(np.linalg.norm(pts[NOSE_WIDTH[0]] - pts[NOSE_WIDTH[1]]))
        if nose_w < 12:
            return None

        down = pts[FACE_DOWN[1]] - pts[FACE_DOWN[0]]
        down /= max(np.linalg.norm(down), 1e-6)
        across = np.array([-down[1], down[0]])
        if np.dot(across, pts[ALAR_BASE[1]] - pts[ALAR_BASE[0]]) < 0:
            across = -across
        centre = pts[SUBNASALE]

        def proj(p):
            d = p - centre
            return float(np.dot(d, across)), float(np.dot(d, down))

        u0 = proj(pts[ALAR_BASE[0]])[0] - self.SIDE * nose_w
        u1 = proj(pts[ALAR_BASE[1]])[0] + self.SIDE * nose_w
        v0 = proj(pts[NOSE_TIP_LOW])[1] - self.TOP * nose_w
        v1 = self.BOTTOM * nose_w                     # subnasale is v = 0
        if u1 - u0 < 4 or v1 - v0 < 2:
            return None

        # Same scale on both axes, so areas are relative to nose width².
        s = self.PATCH_W / (u1 - u0)                  # patch px per image px
        pw, ph = self.PATCH_W, max(8, int(round((v1 - v0) * s)))
        origin = centre + across * u0 + down * v0
        to_img = np.array([[across[0] / s, down[0] / s, origin[0]],
                           [across[1] / s, down[1] / s, origin[1]]], dtype=np.float32)
        patch = cv2.warpAffine(frame_rgb, to_img, (pw, ph),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)

        light = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
        ref = self._skin_brightness(frame_rgb, pts, nose_w)
        dark = (light < self.DARK_RATIO * ref).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mid = min(max(int(round(-u0 * s)), 1), pw - 1)   # u = 0 is the nose middle
        return {"dark": dark, "s": s, "u0": u0, "v0": v0, "nose_w": nose_w,
                "origin": origin, "across": across, "down": down,
                "size": (pw, ph), "mid": mid}

    @staticmethod
    def _to_patch(geo, oval):
        """Oval in nose widths -> centre / half axes / angle in patch pixels."""
        cu, cv, length, height, angle = oval
        nw, s = geo["nose_w"], geo["s"]
        centre = ((cu * nw - geo["u0"]) * s, (cv * nw - geo["v0"]) * s)
        axes = (max(1.0, length * nw * s / 2), max(1.0, height * nw * s / 2))
        return centre, axes, angle

    @staticmethod
    def _to_image(geo, u, v):
        return geo["origin"] + geo["across"] * (u / geo["s"]) + geo["down"] * (v / geo["s"])

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------

    def _fit_blobs(self, geo):
        """Oval (in nose widths) around the largest dark blob on each side."""
        dark, mid = geo["dark"], geo["mid"]
        pw, _ = geo["size"]
        fits = []
        for a, b in ((0, mid), (mid, pw)):
            half = np.zeros_like(dark)
            half[:, a:b] = dark[:, a:b]
            n, labels, stats, _ = cv2.connectedComponentsWithStats(half, connectivity=8)
            if n < 2:
                fits.append(None)
                continue
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[k, cv2.CC_STAT_AREA] < self.MIN_BLOB:
                fits.append(None)
                continue
            vs, us = np.nonzero(labels == k)
            cu, cv = us.mean(), vs.mean()
            cov = np.cov(np.stack([us - cu, vs - cv]))
            evals, evecs = np.linalg.eigh(cov)
            major = evecs[:, 1]
            angle = math.degrees(math.atan2(major[1], major[0]))
            angle = (angle + 90.0) % 180.0 - 90.0        # -90..90
            # A filled ellipse spans about 4 standard deviations.
            length = 4.0 * math.sqrt(max(evals[1], 0.25))
            height = 4.0 * math.sqrt(max(evals[0], 0.25))
            nw, s = geo["nose_w"], geo["s"]
            fits.append(((cu / s + geo["u0"]) / nw, (cv / s + geo["v0"]) / nw,
                         length / s / nw, height / s / nw, angle))
        return fits

    def _median_oval(self, fits, default):
        if len(fits) < 3:
            return default
        cu, cv, length, height, angle = np.median(np.array(fits), axis=0)
        return (float(cu), float(cv), float(length * self.OVAL_SCALE),
                float(height * self.OVAL_SCALE), float(angle))

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def _measure(self, geo):
        """Dark fraction inside each oval, and the dark outlines for display."""
        dark = geo["dark"]
        pw, ph = geo["size"]
        ratios, contours = [], []
        for oval in self.ovals:
            centre, axes, angle = self._to_patch(geo, oval)
            mask = np.zeros((ph, pw), np.uint8)
            cv2.ellipse(mask, (round(centre[0]), round(centre[1])),
                        (round(axes[0]), round(axes[1])), angle, 0, 360, 1, -1)
            area = int(mask.sum())
            inside = dark & mask
            ratios.append(float(inside.sum()) / area if area else 0.0)
            cs, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                c = cv2.approxPolyDP(c, 0.8, True)[:, 0, :]
                contours.append([tuple(self._to_image(geo, u, v)) for u, v in c])
        return ratios, contours

    def _oval_outlines(self, geo, ovals, w, h):
        out = []
        for oval in ovals:
            centre, axes, angle = self._to_patch(geo, oval)
            poly = cv2.ellipse2Poly((round(centre[0] * 8), round(centre[1] * 8)),
                                    (round(axes[0] * 8), round(axes[1] * 8)),
                                    round(angle), 0, 360, 15)
            pts = [self._to_image(geo, u / 8.0, v / 8.0) for u, v in poly]
            out.append([(float(p[0] / w), float(p[1] / h)) for p in pts])
        return out

    @staticmethod
    def _skin_brightness(frame_rgb, pts, nose_w) -> float:
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
        return float(np.mean(values)) if values else 128.0


def pitch_from_matrix(matrix) -> float:
    """Head pitch (nodding up/down) in degrees from MediaPipe's 4x4 face matrix."""
    m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return float(math.degrees(math.atan2(m[2, 1], m[2, 2])))
