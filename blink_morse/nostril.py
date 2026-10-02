"""
Nostril flaring measured from the dark area of the nostrils.

MediaPipe has no blendshape for nostril flare, but the nostril openings
are the darkest thing under the nose. When the nostrils flare, the dark
openings get bigger. This module measures that area.

How it works
------------
1. A small face-aligned patch is cut out under the nose tip: from one alar
   base (landmark 64) to the other (294), and from just above landmark 1
   (lower nose tip) to just below landmark 2 (subnasale). The patch is
   scaled by the nose width, so moving closer or further away does not
   change the area it measures.
2. Every pixel darker than DARK_RATIO x the brightness of the nose bridge
   skin (landmarks 5 and 195) counts as "nostril". Comparing with the skin
   of the same frame keeps it working when the light changes.
3. The patch is split at the middle of the nose. For each side the area of
   the dark blobs is divided by the area of that half -> `ratio`.
4. Calibration: for the first CALIBRATE_TIME seconds (or after pressing C)
   the user keeps the nose relaxed and the average ratio becomes the
   resting level r0. After that r0 keeps following the resting level
   slowly, but only while the nostrils are not flared.
5. flare = (ratio - r0) / r0, e.g. 0.4 = nostrils 40 % bigger than at rest.
   The app compares the average of both sides with the threshold from the
   settings window.

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


@dataclass
class NostrilResult:
    ok: bool = False                     # nostrils could be measured
    calibrating: bool = False            # still learning the resting level
    tilted: bool = False                 # head tilted too far, no flare reported
    ratio: tuple = (0.0, 0.0)            # dark area / half patch (image left, right)
    rest: tuple = (0.0, 0.0)             # resting ratio r0 per side
    flare: float = 0.0                   # mean (ratio - r0) / r0 of both sides
    flare_sides: tuple = (0.0, 0.0)
    pitch: float = 0.0                   # head pitch in degrees
    contours: list = field(default_factory=list)  # outlines, normalised image (x, y)
    box: tuple = ()                      # patch corners, normalised image (x, y)


class NostrilDetector:
    PATCH_W = 96            # patch pixels across the nostril area
    TOP = 0.12              # patch starts this far above landmark 1 (nose widths)
    BOTTOM = 0.06           # ... and ends this far below landmark 2
    SIDE = 0.05             # extra width outside the alar bases
    DARK_RATIO = 0.65       # darker than 65 % of the nose skin = nostril
    MIN_BLOB = 6            # ignore dark specks smaller than this (patch px)
    MIN_REST = 0.02         # floor for r0 so a hidden nostril cannot blow up
    CALIBRATE_TIME = 2.0    # seconds of relaxed nose at the start
    RISE_TIME = 6.0         # r0 follows a slowly growing resting level
    FALL_TIME = 0.8         # ... and a shrinking one faster
    LEARN_BELOW = 0.10      # only adapt r0 while flare is below this
    MAX_PITCH = 10.0        # degrees away from the calibrated pitch
    SMOOTH = 0.4            # flare smoothing (0 = none)

    def __init__(self):
        self.recalibrate()

    def recalibrate(self) -> None:
        """Start learning the resting level again (C key)."""
        self._cal_start: Optional[float] = None
        self._cal_samples = []
        self._cal_pitch = []
        self.rest = None                  # [left, right]
        self.pitch_ref = None
        self._t: Optional[float] = None
        self._flare = 0.0

    def reset(self) -> None:
        """Face lost: keep the calibration, just forget the timing."""
        self._t = None
        self._flare = 0.0

    # ------------------------------------------------------------------

    def update(self, frame_rgb: np.ndarray, pts: np.ndarray, now: float,
               pitch: float = 0.0) -> NostrilResult:
        h, w = frame_rgb.shape[:2]
        measured = self._measure(frame_rgb, pts)
        if measured is None:
            return NostrilResult(pitch=pitch)
        ratios, contours, box = measured
        result = NostrilResult(ok=True, ratio=tuple(ratios), pitch=pitch,
                               contours=[[(float(x / w), float(y / h)) for x, y in c]
                                         for c in contours],
                               box=tuple((float(x / w), float(y / h)) for x, y in box))

        dt = 0.0 if self._t is None else max(0.0, now - self._t)
        self._t = now

        # 1) Calibration: average the first seconds of a relaxed nose.
        if self.rest is None:
            if self._cal_start is None:
                self._cal_start = now
            self._cal_samples.append(ratios)
            self._cal_pitch.append(pitch)
            if now - self._cal_start >= self.CALIBRATE_TIME and len(self._cal_samples) >= 5:
                self.rest = [max(float(v), self.MIN_REST)
                             for v in np.median(np.array(self._cal_samples), axis=0)]
                self.pitch_ref = float(np.median(self._cal_pitch))
            result.calibrating = True
            return result

        # 2) Head tilted too far: the nostrils look bigger, do not trust it.
        if abs(pitch - self.pitch_ref) > self.MAX_PITCH:
            self._flare = 0.0
            result.tilted = True
            result.rest = tuple(self.rest)
            return result

        sides = [(r - r0) / max(r0, self.MIN_REST) for r, r0 in zip(ratios, self.rest)]
        flare = float(np.mean(sides))
        self._flare = self._flare * self.SMOOTH + flare * (1 - self.SMOOTH)

        # 3) Let the resting level follow slow changes (light, posture),
        #    but only while the nose is relaxed.
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

    def _measure(self, frame_rgb, pts):
        """Dark area per half of the nostril patch, plus outlines for display."""
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

        # Same scale on both axes, so the area is relative to nose width².
        s = self.PATCH_W / (u1 - u0)                  # patch px per image px
        pw = self.PATCH_W
        ph = max(8, int(round((v1 - v0) * s)))
        origin = centre + across * u0 + down * v0
        to_img = np.array([[across[0] / s, down[0] / s, origin[0]],
                           [across[1] / s, down[1] / s, origin[1]]], dtype=np.float32)
        patch = cv2.warpAffine(frame_rgb, to_img, (pw, ph),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)

        light = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
        ref = self._skin_brightness(frame_rgb, pts, nose_w)
        mask = (light < self.DARK_RATIO * ref).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        mid = int(round(-u0 * s))                     # u = 0 is the nose middle
        mid = min(max(mid, 1), pw - 1)
        halves = ((0, mid), (mid, pw))
        ratios, contours = [], []
        for a, b in halves:
            half = np.zeros_like(mask)
            half[:, a:b] = mask[:, a:b]
            n, labels, stats, _ = cv2.connectedComponentsWithStats(half, connectivity=8)
            keep = np.zeros_like(mask)
            for k in range(1, n):
                if stats[k, cv2.CC_STAT_AREA] >= self.MIN_BLOB:
                    keep[labels == k] = 1
            ratios.append(float(keep.sum()) / float(max(1, (b - a) * ph)))
            cs, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                c = cv2.approxPolyDP(c, 0.8, True)[:, 0, :].astype(np.float64)
                contours.append([tuple(origin + across * (u / s) + down * (v / s)) for u, v in c])

        box = [origin + across * (u / s) + down * (v / s)
               for u, v in ((0, 0), (pw, 0), (pw, ph), (0, ph))]
        return ratios, contours, box

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
