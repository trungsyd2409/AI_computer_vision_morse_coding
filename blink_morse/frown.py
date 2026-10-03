"""
Frown detection: the vertical furrow lines between the eyebrows.

MediaPipe's landmarks hardly move when you frown (the inner brow points and
the browDown blendshape barely change on most faces), but the skin between
the brows does: one or two thin, dark, vertical lines appear. This module
measures how strong those lines are.

How it works
------------
1. A small face-aligned patch is cut out between the inner ends of the
   eyebrows (landmarks 107 and 336), centred on the glabella (landmark 9),
   from a little above the brows down to the nose bridge (landmark 8). It
   is scaled by the inner-brow distance, so moving closer to or further
   from the camera does not change what it measures.
2. On the brightness channel a morphological "black-hat" with a short
   horizontal kernel keeps only dark details that are thinner than the
   kernel in the horizontal direction, i.e. thin vertical lines. Broad
   shadows and horizontal forehead lines are ignored.
3. `energy` = average black-hat response in the middle of the patch,
   divided by the skin brightness (so the light level does not matter),
   in percent.
4. Calibration (first CALIBRATE_TIME seconds, or C): with a relaxed face
   the resting energy E0 and the resting inner-brow distance d0 are
   learnt. Everyone has some lines at rest; only the change counts.
5. score = TEX_WEIGHT x lines + GEO_WEIGHT x brows, clipped to 0..1:
     lines = (E - E0) / max(E0, E_FLOOR) / TEX_RANGE
     brows = (d0 - d) / d0 / GEO_RANGE      (brows pulled together)
   The app compares the score with the threshold from the settings.

Limits: the lines are only a few pixels wide. Use 1280x720 if you can, and
light from above or the side shows them much better than light straight
from the front. Webcams that smooth skin make them fainter.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

INNER_BROWS = (107, 336)       # inner ends of the eyebrows (image left, right)
GLABELLA, FOREHEAD, NOSE_BRIDGE = 9, 151, 8
EYE_OUTER = (33, 263)          # face-size ruler and the "across" direction
FACE_DOWN = (168, 152)


@dataclass
class FrownResult:
    ok: bool = False                     # the area between the brows was measured
    calibrating: bool = False            # still learning the resting face
    score: float = 0.0                   # 0..1 frown strength (smoothed)
    lines: float = 0.0                   # 0..1 part from the furrow lines
    brows: float = 0.0                   # 0..1 part from the brows moving together
    energy: float = 0.0                  # furrow line energy (% of skin brightness)
    rest_energy: float = 0.0             # resting energy E0
    box: tuple = ()                      # patch corners, normalised image (x, y)
    contours: list = field(default_factory=list)  # visible furrow lines, normalised


class FrownDetector:
    PATCH_W = 64            # patch pixels across the area between the brows
    HALF_WIDTH = 0.30       # patch half width, in inner-brow distances
    TOP = 0.55              # patch top, as a fraction of the way up to landmark 151
    MARGIN = 0.15           # ignore this fraction of the patch on each side
    KERNEL_W = 9            # black-hat kernel width (patch px); lines thinner count
    LINE_LEVEL = 3.0        # black-hat above this (% of brightness) is drawn as a line
    E_FLOOR = 0.15          # floor for E0, so a smooth forehead cannot blow up
    TEX_RANGE = 2.0         # energy 3x the resting level -> lines = 1
    GEO_RANGE = 0.06        # brows 6 % closer than at rest -> brows = 1
    TEX_WEIGHT = 0.8
    GEO_WEIGHT = 0.2
    CALIBRATE_TIME = 2.0    # seconds of relaxed face at the start
    RISE_TIME = 6.0         # resting level follows slow changes ...
    FALL_TIME = 0.8         # ... faster when it should drop
    LEARN_BELOW = 0.15      # only adapt the resting level below this score
    STUCK_TIME = 4.0        # "frowning" longer than this: re-learn the resting face
    STUCK_LEVEL = 0.5
    SMOOTH = 0.4            # score smoothing (0 = none)

    def __init__(self):
        self.recalibrate()

    def recalibrate(self) -> None:
        """Learn the resting face again (C key)."""
        self._cal_start: Optional[float] = None
        self._cal = []
        self.rest_energy: Optional[float] = None
        self.rest_dist: Optional[float] = None
        self._t: Optional[float] = None
        self._score = 0.0
        self._high_since: Optional[float] = None

    def reset(self) -> None:
        """Face lost: keep the calibration, forget the timing."""
        self._t = None
        self._score = 0.0
        self._high_since = None

    # ------------------------------------------------------------------

    def update(self, frame_rgb: np.ndarray, pts: np.ndarray, now: float) -> FrownResult:
        h, w = frame_rgb.shape[:2]
        measured = self._measure(frame_rgb, pts)
        if measured is None:
            return FrownResult()
        energy, dist, box, contours = measured
        result = FrownResult(
            ok=True, energy=energy,
            box=tuple((float(x / w), float(y / h)) for x, y in box),
            contours=[[(float(x / w), float(y / h)) for x, y in c] for c in contours])

        dt = 0.0 if self._t is None else max(0.0, now - self._t)
        self._t = now

        # Calibration: median of the first seconds with a relaxed face.
        if self.rest_energy is None:
            if self._cal_start is None:
                self._cal_start = now
            self._cal.append((energy, dist))
            if now - self._cal_start >= self.CALIBRATE_TIME and len(self._cal) >= 5:
                e0, d0 = np.median(np.array(self._cal), axis=0)
                self.rest_energy, self.rest_dist = float(e0), float(d0)
            result.calibrating = True
            return result

        lines = (energy - self.rest_energy) / max(self.rest_energy, self.E_FLOOR)
        lines = float(np.clip(lines / self.TEX_RANGE, 0.0, 1.0))
        brows = (self.rest_dist - dist) / max(self.rest_dist, 1e-6)
        brows = float(np.clip(brows / self.GEO_RANGE, 0.0, 1.0))
        raw = float(np.clip(self.TEX_WEIGHT * lines + self.GEO_WEIGHT * brows, 0.0, 1.0))
        self._score = self._score * self.SMOOTH + raw * (1.0 - self.SMOOTH)

        # Frowning for far too long: the resting level is wrong (light
        # changed, bad calibration). Learn it again instead of staying stuck.
        if self._score >= self.STUCK_LEVEL:
            if self._high_since is None:
                self._high_since = now
            elif now - self._high_since >= self.STUCK_TIME:
                self.recalibrate()
                result.calibrating = True
                return result
        else:
            self._high_since = None

        # Follow slow changes of the resting face, only while relaxed.
        if dt > 0 and self._score < self.LEARN_BELOW:
            tau = self.FALL_TIME if energy < self.rest_energy else self.RISE_TIME
            self.rest_energy += (energy - self.rest_energy) * (1.0 - math.exp(-dt / tau))
            tau = self.FALL_TIME if dist > self.rest_dist else self.RISE_TIME
            self.rest_dist += (dist - self.rest_dist) * (1.0 - math.exp(-dt / tau))

        result.score = self._score
        result.lines = lines
        result.brows = brows
        result.rest_energy = self.rest_energy
        return result

    # ------------------------------------------------------------------

    def _measure(self, frame_rgb, pts):
        eye_w = float(np.linalg.norm(pts[EYE_OUTER[1]] - pts[EYE_OUTER[0]]))
        brow_w = float(np.linalg.norm(pts[INNER_BROWS[1]] - pts[INNER_BROWS[0]]))
        if eye_w < 20 or brow_w < 8:
            return None

        across = pts[EYE_OUTER[1]] - pts[EYE_OUTER[0]]
        across /= max(np.linalg.norm(across), 1e-6)
        down = np.array([-across[1], across[0]])
        if np.dot(down, pts[FACE_DOWN[1]] - pts[FACE_DOWN[0]]) < 0:
            down = -down
        centre = pts[GLABELLA]

        def v_of(p):
            return float(np.dot(p - centre, down))

        u0, u1 = -self.HALF_WIDTH * brow_w, self.HALF_WIDTH * brow_w
        v0 = v_of(pts[FOREHEAD]) * self.TOP          # negative: above landmark 9
        v1 = v_of(pts[NOSE_BRIDGE])
        if v1 - v0 < 4:
            return None

        s = self.PATCH_W / (u1 - u0)                 # patch px per image px
        pw, ph = self.PATCH_W, max(12, int(round((v1 - v0) * s)))
        origin = centre + across * u0 + down * v0
        to_img = np.array([[across[0] / s, down[0] / s, origin[0]],
                           [across[1] / s, down[1] / s, origin[1]]], dtype=np.float32)
        patch = cv2.warpAffine(frame_rgb, to_img, (pw, ph),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)

        light = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)[:, :, 0]
        light = cv2.GaussianBlur(light, (3, 3), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.KERNEL_W, 1))
        blackhat = cv2.morphologyEx(light, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        ref = max(float(np.median(light)), 1.0)
        response = blackhat / ref * 100.0            # % of skin brightness

        a = int(round(pw * self.MARGIN))
        b = pw - a
        energy = float(response[:, a:b].mean())

        # Visible furrow lines, for display only.
        mask = np.zeros((ph, pw), np.uint8)
        mask[:, a:b] = (response[:, a:b] > self.LINE_LEVEL).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 1), np.uint8))
        contours = []
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if cv2.contourArea(c) < 3:
                continue
            c = cv2.approxPolyDP(c, 0.8, True)[:, 0, :]
            contours.append([tuple(origin + across * (u / s) + down * (v / s)) for u, v in c])

        box = [origin + across * (u / s) + down * (v / s)
               for u, v in ((a, 0), (b, 0), (b, ph), (a, ph))]
        return energy, brow_w / eye_w, box, contours
