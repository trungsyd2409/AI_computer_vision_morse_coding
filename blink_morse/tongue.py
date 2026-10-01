"""
Tongue-out detection and tongue length, built on top of MediaPipe's face
landmarks (MediaPipe itself has no tongue landmarks).

How it works
------------
1. A face-aligned patch is cut out of the frame: it starts at the mouth
   opening (between landmarks 13 and 14) and runs down towards the chin,
   so it follows the face when the head tilts or moves.
2. Everything above the outer line of the lower lip is masked out. Lips
   are red too, so only the part of the tongue that sticks out past the
   lower lip can be measured.
3. The patch is converted to the Lab colour space. The a* channel says how
   red a pixel is. The cheek skin of the same frame is the reference, so a
   pixel counts as tongue when it is clearly redder than the cheeks. Using
   the cheeks of the same frame keeps it working when the light changes.
4. The tongue is the largest red blob that touches the lower lip. Its
   length is the distance from the mouth opening to the farthest point of
   the blob, measured along the face's "down" direction (nose -> chin).
5. Millimetres come from the iris: a human iris is about 11.7 mm across
   for almost everyone, and MediaPipe gives 4 points on each iris.

Limits: this is a 2D measurement, so a tongue pointing at the camera looks
shorter than it is, and a beard or very red chin skin can confuse it.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

IRIS_DIAMETER_MM = 11.7

LIP_TOP_INNER, LIP_BOTTOM_INNER = 13, 14
MOUTH_CORNERS = (61, 291)
NOSE_BRIDGE, CHIN = 168, 152
# Outer line of the lower lip, from one mouth corner to the other.
LOWER_LIP_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
CHEEKS = (50, 280)
# Opposite points on each iris (left/right, top/bottom).
IRIS_PAIRS = ((469, 471), (470, 472), (474, 476), (475, 477))


@dataclass
class TongueResult:
    out: bool = False            # a tongue is visible past the lower lip
    length_mm: float = 0.0       # visible length from the mouth opening
    ratio: float = 0.0           # same length divided by the mouth width
    contour: tuple = ()          # tongue outline, normalised image (x, y)
    lip: Optional[tuple] = None  # mouth opening, normalised image (x, y)
    tip: Optional[tuple] = None  # tongue tip, normalised image (x, y)


class TongueDetector:
    PATCH_W = 64          # patch pixels across one mouth width
    WIDTH = 1.4           # patch width, in mouth widths
    DEPTH = 2.2           # patch depth below the mouth opening, in mouth widths
    REDNESS = 9.0         # a* units redder than the cheeks to count as tongue
    LIP_MARGIN = 0.07     # extra gap under the lower lip line (lip edge is red too)
    MIN_PROTRUDE = 0.03   # tongue must reach at least this far past the lower
                          # lip (in mouth widths) to be shown at all
    SMOOTH = 0.5          # length smoothing (0 = none, 1 = frozen)

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.out = False
        self._mm_per_px: Optional[float] = None
        self._length_px = 0.0

    # ------------------------------------------------------------------

    def update(self, frame_rgb: np.ndarray, pts: np.ndarray) -> TongueResult:
        """
        frame_rgb: the RGB frame given to MediaPipe
        pts:       (478, 2) landmark positions in pixels
        """
        h, w = frame_rgb.shape[:2]
        left, right = pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]]
        mouth_w = float(np.linalg.norm(right - left))
        if mouth_w < 8:
            return self._finish(None, mouth_w, (w, h))

        # Face-aligned axes: "down" from the nose bridge to the chin, and
        # "across" from one mouth corner to the other.
        down = pts[CHIN] - pts[NOSE_BRIDGE]
        down /= max(np.linalg.norm(down), 1e-6)
        across = np.array([-down[1], down[0]])
        if np.dot(across, right - left) < 0:
            across = -across

        lip = (pts[LIP_TOP_INNER] + pts[LIP_BOTTOM_INNER]) / 2
        self._update_scale(pts)

        # Patch coordinates (u, v): u across the face, v down from the lip.
        s = self.PATCH_W / mouth_w                     # patch px per image px
        pw, ph = int(self.PATCH_W * self.WIDTH), int(self.PATCH_W * self.DEPTH)
        origin = lip - across * (mouth_w * self.WIDTH / 2)
        to_img = np.array([[across[0] / s, down[0] / s, origin[0]],
                           [across[1] / s, down[1] / s, origin[1]]], dtype=np.float32)
        patch = cv2.warpAffine(frame_rgb, to_img, (pw, ph),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)

        def to_patch(p):
            d = p - origin
            return np.array([np.dot(d, across) * s, np.dot(d, down) * s])

        # Only the area below the outer line of the lower lip (plus a small
        # margin, because the edge of the lip is red as well).
        lip_line = np.array([to_patch(pts[i]) for i in LOWER_LIP_OUTER])
        lip_line[:, 1] += self.LIP_MARGIN * self.PATCH_W
        lip_bottom_v = float(lip_line[:, 1].max())
        lip_line = lip_line[np.argsort(lip_line[:, 0])]
        poly = np.vstack([[[0, lip_line[0, 1]]], lip_line, [[pw, lip_line[-1, 1]]],
                          [[pw, ph], [0, ph]]])
        allowed = np.zeros((ph, pw), np.uint8)
        cv2.fillPoly(allowed, [np.round(poly).astype(np.int32)], 1)

        # Redness compared with the cheeks of the same frame.
        a = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)[:, :, 1].astype(np.float32)
        cheek_a = self._cheek_redness(frame_rgb, pts, mouth_w)
        mask = ((a - cheek_a) > self.REDNESS).astype(np.uint8) & allowed
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Pixels right under the lower lip: the tongue must touch them.
        above = np.zeros_like(allowed)
        above[6:] = allowed[:-6]
        band = (allowed == 1) & (above == 0)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best, best_area = 0, 0
        for k in range(1, n):
            area = stats[k, cv2.CC_STAT_AREA]
            cx = stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH] / 2
            if area < 25 or abs(cx - pw / 2) > pw * 0.35:
                continue
            if not np.any(band & (labels == k)):
                continue
            if area > best_area:
                best, best_area = k, area

        if best == 0:
            return self._finish(None, mouth_w, (w, h))

        blob = (labels == best).astype(np.uint8)
        vs, us = np.nonzero(blob)
        tip_v = float(vs.max())
        protrude = (tip_v - lip_bottom_v) / self.PATCH_W   # in mouth widths
        tip_u = float(us[vs >= tip_v - 1].mean())
        length_px = tip_v / s

        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True)[:, 0, :]

        def to_norm(uv):
            p = origin + across * (uv[0] / s) + down * (uv[1] / s)
            return (float(p[0] / w), float(p[1] / h))

        found = {
            "protrude": protrude,
            "length_px": length_px,
            "contour": tuple(to_norm(p) for p in outline),
            "lip": (float(lip[0] / w), float(lip[1] / h)),
            "tip": to_norm((tip_u, tip_v)),
        }
        return self._finish(found, mouth_w, (w, h))

    # ------------------------------------------------------------------

    def _update_scale(self, pts: np.ndarray) -> None:
        if len(pts) < 478:
            return
        d = np.mean([np.linalg.norm(pts[a] - pts[b]) for a, b in IRIS_PAIRS])
        if d < 3:
            return
        mm = IRIS_DIAMETER_MM / d
        self._mm_per_px = mm if self._mm_per_px is None else self._mm_per_px * 0.9 + mm * 0.1

    @staticmethod
    def _cheek_redness(frame_rgb, pts, mouth_w) -> float:
        h, w = frame_rgb.shape[:2]
        r = max(2, int(mouth_w * 0.08))
        values = []
        for i in CHEEKS:
            x, y = int(pts[i][0]), int(pts[i][1])
            x0, x1, y0, y1 = max(0, x - r), min(w, x + r), max(0, y - r), min(h, y + r)
            if x1 > x0 and y1 > y0:
                lab = cv2.cvtColor(np.ascontiguousarray(frame_rgb[y0:y1, x0:x1]),
                                   cv2.COLOR_RGB2LAB)
                values.append(float(np.median(lab[:, :, 1])))
        return float(np.mean(values)) if values else 140.0

    def _finish(self, found, mouth_w, size) -> TongueResult:
        """
        Report what is visible this frame. Whether the length is long enough
        to count as a Morse "press" is decided by the app, using the
        threshold from the settings window.
        """
        if found is None or found["protrude"] < self.MIN_PROTRUDE:
            self.out = False
            self._length_px = 0.0
            return TongueResult()

        self.out = True
        if self._length_px <= 0:
            self._length_px = found["length_px"]
        else:
            self._length_px = self._length_px * self.SMOOTH + found["length_px"] * (1 - self.SMOOTH)
        mm = self._length_px * self._mm_per_px if self._mm_per_px else 0.0
        return TongueResult(True, mm, self._length_px / mouth_w,
                            found["contour"], found["lip"], found["tip"])
