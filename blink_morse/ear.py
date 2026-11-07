"""
Ear wiggle detection from the camera image.

MediaPipe's face mesh stops at the edge of the face (the nearest landmarks
are 234 and 454, just in front of each ear) and none of its blendshapes
covers the ears. So the ears are measured straight from the pixels,
relative to the head that MediaPipe tracks. Two things are measured for
each ear the camera can see:

* edge: how far the outer edge of the ear reaches beyond the face edge.
  The skin colour is sampled on the cheek next to the ear, several rows
  next to the face edge are checked for that colour, and the ear is the
  largest skin region that starts near the face edge. Its outermost pixel
  in each row is the edge. When the ear is pulled back towards the head,
  the edge moves inwards.
* move: points on the ear are tracked from frame to frame with optical
  flow (Lucas-Kanade). The head's own motion, estimated from the face
  landmarks, is removed, and what is left is summed with a slow leak, so
  it reads as "how far the ear has moved from where it rests".

All distances are in units of the distance between the outer eye corners
(landmarks 33 and 263), along the head's own axes, so they do not depend
on the distance to the camera or on a tilted head.

On the reference video (1280x720, eyes 230 px apart) a wiggle moved the
outer edge of the ear inwards by 2 to 3 px for 0.2 to 0.4 s. Turning the
head by 1 degree moved the same edge by 2.6 px, so head yaw and pitch are
added as extra "nuisance" features: the calibration (fusion.py) learns how
much of each ear measurement follows the head angle and cancels it.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .fusion import FusedSignal

# Face-edge landmarks just in front of each ear. Which one is on the
# screen's left depends on whether the image is mirrored, so the screen
# side is worked out every frame (see ear_sides).
EAR_LANDMARKS = (234, 454)
SIDES = ("left", "right")       # screen sides
EYE_OUTER = (33, 263)
CHIN, NOSE_BRIDGE = 152, 168
# Landmarks that move with the skull only, used to predict how points on
# the ear would move if the ear itself stayed still.
STABLE = (33, 133, 362, 263, 1, 168, 6, 197, 195, 5, 10, 151, 9, 8)

FEATURES = ("edge_l", "edge_r", "move_lx", "move_ly", "move_rx", "move_ry", "yaw", "pitch")
N_FEATURES = len(FEATURES)

# Where the ear is, relative to the face-edge landmark, in eye distances.
# u runs outwards from the face, v runs down the face.
SCAN_U = (-0.03, 0.55)          # edge scan, from just inside the face edge
SCAN_V = (-0.60, -0.05)         # rows across the upper two thirds of the ear
SCAN_ROWS = 15
TRACK_U = (0.0, 0.45)           # area where points are tracked
TRACK_V = (-0.65, 0.10)
VIEW_U = (-0.12, 0.48)          # crop shown in the HUD
VIEW_V = (-0.70, 0.20)
VIEW_SIZE = (108, 162)          # (width, height) of each HUD crop

MIN_EDGE = 0.08                 # an edge closer than this: ear not visible (the
                                # reference video measured about 0.15 to 0.2)
EAR_NEAR_FACE = 0.2             # the ear region must start this close to the face edge
EAR_MIN_WIDTH = 0.06            # median row width of the ear region, at least
SKIN_SIGMA = 3.5                # skin = within this many spreads of the cheek colour
# Skin is redder than it is blue in YCrCb. The cheek sample must pass this
# (otherwise it landed on hair or glasses and the ear counts as hidden),
# and so must every ear pixel, which keeps grey and white walls out.
MIN_SAMPLE_REDNESS = 10.0       # Cr - Cb of the cheek sample
MIN_PIXEL_REDNESS = 6.0         # Cr - Cb of an ear pixel
MOVE_LEAK = 1.5                 # seconds: the summed motion slowly returns to 0


@dataclass
class EarView:
    """What the HUD shows for one ear."""
    crop: Optional[np.ndarray] = None      # RGB, VIEW_SIZE, head-aligned
    edge_x: Optional[float] = None         # measured ear edge, crop pixels
    face_x: float = 0.0                    # face-edge landmark, crop pixels
    visible: bool = False


@dataclass
class EarFrame:
    features: np.ndarray = field(default_factory=lambda: np.full(N_FEATURES, np.nan))
    views: dict = field(default_factory=dict)       # side -> EarView
    anchors: dict = field(default_factory=dict)     # side -> (x, y) of the ear edge, image px


def head_axes(pts: np.ndarray):
    """(ux, uy, d): unit vector along the eye line, unit vector down the face, eye distance."""
    eye = pts[EYE_OUTER[1]] - pts[EYE_OUTER[0]]
    d = float(np.linalg.norm(eye))
    if d < 1e-6:
        return None
    ux = eye / d
    uy = np.array([-ux[1], ux[0]])
    # The eye line points left in a mirrored image, which would flip the
    # perpendicular. Make it point towards the chin either way.
    if np.dot(uy, pts[CHIN] - pts[NOSE_BRIDGE]) < 0:
        uy = -uy
    return ux, uy, d


def ear_sides(pts: np.ndarray, axes) -> dict:
    """Screen side -> (face-edge landmark, outward sign along ux)."""
    ux = axes[0]
    a, b = EAR_LANDMARKS
    mid = (pts[a] + pts[b]) / 2.0
    out = {}
    left, right = (a, b) if pts[a][0] <= pts[b][0] else (b, a)
    for side, idx in (("left", left), ("right", right)):
        out[side] = (idx, 1.0 if np.dot(pts[idx] - mid, ux) >= 0 else -1.0)
    return out


def _sample(img: np.ndarray, origin, ax_u, ax_v, us, vs) -> np.ndarray:
    """Sample `img` on the grid origin + u*ax_u + v*ax_v (image px), bilinear."""
    uu, vv = np.meshgrid(us, vs)
    mx = (origin[0] + uu * ax_u[0] + vv * ax_v[0]).astype(np.float32)
    my = (origin[1] + uu * ax_u[1] + vv * ax_v[1]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def skin_model(frame_rgb: np.ndarray, pts: np.ndarray, side: str, axes):
    """Median and spread of (Cr, Cb) on the cheek just inside the ear."""
    ux, uy, d = axes
    idx, out = ear_sides(pts, axes)[side]
    centre = pts[idx] - out * 0.12 * d * ux - 0.15 * d * uy
    r = max(3, int(0.04 * d))
    patch = _sample(frame_rgb, centre, ux, uy, np.arange(-r, r + 1), np.arange(-r, r + 1))
    ycc = cv2.cvtColor(patch, cv2.COLOR_RGB2YCrCb).reshape(-1, 3).astype(np.float32)
    med = np.median(ycc, axis=0)
    spread = 1.4826 * np.median(np.abs(ycc - med), axis=0)
    spread = np.maximum(spread, (12.0, 4.0, 4.0))
    return med, spread


def ear_blob(skin: np.ndarray, d: float, start: int) -> np.ndarray:
    """
    Mask of the ear inside the scan strip (rows x columns), or all False.

    There is often a darker band between the cheek and the ear (shadow, hair,
    the fold in front of the ear), so the ear is found as a connected skin
    region rather than as a run that starts at the face edge. Small gaps
    along a row (glasses arm, ear folds) are closed first. The region must
    start near the face edge, span at least 3 rows and be wide enough to be
    an ear, which rules out thin lines such as a strand of hair or the edge
    of a door frame. The largest such region wins.
    """
    gap = max(2, int(0.03 * d))
    mask = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((1, gap + 1), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, best_area = 0, 0
    for k in range(1, n):
        x, _, _, h, area = stats[k]
        if h < 3 or x > start + EAR_NEAR_FACE * d or area <= best_area:
            continue
        widths = (labels == k).sum(axis=1)
        if np.median(widths[widths > 0]) < EAR_MIN_WIDTH * d:
            continue
        best, best_area = k, area
    return labels == best if best else np.zeros_like(skin)


def measure_edge(frame_rgb: np.ndarray, pts: np.ndarray, side: str, axes) -> float:
    """Outer ear edge beyond the face edge, in eye distances, or NaN."""
    ux, uy, d = axes
    idx, out = ear_sides(pts, axes)[side]
    med, spread = skin_model(frame_rgb, pts, side, axes)
    if med[1] - med[2] < MIN_SAMPLE_REDNESS:
        return float("nan")                       # cheek sample is not skin
    us = np.arange(SCAN_U[0] * d, SCAN_U[1] * d, 1.0)   # one sample per image pixel
    vs = np.linspace(SCAN_V[0] * d, SCAN_V[1] * d, SCAN_ROWS)
    strip = _sample(frame_rgb, pts[idx], out * ux, uy, us, vs)
    ycc = cv2.cvtColor(strip, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    dist = np.sqrt(((ycc[..., 1] - med[1]) / spread[1]) ** 2
                   + ((ycc[..., 2] - med[2]) / spread[2]) ** 2)
    skin = ((dist < SKIN_SIGMA) & (ycc[..., 0] > 0.35 * med[0])
            & (ycc[..., 1] - ycc[..., 2] > MIN_PIXEL_REDNESS))
    blob = ear_blob(skin, d, int(round(-SCAN_U[0] * d)))
    edges = []
    for row in blob:
        cols = np.flatnonzero(row)
        # A row that runs to the end of the scan has no edge in view.
        if len(cols) and cols[-1] < len(row) - 2:
            edges.append(us[cols[-1]] / d)
    if len(edges) < 3:
        return float("nan")
    edge = float(np.median(edges))
    return edge if edge >= MIN_EDGE else float("nan")


def ear_view(frame_rgb: np.ndarray, pts: np.ndarray, side: str, axes, edge: float) -> EarView:
    """Head-aligned crop around one ear for the HUD, with the edges in crop pixels."""
    ux, uy, d = axes
    idx, out = ear_sides(pts, axes)[side]
    w, h = VIEW_SIZE
    # The crop's x axis follows the screen (to the right), so it looks like
    # the camera image; `outward` is +1 when the ear sticks out to the right.
    cx = ux if ux[0] >= 0 else -ux
    outward = out * float(np.dot(ux, cx))
    if outward > 0:
        u0, u1 = VIEW_U
    else:
        u0, u1 = -VIEW_U[1], -VIEW_U[0]
    us = np.linspace(u0 * d, u1 * d, w)
    vs = np.linspace(VIEW_V[0] * d, VIEW_V[1] * d, h)
    crop = _sample(frame_rgb, pts[idx], cx, uy, us, vs)
    scale = w / ((u1 - u0) * d)
    face_x = (0.0 - u0 * d) * scale
    edge_x = None
    if math.isfinite(edge):
        edge_x = (outward * edge * d - u0 * d) * scale
    return EarView(crop=np.ascontiguousarray(crop), edge_x=edge_x, face_x=face_x,
                   visible=math.isfinite(edge))


class EarTracker:
    """Per-frame ear measurements; keeps the previous frame for optical flow."""

    def __init__(self):
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None
        self._offset = {idx: np.zeros(2) for idx in EAR_LANDMARKS}

    def lost(self) -> None:
        self._prev_gray = self._prev_pts = self._prev_t = None
        self._offset = {idx: np.zeros(2) for idx in EAR_LANDMARKS}

    def update(self, frame_rgb: np.ndarray, pts: np.ndarray, yaw: float, pitch: float,
               now: float, want_view: bool = True) -> EarFrame:
        result = EarFrame()
        f = result.features
        f[FEATURES.index("yaw")] = yaw
        f[FEATURES.index("pitch")] = pitch
        axes = head_axes(pts)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if axes is None:
            self.lost()
            return result
        ux, uy, d = axes

        dt = 0.0 if self._prev_t is None else max(0.0, now - self._prev_t)
        head = None
        if self._prev_gray is not None:
            head, _ = cv2.estimateAffinePartial2D(
                self._prev_pts[list(STABLE)].astype(np.float32),
                pts[list(STABLE)].astype(np.float32))

        for side, (idx, out) in ear_sides(pts, axes).items():
            key = "l" if side == "left" else "r"
            edge = measure_edge(frame_rgb, pts, side, axes)
            f[FEATURES.index(f"edge_{key}")] = edge
            visible = math.isfinite(edge)

            move = None
            if visible and head is not None:
                move = self._track(gray, idx, out, head, axes)
            leak = math.exp(-dt / MOVE_LEAK) if dt > 0 else 1.0
            if move is None:
                self._offset[idx] *= leak
            else:
                self._offset[idx] = self._offset[idx] * leak + move
            if visible:
                f[FEATURES.index(f"move_{key}x")] = self._offset[idx][0]
                f[FEATURES.index(f"move_{key}y")] = self._offset[idx][1]
                result.anchors[side] = tuple(pts[idx] + out * edge * d * ux - 0.3 * d * uy)
            if want_view:
                result.views[side] = ear_view(frame_rgb, pts, side, axes, edge)

        self._prev_gray, self._prev_pts, self._prev_t = gray, pts.copy(), now
        return result

    def _track(self, gray: np.ndarray, idx: int, out: float, head: np.ndarray,
               axes) -> Optional[np.ndarray]:
        """Median motion of ear points that the head does not explain, (out, down) / d."""
        ux, uy, d = axes
        prev = self._prev_gray
        a = self._prev_pts[idx]
        corners = [a + out * u * d * ux + v * d * uy
                   for u, v in ((TRACK_U[0], TRACK_V[0]), (TRACK_U[1], TRACK_V[0]),
                                (TRACK_U[1], TRACK_V[1]), (TRACK_U[0], TRACK_V[1]))]
        poly = np.array(corners, dtype=np.float32)
        h, w = prev.shape
        x0, y0 = np.floor(poly.min(axis=0)).astype(int)
        x1, y1 = np.ceil(poly.max(axis=0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round(poly - (x0, y0)).astype(np.int32), 255)
        found = cv2.goodFeaturesToTrack(prev[y0:y1, x0:x1], 40, 0.01,
                                        max(3.0, 0.02 * d), mask=mask)
        if found is None or len(found) < 4:
            return None
        p0 = found.reshape(-1, 2) + (x0, y0)
        win = int(max(11, 0.06 * d)) | 1
        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0.astype(np.float32), None,
                                                 winSize=(win, win), maxLevel=2)
        ok = status.ravel() == 1
        if ok.sum() < 4:
            return None
        p0, p1 = p0[ok], p1.reshape(-1, 2)[ok]
        predicted = p0 @ head[:, :2].T + head[:, 2]
        resid = np.median(p1 - predicted, axis=0)
        return np.array([out * float(np.dot(resid, ux)), float(np.dot(resid, uy))]) / d


class EarSignal(FusedSignal):
    FEATURES = FEATURES
    MIN_STD = (0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.3, 0.3)
    # Without a calibration: the edges move inwards during a wiggle.
    FALLBACK = {"edge_l": -1.0, "edge_r": -1.0}
    NUISANCE = ("yaw", "pitch")
    ACTION = "ear"
    REST_STEPS = ("relax", "turn")
    # Less shrinkage than the smile: the head-angle cancellation depends on
    # the covariance between the angles and the ear measurements.
    SHRINK = 0.15
    TURN_TIME = 3.0
    MAX_PITCH = 10.0
    MAX_YAW = 12.0

    @property
    def steps(self) -> list:
        # A head-turn step with the ears relaxed, so the calibration sees
        # how the ear measurements follow the head angle.
        steps = [("relax", self.FIRST_REST), ("turn", self.TURN_TIME)]
        for _ in range(self.ROUNDS):
            steps += [(self.ACTION, self.ACTION_TIME), ("relax", self.RELAX_TIME)]
        return steps
