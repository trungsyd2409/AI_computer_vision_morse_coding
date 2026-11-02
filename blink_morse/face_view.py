"""
Debug view of the tracked face: the MediaPipe mesh redrawn inside a panel.

The face is cut out of the camera image by its outline, scaled to fit the
panel and drawn as lines only:

* the inner mesh faintly (opacity 0.2), so jitter in the landmarks is easy
  to see without hiding everything else,
* the face outline bold,
* each nostril area bold, filled more strongly as the flare score rises.

Lines are drawn with OpenCV into one 8-bit coverage mask per layer (one C
call for all 1322 mesh edges) and the masks are merged into an RGBA image.
Drawing coverage instead of colour keeps anti-aliased edges free of dark
fringes when the image is blended over the glass panel.
"""

from typing import Optional

import cv2
import numpy as np
import pygame

from .mesh_data import FACE_OVAL, NOSTRILS, TESSELATION

MESH_ALPHA = 0.20          # inner mesh lines
OVAL_ALPHA = 0.95          # face outline
LINE_COLOR = (236, 240, 248)
SHIFT = 4                  # OpenCV sub-pixel bits: coordinates are x16


class FaceMeshView:
    def __init__(self):
        self._edges = np.array(TESSELATION, dtype=np.int32)
        self._oval = np.array(FACE_OVAL, dtype=np.int32)
        self._nostrils = [np.array(n, dtype=np.int32) for n in NOSTRILS]
        self._buffer = None            # keeps the bytes behind the surface alive
        self.nostril_centres = []      # panel pixels, for the glow

    def render(self, points: np.ndarray, size: tuple, nostril_color: tuple,
               fill: float = 0.3, pad: int = 6) -> Optional[pygame.Surface]:
        """
        points:        (478, 2) landmarks in camera pixels
        size:          (width, height) of the drawing area
        nostril_color: RGB of the nostril outline and fill
        fill:          0..1 opacity of the nostril fill
        """
        w, h = size
        oval = points[self._oval]
        lo, hi = oval.min(axis=0), oval.max(axis=0)
        span = np.maximum(hi - lo, 1.0)
        scale = min((w - 2 * pad) / span[0], (h - 2 * pad) / span[1])
        q = (points - (lo + hi) / 2.0) * scale + np.array([w / 2.0, h / 2.0])
        q16 = np.round(q * (1 << SHIFT)).astype(np.int32)

        mesh = np.zeros((h, w), np.uint8)
        cv2.polylines(mesh, list(q16[self._edges]), False, 255, 1, cv2.LINE_AA, SHIFT)
        outline = np.zeros((h, w), np.uint8)
        cv2.polylines(outline, [q16[self._oval]], True, 255, 2, cv2.LINE_AA, SHIFT)

        nose_line = np.zeros((h, w), np.uint8)
        nose_fill = np.zeros((h, w), np.uint8)
        self.nostril_centres = []
        for idx in self._nostrils:
            hull = cv2.convexHull(q16[idx])
            cv2.fillConvexPoly(nose_fill, hull, 255, cv2.LINE_AA, SHIFT)
            cv2.polylines(nose_line, [hull], True, 255, 2, cv2.LINE_AA, SHIFT)
            self.nostril_centres.append(tuple(q[idx].mean(axis=0)))

        alpha = np.maximum.reduce([
            mesh * MESH_ALPHA,
            outline * OVAL_ALPHA,
            nose_line.astype(np.float32),
            nose_fill * float(np.clip(fill, 0.0, 1.0)),
        ])
        rgba = np.empty((h, w, 4), np.uint8)
        rgba[:, :, :3] = LINE_COLOR
        rgba[np.maximum(nose_line, nose_fill) > 0, :3] = nostril_color[:3]
        rgba[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)

        self._buffer = rgba.tobytes()
        return pygame.image.frombuffer(self._buffer, (w, h), "RGBA")
