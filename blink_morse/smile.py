"""
Smile detection from several face measurements, fused into one score.

A smile is detected with the lips closed or open. A closed-lip smile does
not open the jaw, so the app looks at what every smile does: the mouth
corners move up and outwards, and the cheeks rise. Five features:

* smile:  mean of MediaPipe's `mouthSmileLeft` / `mouthSmileRight`
* cheek:  mean of `cheekSquintLeft` / `cheekSquintRight` (cheeks pushed up)
* dimple: mean of `mouthDimpleLeft` / `mouthDimpleRight`
* width:  distance between the mouth corners (landmarks 61 and 291)
* lift:   how far the mouth corners sit above the top of the upper lip
          (landmark 0), measured along the face's own up-down axis so a
          tilted head does not change it

Both distances are divided by the distance between the inner eye corners
(133 and 362), so moving closer to the camera does not change them.
Opening the mouth mainly moves the lower lip and the jaw, which these
features do not measure directly, so a smile counts whether the mouth is
open or closed.

The guided calibration (see fusion.py) asks for a smile with the lips
closed. A closed-lip smile is the weakest kind, so once it scores 100 %,
a smile with open lips scores even higher and is detected too.
"""

import math

import numpy as np

from .fusion import FusedSignal

FEATURES = ("smile", "cheek", "dimple", "width", "lift")
N_FEATURES = len(FEATURES)

MOUTH_CORNERS = (61, 291)
UPPER_LIP_TOP = 0
EYE_INNER = (133, 362)
FACE_DOWN = (168, 152)        # nose bridge -> chin, gives the face's "down"


def _pair_mean(blend: dict, name: str) -> float:
    return 0.5 * (blend.get(name + "Left", 0.0) + blend.get(name + "Right", 0.0))


def extract_features(pts: np.ndarray, blend: dict) -> np.ndarray:
    """
    One value per name in FEATURES; NaN when it cannot be measured.

    pts:   landmarks in image pixels, shape (478, 2)
    blend: blendshape name -> score
    """
    out = np.full(N_FEATURES, np.nan)
    out[0] = _pair_mean(blend, "mouthSmile")
    out[1] = _pair_mean(blend, "cheekSquint")
    out[2] = _pair_mean(blend, "mouthDimple")

    a, b = pts[EYE_INNER[0]], pts[EYE_INNER[1]]
    ref = float(math.hypot(*(b - a)))
    down = pts[FACE_DOWN[1]] - pts[FACE_DOWN[0]]
    norm = float(np.linalg.norm(down))
    if ref < 8.0 or norm < 1e-6:                   # face too small to measure
        return out
    down = down / norm

    left, right = pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]]
    out[3] = float(math.hypot(*(right - left))) / ref
    corners_down = 0.5 * (np.dot(left, down) + np.dot(right, down))
    out[4] = float(np.dot(pts[UPPER_LIP_TOP], down) - corners_down) / ref
    return out


class SmileSignal(FusedSignal):
    FEATURES = FEATURES
    MIN_STD = (0.01, 0.01, 0.01, 0.002, 0.002)
    # Without a calibration: the smile blendshape plus the two distances.
    FALLBACK = {"smile": 1.0, "width": 1.0, "lift": 1.0}
    ACTION = "smile"
    # People smile for a while without meaning to type, so a long smile is
    # given more time before the resting level is re-centred.
    STUCK_TIME = 8.0
