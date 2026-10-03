"""Tests for the frown detector using a synthetic forehead image."""

import unittest

import numpy as np

try:
    import cv2  # noqa: F401
    HAVE_CV2 = True
except ImportError:          # pragma: no cover
    HAVE_CV2 = False

if HAVE_CV2:
    import cv2
    from blink_morse.frown import FrownDetector

FPS = 30.0


def face_points() -> np.ndarray:
    """Just the landmarks the detector uses, on a 640x480 frame."""
    pts = np.zeros((478, 2), dtype=np.float64)
    pts[33], pts[263] = (240, 260), (400, 260)       # outer eye corners
    pts[107], pts[336] = (290, 200), (350, 200)      # inner brow ends
    pts[9] = (320, 205)                              # glabella
    pts[151] = (320, 140)                            # forehead
    pts[8] = (320, 240)                              # nose bridge
    pts[168], pts[152] = (320, 245), (320, 420)      # face "down"
    return pts


def skin(lines: bool) -> np.ndarray:
    img = np.full((480, 640, 3), (190, 140, 120), dtype=np.uint8)
    if lines:
        for x in (314, 326):                         # two furrows, 2 px wide
            cv2.line(img, (x, 180), (x, 232), (120, 80, 70), 2)
    return img


@unittest.skipUnless(HAVE_CV2, "OpenCV not installed")
class FrownDetectorTest(unittest.TestCase):
    def run_frames(self, det, img, seconds, start):
        t, result = start, None
        for _ in range(int(seconds * FPS)):
            result = det.update(img, face_points(), t)
            t += 1.0 / FPS
        return result, t

    def test_calibrates_then_detects_lines(self):
        det = FrownDetector()
        result, t = self.run_frames(det, skin(False), 2.2, 0.0)
        self.assertFalse(result.calibrating)
        self.assertLess(result.score, 0.05)

        result, t = self.run_frames(det, skin(True), 0.3, t)
        self.assertGreater(result.score, 0.5)
        self.assertGreater(len(result.contours), 0)

        result, _ = self.run_frames(det, skin(False), 0.3, t)
        self.assertLess(result.score, 0.05)

    def test_reports_calibrating_first(self):
        det = FrownDetector()
        result, _ = self.run_frames(det, skin(True), 0.5, 0.0)
        self.assertTrue(result.calibrating)
        self.assertEqual(result.score, 0.0)

    def test_long_frown_recalibrates(self):
        det = FrownDetector()
        _, t = self.run_frames(det, skin(False), 2.2, 0.0)
        result, _ = self.run_frames(det, skin(True), det.STUCK_TIME + 0.5, t)
        self.assertTrue(result.calibrating)


if __name__ == "__main__":
    unittest.main()
