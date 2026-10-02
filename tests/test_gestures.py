"""Tests for the index/middle finger key using synthetic hand poses."""

import unittest

import numpy as np

from morse_hand.gestures import FINGER_TIPS, FingerKey, PressKind


def apart() -> np.ndarray:
    """An open right hand, index and middle in a V, palm length 100 px."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = (400, 500, 0)          # wrist
    pts[9] = (400, 400, 0)          # middle finger base -> palm length 100
    pts[FINGER_TIPS["index"]] = (350, 300, 0)
    pts[FINGER_TIPS["middle"]] = (410, 290, 0)   # ~0.61 palm apart
    return pts


def together() -> np.ndarray:
    pts = apart()
    pts[FINGER_TIPS["index"]] = pts[FINGER_TIPS["middle"]] + np.array([-10, 5, 0],
                                                                      dtype=np.float32)
    return pts


class FingerKeyTest(unittest.TestCase):
    DT = 1 / 30

    def run_frames(self, key, poses, start=0.0):
        events, t = [], start
        for pose in poses:
            events.extend(key.update(pose, t))
            t += self.DT
        return [e.kind for e in events], t

    def test_short_touch_is_dot(self):
        key = FingerKey(time_unit=0.2)                # dash after 0.3 s
        kinds, _ = self.run_frames(key, [apart()] * 3 + [together()] * 5 + [apart()] * 4)
        self.assertEqual(kinds, [PressKind.DOWN, PressKind.DOT, PressKind.UP])

    def test_long_touch_is_dash_before_release(self):
        key = FingerKey(time_unit=0.2)
        kinds, _ = self.run_frames(key, [apart()] * 3 + [together()] * 15)
        self.assertEqual(kinds, [PressKind.DOWN, PressKind.DASH])
        kinds, _ = self.run_frames(key, [apart()] * 4, start=1.0)
        self.assertEqual(kinds, [PressKind.UP])        # no extra dot on release

    def test_single_noisy_frame_is_ignored(self):
        key = FingerKey()
        kinds, _ = self.run_frames(key, [apart(), together(), apart(), apart()])
        self.assertEqual(kinds, [])

    def test_pause_time_counts_after_release(self):
        key = FingerKey(time_unit=0.2)
        _, t = self.run_frames(key, [together()] * 4 + [apart()] * 2)
        self.assertFalse(key.touching)
        self.assertGreater(key.pause_time(t + 0.5), 0.5)

    def test_scale_invariance(self):
        """The same pose at half size must still register as a touch."""
        key = FingerKey()
        kinds, _ = self.run_frames(key, [together() * 0.5] * 3)
        self.assertEqual(kinds[0], PressKind.DOWN)


if __name__ == "__main__":
    unittest.main()
