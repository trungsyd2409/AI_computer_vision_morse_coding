"""Tests for the pinch detector using synthetic hand poses."""

import unittest

import numpy as np

from morse_hand.gestures import FINGER_TIPS, THUMB_TIP, PinchDetector, TouchPhase


def open_hand() -> np.ndarray:
    """A flat, open right hand roughly 200 px tall."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = (400, 500, 0)          # wrist
    pts[9] = (400, 400, 0)          # middle finger base -> palm length 100
    pts[THUMB_TIP] = (300, 400, 0)
    pts[FINGER_TIPS["index"]] = (360, 300, 0)
    pts[FINGER_TIPS["middle"]] = (400, 290, 0)
    pts[FINGER_TIPS["ring"]] = (440, 300, 0)
    pts[FINGER_TIPS["pinky"]] = (480, 330, 0)
    return pts


def touching(finger: str) -> np.ndarray:
    pts = open_hand()
    pts[THUMB_TIP] = pts[FINGER_TIPS[finger]] + np.array([8, 8, 0], dtype=np.float32)
    return pts


class PinchDetectorTest(unittest.TestCase):
    def run_frames(self, detector, poses, start=0.0, dt=1 / 30):
        events, t = [], start
        for pose in poses:
            events.extend(detector.update(pose, t))
            t += dt
        return events, t

    def test_tap_produces_down_then_up(self):
        d = PinchDetector()
        poses = [open_hand()] * 3 + [touching("middle")] * 4 + [open_hand()] * 4
        events, _ = self.run_frames(d, poses)
        phases = [(e.phase, e.finger) for e in events]
        self.assertEqual(phases, [(TouchPhase.DOWN, "middle"), (TouchPhase.UP, "middle")])

    def test_single_noisy_frame_is_ignored(self):
        d = PinchDetector()
        poses = [open_hand(), touching("index"), open_hand(), open_hand()]
        events, _ = self.run_frames(d, poses)
        self.assertEqual(events, [])

    def test_hold_fires_once(self):
        d = PinchDetector(hold_time=0.5)
        poses = [touching("pinky")] * 30 + [open_hand()] * 3
        events, _ = self.run_frames(d, poses)
        phases = [e.phase for e in events]
        self.assertEqual(phases.count(TouchPhase.HOLD), 1)
        self.assertEqual(phases[0], TouchPhase.DOWN)
        self.assertEqual(phases[-1], TouchPhase.UP)

    def test_scale_invariance(self):
        """The same pose at half size must still register as a touch."""
        d = PinchDetector()
        small = [touching("ring") * 0.5] * 3
        events, _ = self.run_frames(d, small)
        self.assertEqual(events[0].finger, "ring")


if __name__ == "__main__":
    unittest.main()
