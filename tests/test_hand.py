"""
Tests for the index + middle finger gestures.

READY_PHOTO and LOOP_PHOTO are MediaPipe world landmarks (metres) measured
on the two reference photos of the gestures.
"""

import unittest

import numpy as np

from blink_morse.hand import GestureDetector, measure
from blink_morse.press import PressDetector, PressKind

FPS = 30.0

READY_PHOTO = np.array([
    (0.0068, 0.0916, 0.0275),
    (-0.0199, 0.0672, 0.0264),
    (-0.0311, 0.0471, 0.0141),
    (-0.0444, 0.0305, -0.0116),
    (-0.0347, 0.0117, -0.0401),
    (-0.0232, -0.0078, 0.0206),
    (-0.0218, -0.0376, 0.0134),
    (-0.0261, -0.0651, 0.0086),
    (-0.0288, -0.0849, -0.0087),
    (-0.0027, -0.0061, 0.0070),
    (-0.0036, -0.0442, -0.0054),
    (-0.0131, -0.0697, -0.0184),
    (-0.0215, -0.0961, -0.0321),
    (0.0142, 0.0040, -0.0133),
    (-0.0071, -0.0170, -0.0286),
    (-0.0235, 0.0070, -0.0273),
    (-0.0297, 0.0354, -0.0186),
    (0.0177, 0.0240, -0.0272),
    (0.0005, 0.0070, -0.0342),
    (-0.0189, 0.0148, -0.0324),
    (-0.0239, 0.0331, -0.0218),
])

LOOP_PHOTO = np.array([
    (0.0004, 0.0890, 0.0180),
    (-0.0240, 0.0630, 0.0274),
    (-0.0385, 0.0427, 0.0207),
    (-0.0534, 0.0218, 0.0028),
    (-0.0466, 0.0052, -0.0200),
    (-0.0185, -0.0043, 0.0250),
    (-0.0287, -0.0334, 0.0191),
    (-0.0391, -0.0542, 0.0106),
    (-0.0493, -0.0649, -0.0101),
    (-0.0020, -0.0047, 0.0069),
    (0.0061, -0.0449, 0.0014),
    (-0.0116, -0.0652, -0.0122),
    (-0.0300, -0.0743, -0.0252),
    (0.0137, 0.0032, -0.0166),
    (-0.0087, -0.0168, -0.0264),
    (-0.0281, -0.0021, -0.0196),
    (-0.0340, 0.0230, -0.0070),
    (0.0120, 0.0178, -0.0316),
    (-0.0097, 0.0044, -0.0346),
    (-0.0289, 0.0087, -0.0260),
    (-0.0316, 0.0267, -0.0146),
])


def play(gesture, detector, script, t=0.0):
    """script: list of (seconds, world landmarks or None). Returns (kinds, results, t)."""
    kinds, results = [], []
    for seconds, world in script:
        for _ in range(max(1, round(seconds * FPS))):
            res = gesture.update(world, t)
            results.append(res)
            if res.armed and not res.holding:
                kinds += [e.kind for e in detector.update(res.score, t)]
            elif not res.armed:
                detector.reset()
            t += 1 / FPS
    return kinds, results, t


class MeasureTest(unittest.TestCase):
    def test_photos_pass_the_pose_rules(self):
        g = GestureDetector()
        for world in (READY_PHOTO, LOOP_PHOTO):
            self.assertTrue(all(g.checks(measure(world)).values()))

    def test_curl_tells_the_photos_apart(self):
        g = GestureDetector(loop_angle=85)
        ready, loop = measure(READY_PHOTO), measure(LOOP_PHOTO)
        self.assertLess(ready.curl, 80)
        self.assertGreater(loop.curl, 100)
        self.assertLess(g.score_for(ready.curl), g.READY_SCORE)
        self.assertGreater(g.score_for(loop.curl), 0.5)

    def test_open_hand_fails_the_folded_rule(self):
        world = READY_PHOTO.copy()
        # Stretch ring and pinky out like the raised fingers.
        for mcp, tip in ((13, 16), (17, 20)):
            direction = world[12] - world[9]
            for k, i in enumerate(range(mcp + 1, tip + 1), start=1):
                world[i] = world[mcp] + direction * k / 3
        checks = GestureDetector().checks(measure(world))
        self.assertFalse(checks["ring folded"])
        self.assertTrue(GestureDetector(need_folded=False).pose_ok(world))

    def test_spread_fingers_are_not_together(self):
        world = READY_PHOTO.copy()
        world[8] += (world[5] - world[17]) * 1.2     # index tip far to the side
        self.assertFalse(GestureDetector().checks(measure(world))["tips together"])


class GestureFlowTest(unittest.TestCase):
    def setUp(self):
        self.g = GestureDetector()
        self.d = PressDetector(0.5, 0.25)        # dash from 0.375 s

    def test_ready_arms_then_loops_type_dot_and_dash(self):
        script = [(0.4, READY_PHOTO), (0.2, LOOP_PHOTO), (0.3, READY_PHOTO),
                  (0.6, LOOP_PHOTO), (0.3, READY_PHOTO)]
        kinds, results, _ = play(self.g, self.d, script)
        self.assertTrue(any("armed" in r.events for r in results))
        self.assertEqual(kinds, [PressKind.DOT, PressKind.DASH])

    def test_loop_without_ready_first_types_nothing(self):
        kinds, results, _ = play(self.g, self.d, [(1.0, LOOP_PHOTO), (0.3, None)])
        self.assertEqual(kinds, [])
        self.assertFalse(any(r.armed for r in results))

    def test_losing_the_hand_switches_recording_off(self):
        _, results, t = play(self.g, self.d, [(0.4, READY_PHOTO), (0.5, None)])
        self.assertFalse(results[-1].armed)
        self.assertTrue(any("disarmed" in r.events for r in results))

    def test_one_lost_frame_does_not_break_a_dash(self):
        script = [(0.4, READY_PHOTO), (0.2, LOOP_PHOTO), (1 / FPS, None),
                  (0.4, LOOP_PHOTO), (0.3, READY_PHOTO)]
        kinds, _, _ = play(self.g, self.d, script)
        self.assertEqual(kinds, [PressKind.DASH])

    def test_hand_lost_mid_loop_drops_the_symbol(self):
        script = [(0.4, READY_PHOTO), (0.2, LOOP_PHOTO), (0.6, None)]
        kinds, _, _ = play(self.g, self.d, script)
        self.assertEqual(kinds, [])

    def test_states(self):
        _, results, _ = play(self.g, self.d, [(0.4, READY_PHOTO), (0.2, LOOP_PHOTO)])
        self.assertEqual(results[0].state, "ready")
        self.assertEqual(results[-1].state, "loop")
        none = GestureDetector().update(None, 0.0)
        self.assertEqual(none.state, "none")


if __name__ == "__main__":
    unittest.main()
