"""Tests for the arm-curl detector and the elbow angle maths."""

import unittest

from blink_morse.arm_tracker import (ArmPose, build_result, curl_from_angle,
                                     elbow_angle)
from blink_morse.config import PushupSettings
from blink_morse.curls import CurlDetector, CurlKind
from blink_morse.morse import MorseComposer

FPS = 30.0
DOWN, UP = 0.05, 0.9          # arm straight / curled


def play(detector, script, start=0.0, composer=None):
    """script: list of (seconds, left_curl, right_curl). Returns (events, t)."""
    events, t = [], start
    for seconds, left, right in script:
        for _ in range(max(1, round(seconds * FPS))):
            new = detector.update(left, right, t)
            events.extend(new)
            if composer is not None:
                for e in new:
                    composer.add_symbol("." if e.kind == CurlKind.DOT else "-")
                composer.update(detector.pause_time(t))
            t += 1.0 / FPS
    return events, t


def kinds(events):
    return [e.kind for e in events]


class AngleTest(unittest.TestCase):
    def test_straight_arm(self):
        self.assertAlmostEqual(elbow_angle((0, 0), (0, 1), (0, 2)), 180, delta=0.01)
        self.assertEqual(curl_from_angle(180), 0.0)

    def test_right_angle(self):
        self.assertAlmostEqual(elbow_angle((0, 0, 0), (0, 1, 0), (1, 1, 0)), 90, delta=0.01)

    def test_full_curl(self):
        self.assertEqual(curl_from_angle(30), 1.0)
        self.assertGreater(curl_from_angle(90), 0.5)

    def test_mirror_swaps_arms(self):
        a, b = ArmPose(90, 0.6), ArmPose(170, 0.0)
        self.assertIs(build_result(a, b, mirrored=False).left, a)
        self.assertIs(build_result(a, b, mirrored=True).left, b)
        self.assertIs(build_result(a, b, mirrored=True, swap_arms=True).left, a)


class CurlDetectorTest(unittest.TestCase):
    """t = 0.5 s, so a dash starts at 0.75 s of curl."""

    def setUp(self):
        self.d = CurlDetector(curl_threshold=0.5, time_unit=0.5)
        _, self.t = play(self.d, [(0.5, DOWN, DOWN)])

    def test_short_curl_is_a_dot(self):
        events, _ = play(self.d, [(0.4, DOWN, UP), (0.3, DOWN, DOWN)], self.t)
        self.assertEqual(kinds(events), [CurlKind.DOT])

    def test_long_curl_is_a_dash_fired_while_curled(self):
        events, t = play(self.d, [(1.2, UP, DOWN)], self.t)
        self.assertEqual(kinds(events), [CurlKind.DASH])
        self.assertAlmostEqual(events[0].fired - events[0].started, 0.75, delta=0.05)
        events, _ = play(self.d, [(0.3, DOWN, DOWN)], t)
        self.assertEqual(events, [])

    def test_either_arm_works(self):
        events, _ = play(self.d, [(0.3, UP, DOWN), (0.3, DOWN, DOWN),
                                  (0.3, DOWN, UP), (0.3, DOWN, DOWN)], self.t)
        self.assertEqual(kinds(events), [CurlKind.DOT, CurlKind.DOT])

    def test_shaky_arm_near_threshold_is_one_curl(self):
        script = [(0.1, DOWN, 0.55), (0.1, DOWN, 0.46), (0.1, DOWN, 0.53),
                  (0.1, DOWN, 0.47), (0.3, DOWN, DOWN)]
        events, _ = play(self.d, script, self.t)
        self.assertEqual(kinds(events), [CurlKind.DOT])

    def test_hidden_arm_counts_as_none(self):
        events, _ = play(self.d, [(0.3, None, UP), (0.3, None, DOWN)], self.t)
        self.assertEqual(kinds(events), [CurlKind.DOT])

    def test_pushup_letter_after_3t_and_space_after_5t(self):
        s = PushupSettings(time_unit=0.5)
        self.assertEqual((s.dash_after, s.letter_gap, s.word_gap), (0.75, 1.5, 2.5))
        d = CurlDetector(s.down_threshold, s.time_unit)
        composer = MorseComposer(s.letter_gap, s.word_gap)
        _, t = play(d, [(0.5, None, DOWN)])
        # quick down (dot), up, long down (dash), then stay up 1.2 s
        _, t = play(d, [(0.3, None, UP), (0.3, None, DOWN), (1.0, None, UP),
                        (1.2, None, DOWN)], t, composer)
        self.assertEqual(composer.code, ".-")          # 1.2 s < 1.5 s
        _, t = play(d, [(0.5, None, DOWN)], t, composer)
        self.assertEqual(composer.text, "A")
        _, t = play(d, [(1.0, None, DOWN)], t, composer)
        self.assertEqual(composer.text, "A ")


class PushupDepthTest(unittest.TestCase):
    def test_depth_from_elbow_angle(self):
        from blink_morse.arm_tracker import pushup_depth
        self.assertEqual(pushup_depth(175), 0.0)      # arms straight, up
        self.assertEqual(pushup_depth(70), 1.0)       # chest down
        self.assertAlmostEqual(pushup_depth(120), 0.5)


class GateTest(unittest.TestCase):
    @staticmethod
    def hand(ratio):
        """Synthetic hand: knuckles 1 unit from the wrist, tips `ratio` units."""
        import numpy as np
        pts = np.zeros((21, 3))
        for k, (knuckle, tip) in enumerate(((5, 8), (9, 12), (13, 16), (17, 20))):
            x = k * 0.2
            pts[knuckle] = (x, -1.0, 0)
            pts[tip] = (x * ratio, -ratio, 0)
        return pts

    def test_fist_and_open(self):
        from blink_morse.hand_tracker import FistGate
        g = FistGate()
        self.assertFalse(g.update(self.hand(1.9)))
        self.assertTrue(g.update(self.hand(1.0)))
        self.assertTrue(g.update(self.hand(1.4)))      # hysteresis keeps it on
        self.assertFalse(g.update(self.hand(1.8)))
        self.assertFalse(g.update(None))

    def test_cancel_needs_arm_down_first(self):
        d = CurlDetector(0.5, 0.5)
        _, t = play(d, [(0.3, None, DOWN)])
        d.cancel(t)                                   # hand opened
        events, t = play(d, [(0.3, None, UP), (0.3, None, DOWN)], t)
        self.assertEqual(events, [])                  # arm was already up
        events, _ = play(d, [(0.3, None, UP), (0.3, None, DOWN)], t)
        self.assertEqual(kinds(events), [CurlKind.DOT])


class PickHandTest(unittest.TestCase):
    @staticmethod
    def hand_at(x, y):
        import numpy as np
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[:, 0], pts[:, 1] = x, y
        return pts

    def test_hand_on_typing_arm_is_never_the_switch(self):
        from blink_morse.hand_tracker import pick_switch_hand
        typing_hand = self.hand_at(0.45, 0.35)
        # Only the typing arm's own hand is in view -> no switch hand.
        self.assertIsNone(pick_switch_hand([typing_hand], None, (0.46, 0.36)))
        self.assertIsNone(pick_switch_hand([typing_hand], (0.2, 0.9), (0.46, 0.36)))

    def test_switch_hand_matched_to_its_wrist(self):
        from blink_morse.hand_tracker import pick_switch_hand
        typing_hand, switch_hand = self.hand_at(0.7, 0.4), self.hand_at(0.25, 0.5)
        got = pick_switch_hand([typing_hand, switch_hand], (0.26, 0.52), (0.7, 0.41))
        self.assertIs(got, switch_hand)
        got = pick_switch_hand([typing_hand, switch_hand], None, (0.7, 0.41))
        self.assertIs(got, switch_hand)


if __name__ == "__main__":
    unittest.main()
