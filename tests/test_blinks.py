"""Tests for the blink detector using synthetic score sequences."""

import unittest

from blink_morse.blinks import BlinkDetector, BlinkKind, EyeSignal
from blink_morse.face_tracker import build_result

FPS = 30.0
OPEN, SHUT, SQUINT = 0.05, 0.85, 0.30


def play(detector, script, start=0.0):
    """
    script: list of (seconds, left_score, right_score) segments.
    Returns (events, end time).
    """
    events, t = [], start
    for seconds, left, right in script:
        for _ in range(max(1, round(seconds * FPS))):
            events.extend(detector.update(left, right, t))
            t += 1.0 / FPS
    return events, t


def kinds(events):
    return [e.kind for e in events]


class BlinkDetectorTest(unittest.TestCase):
    """Default time unit t = 0.1 s, so a dash starts at 0.15 s of closure."""

    def setUp(self):
        self.d = BlinkDetector(time_unit=0.1)
        _, self.t = play(self.d, [(1.0, OPEN, OPEN)])     # settle the baselines

    def test_short_blink_is_a_dot(self):
        events, _ = play(self.d, [(0.07, SHUT, SHUT), (0.2, OPEN, OPEN)], self.t)
        self.assertEqual(kinds(events), [BlinkKind.DOT])

    def test_long_blink_is_a_dash(self):
        events, _ = play(self.d, [(0.3, SHUT, SHUT), (0.2, OPEN, OPEN)], self.t)
        self.assertEqual(kinds(events), [BlinkKind.DASH])

    def test_dash_fires_while_eyes_are_still_closed(self):
        events, t = play(self.d, [(0.5, SHUT, SHUT)], self.t)
        self.assertEqual(kinds(events), [BlinkKind.DASH])
        self.assertAlmostEqual(events[0].fired - events[0].started, 0.15, delta=0.04)
        self.assertTrue(self.d.closed)
        events, _ = play(self.d, [(0.2, OPEN, OPEN)], t)
        self.assertEqual(events, [])          # no second symbol on opening

    def test_split_point_scales_with_time_unit(self):
        self.d.time_unit = 0.2                # dash from 0.3 s
        events, _ = play(self.d, [(0.2, SHUT, SHUT), (0.2, OPEN, OPEN)], self.t)
        self.assertEqual(kinds(events), [BlinkKind.DOT])

    def test_single_eye_is_ignored(self):
        events, _ = play(self.d, [(0.3, SHUT, OPEN), (0.2, OPEN, OPEN),
                                  (0.3, OPEN, SHUT), (0.2, OPEN, OPEN)], self.t)
        self.assertEqual(events, [])

    def test_one_noisy_open_frame_does_not_split_a_blink(self):
        script = [(0.1, SHUT, SHUT), (1 / FPS, OPEN, OPEN), (0.1, SHUT, SHUT),
                  (0.3, OPEN, OPEN)]
        events, _ = play(self.d, script, self.t)
        self.assertEqual(kinds(events), [BlinkKind.DASH])

    def test_sequence(self):
        script = [(0.07, SHUT, SHUT), (0.15, OPEN, OPEN),
                  (0.3, SHUT, SHUT), (0.15, OPEN, OPEN),
                  (0.07, SHUT, SHUT), (0.3, OPEN, OPEN)]
        events, _ = play(self.d, script, self.t)
        self.assertEqual(kinds(events), [BlinkKind.DOT, BlinkKind.DASH, BlinkKind.DOT])

    def test_pause_time_counts_from_when_the_eyes_open(self):
        _, t = play(self.d, [(0.3, SHUT, SHUT)], self.t)
        self.assertEqual(self.d.pause_time(t), 0.0)      # still closed
        _, t = play(self.d, [(0.5, OPEN, OPEN)], t)
        self.assertAlmostEqual(self.d.pause_time(t), 0.5, delta=0.05)

    def test_reset_restarts_the_pause(self):
        _, t = play(self.d, [(0.1, SHUT, SHUT), (2.0, OPEN, OPEN)], self.t)
        self.d.reset(t)
        self.assertEqual(self.d.pause_time(t), 0.0)


class EyeSignalTest(unittest.TestCase):
    def test_learns_high_resting_level(self):
        """Someone whose open eyes read 0.35 must not look half closed."""
        s = EyeSignal()
        t = 0.0
        for _ in range(int(8 * FPS)):
            s.update(0.35, t, is_closed=False)
            t += 1 / FPS
        self.assertAlmostEqual(s.baseline, 0.35, places=2)
        self.assertLess(s.update(0.35, t, False), 0.05)
        self.assertGreater(s.update(0.9, t + 0.03, False), 0.8)


class EyeMappingTest(unittest.TestCase):
    def test_mirrored_frame_swaps_sides(self):
        r = build_result(mp_left=0.9, mp_right=0.1, mirrored=True)
        self.assertEqual((r.left_score, r.right_score), (0.1, 0.9))

    def test_corners_follow_the_scores(self):
        mp_left_corners, mp_right_corners = ("L", "l"), ("R", "r")
        r = build_result(0.9, 0.1, mirrored=True,
                         corners=(mp_left_corners, mp_right_corners))
        self.assertEqual(r.left_corners, mp_right_corners)
        self.assertEqual(r.right_corners, mp_left_corners)

    def test_plain_frame_keeps_sides(self):
        r = build_result(mp_left=0.9, mp_right=0.1, mirrored=False)
        self.assertEqual((r.left_score, r.right_score), (0.9, 0.1))

    def test_manual_swap(self):
        r = build_result(0.9, 0.1, mirrored=True, swap_eyes=True)
        self.assertEqual(r.left_score, 0.9)


if __name__ == "__main__":
    unittest.main()
