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
    def setUp(self):
        self.d = BlinkDetector()
        _, self.t = play(self.d, [(1.0, OPEN, OPEN)])     # settle the baselines

    def test_both_eyes_type_a_dash_on_the_first_closed_frame(self):
        events, _ = play(self.d, [(0.15, SHUT, SHUT), (0.3, OPEN, OPEN)], self.t)
        self.assertEqual(kinds(events), [BlinkKind.DASH])
        self.assertAlmostEqual(events[0].fired, events[0].started)   # no waiting

    def test_right_wink_types_a_dot_within_a_few_frames(self):
        events, _ = play(self.d, [(0.3, OPEN, SHUT), (0.3, OPEN, OPEN)], self.t)
        self.assertEqual(kinds(events), [BlinkKind.DOT])
        self.assertLess(events[0].fired - events[0].started, 0.1)

    def test_left_wink_ends_the_letter(self):
        events, _ = play(self.d, [(0.3, SHUT, OPEN), (0.3, OPEN, OPEN)], self.t)
        self.assertEqual(kinds(events), [BlinkKind.END])
        self.assertLess(events[0].fired - events[0].started, 0.1)

    def test_blink_led_by_left_eye_is_still_a_dash(self):
        script = [(1 / FPS, SHUT, OPEN), (0.12, SHUT, SHUT), (0.3, OPEN, OPEN)]
        events, _ = play(self.d, script, self.t)
        self.assertEqual(kinds(events), [BlinkKind.DASH])

    def test_blink_led_by_right_eye_is_still_a_dash(self):
        # The right eye closes one frame before the left one.
        script = [(1 / FPS, OPEN, SHUT), (0.12, SHUT, SHUT), (0.3, OPEN, OPEN)]
        events, _ = play(self.d, script, self.t)
        self.assertEqual(kinds(events), [BlinkKind.DASH])

    def test_left_eye_on_its_way_down_blocks_the_dot(self):
        # Left eye half closed (not yet past the threshold) while the right
        # is shut: this is the start of a blink, not a wink.
        script = [(2 / FPS, 0.30, SHUT), (0.1, SHUT, SHUT), (0.3, OPEN, OPEN)]
        events, _ = play(self.d, script, self.t)
        self.assertEqual(kinds(events), [BlinkKind.DASH])

    def test_wink_with_squinting_left_eye_still_counts(self):
        events, _ = play(self.d, [(0.4, 0.55, 0.95), (0.3, OPEN, OPEN)], self.t)
        self.assertEqual(kinds(events), [BlinkKind.DOT])

    def test_long_close_types_only_one_dash(self):
        events, _ = play(self.d, [(1.0, SHUT, SHUT), (0.3, OPEN, OPEN)], self.t)
        self.assertEqual(kinds(events), [BlinkKind.DASH])

    def test_one_noisy_open_frame_does_not_split_a_blink(self):
        script = [(0.1, SHUT, SHUT), (1 / FPS, OPEN, OPEN), (0.1, SHUT, SHUT),
                  (0.3, OPEN, OPEN)]
        events, _ = play(self.d, script, self.t)
        self.assertEqual(kinds(events), [BlinkKind.DASH])

    def test_fast_sequence(self):
        script = [(0.1, OPEN, SHUT), (0.15, OPEN, OPEN),     # dot
                  (0.1, SHUT, SHUT), (0.15, OPEN, OPEN),     # dash
                  (0.1, OPEN, SHUT), (0.15, OPEN, OPEN),     # dot
                  (0.1, SHUT, OPEN), (0.3, OPEN, OPEN)]      # end of letter
        events, _ = play(self.d, script, self.t)
        self.assertEqual(kinds(events), [BlinkKind.DOT, BlinkKind.DASH,
                                         BlinkKind.DOT, BlinkKind.END])

    def test_blink_filter_ignores_very_short_blinks(self):
        self.d.blink_filter = 0.1
        events, _ = play(self.d, [(0.066, SHUT, SHUT), (0.3, OPEN, OPEN)], self.t)
        self.assertEqual(events, [])
        events, _ = play(self.d, [(0.2, SHUT, SHUT), (0.3, OPEN, OPEN)], self.t + 1)
        self.assertEqual(kinds(events), [BlinkKind.DASH])

    def test_pause_time_counts_from_when_the_eyes_open(self):
        _, t = play(self.d, [(0.3, SHUT, SHUT)], self.t)
        self.assertEqual(self.d.pause_time(t), 0.0)      # still closed
        _, t = play(self.d, [(0.5, OPEN, OPEN)], t)
        self.assertAlmostEqual(self.d.pause_time(t), 0.5, delta=0.05)


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
