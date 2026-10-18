"""Tests for the forearm-tilt gestures and the gesture-driven Morse flow."""

import unittest

import numpy as np

from blink_morse.arm_gestures import TiltKey, forearm_angle
from blink_morse.config import ArmSettings
from blink_morse.hand_tracker import FistGate, assign_hands
from blink_morse.morse import MorseComposer

FPS = 30.0
LEVEL, UP, DOWN, HANG = 0.0, 45.0, -45.0, -88.0


def play(key, script, t=0.0):
    """script: list of (seconds, angle). Returns (gestures, t)."""
    out = []
    for seconds, angle in script:
        for _ in range(max(1, round(seconds * FPS))):
            g = key.update(angle, t)
            if g:
                out.append(g)
            t += 1.0 / FPS
    return out, t


class ForearmAngleTest(unittest.TestCase):
    def test_signs_match_the_screen(self):
        # y grows downwards on the image
        self.assertAlmostEqual(forearm_angle((100, 100), (200, 100)), 0.0)
        self.assertAlmostEqual(forearm_angle((100, 100), (200, 0)), 45.0)
        self.assertAlmostEqual(forearm_angle((100, 100), (200, 200)), -45.0)

    def test_same_for_both_directions(self):
        # A forearm pointing left (other arm) reads the same as one pointing right.
        self.assertAlmostEqual(forearm_angle((200, 100), (100, 0)), 45.0)
        self.assertAlmostEqual(forearm_angle((200, 100), (100, 200)), -45.0)


class TiltKeyTest(unittest.TestCase):
    def setUp(self):
        self.k = TiltKey(45, 20, 15, 0.10)

    def test_needs_level_first(self):
        got, _ = play(self.k, [(0.5, UP)])
        self.assertEqual(got, [])

    def test_up_and_down(self):
        got, _ = play(self.k, [(0.3, LEVEL), (0.3, UP), (0.3, LEVEL), (0.3, DOWN)])
        self.assertEqual(got, ["up", "down"])

    def test_holding_fires_once(self):
        got, _ = play(self.k, [(0.3, LEVEL), (2.0, UP)])
        self.assertEqual(got, ["up"])

    def test_two_dots_need_level_between(self):
        got, _ = play(self.k, [(0.3, LEVEL), (0.3, DOWN), (0.3, -70), (0.3, DOWN)])
        self.assertEqual(got, ["down"])
        got, _ = play(self.k, [(0.3, LEVEL), (0.3, DOWN)], 5.0)
        self.assertEqual(got, ["down"])

    def test_sweeping_past_does_not_type(self):
        # Dropping the arm from level to hanging passes -45 in one frame.
        got, _ = play(self.k, [(0.3, LEVEL), (1 / FPS, DOWN), (0.5, HANG)])
        self.assertEqual(got, [])

    def test_lost_arm_resets(self):
        got, _ = play(self.k, [(0.3, LEVEL), (0.2, None), (0.3, UP)])
        self.assertEqual(got, [])


class GestureFlowTest(unittest.TestCase):
    def test_spell_a_word(self):
        """dot dash, end letter, dash dot dot, space -> 'AD '"""
        sym, ctl = TiltKey(), TiltKey()
        c = MorseComposer()
        t = 0.0
        for key, angle in [(sym, LEVEL), (ctl, LEVEL),
                           (sym, DOWN), (sym, LEVEL), (sym, UP), (sym, LEVEL),
                           (ctl, UP), (ctl, LEVEL),
                           (sym, UP), (sym, LEVEL), (sym, DOWN), (sym, LEVEL),
                           (sym, DOWN), (sym, LEVEL), (ctl, DOWN)]:
            got, t = play(key, [(0.2, angle)], t)
            for g in got:
                if key is sym:
                    c.add_symbol("-" if g == "up" else ".")
                elif g == "up":
                    c.end_letter()
                else:
                    c.add_space()
        self.assertEqual(c.text, "AD ")
        self.assertEqual(c.code, "")

    def test_settings_roles(self):
        s = ArmSettings()
        self.assertEqual((s.symbol_side, s.control_side), ("left", "right"))
        s.symbol_side = "right"
        self.assertEqual(s.control_side, "left")


class HandsTest(unittest.TestCase):
    @staticmethod
    def hand_at(x, y):
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[:, 0], pts[:, 1] = x, y
        return pts

    def test_hands_go_to_nearest_wrist(self):
        a, b = self.hand_at(0.2, 0.5), self.hand_at(0.8, 0.5)
        got = assign_hands([b, a], {"left": (0.22, 0.5), "right": (0.78, 0.5)})
        self.assertIs(got["left"], a)
        self.assertIs(got["right"], b)

    def test_far_hand_is_ignored(self):
        got = assign_hands([self.hand_at(0.5, 0.1)], {"left": (0.2, 0.5), "right": None})
        self.assertEqual(got, {"left": None, "right": None})

    def test_fist_survives_short_dropout(self):
        fist = np.zeros((21, 3))
        fist[[5, 9, 13, 17], 1] = 10
        fist[[8, 12, 16, 20], 1] = 11      # tips folded back near the knuckles
        g = FistGate()
        self.assertTrue(g.update(fist, 0.0))
        self.assertTrue(g.update(None, 0.2))
        self.assertFalse(g.update(None, 0.5))


if __name__ == "__main__":
    unittest.main()
