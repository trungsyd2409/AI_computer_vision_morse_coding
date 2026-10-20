"""Tests for hip sway as Morse input, and for the body landmark mapping."""

import json
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

from blink_morse.body_tracker import BONES, build_body
from blink_morse.config import AppSettings, BodySettings
from blink_morse.hip_gestures import HipKey, hip_tilt
from blink_morse.morse import MorseComposer

FPS = 30.0
WAIT, RIGHT, LEFT = 0.0, 9.0, -9.0             # well past the default A = 5 deg


def play(key, script, t=0.0, active=True):
    """script: list of (seconds, angle). Returns (events, t)."""
    out = []
    for seconds, angle in script:
        for _ in range(max(1, round(seconds * FPS))):
            out.extend(key.update(angle, t, active))
            t += 1.0 / FPS
    return out, t


class HipTiltTest(unittest.TestCase):
    def test_right_hip_higher_is_positive(self):
        # image y grows downwards; user's right hip is the second argument
        self.assertAlmostEqual(hip_tilt((100, 100), (200, 100)), 0.0)
        self.assertAlmostEqual(hip_tilt((100, 100), (200, 0)), 45.0)
        self.assertAlmostEqual(hip_tilt((100, 100), (200, 200)), -45.0)

    def test_same_without_mirror(self):
        # Not mirrored: the user's right hip is on the left of the image.
        self.assertAlmostEqual(hip_tilt((200, 100), (100, 0)), 45.0)
        self.assertAlmostEqual(hip_tilt((200, 100), (100, 200)), -45.0)

    def test_turned_sideways_gives_none(self):
        self.assertIsNone(hip_tilt((100, 100), (100.2, 40), min_width=5))


class HipKeyTest(unittest.TestCase):
    def setUp(self):
        self.k = HipKey()            # A = 5 deg, margin 1, t = 0.6 s, no smoothing

    def test_sway_fires_at_once_and_only_once(self):
        got, _ = play(self.k, [(0.3, WAIT), (1.0, RIGHT)])
        self.assertEqual(got, ["right"])
        self.assertEqual(self.k.latched, "right")

    def test_fires_on_the_frame_it_crosses(self):
        play(self.k, [(0.2, WAIT)])
        self.assertEqual(self.k.update(5.0, 1.0), ["right"])
        k = HipKey()
        play(k, [(0.2, WAIT)])
        self.assertEqual(k.update(-5.0, 1.0), ["left"])

    def test_just_inside_the_angle_is_still_waiting(self):
        got, _ = play(self.k, [(0.1, WAIT), (0.2, 4.9), (0.2, -4.9)])
        self.assertEqual(got, [])
        self.assertEqual(self.k.zone, "waiting")

    def test_right_and_left(self):
        got, _ = play(self.k, [(0.2, WAIT), (0.3, RIGHT), (0.2, WAIT), (0.3, LEFT), (0.2, WAIT)])
        self.assertEqual(got, ["right", "left"])

    def test_must_return_to_waiting_before_the_next_one(self):
        got, _ = play(self.k, [(0.2, WAIT), (0.3, RIGHT), (0.2, 5.5), (0.3, RIGHT)])
        self.assertEqual(got, ["right"])                  # never left the sway
        got, _ = play(self.k, [(0.2, 3.5), (0.3, RIGHT)], 5.0)
        self.assertEqual(got, ["right"])                  # 3.5 is below A - margin = 4

    def test_edge_noise_does_not_type_twice(self):
        # Hovering around 5 deg: 4.5 is inside the 1 deg margin, so still recorded.
        got, _ = play(self.k, [(0.2, WAIT), (0.2, 5.5), (0.1, 4.5), (0.1, 5.2), (0.1, 4.2)])
        self.assertEqual(got, ["right"])

    def test_swing_across_in_one_go(self):
        got, _ = play(self.k, [(0.2, WAIT), (0.3, RIGHT), (0.3, LEFT)])
        self.assertEqual(got, ["right", "left"])

    def test_end_after_t_and_space_after_3t(self):
        got, _ = play(self.k, [(0.3, RIGHT), (0.3, WAIT)])
        self.assertEqual(got, [])                         # first sample inside a sway: silent
        got, _ = play(self.k, [(0.9, WAIT)], 1.0)
        self.assertEqual(got, ["end"])                    # 0.6 s passed, 1.8 s not yet
        got, _ = play(self.k, [(1.5, WAIT)], 2.0)
        self.assertEqual(got, ["space"])
        got, _ = play(self.k, [(3.0, WAIT)], 4.0)
        self.assertEqual(got, [])                         # once per waiting period

    def test_sway_before_t_types_the_symbol_and_no_end(self):
        got, _ = play(self.k, [(0.2, WAIT), (0.4, WAIT), (0.2, RIGHT)])
        self.assertEqual(got, ["right"])

    def test_sway_after_t_still_types(self):
        got, _ = play(self.k, [(0.2, WAIT), (0.8, WAIT), (0.2, LEFT)])
        self.assertEqual(got, ["end", "left"])

    def test_timer_restarts_when_waiting_starts_again(self):
        got, _ = play(self.k, [(0.1, WAIT), (0.4, WAIT), (0.2, RIGHT), (0.5, WAIT)])
        self.assertEqual(got, ["right"])                  # 0.5 s waiting twice, never 0.6
        self.assertAlmostEqual(self.k.wait_for, 0.5, delta=0.05)

    def test_pose_held_when_typing_starts_types_nothing(self):
        got, t = play(self.k, [(0.5, RIGHT)], active=False)
        self.assertEqual(got, [])
        self.assertEqual(self.k.zone, "right")            # still shown on the HUD
        got, t = play(self.k, [(0.5, RIGHT)], t)
        self.assertEqual(got, [])                         # primed silently
        got, t = play(self.k, [(0.3, WAIT), (0.3, RIGHT)], t)
        self.assertEqual(got, ["right"])

    def test_inactive_keeps_no_timer(self):
        play(self.k, [(2.0, WAIT)], active=False)
        self.assertEqual(self.k.wait_for, 0.0)
        got, _ = play(self.k, [(0.3, WAIT)], 3.0)
        self.assertEqual(got, [])                         # timer starts when typing starts

    def test_lost_body_resets(self):
        play(self.k, [(0.2, WAIT), (0.3, RIGHT)])
        self.assertEqual(self.k.update(None, 5.0), [])
        self.assertIsNone(self.k.angle)
        self.assertIsNone(self.k.latched)

    def test_zone_names(self):
        play(self.k, [(0.3, WAIT)])
        self.assertEqual(self.k.zone, "waiting")
        play(self.k, [(0.3, LEFT)], 1.0)
        self.assertEqual(self.k.zone, "left")
        play(self.k, [(0.5, -4.5)], 2.0)
        self.assertEqual(self.k.zone, "left")              # inside the margin: still recorded
        play(self.k, [(0.5, -3.0)], 3.0)
        self.assertEqual(self.k.zone, "waiting")

    def test_optional_smoothing(self):
        k = HipKey(smooth=0.5)
        _, t = play(k, [(0.2, WAIT)])
        self.assertEqual(k.update(8.0, t), [])             # 4 after smoothing
        self.assertEqual(k.update(8.0, t + 0.03), ["right"])   # 6


class SpellingTest(unittest.TestCase):
    def test_spell_a_word(self):
        """A = dot dash, D = dash dot dot, then a long wait for the space"""
        k, c = HipKey(), MorseComposer()
        symbols = {"left": ".", "right": "-"}              # angle > A = dash, < -A = dot

        def feed(script, t):
            events, t = play(k, script, t)
            for e in events:
                if e in symbols:
                    c.add_symbol(symbols[e])
                elif e == "end":
                    c.end_letter()
                else:
                    c.add_space()
            return t

        t = feed([(0.3, WAIT)], 0.0)
        t = feed([(0.25, LEFT), (0.25, WAIT), (0.25, RIGHT), (0.9, WAIT)], t)
        self.assertEqual(c.text, "A")
        t = feed([(0.25, RIGHT), (0.25, WAIT), (0.25, LEFT), (0.25, WAIT),
                  (0.25, LEFT), (2.5, WAIT)], t)
        self.assertEqual(c.text, "AD ")
        self.assertEqual(c.code, "")


LM = namedtuple("LM", "x y visibility")


def landmarks(**named):
    """33 hidden landmarks, with the listed indices filled in."""
    lms = [LM(0.5, 0.5, 0.0)] * 33
    for index, (x, y) in named.items():
        lms[int(index[1:])] = LM(x, y, 1.0)
    return lms


class BuildBodyTest(unittest.TestCase):
    # MediaPipe left hip (23) high on the image, right hip (24) lower.
    lms = landmarks(i23=(0.6, 0.50), i24=(0.4, 0.55), i25=(0.6, 0.7), i11=(0.6, 0.3),
                    i12=(0.4, 0.3), i15=(0.7, 0.5), i16=(0.3, 0.5))

    def test_mirrored_image_swaps_the_sides(self):
        body = build_body(self.lms, (640, 480), mirrored=True)
        # MediaPipe's left body is the user's right when the frame is mirrored.
        self.assertEqual(body.points["right_hip"], (0.6, 0.50))
        self.assertEqual(body.points["left_hip"], (0.4, 0.55))
        self.assertIn("right_knee", body.points)
        self.assertEqual(body.wrists["right"], (0.7, 0.5))
        # user's right hip is the higher one -> positive
        self.assertGreater(body.hip_angle, 0)

    def test_not_mirrored_keeps_the_sides(self):
        body = build_body(self.lms, (640, 480), mirrored=False)
        self.assertEqual(body.points["left_hip"], (0.6, 0.50))
        self.assertLess(body.hip_angle, 0)

    def test_swap_option_flips_back(self):
        a = build_body(self.lms, (640, 480), mirrored=True, swap_sides=True)
        b = build_body(self.lms, (640, 480), mirrored=False)
        self.assertEqual(a.points, b.points)

    def test_angle_uses_pixels(self):
        lms = landmarks(i23=(0.6, 0.5), i24=(0.4, 0.5 - 0.1))      # right hip 48 px up
        body = build_body(lms, (640, 480), mirrored=False)
        # not mirrored: user's right hip is MediaPipe's right (index 24), at x = 0.4
        self.assertAlmostEqual(body.hip_angle, 20.56, delta=0.05)    # atan(48 / 128 px), not atan(0.1 / 0.2)

    def test_hidden_hip_gives_no_angle(self):
        body = build_body(landmarks(i23=(0.6, 0.5)), (640, 480), mirrored=True)
        self.assertIsNone(body.hip_angle)

    def test_bones_only_use_known_joints(self):
        names = {f"{s}_{j}" for s in ("left", "right")
                 for j in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "toe")}
        for a, b in BONES:
            self.assertIn(a, names)
            self.assertIn(b, names)


class SettingsTest(unittest.TestCase):
    def test_symbol_roles(self):
        s = BodySettings()
        self.assertEqual((s.up_symbol, s.down_symbol), ("dash", "dot"))
        s.up_symbol = "dot"
        self.assertEqual(s.down_symbol, "dash")

    def test_old_arm_settings_file_still_loads(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "settings.json"
            path.write_text(json.dumps({"camera": {"index": 2},
                                        "arm": {"swap_arms": True, "together_ratio": 0.5,
                                                "curl_side": "right"}}))
            s = AppSettings.load(path)
        self.assertEqual(s.camera.index, 2)
        self.assertTrue(s.body.swap_arms)
        self.assertAlmostEqual(s.body.together_ratio, 0.5)
        self.assertEqual(s.body.trigger_angle, 5.0)

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "settings.json"
            s = AppSettings()
            s.body.trigger_angle = 11.0
            s.body.up_symbol = "dot"
            s.save(path)
            back = AppSettings.load(path)
        self.assertEqual(back.body.trigger_angle, 11.0)
        self.assertEqual(back.body.up_symbol, "dot")


if __name__ == "__main__":
    unittest.main()
