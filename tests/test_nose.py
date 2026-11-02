"""Tests for the nose feature fusion and the press detector, without a camera."""

import math
import unittest

import numpy as np

from blink_morse.nose import (DARK_INDEX, FEATURES, N_FEATURES, WIDTH_PAIRS, NoseSignal,
                              extract_features, fit_calibration)
from blink_morse.press import PressDetector, PressKind

FPS = 30.0
REST = np.array([1.30, 1.17, 1.20, 1.10, 0.02, 0.20])
# How far each feature moves during a flare, in its own units. "alar" and
# "dark" do not react at all for this simulated person.
FLARE_SHIFT = np.array([0.04, 0.0, 0.03, 0.025, 0.15, 0.0])
NOISE = np.array([0.006, 0.006, 0.006, 0.006, 0.02, 0.01])
WING = FEATURES.index("wing")


def sample(rng, flare: float = 0.0) -> np.ndarray:
    return REST + FLARE_SHIFT * flare + rng.normal(0.0, NOISE)


def run_calibration(nose: NoseSignal, rng, react: float = 0.25, t: float = 0.0) -> float:
    """Follow the prompts like a user would, flaring `react` s after the cue."""
    step_start, last_step = t, None
    while nose.calibrating:
        step = nose.step
        if step != last_step:
            step_start, last_step = t, step
        flaring = step == "flare" and t - step_start >= react
        nose.update(sample(rng, 1.0 if flaring else 0.0), t)
        t += 1.0 / FPS
    return t


def play(nose: NoseSignal, rng, seconds: float, flare: float, t: float, **kw):
    scores = []
    for _ in range(round(seconds * FPS)):
        scores.append(nose.update(sample(rng, flare), t, **kw).score)
        t += 1.0 / FPS
    return scores, t


class FitCalibrationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.rest = np.array([sample(rng) for _ in range(150)])
        self.flare = np.array([sample(rng, 1.0) for _ in range(90)])

    def test_average_flare_scores_one(self):
        cal = fit_calibration(self.rest, self.flare)
        self.assertFalse(cal.fallback)
        z = lambda x: (x - cal.mean) / cal.std
        self.assertAlmostEqual(float(np.mean(z(self.flare) @ cal.weights)), 1.0, delta=0.05)
        self.assertAlmostEqual(float(np.mean(z(self.rest) @ cal.weights)), 0.0, delta=0.05)

    def test_feature_that_never_moves_gets_little_weight(self):
        cal = fit_calibration(self.rest, self.flare)
        dark = abs(cal.weights[DARK_INDEX])
        self.assertLess(dark, 0.25 * abs(cal.weights[WING]))
        self.assertGreater(cal.quality, 5.0)

    def test_no_flare_seen_falls_back(self):
        rng = np.random.default_rng(2)
        still = np.array([sample(rng) for _ in range(90)])
        cal = fit_calibration(self.rest, still)
        self.assertTrue(cal.fallback)
        self.assertGreater(cal.weights[WING], 0)

    def test_disabled_feature_has_zero_weight(self):
        mask = np.ones(N_FEATURES, dtype=bool)
        mask[WING] = False
        cal = fit_calibration(self.rest, self.flare, mask)
        self.assertEqual(cal.weights[WING], 0.0)

    def test_missing_feature_is_ignored(self):
        rest, flare = self.rest.copy(), self.flare.copy()
        rest[:, DARK_INDEX] = np.nan
        flare[:, DARK_INDEX] = np.nan
        cal = fit_calibration(rest, flare)
        self.assertEqual(cal.weights[DARK_INDEX], 0.0)
        self.assertTrue(np.all(np.isfinite(cal.weights)))


class NoseSignalTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.nose = NoseSignal(smile_gate=0.35, threshold=0.5)
        self.t = run_calibration(self.nose, self.rng)

    def test_calibration_finishes_with_good_quality(self):
        cal = self.nose.calibration
        self.assertIsNotNone(cal)
        self.assertFalse(cal.fallback)
        self.assertGreater(cal.quality, 3.0)

    def test_rest_and_flare_scores(self):
        rest, t = play(self.nose, self.rng, 1.0, 0.0, self.t)
        flare, t = play(self.nose, self.rng, 1.0, 1.0, t)
        self.assertLess(np.mean(rest[10:]), 0.2)
        self.assertGreater(np.mean(flare[10:]), 0.7)

    def test_smile_is_ignored(self):
        scores, _ = play(self.nose, self.rng, 1.0, 1.0, self.t, smile=0.8)
        self.assertEqual(max(scores), 0.0)

    def test_turned_head_is_ignored(self):
        scores, _ = play(self.nose, self.rng, 1.0, 1.0, self.t, yaw=35.0)
        self.assertEqual(max(scores), 0.0)

    def test_slow_drift_is_followed(self):
        # The resting nose looks 1 % wider after the camera moved a little.
        global REST
        old = REST.copy()
        try:
            REST = old * np.array([1.01, 1.01, 1.01, 1.01, 1.0, 1.0])
            scores, _ = play(self.nose, self.rng, 20.0, 0.0, self.t)
        finally:
            REST = old
        self.assertLess(np.mean(scores[-60:]), 0.2)

    def test_stuck_flare_is_recentred(self):
        res = None
        t = self.t
        for _ in range(round(6 * FPS)):
            res = self.nose.update(sample(self.rng, 1.0), t)
            t += 1 / FPS
            if res.recentred:
                break
        self.assertTrue(res.recentred)

    def test_recalibrate_starts_with_relax_prompt(self):
        self.nose.start_calibration()
        res = self.nose.update(sample(self.rng), self.t)
        self.assertTrue(res.calibrating)
        self.assertEqual(res.step, "relax")
        self.assertIn("relax", res.events)


class ExtractFeaturesTest(unittest.TestCase):
    def test_widths_are_relative_to_the_eyes(self):
        pts = np.zeros((478, 2))
        pts[133], pts[362] = (100, 100), (160, 100)          # 60 px apart
        for a, b in WIDTH_PAIRS.values():
            pts[a], pts[b] = (100, 150), (178, 150)          # 78 px apart
        f = extract_features(pts, {"noseSneerLeft": 0.2, "noseSneerRight": 0.4})
        self.assertAlmostEqual(f[WING], 1.3)
        self.assertAlmostEqual(f[FEATURES.index("sneer")], 0.3)
        self.assertTrue(math.isnan(f[DARK_INDEX]))           # no frame given

    def test_tiny_face_gives_nan(self):
        pts = np.zeros((478, 2))
        pts[362] = (3, 0)
        self.assertTrue(np.all(np.isnan(extract_features(pts, {}))))


class PressDetectorTest(unittest.TestCase):
    """t = 0.25 s, so a dash starts at 0.375 s of flare."""

    def setUp(self):
        self.d = PressDetector(threshold=0.5, time_unit=0.25)
        self.t = 0.0
        self.feed(0.5, 0.0)

    def feed(self, seconds, score):
        events = []
        for _ in range(max(1, round(seconds * FPS))):
            events.extend(self.d.update(score, self.t))
            self.t += 1 / FPS
        return [e.kind for e in events]

    def test_short_flare_is_a_dot(self):
        kinds = self.feed(0.2, 1.0) + self.feed(0.3, 0.0)
        self.assertEqual(kinds, [PressKind.DOT])

    def test_long_flare_is_a_dash_while_still_flared(self):
        kinds = self.feed(0.5, 1.0)
        self.assertEqual(kinds, [PressKind.DASH])
        self.assertTrue(self.d.pressed)
        self.assertEqual(self.feed(0.3, 0.0), [])

    def test_hysteresis_keeps_one_press(self):
        # Score dips to 0.4: below the threshold but above the release level.
        kinds = self.feed(0.1, 0.9) + self.feed(0.1, 0.4) + self.feed(0.1, 0.9) \
            + self.feed(0.3, 0.0)
        self.assertEqual(len(kinds), 1)

    def test_slow_fade_after_a_short_flare_stays_a_dot(self):
        # Score falls back slowly through the band between the release
        # level (0.35) and the threshold (0.5), well past 1.5 t in total.
        kinds = self.feed(0.2, 1.0) + self.feed(0.4, 0.42) + self.feed(0.3, 0.0)
        self.assertEqual(kinds, [PressKind.DOT])

    def test_pause_time_after_release(self):
        self.feed(0.2, 1.0)
        self.feed(0.6, 0.0)
        self.assertAlmostEqual(self.d.pause_time(self.t), 0.6, delta=0.06)


if __name__ == "__main__":
    unittest.main()
