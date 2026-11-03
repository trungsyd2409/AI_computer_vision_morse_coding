"""Tests for the smile features, the feature fusion and the press detector, without a camera."""

import math
import unittest

import numpy as np

from blink_morse.fusion import fit_calibration
from blink_morse.press import PressDetector, PressKind
from blink_morse.smile import FEATURES, N_FEATURES, SmileSignal, extract_features

FPS = 30.0
SMILE, CHEEK, DIMPLE, WIDTH, LIFT = range(5)
REST = np.array([0.05, 0.02, 0.05, 1.60, -0.10])
# A closed-lip smile: the blendshape and the two distances move, the cheek
# and dimple scores do not react for this simulated person.
SMILE_SHIFT = np.array([0.35, 0.0, 0.0, 0.12, 0.08])
NOISE = np.array([0.03, 0.01, 0.02, 0.01, 0.008])
MIN_STD = np.array(SmileSignal.MIN_STD)
FALLBACK = np.array([1.0, 0, 0, 1.0, 1.0])


def sample(rng, smile: float = 0.0) -> np.ndarray:
    return REST + SMILE_SHIFT * smile + rng.normal(0.0, NOISE)


def run_calibration(sig: SmileSignal, rng, react: float = 0.25, t: float = 0.0) -> float:
    """Follow the prompts like a user would, smiling `react` s after the cue."""
    step_start, last_step = t, None
    while sig.calibrating:
        step = sig.step
        if step != last_step:
            step_start, last_step = t, step
        smiling = step == "smile" and t - step_start >= react
        sig.update(sample(rng, 1.0 if smiling else 0.0), t)
        t += 1.0 / FPS
    return t


def play(sig: SmileSignal, rng, seconds: float, smile: float, t: float, **kw):
    scores = []
    for _ in range(round(seconds * FPS)):
        scores.append(sig.update(sample(rng, smile), t, **kw).score)
        t += 1.0 / FPS
    return scores, t


class ExtractFeaturesTest(unittest.TestCase):
    def face(self, corner_y=200.0, corner_dx=40.0):
        pts = np.zeros((478, 2))
        pts[133], pts[362] = (100, 100), (160, 100)          # eyes 60 px apart
        pts[168], pts[152] = (130, 100), (130, 260)          # face points down
        pts[61], pts[291] = (130 - corner_dx, corner_y), (130 + corner_dx, corner_y)
        pts[0] = (130, 190)                                  # top of upper lip
        return pts

    def test_blendshapes_are_averaged(self):
        blend = {"mouthSmileLeft": 0.2, "mouthSmileRight": 0.4,
                 "cheekSquintLeft": 0.1, "cheekSquintRight": 0.3}
        f = extract_features(self.face(), blend)
        self.assertAlmostEqual(f[SMILE], 0.3)
        self.assertAlmostEqual(f[CHEEK], 0.2)
        self.assertEqual(f[DIMPLE], 0.0)

    def test_width_and_lift(self):
        relaxed = extract_features(self.face(200, 40), {})
        smiling = extract_features(self.face(192, 48), {})
        self.assertAlmostEqual(relaxed[WIDTH], 80 / 60)
        self.assertAlmostEqual(relaxed[LIFT], -10 / 60)
        self.assertGreater(smiling[WIDTH], relaxed[WIDTH])
        self.assertGreater(smiling[LIFT], relaxed[LIFT])

    def test_lift_ignores_head_roll(self):
        pts = self.face(192, 48)
        angle = math.radians(25)
        rot = np.array([[math.cos(angle), -math.sin(angle)],
                        [math.sin(angle), math.cos(angle)]])
        rolled = (pts - 130) @ rot.T + 130
        a, b = extract_features(pts, {}), extract_features(rolled, {})
        self.assertAlmostEqual(a[LIFT], b[LIFT], places=6)
        self.assertAlmostEqual(a[WIDTH], b[WIDTH], places=6)

    def test_tiny_face_keeps_blendshapes_only(self):
        pts = self.face()
        pts[362] = pts[133] + (3, 0)
        f = extract_features(pts, {"mouthSmileLeft": 0.5, "mouthSmileRight": 0.5})
        self.assertEqual(f[SMILE], 0.5)
        self.assertTrue(math.isnan(f[WIDTH]))


class FitCalibrationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.rest = np.array([sample(rng) for _ in range(150)])
        self.smile = np.array([sample(rng, 1.0) for _ in range(90)])

    def fit(self, active, enabled=None):
        return fit_calibration(self.rest, active, MIN_STD, FALLBACK, enabled)

    def test_average_smile_scores_one(self):
        cal = self.fit(self.smile)
        self.assertFalse(cal.fallback)
        z = lambda x: (x - cal.mean) / cal.std
        self.assertAlmostEqual(float(np.mean(z(self.smile) @ cal.weights)), 1.0, delta=0.05)
        self.assertAlmostEqual(float(np.mean(z(self.rest) @ cal.weights)), 0.0, delta=0.05)

    def test_feature_that_never_moves_gets_little_weight(self):
        cal = self.fit(self.smile)
        self.assertLess(abs(cal.weights[CHEEK]), 0.25 * abs(cal.weights[SMILE]))
        self.assertGreater(cal.quality, 5.0)

    def test_no_smile_seen_falls_back(self):
        rng = np.random.default_rng(2)
        still = np.array([sample(rng) for _ in range(90)])
        cal = self.fit(still)
        self.assertTrue(cal.fallback)
        self.assertGreater(cal.weights[SMILE], 0)
        self.assertEqual(cal.weights[CHEEK], 0)

    def test_disabled_feature_has_zero_weight(self):
        mask = np.ones(N_FEATURES, dtype=bool)
        mask[SMILE] = False
        self.assertEqual(self.fit(self.smile, mask).weights[SMILE], 0.0)

    def test_missing_feature_is_ignored(self):
        rest, smile = self.rest.copy(), self.smile.copy()
        rest[:, WIDTH] = np.nan
        smile[:, WIDTH] = np.nan
        cal = fit_calibration(rest, smile, MIN_STD, FALLBACK)
        self.assertEqual(cal.weights[WIDTH], 0.0)
        self.assertTrue(np.all(np.isfinite(cal.weights)))


class SmileSignalTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.sig = SmileSignal(threshold=0.5)
        self.t = run_calibration(self.sig, self.rng)

    def test_calibration_finishes_with_good_quality(self):
        cal = self.sig.calibration
        self.assertIsNotNone(cal)
        self.assertFalse(cal.fallback)
        self.assertGreater(cal.quality, 3.0)
        self.assertEqual(len(cal.weights), len(FEATURES))

    def test_rest_and_smile_scores(self):
        rest, t = play(self.sig, self.rng, 1.0, 0.0, self.t)
        smile, t = play(self.sig, self.rng, 1.0, 1.0, t)
        self.assertLess(np.mean(rest[10:]), 0.2)
        self.assertGreater(np.mean(smile[10:]), 0.7)

    def test_open_mouth_smile_scores_higher(self):
        # Calibrated with a closed-lip smile; a big open smile moves more.
        smile, _ = play(self.sig, self.rng, 1.0, 1.8, self.t)
        self.assertGreater(np.mean(smile[10:]), 1.2)

    def test_turned_head_is_ignored(self):
        scores, _ = play(self.sig, self.rng, 1.0, 1.0, self.t, yaw=35.0)
        self.assertEqual(max(scores), 0.0)

    def test_caller_gate_is_respected(self):
        scores, _ = play(self.sig, self.rng, 1.0, 1.0, self.t, gate="paused")
        self.assertEqual(max(scores), 0.0)

    def test_long_smile_is_recentred_after_stuck_time(self):
        res, t = None, self.t
        for _ in range(round((SmileSignal.STUCK_TIME + 2) * FPS)):
            res = self.sig.update(sample(self.rng, 1.0), t)
            t += 1 / FPS
            if res.recentred:
                break
        self.assertTrue(res.recentred)
        self.assertGreaterEqual(t - self.t, SmileSignal.STUCK_TIME)

    def test_recalibrate_starts_with_relax_prompt(self):
        self.sig.start_calibration()
        res = self.sig.update(sample(self.rng), self.t)
        self.assertTrue(res.calibrating)
        self.assertEqual(res.step, "relax")
        self.assertIn("relax", res.events)


class PressDetectorTest(unittest.TestCase):
    """t = 0.25 s, so a dash starts at 0.375 s of smiling."""

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

    def test_short_smile_is_a_dot(self):
        kinds = self.feed(0.2, 1.0) + self.feed(0.3, 0.0)
        self.assertEqual(kinds, [PressKind.DOT])

    def test_long_smile_is_a_dash_while_still_smiling(self):
        kinds = self.feed(0.5, 1.0)
        self.assertEqual(kinds, [PressKind.DASH])
        self.assertTrue(self.d.pressed)
        self.assertEqual(self.feed(0.3, 0.0), [])

    def test_hysteresis_keeps_one_press(self):
        kinds = self.feed(0.1, 0.9) + self.feed(0.1, 0.4) + self.feed(0.1, 0.9) \
            + self.feed(0.3, 0.0)
        self.assertEqual(len(kinds), 1)

    def test_slow_fade_after_a_short_smile_stays_a_dot(self):
        kinds = self.feed(0.2, 1.0) + self.feed(0.4, 0.42) + self.feed(0.3, 0.0)
        self.assertEqual(kinds, [PressKind.DOT])

    def test_pause_time_after_release(self):
        self.feed(0.2, 1.0)
        self.feed(0.6, 0.0)
        self.assertAlmostEqual(self.d.pause_time(self.t), 0.6, delta=0.06)


if __name__ == "__main__":
    unittest.main()
