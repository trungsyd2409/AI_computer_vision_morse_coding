"""Tests for the ear measurements, head-angle cancellation and twitch mode, on synthetic images."""

import math
import unittest

import cv2
import numpy as np

from blink_morse.ear import FEATURES, EarSignal, EarTracker, head_axes, measure_edge
from blink_morse.fusion import fit_calibration
from blink_morse.press import PressKind, TwitchDetector

SKIN = (205, 150, 125)          # RGB
EAR = (215, 145, 130)           # a little pinker than the cheek
FPS = 30.0


def landmarks(shift=(0.0, 0.0)) -> np.ndarray:
    """A face 100 px between the outer eye corners, edges at x = 170 and 330."""
    pts = np.tile(np.array([[250.0, 250.0]]), (478, 1))
    pts[33], pts[263] = (200, 200), (300, 200)
    pts[133], pts[362] = (225, 200), (275, 200)
    pts[234], pts[454] = (170, 260), (330, 260)
    for i, p in zip((1, 168, 6, 197, 195, 5, 10, 151, 9, 8),
                    ((250, 270), (250, 195), (250, 215), (250, 230), (250, 245),
                     (250, 255), (250, 120), (250, 140), (250, 175), (250, 185))):
        pts[i] = p
    return pts + np.array(shift)


def frame(ear_dx=0.0, ear_dy=0.0, shift=(0, 0), right_ear=True) -> np.ndarray:
    """White background, face ellipse, a textured right ear, no left ear."""
    img = np.full((480, 640, 3), 245, np.uint8)
    sx, sy = shift
    cv2.ellipse(img, (250 + sx, 250 + sy), (82, 130), 0, 0, 360, SKIN, -1)
    if right_ear:
        cx, cy = 345 + sx + ear_dx, 225 + sy + ear_dy
        cv2.ellipse(img, (int(round(cx)), int(round(cy))), (18, 40), 0, 0, 360, EAR, -1)
        # Folds of the ear: give the tracker some corners to follow.
        for k, (dx, dy) in enumerate(((-6, -25), (4, -12), (-3, 0), (6, 14), (-5, 26))):
            cv2.circle(img, (int(round(cx + dx)), int(round(cy + dy))), 3,
                       (150 - 10 * k, 90, 80), -1)
    return img


class EdgeTest(unittest.TestCase):
    def test_right_ear_edge(self):
        pts = landmarks()
        edge = measure_edge(frame(), pts, "right", head_axes(pts))
        # The ear reaches x = 363 at its widest row (33 px beyond the face
        # edge at 330, eyes 100 px apart). The median over the scanned rows
        # of a rounded ear is a little less.
        self.assertGreater(edge, 0.28)
        self.assertLessEqual(edge, 0.335)

    def test_pulled_back_ear_is_closer(self):
        pts = landmarks()
        rest = measure_edge(frame(), pts, "right", head_axes(pts))
        pulled = measure_edge(frame(ear_dx=-5), pts, "right", head_axes(pts))
        self.assertAlmostEqual(rest - pulled, 0.05, delta=0.015)

    def test_hidden_ear_is_nan(self):
        pts = landmarks()
        self.assertTrue(math.isnan(measure_edge(frame(), pts, "left", head_axes(pts))))

    def test_mirrored_image_gives_the_same_edge(self):
        # The app mirrors the camera: the same ear then sits on the screen's
        # left, and the eye line points left. Landmark indices stay the same.
        pts = landmarks()
        img = frame()
        w = img.shape[1]
        mirrored_pts = pts.copy()
        mirrored_pts[:, 0] = w - 1 - pts[:, 0]
        mirrored = np.ascontiguousarray(img[:, ::-1])
        a = measure_edge(img, pts, "right", head_axes(pts))
        b = measure_edge(mirrored, mirrored_pts, "left", head_axes(mirrored_pts))
        self.assertAlmostEqual(a, b, delta=0.02)
        self.assertTrue(math.isnan(measure_edge(mirrored, mirrored_pts, "right",
                                                head_axes(mirrored_pts))))

    def test_hair_on_the_cheek_hides_the_ear(self):
        img = frame()
        pts = landmarks()
        cv2.circle(img, (318, 245), 12, (30, 25, 25), -1)    # dark hair where the cheek is sampled
        self.assertTrue(math.isnan(measure_edge(img, pts, "right", head_axes(pts))))


class TrackerTest(unittest.TestCase):
    def run_frames(self, frames_and_points):
        tracker = EarTracker()
        out = None
        for i, (img, pts) in enumerate(frames_and_points):
            out = tracker.update(img, pts, 0.0, 0.0, i / FPS)
        return out

    def test_ear_moving_up_is_seen(self):
        pts = landmarks()
        out = self.run_frames([(frame(), pts), (frame(ear_dy=-3), pts)])
        self.assertAlmostEqual(out.features[FEATURES.index("move_ry")], -0.03, delta=0.01)
        self.assertTrue(math.isnan(out.features[FEATURES.index("move_ly")]))

    def test_head_moving_with_the_ear_is_not_ear_motion(self):
        out = self.run_frames([(frame(), landmarks()),
                               (frame(shift=(4, 2)), landmarks((4, 2)))])
        self.assertLess(abs(out.features[FEATURES.index("move_rx")]), 0.008)
        self.assertLess(abs(out.features[FEATURES.index("move_ry")]), 0.008)

    def test_views_for_the_hud(self):
        out = self.run_frames([(frame(), landmarks())])
        right, left = out.views["right"], out.views["left"]
        self.assertTrue(right.visible)
        self.assertFalse(left.visible)
        self.assertGreater(right.edge_x, right.face_x)
        self.assertEqual(right.crop.shape[:2], (162, 108))


class NuisanceTest(unittest.TestCase):
    """Head yaw moves the ear edge; the calibration must learn to cancel it."""

    def setUp(self):
        rng = np.random.default_rng(0)
        n_rest, n_act = 300, 90
        yaw_rest = rng.normal(0, 2.0, n_rest)            # head turn step included
        yaw_act = rng.normal(0, 0.3, n_act)
        def rows(yaw, wiggle):
            edge = 0.15 + 0.0115 * yaw - 0.012 * wiggle + rng.normal(0, 0.002, len(yaw))
            out = np.full((len(yaw), len(FEATURES)), np.nan)
            out[:, FEATURES.index("edge_r")] = edge
            out[:, FEATURES.index("yaw")] = yaw
            out[:, FEATURES.index("pitch")] = rng.normal(0, 0.3, len(yaw))
            return out
        self.rest, self.active = rows(yaw_rest, 0.0), rows(yaw_act, 1.0)
        self.min_std = np.array(EarSignal.MIN_STD)
        self.fallback = np.zeros(len(FEATURES))
        self.nuisance = np.array([f in EarSignal.NUISANCE for f in FEATURES])

    def score(self, cal, edge, yaw):
        x = np.zeros(len(FEATURES))
        x[FEATURES.index("edge_r")] = edge
        x[FEATURES.index("yaw")] = yaw
        z = np.nan_to_num((x - cal.mean) / cal.std)
        return float(np.dot(cal.weights, z))

    def test_head_turn_is_cancelled(self):
        with_n = fit_calibration(self.rest, self.active, self.min_std, self.fallback,
                                 shrink=0.15, nuisance=self.nuisance)
        # Baseline: the same data without the head angles at all.
        no_angles = ~self.nuisance
        without = fit_calibration(self.rest, self.active, self.min_std, self.fallback,
                                  enabled=no_angles, shrink=0.15)
        # Without the angles the head turns hide the wiggle completely.
        self.assertGreater(with_n.quality, 3.0)
        self.assertLess(without.quality, with_n.quality / 2)
        # Head turned 3 degrees, ears relaxed: close to 0.
        turned = (0.15 + 0.0115 * 3, 3.0)
        self.assertLess(abs(self.score(with_n, *turned)), 0.25)
        # A wiggle with the head still scores about 1.
        self.assertAlmostEqual(self.score(with_n, 0.15 - 0.012, 0.0), 1.0, delta=0.3)

    def test_nuisance_never_counts_as_the_action(self):
        cal = fit_calibration(self.rest, self.active, self.min_std, self.fallback,
                              shrink=0.15, nuisance=self.nuisance)
        self.assertEqual(cal.effect[FEATURES.index("yaw")], 0.0)


class EarSignalTest(unittest.TestCase):
    def test_calibration_includes_a_head_turn_rest_step(self):
        sig = EarSignal()
        kinds = [k for k, _ in sig.steps]
        self.assertEqual(kinds[:2], ["relax", "turn"])
        self.assertEqual(kinds.count("ear"), sig.ROUNDS)
        self.assertIn("turn", sig.REST_STEPS)


class TwitchTest(unittest.TestCase):
    def feed(self, d, script, t=0.0):
        out = []
        for seconds, score in script:
            for _ in range(max(1, round(seconds * FPS))):
                out += [(t, e.kind) for e in d.update(score, t)]
                t += 1 / FPS
        return out, t

    def test_one_wiggle_is_a_dot_after_the_wait(self):
        d = TwitchDetector(0.5, 0.25)
        events, _ = self.feed(d, [(0.3, 0), (0.2, 1), (0.8, 0)])
        self.assertEqual([k for _, k in events], [PressKind.DOT])
        # Typed 2 t (0.5 s) after the wiggle ended at 0.5 s.
        self.assertAlmostEqual(events[0][0], 1.0, delta=0.08)

    def test_two_quick_wiggles_are_a_dash(self):
        d = TwitchDetector(0.5, 0.25)
        events, _ = self.feed(d, [(0.3, 0), (0.2, 1), (0.2, 0), (0.2, 1), (0.8, 0)])
        self.assertEqual([k for _, k in events], [PressKind.DASH])

    def test_letter_pause_waits_for_the_decision(self):
        d = TwitchDetector(0.5, 0.25)
        self.feed(d, [(0.3, 0), (0.2, 1), (0.2, 0)])
        self.assertTrue(d.waiting)
        self.assertEqual(d.pause_time(1.0), 0.0)

    def test_long_hold_is_still_one_wiggle(self):
        d = TwitchDetector(0.5, 0.25)
        events, _ = self.feed(d, [(0.3, 0), (1.5, 1), (0.8, 0)])
        self.assertEqual([k for _, k in events], [PressKind.DOT])


if __name__ == "__main__":
    unittest.main()
