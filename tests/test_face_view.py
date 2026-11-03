"""Smoke test for the face mesh debug view, with made-up landmarks."""

import importlib.util
import unittest

import numpy as np

from blink_morse.mesh_data import FACE_OVAL, INNER_LIPS, NOSTRILS, OUTER_LIPS, TESSELATION


class MeshDataTest(unittest.TestCase):
    def test_indices_are_valid(self):
        flat = [i for edge in TESSELATION for i in edge]
        self.assertEqual(len(TESSELATION), 1322)
        self.assertLess(max(flat), 468)
        self.assertEqual(len(FACE_OVAL), 36)
        self.assertEqual(len(NOSTRILS[0]), len(NOSTRILS[1]))
        self.assertEqual(len(OUTER_LIPS), 20)
        self.assertEqual(len(INNER_LIPS), 20)


class FaceMeshViewTest(unittest.TestCase):
    def test_render_draws_outline_and_lips(self):
        if importlib.util.find_spec("pygame") is None:
            self.skipTest("pygame not installed")
        from blink_morse.face_view import FaceMeshView

        rng = np.random.default_rng(0)
        pts = rng.uniform(200, 300, (478, 2))
        # Put the outline on a circle so the face has a clear size.
        angles = np.linspace(0, 2 * np.pi, len(FACE_OVAL), endpoint=False)
        pts[list(FACE_OVAL)] = np.stack([250 + 80 * np.cos(angles),
                                         250 + 100 * np.sin(angles)], axis=1)
        view = FaceMeshView()
        surf = view.render(pts, (136, 196), (103, 232, 249), 0.5)
        self.assertEqual(surf.get_size(), (136, 196))
        alpha = np.frombuffer(view._buffer, np.uint8).reshape(196, 136, 4)[:, :, 3]
        self.assertGreater(alpha.max(), 200)          # bold lines exist
        self.assertEqual(len(view.highlight_centres), 1)      # the lips


if __name__ == "__main__":
    unittest.main()
