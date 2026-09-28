"""
One Euro filter (Casiez et al., CHI 2012) applied to all landmarks at once.

Raw landmarks jitter by a pixel or two even when the head is still. A plain
moving average would remove the jitter but make fast movements lag behind.
The One Euro filter adapts: it smooths hard when the head is slow and lets
the signal through when it moves quickly.

It is only used for drawing the eye outlines. Blink detection works on the
raw scores so that a quick wink is never delayed.
"""

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.4, beta: float = 0.02,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = None
        self._dx = None
        self._t = None

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._x is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return x

        dt = max(t - self._t, 1e-4)
        self._t = t

        # Estimate the speed, smoothed with its own fixed cutoff.
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx

        # Faster motion -> higher cutoff -> less smoothing.
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x
