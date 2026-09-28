"""
Small drawing helpers on top of pygame-ce.

pygame's built-in shapes are either aliased or single-pixel wide. These
helpers build the few smooth shapes the HUD needs (capsules, thick arcs,
soft glows) out of the anti-aliased primitives pygame-ce provides.
"""

import math
from functools import lru_cache

import numpy as np
import pygame


def with_alpha(color, alpha: float) -> tuple:
    """Return an RGBA tuple with alpha given as 0..1."""
    return (color[0], color[1], color[2], int(max(0.0, min(1.0, alpha)) * 255))


def scale_rgb(color, k: float) -> tuple:
    """Multiply an RGB colour by k (used to fade additive glows)."""
    k = max(0.0, min(1.0, k))
    return (int(color[0] * k), int(color[1] * k), int(color[2] * k))


def circle(surface, color, center, radius) -> None:
    pygame.draw.aacircle(surface, color, (round(center[0]), round(center[1])),
                         max(1, round(radius)))


def ring(surface, color, center, radius, width: int = 2) -> None:
    pygame.draw.aacircle(surface, color, (round(center[0]), round(center[1])),
                         max(1, round(radius)), width)


def line(surface, color, a, b, width: int = 2) -> None:
    pygame.draw.aaline(surface, color, (float(a[0]), float(a[1])),
                       (float(b[0]), float(b[1])), width)


def capsule(surface, color, center, length: float, thickness: float) -> None:
    """Horizontal pill shape, used for the dash symbol."""
    r = thickness / 2.0
    cx, cy = center
    rect = pygame.Rect(0, 0, round(length), round(thickness))
    rect.center = (round(cx), round(cy))
    pygame.draw.rect(surface, color, rect, border_radius=round(r))


def rounded_rect(surface, color, rect, radius: int, width: int = 0) -> None:
    pygame.draw.rect(surface, color, rect, width=width, border_radius=radius)


def arc(surface, color, center, radius: float, start: float, sweep: float,
        width: float = 3.0, segments: int = 48) -> None:
    """
    Thick anti-aliased arc drawn as a polygon.
    Angles are in radians, 0 points up and positive runs clockwise, which
    is how a progress ring is usually read.
    """
    if sweep <= 0:
        return
    steps = max(2, int(segments * sweep / (2 * math.pi)))
    center = (float(center[0]), float(center[1]))
    outer, inner = [], []
    for i in range(steps + 1):
        a = start + sweep * i / steps
        sx, sy = math.sin(a), -math.cos(a)
        outer.append((center[0] + sx * (radius + width / 2),
                      center[1] + sy * (radius + width / 2)))
        inner.append((center[0] + sx * (radius - width / 2),
                      center[1] + sy * (radius - width / 2)))
    points = outer + inner[::-1]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.aalines(surface, color, True, points)


# ---------------------------------------------------------------------------
# Soft glow sprites for the additive glow layer
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _glow_mask(size: int = 128) -> np.ndarray:
    """Gaussian falloff from 1 in the centre to 0 at the edge."""
    y, x = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    d2 = ((x - c) ** 2 + (y - c) ** 2) / (c * c)
    return np.clip(np.exp(-d2 * 4.0) - np.exp(-4.0), 0.0, 1.0)


@lru_cache(maxsize=256)
def glow_sprite(color: tuple, diameter: int) -> pygame.Surface:
    """Pre-tinted, pre-scaled glow blob. Cached because it is reused a lot."""
    mask = _glow_mask()
    rgb = (mask[..., None] * np.array(color, dtype=np.float32)[None, None, :])
    surf = pygame.surfarray.make_surface(rgb.astype(np.uint8).swapaxes(0, 1))
    return pygame.transform.smoothscale(surf, (diameter, diameter))


def glow(surface, color, center, radius: float, intensity: float = 1.0) -> None:
    """Add a soft light blob to an opaque, black-cleared glow surface."""
    if intensity <= 0.01 or radius < 1:
        return
    # Quantise so the sprite cache stays small.
    diameter = int(max(4, round(radius * 2 / 4) * 4))
    level = round(max(0.0, min(1.0, intensity)) * 16) / 16
    sprite = glow_sprite(scale_rgb(color, level), diameter)
    surface.blit(sprite, (round(center[0] - diameter / 2), round(center[1] - diameter / 2)),
                 special_flags=pygame.BLEND_RGB_ADD)


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------

def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def ease_out_back(t: float, overshoot: float = 1.6) -> float:
    t = max(0.0, min(1.0, t))
    t -= 1
    return t * t * ((overshoot + 1) * t + overshoot) + 1
