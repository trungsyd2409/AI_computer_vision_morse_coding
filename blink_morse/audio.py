"""
Tiny synthesiser for the feedback sounds.

All sounds are generated with NumPy at start-up, so the project ships
without any audio files. Each tone gets a short fade in / fade out to avoid
the "click" you hear when a sine wave starts or stops abruptly.
"""

import numpy as np
import pygame

SAMPLE_RATE = 44100


def _tone(freq: float, duration: float, volume: float = 0.35,
          attack: float = 0.006, release: float = 0.04,
          overtone: float = 0.18) -> np.ndarray:
    """Sine tone with a soft second harmonic and a linear envelope."""
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n) / SAMPLE_RATE
    wave = np.sin(2 * np.pi * freq * t) + overtone * np.sin(4 * np.pi * freq * t)

    env = np.ones(n)
    a = max(int(SAMPLE_RATE * attack), 1)
    r = max(int(SAMPLE_RATE * release), 1)
    env[:a] = np.linspace(0.0, 1.0, a)
    env[-r:] = np.linspace(1.0, 0.0, r)
    return wave * env * volume


def _sweep(f0: float, f1: float, duration: float, volume: float = 0.3) -> np.ndarray:
    """Tone that glides from f0 to f1, used for delete and clear."""
    n = int(SAMPLE_RATE * duration)
    freqs = np.linspace(f0, f1, n)
    phase = 2 * np.pi * np.cumsum(freqs) / SAMPLE_RATE
    env = np.linspace(1.0, 0.0, n) ** 1.5
    return np.sin(phase) * env * volume


def _to_sound(mono: np.ndarray) -> "pygame.mixer.Sound":
    pcm = np.clip(mono, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([pcm, pcm]))
    return pygame.sndarray.make_sound(stereo)


class SoundBank:
    def __init__(self):
        self.enabled = True
        self._sounds = {}
        try:
            # pygame.init() may already have opened the mixer with a larger
            # buffer. Re-open it with a small one so the sound of a dot
            # follows the blink without a noticeable lag.
            pygame.mixer.quit()
            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=256)
        except pygame.error:
            # No audio device: the app still works, just silently.
            self.enabled = False
            return

        gap = np.zeros(int(SAMPLE_RATE * 0.03))
        self._sounds = {
            # Dash is three times the length of a dot, as in real Morse.
            "dot": _to_sound(_tone(740, 0.07)),
            "dash": _to_sound(_tone(740, 0.21)),
            "letter": _to_sound(np.concatenate(
                [_tone(880, 0.07, 0.28), gap, _tone(1320, 0.12, 0.28)])),
            "space": _to_sound(_tone(520, 0.12, 0.22)),
            "invalid": _to_sound(_tone(196, 0.18, 0.3, overtone=0.5)),
            "delete": _to_sound(_sweep(900, 500, 0.09, 0.22)),
            "clear": _to_sound(_sweep(1100, 220, 0.32, 0.25)),
            "pause": _to_sound(np.concatenate(
                [_tone(660, 0.06, 0.2), gap, _tone(440, 0.09, 0.2)])),
            "resume": _to_sound(np.concatenate(
                [_tone(440, 0.06, 0.2), gap, _tone(660, 0.09, 0.2)])),
        }

    def play(self, name: str) -> None:
        if self.enabled and name in self._sounds:
            self._sounds[name].play()

    def toggle(self) -> bool:
        if self._sounds:
            self.enabled = not self.enabled
        return self.enabled
