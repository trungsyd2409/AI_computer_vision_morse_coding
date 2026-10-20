"""
Hip Morse - type Morse code by swaying your hips in front of a webcam.

Put both hands together to switch typing on or off. Then, with angle A:
hip angle >= A is a dash, <= -A is a dot. Waiting (-A < angle < A) for more
than t ends the letter, for more than 3t adds a space. H shows the angle.

Run:  python main.py
Keys: S swap dot / dash, X settings, Z hide UI, P pause, H hip angle,
      C chart, M mute, Backspace / Delete edit, Esc quit.
"""

import multiprocessing

from blink_morse.app import BlinkMorseApp


def main() -> None:
    BlinkMorseApp().run()


if __name__ == "__main__":
    # Needed on Windows because the settings window runs in its own process.
    multiprocessing.freeze_support()
    main()
