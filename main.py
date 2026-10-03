"""
Frown Morse - type Morse code by frowning in front of a webcam. The vertical
furrow lines between the eyebrows are measured: frown < 1.5 t = dot,
>= 1.5 t = dash; relaxed 3 t = end of letter, 7 t = space.

Run:  python main.py  (keep your face relaxed for the first 2 seconds)
Keys: X settings, Z hide UI, P pause listening, H toggle chart, M mute,
      C recalibrate, Backspace / Delete edit, Esc quit.
"""

import multiprocessing

from blink_morse.app import BlinkMorseApp


def main() -> None:
    BlinkMorseApp().run()


if __name__ == "__main__":
    # Needed on Windows because the settings window runs in its own process.
    multiprocessing.freeze_support()
    main()
