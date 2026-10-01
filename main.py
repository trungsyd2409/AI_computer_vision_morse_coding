"""
Tongue Morse - type Morse code by sticking your tongue out in front of a
webcam. Short = dot, long = dash; the length that counts is set with X.

Run:  python main.py
Keys: X settings, Z hide UI, P pause listening, H toggle chart, M mute,
      Backspace / Delete edit, Esc quit.
"""

import multiprocessing

from blink_morse.app import BlinkMorseApp


def main() -> None:
    BlinkMorseApp().run()


if __name__ == "__main__":
    # Needed on Windows because the settings window runs in its own process.
    multiprocessing.freeze_support()
    main()
