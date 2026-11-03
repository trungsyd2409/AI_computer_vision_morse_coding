"""
Smile Morse - type Morse code by smiling in front of a webcam.

Run:  python main.py
Keys: C calibrate, X settings, Z hide UI, P pause listening, H toggle chart,
      F feature weights, G face mesh / timing legend, L record features to
      CSV (hold N to label a smile), M mute, Backspace / Delete edit, Esc quit.
"""

import multiprocessing

from blink_morse.app import BlinkMorseApp


def main() -> None:
    BlinkMorseApp().run()


if __name__ == "__main__":
    # Needed on Windows because the settings window runs in its own process.
    multiprocessing.freeze_support()
    main()
