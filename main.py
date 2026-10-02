"""
Nostril Morse - type Morse code by flaring your nostrils in front of a
webcam. Short flare = dot, long flare = dash; the threshold is set with X.
C recalibrates the resting nostril size, L logs measurements to logs/*.csv.

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
