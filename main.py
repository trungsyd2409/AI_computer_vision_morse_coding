"""
Hand Morse - type Morse code with your index and middle fingers.

Hold the two fingers straight and together to start recording, then bend
them into a loop: a short loop is a dot, a long loop a dash.

Run:  python main.py
Keys: X settings, Z hide UI, P pause listening, H toggle chart,
      F gesture checks, G hand skeleton / timing legend, L record hand
      measurements to CSV (hold N to label a loop), M mute,
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
