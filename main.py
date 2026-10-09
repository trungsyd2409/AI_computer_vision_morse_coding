"""
Squat Morse - type Morse code by doing squats in front of a webcam.
Touch your hands together to switch typing on. Down < 1.5 t = dot,
down >= 1.5 t = dash, standing 5 t = end of letter, standing 10 t = space.

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
