"""
Sit-up Morse - type Morse code by doing sit-ups in front of a webcam.
Up < 1.5 t = dot, up >= 1.5 t = dash, lying down 3 t = end of letter,
lying down 5 t = space.

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
