"""
Push-up Morse - type Morse code by doing push-ups in front of a webcam.
Down < 1.5 t = dot, down >= 1.5 t = dash, up 3 t = end of letter,
up 5 t = space.

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
