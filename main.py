"""
Curl Morse - type Morse code with dumbbell-style arm curls in front of a
webcam. Short curl = dot, long curl = dash, keep the arm straight to end
a letter (5 t) or a word (7 t).

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
