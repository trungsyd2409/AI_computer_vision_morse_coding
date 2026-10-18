"""
Arm Morse - type Morse code with diagonal forearm poses in front of a webcam.

Hold a right-hand fist to type. Left forearm raised 45 deg = dash, lowered
45 deg = dot. Right forearm raised 45 deg = end of letter, lowered 45 deg =
space. Bring each forearm back to level between gestures.

Run:  python main.py
Keys: S swap arms, X settings, Z hide UI, P pause, H toggle chart, M mute,
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
