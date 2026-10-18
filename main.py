"""
Arm Morse - type Morse code with diagonal forearm poses in front of a webcam.

Hold a right-hand fist to type. Left forearm angle against the X axis
entering 35..145 deg = dash, -135..-35 deg = dot (instant; leave the zone
before the next one). Right forearm raised 45 deg = end of letter, lowered
45 deg = space, back to level between gestures.

Run:  python main.py
Keys: S swap arms, X settings, Z hide UI, P pause, H angles, C chart, M mute,
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
