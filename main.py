"""
Arm Morse - type Morse code with forearm angles in front of a webcam.

Put both hands together to switch typing on or off. Each forearm's angle
against the X axis (right arm mirrored) fires the moment it enters a zone:
left 35..145 = dash, -135..-35 = dot; right 35..145 = end of letter,
-135..-35 = space. An arm must leave its zone before it fires again.

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
