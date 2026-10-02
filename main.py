"""
Morse Hand - type Morse code with your right hand in front of a webcam.
Touch the index and middle fingertips together: < 1.5 t = dot, >= 1.5 t =
dash; keep them apart 3 t to end the letter, 7 t to add a space.

Run:  python main.py
Keys: Enter send message, X camera settings, H toggle chart, M mute,
      Backspace / Delete edit, Esc quit.
"""

import multiprocessing

from morse_hand.app import MorseHandApp


def main() -> None:
    MorseHandApp().run()


if __name__ == "__main__":
    # Needed on Windows because the settings window runs in its own process.
    multiprocessing.freeze_support()
    main()
