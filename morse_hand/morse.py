"""
International Morse code table plus the state machine that turns
dot / dash / end-of-letter / delete events into text.

Nothing in this file knows about cameras or hands, which makes it easy to
unit test and reuse with any other input (keyboard, button, sensor...).
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

# ITU-R M.1677-1 codes for letters, digits and the most common punctuation.
MORSE_TABLE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "!": "-.-.--",
    "'": ".----.", "/": "-..-.", "(": "-.--.", ")": "-.--.-",
    ":": "---...", "=": "-...-", "+": ".-.-.", "-": "-....-",
    "@": ".--.-.",
}

# Reverse lookup used when a letter is committed.
CODE_TO_CHAR = {code: char for char, code in MORSE_TABLE.items()}

# The longest code in the table; anything longer can never be valid.
MAX_CODE_LENGTH = max(len(code) for code in MORSE_TABLE.values())


def decode(code: str) -> Optional[str]:
    """Return the character for a complete code, or None if it is unknown."""
    return CODE_TO_CHAR.get(code)


def candidates(prefix: str) -> list:
    """
    All characters whose code starts with `prefix`, ordered by code length.

    The HUD uses this to show which letters are still reachable while the
    user is halfway through a code.
    """
    if not prefix:
        return []
    matches = [c for c, code in MORSE_TABLE.items() if code.startswith(prefix)]
    return sorted(matches, key=lambda c: (len(MORSE_TABLE[c]), c))


class EventKind(Enum):
    SYMBOL = auto()        # a dot or dash was added to the current code
    LETTER = auto()        # the current code was committed as a character
    SPACE = auto()         # a space was typed
    INVALID = auto()       # the code did not match any character
    DELETE = auto()        # last symbol or last character removed
    CLEAR = auto()         # everything wiped
    ARMED = auto()         # first ring tap, waiting to see if a second follows


@dataclass
class ComposerEvent:
    kind: EventKind
    value: str = ""        # the symbol, the character, or the rejected code


class MorseComposer:
    """
    Builds a message from Morse input.

    Space rule: one ring tap ends the current letter, a second ring tap
    inside `double_tap_window` seconds inserts a space. Any dot or dash in
    between cancels the pending double tap, so "ring, dot, ring" is never
    mistaken for a space.
    """

    def __init__(self, double_tap_window: float = 0.45, max_text: int = 200):
        self.double_tap_window = double_tap_window
        self.max_text = max_text
        self.code = ""                     # symbols of the letter in progress
        self.text = ""                     # the decoded message
        self._ring_armed = False
        self._last_ring_time = 0.0

    # -- symbol input -------------------------------------------------------

    def add_symbol(self, symbol: str) -> ComposerEvent:
        """Append '.' or '-' to the current code."""
        if symbol not in (".", "-"):
            raise ValueError(f"Unsupported Morse symbol: {symbol!r}")
        self._ring_armed = False

        # A code longer than anything in the table can never resolve, so
        # reject it right away instead of letting the user keep tapping.
        if len(self.code) >= MAX_CODE_LENGTH:
            rejected = self.code + symbol
            self.code = ""
            return ComposerEvent(EventKind.INVALID, rejected)

        self.code += symbol
        return ComposerEvent(EventKind.SYMBOL, symbol)

    # -- ring finger: end of letter or space ---------------------------------

    def ring_tap(self, now: float) -> Optional[ComposerEvent]:
        """Handle a ring-finger tap at time `now` (seconds)."""
        is_double = (
            self._ring_armed
            and now - self._last_ring_time <= self.double_tap_window
        )

        if is_double:
            self._ring_armed = False
            return self._insert_space()

        self._ring_armed = True
        self._last_ring_time = now
        if self.code:
            return self._commit()
        return ComposerEvent(EventKind.ARMED)

    def pending_space_progress(self, now: float) -> float:
        """
        0..1 progress of the double-tap window, or -1 when nothing is pending.
        Used by the HUD to draw a small countdown.
        """
        if not self._ring_armed:
            return -1.0
        elapsed = now - self._last_ring_time
        if elapsed > self.double_tap_window:
            self._ring_armed = False
            return -1.0
        return elapsed / self.double_tap_window

    # -- pinky: delete and clear ---------------------------------------------

    def delete(self) -> Optional[ComposerEvent]:
        """Remove the last symbol if a code is in progress, else the last char."""
        self._ring_armed = False
        if self.code:
            removed = self.code[-1]
            self.code = self.code[:-1]
            return ComposerEvent(EventKind.DELETE, removed)
        if self.text:
            removed = self.text[-1]
            self.text = self.text[:-1]
            return ComposerEvent(EventKind.DELETE, removed)
        return None

    def take_text(self) -> str:
        """Return the finished message and empty it, ready for the next one."""
        text = self.text.strip()
        self.text = ""
        self._ring_armed = False
        return text

    def clear(self) -> ComposerEvent:
        self._ring_armed = False
        self.code = ""
        self.text = ""
        return ComposerEvent(EventKind.CLEAR)

    # -- internals ------------------------------------------------------------

    def _commit(self) -> ComposerEvent:
        code, self.code = self.code, ""
        char = decode(code)
        if char is None:
            return ComposerEvent(EventKind.INVALID, code)
        self._append(char)
        return ComposerEvent(EventKind.LETTER, char)

    def _insert_space(self) -> Optional[ComposerEvent]:
        # Collapse repeated spaces and ignore a space at the very start.
        if not self.text or self.text.endswith(" "):
            return None
        self._append(" ")
        return ComposerEvent(EventKind.SPACE, " ")

    def _append(self, char: str) -> None:
        self.text = (self.text + char)[-self.max_text:]
