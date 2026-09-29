"""
International Morse code table plus the state machine that turns
dot / dash / end-of-letter / delete events into text.

Nothing in this file knows about cameras or faces, which makes it easy to
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


@dataclass
class ComposerEvent:
    kind: EventKind
    value: str = ""        # the symbol, the character, or the rejected code


class MorseComposer:
    """
    Builds a message from Morse input, using silences as separators, like
    real Morse code.

    `update(pause)` is called every frame with how long the input has been
    idle (the eyes open since the last blink):

    * idle for `letter_gap` with symbols pending -> the letter ends
    * idle for `word_gap` after a letter          -> a space is added

    Both gaps are measured from the same moment, so a new symbol typed in
    between simply continues the same word.
    """

    def __init__(self, letter_gap: float = 0.3, word_gap: float = 0.7,
                 max_text: int = 200):
        self.letter_gap = letter_gap
        self.word_gap = word_gap
        self.max_text = max_text
        self.code = ""                     # symbols of the letter in progress
        self.text = ""                     # the decoded message
        self._space_pending = False        # a letter ended, a space may follow

    # -- symbol input -------------------------------------------------------

    def add_symbol(self, symbol: str) -> ComposerEvent:
        """Append '.' or '-' to the current code."""
        if symbol not in (".", "-"):
            raise ValueError(f"Unsupported Morse symbol: {symbol!r}")
        # A new symbol before the word gap means the word is not over yet.
        self._space_pending = False

        # A code longer than anything in the table can never resolve, so
        # reject it right away instead of letting the user keep going.
        if len(self.code) >= MAX_CODE_LENGTH:
            rejected = self.code + symbol
            self.code = ""
            return ComposerEvent(EventKind.INVALID, rejected)

        self.code += symbol
        return ComposerEvent(EventKind.SYMBOL, symbol)

    def end_letter(self) -> Optional[ComposerEvent]:
        """Finish the letter in progress. Does nothing if no symbol is pending."""
        if not self.code:
            return None
        event = self._commit()
        self._space_pending = event.kind == EventKind.LETTER
        return event

    # -- pauses ----------------------------------------------------------------

    def update(self, pause: float) -> list:
        """
        Call every frame with the number of seconds the input has been idle.
        Returns the events (letter, space) that the pause produced.
        """
        events = []
        if self.code and pause >= self.letter_gap:
            events.append(self.end_letter())
        if self._space_pending and pause >= self.word_gap:
            self._space_pending = False
            space = self._insert_space()
            if space is not None:
                events.append(space)
        return events

    def pause_state(self, pause: float) -> tuple:
        """
        What the current pause is counting towards, for the HUD:
        ("letter", progress 0..1), ("space", progress 0..1) or (None, 0).
        """
        if self.code:
            return "letter", min(1.0, pause / self.letter_gap)
        if self._space_pending:
            return "space", min(1.0, pause / self.word_gap)
        return None, 0.0

    # -- delete and clear ------------------------------------------------------

    def delete(self) -> Optional[ComposerEvent]:
        """Remove the last symbol if a code is in progress, else the last char."""
        self._space_pending = False
        if self.code:
            removed = self.code[-1]
            self.code = self.code[:-1]
            return ComposerEvent(EventKind.DELETE, removed)
        if self.text:
            removed = self.text[-1]
            self.text = self.text[:-1]
            return ComposerEvent(EventKind.DELETE, removed)
        return None

    def clear(self) -> ComposerEvent:
        self._space_pending = False
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
