"""Tests for the Morse table and the pause-driven composer."""

import unittest

from blink_morse.morse import (MORSE_TABLE, EventKind, MorseComposer, candidates,
                               decode)


class MorseTableTest(unittest.TestCase):
    def test_codes_are_unique(self):
        codes = list(MORSE_TABLE.values())
        self.assertEqual(len(codes), len(set(codes)))

    def test_decode(self):
        self.assertEqual(decode(".-"), "A")
        self.assertEqual(decode("..--.."), "?")
        self.assertIsNone(decode("......."))

    def test_candidates_are_sorted_by_length(self):
        self.assertEqual(candidates(".")[:3], ["E", "A", "I"])
        self.assertEqual(candidates(""), [])


def pause(composer, seconds, step=1 / 30):
    """Simulate `seconds` of idle time; return the events it produced."""
    events, t = [], 0.0
    while t <= seconds + 1e-9:
        e = composer.update(t)
        if e is not None:
            events.append(e)
        t += step
    return events


class ComposerTest(unittest.TestCase):
    def type_code(self, composer, code):
        for symbol in code:
            composer.add_symbol(symbol)

    def test_short_pause_does_nothing(self):
        c = MorseComposer()
        self.type_code(c, "..")
        self.assertEqual(pause(c, 0.3), [])
        self.assertEqual(c.code, "..")

    def test_letter_pause_commits_letter(self):
        c = MorseComposer()
        self.type_code(c, "....")
        events = pause(c, 0.6)
        self.assertEqual([e.kind for e in events], [EventKind.LETTER])
        self.assertEqual(c.text, "H")

    def test_word_pause_adds_space_once(self):
        c = MorseComposer()
        self.type_code(c, "..")
        events = pause(c, 3.0)
        self.assertEqual([e.kind for e in events], [EventKind.LETTER, EventKind.SPACE])
        self.assertEqual(c.text, "I ")
        self.assertEqual(pause(c, 5.0), [])          # no second space

    def test_typing_before_word_pause_keeps_the_word(self):
        c = MorseComposer()
        self.type_code(c, ".-")
        pause(c, 1.0)                                # letter, no space yet
        self.type_code(c, "-")
        pause(c, 0.6)
        self.assertEqual(c.text, "AT")

    def test_hello_world(self):
        c = MorseComposer()
        for i, word in enumerate(("HELLO", "WORLD")):
            for char in word:
                self.type_code(c, MORSE_TABLE[char])
                pause(c, 0.7)
            if i == 0:
                pause(c, 2.1)
        self.assertEqual(c.text, "HELLO WORLD")

    def test_invalid_code_adds_no_space(self):
        c = MorseComposer()
        self.type_code(c, "..--")
        events = pause(c, 3.0)
        self.assertEqual([e.kind for e in events], [EventKind.INVALID])
        self.assertEqual(c.text, "")

    def test_too_long_code_rejected_early(self):
        c = MorseComposer()
        for _ in range(6):
            c.add_symbol("-")
        event = c.add_symbol("-")
        self.assertEqual(event.kind, EventKind.INVALID)
        self.assertEqual(c.code, "")

    def test_pause_state_for_hud(self):
        c = MorseComposer(letter_gap=0.5, word_gap=2.0)
        self.assertEqual(c.pause_state(0.0), (None, 0.0))
        c.add_symbol(".")
        self.assertEqual(c.pause_state(0.25), ("letter", 0.5))
        c.update(0.5)
        self.assertEqual(c.pause_state(1.0), ("space", 0.5))

    def test_delete_symbol_then_char(self):
        c = MorseComposer()
        self.type_code(c, ".-")
        pause(c, 0.6)
        c.add_symbol("-")
        self.assertEqual(c.delete().value, "-")
        self.assertEqual(c.delete().value, "A")
        self.assertIsNone(c.delete())

    def test_delete_cancels_pending_space(self):
        c = MorseComposer()
        self.type_code(c, ".-")
        pause(c, 0.6)
        c.delete()
        self.assertEqual(pause(c, 3.0), [])

    def test_clear(self):
        c = MorseComposer()
        self.type_code(c, ".-")
        pause(c, 0.6)
        c.clear()
        self.assertEqual((c.text, c.code), ("", ""))


if __name__ == "__main__":
    unittest.main()
