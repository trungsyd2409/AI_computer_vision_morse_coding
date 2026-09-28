"""Tests for the Morse table and the composer."""

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
        events.extend(composer.update(t))
        t += step
    return events


class ComposerTest(unittest.TestCase):
    def letter(self, composer, code):
        for symbol in code:
            composer.add_symbol(symbol)
        return composer.end_letter()

    def test_end_letter_commits(self):
        c = MorseComposer()
        event = self.letter(c, "....")
        self.assertEqual(event.kind, EventKind.LETTER)
        self.assertEqual(c.text, "H")

    def test_end_letter_without_symbols_does_nothing(self):
        c = MorseComposer()
        self.assertIsNone(c.end_letter())

    def test_short_pause_does_nothing(self):
        c = MorseComposer(word_gap=3.0)
        self.letter(c, "..")
        self.assertEqual(pause(c, 2.5), [])
        self.assertEqual(c.text, "I")

    def test_word_pause_adds_space_once(self):
        c = MorseComposer(word_gap=3.0)
        self.letter(c, "..")
        events = pause(c, 3.2)
        self.assertEqual([e.kind for e in events], [EventKind.SPACE])
        self.assertEqual(c.text, "I ")
        self.assertEqual(pause(c, 5.0), [])          # no second space

    def test_word_pause_finishes_a_pending_letter_first(self):
        c = MorseComposer(word_gap=3.0)
        c.add_symbol("-")
        events = pause(c, 3.2)
        self.assertEqual([e.kind for e in events], [EventKind.LETTER, EventKind.SPACE])
        self.assertEqual(c.text, "T ")

    def test_typing_before_word_pause_keeps_the_word(self):
        c = MorseComposer(word_gap=3.0)
        self.letter(c, ".-")
        pause(c, 1.5)
        self.letter(c, "-")
        self.assertEqual(c.text, "AT")

    def test_hello_world(self):
        c = MorseComposer()
        for i, word in enumerate(("HELLO", "WORLD")):
            for char in word:
                self.letter(c, MORSE_TABLE[char])
            if i == 0:
                pause(c, 3.1)
        self.assertEqual(c.text, "HELLO WORLD")

    def test_invalid_code_adds_no_space(self):
        c = MorseComposer()
        event = self.letter(c, "..--")
        self.assertEqual(event.kind, EventKind.INVALID)
        self.assertEqual(pause(c, 4.0), [])
        self.assertEqual(c.text, "")

    def test_too_long_code_rejected_early(self):
        c = MorseComposer()
        for _ in range(6):
            c.add_symbol("-")
        event = c.add_symbol("-")
        self.assertEqual(event.kind, EventKind.INVALID)
        self.assertEqual(c.code, "")

    def test_space_progress_for_hud(self):
        c = MorseComposer(word_gap=3.0)
        self.assertEqual(c.space_progress(1.0), -1.0)
        self.letter(c, ".")
        self.assertAlmostEqual(c.space_progress(1.5), 0.5)

    def test_delete_symbol_then_char(self):
        c = MorseComposer()
        self.letter(c, ".-")
        c.add_symbol("-")
        self.assertEqual(c.delete().value, "-")
        self.assertEqual(c.delete().value, "A")
        self.assertIsNone(c.delete())

    def test_delete_cancels_pending_space(self):
        c = MorseComposer()
        self.letter(c, ".-")
        c.delete()
        self.assertEqual(pause(c, 4.0), [])

    def test_clear(self):
        c = MorseComposer()
        self.letter(c, ".-")
        c.clear()
        self.assertEqual((c.text, c.code), ("", ""))


if __name__ == "__main__":
    unittest.main()
