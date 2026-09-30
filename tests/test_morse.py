"""Tests for the Morse table and the composer state machine."""

import unittest

from morse_hand.morse import (MORSE_TABLE, EventKind, MorseComposer, candidates,
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


class ComposerTest(unittest.TestCase):
    def type_letter(self, composer, code, now):
        for symbol in code:
            composer.add_symbol(symbol)
        return composer.ring_tap(now)

    def test_single_ring_commits_letter(self):
        c = MorseComposer()
        event = self.type_letter(c, "....", 1.0)
        self.assertEqual(event.kind, EventKind.LETTER)
        self.assertEqual(c.text, "H")
        self.assertEqual(c.code, "")

    def test_double_ring_adds_space(self):
        c = MorseComposer(double_tap_window=0.4)
        self.type_letter(c, "..", 1.0)
        event = c.ring_tap(1.3)
        self.assertEqual(event.kind, EventKind.SPACE)
        self.assertEqual(c.text, "I ")

    def test_slow_second_ring_is_not_space(self):
        c = MorseComposer(double_tap_window=0.4)
        self.type_letter(c, "..", 1.0)
        event = c.ring_tap(2.0)
        self.assertEqual(event.kind, EventKind.ARMED)
        self.assertEqual(c.text, "I")

    def test_symbol_between_rings_cancels_space(self):
        c = MorseComposer(double_tap_window=0.4)
        self.type_letter(c, "..", 1.0)
        c.add_symbol(".")
        event = c.ring_tap(1.2)
        self.assertEqual(event.kind, EventKind.LETTER)
        self.assertEqual(c.text, "IE")

    def test_hello_world(self):
        c = MorseComposer(double_tap_window=0.4)
        t = 0.0
        for word in ("HELLO", "WORLD"):
            if c.text:
                t += 0.2
                c.ring_tap(t)          # second tap right after the last letter
            for char in word:
                t += 1.0
                self.type_letter(c, MORSE_TABLE[char], t)
        self.assertEqual(c.text, "HELLO WORLD")

    def test_invalid_code(self):
        c = MorseComposer()
        event = self.type_letter(c, "..--", 1.0)
        self.assertEqual(event.kind, EventKind.INVALID)
        self.assertEqual(c.text, "")

    def test_too_long_code_rejected_early(self):
        c = MorseComposer()
        for _ in range(6):
            c.add_symbol("-")
        event = c.add_symbol("-")
        self.assertEqual(event.kind, EventKind.INVALID)
        self.assertEqual(c.code, "")

    def test_delete_symbol_then_char(self):
        c = MorseComposer()
        self.type_letter(c, ".-", 1.0)
        c.add_symbol("-")
        self.assertEqual(c.delete().value, "-")
        self.assertEqual(c.delete().value, "A")
        self.assertIsNone(c.delete())

    def test_take_text_empties_the_message(self):
        c = MorseComposer()
        self.type_letter(c, ".-", 1.0)
        self.assertEqual(c.take_text(), "A")
        self.assertEqual(c.text, "")
        self.assertEqual(c.take_text(), "")

    def test_clear(self):
        c = MorseComposer()
        self.type_letter(c, ".-", 1.0)
        c.clear()
        self.assertEqual((c.text, c.code), ("", ""))


if __name__ == "__main__":
    unittest.main()
