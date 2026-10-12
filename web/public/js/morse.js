// International Morse table and the state machine that turns symbols and
// pauses into text. Nothing here touches the camera or the DOM, so it can
// be tested on its own. This is a direct port of blink_morse/morse.py.

// ITU-R M.1677-1 codes for letters, digits and the common punctuation.
export const MORSE_TABLE = {
  A: ".-", B: "-...", C: "-.-.", D: "-..", E: ".", F: "..-.",
  G: "--.", H: "....", I: "..", J: ".---", K: "-.-", L: ".-..",
  M: "--", N: "-.", O: "---", P: ".--.", Q: "--.-", R: ".-.",
  S: "...", T: "-", U: "..-", V: "...-", W: ".--", X: "-..-",
  Y: "-.--", Z: "--..",
  0: "-----", 1: ".----", 2: "..---", 3: "...--", 4: "....-",
  5: ".....", 6: "-....", 7: "--...", 8: "---..", 9: "----.",
  ".": ".-.-.-", ",": "--..--", "?": "..--..", "!": "-.-.--",
  "'": ".----.", "/": "-..-.", "(": "-.--.", ")": "-.--.-",
  ":": "---...", "=": "-...-", "+": ".-.-.", "-": "-....-",
  "@": ".--.-.",
};

export const CODE_TO_CHAR = Object.fromEntries(
  Object.entries(MORSE_TABLE).map(([char, code]) => [code, char])
);

export const MAX_CODE_LENGTH = Math.max(
  ...Object.values(MORSE_TABLE).map((code) => code.length)
);

export function decode(code) {
  return Object.prototype.hasOwnProperty.call(CODE_TO_CHAR, code)
    ? CODE_TO_CHAR[code]
    : null;
}

// Characters whose code starts with `prefix`, shortest code first.
export function candidates(prefix) {
  if (!prefix) return [];
  const matches = Object.keys(MORSE_TABLE).filter((c) =>
    MORSE_TABLE[c].startsWith(prefix)
  );
  return matches.sort((a, b) => {
    const diff = MORSE_TABLE[a].length - MORSE_TABLE[b].length;
    if (diff !== 0) return diff;
    return a < b ? -1 : a > b ? 1 : 0;
  });
}

export const EventKind = Object.freeze({
  SYMBOL: "symbol",
  LETTER: "letter",
  SPACE: "space",
  INVALID: "invalid",
  DELETE: "delete",
  CLEAR: "clear",
});

// Builds a message from Morse input and uses silences as separators.
// `update(pause)` is called every frame with the idle time since the last
// blink: a pause of `letterGap` ends the letter, `wordGap` adds a space.
export class MorseComposer {
  constructor(letterGap = 0.3, wordGap = 0.7, maxText = 200) {
    this.letterGap = letterGap;
    this.wordGap = wordGap;
    this.maxText = maxText;
    this.code = ""; // symbols of the letter in progress
    this.text = ""; // the decoded message
    this._spacePending = false;
  }

  addSymbol(symbol) {
    if (symbol !== "." && symbol !== "-") {
      throw new Error(`Unsupported Morse symbol: ${symbol}`);
    }
    // A new symbol before the word gap means the word goes on.
    this._spacePending = false;

    // No code is longer than MAX_CODE_LENGTH, so reject early.
    if (this.code.length >= MAX_CODE_LENGTH) {
      const rejected = this.code + symbol;
      this.code = "";
      return { kind: EventKind.INVALID, value: rejected };
    }
    this.code += symbol;
    return { kind: EventKind.SYMBOL, value: symbol };
  }

  endLetter() {
    if (!this.code) return null;
    const event = this._commit();
    this._spacePending = event.kind === EventKind.LETTER;
    return event;
  }

  update(pause) {
    const events = [];
    if (this.code && pause >= this.letterGap) {
      events.push(this.endLetter());
    }
    if (this._spacePending && pause >= this.wordGap) {
      this._spacePending = false;
      const space = this._insertSpace();
      if (space) events.push(space);
    }
    return events;
  }

  // What the current pause is counting towards: ["letter", progress],
  // ["space", progress] or [null, 0].
  pauseState(pause) {
    if (this.code) return ["letter", Math.min(1, pause / this.letterGap)];
    if (this._spacePending) return ["space", Math.min(1, pause / this.wordGap)];
    return [null, 0];
  }

  delete() {
    this._spacePending = false;
    if (this.code) {
      const removed = this.code[this.code.length - 1];
      this.code = this.code.slice(0, -1);
      return { kind: EventKind.DELETE, value: removed };
    }
    if (this.text) {
      const removed = this.text[this.text.length - 1];
      this.text = this.text.slice(0, -1);
      return { kind: EventKind.DELETE, value: removed };
    }
    return null;
  }

  clear() {
    this._spacePending = false;
    this.code = "";
    this.text = "";
    return { kind: EventKind.CLEAR, value: "" };
  }

  _commit() {
    const code = this.code;
    this.code = "";
    const char = decode(code);
    if (char === null) return { kind: EventKind.INVALID, value: code };
    this._append(char);
    return { kind: EventKind.LETTER, value: char };
  }

  _insertSpace() {
    // Collapse repeated spaces and ignore a space at the very start.
    if (!this.text || this.text.endsWith(" ")) return null;
    this._append(" ");
    return { kind: EventKind.SPACE, value: " " };
  }

  _append(char) {
    this.text = (this.text + char).slice(-this.maxText);
  }
}
