// Plays a text as Morse "blinks" on a timeline, the way the AI answers.
// One time unit U (ms): dot = 1U closed, dash = 3U closed, 1U between
// symbols, 3U between letters, 7U between words. These are the same rules
// the user's blinks follow in the detector.

import { MORSE_TABLE } from "./morse.js";

export const START_DELAY_MS = 600;
export const END_PAD_MS = 500;

// Returns the events of the reply in time order (times in ms from start).
// kinds: close {sym}, open {sym}, letter {ch}, space, done
export function buildTimeline(text, unitMs) {
  const U = unitMs;
  const events = [];
  let t = START_DELAY_MS;
  const words = String(text).split(/\s+/).filter(Boolean);
  words.forEach((word, wi) => {
    for (const ch of word) {
      const code = MORSE_TABLE[ch];
      if (!code) continue;
      for (const sym of code) {
        events.push({ t, k: "close", sym });
        t += (sym === "." ? 1 : 3) * U;
        events.push({ t, k: "open", sym });
        t += U; // gap after every symbol
      }
      t += 2 * U; // 3U in total between letters
      events.push({ t, k: "letter", ch });
    }
    if (wi < words.length - 1) {
      t += 4 * U; // 7U in total between words
      events.push({ t, k: "space" });
    }
  });
  t += END_PAD_MS;
  events.push({ t, k: "done" });
  return events;
}

export function durationMs(text, unitMs) {
  const tl = buildTimeline(text, unitMs);
  return tl[tl.length - 1].t;
}

// Drives a timeline. Call tick(nowMs) every frame. Events that arrive more
// than LATE_MS after their time (tab was in the background) are flagged
// `late` so the caller can skip sounds.
export class Player {
  static LATE_MS = 150;

  constructor(onEvent) {
    this.onEvent = onEvent;
    this.events = [];
    this.index = 0;
    this.startedAt = 0;
    this.running = false;
  }

  start(text, unitMs, nowMs) {
    this.events = buildTimeline(text, unitMs);
    this.index = 0;
    this.startedAt = nowMs;
    this.running = true;
  }

  stop() {
    this.running = false;
    this.events = [];
    this.index = 0;
  }

  tick(nowMs) {
    if (!this.running) return;
    const elapsed = nowMs - this.startedAt;
    while (this.index < this.events.length && this.events[this.index].t <= elapsed) {
      const e = this.events[this.index++];
      this._emit(e, elapsed - e.t > Player.LATE_MS);
    }
  }

  // Reveals the rest of the reply at once (skip button).
  skip() {
    if (!this.running) return;
    while (this.index < this.events.length) {
      const e = this.events[this.index++];
      if (e.k === "close" || e.k === "open") continue;
      this._emit(e, true);
    }
  }

  _emit(e, late) {
    if (e.k === "done") this.running = false;
    this.onEvent({ ...e, late });
  }
}
