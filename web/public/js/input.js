// Joins the blink detector and the Morse composer, and keeps their timing
// settings in step with the user's time unit.

import { BlinkDetector, BlinkKind } from "./blinks.js";
import { MorseComposer } from "./morse.js";

export class MorseInput {
  constructor(settings) {
    this.detector = new BlinkDetector(settings.closeThreshold, settings.unitMs / 1000);
    this.composer = new MorseComposer(settings.unitMs * 3 / 1000, settings.unitMs * 7 / 1000, 200);
    this.apply(settings);
  }

  apply(settings) {
    const unit = settings.unitMs / 1000;
    this.detector.closeThreshold = settings.closeThreshold;
    this.detector.timeUnit = unit;
    this.composer.letterGap = unit * 3;
    this.composer.wordGap = unit * 7;
  }

  // One camera (or keyboard) sample. The detector always runs so the eye
  // chips stay alive; symbols only reach the composer when `accept` is true
  // (it is the user's turn). Returns the composer events it caused.
  feed(leftRaw, rightRaw, t, accept = true) {
    const events = [];
    for (const blink of this.detector.update(leftRaw, rightRaw, t)) {
      if (!accept) continue;
      events.push(this.composer.addSymbol(blink.kind === BlinkKind.DOT ? "." : "-"));
    }
    return events;
  }

  // Call every animation frame while input is live.
  tick(t) {
    return this.composer.update(this.detector.pauseTime(t));
  }

  reset(t) {
    this.detector.reset(t);
  }

  clear() {
    this.composer.clear();
  }

  // Seconds since the last blink ended (0 while the eyes are shut).
  pause(t) {
    return this.detector.pauseTime(t);
  }
}
