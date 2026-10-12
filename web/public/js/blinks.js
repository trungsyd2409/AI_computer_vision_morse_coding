// Turns two eye-closure scores per frame into Morse dots and dashes, using
// the length of each blink like a telegraph key. Port of blink_morse/blinks.py.
//
// With time unit t (0.1 s by default):
//   both eyes closed for less than 1.5 t -> DOT  (typed when they open)
//   both eyes closed for 1.5 t or longer -> DASH (typed at 1.5 t, eyes still shut)
// The letter and word gaps are handled by the Morse composer.
// Winks (one eye only) are ignored. All times are in seconds.

export const BlinkKind = Object.freeze({ DOT: "dot", DASH: "dash" });

// Removes a person's resting eye-closure level from the raw score.
export class EyeSignal {
  constructor(initial = 0.1) {
    this.baseline = initial;
    this.value = 0; // normalised closure, 0 = open, 1 = shut
    this._t = null;
  }

  reset() {
    this._t = null;
  }

  update(raw, now, isClosed) {
    const dt = this._t === null ? 0 : Math.max(0, now - this._t);
    this._t = now;

    // Learn only from frames where the eye is open. The baseline follows the
    // lower envelope of the signal: quick to drop, slow to rise.
    if (!isClosed && raw < this.baseline + 0.3) {
      const tau = raw < this.baseline ? EyeSignal.FALL_TIME : EyeSignal.RISE_TIME;
      const k = dt > 0 ? 1 - Math.exp(-dt / tau) : 0;
      this.baseline += (raw - this.baseline) * k;
      this.baseline = Math.min(Math.max(this.baseline, 0), EyeSignal.MAX_BASELINE);
    }

    const span = Math.max(1 - this.baseline, 0.2);
    this.value = Math.min(Math.max((raw - this.baseline) / span, 0), 1);
    return this.value;
  }
}
EyeSignal.MAX_BASELINE = 0.4;
EyeSignal.RISE_TIME = 2.0;
EyeSignal.FALL_TIME = 0.25;

export class BlinkDetector {
  constructor(closeThreshold = 0.45, timeUnit = 0.1) {
    this.closeThreshold = closeThreshold;
    this.timeUnit = timeUnit;

    this.left = new EyeSignal();
    this.right = new EyeSignal();
    this.leftClosed = false;
    this.rightClosed = false;

    this.pose = "open"; // open / left / right / both
    this.closed = false; // inside a both-eye blink
    this.closedSince = 0;
    this.openSince = null;

    this._dashFired = false;
    this._openFrames = 0;
    this._openStart = 0;
  }

  get dashAfter() {
    return BlinkDetector.DASH_SPLIT * this.timeUnit;
  }

  // True once the current blink has already been typed as a dash.
  get dashFired() {
    return this._dashFired;
  }

  reset(now = null) {
    this.left.reset();
    this.right.reset();
    this.leftClosed = false;
    this.rightClosed = false;
    this.pose = "open";
    this.closed = false;
    this._dashFired = false;
    this._openFrames = 0;
    if (now !== null) this.openSince = now;
  }

  closedTime(now) {
    return this.closed ? now - this.closedSince : 0;
  }

  // Seconds since the eyes opened after the last blink (0 while shut).
  pauseTime(now) {
    if (this.closed || this.openSince === null) return 0;
    return Math.max(0, now - this.openSince);
  }

  // Feeds one frame of raw blink scores and returns the symbols it typed.
  update(leftRaw, rightRaw, now) {
    const nl = this.left.update(leftRaw, now, this.leftClosed);
    const nr = this.right.update(rightRaw, now, this.rightClosed);
    this.leftClosed = this._hysteresis(nl, this.leftClosed);
    this.rightClosed = this._hysteresis(nr, this.rightClosed);
    this.pose = this._classify();
    if (this.openSince === null) this.openSince = now;

    const both = this.pose === "both";
    if (!this.closed) {
      if (both) {
        this.closed = true;
        this.closedSince = now;
        this._dashFired = false;
        this._openFrames = 0;
      }
      return [];
    }

    // Inside a blink.
    if (both) {
      this._openFrames = 0;
      if (!this._dashFired && now - this.closedSince >= this.dashAfter) {
        this._dashFired = true;
        return [{ kind: BlinkKind.DASH, started: this.closedSince, fired: now }];
      }
      return [];
    }

    // The eyes are opening. End the blink only after they stay open for a
    // couple of frames, so one noisy frame cannot split it in two.
    if (this._openFrames === 0) this._openStart = now;
    this._openFrames += 1;
    if (this._openFrames < BlinkDetector.OPEN_FRAMES) return [];

    this.closed = false;
    this.openSince = this._openStart;
    if (this._dashFired) return [];
    return [{ kind: BlinkKind.DOT, started: this.closedSince, fired: this._openStart }];
  }

  _hysteresis(value, wasClosed) {
    if (wasClosed) return value > this.closeThreshold - BlinkDetector.RELEASE_GAP;
    return value > this.closeThreshold;
  }

  _classify() {
    if (this.leftClosed && this.rightClosed) return "both";
    if (this.leftClosed) return "left";
    if (this.rightClosed) return "right";
    return "open";
  }
}
BlinkDetector.RELEASE_GAP = 0.15;
BlinkDetector.DASH_SPLIT = 1.5;
BlinkDetector.OPEN_FRAMES = 2;
