// Small Web Audio synth for the Morse beeps. One voice per speaker, held for
// as long as the eyes are closed, like a telegraph sounder.

const VOICES = {
  ai: { freq: 440, type: "triangle", gain: 1.0 },
  user: { freq: 660, type: "sine", gain: 0.55 },
};

export class Sound {
  constructor() {
    this.ctx = null;
    this.enabled = true;
    this.volume = 0.5;
    this.active = {};
  }

  // Must be called from a user gesture (the start button) so browsers allow audio.
  unlock() {
    try {
      if (!this.ctx) {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx) return;
        this.ctx = new Ctx();
      }
      if (this.ctx.state === "suspended") this.ctx.resume();
    } catch { /* audio is optional */ }
  }

  start(who) {
    if (!this.enabled || !this.ctx || this.active[who] || this.volume <= 0) return;
    const spec = VOICES[who];
    const ctx = this.ctx;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = spec.type;
    osc.frequency.value = spec.freq;
    const peak = 0.25 * this.volume * spec.gain;
    const now = ctx.currentTime;
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(peak, now + 0.008);
    osc.connect(gain).connect(ctx.destination);
    osc.start(now);
    this.active[who] = { osc, gain };
  }

  stop(who) {
    const voice = this.active[who];
    if (!voice || !this.ctx) return;
    const now = this.ctx.currentTime;
    voice.gain.gain.cancelScheduledValues(now);
    voice.gain.gain.setValueAtTime(voice.gain.gain.value, now);
    voice.gain.gain.linearRampToValueAtTime(0, now + 0.012);
    voice.osc.stop(now + 0.02);
    this.active[who] = null;
  }

  stopAll() {
    Object.keys(VOICES).forEach((who) => this.stop(who));
  }
}
