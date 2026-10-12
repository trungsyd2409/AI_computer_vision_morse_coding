// Small Web Audio synth. Two kinds of sound:
//  - held tones, one voice per speaker, kept for as long as the eyes are
//    closed (like a telegraph sounder): the AI is a lower triangle wave,
//    the user a higher, quieter sine;
//  - short cues for the user: a tick when a letter is accepted, a low
//    buzz when a code matches nothing, two rising notes when a message sends.

const VOICES = {
  ai: { freq: 440, type: "triangle", gain: 1.0 },
  user: { freq: 660, type: "sine", gain: 0.6 },
};

const CUES = {
  letter: [{ freq: 988, ms: 45, gain: 0.5 }],
  invalid: [{ freq: 180, ms: 110, gain: 0.7, type: "square" }],
  send: [{ freq: 523, ms: 70, gain: 0.55 }, { freq: 784, ms: 110, gain: 0.55, at: 80 }],
};

export class Sound {
  constructor() {
    this.ctx = null;
    // Xeno's beep and the user's own sounds can be switched separately.
    this.channel = { ai: true, user: true };
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
    if (!this.channel[who] || !this.ctx || this.active[who] || this.volume <= 0) return;
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

  // A named short cue: "letter", "invalid" or "send".
  cue(name) {
    if (!this.channel.user || !this.ctx || this.volume <= 0) return;
    const notes = CUES[name];
    if (!notes) return;
    const ctx = this.ctx;
    const base = ctx.currentTime;
    for (const note of notes) {
      const start = base + (note.at || 0) / 1000;
      const end = start + note.ms / 1000;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = note.type || "sine";
      osc.frequency.value = note.freq;
      const peak = 0.22 * this.volume * note.gain;
      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(peak, start + 0.006);
      gain.gain.linearRampToValueAtTime(0, end);
      osc.connect(gain).connect(ctx.destination);
      osc.start(start);
      osc.stop(end + 0.02);
    }
  }
}
