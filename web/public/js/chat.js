// The conversation: whose turn it is, the auto-send countdown, the request to
// the server, and playing the AI's answer as Morse blinks.
//
//   idle -> user -> sending -> ai -> user -> ...
//
// While it is not the user's turn the detector keeps running (so the eye
// chips stay live) but its symbols are thrown away.

import { Player } from "./player.js";
import { sanitize } from "./sanitize.js";
import { personaText } from "./persona.js";
import { getLang, t } from "./i18n.js";

// The AI's time unit never goes below this, so a dot lasts at least two
// frames at 60 Hz and the eyes can be seen closing at the 3x speed.
export const MIN_AI_UNIT_MS = 30;

export function aiUnitMs(settings) {
  return Math.max(MIN_AI_UNIT_MS, settings.unitMs / settings.aiSpeed);
}

export class ApiClient {
  async status() {
    try {
      const res = await fetch("/api/status", { cache: "no-store" });
      return res.ok ? await res.json() : null;
    } catch {
      return null;
    }
  }

  async chat(payload) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 30000);
    let res;
    try {
      res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: ctrl.signal,
      });
    } catch {
      throw new Error(t("errNetwork"));
    } finally {
      clearTimeout(timer);
    }
    let data = null;
    try { data = await res.json(); } catch { /* not JSON */ }
    if (res.status === 401) {
      this.onAuthLost?.();   // the session ended: the page asks for the password again
      throw new Error(t("sessionExpired"));
    }
    if (!res.ok) throw new Error(data?.message || t("errNetwork"));
    return data;
  }
}

export function sentenceCase(text) {
  const s = String(text || "").toLowerCase();
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : "";
}

export class Conversation {
  // hooks: { onMessage(who, text), onNote(text), onPhase(phase) }
  constructor({ input, getSettings, sound, api, hooks }) {
    this.input = input;
    this.getSettings = getSettings;
    this.sound = sound;
    this.api = api;
    this.hooks = hooks;

    this.phase = "idle";
    this.paused = false;
    this.lastTurn = null;     // { user, assistant }, the one remembered turn
    this._token = 0;          // changes when the chat is cleared mid-request
    this._pendingUser = "";
    this._record = true;

    this.ai = this._blankAi();
    this.player = new Player((e) => this._onPlayerEvent(e));
  }

  _blankAi() {
    return { code: "", live: "", text: "", closed: false };
  }

  // -- turns ---------------------------------------------------------------

  begin(nowMs, greet) {
    if (greet) {
      this._playAi(nowMs, greet, false, "");
    } else {
      this.userTurn(nowMs);
    }
  }

  userTurn(nowMs) {
    this._setPhase("user");
    this.ai = this._blankAi();
    this.sound.stop("ai");
    this.input.clear();
    this.input.reset(nowMs / 1000);
  }

  _setPhase(phase) {
    this.phase = phase;
    this.hooks.onPhase?.(phase);
  }

  get acceptsInput() {
    return this.phase === "user" && !this.paused;
  }

  // One sample from the camera or keyboard. Returns the composer events.
  onSample(left, right, tSec) {
    const events = this.input.feed(left, right, tSec, this.acceptsInput);
    if (events.length) this.hooks.onEvents?.(events);
    return events;
  }

  // Call every animation frame. `live` is false while no face is visible.
  frame(nowMs, live) {
    this.player.tick(nowMs);
    if (!this.acceptsInput || !live) return;
    const t = nowMs / 1000;
    this.hooks.onEvents?.(this.input.tick(t));
    const text = this.input.composer.text.trim();
    if (!this.input.composer.code && text && !this.input.detector.closed) {
      if (this.input.pause(t) >= this.getSettings().sendDelay) this.send(nowMs);
    }
  }

  // Seconds-based view of the countdown, or null when nothing is waiting.
  countdown(nowMs) {
    if (!this.acceptsInput) return null;
    const composer = this.input.composer;
    if (composer.code || !composer.text.trim() || this.input.detector.closed) return null;
    const total = this.getSettings().sendDelay;
    const pause = this.input.pause(nowMs / 1000);
    return { progress: pause / total, left: total - pause };
  }

  canSend() {
    return this.acceptsInput && !!this.input.composer.text.trim();
  }

  deleteLast() {
    if (!this.acceptsInput) return;
    this.input.composer.delete();
  }

  togglePause(nowMs) {
    this.paused = !this.paused;
    if (!this.paused) this.input.reset(nowMs / 1000);
    return this.paused;
  }

  // -- sending -------------------------------------------------------------

  async send(nowMs) {
    if (!this.canSend()) return;
    const composer = this.input.composer;
    // A letter still being typed is finished first, so nothing is lost.
    if (composer.code) composer.endLetter();
    const text = composer.text.trim().toLowerCase();
    if (!text) return;

    const settings = this.getSettings();
    const lang = getLang();
    this._pendingUser = text;
    this.hooks.onMessage("user", sentenceCase(text));
    composer.clear();
    this.hooks.onSent?.();
    this._setPhase("sending");
    const token = this._token;

    try {
      const data = await this.api.chat({
        message: text,
        history: this.lastTurn ? [this.lastTurn] : [],
        lang,
        persona: personaText(settings, lang),
        maxChars: settings.maxChars,
      });
      if (token !== this._token) return;
      const reply = sanitize(data.reply, settings.maxChars);
      if (!reply) throw new Error("empty reply");
      // Models were set up but none answered: say so, so the canned line is not a surprise.
      if (data.fallback) this.hooks.onNote(t("fallbackNote"));
      this._playAi(performance.now(), reply, true, text);
    } catch (err) {
      if (token !== this._token) return;
      this.hooks.onNote(t("errReply", { msg: err.message }));
      this.userTurn(performance.now());
    }
  }

  // -- the AI speaks ----------------------------------------------------------

  _playAi(nowMs, text, record, userText) {
    this._setPhase("ai");
    this.ai = this._blankAi();
    this._record = record;
    this._pendingUser = userText;
    this._reply = text;
    const settings = this.getSettings();
    // The AI uses the same time unit t as the user, scaled by its speed.
    this.player.start(text, aiUnitMs(settings), nowMs);
  }

  skipAi() {
    if (this.phase === "ai") this.player.skip();
  }

  _onPlayerEvent(e) {
    const ai = this.ai;
    switch (e.k) {
      case "close":
        ai.closed = true;
        ai.live = e.sym;
        if (!e.late) this.sound.start("ai");
        break;
      case "open":
        ai.closed = false;
        ai.live = "";
        ai.code += e.sym;
        this.sound.stop("ai");
        break;
      case "letter":
        ai.text += e.ch;
        ai.code = "";
        break;
      case "space":
        ai.text += " ";
        break;
      case "done": {
        this.sound.stop("ai");
        const reply = this._reply;
        this.hooks.onMessage("ai", sentenceCase(reply));
        if (this._record) this.lastTurn = { user: this._pendingUser, assistant: reply.toLowerCase() };
        this.userTurn(performance.now());
        break;
      }
      default:
        break;
    }
  }

  // -- housekeeping ---------------------------------------------------------------

  clearChat(nowMs) {
    this._token += 1;
    this.lastTurn = null;
    this.player.stop();
    this.sound.stop("ai");
    if (this.phase !== "idle") this.userTurn(nowMs);
  }

  statusKeys(faceOk) {
    const c = this.input.composer;
    const typing = !!(c.code || c.text || this.input.detector.closed);
    let user;
    if (this.phase === "idle") user = "stWaitStart";
    else if (this.paused) user = "stPaused";
    else if (this.phase === "sending") user = "stSent";
    else if (this.phase === "ai") user = "stReadingAi";
    else if (!faceOk) user = "stNoFace";
    else user = typing ? "stTyping" : "stYourTurn";

    let ai;
    if (this.phase === "ai") ai = "stAiTyping";
    else if (this.phase === "sending") ai = "stThinking";
    else if (this.phase === "user" && typing) ai = "stReadingYou";
    else ai = "stReady";
    return { user, ai };
  }
}
