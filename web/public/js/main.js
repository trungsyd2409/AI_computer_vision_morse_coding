// Wires everything together: start screen, camera, the frame loop, keyboard
// shortcuts and the settings drawer.

import { EventKind, MORSE_TABLE, decode } from "./morse.js";
import { Sound } from "./audio.js";
import { Character } from "./character.js";
import { Chart, Chips, Ring, Tape } from "./hud.js";
import { MorseInput } from "./input.js";
import { ApiClient, Conversation, aiUnitMs, sentenceCase } from "./chat.js";
import { applyI18n, getLang, onLang, setLang, t } from "./i18n.js";
import { createDrawer } from "./settings.js";
import { setPresets } from "./persona.js";
import { fetchDefaults, gate, logout } from "./auth.js";
import {
  USER_KEYS, applyServerDefaults, baseSettings, clearSettings, loadImages, loadSettings,
  sanitizeSettings, saveImages, saveSettings,
} from "./store.js";
import { Tracker } from "./tracker.js";

const $ = (id) => document.getElementById(id);

// The tape only needs the most recent letters; older ones are off screen.
const TAPE_LETTERS = 40;

// -- state --------------------------------------------------------------------

// -- sign-in and defaults ---------------------------------------------------------

// The language for the password sheet comes from what this browser stored.
setLang(loadSettings().lang);
const session = await gate();
const isMaster = session.role === "master";
// Visitors with the user password may use the app but not open Settings, so
// only the header buttons (language, sound) can change and be remembered.
const ownKeys = isMaster ? null : USER_KEYS;

const defaults = (await fetchDefaults()) || {};
setPresets(defaults.personas);               // before the defaults: they may name one of your own personalities
applyServerDefaults(defaults.settings);
const defaultImages = { open: null, closed: null, ...(defaults.images || {}) };

let settings = loadSettings(ownKeys);
let storedImages = isMaster ? loadImages() : { open: null, closed: null };
const pickedImages = () => ({
  open: storedImages.open || defaultImages.open,
  closed: storedImages.closed || defaultImages.closed,
});
let images = pickedImages();
let mode = null;            // null | "camera" | "keyboard"
let faceOk = false;
let landmarks = null;
let lastCamAt = 0;
let keyHeld = false;
let serverStatus = null;
let toastTimer = 0;
let lastFpsText = 0;

setLang(settings.lang);

const sound = new Sound();
const input = new MorseInput(settings);
const api = new ApiClient();
api.onAuthLost = () => setTimeout(() => location.reload(), 1500);

const video = $("cam");
const camStage = $("cam-stage");
const character = new Character($("character"));
character.setMode(settings.characterMode, images);

const chart = new Chart($("chart-grid"), $("chart-note"));
const tape = new Tape($("tape-view"), $("tape-svg"));
const ring = new Ring($("ring"), $("ring-fg"), $("ring-text"));
const chips = new Chips(camStage, video, $("chip-l"), $("chip-r"));

const conv = new Conversation({
  input,
  getSettings: () => settings,
  sound,
  api,
  hooks: {
    onMessage: addMessage,
    onNote: addNote,
    onPhase: () => {},
    // Short cues for the user's own typing.
    onEvents(events) {
      for (const e of events) {
        if (!e) continue;
        if (e.kind === EventKind.LETTER) sound.cue("letter");
        else if (e.kind === EventKind.INVALID) sound.cue("invalid");
      }
    },
    onSent: () => sound.cue("send"),
  },
});

const tracker = new Tracker({
  video,
  onSample(mpLeft, mpRight, tSec, lm) {
    lastCamAt = performance.now();
    landmarks = lm;
    let left = settings.swapEyes ? mpRight : mpLeft;
    let right = settings.swapEyes ? mpLeft : mpRight;
    if (keyHeld) { left = 1; right = 1; }
    conv.onSample(left, right, tSec);
  },
  onFace(ok) {
    faceOk = ok;
    if (!ok) landmarks = null;
    input.reset(performance.now() / 1000);
  },
  onStats({ fps }) {
    const now = performance.now();
    if (now - lastFpsText < 400) return;
    lastFpsText = now;
    const badge = $("fps");
    badge.textContent = `${Math.round(fps)} FPS`;
    badge.classList.toggle("good", fps >= 20);
    badge.classList.toggle("bad", fps < 20);
  },
});

function syncAudio() {
  sound.channel.ai = settings.sound;
  sound.channel.user = settings.userSound;
  sound.volume = settings.volume;
  if (!settings.sound) sound.stop("ai");
  if (!settings.userSound) sound.stop("user");
  $("btn-mute").setAttribute("aria-pressed", String(!settings.sound && !settings.userSound));
}
syncAudio();
character.setUnit(aiUnitMs(settings));

// -- small DOM helpers -------------------------------------------------------------

function toast(message, isError = false) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("error", isError);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3500);
}

function scrollLog() {
  const log = $("chat-log");
  log.scrollTop = log.scrollHeight;
}

// Finished lines go above the row that shows the reply being typed.
function addRow(row) {
  $("chat-log").insertBefore(row, $("ai-live"));
  scrollLog();
}

function addMessage(who, text) {
  const row = document.createElement("div");
  row.className = `msg ${who}`;
  const label = document.createElement("span");
  label.className = "who";
  label.dataset.who = who;
  label.textContent = t(`who_${who}`);
  const body = document.createElement("p");
  body.className = "text";
  body.textContent = text;
  row.append(label, body);
  addRow(row);
}

function addNote(text) {
  const row = document.createElement("div");
  row.className = "msg note";
  const body = document.createElement("p");
  body.className = "text";
  body.textContent = text;
  row.append(body);
  addRow(row);
}

function setText(el, value) {
  if (el.textContent !== value) el.textContent = value;
}

// A line of text plus the letter being typed, shown in the lamp colour.
function renderLine(el, text, pendingCode) {
  const key = `${text}|${pendingCode}`;
  if (el._key === key) return;
  el._key = key;
  const nodes = [];
  const body = sentenceCase(text);
  if (body) nodes.push(document.createTextNode(body));
  if (pendingCode) {
    const span = document.createElement("span");
    span.className = "pending";
    span.textContent = (decode(pendingCode) || "?").toLowerCase();
    nodes.push(span);
  }
  el.replaceChildren(...nodes);
}

function setLamp(dot, state) {
  dot.className = `lamp ${state}`.trim();
}

// Text already typed, as tape groups: one per letter, one per word gap.
function tapeGroups(text) {
  const groups = [];
  for (const ch of text.slice(-TAPE_LETTERS)) {
    if (ch === " ") {
      if (groups.length) groups.push({ space: true });
      continue;
    }
    const up = ch.toUpperCase();
    if (MORSE_TABLE[up]) groups.push({ ch: up, code: MORSE_TABLE[up] });
  }
  return groups;
}

function renderLangButton() {
  const lang = getLang();
  $("btn-lang").children[0].classList.toggle("on", lang === "vi");
  $("btn-lang").children[1].classList.toggle("on", lang === "en");
}

// -- frame loop ----------------------------------------------------------------------

function loop() {
  const nowMs = performance.now();
  const det = input.detector;

  // Keyboard samples fill in when the camera is not delivering frames.
  const camFresh = mode === "camera" && nowMs - lastCamAt < 150;
  if (mode && !camFresh && (mode === "keyboard" || keyHeld) && (keyHeld || det.closed || det.pose !== "open")) {
    const v = keyHeld ? 1 : 0;
    conv.onSample(v, v, nowMs / 1000);
  }

  const live = mode === "keyboard" || faceOk || keyHeld || det.closed;
  conv.frame(nowMs, live);
  render(nowMs, live);
  requestAnimationFrame(loop);
}

function render(nowMs, live) {
  const det = input.detector;
  const composer = input.composer;
  const phase = conv.phase;
  const userTurn = phase === "user";
  const aiTurn = phase === "ai";
  const started = phase !== "idle";
  const ai = conv.ai;

  // the tape follows whoever is typing
  if (aiTurn) {
    tape.set(tapeGroups(ai.text), ai.code, ai.live);
  } else if (userTurn) {
    const liveSym = det.closed ? (det.dashFired ? "-" : ".") : "";
    tape.set(tapeGroups(composer.text), composer.code, liveSym);
  } else {
    tape.set([], "", "");
  }
  const tapeEmpty = aiTurn
    ? !(ai.text || ai.code || ai.live)
    : userTurn ? !(composer.text || composer.code || det.closed) : true;
  const idle = $("tape-idle");
  idle.hidden = !tapeEmpty;
  if (tapeEmpty) {
    setText(idle, t(phase === "sending" ? "tapeIdleSend" : aiTurn ? "tapeIdleAi" : "tapeIdleUser"));
  }
  setText($("tape-who"), t(aiTurn || phase === "sending" ? "who_ai" : "who_user"));

  // what the user has typed
  renderLine($("user-line"), userTurn ? composer.text : "", userTurn ? composer.code : "");
  $("user-caret").hidden = !(userTurn && !conv.paused);
  $("user-placeholder").hidden = !userTurn;

  // what Xeno is typing
  $("ai-live").hidden = !aiTurn;
  renderLine($("ai-line"), aiTurn ? ai.text : "", "");
  $("ai-caret").hidden = !aiTurn;
  $("btn-skip").hidden = !aiTurn;
  if (aiTurn) scrollLog();
  character.setClosed(ai.closed);
  character.setState(phase === "sending" ? "thinking" : aiTurn ? "talking" : "idle");

  // chart follows whoever is typing
  chart.update(aiTurn ? ai.code : userTurn ? composer.code : "");

  // countdown ring and send button
  const cd = conv.countdown(nowMs);
  if (cd && live) ring.show(cd.progress, cd.left);
  else ring.hide();
  $("btn-send").disabled = !conv.canSend();

  // eye gauges
  if (mode === "camera" && faceOk && landmarks) {
    chips.update(landmarks, {
      leftValue: det.left.value,
      rightValue: det.right.value,
      swap: settings.swapEyes,
      mirror: settings.mirror,
    });
  } else {
    chips.hide();
  }

  // status text
  const keys = conv.statusKeys(mode === "keyboard" || faceOk);
  setText($("user-status"), t(keys.user));
  setText($("ai-status"), t(keys.ai));
  setLamp($("user-dot"), !started ? "" : keys.user === "stNoFace" ? "bad" : keys.user === "stYourTurn" || keys.user === "stTyping" ? "on" : "");
  setLamp($("ai-dot"), keys.ai === "stAiTyping" ? "on" : keys.ai === "stThinking" ? "wait" : "");

  const hint = $("cam-hint");
  const showHint = mode === "camera" && started && !faceOk;
  hint.hidden = !showHint;
  if (showHint) setText(hint, t("hintNoFace"));

  // the user's own tone follows the eyes
  if (conv.acceptsInput && det.closed) sound.start("user");
  else sound.stop("user");
}

// -- start screen ------------------------------------------------------------------------

function showModeLine() {
  const el = $("ov-mode");
  if (!serverStatus) el.textContent = t("ovModeUnknown");
  else if (serverStatus.mode === "live") el.textContent = t("ovModeLive", { model: serverStatus.model });
  else el.textContent = t("ovModeMock");
}

function overlayStatus(message, isError = false) {
  const el = $("ov-status");
  el.textContent = message;
  el.classList.toggle("error", isError);
}

function setBusy(busy) {
  $("btn-start-cam").disabled = busy;
  $("btn-start-key").disabled = busy;
}

function enter(nextMode) {
  mode = nextMode;
  $("overlay").hidden = true;
  camStage.classList.toggle("live", nextMode === "camera");
  video.classList.toggle("mirror", settings.mirror);
  if (nextMode === "keyboard") {
    $("fps").textContent = "KEY";
    setText($("user-status"), t("stKeyboard"));
  }
  conv.begin(performance.now(), t("greeting"));
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    overlayStatus(t("ovNoApi"), true);
    return;
  }
  sound.unlock();
  setBusy(true);
  try {
    if (serverStatus && (!serverStatus.mediapipe || !serverStatus.model_file)) {
      overlayStatus(t("ovNoModel"), true);
      return;
    }
    overlayStatus(t("ovLoadingModel"));
    try {
      await tracker.load();
    } catch (err) {
      overlayStatus(t("ovLoadFail", { err: err?.message || String(err) }), true);
      return;
    }
    overlayStatus(t("ovOpeningCam"));
    try {
      await tracker.startCamera(settings.deviceId);
    } catch {
      overlayStatus(t("ovCamDenied"), true);
      return;
    }
    enter("camera");
  } finally {
    setBusy(false);
  }
}

function startKeyboard() {
  sound.unlock();
  enter("keyboard");
}

// -- settings ------------------------------------------------------------------------------

function update(patch) {
  if (!isMaster) patch = Object.fromEntries(Object.entries(patch).filter(([k]) => USER_KEYS.includes(k)));
  const prev = settings;
  settings = sanitizeSettings({ ...settings, ...patch });
  saveSettings(settings, ownKeys);
  applySettings(prev);
}

function applySettings(prev) {
  input.apply(settings);
  syncAudio();
  character.setUnit(aiUnitMs(settings));
  video.classList.toggle("mirror", settings.mirror);
  if (settings.lang !== prev.lang) {
    setLang(settings.lang);
  }
  if (settings.characterMode !== prev.characterMode) {
    character.setMode(settings.characterMode, images);
  }
  if (settings.deviceId !== prev.deviceId && mode === "camera") {
    tracker.startCamera(settings.deviceId).catch(() => toast(t("ovCamDenied"), true));
  }
}

const drawer = createDrawer({
  getSettings: () => settings,
  update,
  getImages: () => images,
  setImages(next) {
    // Only pictures the visitor picked are stored. The defaults come from the server.
    const own = (v) => (typeof v === "string" && v.startsWith("data:image/") ? v : null);
    storedImages = { open: own(next.open), closed: own(next.closed) };
    const ok = saveImages(storedImages);
    images = pickedImages();
    character.setMode(settings.characterMode, images);
    return ok;
  },
  defaultCharacterMode: () => (defaultImages.open && defaultImages.closed ? baseSettings().characterMode : "designed"),
  listCameras: () => Tracker.listCameras(),
  onReset() {
    clearSettings();
    const prev = settings;
    settings = sanitizeSettings(null);
    applySettings(prev);
  },
  toast,
});

onLang(() => {
  document.querySelectorAll(".msg .who").forEach((el) => { el.textContent = t(`who_${el.dataset.who}`); });
  chart.refreshText();
  showModeLine();
  refreshPauseLabel();
  drawer.sync();
  renderLangButton();
  // force the lines to redraw in the new language
  $("user-line")._key = null;
  $("ai-line")._key = null;
});

function refreshPauseLabel() {
  const btn = $("btn-pause");
  const span = btn.querySelector("span");
  span.dataset.i18n = conv.paused ? "resume" : "pause";
  span.textContent = t(span.dataset.i18n);
  btn.setAttribute("aria-pressed", String(conv.paused));
}

function togglePause() {
  if (conv.phase === "idle") return;
  const paused = conv.togglePause(performance.now());
  refreshPauseLabel();
  toast(t(paused ? "paused" : "resumed"));
}

// M mutes everything, or turns both channels back on.
function toggleMute() {
  const anyOn = settings.sound || settings.userSound;
  update({ sound: !anyOn, userSound: !anyOn });
  drawer.sync();
  toast(t(anyOn ? "muted" : "unmuted"));
}

function toggleDrawer() {
  if (drawer.isOpen) drawer.close();
  else drawer.open();
}

// -- keyboard ------------------------------------------------------------------------------------

function inTextField(target) {
  return target instanceof HTMLElement &&
    (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) &&
      !(target.tagName === "INPUT" && ["checkbox", "range", "button"].includes(target.type)));
}

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const overlayOpen = !$("overlay").hidden;
  if (e.key === "Escape") {
    if (drawer.isOpen) { drawer.close(); e.preventDefault(); }
    return;
  }
  if (inTextField(e.target)) return;

  if (e.code === "Space") {
    if (overlayOpen || drawer.isOpen) return;
    e.preventDefault();
    if (!e.repeat) keyHeld = true;
    return;
  }
  if (overlayOpen) return;
  const key = e.key.toLowerCase();
  if (key === "x") { e.preventDefault(); toggleDrawer(); return; }
  if (drawer.isOpen) return;
  if (e.key === "Enter" && !(e.target instanceof HTMLButtonElement)) {
    e.preventDefault();
    conv.send(performance.now());
  } else if (e.key === "Backspace" || e.key === "Delete") {
    e.preventDefault();
    conv.deleteLast();
  } else if (key === "p") {
    togglePause();
  } else if (key === "m") {
    toggleMute();
  }
});

document.addEventListener("keyup", (e) => {
  if (e.code !== "Space") return;
  if (keyHeld) e.preventDefault();
  keyHeld = false;
});

window.addEventListener("blur", () => { keyHeld = false; });

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) input.reset(performance.now() / 1000);
});

// Mouse clicks should not leave a button focused, or the next Space press
// would click it again.
document.addEventListener("click", (e) => {
  const button = e.target instanceof Element ? e.target.closest("button") : null;
  if (button && e.detail > 0) button.blur();
});

// -- buttons -----------------------------------------------------------------------------------------

$("btn-start-cam").addEventListener("click", startCamera);
$("btn-start-key").addEventListener("click", startKeyboard);
$("btn-settings").addEventListener("click", toggleDrawer);
$("btn-logout").addEventListener("click", async () => { await logout(); location.reload(); });
$("btn-close-settings").addEventListener("click", () => drawer.close());
$("btn-pause").addEventListener("click", togglePause);
$("btn-mute").addEventListener("click", toggleMute);
$("btn-send").addEventListener("click", () => conv.send(performance.now()));
$("btn-skip").addEventListener("click", () => conv.skipAi());
$("btn-lang").addEventListener("click", () => update({ lang: getLang() === "vi" ? "en" : "vi" }));
$("btn-clear-chat").addEventListener("click", () => {
  document.querySelectorAll("#chat-log .msg:not(#ai-live)").forEach((el) => el.remove());
  conv.clearChat(performance.now());
  toast(t("cleared"));
});
$("chart-letters").addEventListener("click", () => setChartSet("letters"));
$("chart-digits").addEventListener("click", () => setChartSet("digits"));

function setChartSet(name) {
  chart.setSet(name);
  $("chart-letters").setAttribute("aria-pressed", String(name === "letters"));
  $("chart-digits").setAttribute("aria-pressed", String(name === "digits"));
}

// -- boot ---------------------------------------------------------------------------------------------------

$("btn-settings").hidden = false;
if (!isMaster) $("drawer").dataset.lite = "";   // user password: only the three tuning sliders
$("btn-logout").hidden = !session.auth;
$("overlay").hidden = false;
applyI18n(document);
renderLangButton();
refreshPauseLabel();
if (!navigator.mediaDevices?.getUserMedia) $("btn-start-cam").disabled = true;

api.status().then((status) => {
  serverStatus = status;
  showModeLine();
});
showModeLine();

requestAnimationFrame(loop);
