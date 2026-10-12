// Wires everything together: start screen, camera, the frame loop, keyboard
// shortcuts and the settings drawer.

import { decode } from "./morse.js";
import { Sound } from "./audio.js";
import { Character } from "./character.js";
import { Chart, Chips, Ring, Strip } from "./hud.js";
import { MorseInput } from "./input.js";
import { ApiClient, Conversation, sentenceCase } from "./chat.js";
import { applyI18n, getLang, onLang, setLang, t } from "./i18n.js";
import { createDrawer } from "./settings.js";
import {
  clearSettings, loadImages, loadSettings, sanitizeSettings, saveImages, saveSettings,
} from "./store.js";
import { Tracker } from "./tracker.js";

const $ = (id) => document.getElementById(id);

// -- state --------------------------------------------------------------------

let settings = loadSettings();
let images = loadImages();
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
sound.enabled = settings.sound;
sound.volume = settings.volume;

const input = new MorseInput(settings);
const api = new ApiClient();

const video = $("cam");
const camStage = $("cam-stage");
const character = new Character($("character"));
character.setMode(settings.characterMode, images);

const chart = new Chart($("chart-grid"), $("chart-note"));
const userStrip = new Strip($("user-strip"), $("user-strip-live"));
const aiStrip = new Strip($("ai-strip"), $("ai-strip-live"));
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

// -- small DOM helpers -------------------------------------------------------------

function toast(message, isError = false) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("error", isError);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3500);
}

function addMessage(who, text) {
  const log = $("chat-log");
  const wrap = document.createElement("div");
  wrap.className = `msg ${who}`;
  const label = document.createElement("span");
  label.className = "who";
  label.dataset.who = who;
  label.textContent = t(`who_${who}`);
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.append(label, bubble);
  log.append(wrap);
  log.scrollTop = log.scrollHeight;
}

function addNote(text) {
  const log = $("chat-log");
  const wrap = document.createElement("div");
  wrap.className = "msg note";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.append(bubble);
  log.append(wrap);
  log.scrollTop = log.scrollHeight;
}

function setText(el, value) {
  if (el.textContent !== value) el.textContent = value;
}

// The line under a face: finished text plus the letter being typed.
function renderLine(el, text, pendingCode) {
  const key = `${text}|${pendingCode}`;
  if (el._key === key) return;
  el._key = key;
  const nodes = [document.createTextNode(sentenceCase(text))];
  if (pendingCode) {
    const span = document.createElement("span");
    span.className = "pending";
    span.textContent = (decode(pendingCode) || "?").toLowerCase();
    nodes.push(span);
  }
  el.replaceChildren(...nodes);
}

function setStatusDot(dot, state) {
  dot.className = `dot ${state}`.trim();
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
  const userTurn = conv.phase === "user";
  const started = conv.phase !== "idle";

  // user side
  const liveSym = userTurn && det.closed && !det.dashFired ? "." : "";
  userStrip.set(userTurn ? composer.code : "", liveSym);
  renderLine($("user-line"), userTurn ? composer.text : "", userTurn ? composer.code : "");
  $("user-caret").hidden = !(userTurn && !conv.paused);

  // AI side
  const ai = conv.ai;
  aiStrip.set(ai.code, ai.live);
  renderLine($("ai-line"), ai.text, "");
  $("ai-caret").hidden = conv.phase !== "ai";
  $("btn-skip").hidden = conv.phase !== "ai";
  character.setClosed(ai.closed);
  character.setState(conv.phase === "sending" ? "thinking" : conv.phase === "ai" ? "talking" : "idle");

  // chart follows whoever is typing
  chart.update(conv.phase === "ai" ? ai.code : userTurn ? composer.code : "");

  // countdown ring and send button
  const cd = conv.countdown(nowMs);
  if (cd && live) ring.show(cd.progress, cd.left);
  else ring.hide();
  $("btn-send").disabled = !conv.canSend();

  // eye chips
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
  setStatusDot($("user-dot"), !started ? "" : keys.user === "stNoFace" ? "bad" : keys.user === "stYourTurn" || keys.user === "stTyping" ? "on" : "");
  setStatusDot($("ai-dot"), keys.ai === "stAiTyping" ? "on" : keys.ai === "stThinking" ? "wait" : "");

  const hint = $("cam-hint");
  const showHint = mode === "camera" && started && !faceOk;
  hint.hidden = !showHint;
  if (showHint) setText(hint, t("hintNoFace"));

  // the user's own beep follows the eyes
  if (settings.userBeep && conv.acceptsInput && det.closed) sound.start("user");
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
  const prev = settings;
  settings = sanitizeSettings({ ...settings, ...patch });
  saveSettings(settings);
  applySettings(prev);
}

function applySettings(prev) {
  input.apply(settings);
  sound.enabled = settings.sound;
  sound.volume = settings.volume;
  if (!settings.sound) sound.stopAll();
  video.classList.toggle("mirror", settings.mirror);
  $("btn-mute").setAttribute("aria-pressed", String(!settings.sound));
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
    images = next;
    const ok = saveImages(next);
    character.setMode(settings.characterMode, images);
    return ok;
  },
  listCameras: () => Tracker.listCameras(),
  onReset() {
    clearSettings();
    const prev = settings;
    settings = sanitizeSettings(null);
    applySettings(prev);
    saveSettings(settings);
  },
  toast,
});

onLang(() => {
  document.querySelectorAll(".msg .who").forEach((el) => { el.textContent = t(`who_${el.dataset.who}`); });
  chart.refreshText();
  showModeLine();
  refreshPauseLabel();
  drawer.sync();
  $("btn-lang").textContent = getLang().toUpperCase();
  redrawLines();
});

function redrawLines() {
  // force the lines and statuses to redraw in the new language
  $("user-line")._key = null;
  $("ai-line")._key = null;
}

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

function toggleMute() {
  update({ sound: !settings.sound });
  drawer.sync();
  toast(t(settings.sound ? "unmuted" : "muted"));
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
$("btn-close-settings").addEventListener("click", () => drawer.close());
$("btn-pause").addEventListener("click", togglePause);
$("btn-mute").addEventListener("click", toggleMute);
$("btn-send").addEventListener("click", () => conv.send(performance.now()));
$("btn-skip").addEventListener("click", () => conv.skipAi());
$("btn-lang").addEventListener("click", () => update({ lang: getLang() === "vi" ? "en" : "vi" }));
$("btn-clear-chat").addEventListener("click", () => {
  $("chat-log").replaceChildren();
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

applyI18n(document);
$("btn-lang").textContent = getLang().toUpperCase();
$("btn-mute").setAttribute("aria-pressed", String(!settings.sound));
refreshPauseLabel();
if (!navigator.mediaDevices?.getUserMedia) $("btn-start-cam").disabled = true;

api.status().then((status) => {
  serverStatus = status;
  showModeLine();
});
showModeLine();

requestAnimationFrame(loop);
