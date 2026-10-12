// Settings kept in localStorage. Every access is wrapped in try/catch because
// storage can be blocked (private window, strict privacy settings).

const SETTINGS_KEY = "xeno.settings.v1";
const IMAGES_KEY = "xeno.images.v1";

import { PRESETS } from "./persona.js";

// Built-in defaults. The server can lay its own on top (config.json, "defaults"),
// see applyServerDefaults().
export const DEFAULTS = Object.freeze({
  lang: "vi",
  // camera
  deviceId: "",
  mirror: true,
  swapEyes: false,
  closeThreshold: 0.45,
  // typing
  unitMs: 100,
  sendDelay: 3,
  userSound: true,
  // AI
  personaPreset: "funny",
  personaCustom: { vi: "", en: "" },
  aiSpeed: 1,
  maxChars: 40,
  // sound
  sound: true,
  volume: 0.5,
  // character
  characterMode: "designed",
});

const RANGES = {
  closeThreshold: [0.25, 0.7],
  unitMs: [60, 300],
  sendDelay: [1.5, 8],
  aiSpeed: [0.5, 3],
  maxChars: [20, 120],
  volume: [0, 1],
};

// What a fresh visitor starts with: DEFAULTS plus whatever the server sends.
let BASE = { ...DEFAULTS, personaCustom: { ...DEFAULTS.personaCustom } };

// Settings a visitor with the user password may still change: the header buttons
// plus the three personal tuning sliders (eye threshold, time unit t, auto-send delay).
export const USER_KEYS = ["lang", "sound", "userSound", "volume", "closeThreshold", "unitMs", "sendDelay"];

function clamp(value, [lo, hi], fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
}

export function sanitizeSettings(raw) {
  const s = { ...BASE, personaCustom: { ...BASE.personaCustom } };
  if (!raw || typeof raw !== "object") return s;
  if (raw.lang === "vi" || raw.lang === "en") s.lang = raw.lang;
  if (typeof raw.deviceId === "string") s.deviceId = raw.deviceId.slice(0, 200);
  for (const key of ["mirror", "swapEyes", "userSound", "sound"]) {
    if (typeof raw[key] === "boolean") s[key] = raw[key];
  }
  for (const [key, range] of Object.entries(RANGES)) {
    if (key in raw) s[key] = clamp(raw[key], range, BASE[key]);
  }
  s.maxChars = Math.round(s.maxChars);
  // The built-in names, or any personality the server listed in config.json ("personas").
  if (["funny", "calm", "teacher", "custom"].includes(raw.personaPreset) ||
      (typeof raw.personaPreset === "string" && Object.hasOwn(PRESETS, raw.personaPreset))) {
    s.personaPreset = raw.personaPreset;
  }
  if (raw.personaCustom && typeof raw.personaCustom === "object") {
    for (const l of ["vi", "en"]) {
      if (typeof raw.personaCustom[l] === "string") s.personaCustom[l] = raw.personaCustom[l].slice(0, 800);
    }
  }
  if (raw.characterMode === "designed" || raw.characterMode === "custom") s.characterMode = raw.characterMode;
  return s;
}

// Lays the server's defaults over the built-in ones. Call once, before loadSettings().
export function applyServerDefaults(serverSettings) {
  BASE = { ...DEFAULTS, personaCustom: { ...DEFAULTS.personaCustom } };
  BASE = sanitizeSettings(serverSettings && typeof serverSettings === "object" ? serverSettings : null);
}

export function baseSettings() {
  return { ...BASE, personaCustom: { ...BASE.personaCustom } };
}

function sameValue(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function readStored() {
  try {
    const text = localStorage.getItem(SETTINGS_KEY);
    const raw = text ? JSON.parse(text) : null;
    return raw && typeof raw === "object" ? raw : null;
  } catch {
    return null;
  }
}

// Only what differs from the defaults is stored, so a later change to the
// defaults reaches everyone who never touched that setting.
// `onlyKeys` limits the visitor to those settings: the rest of what is stored
// (for example the master's choices on a shared browser) is left alone and ignored.
export function loadSettings(onlyKeys = null) {
  let raw = readStored();
  if (raw && onlyKeys) raw = Object.fromEntries(Object.entries(raw).filter(([k]) => onlyKeys.includes(k)));
  return sanitizeSettings(raw);
}

export function saveSettings(settings, onlyKeys = null) {
  const diff = {};
  for (const key of Object.keys(BASE)) {
    if (!sameValue(settings[key], BASE[key])) diff[key] = settings[key];
  }
  let out = diff;
  if (onlyKeys) {
    out = { ...(readStored() || {}) };
    for (const key of onlyKeys) {
      if (key in diff) out[key] = diff[key];
      else delete out[key];
    }
  }
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(out));
  } catch { /* storage unavailable, keep working in memory */ }
}

export function clearSettings() {
  try { localStorage.removeItem(SETTINGS_KEY); } catch { /* ignore */ }
}

export function loadImages() {
  try {
    const raw = JSON.parse(localStorage.getItem(IMAGES_KEY) || "null");
    const ok = (v) => typeof v === "string" && v.startsWith("data:image/");
    return { open: ok(raw?.open) ? raw.open : null, closed: ok(raw?.closed) ? raw.closed : null };
  } catch {
    return { open: null, closed: null };
  }
}

// Returns false when the browser refuses (quota), so the caller can warn.
export function saveImages(images) {
  try {
    if (!images.open && !images.closed) localStorage.removeItem(IMAGES_KEY);
    else localStorage.setItem(IMAGES_KEY, JSON.stringify(images));
    return true;
  } catch {
    return false;
  }
}
