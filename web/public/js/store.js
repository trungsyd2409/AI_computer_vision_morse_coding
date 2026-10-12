// Settings kept in localStorage. Every access is wrapped in try/catch because
// storage can be blocked (private window, strict privacy settings).

const SETTINGS_KEY = "xeno.settings.v1";
const IMAGES_KEY = "xeno.images.v1";

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
  userBeep: false,
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
  aiSpeed: [0.5, 2],
  maxChars: [20, 120],
  volume: [0, 1],
};

function clamp(value, [lo, hi], fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
}

export function sanitizeSettings(raw) {
  const s = { ...DEFAULTS, personaCustom: { ...DEFAULTS.personaCustom } };
  if (!raw || typeof raw !== "object") return s;
  if (raw.lang === "vi" || raw.lang === "en") s.lang = raw.lang;
  if (typeof raw.deviceId === "string") s.deviceId = raw.deviceId.slice(0, 200);
  for (const key of ["mirror", "swapEyes", "userBeep", "sound"]) {
    if (typeof raw[key] === "boolean") s[key] = raw[key];
  }
  for (const [key, range] of Object.entries(RANGES)) {
    if (key in raw) s[key] = clamp(raw[key], range, DEFAULTS[key]);
  }
  s.maxChars = Math.round(s.maxChars);
  if (["funny", "calm", "teacher", "custom"].includes(raw.personaPreset)) s.personaPreset = raw.personaPreset;
  if (raw.personaCustom && typeof raw.personaCustom === "object") {
    for (const l of ["vi", "en"]) {
      if (typeof raw.personaCustom[l] === "string") s.personaCustom[l] = raw.personaCustom[l].slice(0, 800);
    }
  }
  if (raw.characterMode === "designed" || raw.characterMode === "custom") s.characterMode = raw.characterMode;
  return s;
}

export function loadSettings() {
  try {
    const text = localStorage.getItem(SETTINGS_KEY);
    return sanitizeSettings(text ? JSON.parse(text) : null);
  } catch {
    return sanitizeSettings(null);
  }
}

export function saveSettings(settings) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
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
