// Personality presets for Xeno Assistant. The texts live on the server
// (server.py, and config.json under "personas" to change them). The page gets
// them from /api/defaults so there is one copy to edit.

export const PRESETS = {};

export function setPresets(presets) {
  for (const key of Object.keys(PRESETS)) delete PRESETS[key];
  if (presets && typeof presets === "object") Object.assign(PRESETS, presets);
}

export const PRESET_IDS = ["funny", "calm", "teacher", "custom"];

// The text that goes to the server for the current settings.
// (The server ignores it unless the visitor signed in with the master password.)
export function personaText(settings, lang) {
  if (settings.personaPreset === "custom") {
    return (settings.personaCustom?.[lang] || "").trim();
  }
  return (PRESETS[settings.personaPreset] || PRESETS.funny || {})[lang] || "";
}
