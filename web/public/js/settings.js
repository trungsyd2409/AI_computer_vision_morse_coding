// The settings drawer: reads the current settings into the controls and sends
// every change back through `update`.

import { PRESETS, personaText } from "./persona.js";
import { getLang, t } from "./i18n.js";

const $ = (id) => document.getElementById(id);

// Shrinks a picked image to at most 512 px and returns a data URL, so it
// fits comfortably in localStorage.
async function readImage(file) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, 512 / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close?.();
  return canvas.toDataURL("image/webp", 0.9);
}

export function createDrawer({ getSettings, update, getImages, setImages, defaultCharacterMode, listCameras, onReset, toast }) {
  const drawer = $("drawer");
  const els = {
    device: $("set-device"), mirror: $("set-mirror"), swap: $("set-swap"),
    threshold: $("set-threshold"), unit: $("set-unit"), delay: $("set-delay"),
    userSound: $("set-usersound"), preset: $("set-preset"), persona: $("set-persona"),
    aiSpeed: $("set-aispeed"), maxChars: $("set-maxchars"),
    sound: $("set-sound"), volume: $("set-volume"),
  };
  const outs = {
    threshold: $("out-threshold"), unit: $("out-unit"), delay: $("out-delay"),
    aiSpeed: $("out-aispeed"), maxChars: $("out-maxchars"), volume: $("out-volume"),
  };

  function showOutputs(s) {
    outs.threshold.textContent = s.closeThreshold.toFixed(2);
    outs.unit.textContent = t("unitMs", { n: Math.round(s.unitMs) });
    outs.delay.textContent = t("seconds", { n: s.sendDelay.toFixed(1) });
    outs.aiSpeed.textContent = `${s.aiSpeed.toFixed(1)}x`;
    outs.maxChars.textContent = String(s.maxChars);
    outs.volume.textContent = `${Math.round(s.volume * 100)}%`;
    $("persona-count").textContent = String(els.persona.value.length);
  }

  async function fillDevices() {
    const s = getSettings();
    const cams = await listCameras();
    els.device.replaceChildren();
    const auto = document.createElement("option");
    auto.value = "";
    auto.textContent = "Auto";
    els.device.append(auto);
    cams.forEach((cam, i) => {
      const o = document.createElement("option");
      o.value = cam.deviceId;
      o.textContent = cam.label || `Camera ${i + 1}`;
      els.device.append(o);
    });
    els.device.value = cams.some((c) => c.deviceId === s.deviceId) ? s.deviceId : "";
  }

  function fillImages() {
    const imgs = getImages();
    for (const key of ["open", "closed"]) {
      const img = $(`prev-${key}`);
      if (imgs[key]) { img.src = imgs[key]; img.hidden = false; } else { img.hidden = true; img.removeAttribute("src"); }
    }
  }

  function sync() {
    const s = getSettings();
    const lang = getLang();
    els.mirror.checked = s.mirror;
    els.swap.checked = s.swapEyes;
    els.threshold.value = s.closeThreshold;
    els.unit.value = s.unitMs;
    els.delay.value = s.sendDelay;
    els.userSound.checked = s.userSound;
    els.preset.value = s.personaPreset;
    els.persona.value = personaText(s, lang);
    els.aiSpeed.value = s.aiSpeed;
    els.maxChars.value = s.maxChars;
    els.sound.checked = s.sound;
    els.volume.value = s.volume;
    drawer.querySelectorAll("[data-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
    drawer.querySelectorAll("[data-char]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.char === s.characterMode)));
    showOutputs(s);
    fillImages();
  }

  // -- wiring ---------------------------------------------------------------

  els.mirror.addEventListener("change", () => update({ mirror: els.mirror.checked }));
  els.swap.addEventListener("change", () => update({ swapEyes: els.swap.checked }));
  els.userSound.addEventListener("change", () => update({ userSound: els.userSound.checked }));
  els.sound.addEventListener("change", () => update({ sound: els.sound.checked }));
  els.device.addEventListener("change", () => update({ deviceId: els.device.value }));

  const sliders = [
    [els.threshold, "closeThreshold"], [els.unit, "unitMs"], [els.delay, "sendDelay"],
    [els.aiSpeed, "aiSpeed"], [els.maxChars, "maxChars"], [els.volume, "volume"],
  ];
  for (const [el, key] of sliders) {
    el.addEventListener("input", () => {
      update({ [key]: Number(el.value) });
      showOutputs(getSettings());
    });
  }

  els.preset.addEventListener("change", () => {
    const s = getSettings();
    const lang = getLang();
    const preset = els.preset.value;
    if (preset === "custom") {
      // Start the custom text from whatever is showing now.
      const current = els.persona.value || PRESETS.funny?.[lang] || "";
      update({ personaPreset: "custom", personaCustom: { ...s.personaCustom, [lang]: current } });
    } else {
      update({ personaPreset: preset });
    }
    els.persona.value = personaText(getSettings(), lang);
    showOutputs(getSettings());
  });

  els.persona.addEventListener("input", () => {
    const s = getSettings();
    const lang = getLang();
    update({ personaPreset: "custom", personaCustom: { ...s.personaCustom, [lang]: els.persona.value } });
    els.preset.value = "custom";
    showOutputs(getSettings());
  });

  drawer.querySelectorAll("[data-lang]").forEach((b) => {
    b.addEventListener("click", () => update({ lang: b.dataset.lang }));
  });
  drawer.querySelectorAll("[data-char]").forEach((b) => {
    b.addEventListener("click", () => {
      const imgs = getImages();
      if (b.dataset.char === "custom" && !(imgs.open && imgs.closed)) toast(t("imgNeedBoth"));
      update({ characterMode: b.dataset.char });
      sync();
    });
  });

  for (const key of ["open", "closed"]) {
    $(`img-${key}`).addEventListener("change", async (ev) => {
      const file = ev.target.files?.[0];
      ev.target.value = "";
      if (!file) return;
      try {
        const url = await readImage(file);
        const next = { ...getImages(), [key]: url };
        if (!setImages(next)) toast(t("imgBad"), true);
        fillImages();
      } catch {
        toast(t("imgBad"), true);
      }
    });
  }
  $("btn-img-reset").addEventListener("click", () => {
    setImages({ open: null, closed: null });
    update({ characterMode: defaultCharacterMode() });
    sync();
  });
  $("btn-reset-settings").addEventListener("click", () => { onReset(); sync(); });

  // The drawer slides in. `_open` is the truth; `hidden` follows once the
  // slide-out has finished.
  let _open = false;
  let hideTimer = 0;
  const SLIDE_MS = 240;

  return {
    sync,
    get isOpen() { return _open; },
    async open() {
      _open = true;
      clearTimeout(hideTimer);
      drawer.hidden = false;
      void drawer.offsetWidth; // let the browser see the closed position first
      drawer.classList.add("in");
      $("btn-settings").setAttribute("aria-expanded", "true");
      sync();
      await fillDevices();
      if (_open) $("btn-close-settings").focus();
    },
    close() {
      _open = false;
      drawer.classList.remove("in");
      $("btn-settings").setAttribute("aria-expanded", "false");
      $("btn-settings").focus({ preventScroll: true });
      clearTimeout(hideTimer);
      hideTimer = setTimeout(() => { if (!_open) drawer.hidden = true; }, SLIDE_MS);
    },
  };
}
