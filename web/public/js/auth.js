// Sign-in. The server decides who may enter and what they may do; this file
// only shows the password sheet and asks the server.

import { applyI18n, getLang, setLang, t } from "./i18n.js";
import { loadSettings, saveSettings } from "./store.js";

const $ = (id) => document.getElementById(id);
const JSON_HEADERS = { "Content-Type": "application/json" };

async function readJson(res) {
  try { return await res.json(); } catch { return null; }
}

// { auth, role } or null when the server cannot be reached.
export async function getSession() {
  try {
    const res = await fetch("/api/session", { cache: "no-store" });
    return res.ok ? await readJson(res) : null;
  } catch {
    return null;
  }
}

// What a fresh visitor starts with (settings, personalities, default pictures).
export async function fetchDefaults() {
  try {
    const res = await fetch("/api/defaults", { cache: "no-store" });
    return res.ok ? await readJson(res) : null;
  } catch {
    return null;
  }
}

export async function logout() {
  try {
    await fetch("/api/logout", { method: "POST", headers: JSON_HEADERS, body: "{}" });
  } catch { /* the page reloads next anyway */ }
}

async function login(password) {
  let res;
  try {
    res = await fetch("/api/login", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify({ password }) });
  } catch {
    return { error: "loginNet" };
  }
  const data = await readJson(res);
  if (res.ok && data?.role) return { role: data.role, auth: data.auth !== false };
  if (data?.error === "bad_password") return { error: "loginBad" };
  if (data?.error === "too_many_attempts") return { error: "loginLocked" };
  return { error: "loginNet" };
}

// Resolves to { auth, role } once the visitor may use the app. Shows the
// password sheet only when the server asks for one.
export async function gate() {
  const session = await getSession();
  if (session && (!session.auth || session.role)) return session;

  const sheet = $("login");
  const form = $("login-form");
  const field = $("login-password");
  const error = $("login-error");
  const langButton = $("btn-login-lang");
  if (!session) error.textContent = t("loginNet");

  const refreshLang = () => {
    applyI18n(document);
    langButton.textContent = getLang() === "vi" ? "English" : "Tiếng Việt";
    if (error.dataset.key) error.textContent = t(error.dataset.key);
  };
  const show = (key) => {
    error.dataset.key = key || "";
    error.textContent = key ? t(key) : "";
  };
  refreshLang();
  langButton.addEventListener("click", () => {
    const next = getLang() === "vi" ? "en" : "vi";
    setLang(next);
    saveSettings({ ...loadSettings(), lang: next }, ["lang"]);
    refreshLang();
  });

  sheet.hidden = false;
  field.focus();
  return new Promise((resolve) => {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const password = field.value;
      if (!password) return show("loginEmpty");
      $("btn-login").disabled = true;
      show("");
      const result = await login(password);
      $("btn-login").disabled = false;
      if (result.error) {
        show(result.error);
        field.select();
        return;
      }
      field.value = "";
      sheet.hidden = true;
      resolve({ auth: result.auth, role: result.role });
    });
  });
}
