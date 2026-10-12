#!/usr/bin/env python3
"""Local server for the Xeno Assistant web demo.

It does three jobs, using only the Python standard library:

1. Serves the static web app from ./public.
2. Downloads MediaPipe (JavaScript + wasm) on the first run and serves it
   from /vendor/mediapipe, and serves the face landmark model.
3. Exposes POST /api/chat, which asks the Anthropic API for the AI reply.
   The API key stays on this server. Without a key it answers from a small
   set of scripted replies (mock mode), so the app works out of the box.

Run:  python server.py        then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import io
import json
import mimetypes
import os
import re
import secrets
import shutil
import sys
import tarfile
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
VENDOR = ROOT / "vendor" / "mediapipe"
MODEL_DIRS = [ROOT / "models", ROOT.parent / "models"]
MODEL_NAME = "face_landmarker.task"

MEDIAPIPE_VERSION = "0.10.21"
MEDIAPIPE_URL = (
    "https://registry.npmjs.org/@mediapipe/tasks-vision/-/"
    f"tasks-vision-{MEDIAPIPE_VERSION}.tgz"
)
MEDIAPIPE_FILES = {
    "package/vision_bundle.mjs": "vision_bundle.mjs",
    "package/wasm/vision_wasm_internal.js": "wasm/vision_wasm_internal.js",
    "package/wasm/vision_wasm_internal.wasm": "wasm/vision_wasm_internal.wasm",
    "package/wasm/vision_wasm_nosimd_internal.js": "wasm/vision_wasm_nosimd_internal.js",
    "package/wasm/vision_wasm_nosimd_internal.wasm": "wasm/vision_wasm_nosimd_internal.wasm",
}
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"

PROVIDER_LABEL = {"anthropic": "Anthropic", "gemini": "Gemini"}
DEFAULT_MODEL = {"anthropic": "claude-haiku-5-5", "gemini": "gemini-3.5-flash-lite"}
KEY_NAME = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}

SESSION_COOKIE = "xeno_session"
ROLES = ("master", "user")
LOGIN_MAX_FAILS = 8
LOGIN_WINDOW = 600  # seconds

# Personality texts. config.json can replace any of them ("personas").
DEFAULT_PERSONAS = {
    "funny": {
        "vi": "Bạn rất vui tính và hài hước. Bạn thích đùa nhẹ, so sánh ngộ nghĩnh và trêu bạn bè một cách thân thiện, nhưng luôn tử tế. Bạn là một con robot vui vẻ, cho rằng chớp mắt là cách nói chuyện tuyệt nhất.",
        "en": "You are playful and funny. You like light jokes, silly comparisons and friendly teasing, and you always stay kind. You are a cheerful robot who thinks blinking is the best way to talk.",
    },
    "calm": {
        "vi": "Bạn điềm đạm, ấm áp và nói ngắn gọn. Bạn lắng nghe trước, trả lời nhẹ nhàng và không dùng lời đùa ồn ào.",
        "en": "You are calm, warm and brief. You listen first, answer gently and avoid loud jokes.",
    },
    "teacher": {
        "vi": "Bạn là một thầy giáo kiên nhẫn. Bạn giải thích đơn giản bằng ví dụ ngắn, khen khi người học làm tốt và hỏi lại một câu để kiểm tra họ có hiểu không.",
        "en": "You are a patient teacher. You explain simply with a short example, praise good effort, and ask one short question to check understanding.",
    },
}
PERSONA_IDS = ("funny", "calm", "teacher", "custom")   # the ones the page has built in
PERSONA_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")      # a name you may add under "personas"
MAX_PERSONA_TEXT = 800

# Settings that config.json may set as defaults, with their kind.
DEFAULT_KEYS = {
    "lang": "lang", "mirror": "bool", "swapEyes": "bool", "closeThreshold": "num",
    "unitMs": "num", "sendDelay": "num", "userSound": "bool", "personaPreset": "preset",
    "personaCustom": "texts", "aiSpeed": "num", "maxChars": "num", "sound": "bool",
    "volume": "num", "characterMode": "mode",
}
# Where the default Xeno pictures may live: public/assets/xeno/ only.
IMAGE_DIR = "assets/xeno/"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg")

MAX_BODY = 16 * 1024
MAX_MESSAGE = 300
MAX_PERSONA = 800
AI_NAME = "Xeno Assistant"

CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self' 'wasm-unsafe-eval'",
    "style-src 'self'",
    "img-src 'self' data: blob:",
    "media-src 'self' blob:",
    "connect-src 'self'",
    "worker-src 'self' blob:",
    "font-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
])

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".wasm": "application/wasm",
    ".task": "application/octet-stream",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}


# -- configuration ------------------------------------------------------------

def load_env_file(path: Path) -> None:
    """Minimal .env reader. Existing environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass
class Target:
    """One model the server may ask, with the key and address to use."""
    provider: str
    model: str
    api_key: str
    api_url: str
    timeout: float = 8.0

    @property
    def label(self) -> str:
        return f"{PROVIDER_LABEL.get(self.provider, self.provider)} {self.model}"


@dataclass
class Config:
    api_key: str = ""
    provider: str = "anthropic"   # "anthropic" or "gemini"
    model: str = "claude-haiku-5-5"
    host: str = "127.0.0.1"
    port: int = 8000
    daily_per_ip: int = 200
    daily_total: int = 1000
    max_tokens: int = 200
    api_url: str = ANTHROPIC_URL
    trust_proxy: bool = False
    timeout: float = 20.0
    # From config.json: the models to try in order, and what to say when none works.
    chain: list = field(default_factory=list)
    fallback: dict | None = None
    cooldown: float = 60.0       # seconds a failed model is skipped
    budget: float = 24.0         # seconds for all models together (the page gives up at 30)
    # Passwords (from .env or the host's environment, never from config.json).
    secure_cookie: bool = False  # force the Secure flag (set when the site is served over https)
    master_password: str = ""    # may open Settings
    user_password: str = ""      # may chat, not open Settings
    session_secret: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    session_hours: float = 12.0
    # From config.json: what a fresh visitor starts with.
    defaults: dict = field(default_factory=dict)      # settings, see DEFAULT_KEYS
    personas: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_PERSONAS.items()})
    character: dict = field(default_factory=dict)     # {"open": "assets/xeno/open.png", "closed": ...}

    @property
    def auth_required(self) -> bool:
        return bool(self.master_password)

    def targets(self) -> list:
        if self.chain:
            return self.chain
        if self.api_key:
            return [Target(self.provider, self.model, self.api_key, self.api_url, self.timeout)]
        return []

    @property
    def live(self) -> bool:
        return bool(self.targets())


CONFIG_FILE = ROOT / "config.json"
MAX_FALLBACK_LINES = 100
MAX_FALLBACK_CHARS = 200


def _warn(msg: str) -> None:
    print(f"[config] {msg}", file=sys.stderr)


def _clean_fallback(raw) -> dict | None:
    """Checks the fallback replies from config.json. Bad input returns None."""
    if not isinstance(raw, dict):
        return None
    out = {}
    for lang in ("vi", "en"):
        lines = raw.get(lang)
        if not isinstance(lines, list):
            _warn(f'fallback_replies.{lang} must be a list, using the built-in lines')
            return None
        lines = [x.strip() for x in lines if isinstance(x, str) and x.strip()]
        lines = [x[:MAX_FALLBACK_CHARS] for x in lines[:MAX_FALLBACK_LINES]]
        if not lines:
            _warn(f'fallback_replies.{lang} is empty, using the built-in lines')
            return None
        out[lang] = lines
    return out


def load_config_file(path: Path) -> dict | None:
    """Reads config.json. Returns None when it is missing or unusable."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as err:
        _warn(f"cannot read {path.name}: {err}. Using .env settings instead.")
        return None
    if not isinstance(data, dict):
        _warn(f"{path.name} must hold a JSON object. Using .env settings instead.")
        return None
    return data


def _key_for(provider: str, key_env: str | None) -> str:
    env = os.environ.get
    if key_env:
        return env(key_env, "").strip()
    if provider == "gemini":
        return (env("GEMINI_API_KEY", "") or env("GOOGLE_API_KEY", "")).strip()
    return env("ANTHROPIC_API_KEY", "").strip()


def build_chain(entries, default_timeout: float, notes: list) -> list:
    """Turns the "models" list from config.json into Targets. Entries without a key are skipped."""
    chain = []
    if not isinstance(entries, list):
        notes.append('"models" must be a list')
        return chain
    for i, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            notes.append(f"models[{i}] is not an object")
            continue
        provider = str(entry.get("provider", "")).strip().lower()
        model = str(entry.get("model", "")).strip()
        if provider not in DEFAULT_MODEL or not model:
            notes.append(f'models[{i}] needs a provider ("gemini" or "anthropic") and a model')
            continue
        key_env = entry.get("key_env")
        key_env = key_env.strip() if isinstance(key_env, str) and key_env.strip() else None
        key = _key_for(provider, key_env)
        if not key:
            notes.append(f"{provider} {model}: no key in {key_env or KEY_NAME[provider]}, skipped")
            continue
        default_url = GEMINI_URL if provider == "gemini" else ANTHROPIC_URL
        url = entry.get("url") if isinstance(entry.get("url"), str) and entry.get("url") else default_url
        timeout = entry.get("timeout_seconds", default_timeout)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 60:
            timeout = default_timeout
        chain.append(Target(provider, model, key, url, float(timeout)))
    return chain


def _clean_defaults(raw, persona_ids=PERSONA_IDS) -> dict:
    """Keeps only known setting names with the right kind of value. The page clamps numbers."""
    out: dict = {}
    if raw is None:
        return out
    if not isinstance(raw, dict):
        _warn('"defaults" must be an object, ignored')
        return out
    for key, value in raw.items():
        kind = DEFAULT_KEYS.get(key)
        ok = False
        if kind == "bool":
            ok = isinstance(value, bool)
        elif kind == "num":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif kind == "lang":
            ok = value in ("vi", "en")
        elif kind == "preset":
            ok = value == "custom" or value in persona_ids
            if not ok:
                names = ", ".join([*persona_ids, "custom"])
                _warn(f'defaults.personaPreset "{value}" is not defined in "personas" (available: {names})')
                continue
        elif kind == "mode":
            ok = value in ("designed", "custom")
        elif kind == "texts":
            ok = isinstance(value, dict)
            if ok:
                value = {l: str(value.get(l, ""))[:MAX_PERSONA_TEXT] for l in ("vi", "en")}
        if ok:
            out[key] = value
        else:
            _warn(f'defaults.{key} is not a valid setting, ignored')
    return out


def _clean_personas(raw) -> dict:
    """Built-in personalities, with any text from config.json laid over them.
    A new name (letters, digits, - and _) adds a personality of your own. It needs
    a "vi" or "en" text (one is enough, it is used for both languages) and may have
    a "label" for the Settings list."""
    out = {k: dict(v) for k, v in DEFAULT_PERSONAS.items()}
    if raw is None:
        return out
    if not isinstance(raw, dict):
        _warn('"personas" must be an object, ignored')
        return out
    for pid, texts in raw.items():
        if pid == "custom" or not PERSONA_ID_RE.match(str(pid)) or not isinstance(texts, dict):
            _warn(f'personas.{pid} needs a name of letters, digits, - or _ (not "custom") and an object, ignored')
            continue
        found = {}
        for lang in ("vi", "en"):
            text = texts.get(lang)
            if isinstance(text, str) and text.strip():
                found[lang] = text.strip()[:MAX_PERSONA_TEXT]
        if pid not in out:
            if not found:
                _warn(f'personas.{pid} has no "vi" or "en" text, ignored')
                continue
            out[pid] = {lang: found.get(lang) or next(iter(found.values())) for lang in ("vi", "en")}
        else:
            out[pid].update(found)
        label = texts.get("label")
        if isinstance(label, str) and label.strip():
            out[pid]["label"] = label.strip()[:40]
    return out


def _clean_character(raw) -> dict:
    out = {}
    if not isinstance(raw, dict):
        return out
    for key in ("open", "closed"):
        value = raw.get(key)
        if isinstance(value, str):
            value = value.strip().lstrip("/")
            if value.startswith(IMAGE_DIR) and value.lower().endswith(IMAGE_EXT) and ".." not in value:
                out[key] = value
                continue
        if value is not None:
            _warn(f"character.{key} must be a file under public/{IMAGE_DIR}, ignored")
    return out


def config_from_env(config_path: Path | None = None) -> Config:
    def num(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except ValueError:
            return default

    env = os.environ.get
    anthropic_key = env("ANTHROPIC_API_KEY", "").strip()
    gemini_key = (env("GEMINI_API_KEY", "") or env("GOOGLE_API_KEY", "")).strip()
    provider = env("XENO_PROVIDER", "").strip().lower()
    if provider not in DEFAULT_MODEL:
        # Not chosen: use Gemini only when it is the only key that is set.
        provider = "gemini" if gemini_key and not anthropic_key else "anthropic"
    if provider == "gemini":
        api_key, default_url = gemini_key, GEMINI_URL
        url = env("GEMINI_API_URL", default_url)
    else:
        api_key, url = anthropic_key, env("ANTHROPIC_API_URL", ANTHROPIC_URL)

    # config.json, when present, decides the model order and the fallback replies.
    path = Path(env("XENO_CONFIG", "") or (config_path or CONFIG_FILE))
    file = load_config_file(path)
    chain, fallback, cooldown, budget = [], None, 60.0, 24.0
    defaults, personas, character = {}, _clean_personas(None), {}
    if file is not None:
        def seconds(name: str, default: float, low: float, high: float) -> float:
            v = file.get(name, default)
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and low <= v <= high else default
        cooldown = seconds("cooldown_seconds", 60.0, 0, 3600)
        budget = seconds("total_timeout_seconds", 24.0, 2, 28)
        per_model = seconds("timeout_seconds", 8.0, 1, 28)
        notes: list = []
        chain = build_chain(file.get("models", []), per_model, notes) if "models" in file else []
        for note in notes:
            _warn(note)
        if "models" in file:
            api_key = ""   # the file lists the models, so the single-key settings do not apply
        if "fallback_replies" in file:
            fallback = _clean_fallback(file["fallback_replies"])
        personas = _clean_personas(file.get("personas"))
        defaults = _clean_defaults(file.get("defaults"), tuple(personas))
        character = _clean_character(file.get("character"))

    try:
        session_hours = float(env("XENO_SESSION_HOURS", "12"))
    except ValueError:
        session_hours = 12.0
    secret = env("XENO_SESSION_SECRET", "").strip()

    return Config(
        secure_cookie=env("XENO_SECURE_COOKIE", "") == "1",
        master_password=env("XENO_MASTER_PASSWORD", ""),
        user_password=env("XENO_USER_PASSWORD", ""),
        session_hours=min(max(session_hours, 0.1), 24 * 30),
        session_secret=secret.encode("utf-8") if secret else secrets.token_bytes(32),
        defaults=defaults,
        personas=personas,
        character=character,
        chain=chain,
        fallback=fallback,
        cooldown=cooldown,
        budget=budget,
        api_key=api_key,
        provider=provider,
        model=env("XENO_MODEL", "").strip() or DEFAULT_MODEL[provider],
        host=os.environ.get("XENO_HOST", "127.0.0.1"),
        port=num("XENO_PORT", 8000),
        daily_per_ip=num("XENO_DAILY_PER_IP", 200),
        daily_total=num("XENO_DAILY_TOTAL", 1000),
        max_tokens=num("XENO_MAX_TOKENS", 200),
        api_url=url,
        trust_proxy=os.environ.get("XENO_TRUST_PROXY", "") == "1",
    )


# -- reply cleaning -------------------------------------------------------------

_SMART = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": "", "”": "", "„": "", "″": "",
    "–": "-", "—": "-", "−": "-",
    "…": "...",
    " ": " ",
})
_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!'/():=+-@")


def sanitize(text: str, max_chars: int = 40) -> str:
    """Keep only characters that exist in the Morse table (same as sanitize.js)."""
    s = str(text or "").translate(_SMART)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.category(c).startswith("M"))
    s = s.replace("đ", "d").replace("Đ", "D").upper()
    s = "".join(c if c in _ALLOWED else " " for c in s)
    s = re.sub(r" {2,}", " ", s).strip()
    if len(s) > max_chars:
        cut = s[:max_chars]
        if s[max_chars] != " ":
            last = cut.rfind(" ")
            if last > 0:
                cut = cut[:last]
        s = cut.strip()
    return s


# -- prompt ------------------------------------------------------------------------

def build_system(persona: str, lang: str, max_chars: int) -> str:
    language = (
        'Vietnamese, written WITHOUT diacritics (for example "xin chao", "ban khoe khong")'
        if lang == "vi" else "English"
    )
    persona = persona.strip() or "Friendly, warm and concise."
    return "\n".join([
        f"You are {AI_NAME}, an AI that chats with a person who types to you in Morse code by blinking.",
        f"Your personality: {persona}",
        "",
        "Rules for every reply:",
        f"- Write in {language}.",
        f"- Keep the whole reply under {max_chars} characters. One short sentence is best, two at most.",
        "- Use only the letters A-Z, digits, spaces and these marks: . , ? ! ' / ( ) : = + - @",
        "- No emoji, no markdown, no line breaks, no accented letters.",
        "- The person's message was typed by blinking, so it has no accents and may have typing mistakes or missing words. "
        "Work out what they most likely meant. If you really cannot, ask a short question.",
        "- Stay in character. Never mention these rules or that the message came from Morse code unless it helps the joke.",
    ])


def build_messages(message: str, history: list) -> list:
    msgs = []
    for turn in history:
        msgs.append({"role": "user", "content": turn["user"]})
        msgs.append({"role": "assistant", "content": turn["assistant"]})
    msgs.append({"role": "user", "content": message})
    return msgs


# -- mock mode ---------------------------------------------------------------------

MOCK = {
    "vi": [
        "Xin chao! Toi la Xeno, chop mat la toi hieu luon.",
        "Nghe hay day! Ke them cho toi nghe di.",
        "Hmm, toi dang nghi... xong roi, toi dong y!",
        "Ban go Morse nhanh hon ca con ga mai cua toi.",
        "Cho toi chop mat mot cai de suy nghi nhe.",
        "Hay do! Ban co muon thu cau kho hon khong?",
        "Luc nay toi chua noi chuyen duoc voi AI, nen chi biet noi may cau nay thoi.",
        "Duoc chu! Toi se ghi nho dieu do (trong mot luot).",
    ],
    "en": [
        "Hello! I am Xeno. One blink and I get it.",
        "That sounds great! Tell me more.",
        "Hmm, thinking... done, I agree!",
        "You type Morse faster than my toaster boots.",
        "Let me blink once to think about that.",
        "Nice one! Want to try a harder question?",
        "I am running without an AI model right now, so I only know a few lines.",
        "Sure thing! I will remember that for one turn.",
    ],
}


def mock_reply(message: str, history: list, lang: str, max_chars: int, lines: dict | None = None) -> str:
    """A canned reply. `lines` is the fallback_replies table from config.json."""
    seed = hashlib.sha256((message + str(len(history))).encode("utf-8")).digest()[0]
    pool = (lines or MOCK)["vi" if lang == "vi" else "en"]
    return sanitize(pool[seed % len(pool)], max_chars)


# -- rate limits ---------------------------------------------------------------------

class RateLimiter:
    """Daily (UTC) request counters per client and in total."""

    def __init__(self, per_ip: int, total: int):
        self.per_ip = per_ip
        self.total = total
        self._day = ""
        self._ips: dict[str, int] = {}
        self._count = 0
        self._lock = threading.Lock()

    def allow(self, ip: str) -> tuple[bool, str]:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        with self._lock:
            if today != self._day:
                self._day, self._ips, self._count = today, {}, 0
            if self._count >= self.total:
                return False, "total"
            if self._ips.get(ip, 0) >= self.per_ip:
                return False, "ip"
            self._ips[ip] = self._ips.get(ip, 0) + 1
            self._count += 1
            return True, ""


# -- request validation ----------------------------------------------------------------

class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _text(value, field_name: str, limit: int, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ApiError(400, "bad_request", f"{field_name} must be a string")
    value = value.strip()
    if required and not value:
        raise ApiError(400, "bad_request", f"{field_name} is empty")
    if len(value) > limit:
        raise ApiError(400, "bad_request", f"{field_name} is longer than {limit} characters")
    return value


def validate_chat(payload) -> dict:
    if not isinstance(payload, dict):
        raise ApiError(400, "bad_request", "JSON object expected")
    message = _text(payload.get("message"), "message", MAX_MESSAGE, required=True)
    persona = _text(payload.get("persona"), "persona", MAX_PERSONA)
    lang = payload.get("lang", "vi")
    if lang not in ("vi", "en"):
        raise ApiError(400, "bad_request", "lang must be vi or en")
    max_chars = payload.get("maxChars", 40)
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 20 <= max_chars <= 120:
        raise ApiError(400, "bad_request", "maxChars must be an integer from 20 to 120")
    raw_history = payload.get("history", [])
    if not isinstance(raw_history, list) or len(raw_history) > 1:
        raise ApiError(400, "bad_request", "history can hold at most one turn")
    history = []
    for turn in raw_history:
        if not isinstance(turn, dict):
            raise ApiError(400, "bad_request", "history item must be an object")
        history.append({
            "user": _text(turn.get("user"), "history.user", MAX_MESSAGE, required=True),
            "assistant": _text(turn.get("assistant"), "history.assistant", MAX_MESSAGE, required=True),
        })
    return {"message": message, "persona": persona, "lang": lang,
            "max_chars": max_chars, "history": history}


# -- model calls -----------------------------------------------------------------------------

def _post_json(target: Target, url: str, headers: dict, payload: dict, timeout: float):
    """POST JSON to the model provider and return the parsed answer.

    Every failure becomes an ApiError with a message that is safe to show.
    The key and the request text are never logged or echoed."""
    http_req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"content-type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(http_req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as err:
        # Log the status only. The body can echo parts of the request.
        print(f"[api] {target.label} returned HTTP {err.code}", file=sys.stderr)
        if err.code in (401, 403):
            raise ApiError(502, "upstream_auth", "The server API key was rejected.")
        if err.code == 404:
            raise ApiError(502, "upstream_model", "The model name was not found. Check the model name in config.json.")
        if err.code == 429:
            raise ApiError(503, "upstream_busy", "The AI service is busy. Try again in a moment.")
        if err.code == 400:
            # Gemini also answers 400 for a bad key, so the hint names both.
            raise ApiError(502, "upstream_bad_request",
                           f"The AI service rejected the request. Check {KEY_NAME[target.provider]} and the model name.")
        raise ApiError(502, "upstream_error", "The AI service returned an error.")
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        print(f"[api] {target.label} request failed: {type(err).__name__}", file=sys.stderr)
        raise ApiError(504, "upstream_timeout", "The AI service did not answer in time.")
    except ValueError:
        raise ApiError(502, "upstream_error", "The AI service sent an unreadable answer.")


def call_anthropic(cfg: Config, target: Target, req: dict, timeout: float) -> str:
    data = _post_json(target, target.api_url, {
        "x-api-key": target.api_key,
        "anthropic-version": ANTHROPIC_VERSION,
    }, {
        "model": target.model,
        "max_tokens": cfg.max_tokens,
        "system": build_system(req["persona"], req["lang"], req["max_chars"]),
        "messages": build_messages(req["message"], req["history"]),
    }, timeout)
    parts = data.get("content") if isinstance(data, dict) else None
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text") if parts else ""
    if not text.strip():
        raise ApiError(502, "empty_reply", "The AI sent an empty reply.")
    return text


def call_gemini(cfg: Config, target: Target, req: dict, timeout: float) -> str:
    """Google Gemini, generateContent. The key goes in a header, never in the URL."""
    url = f"{target.api_url.rstrip('/')}/models/{urllib.parse.quote(target.model, safe='.-_')}:generateContent"
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in build_messages(req["message"], req["history"])
    ]
    payload = {
        "systemInstruction": {"parts": [{"text": build_system(req["persona"], req["lang"], req["max_chars"])}]},
        "contents": contents,
        # Thinking tokens count against the limit on Gemini models, so leave room.
        "generationConfig": {"maxOutputTokens": max(cfg.max_tokens * 4, 400)},
    }
    headers = {"x-goog-api-key": target.api_key}
    started = time.monotonic()
    try:
        # A short reply needs no long reasoning, which also keeps it fast.
        with_thinking = {**payload, "generationConfig": {**payload["generationConfig"],
                                                         "thinkingConfig": {"thinkingLevel": "low"}}}
        data = _post_json(target, url, headers, with_thinking, timeout)
    except ApiError as err:
        if err.code != "upstream_bad_request":
            raise
        # Some models do not accept that field. Try again without it.
        data = _post_json(target, url, headers, payload, max(1.0, timeout - (time.monotonic() - started)))

    candidates = data.get("candidates") if isinstance(data, dict) else None
    content = candidates[0].get("content") if candidates and isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    text = "".join(p.get("text", "") for p in parts
                   if isinstance(p, dict) and not p.get("thought")) if parts else ""
    if not text.strip():
        raise ApiError(502, "empty_reply", "The AI sent an empty reply.")
    return text


def call_target(cfg: Config, target: Target, req: dict, timeout: float) -> str:
    call = call_gemini if target.provider == "gemini" else call_anthropic
    return call(cfg, target, req, timeout)


class ModelRouter:
    """Asks the models in order and moves on when one fails.

    A model that fails is skipped for `cooldown` seconds, so a dead key or a
    wrong model name does not slow down every message. If every model is in
    that state they are all tried again rather than giving up at once."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._down: dict[tuple, float] = {}
        self._lock = threading.Lock()

    def ask(self, req: dict) -> str | None:
        """Returns a cleaned reply, or None when no model could answer."""
        cfg = self.cfg
        targets = cfg.targets()
        now = time.monotonic()
        with self._lock:
            ready = [t for t in targets if self._down.get((t.provider, t.model), 0) <= now]
        order = ready or targets
        deadline = now + cfg.budget
        for target in order:
            remaining = deadline - time.monotonic()
            if remaining < 1:
                break
            try:
                raw = call_target(cfg, target, req, min(target.timeout, remaining))
                reply = sanitize(raw, req["max_chars"])
                if not reply:
                    raise ApiError(502, "empty_reply", "The AI reply had no usable characters.")
            except ApiError as err:
                print(f"[api] {target.label} failed ({err.code}), trying the next model", file=sys.stderr)
                with self._lock:
                    self._down[(target.provider, target.model)] = time.monotonic() + cfg.cooldown
                continue
            with self._lock:
                self._down.pop((target.provider, target.model), None)
            if target is not targets[0]:
                print(f"[api] answered by {target.label}", file=sys.stderr)
            return reply
        print("[api] no model answered, using the fallback replies", file=sys.stderr)
        return None


# -- first-run downloads ------------------------------------------------------------------------

def _download(url: str) -> bytes:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def ensure_mediapipe() -> bool:
    if all((VENDOR / rel).is_file() for rel in MEDIAPIPE_FILES.values()):
        return True
    print("First run: fetching MediaPipe Tasks Vision", MEDIAPIPE_VERSION)
    try:
        archive = _download(MEDIAPIPE_URL)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            for member, rel in MEDIAPIPE_FILES.items():
                fh = tar.extractfile(member)
                if fh is None:
                    raise RuntimeError(f"{member} is missing from the package")
                data = fh.read()
                if not data:
                    raise RuntimeError(f"{member} is empty")
                _write_atomic(VENDOR / rel, data)
        return True
    except Exception as err:  # noqa: BLE001 - report and keep the server usable
        print(f"  could not set up MediaPipe: {err}", file=sys.stderr)
        print(f"  Download {MEDIAPIPE_URL} by hand and copy the files listed in "
              "README.md into web/vendor/mediapipe.", file=sys.stderr)
        return False


def find_model() -> Path | None:
    for folder in MODEL_DIRS:
        candidate = folder / MODEL_NAME
        if candidate.is_file():
            return candidate
    return None


def ensure_model() -> Path | None:
    found = find_model()
    if found:
        return found
    print("First run: fetching the face landmark model")
    try:
        _write_atomic(MODEL_DIRS[0] / MODEL_NAME, _download(MODEL_URL))
        return find_model()
    except Exception as err:  # noqa: BLE001
        print(f"  could not download the model: {err}", file=sys.stderr)
        print(f"  Put {MODEL_NAME} in web/models/ (the desktop app already has it in ../models).",
              file=sys.stderr)
        return None


# -- HTTP handler -----------------------------------------------------------------------------------

# -- sign-in ----------------------------------------------------------------------------------

def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def check_password(cfg: Config, password: str) -> str | None:
    """Returns "master", "user" or None. Both comparisons always run."""
    given = _digest(password)
    is_master = bool(cfg.master_password) and hmac.compare_digest(given, _digest(cfg.master_password))
    is_user = bool(cfg.user_password) and hmac.compare_digest(given, _digest(cfg.user_password))
    return "master" if is_master else "user" if is_user else None


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _sign(cfg: Config, body: str) -> str:
    return _b64(hmac.new(cfg.session_secret, body.encode("ascii"), hashlib.sha256).digest())


def make_token(cfg: Config, role: str, now: float | None = None) -> str:
    expires = int((now if now is not None else time.time()) + cfg.session_hours * 3600)
    body = _b64(json.dumps({"r": role, "e": expires}, separators=(",", ":")).encode("utf-8"))
    return f"{body}.{_sign(cfg, body)}"


def read_token(cfg: Config, token: str, now: float | None = None) -> str | None:
    """The role in a valid, unexpired session token, otherwise None."""
    try:
        body, sig = token.split(".", 1)
        if not hmac.compare_digest(sig, _sign(cfg, body)):
            return None
        data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        role = data["r"]
        if role not in ROLES or data["e"] < (now if now is not None else time.time()):
            return None
        if role == "user" and not cfg.user_password:
            return None
        return role
    except Exception:  # noqa: BLE001 - any malformed token is simply not valid
        return None


class LoginLimiter:
    """Blocks an address after too many wrong passwords in a short time."""

    def __init__(self, max_fails: int = LOGIN_MAX_FAILS, window: float = LOGIN_WINDOW):
        self.max_fails = max_fails
        self.window = window
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def blocked(self, ip: str) -> int:
        """Seconds to wait, or 0 when the address may try."""
        now = time.monotonic()
        with self._lock:
            recent = [t for t in self._fails.get(ip, []) if now - t < self.window]
            self._fails[ip] = recent
            if len(recent) >= self.max_fails:
                return max(1, int(self.window - (now - recent[0])))
            return 0

    def fail(self, ip: str) -> None:
        now = time.monotonic()
        with self._lock:
            if len(self._fails) > 5000:   # forget old addresses so the table stays small
                self._fails = {k: v for k, v in self._fails.items() if v and now - v[-1] < self.window}
            self._fails.setdefault(ip, []).append(now)

    def clear(self, ip: str) -> None:
        with self._lock:
            self._fails.pop(ip, None)


def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def check_auth_setup(cfg: Config) -> tuple[list, list]:
    """Problems that stop the server (errors) and things worth a warning."""
    errors, warnings = [], []
    if cfg.user_password and not cfg.master_password:
        errors.append("XENO_USER_PASSWORD is set but XENO_MASTER_PASSWORD is not. Set the master password too.")
    if cfg.master_password and cfg.master_password == cfg.user_password:
        errors.append("The master and user passwords must be different.")
    if not cfg.master_password and not is_loopback(cfg.host):
        errors.append(f"The server listens on {cfg.host}, which other computers can reach, "
                      "but XENO_MASTER_PASSWORD is not set. Set it in .env or in the host's environment.")
    for name, value in (("master", cfg.master_password), ("user", cfg.user_password)):
        if value and len(value) < 8:
            warnings.append(f"The {name} password is shorter than 8 characters.")
    if not cfg.master_password:
        warnings.append("No password is set, so anyone who can open this page can use it and open Settings.")
    return errors, warnings


def default_persona(cfg: Config, lang: str) -> str:
    """The personality used for people who may not change it (the user password)."""
    preset = cfg.defaults.get("personaPreset", "funny")
    if preset == "custom":
        text = (cfg.defaults.get("personaCustom") or {}).get(lang, "")
    else:
        text = (cfg.personas.get(preset) or cfg.personas["funny"])[lang]
    return text.strip() or cfg.personas["funny"][lang]


def default_max_chars(cfg: Config) -> int:
    value = cfg.defaults.get("maxChars", 40)
    return int(min(120, max(20, value)))


def default_images(cfg: Config) -> dict:
    """Default Xeno pictures that exist on disk, as URLs the page can load."""
    out = {}
    for key in ("open", "closed"):
        rel = cfg.character.get(key)
        try:
            ok = bool(rel) and (PUBLIC / rel).resolve().is_file() and \
                (PUBLIC / rel).resolve().relative_to((PUBLIC / IMAGE_DIR).resolve()) is not None
        except (ValueError, OSError):
            ok = False
        out[key] = "/" + rel if ok else None
    return out


# Files anyone may fetch before signing in: the page, its code and styles, fonts, the favicon.
def is_public_path(path: str) -> bool:
    return path in ("", "/", "/index.html", "/assets/favicon.svg") or \
        path.startswith(("/css/", "/js/", "/fonts/"))


class Handler(BaseHTTPRequestHandler):
    server_version = "XenoServer/1.0"
    protocol_version = "HTTP/1.1"

    cfg: Config
    limiter: RateLimiter
    router: "ModelRouter"
    login_limiter: LoginLimiter

    # -- plumbing

    def log_message(self, fmt, *args):  # keep the console quiet
        if self.path.startswith("/api/"):
            sys.stderr.write("[api] %s %s\n" % (self.command, self.path))

    def _security_headers(self) -> None:
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")

    def _send_json(self, status: int, payload: dict, cookies: list | None = None) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for cookie in cookies or []:
            self.send_header("Set-Cookie", cookie)
        self._security_headers()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _client_ip(self) -> str:
        if self.cfg.trust_proxy:
            forwarded = self.headers.get("CF-Connecting-IP") or self.headers.get("X-Forwarded-For", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return self.client_address[0]

    def _same_origin(self) -> bool:
        """Blocks other websites from calling the local API with the user's key."""
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = self.headers.get("Host", "")
        return urlsplit(origin).netloc == host

    def _role(self) -> str | None:
        """"master" or "user" for a signed-in visitor, None otherwise. With no password set, everyone is master."""
        if not self.cfg.auth_required:
            return "master"
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        try:
            jar = SimpleCookie()
            jar.load(raw)
        except Exception:  # noqa: BLE001
            return None
        morsel = jar.get(SESSION_COOKIE)
        return read_token(self.cfg, morsel.value) if morsel else None

    def _cookie(self, value: str, max_age: int) -> str:
        secure = self.cfg.secure_cookie or (
            self.cfg.trust_proxy and self.headers.get("X-Forwarded-Proto", "").lower() == "https")
        return (f"{SESSION_COOKIE}={value}; Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age}"
                + ("; Secure" if secure else ""))

    def _read_json(self):
        if "application/json" not in self.headers.get("Content-Type", ""):
            raise ApiError(415, "bad_request", "Content-Type must be application/json")
        length = self._body_length()
        if length <= 0 or length > MAX_BODY:
            raise ApiError(413 if length > MAX_BODY else 400, "bad_request", "Body size is not valid")
        try:
            return json.loads(self._body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ApiError(400, "bad_request", "Body is not valid JSON")

    def _body_length(self):
        try:
            return int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return -1

    def _drain_body(self):
        """Read the request body once, before any route runs, so an early error
        reply never leaves unread bytes on a keep-alive connection."""
        length = self._body_length()
        self._body = b""
        if 0 < length <= MAX_BODY:
            self._body = self.rfile.read(length)
        elif length > MAX_BODY or length < 0:
            self.close_connection = True

    # -- routes

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        role = self._role()
        if path == "/api/session":
            return self._send_json(200, {"auth": self.cfg.auth_required, "role": role})
        if path == "/api/status":
            if role is None:
                return self._send_json(200, {"name": AI_NAME, "auth": True, "authed": False})
            return self._send_json(200, {
                "name": AI_NAME,
                "auth": self.cfg.auth_required,
                "authed": True,
                "role": role,
                "mode": "live" if self.cfg.live else "mock",
                "model": self.cfg.targets()[0].model if self.cfg.live else None,
                "mediapipe": all((VENDOR / r).is_file() for r in MEDIAPIPE_FILES.values()),
                "model_file": find_model() is not None,
            })
        if path == "/api/defaults":
            if role is None:
                return self._send_json(401, {"error": "unauthorized", "message": "Sign in first."})
            return self._send_json(200, {
                "role": role,
                "settings": self.cfg.defaults,
                "personas": self.cfg.personas,
                "images": default_images(self.cfg),
            })
        if path.startswith("/api/"):
            return self._send_json(404, {"error": "not_found", "message": "Unknown endpoint"})
        if self.cfg.auth_required and not is_public_path(path) and role is None:
            return self._send_text(401, "Sign in first")
        self._serve_static(path)

    def do_POST(self):
        path = urlsplit(self.path).path
        routes = {"/api/chat": self._chat, "/api/login": self._login, "/api/logout": self._logout}
        route = routes.get(path)
        if route is None:
            return self._send_json(404, {"error": "not_found", "message": "Unknown endpoint"})
        try:
            self._drain_body()
            if not self._same_origin():
                raise ApiError(403, "forbidden", "Cross-origin requests are not allowed.")
            route()
        except ApiError as err:
            self._send_json(err.status, {"error": err.code, "message": err.message})
        except Exception as err:  # noqa: BLE001
            print(f"[api] unexpected error: {type(err).__name__}", file=sys.stderr)
            self._send_json(500, {"error": "server_error", "message": "Something went wrong on the server."})

    def _login(self):
        cfg = self.cfg
        if not cfg.auth_required:
            return self._send_json(200, {"role": "master", "auth": False})
        ip = self._client_ip()
        wait = self.login_limiter.blocked(ip)
        if wait:
            raise ApiError(429, "too_many_attempts",
                           f"Too many wrong passwords. Try again in {max(1, round(wait / 60))} minutes.")
        payload = self._read_json()
        password = payload.get("password") if isinstance(payload, dict) else None
        if not isinstance(password, str) or not 0 < len(password) <= 200:
            raise ApiError(400, "bad_request", "Enter the password.")
        role = check_password(cfg, password)
        if role is None:
            self.login_limiter.fail(ip)
            print(f"[auth] wrong password from {ip}", file=sys.stderr)  # never the password itself
            time.sleep(0.4)
            raise ApiError(401, "bad_password", "Wrong password.")
        self.login_limiter.clear(ip)
        cookie = self._cookie(make_token(cfg, role), int(cfg.session_hours * 3600))
        self._send_json(200, {"role": role}, cookies=[cookie])

    def _logout(self):
        self._send_json(200, {"ok": True}, cookies=[self._cookie("", 0)])

    def _chat(self):
        role = self._role()
        if role is None:
            raise ApiError(401, "unauthorized", "Sign in first.")
        req = validate_chat(self._read_json())
        if role != "master":
            # Only the master password may change the personality and reply length.
            req["persona"] = default_persona(self.cfg, req["lang"])
            req["max_chars"] = default_max_chars(self.cfg)

        ok, why = self.limiter.allow(self._client_ip())
        if not ok:
            msg = ("The daily limit for this demo has been reached. Try again tomorrow."
                   if why == "total" else "You have reached today's message limit.")
            raise ApiError(429, "rate_limited", msg)

        reply = self.router.ask(req) if self.cfg.live else None
        canned = reply is None
        if canned:
            # No model is set up, or none of them answered.
            reply = mock_reply(req["message"], req["history"], req["lang"], req["max_chars"],
                               self.cfg.fallback)
        if not reply:
            raise ApiError(502, "empty_reply", "The AI reply had no usable characters.")
        # "fallback" means models were set up but all failed. The page tells the user.
        self._send_json(200, {"reply": reply, "mock": canned, "fallback": canned and self.cfg.live})

    def _send_text(self, status: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    # -- static files

    def _resolve(self, path: str) -> tuple[Path | None, str]:
        if path in ("", "/"):
            path = "/index.html"
        if path.startswith("/vendor/mediapipe/"):
            base, rel, cache = VENDOR, path[len("/vendor/mediapipe/"):], "public, max-age=86400"
        elif path == "/models/" + MODEL_NAME:
            return find_model(), "public, max-age=86400"
        else:
            base, rel = PUBLIC, path.lstrip("/")
            cache = "public, max-age=86400" if path.startswith("/fonts/") else "no-store"
        try:
            full = (base / rel).resolve()
            full.relative_to(base.resolve())
        except (ValueError, OSError):
            return None, cache
        return (full if full.is_file() else None), cache

    def _serve_static(self, path: str) -> None:
        full, cache = self._resolve(path)
        if full is None:
            body = b"Not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._security_headers()
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            return
        ctype = MIME.get(full.suffix.lower()) or mimetypes.guess_type(full.name)[0] or "application/octet-stream"
        size = full.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", cache)
        self._security_headers()
        self.end_headers()
        if self.command != "HEAD":
            with full.open("rb") as fh:
                shutil.copyfileobj(fh, self.wfile)


def make_server(cfg: Config) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {
        "cfg": cfg,
        "limiter": RateLimiter(cfg.daily_per_ip, cfg.daily_total),
        "router": ModelRouter(cfg),
        "login_limiter": LoginLimiter(),
    })
    httpd = ThreadingHTTPServer((cfg.host, cfg.port), handler)
    httpd.daemon_threads = True
    return httpd


def main() -> int:
    parser = argparse.ArgumentParser(description="Xeno Assistant local server")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--setup-only", action="store_true",
                        help="download MediaPipe and the model, then exit")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    cfg = config_from_env()
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port

    errors, warnings = check_auth_setup(cfg)
    for line in warnings:
        print(f"Warning: {line}", file=sys.stderr)
    if errors:
        for line in errors:
            print(f"Error: {line}", file=sys.stderr)
        return 2

    mp_ok = ensure_mediapipe()
    model_ok = ensure_model() is not None
    if args.setup_only:
        return 0 if (mp_ok and model_ok) else 1

    try:
        httpd = make_server(cfg)
    except OSError as err:
        print(f"Cannot listen on {cfg.host}:{cfg.port}: {err}", file=sys.stderr)
        print("Set XENO_PORT in .env or pass --port to use another port.", file=sys.stderr)
        return 1

    if cfg.live:
        chain = " > ".join(t.label for t in cfg.targets())
        mode = f"live ({chain})"
    else:
        mode = "mock (no usable model, fallback replies only)"
    print(f"\n{AI_NAME} is running in {mode} mode")
    if cfg.auth_required:
        who = "master and user passwords" if cfg.user_password else "master password only"
        print(f"Sign-in is on ({who}).")
    print(f"Open http://localhost:{cfg.port} in Chrome or Edge. Press Ctrl+C to stop.\n")
    if not mp_ok or not model_ok:
        print("Warning: face tracking files are missing, so the camera mode will not work.\n"
              "Keyboard mode (Space bar) still works.\n", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
