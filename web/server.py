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
import hashlib
import io
import json
import mimetypes
import os
import re
import shutil
import sys
import tarfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
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
class Config:
    api_key: str = ""
    model: str = "claude-haiku-5-5"
    host: str = "127.0.0.1"
    port: int = 8000
    daily_per_ip: int = 200
    daily_total: int = 1000
    max_tokens: int = 200
    api_url: str = ANTHROPIC_URL
    trust_proxy: bool = False
    timeout: float = 20.0

    @property
    def live(self) -> bool:
        return bool(self.api_key)


def config_from_env() -> Config:
    def num(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except ValueError:
            return default

    return Config(
        api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        model=os.environ.get("XENO_MODEL", "claude-haiku-5-5").strip() or "claude-haiku-5-5",
        host=os.environ.get("XENO_HOST", "127.0.0.1"),
        port=num("XENO_PORT", 8000),
        daily_per_ip=num("XENO_DAILY_PER_IP", 200),
        daily_total=num("XENO_DAILY_TOTAL", 1000),
        max_tokens=num("XENO_MAX_TOKENS", 200),
        api_url=os.environ.get("ANTHROPIC_API_URL", ANTHROPIC_URL),
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
        "Toi la ban demo, chua co API key nen chi biet noi may cau nay thoi.",
        "Duoc chu! Toi se ghi nho dieu do (trong mot luot).",
    ],
    "en": [
        "Hello! I am Xeno. One blink and I get it.",
        "That sounds great! Tell me more.",
        "Hmm, thinking... done, I agree!",
        "You type Morse faster than my toaster boots.",
        "Let me blink once to think about that.",
        "Nice one! Want to try a harder question?",
        "I am the demo version with no API key, so I only know a few lines.",
        "Sure thing! I will remember that for one turn.",
    ],
}


def mock_reply(message: str, history: list, lang: str, max_chars: int) -> str:
    seed = hashlib.sha256((message + str(len(history))).encode("utf-8")).digest()[0]
    lines = MOCK["vi" if lang == "vi" else "en"]
    return sanitize(lines[seed % len(lines)], max_chars)


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


# -- Anthropic call -------------------------------------------------------------------------

def call_anthropic(cfg: Config, req: dict) -> str:
    body = json.dumps({
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "system": build_system(req["persona"], req["lang"], req["max_chars"]),
        "messages": build_messages(req["message"], req["history"]),
    }).encode("utf-8")
    http_req = urllib.request.Request(
        cfg.api_url,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": cfg.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(http_req, timeout=cfg.timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as err:
        # Log the status only. The body can echo parts of the request.
        print(f"[api] Anthropic returned HTTP {err.code}", file=sys.stderr)
        if err.code in (401, 403):
            raise ApiError(502, "upstream_auth", "The server API key was rejected.")
        if err.code == 404:
            raise ApiError(502, "upstream_model", "The model name was not found. Check XENO_MODEL.")
        if err.code == 429:
            raise ApiError(503, "upstream_busy", "The AI service is busy. Try again in a moment.")
        raise ApiError(502, "upstream_error", "The AI service returned an error.")
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        print(f"[api] request failed: {type(err).__name__}", file=sys.stderr)
        raise ApiError(504, "upstream_timeout", "The AI service did not answer in time.")
    except ValueError:
        raise ApiError(502, "upstream_error", "The AI service sent an unreadable answer.")

    parts = data.get("content") if isinstance(data, dict) else None
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text") if parts else ""
    if not text.strip():
        raise ApiError(502, "empty_reply", "The AI sent an empty reply.")
    return text


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

class Handler(BaseHTTPRequestHandler):
    server_version = "XenoServer/1.0"
    protocol_version = "HTTP/1.1"

    cfg: Config
    limiter: RateLimiter

    # -- plumbing

    def log_message(self, fmt, *args):  # keep the console quiet
        if self.path.startswith("/api/"):
            sys.stderr.write("[api] %s %s\n" % (self.command, self.path))

    def _security_headers(self) -> None:
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
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

    # -- routes

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/status":
            return self._send_json(200, {
                "name": AI_NAME,
                "mode": "live" if self.cfg.live else "mock",
                "model": self.cfg.model if self.cfg.live else None,
                "mediapipe": all((VENDOR / r).is_file() for r in MEDIAPIPE_FILES.values()),
                "model_file": find_model() is not None,
            })
        if path.startswith("/api/"):
            return self._send_json(404, {"error": "not_found", "message": "Unknown endpoint"})
        self._serve_static(path)

    def do_POST(self):
        path = urlsplit(self.path).path
        if path != "/api/chat":
            return self._send_json(404, {"error": "not_found", "message": "Unknown endpoint"})
        try:
            if not self._same_origin():
                raise ApiError(403, "forbidden", "Cross-origin requests are not allowed.")
            if "application/json" not in self.headers.get("Content-Type", ""):
                raise ApiError(415, "bad_request", "Content-Type must be application/json")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_BODY:
                raise ApiError(413 if length > MAX_BODY else 400, "bad_request", "Body size is not valid")
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                raise ApiError(400, "bad_request", "Body is not valid JSON")
            req = validate_chat(payload)

            ok, why = self.limiter.allow(self._client_ip())
            if not ok:
                msg = ("The daily limit for this demo has been reached. Try again tomorrow."
                       if why == "total" else "You have reached today's message limit.")
                raise ApiError(429, "rate_limited", msg)

            if self.cfg.live:
                raw = call_anthropic(self.cfg, req)
                reply = sanitize(raw, req["max_chars"])
                mock = False
            else:
                reply = mock_reply(req["message"], req["history"], req["lang"], req["max_chars"])
                mock = True
            if not reply:
                raise ApiError(502, "empty_reply", "The AI reply had no usable characters.")
            self._send_json(200, {"reply": reply, "mock": mock})
        except ApiError as err:
            self._send_json(err.status, {"error": err.code, "message": err.message})
        except Exception as err:  # noqa: BLE001
            print(f"[api] unexpected error: {type(err).__name__}", file=sys.stderr)
            self._send_json(500, {"error": "server_error", "message": "Something went wrong on the server."})

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

    mode = f"live ({cfg.model})" if cfg.live else "mock (no ANTHROPIC_API_KEY, scripted replies)"
    print(f"\n{AI_NAME} is running in {mode} mode")
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
