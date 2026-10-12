"""Tests for server.py: sanitizer, validation, mock mode, rate limits, and the
Anthropic call against a local stub. Run from web/:  python -m unittest discover -s tests
"""

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import server  # noqa: E402


def post(base, path, payload, headers=None, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=data, method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as err:
        return err.code, json.load(err)


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=5) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read(), dict(err.headers)


class Running:
    """Runs a server in a thread for the duration of a with-block."""

    def __init__(self, cfg):
        cfg.port = 0
        self.httpd = server.make_server(cfg)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class SanitizeTests(unittest.TestCase):
    def test_shared_vectors(self):
        vectors = json.loads((HERE / "sanitize_vectors.json").read_text(encoding="utf-8"))
        for v in vectors:
            self.assertEqual(server.sanitize(v["in"], v["max"]), v["out"], v["in"])


class ValidationTests(unittest.TestCase):
    def test_ok(self):
        req = server.validate_chat({"message": " hi ", "lang": "en", "maxChars": 40})
        self.assertEqual(req["message"], "hi")
        self.assertEqual(req["history"], [])

    def test_rejects(self):
        bad = [
            None, [], {"message": ""}, {"message": "x" * 301},
            {"message": "x", "lang": "fr"},
            {"message": "x", "maxChars": 5}, {"message": "x", "maxChars": 500},
            {"message": "x", "maxChars": True},
            {"message": "x", "persona": "p" * 801},
            {"message": "x", "history": [{"user": "a", "assistant": "b"}] * 2},
            {"message": "x", "history": [{"user": "a"}]},
            {"message": "x", "history": "no"},
        ]
        for payload in bad:
            with self.assertRaises(server.ApiError, msg=str(payload)[:60]):
                server.validate_chat(payload)


class PromptTests(unittest.TestCase):
    def test_system_prompt(self):
        vi = server.build_system("Funny", "vi", 40)
        self.assertIn("WITHOUT diacritics", vi)
        self.assertIn("under 40 characters", vi)
        self.assertIn("Funny", vi)
        self.assertIn("English", server.build_system("", "en", 60))

    def test_messages_include_one_turn(self):
        msgs = server.build_messages("now", [{"user": "a", "assistant": "b"}])
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user"])
        self.assertEqual(msgs[-1]["content"], "now")


class MockModeTests(unittest.TestCase):
    def test_chat_and_status(self):
        with Running(server.Config()) as s:
            code, body = post(s.base, "/api/chat", {"message": "xin chao", "lang": "vi", "maxChars": 40})
            self.assertEqual(code, 200)
            self.assertTrue(body["mock"])
            self.assertLessEqual(len(body["reply"]), 40)
            self.assertEqual(body["reply"], server.sanitize(body["reply"], 40))
            code, raw, _ = get(s.base, "/api/status")
            self.assertEqual(json.loads(raw)["mode"], "mock")

    def test_bad_requests(self):
        with Running(server.Config()) as s:
            self.assertEqual(post(s.base, "/api/chat", {"message": ""})[0], 400)
            self.assertEqual(post(s.base, "/api/chat", None, raw=b"not json")[0], 400)
            self.assertEqual(post(s.base, "/api/nope", {})[0], 404)
            big = b'{"message":"' + b"a" * 20000 + b'"}'
            self.assertEqual(post(s.base, "/api/chat", None, raw=big)[0], 413)

    def test_cross_origin_blocked(self):
        with Running(server.Config()) as s:
            code, _ = post(s.base, "/api/chat", {"message": "hi"}, headers={"Origin": "https://evil.example"})
            self.assertEqual(code, 403)
            host = s.base.split("//")[1]
            code, _ = post(s.base, "/api/chat", {"message": "hi"}, headers={"Origin": "http://" + host})
            self.assertEqual(code, 200)


class RateLimitTests(unittest.TestCase):
    def test_per_ip(self):
        with Running(server.Config(daily_per_ip=2, daily_total=100)) as s:
            codes = [post(s.base, "/api/chat", {"message": "hi"})[0] for _ in range(3)]
            self.assertEqual(codes, [200, 200, 429])

    def test_total(self):
        with Running(server.Config(daily_per_ip=100, daily_total=1)) as s:
            codes = [post(s.base, "/api/chat", {"message": "hi"})[0] for _ in range(2)]
            self.assertEqual(codes, [200, 429])


class StubHandler(BaseHTTPRequestHandler):
    seen = []
    mode = "ok"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubHandler.seen.append((dict(self.headers), body))
        if StubHandler.mode == "401":
            return self._reply(401, {"error": {"type": "authentication_error"}})
        if StubHandler.mode == "empty":
            return self._reply(200, {"content": [{"type": "text", "text": "   "}]})
        self._reply(200, {"content": [{"type": "text",
                                       "text": "Xin chào! Tôi là “Xeno” 😀 và tôi rất vui được gặp bạn hôm nay nhé"}]})

    def _reply(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class LiveModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.stub.server_address[1]}/v1/messages"

    @classmethod
    def tearDownClass(cls):
        cls.stub.shutdown()
        cls.stub.server_close()

    def cfg(self):
        return server.Config(api_key="test-key", model="test-model", api_url=self.url)

    def test_request_shape_and_cleaning(self):
        StubHandler.seen.clear()
        StubHandler.mode = "ok"
        with Running(self.cfg()) as s:
            payload = {"message": "toi dang di choi", "lang": "vi", "maxChars": 40,
                       "persona": "Very funny", "history": [{"user": "chao", "assistant": "XIN CHAO"}]}
            code, body = post(s.base, "/api/chat", payload)
            self.assertEqual(code, 200)
            self.assertFalse(body["mock"])
            self.assertLessEqual(len(body["reply"]), 40)
            self.assertTrue(body["reply"].startswith("XIN CHAO! TOI LA XENO"))
            self.assertEqual(body["reply"], server.sanitize(body["reply"], 40))
        headers, sent = StubHandler.seen[-1]
        lower = {k.lower(): v for k, v in headers.items()}
        self.assertEqual(lower["x-api-key"], "test-key")
        self.assertEqual(lower["anthropic-version"], "2023-06-01")
        self.assertEqual(sent["model"], "test-model")
        self.assertIn("Very funny", sent["system"])
        self.assertEqual([m["role"] for m in sent["messages"]], ["user", "assistant", "user"])
        self.assertNotIn("temperature", sent)
        self.assertNotIn("top_p", sent)

    def test_upstream_errors(self):
        for mode, expected in (("401", 502), ("empty", 502)):
            StubHandler.mode = mode
            with Running(self.cfg()) as s:
                code, body = post(s.base, "/api/chat", {"message": "hi"})
                self.assertEqual(code, expected, mode)
                self.assertNotIn("test-key", json.dumps(body))
        StubHandler.mode = "ok"

    def test_status_never_leaks_key(self):
        with Running(self.cfg()) as s:
            _, raw, _ = get(s.base, "/api/status")
            self.assertNotIn(b"test-key", raw)
            self.assertEqual(json.loads(raw)["mode"], "live")


class StaticTests(unittest.TestCase):
    def test_index_headers_and_traversal(self):
        with Running(server.Config()) as s:
            code, _, headers = get(s.base, "/")
            if (server.PUBLIC / "index.html").is_file():
                self.assertEqual(code, 200)
                self.assertIn("text/html", headers["Content-Type"])
                self.assertIn("wasm-unsafe-eval", headers["Content-Security-Policy"])
                self.assertEqual(headers["Cache-Control"], "no-store")
            for bad in ("/../server.py", "/%2e%2e/server.py", "/js/../../server.py", "/vendor/mediapipe/../../server.py"):
                code, body, _ = get(s.base, bad)
                self.assertEqual(code, 404, bad)
                self.assertNotIn(b"ThreadingHTTPServer", body)

    def test_mime_types(self):
        self.assertEqual(server.MIME[".mjs"], "text/javascript; charset=utf-8")
        self.assertEqual(server.MIME[".wasm"], "application/wasm")


if __name__ == "__main__":
    unittest.main()
