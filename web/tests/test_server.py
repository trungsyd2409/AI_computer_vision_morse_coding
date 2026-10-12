"""Tests for server.py: sanitizer, validation, mock mode, rate limits, and the
Anthropic call against a local stub. Run from web/:  python -m unittest discover -s tests
"""

import json
import os
import sys
import threading
import unittest
import unittest.mock
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
    paths = []
    mode = "ok"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubHandler.seen.append((dict(self.headers), body))
        StubHandler.paths.append(self.path)
        if "bad-model" in self.path or body.get("model") == "bad-model":
            return self._reply(404, {"error": {"status": "NOT_FOUND"}})
        if "generateContent" in self.path:
            return self._gemini(body)
        if StubHandler.mode == "401":
            return self._reply(401, {"error": {"type": "authentication_error"}})
        if StubHandler.mode == "empty":
            return self._reply(200, {"content": [{"type": "text", "text": "   "}]})
        self._reply(200, {"content": [{"type": "text",
                                       "text": "Xin chào! Tôi là “Xeno” 😀 và tôi rất vui được gặp bạn hôm nay nhé"}]})

    def _gemini(self, body):
        if StubHandler.mode == "401":
            return self._reply(400, {"error": {"status": "INVALID_ARGUMENT"}})  # Gemini's answer for a bad key
        if StubHandler.mode == "no-thinking" and "thinkingConfig" in body.get("generationConfig", {}):
            return self._reply(400, {"error": {"status": "INVALID_ARGUMENT"}})
        if StubHandler.mode == "empty":
            return self._reply(200, {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]})
        self._reply(200, {"candidates": [{"content": {"role": "model", "parts": [
            {"text": "hidden reasoning", "thought": True},
            {"text": "Xin chào! Tôi là “Xeno” 😀 và tôi rất vui được gặp bạn hôm nay nhé"}]}}]})

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
        for mode in ("401", "empty"):
            StubHandler.mode = mode
            with Running(self.cfg()) as s:
                code, body = post(s.base, "/api/chat", {"message": "hi"})
                self.assertEqual(code, 200, mode)           # the fallback replies answer
                self.assertTrue(body["mock"] and body["fallback"], mode)
                self.assertNotIn("test-key", json.dumps(body))
        StubHandler.mode = "ok"

    def test_status_never_leaks_key(self):
        with Running(self.cfg()) as s:
            _, raw, _ = get(s.base, "/api/status")
            self.assertNotIn(b"test-key", raw)
            self.assertEqual(json.loads(raw)["mode"], "live")


class GeminiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.stub.server_address[1]}/v1beta"

    @classmethod
    def tearDownClass(cls):
        cls.stub.shutdown()
        cls.stub.server_close()

    def cfg(self):
        return server.Config(api_key="gem-key", provider="gemini", model="gemini-test", api_url=self.url)

    def test_request_shape_and_cleaning(self):
        StubHandler.seen.clear()
        StubHandler.paths.clear()
        StubHandler.mode = "ok"
        with Running(self.cfg()) as s:
            payload = {"message": "toi dang di choi", "lang": "vi", "maxChars": 40,
                       "persona": "Very funny", "history": [{"user": "chao", "assistant": "XIN CHAO"}]}
            code, body = post(s.base, "/api/chat", payload)
            self.assertEqual(code, 200)
            self.assertFalse(body["mock"])
            self.assertTrue(body["reply"].startswith("XIN CHAO! TOI LA XENO"))
            self.assertNotIn("HIDDEN", body["reply"])  # thought parts are dropped
        headers, sent = StubHandler.seen[-1]
        lower = {k.lower(): v for k, v in headers.items()}
        self.assertEqual(lower["x-goog-api-key"], "gem-key")
        self.assertEqual(StubHandler.paths[-1], "/v1beta/models/gemini-test:generateContent")
        self.assertNotIn("gem-key", StubHandler.paths[-1])
        self.assertIn("Very funny", sent["systemInstruction"]["parts"][0]["text"])
        self.assertEqual([c["role"] for c in sent["contents"]], ["user", "model", "user"])

    def test_retries_without_thinking_field(self):
        StubHandler.seen.clear()
        StubHandler.mode = "no-thinking"
        with Running(self.cfg()) as s:
            code, body = post(s.base, "/api/chat", {"message": "hi"})
            self.assertEqual(code, 200)
        self.assertEqual(len(StubHandler.seen), 2)
        self.assertNotIn("thinkingConfig", StubHandler.seen[-1][1]["generationConfig"])
        StubHandler.mode = "ok"

    def test_errors_do_not_leak_key(self):
        for mode in ("401", "empty"):
            StubHandler.mode = mode
            with Running(self.cfg()) as s:
                code, body = post(s.base, "/api/chat", {"message": "hi"})
                self.assertEqual(code, 200, mode)
                self.assertTrue(body["fallback"], mode)
                self.assertNotIn("gem-key", json.dumps(body))
        StubHandler.mode = "ok"

    def test_provider_choice_from_env(self):
        keep = {k: os.environ.pop(k, None) for k in
                ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "XENO_PROVIDER", "XENO_MODEL",
                 "XENO_CONFIG")}
        os.environ["XENO_CONFIG"] = os.path.join(os.path.dirname(__file__), "no-such-config.json")
        try:
            self.assertEqual(server.config_from_env().provider, "anthropic")
            os.environ["GEMINI_API_KEY"] = "g"
            cfg = server.config_from_env()
            self.assertEqual((cfg.provider, cfg.api_key, cfg.model), ("gemini", "g", "gemini-3.5-flash-lite"))
            os.environ["ANTHROPIC_API_KEY"] = "a"
            self.assertEqual(server.config_from_env().provider, "anthropic")  # both set: Anthropic unless chosen
            os.environ["XENO_PROVIDER"] = "gemini"
            self.assertEqual(server.config_from_env().api_key, "g")
            os.environ["XENO_MODEL"] = "my-model"
            self.assertEqual(server.config_from_env().model, "my-model")
        finally:
            for k in keep:
                os.environ.pop(k, None)
                if keep[k] is not None:
                    os.environ[k] = keep[k]


class FallbackChainTests(unittest.TestCase):
    """config.json: models are tried in order, then the canned replies."""

    @classmethod
    def setUpClass(cls):
        cls.stub = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        port = cls.stub.server_address[1]
        cls.gem = f"http://127.0.0.1:{port}/v1beta"
        cls.ant = f"http://127.0.0.1:{port}/v1/messages"

    @classmethod
    def tearDownClass(cls):
        cls.stub.shutdown()
        cls.stub.server_close()

    def setUp(self):
        StubHandler.mode = "ok"
        StubHandler.seen.clear()
        StubHandler.paths.clear()

    def cfg(self, *targets, **kw):
        chain = [server.Target(p, m, "k-" + m, self.gem if p == "gemini" else self.ant, 5.0)
                 for p, m in targets]
        return server.Config(chain=chain, **kw)

    def test_next_model_answers_and_failed_one_is_skipped_for_a_while(self):
        cfg = self.cfg(("gemini", "bad-model"), ("gemini", "good-model"))
        with Running(cfg) as s:
            code, body = post(s.base, "/api/chat", {"message": "hi"})
            self.assertEqual(code, 200)
            self.assertFalse(body["mock"])
            self.assertTrue(body["reply"].startswith("XIN CHAO"))
            self.assertEqual([p.split("/")[-1].split(":")[0] for p in StubHandler.paths],
                             ["bad-model", "good-model"])
            StubHandler.paths.clear()
            post(s.base, "/api/chat", {"message": "again"})
            self.assertEqual([p.split("/")[-1].split(":")[0] for p in StubHandler.paths], ["good-model"])

    def test_cooldown_zero_tries_the_first_model_again(self):
        cfg = self.cfg(("gemini", "bad-model"), ("gemini", "good-model"), cooldown=0)
        with Running(cfg) as s:
            post(s.base, "/api/chat", {"message": "hi"})
            StubHandler.paths.clear()
            post(s.base, "/api/chat", {"message": "again"})
            self.assertEqual(len(StubHandler.paths), 2)

    def test_falls_over_to_another_provider(self):
        cfg = self.cfg(("gemini", "bad-model"), ("anthropic", "good-model"))
        with Running(cfg) as s:
            code, body = post(s.base, "/api/chat", {"message": "hi"})
            self.assertEqual(code, 200)
            self.assertFalse(body["mock"])
        headers, sent = StubHandler.seen[-1]
        self.assertEqual({k.lower(): v for k, v in headers.items()}["x-api-key"], "k-good-model")
        self.assertEqual(sent["model"], "good-model")

    def test_no_model_answers_uses_config_replies(self):
        lines = {"vi": ["DONG DU PHONG"], "en": ["CANNED LINE"]}
        cfg = self.cfg(("gemini", "bad-model"), ("anthropic", "bad-model"), fallback=lines)
        with Running(cfg) as s:
            code, body = post(s.base, "/api/chat", {"message": "hi", "lang": "en"})
            self.assertEqual((code, body["reply"]), (200, "CANNED LINE"))
            self.assertTrue(body["mock"] and body["fallback"])
            # every model is now cooling down, yet they are all tried again
            StubHandler.paths.clear()
            code, body = post(s.base, "/api/chat", {"message": "hi", "lang": "vi"})
            self.assertEqual(body["reply"], "DONG DU PHONG")
            self.assertEqual(len(StubHandler.paths), 2)

    def test_no_models_at_all_is_plain_mock_without_notice(self):
        lines = {"vi": ["A"], "en": ["ONLY LINE"]}
        with Running(server.Config(fallback=lines)) as s:
            code, body = post(s.base, "/api/chat", {"message": "hi", "lang": "en"})
            self.assertEqual(body["reply"], "ONLY LINE")
            self.assertTrue(body["mock"])
            self.assertFalse(body["fallback"])

    def test_reply_with_no_usable_characters_moves_on(self):
        # the first model answers, but only with characters Morse cannot send
        cfg = self.cfg(("gemini", "good-model"), ("anthropic", "good-model"))
        original = server.sanitize
        calls = []

        def picky(text, n):
            calls.append(1)
            return "" if len(calls) == 1 else original(text, n)
        server.sanitize = picky
        try:
            with Running(cfg) as s:
                code, body = post(s.base, "/api/chat", {"message": "hi"})
        finally:
            server.sanitize = original
        self.assertEqual(code, 200)
        self.assertFalse(body["mock"])
        self.assertEqual(len(StubHandler.seen), 2)

    def test_config_file_is_read(self):
        import tempfile
        keep = {k: os.environ.pop(k, None) for k in
                ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "XENO_CONFIG", "MY_KEY")}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "config.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({
                        "models": [
                            {"provider": "gemini", "model": "m1"},
                            {"provider": "gemini", "model": "m2", "key_env": "MY_KEY", "timeout_seconds": 3},
                            {"provider": "anthropic", "model": "m3"},
                            {"provider": "nope", "model": "m4"},
                        ],
                        "cooldown_seconds": 5,
                        "fallback_replies": {"vi": ["xin chao"], "en": ["hello"]},
                    }, fh)
                os.environ["XENO_CONFIG"] = path
                # no keys: nothing usable, and the single-key settings are not used either
                cfg = server.config_from_env()
                self.assertEqual(cfg.targets(), [])
                self.assertFalse(cfg.live)
                self.assertEqual(cfg.fallback, {"vi": ["xin chao"], "en": ["hello"]})
                os.environ["GEMINI_API_KEY"] = "g"
                os.environ["MY_KEY"] = "other"
                os.environ["ANTHROPIC_API_KEY"] = "a"
                cfg = server.config_from_env()
                self.assertEqual([(t.provider, t.model, t.api_key, t.timeout) for t in cfg.targets()],
                                 [("gemini", "m1", "g", 8.0), ("gemini", "m2", "other", 3.0),
                                  ("anthropic", "m3", "a", 8.0)])
                self.assertEqual(cfg.cooldown, 5.0)
                # a broken file falls back to the .env settings
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("{ not json")
                cfg = server.config_from_env()
                self.assertEqual([(t.provider, t.api_key) for t in cfg.targets()], [("anthropic", "a")])
                self.assertIsNone(cfg.fallback)
        finally:
            for k in keep:
                os.environ.pop(k, None)
                if keep[k] is not None:
                    os.environ[k] = keep[k]

    def test_bad_fallback_replies_are_ignored(self):
        self.assertIsNone(server._clean_fallback({"vi": [], "en": ["x"]}))
        self.assertIsNone(server._clean_fallback({"vi": ["x"]}))
        self.assertIsNone(server._clean_fallback("text"))
        ok = server._clean_fallback({"vi": [" a ", 3, ""], "en": ["b" * 500]})
        self.assertEqual(ok["vi"], ["a"])
        self.assertEqual(len(ok["en"][0]), server.MAX_FALLBACK_CHARS)


def call(base, path, payload=None, cookie=None, headers=None, method=None):
    """Like post/get but returns (status, json or bytes, response headers) and can send a cookie."""
    h = dict(headers or {})
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        h["Content-Type"] = "application/json"
    if cookie:
        h["Cookie"] = cookie
    req = urllib.request.Request(base + path, data=data, headers=h,
                                 method=method or ("POST" if payload is not None else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            code, body, hdrs = resp.status, resp.read(), resp.headers
    except urllib.error.HTTPError as err:
        code, body, hdrs = err.code, err.read(), err.headers
    try:
        body = json.loads(body)
    except ValueError:
        pass
    return code, body, hdrs


def sign_in(base, password):
    code, body, hdrs = call(base, "/api/login", {"password": password})
    cookie = hdrs.get("Set-Cookie", "").split(";")[0] if code == 200 else None
    return code, body, cookie


class AuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.stub.server_address[1]}/v1/messages"

    @classmethod
    def tearDownClass(cls):
        cls.stub.shutdown()
        cls.stub.server_close()

    def cfg(self, **kw):
        base = dict(master_password="master-pass-1", user_password="user-pass-22",
                    session_secret=b"s" * 32)
        base.update(kw)
        return server.Config(**base)

    def test_error_reply_keeps_keep_alive_connection_clean(self):
        # A rejected POST must still consume its body, or the next request on the
        # same connection is read as garbage ("{}GET ...").
        import http.client
        with Running(self.cfg()) as s:
            conn = http.client.HTTPConnection("127.0.0.1", s.httpd.server_address[1], timeout=5)
            hdr = {"Content-Type": "application/json", "Origin": s.base}
            conn.request("POST", "/api/login", json.dumps({"password": "nope"}), hdr)
            r = conn.getresponse(); r.read()
            self.assertEqual(r.status, 401)
            conn.request("POST", "/api/logout", "{}", hdr)
            r = conn.getresponse(); r.read()
            self.assertEqual(r.status, 200)
            conn.request("GET", "/api/session")
            r = conn.getresponse(); r.read()
            self.assertEqual(r.status, 200)
            conn.close()

    def test_locked_until_signed_in(self):
        with Running(self.cfg()) as s:
            code, body, _ = call(s.base, "/api/session")
            self.assertEqual((code, body), (200, {"auth": True, "role": None}))
            self.assertEqual(call(s.base, "/api/chat", {"message": "hi"})[0], 401)
            self.assertEqual(call(s.base, "/api/defaults")[0], 401)
            code, body, _ = call(s.base, "/api/status")
            self.assertEqual(body, {"name": server.AI_NAME, "auth": True, "authed": False})
            # the page and its code load (they hold no secrets) but the heavy files do not
            self.assertEqual(call(s.base, "/")[0], 200)
            self.assertEqual(call(s.base, "/css/app.css")[0], 200)
            for path in ("/models/face_landmarker.task", "/vendor/mediapipe/vision_bundle.mjs",
                         "/assets/xeno/open.png", "/server.py", "/config.json"):
                self.assertEqual(call(s.base, path)[0], 401, path)

    def test_wrong_and_right_passwords(self):
        with Running(self.cfg()) as s:
            with unittest.mock.patch("time.sleep"):
                self.assertEqual(sign_in(s.base, "nope")[0], 401)
            code, body, cookie = sign_in(s.base, "user-pass-22")
            self.assertEqual((code, body), (200, {"role": "user"}))
            code, body, hdrs = call(s.base, "/api/login", {"password": "master-pass-1"})
            self.assertEqual(body, {"role": "master"})
            flags = hdrs["Set-Cookie"]
            self.assertIn("HttpOnly", flags)
            self.assertIn("SameSite=Strict", flags)
            self.assertNotIn("Secure", flags)             # plain http on localhost
            self.assertNotIn("master-pass-1", flags)      # the cookie never holds the password
            code, body, _ = call(s.base, "/api/session", cookie=cookie)
            self.assertEqual(body, {"auth": True, "role": "user"})
            self.assertEqual(call(s.base, "/api/chat", {"message": "hi"}, cookie=cookie)[0], 200)
            self.assertEqual(call(s.base, "/api/status", cookie=cookie)[1]["authed"], True)

    def test_master_only_setup_has_no_user_role(self):
        with Running(self.cfg(user_password="")) as s:
            with unittest.mock.patch("time.sleep"):
                self.assertEqual(sign_in(s.base, "")[0], 400)
                self.assertEqual(sign_in(s.base, "x")[0], 401)
            self.assertEqual(sign_in(s.base, "master-pass-1")[1], {"role": "master"})

    def test_logout_clears_cookie(self):
        with Running(self.cfg()) as s:
            _, _, cookie = sign_in(s.base, "user-pass-22")
            code, _, hdrs = call(s.base, "/api/logout", {}, cookie=cookie)
            self.assertEqual(code, 200)
            self.assertIn("Max-Age=0", hdrs["Set-Cookie"])

    def test_cross_origin_login_blocked(self):
        with Running(self.cfg()) as s:
            code, _, _ = call(s.base, "/api/login", {"password": "master-pass-1"},
                              headers={"Origin": "https://evil.example"})
            self.assertEqual(code, 403)

    def test_too_many_wrong_passwords_lock_the_address(self):
        with Running(self.cfg()) as s, unittest.mock.patch("time.sleep"):
            codes = [sign_in(s.base, "wrong")[0] for _ in range(server.LOGIN_MAX_FAILS)]
            self.assertEqual(set(codes), {401})
            code, body, _ = call(s.base, "/api/login", {"password": "master-pass-1"})
            self.assertEqual((code, body["error"]), (429, "too_many_attempts"))

    def test_secure_cookie_behind_https_proxy(self):
        with Running(self.cfg(trust_proxy=True)) as s:
            _, _, hdrs = call(s.base, "/api/login", {"password": "user-pass-22"},
                              headers={"X-Forwarded-Proto": "https"})
            self.assertIn("Secure", hdrs["Set-Cookie"])
        with Running(self.cfg(secure_cookie=True)) as s:
            _, _, hdrs = call(s.base, "/api/login", {"password": "user-pass-22"})
            self.assertIn("Secure", hdrs["Set-Cookie"])

    def test_tokens(self):
        cfg = self.cfg()
        token = server.make_token(cfg, "user", now=1000)
        self.assertEqual(server.read_token(cfg, token, now=1000), "user")
        self.assertIsNone(server.read_token(cfg, token, now=1000 + 13 * 3600))   # expired after 12 h
        body, sig = token.split(".")
        forged = server._b64(json.dumps({"r": "master", "e": 99999999999}).encode()) + "." + sig
        self.assertIsNone(server.read_token(cfg, forged, now=1000))
        self.assertIsNone(server.read_token(self.cfg(session_secret=b"t" * 32), token, now=1000))
        self.assertIsNone(server.read_token(cfg, "garbage", now=1000))
        self.assertIsNone(server.read_token(cfg, "", now=1000))
        # a user token stops working when the user password is removed
        self.assertIsNone(server.read_token(self.cfg(user_password=""), token, now=1000))

    def test_only_master_controls_persona_and_length(self):
        chain = [server.Target("anthropic", "m", "k", self.url, 5.0)]
        cfg = self.cfg(chain=chain, defaults={"personaPreset": "calm", "maxChars": 30})
        payload = {"message": "hi", "lang": "en", "maxChars": 100, "persona": "A pirate who shouts."}
        StubHandler.mode = "ok"
        with Running(cfg) as s:
            _, _, user_cookie = sign_in(s.base, "user-pass-22")
            _, _, master_cookie = sign_in(s.base, "master-pass-1")
            StubHandler.seen.clear()
            self.assertEqual(call(s.base, "/api/chat", payload, cookie=user_cookie)[0], 200)
            system = StubHandler.seen[-1][1]["system"]
            self.assertNotIn("pirate", system)
            self.assertIn(server.DEFAULT_PERSONAS["calm"]["en"], system)
            self.assertIn("under 30 characters", system)
            self.assertEqual(call(s.base, "/api/chat", payload, cookie=master_cookie)[0], 200)
            system = StubHandler.seen[-1][1]["system"]
            self.assertIn("A pirate who shouts.", system)
            self.assertIn("under 100 characters", system)

    def test_auth_setup_checks(self):
        errors, _ = server.check_auth_setup(server.Config(user_password="abcdefgh"))
        self.assertTrue(any("MASTER" in e for e in errors))
        errors, _ = server.check_auth_setup(server.Config(master_password="abcdefgh", user_password="abcdefgh"))
        self.assertTrue(any("different" in e for e in errors))
        errors, _ = server.check_auth_setup(server.Config(host="0.0.0.0"))
        self.assertTrue(any("0.0.0.0" in e for e in errors))
        errors, warnings = server.check_auth_setup(server.Config())          # local, no password
        self.assertEqual(errors, [])
        self.assertTrue(warnings)
        errors, warnings = server.check_auth_setup(server.Config(host="0.0.0.0", master_password="long-enough-1"))
        self.assertEqual((errors, warnings), ([], []))
        _, warnings = server.check_auth_setup(server.Config(master_password="short"))
        self.assertTrue(any("shorter" in w for w in warnings))


class DefaultsTests(unittest.TestCase):
    def test_defaults_endpoint_and_gated_pictures(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            public = Path(tmp)
            (public / "assets" / "xeno").mkdir(parents=True)
            (public / "assets" / "xeno" / "open.png").write_bytes(b"\x89PNG fake")
            (public / "index.html").write_text("<p>hi</p>")
            real = server.PUBLIC
            server.PUBLIC = public
            try:
                cfg = server.Config(master_password="master-pass-1", session_secret=b"s" * 32,
                                    defaults={"characterMode": "custom", "unitMs": 120},
                                    character={"open": "assets/xeno/open.png", "closed": "assets/xeno/closed.png"})
                with Running(cfg) as s:
                    self.assertEqual(call(s.base, "/assets/xeno/open.png")[0], 401)
                    _, _, cookie = sign_in(s.base, "master-pass-1")
                    code, body, _ = call(s.base, "/api/defaults", cookie=cookie)
                    self.assertEqual(code, 200)
                    self.assertEqual(body["role"], "master")
                    self.assertEqual(body["settings"], {"characterMode": "custom", "unitMs": 120})
                    # only the picture that exists on disk is offered
                    self.assertEqual(body["images"], {"open": "/assets/xeno/open.png", "closed": None})
                    self.assertEqual(body["personas"]["funny"], server.DEFAULT_PERSONAS["funny"])
                    code, raw, hdrs = call(s.base, "/assets/xeno/open.png", cookie=cookie)
                    self.assertEqual((code, raw), (200, b"\x89PNG fake"))
            finally:
                server.PUBLIC = real

    def test_config_file_defaults(self):
        import tempfile
        keep = {k: os.environ.pop(k, None) for k in
                ("XENO_CONFIG", "XENO_MASTER_PASSWORD", "XENO_USER_PASSWORD", "XENO_SESSION_SECRET")}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "config.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({
                        "defaults": {"unitMs": 120, "characterMode": "custom", "personaPreset": "pirate",
                                     "volume": "loud", "personaCustom": {"vi": "xin chao", "en": "hello"},
                                     "mystery": 1},
                        "personas": {"funny": {"vi": "Moi vui", "en": ""}, "ghost": {"vi": "x"}},
                        "character": {"open": "assets/xeno/open.png", "closed": "../../etc/passwd"},
                    }, fh)
                os.environ["XENO_CONFIG"] = path
                os.environ["XENO_MASTER_PASSWORD"] = "master-pass-1"
                os.environ["XENO_SESSION_SECRET"] = "fixed"
                cfg = server.config_from_env()
                self.assertEqual(cfg.master_password, "master-pass-1")
                self.assertEqual(cfg.session_secret, b"fixed")
                self.assertEqual(cfg.defaults, {"unitMs": 120, "characterMode": "custom",
                                                "personaCustom": {"vi": "xin chao", "en": "hello"}})
                self.assertEqual(cfg.personas["funny"]["vi"], "Moi vui")
                self.assertEqual(cfg.personas["funny"]["en"], server.DEFAULT_PERSONAS["funny"]["en"])
                # a new name adds a personality of your own; one language is enough
                self.assertEqual(cfg.personas["ghost"], {"vi": "x", "en": "x"})
                self.assertEqual(cfg.character, {"open": "assets/xeno/open.png"})
        finally:
            for k in keep:
                os.environ.pop(k, None)
                if keep[k] is not None:
                    os.environ[k] = keep[k]


    def test_own_persona_is_the_default_for_the_user_password(self):
        import tempfile
        keep = {k: os.environ.pop(k, None) for k in ("XENO_CONFIG", "XENO_MASTER_PASSWORD", "XENO_USER_PASSWORD")}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "config.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({"defaults": {"personaPreset": "pirate"},
                               "personas": {"pirate": {"vi": "Noi nhu hai tac", "en": "Talk like a pirate",
                                                        "label": "Pirate"}}}, fh)
                os.environ["XENO_CONFIG"] = path
                cfg = server.config_from_env()
                self.assertEqual(cfg.defaults["personaPreset"], "pirate")
                self.assertEqual(server.default_persona(cfg, "vi"), "Noi nhu hai tac")
                self.assertEqual(server.default_persona(cfg, "en"), "Talk like a pirate")
                self.assertEqual(cfg.personas["pirate"]["label"], "Pirate")
                # a name that is not under "personas" is dropped, not silently used
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({"defaults": {"personaPreset": "pirat"},
                               "personas": {"pirate": {"vi": "x"}}}, fh)
                self.assertNotIn("personaPreset", server.config_from_env().defaults)
        finally:
            for k in keep:
                os.environ.pop(k, None)
                if keep[k] is not None:
                    os.environ[k] = keep[k]


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
