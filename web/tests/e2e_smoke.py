"""End-to-end smoke test with a headless Chromium (optional).

Needs:  pip install playwright   (and a Chromium that Playwright can find)
Run from web/:  python tests/e2e_smoke.py

It starts server.py in mock mode on a spare port, opens the app with a fake
camera, types with the Space bar (keyboard mode) and checks the whole chat
loop. A fake camera has no face, so real blink detection needs a person in
front of a real webcam. Screenshots go to tests/shots/.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
WEB = HERE.parent
SHOTS = HERE / "shots"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"

UNIT = 150  # ms, slower than default so test timing jitter does not matter
SETTINGS = {"unitMs": UNIT, "sendDelay": 2, "sound": False}

failures = []


def check(cond, label):
    print(("PASS  " if cond else "FAIL  ") + label)
    if not cond:
        failures.append(label)


def start_server():
    env = dict(os.environ, XENO_PORT=str(PORT), ANTHROPIC_API_KEY="")
    proc = subprocess.Popen([sys.executable, str(WEB / "server.py"), "--port", str(PORT)],
                            cwd=WEB, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(100):
        try:
            urllib.request.urlopen(BASE + "/api/status", timeout=1).read()
            return proc
        except Exception:
            time.sleep(0.2)
    proc.kill()
    raise SystemExit("server did not start")


def tap(page, ms):
    page.keyboard.down("Space")
    page.wait_for_timeout(ms)
    page.keyboard.up("Space")


def dot(page):
    tap(page, 70)
    page.wait_for_timeout(UNIT)


def type_dots(page, n):
    for _ in range(n):
        dot(page)


def main():
    SHOTS.mkdir(exist_ok=True)
    server = start_server()
    problems = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=[
                "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                "--autoplay-policy=no-user-gesture-required",
            ])
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, permissions=["camera"])
            ctx.add_init_script(
                "try{ if(!localStorage.getItem('xeno.settings.v1')) localStorage.setItem('xeno.settings.v1', %s) }catch(e){}"
                % json.dumps(json.dumps(SETTINGS)))
            page = ctx.new_page()
            requests = []
            page.on("console", lambda m: problems.append(m.text) if m.type in ("error",) else None)
            page.on("pageerror", lambda e: problems.append(str(e)))
            page.on("request", lambda r: requests.append(r) if r.url.endswith("/api/chat") else None)

            # 1. start screen
            page.goto(BASE)
            page.wait_for_selector("#overlay:not([hidden])")
            for _ in range(40):
                if page.inner_text("#ov-mode").strip():
                    break
                page.wait_for_timeout(100)
            check("giả lập" in page.inner_text("#ov-mode").lower() or "mock" in page.inner_text("#ov-mode").lower(),
                  "overlay shows mock mode")
            page.screenshot(path=str(SHOTS / "1_overlay.png"))

            # 2. camera path under the real CSP: MediaPipe + model + fake camera.
            # Playwright's string waits use eval, which the CSP blocks, so this
            # part only uses locator waits and polling.
            page.click("#btn-start-cam")
            try:
                page.locator("#overlay").wait_for(state="hidden", timeout=90000)
                check(True, "camera mode starts (MediaPipe, model and camera load)")
                fps = 0.0
                for _ in range(60):
                    text = page.inner_text("#fps")
                    digits = "".join(c for c in text if c.isdigit())
                    fps = float(digits) if digits else 0.0
                    if fps > 0:
                        break
                    page.wait_for_timeout(250)
                check(fps > 0, "FPS badge updates: " + page.inner_text("#fps"))
                check(page.is_visible("#cam-hint"), "no-face hint is shown for the fake camera")
                page.wait_for_timeout(500)
                page.screenshot(path=str(SHOTS / "2_camera.png"))
            except Exception as err:  # noqa: BLE001
                check(False, f"camera mode starts ({type(err).__name__}: {str(err)[:160]})")
                page.screenshot(path=str(SHOTS / "2_camera_fail.png"))

            # The remaining steps use string waits, so they run with CSP bypassed.
            ctx.close()
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, permissions=["camera"], bypass_csp=True)
            ctx.add_init_script(
                "try{ if(!localStorage.getItem('xeno.settings.v1')) localStorage.setItem('xeno.settings.v1', %s) }catch(e){}"
                % json.dumps(json.dumps(SETTINGS)))
            page = ctx.new_page()
            page.on("pageerror", lambda e: problems.append(str(e)))
            page.on("request", lambda r: requests.append(r) if r.url.endswith("/api/chat") else None)

            # 3. keyboard path in a fresh page
            page.goto(BASE)
            page.click("#btn-start-key")
            page.wait_for_function("document.getElementById('overlay').hidden")
            page.wait_for_function("document.getElementById('ai-line').textContent.length > 0", timeout=15000)
            page.screenshot(path=str(SHOTS / "3_ai_greeting.png"))
            page.click("#btn-skip")
            page.wait_for_function("document.querySelectorAll('#chat-log .msg.ai').length === 1")
            check(page.inner_text("#chat-log .msg.ai .bubble") == "Xin chao toi la xeno", "greeting lands in the chat")
            page.wait_for_function("document.getElementById('user-status').textContent.length > 0")

            # type HI: H = 4 dots, I = 2 dots
            type_dots(page, 4)
            page.wait_for_timeout(UNIT * 3 + 200)
            check(page.inner_text("#user-line").lower() == "h", "letter H committed after the letter gap")
            type_dots(page, 2)
            page.wait_for_timeout(UNIT * 3 + 200)
            check(page.inner_text("#user-line").lower() == "hi", "text is 'hi'")
            page.wait_for_timeout(300)
            check(page.is_visible("#ring"), "countdown ring is visible while waiting")
            page.screenshot(path=str(SHOTS / "4_countdown.png"))

            page.wait_for_function("document.querySelectorAll('#chat-log .msg.user').length === 1", timeout=8000)
            check(page.inner_text("#chat-log .msg.user .bubble") == "Hi", "message auto-sent after the pause")
            page.wait_for_function("document.getElementById('ai-line').textContent.length > 0", timeout=15000)
            page.screenshot(path=str(SHOTS / "5_ai_replying.png"))
            page.click("#btn-skip")
            page.wait_for_function("document.querySelectorAll('#chat-log .msg.ai').length === 2", timeout=8000)
            check(len(requests) == 1 and json.loads(requests[0].post_data)["history"] == [], "first request has no history")

            # second message must carry exactly one remembered turn
            type_dots(page, 1)
            page.wait_for_timeout(UNIT * 3 + 200)
            page.keyboard.press("Enter")
            page.wait_for_function("document.querySelectorAll('#chat-log .msg.user').length === 2", timeout=8000)
            body = json.loads(requests[1].post_data)
            check(len(body["history"]) == 1 and body["history"][0]["user"] == "hi", "second request remembers one turn")
            check(body["message"] == "e" and body["lang"] == "vi" and body["maxChars"] == 40, "request fields are right")
            check("vui tính" in body["persona"].lower(), "default persona is the funny one")
            page.wait_for_function("document.getElementById('btn-skip') && !document.getElementById('btn-skip').hidden", timeout=8000)
            page.click("#btn-skip")
            page.wait_for_function("document.querySelectorAll('#chat-log .msg.ai').length === 3", timeout=8000)

            # backspace removes the last typed character
            type_dots(page, 1)
            page.wait_for_timeout(UNIT * 3 + 200)
            before = page.inner_text("#user-line")
            page.keyboard.press("Backspace")
            page.wait_for_timeout(100)
            check(before.lower() == "e" and page.inner_text("#user-line") == "",
                  f"Backspace deletes the last character (before={before!r})")

            # 4. settings drawer and language switch
            page.keyboard.press("x")
            page.wait_for_selector("#drawer:not([hidden])")
            page.screenshot(path=str(SHOTS / "6_settings.png"))
            page.click("[data-lang='en']")
            check(page.inner_text("#dr-title") == "Settings", "language switch to English")
            check(page.is_visible("#set-persona") and "playful" in page.input_value("#set-persona").lower(),
                  "persona text follows the language")
            page.fill("#set-persona", "You are a pirate.")
            check(page.input_value("#set-preset") == "custom", "editing the persona switches to Custom")
            page.keyboard.press("Escape")
            check(page.is_hidden("#drawer"), "Escape closes the drawer")
            page.screenshot(path=str(SHOTS / "7_english.png"))

            # custom character images: two small PNGs, kept after a reload
            import struct
            import zlib

            def png(rgb):
                raw = b"".join(b"\x00" + bytes(rgb) * 64 for _ in range(64))
                def chunk(tag, data):
                    body = tag + data
                    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
                return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
                        + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

            (SHOTS / "open.png").write_bytes(png((200, 80, 80)))
            (SHOTS / "closed.png").write_bytes(png((80, 80, 200)))
            page.keyboard.press("x")
            page.set_input_files("#img-open", str(SHOTS / "open.png"))
            page.set_input_files("#img-closed", str(SHOTS / "closed.png"))
            page.wait_for_function("!document.getElementById('prev-closed').hidden")
            page.click("[data-char='custom']")
            check(page.locator("#character img").count() == 2, "custom images replace the designed face")
            page.keyboard.press("Escape")
            page.reload()
            page.click("#btn-start-key")
            page.wait_for_function("document.getElementById('overlay').hidden")
            check(page.locator("#character img").count() == 2, "custom images survive a reload")
            page.screenshot(path=str(SHOTS / "9_custom_character.png"))
            page.click("#btn-skip")

            # 5. narrow screen
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(300)
            overflow = page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
            check(not overflow, "no horizontal scroll at 390 px")
            page.screenshot(path=str(SHOTS / "8_mobile.png"), full_page=True)

            browser.close()
    finally:
        server.terminate()

    # MediaPipe prints an INFO line through console.error, which is not an error.
    problems[:] = [m for m in problems if not m.startswith("INFO:")]
    csp = [m for m in problems if "Content Security Policy" in m or "Refused to" in m]
    check(not csp, f"no CSP violations {csp[:2]}")
    other = [m for m in problems if m not in csp]
    check(not other, f"no console errors {other[:3]}")
    print("\n%d failure(s)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
