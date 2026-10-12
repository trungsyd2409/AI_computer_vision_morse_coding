# Xeno Assistant (web demo)

A browser version of Blink Morse. You type Morse code by blinking, and an AI called Xeno Assistant answers by blinking Morse back at you. This folder runs on your own computer. It does not change the desktop app in the parent folder.

## Quick start

You need Python 3.9 or newer and Chrome or Edge on a desktop. No Node.js and no extra Python packages.

```bash
cd web
python server.py
```

Open http://localhost:8000 and press **Turn on camera and start**.

On the first run the server downloads MediaPipe (about 20 MB) into `vendor/`. It reuses the face model `../models/face_landmarker.task` from the desktop app. If that file is missing it downloads the model into `web/models/`.

### Use a real AI

Without a key the app runs in **mock mode**: Xeno answers with a few scripted lines, which is enough to test the whole screen. To get real answers:

1. Copy `.env.example` to `.env`.
2. Put your Anthropic API key after `ANTHROPIC_API_KEY=`.
3. Restart `python server.py`. The start screen now says "Live AI".

The key stays on the server. The browser never sees it. `.env` is ignored by git.

## How to use it

| Action | How |
| --- | --- |
| Dot | Close both eyes briefly (less than 1.5 t) |
| Dash | Keep both eyes closed 1.5 t or longer |
| Next letter | Pause about 3 t with your eyes open |
| Next word | Pause about 7 t |
| Send | Stop for 3 seconds after the last letter, or press Enter |
| Delete | Backspace or Delete removes the last symbol or character |

t is the time unit, 100 ms by default. Slow it down in Settings if you are new to this. A wink (one eye) is ignored.

Keys: `Space` blink for testing without a camera, `Enter` send, `Backspace` delete, `P` pause, `M` mute, `X` settings, `Esc` close settings.

Settings let you change the language (Vietnamese without accents, or English), the camera and its mirror and eye-swap options, the closing threshold, the time unit, the auto-send delay, Xeno's personality (funny by default, or write your own), Xeno's blink speed and reply length, sound, and Xeno's picture (built-in robot, or your own open-eyes and closed-eyes images).

## How it works (short version)

1. Your webcam picture is analysed inside your browser by Google's MediaPipe Face Landmarker. It reports how closed each eye is, many times a second. The video is never uploaded.
2. A small program turns "eyes closed for this long" into dots and dashes, and a second one turns dots, dashes and pauses into letters. This is the same logic as the desktop app, ported to JavaScript.
3. When you stop typing for 3 seconds, your sentence goes to the local server. The server asks the AI for a short answer, cleans it so it only has characters that exist in Morse, and sends it back.
4. Xeno "types" the answer: its eyes close for a dot or a dash, using the same time unit as yours, with a beep for each symbol.

## How it works (technical)

```
web/
  server.py            stdlib HTTP server: static files, /api/chat, /api/status, first-run downloads
  public/
    index.html, css/app.css, fonts/ (Geist, SIL OFL)
    js/blinks.js       EyeSignal + BlinkDetector      (port of blink_morse/blinks.py)
    js/morse.js        MORSE_TABLE + MorseComposer    (port of blink_morse/morse.py)
    js/tracker.js      camera + FaceLandmarker (VIDEO mode, GPU then CPU delegate)
    js/input.js        detector + composer wiring
    js/chat.js         turn state machine, auto-send, API client, 1-turn memory
    js/player.js       AI reply timeline (1/3/1/3/7 units) and playback
    js/sanitize.js     keeps only characters in the Morse table
    js/character.js    designed SVG face and custom-image face
    js/hud.js          Morse strips, chart, eye chips, countdown ring
    js/audio.js        Web Audio beeps (AI 440 Hz triangle, optional user 660 Hz)
    js/settings.js, store.js, persona.js, i18n.js, main.js
  tests/               see below
```

- **Eye mapping.** MediaPipe names the eyes after the face in the picture. The desktop app mirrors the frame before detection, so there the user's left eye is `eyeBlinkRight`. The web app sends the unmirrored video to MediaPipe, so your left eye is `eyeBlinkLeft`. Mirroring is a display option only. "Swap eyes" in Settings exists for unusual cameras.
- **Detector.** Per eye, a baseline follows the resting closure level (fast down, slow up, only from open frames). Normalised closure goes through hysteresis (close at the threshold, release 0.15 below it). Only both eyes closed counts. A dash fires at 1.5 t while the eyes are still shut, a dot fires when they open after 2 open frames.
- **Auto-send.** The countdown uses the time since the eyes last opened. It shows only when no letter is pending and the text is not empty. Any new blink restarts it.
- **API.** `POST /api/chat` with `message` (up to 300 characters), `lang` (`vi` or `en`), `persona` (up to 800), `maxChars` (20 to 120) and `history` (at most one `{user, assistant}` turn). The server builds the system prompt, calls the Anthropic Messages API, then runs the reply through a sanitizer (strip accents, upper case, map smart quotes, replace anything outside the Morse table with a space, cut at a word boundary). The same sanitizer exists in JavaScript and both are tested against `tests/sanitize_vectors.json`.
- **Limits.** Per IP per UTC day (default 200) and total per day (default 1000). Over the limit gives HTTP 429. Requests whose `Origin` does not match the server host are rejected, so other websites cannot use your key through localhost.
- **Security headers.** Strict CSP (`script-src 'self' 'wasm-unsafe-eval'`, no inline scripts or styles), `nosniff`, no referrer, camera allowed for this origin only. The server listens on 127.0.0.1 only.
- **Privacy.** Message text goes to the server and then to the AI provider. Nothing is logged by the server except request lines for `/api/*` and error types. Chat history lives in the page only. Settings and chosen images live in your browser's `localStorage`.

## Configuration (`.env`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | empty | Empty means mock mode |
| `XENO_MODEL` | `claude-haiku-5-5` | Model for replies. Check the model name in Anthropic's docs if you get a "model not found" message |
| `XENO_PORT` | 8000 | Port |
| `XENO_HOST` | 127.0.0.1 | Keep this unless you know why you change it |
| `XENO_DAILY_PER_IP` | 200 | Messages per client per UTC day |
| `XENO_DAILY_TOTAL` | 1000 | Messages for everyone per UTC day |
| `XENO_MAX_TOKENS` | 200 | Cap on tokens per reply |

`python server.py --port 8123` overrides the port. `python server.py --setup-only` only downloads the files and exits.

## Tests

```bash
# JavaScript logic: parity with the Python detector, sanitizer, AI timeline (Node 20+)
node --test tests/logic.test.mjs tests/timeline.test.mjs

# Server: validation, mock mode, rate limits, a stubbed AI endpoint
python -m unittest discover -s tests

# Rebuild the golden traces from the real Python modules
python tests/gen_golden.py ../blink_morse

# Optional browser test with a fake camera (pip install playwright)
python tests/e2e_smoke.py
```

`golden.json` holds blink traces and the outputs the Python detector and composer produced for them. The JavaScript port must reproduce every output (event kinds exactly, numbers within 1e-9). If you change the Python logic, regenerate the file and the JavaScript tests tell you what to port.

The browser test uses a fake camera without a face, so it checks the page, the chat loop and the keyboard mode. Real blink detection needs a person in front of a real webcam.

## Troubleshooting

- **The camera does not start.** Allow camera access for localhost in the browser's address bar. Close other apps that use the camera. Pick another device in Settings.
- **"The face tracking file is missing."** The first-run download failed. Run `python server.py --setup-only` with internet access, or put `face_landmarker.task` in `web/models/`. For MediaPipe, download `@mediapipe/tasks-vision@0.10.21` from npm and copy `vision_bundle.mjs` and the `wasm` folder into `web/vendor/mediapipe/`.
- **Blinks are missed or doubled.** Change "Eye closing threshold" in Settings (lower for more sensitive). Face the camera, light your face evenly, and avoid glasses glare. Raise the time unit t if dots and dashes get mixed up.
- **Stray letters while waiting for auto-send.** A natural blink during the 3-second wait is read as a dot and can add an E or a T. Use `P` to pause while you think, Backspace to remove the extra letter, or raise the auto-send delay. A deliberate "send" gesture is a candidate for a later version.
- **Low FPS.** The badge in the top right shows face tracking speed. Below 20 FPS typing gets unreliable. Close heavy tabs and prefer a machine with a working GPU in Chrome.

## Going public later

Plan: serve `public/` from Cloudflare Pages, replace `/api/chat` with a Cloudflare Worker that holds the key, and move the daily counters to Workers KV or a Durable Object. `server.py` and the Worker should share the prompt and the sanitizer rules (`tests/sanitize_vectors.json` covers the sanitizer). Camera access needs HTTPS, which Pages provides.

## Credits

Face tracking: Google MediaPipe Tasks Vision 0.10.21 (Apache 2.0). Fonts: Geist and Geist Mono by Vercel (SIL Open Font License, see `public/fonts/LICENSE-Geist.txt`). Morse table: ITU-R M.1677-1.
