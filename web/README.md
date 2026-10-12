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

Without a key the app runs in **mock mode**: Xeno answers with a few canned lines, which is enough to test the whole screen. To get real answers:

1. Copy `.env.example` to `.env`.
2. Put a key in it: `GEMINI_API_KEY=` (from Google AI Studio), `ANTHROPIC_API_KEY=` (from the Anthropic Console), or both.
3. Restart `python server.py`. The console prints `live (Gemini gemini-3.5-flash-lite > ...)`, the models it will try in order.

The keys stay on the server. The browser never sees them. `.env` is ignored by git.

### Which models, and what happens when they fail (`config.json`)

`config.json` lists the models in the order the server tries them:

```json
"models": [
  { "provider": "gemini", "model": "gemini-3.5-flash-lite" },
  { "provider": "gemini", "model": "gemini-3.8-flash" },
  { "provider": "anthropic", "model": "claude-haiku-5-5" }
]
```

- A call that fails (wrong key, unknown model, busy, too slow, empty answer) moves on to the next model.
- A model that failed is skipped for `cooldown_seconds` (60 by default), so one broken entry does not slow every message. If all of them are cooling down, they are all tried again.
- `timeout_seconds` (8) is the wait per model and `total_timeout_seconds` (24) the wait for all of them together. The page gives up at 30 seconds, so keep the total below that.
- Entries whose key is missing in `.env` are skipped, and the console says so at start. Use `key_env` to read the key from another variable, for example two Gemini projects: `{ "provider": "gemini", "model": "...", "key_env": "GEMINI_KEY_2" }`.
- When no model answers, Xeno picks one line from `fallback_replies` (one list for `vi`, one for `en`) and the page adds a short note that a default reply was used. With no key at all the same lines are used, without the note. Edit the lists freely, one line is picked per message by a hash of the user's text. Keep lines short and use only letters, digits and basic punctuation, because Xeno has to blink them.
- If `config.json` is missing or broken, the server falls back to the single-key settings in `.env` (`XENO_PROVIDER`, `XENO_MODEL`). If the file has a `models` list, those two variables are ignored.

### Passwords and roles

The page can ask for a password before anything loads. Set them in `.env` (never in `config.json`, and `.env` is in `.gitignore`):

```
XENO_MASTER_PASSWORD=a-long-phrase-for-you
XENO_USER_PASSWORD=a-different-phrase-for-guests
```

| Password | Can chat | Can open Settings |
| --- | --- | --- |
| Master | yes | yes |
| User | yes | no |

- With no master password the sign-in screen is off and everyone is treated as master. That is fine on your own computer. The server refuses to start if you bind it to a public address without a master password.
- The user role also gets the default personality and the default reply length from the server, so changing values in the browser tools does not help. It still has the language, sound and volume controls.
- Wrong passwords are limited to 8 tries per 10 minutes per address.
- A sign-in lasts `XENO_SESSION_HOURS` (12 by default). The **Sign out** button in the header ends it. To sign everyone out at once, change `XENO_SESSION_SECRET` or restart the server when that is left empty.
- Check that git ignores your file: `git check-ignore -v .env` should print a line.

### Where to change the defaults

Everything a new visitor sees first comes from `config.json`:

| Section | What it controls |
| --- | --- |
| `defaults` | Starting values for Settings: `characterMode` (`custom` shows your pictures, `designed` the drawn face), `personaPreset` (`funny`, `calm`, `teacher` or `custom`), `personaCustom` (the `vi` and `en` text used by `custom`), `aiSpeed`, `maxChars`, `unitMs`, `sendDelay`, `closeThreshold`, `mirror`, `swapEyes`, `sound`, `userSound`, `volume`, `lang` |
| `personas` | The text of the three ready-made personalities, in Vietnamese and English |
| `character` | The two default pictures of Xeno |

**Your own Xeno pictures.** Copy them into `public/assets/xeno/` as `open.png` and `closed.png` (or use another name and update `character`). With `characterMode` set to `custom`, they are what everyone sees. If a file is missing the app falls back to the drawn face.

**Your own personality.** Edit the text in `personas.funny` (or change `defaults.personaPreset` to another one), or write a new one in `defaults.personaCustom` and set `personaPreset` to `custom`. Restart the server after editing `config.json`.

Settings > Reset in the browser goes back to these defaults. A master's changes live in that browser only, as the difference from the defaults, so a new default in `config.json` reaches everyone except the values they changed themselves.

### Deploy

Use the same variables on the host as secrets or environment variables: `XENO_MASTER_PASSWORD`, `XENO_USER_PASSWORD`, `XENO_SESSION_SECRET` (long random), the API keys, and `XENO_HOST=0.0.0.0`. Behind an https proxy also set `XENO_SECURE_COOKIE=1` and `XENO_TRUST_PROXY=1`. Do not upload `.env`. Camera access needs https.

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

Your own typing makes sound too: a held tone while your eyes are closed, a short tick when a letter is accepted, a low buzz when a code matches no letter, and two rising notes when a message sends. Turn it off in Settings under Sound. At 3x the AI's time unit is clamped to 30 ms, so a dot is still visible.

Keys: `Space` blink for testing without a camera, `Enter` send, `Backspace` delete, `P` pause, `M` mute, `X` settings, `Esc` close settings.

Settings let you change the language (Vietnamese without accents, or English), the camera and its mirror and eye-swap options, the closing threshold, the time unit, the auto-send delay, Xeno's personality (funny by default, or write your own), Xeno's blink speed (0.5x to 3x) and reply length, sound (Xeno's beep and your own typing sounds are separate switches), and Xeno's picture (built-in robot, or your own open-eyes and closed-eyes images).

## How it works (short version)

1. Your webcam picture is analysed inside your browser by Google's MediaPipe Face Landmarker. It reports how closed each eye is, many times a second. The video is never uploaded.
2. A small program turns "eyes closed for this long" into dots and dashes, and a second one turns dots, dashes and pauses into letters. This is the same logic as the desktop app, ported to JavaScript.
3. When you stop typing for 3 seconds, your sentence goes to the local server. The server asks the AI for a short answer, cleans it so it only has characters that exist in Morse, and sends it back.
4. Xeno "types" the answer: its eyes close for a dot or a dash, using the same time unit as yours, with a beep for each symbol. Everything you and Xeno type is printed on a paper tape across the middle of the page, one letter under its own dots and dashes.

## How it works (technical)

```
web/
  server.py            stdlib HTTP server: static files, /api/chat, /api/status, first-run downloads
  public/
    index.html, css/app.css, fonts/ (Bricolage Grotesque and IBM Plex Mono, SIL OFL)
    js/blinks.js       EyeSignal + BlinkDetector      (port of blink_morse/blinks.py)
    js/morse.js        MORSE_TABLE + MorseComposer    (port of blink_morse/morse.py)
    js/tracker.js      camera + FaceLandmarker (VIDEO mode, GPU then CPU delegate)
    js/input.js        detector + composer wiring
    js/chat.js         turn state machine, auto-send, API client, 1-turn memory
    js/player.js       AI reply timeline (1/3/1/3/7 units) and playback
    js/sanitize.js     keeps only characters in the Morse table
    js/character.js    designed SVG face and custom-image face
    js/hud.js          paper tape, chart, eye gauges, countdown ring
    js/audio.js        Web Audio: held tones (AI 440 Hz triangle, you 660 Hz sine) and short cues (letter, invalid, send)
    js/settings.js, store.js, persona.js, i18n.js, main.js
  tests/               see below
```

- **Eye mapping.** MediaPipe names the eyes after the face in the picture. The desktop app mirrors the frame before detection, so there the user's left eye is `eyeBlinkRight`. The web app sends the unmirrored video to MediaPipe, so your left eye is `eyeBlinkLeft`. Mirroring is a display option only. "Swap eyes" in Settings exists for unusual cameras.
- **Detector.** Per eye, a baseline follows the resting closure level (fast down, slow up, only from open frames). Normalised closure goes through hysteresis (close at the threshold, release 0.15 below it). Only both eyes closed counts. A dash fires at 1.5 t while the eyes are still shut, a dot fires when they open after 2 open frames.
- **Auto-send.** The countdown uses the time since the eyes last opened. It shows only when no letter is pending and the text is not empty. Any new blink restarts it.
- **API.** `POST /api/chat` with `message` (up to 300 characters), `lang` (`vi` or `en`), `persona` (up to 800), `maxChars` (20 to 120) and `history` (at most one `{user, assistant}` turn). The server builds the system prompt, calls the Gemini `generateContent` API or the Anthropic Messages API, then runs the reply through a sanitizer (strip accents, upper case, map smart quotes, replace anything outside the Morse table with a space, cut at a word boundary). The same sanitizer exists in JavaScript and both are tested against `tests/sanitize_vectors.json`.
- **Limits.** Per IP per UTC day (default 200) and total per day (default 1000). Over the limit gives HTTP 429. Requests whose `Origin` does not match the server host are rejected, so other websites cannot use your key through localhost.
- **Security headers.** Strict CSP (`script-src 'self' 'wasm-unsafe-eval'`, no inline scripts or styles), `nosniff`, no referrer, camera allowed for this origin only. The server listens on 127.0.0.1 only.
- **Privacy.** Message text goes to the server and then to the AI provider. Nothing is logged by the server except request lines for `/api/*` and error types. Chat history lives in the page only. Settings and chosen images live in your browser's `localStorage`.

## Configuration (`.env`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` | empty | Google Gemini key. `GOOGLE_API_KEY` also works |
| `ANTHROPIC_API_KEY` | empty | Anthropic key. With neither key set the app runs in mock mode |
| `XENO_PROVIDER` | auto | `gemini` or `anthropic`. Only used when `config.json` has no `models` list |
| `XENO_MODEL` | `gemini-3.5-flash-lite` or `claude-haiku-5-5` | Model for replies. Only used when `config.json` has no `models` list. Check the name in the provider's docs if you get a "model not found" message |
| `XENO_MASTER_PASSWORD` | empty | Password that can open Settings. Empty turns the sign-in screen off |
| `XENO_USER_PASSWORD` | empty | Password that can chat but not open Settings. Needs a master password |
| `XENO_SESSION_SECRET` | random per start | Signs the login cookie. Set it to keep sign-ins across restarts |
| `XENO_SESSION_HOURS` | 12 | How long one sign-in lasts |
| `XENO_SECURE_COOKIE` | 0 | Set to 1 behind https |
| `XENO_TRUST_PROXY` | 0 | Set to 1 behind a proxy, so the client address comes from `X-Forwarded-For` |
| `XENO_CONFIG` | `config.json` | Path to the model and fallback file |
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

Face tracking: Google MediaPipe Tasks Vision 0.10.21 (Apache 2.0). Fonts: Bricolage Grotesque and IBM Plex Mono (SIL Open Font License, see the `LICENSE-*.txt` files in `public/fonts/`). Morse table: ITU-R M.1677-1.
