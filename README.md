# Frown Morse

Type Morse code by frowning, the way a telegraph operator uses a key: a short frown is a dot, a long frown is a dash, and a relaxed face marks the end of a letter or a word. The app measures the vertical furrow lines that appear between your eyebrows when you frown. Everything is timed in one adjustable unit, and the decoded text appears on screen as you go, no hands needed.

![Interface preview](docs/preview.png)
<sub>Interface preview, rendered with a placeholder camera image.</sub>

---

## Quick start

### 1. Requirements

| | |
|---|---|
| Python | 3.10, 3.11 or 3.12 (MediaPipe does not always support the newest Python right away) |
| Webcam | Any USB, built-in or phone camera. 30 FPS or more is recommended |
| GPU | Anything that supports OpenGL 3.3 (every integrated GPU from the last ten years) |
| OS | Windows 10/11 (main target), macOS and Linux also work |

### 2. Install

**Windows (PowerShell)**

```powershell
cd AI_computer_vision_morse_coding
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**

```bash
cd AI_computer_vision_morse_coding
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> The project uses **pygame-ce**, a maintained fork of pygame. If the classic `pygame` package is already installed in the same environment, remove it first with `pip uninstall pygame`, because both packages provide the `pygame` module.

### 3. Run

```bash
python main.py
```

On the first launch the face tracking model (`face_landmarker.task`, about 3.7 MB) is downloaded into `models/`. Later launches work offline.

### 4. Run the tests (optional)

```bash
python -m unittest discover -s tests -v
```

The tests cover the Morse logic, the blink detector and the left/right eye mapping. They do not need a camera.

---

## How to use it

Sit facing the camera with your face well lit. **Keep your face relaxed for the first 2 seconds**: the app learns what your forehead looks like at rest (the frame between your eyebrows is white while it learns). Then frown, pulling your eyebrows together. All timings come from one time unit **t** (0.1 s by default, adjustable in the settings window):

| Face | Result | With t = 0.1 s |
|---|---|---|
| **Short frown**, shorter than 1.5 t | Dot `·` | under 0.15 s |
| **Long frown**, 1.5 t or longer | Dash `–` | 0.15 s or more |
| **Relaxed for 3 t** | End of letter, the Latin letter pops up | 0.3 s |
| **Relaxed for 7 t** | End of word, a space is added | 0.7 s |

A dash is typed the moment the frown reaches 1.5 t, while you are still frowning, so it never waits for you to relax. A dot can only be known once you relax.

**What is shown on the face.** A thin frame between your eyebrows marks the area that is measured, and the furrow lines found there are outlined. Both are pink while the frown is not strong enough, cyan once it counts (a dot) and violet once it has become a dash. The **Frown** panel shows the score from 0 to 100 % with a white tick at the threshold.

**Recalibrate with C** when the light changes or you move, keeping your face relaxed for 2 seconds. If the app sees a "frown" that lasts more than 4 seconds it assumes the resting face has changed and recalibrates by itself.

**Example: typing "HI"**

1. Frown quickly four times (`····`), then relax. **H** pops up after 0.3 s.
2. Frown quickly twice (`··`) and relax. **I** pops up.
3. Stay relaxed a little longer (0.7 s in total) and a space is added.

**Choosing t.** A frown is slower than a blink, so 0.15 to 0.25 s is often more comfortable than the default. The settings window shows what every rule means in seconds for the current value.

**Hiding the interface.** Press **Z** to hide every panel and show the camera image with its normal colours. While the interface is hidden, face tracking and typing are switched off. Press **Z** again to bring everything back.

Deleting is done on the keyboard.

While a letter is in progress, the chart on the right dims every character you can no longer reach and highlights the exact match, so you never need to memorise the whole alphabet.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `X` | Open or close the settings window |
| `Z` | Hide or show the interface. Hidden = normal camera colours and no typing |
| `P` | Pause or resume listening |
| `C` | Recalibrate the relaxed face (keep it relaxed for 2 seconds) |
| `H` | Show or hide the Morse chart |
| `M` | Mute or unmute sounds |
| `Backspace` | Delete the last symbol, or the last letter if no symbol is pending |
| `Delete` | Clear everything |
| `Esc` / `Q` | Quit |

### Settings window

![Settings window](docs/settings.png)

Press `X` to open it. It has two tabs. Changes apply instantly and are saved to `settings.json` when the app closes.

**Camera tab**
- Live stats: the real capture resolution, camera FPS and app FPS. Green is 30 FPS or more, amber is 15 to 30, red is below 15.
- Device number, with the device name underneath on Windows (for example "Integrated Webcam" or "Camo"), resolution, target FPS (30 or 60), mirror view, auto exposure, and manual exposure, brightness, contrast and gain.
- **Driver settings** (Windows) opens the camera maker's own settings dialog.

**Frown tab**
- **Frown threshold**: how strong the frown score must be to count (0.30 by default), marked by the white tick on the Frown bar.
- **Time unit t**: the base timing, from 0.05 to 0.5 s. A card below the slider spells out the dot, dash, letter and space timings for the current value.

### Using an iPhone or Android phone as the camera

A phone camera is usually much sharper than a laptop webcam, and sharper eyes make blinks easier to read. Install a "phone as webcam" app such as Camo, iVCam or EpocCam on both the phone and the PC, connect the phone by USB, and the phone shows up as an extra camera. In the settings window, step **Device** to the new camera (its name is shown on Windows). Put the phone at eye level and light your face from the front.

### Troubleshooting

| Problem | Fix |
|---|---|
| "Camera unavailable" | Close other apps using the camera (Teams, Zoom, OBS), or pick another device in the settings window. |
| FPS below 15 | Choose 640×480 in settings, turn off auto exposure in a dark room (long exposure halves the frame rate), and plug the laptop in. |
| Frowns are missed | Lower **Frown threshold**, use 1280×720, sit closer, and light your face from above or the side. Light straight from the front hides the furrow lines. |
| Symbols appear without frowning | Raise **Frown threshold**, or press **C** to recalibrate with a relaxed face. |
| Quick frowns come out as dashes | Raise **Time unit t**. |
| Letters or words end before you are done | Raise **Time unit t**. |
| Typing feels slow | Lower **Time unit t**. |
| Input feels slow | Choose 60 FPS in settings. At 30 FPS each frame is 33 ms apart, at 60 FPS only 17 ms. |
| Glasses | Usually fine. Strong reflections on the lenses can hide the eyelids, so tilt the screen or the light slightly. |
| Model download fails | Download the file from the URL in `blink_morse/config.py` and save it as `models/face_landmarker.task`. |

---

## Tech stack

| Part | Library | Why |
|---|---|---|
| Face tracking | [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) | 478 face points plus 52 expression scores, including one blink score per eye, in real time on a normal CPU |
| Camera | [OpenCV](https://opencv.org/) | Reads the webcam and controls exposure, brightness and so on |
| Window and input | [pygame-ce](https://pyga.me/) | Window, keyboard, text rendering and sound playback |
| Rendering | [ModernGL](https://github.com/moderngl/moderngl) (OpenGL 3.3) | Draws the frosted glass panels and the neon glow on the graphics card |
| Maths | [NumPy](https://numpy.org/) | Signal maths and sound synthesis |
| Camera names | [pygrabber](https://github.com/andreaschiavinato/python_grabber) (Windows, optional) | Lists DirectShow device names for the settings window |
| Fonts | Inter and JetBrains Mono | Bundled in `assets/fonts` under the SIL Open Font License |

No sound files and no image assets are shipped: every sound is synthesised at start-up and every shape is drawn in code.

---

## How it works (non-technical overview)

Think of the app as a small assembly line with five stations.

1. **The camera takes pictures.** About 30 to 60 times a second, the webcam sends a new picture of you.

2. **The face reader finds the spot between your eyebrows.** A pre-trained model from Google places 478 points on your face. The app uses them to cut out the small patch of skin between the inner ends of your eyebrows, and looks for thin, dark, vertical lines there: the furrows that appear when you frown. Everybody has some lines at rest, so the app first learns what your relaxed face looks like and measures only the change.

3. **The frown reader times each frown.** It works like a telegraph key, where the length of each press matters. A short frown is a dot; as soon as a frown lasts longer than one and a half time units, it becomes a dash, without waiting for you to relax.

4. **The Morse translator listens to the silence.** Like a radio operator, it treats a short silence as the end of a letter and a longer one as the end of a word. After three time units with a relaxed face, the dots and dashes collected so far are looked up in the international Morse table. After seven, a space is added. If a code does not exist, the app shows a red question mark instead of guessing.

5. **The screen shows the result.** The camera picture is dimmed and information panels that look like frosted glass float around it. A thin frame between your eyebrows shows the furrow lines that were found, the newest symbol ripples when it lands, and the finished letter pops up and is added to the message at the bottom. A short sound confirms every symbol, so you can type without looking.

The app never records or uploads anything. Every picture is processed in memory and thrown away straight after.

## How it works (technical deep dive)

### Frown signal (`frown.py`)

MediaPipe's landmarks and its `browDown` blendshapes hardly move when most people frown, so the input is measured from the image instead:

1. A face-aligned patch (64 px wide) is cut out between the inner brow ends (landmarks 107 and 336), centred on the glabella (landmark 9), from a little above the brows down to landmark 8. Its scale follows the inner-brow distance, so distance to the camera does not matter.
2. A morphological black-hat with a 9×1 horizontal kernel on the L channel keeps only dark details thinner than the kernel horizontally, i.e. thin vertical lines. Broad shadows and horizontal forehead lines are ignored.
3. `energy` = mean response in the middle 70 % of the patch, divided by the median skin brightness, in percent.
4. A 2-second calibration learns the resting energy `E0` and inner-brow distance `d0`; both keep following slow changes while the face is relaxed.
5. `score = 0.8 × clip((E − E0) / max(E0, 0.15) / 2) + 0.2 × clip((d0 − d) / d0 / 0.06)`, smoothed with an exponential filter. A score above 0.5 for more than 4 s triggers a recalibration.

The score is fed to the existing press detector (`blinks.py`) as both "eyes", with its own baseline learning switched off, so the dot/dash timing and the letter/word gaps are unchanged.

### Pipeline

```mermaid
flowchart LR
    A[Camera thread<br/>OpenCV, MJPG] -->|latest frame| B[Mirror + BGR to RGB]
    B --> C[MediaPipe Face Landmarker<br/>VIDEO mode, blendshapes]
    C -->|eyeBlinkLeft / Right| M[Side mapping<br/>image face to user]
    M --> D[BlinkDetector<br/>baseline, hysteresis, episodes]
    D -->|DOT / DASH| F[MorseComposer<br/>pause-driven state machine]
    D -->|pause time| F
    C -->|eye corners| G
    F --> G[HUD<br/>pygame surfaces]
    G --> H[ModernGL compositor<br/>glass + bloom shader]
    B --> H
    I[Settings process] <-->|Pipe| J[Main loop]
```

### Threads and processes

- **Camera thread** (`camera.py`) blocks on `VideoCapture.read()` and publishes only the newest frame through a `threading.Condition`. The main loop wakes up as soon as a frame arrives and never processes the same frame twice. `CAP_PROP_BUFFERSIZE = 1` and MJPG are requested so frames are fresh and 720p can still reach 30 to 60 FPS. On Windows the backends are tried in the order DirectShow, Media Foundation, default, which also covers virtual cameras created by phone-webcam apps.
- **Main thread** runs detection, the blink detector, HUD drawing and the OpenGL composition, in that order. The symbol and its sound are produced before any texture upload or drawing, so rendering never adds to the reaction time.
- **Settings process** (`settings_window.py`). The main window owns an OpenGL context, and a second SDL window in the same process can steal or invalidate it. Running the settings UI in a `spawn` child process avoids that and keeps slider dragging from stalling the render loop. Messages are small tuples over a `multiprocessing.Pipe`.

### Signal: blendshapes and side mapping (`face_tracker.py`)

The Face Landmarker runs in `VIDEO` mode with `output_face_blendshapes=True`. Of the 52 ARKit-style blendshapes, `eyeBlinkLeft` and `eyeBlinkRight` are used. They respond to true lid closure much better than an eye-aspect-ratio computed from the mesh, whose eyelid points tend not to close fully.

MediaPipe names both landmarks and blendshapes after the face **as depicted in the image** (see [google-ai-edge/mediapipe#6368](https://github.com/google-ai-edge/mediapipe/issues/6368)). Mirroring the frame therefore swaps them: in the default selfie view, `eyeBlinkLeft` is the user's right eye. `build_result()` applies that swap once, so every other module only sees the user's own left and right. A unit test pins this down.

### Blink detector (`blinks.py`)

**Per-user baseline.** `EyeSignal` tracks the lower envelope of each raw score while the eye is open (it falls quickly, with a 0.25 s time constant, and rises slowly, with 2 s), capped at 0.4. The normalised closure is `(raw - baseline) / (1 - baseline)`. This matters for people with narrower eyes and for laptop cameras, where looking down at the screen lowers the lids and raises the resting score.

**Hysteresis.** An eye becomes closed above `close_threshold` (0.45) and opens again below `close_threshold - 0.15`.

**Pose per frame.** `open`, `left`, `right` or `both`. Only `both` counts as a blink; winks are ignored.

**Blink timing.** A blink starts on the first frame both eyes are closed. It ends when the eyes have looked open for two frames in a row (so one noisy frame cannot split a blink in two), and its end time is the first of those open frames. With time unit `t`:

| Situation | Result | When |
|---|---|---|
| Blink still closed after 1.5 t | `DASH` | at 1.5 t, eyes still closed |
| Blink ended before 1.5 t | `DOT` | when the eyes open |

The dash is emitted as soon as it can be told apart from a dot, which removes the wait for the eyes to open. The 1.5 t split point sits halfway between the "shorter than t" and "longer than 2 t" rule of thumb, so there is no dead zone where a blink types nothing.

The detector also reports `closed_time` (for the live dot-to-dash preview) and `pause_time`: seconds since the eyes opened after the last blink, which the composer uses for the letter and word gaps. Calling `reset(now)` restarts that clock; the app does this when the interface is shown again with Z, so time spent hidden never counts as a pause.

### Morse state machine (`morse.py`)

`MorseComposer` knows nothing about cameras, so it can be tested with plain function calls. It is driven by symbols and by the idle time reported every frame:

| State | Input | Result |
|---|---|---|
| any | dot / dash | append to `code`, cancel a pending space |
| `code` not empty | idle ≥ `letter_gap` (3 t) | decode, `LETTER` (arms the space) or `INVALID` |
| space armed | idle ≥ `word_gap` (7 t) | `SPACE` (never leading, never doubled) |
| `code` already 6 symbols long | dot / dash | `INVALID` right away, since no code is longer |
| any | Backspace | remove last symbol, otherwise last character |

Both gaps are measured from the same moment, so a new symbol typed between 3 t and 7 t simply continues the same word. The table follows ITU-R M.1677-1 for A to Z, 0 to 9 and common punctuation.

### Rendering (`compositor.py` and `hud.py`)

Each frame uploads three textures and draws one full-screen triangle strip with a single fragment shader:

1. **Camera texture** with mipmaps. A cover crop (like CSS `object-fit: cover`) fills the 800×600 window at any camera aspect ratio without stretching. A moderate filter is applied: 12 % desaturation, a gentle cool tint, 14 % dimming and a soft vignette. A `u_filter` uniform blends between this look and the untouched image; the Z key animates it to 0 over about 0.2 s, which also removes the panels, glow and grain.
2. **Glass panels.** Up to 10 rectangles are passed as uniforms, including the two openness labels that follow the eyes. A rounded-rectangle signed distance field gives anti-aliased edges, a soft drop shadow, a hairline border that is brighter at the top, and an accent colour for flashes. Inside a panel the camera is sampled at mip level 3.4 with five taps, which gives a smooth frosted blur almost for free. The glass keeps about 45 % of the blurred background's brightness with a light smoky tint, clear enough to see the room through it while white text stays readable.
3. **Glow layer.** An opaque black pygame surface where the symbols, meters and letters are drawn again in colour. The shader adds three mip levels of it (1.0, 2.6 and 4.2), which gives the neon bloom without extra framebuffers or blur passes.
4. **UI layer.** A transparent pygame surface with text, meters and icons, blended last. The only things placed near the face are the two openness labels: each sits just outside the outer corner of its eye (landmarks 33 and 263, mapped to the user's side like the blink scores), and its position is smoothed with a light exponential filter so the numbers do not jitter. The smoothing affects only where the label is drawn, never the input.
5. A little animated film grain removes banding in the dark gradients.

### Audio (`audio.py`)

Tones are built with NumPy: a sine wave plus a soft second harmonic and a short attack and release to avoid clicks. The dash lasts three times as long as the dot, as in real Morse timing. The mixer is opened with a 256-sample buffer (about 6 ms) so the sound follows the blink closely.

### Performance budget

On the CPU side (camera upload, HUD drawing, layer upload) a frame takes about 10 ms. The Face Landmarker with blendshapes adds roughly 8 to 20 ms depending on the CPU, which keeps the app between 30 and 60 FPS on a typical laptop. VSync is off so the camera sets the pace and no extra frame of latency is added.

End-to-end, a dash is typed one camera frame plus one detection after the blink reaches 1.5 t, and a dot two frames after the eyes open (the second frame confirms they really are open).

---

## Project structure

```
AI_computer_vision_morse_coding/
├── main.py                  entry point
├── requirements.txt
├── blink_morse/
│   ├── app.py               main loop, wires everything together
│   ├── config.py            constants, colours, saved settings
│   ├── camera.py            threaded webcam reader
│   ├── face_tracker.py      MediaPipe wrapper, model download, eye mapping
│   ├── frown.py             furrow lines between the eyebrows -> frown score
│   ├── blinks.py            press timing detector (dot / dash)
│   ├── morse.py             Morse table and composer state machine
│   ├── hud.py               everything drawn on top of the camera
│   ├── draw.py              anti-aliased shapes, glow sprites, easing
│   ├── compositor.py        ModernGL shader: glass panels and bloom
│   ├── audio.py             synthesised feedback sounds
│   └── settings_window.py   settings window (separate process)
├── assets/fonts/            Inter and JetBrains Mono (OFL)
├── models/                  face model, downloaded on first run
├── docs/                    screenshots for this README
└── tests/                   unit tests for Morse logic and blinks
```

## Credits

- Face tracking model: Google MediaPipe, Apache License 2.0.
- Fonts: Inter by Rasmus Andersson and JetBrains Mono by JetBrains, both under the SIL Open Font License 1.1 (licence files in `assets/fonts`).
