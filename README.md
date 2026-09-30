# Morse Hand

Type Morse code in the air with your right hand. A webcam watches your fingers, each thumb touch becomes a dot, a dash or a command, and the decoded letters appear on screen as you go.

![Interface preview](docs/preview.png)
<sub>Interface preview, rendered with a placeholder camera image.</sub>

---

## Quick start

### 1. Requirements

| | |
|---|---|
| Python | 3.10, 3.11 or 3.12 (MediaPipe does not always support the newest Python right away) |
| Webcam | Any USB or built-in camera. 30 FPS or more is recommended |
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

On the first launch the hand tracking model (`hand_landmarker.task`, about 7.5 MB) is downloaded into `models/`. Later launches work offline.

### 4. Run the tests (optional)

```bash
python -m unittest discover -s tests -v
```

The tests cover the Morse logic and the gesture detector and do not need a camera.

---

## How to use it

Hold your **right hand** up to the camera, palm facing the screen. The left hand is ignored on purpose, so it can rest or hold a coffee.

| Gesture (thumb touches...) | Action |
|---|---|
| **Index** finger | Add a dot `·` |
| **Middle** finger | Add a dash `–` |
| **Ring** finger, once | End the letter and show it |
| **Ring** finger, twice quickly | Add a space |
| **Pinky**, short tap | Delete the last symbol, or the last letter if no symbol is pending |
| **Pinky**, hold for 1 second | Clear everything |

**Example: typing "HI"**

1. Index, index, index, index (`····`) then ring once. **H** pops up.
2. Index, index (`··`) then ring once. **I** pops up.
3. Ring again straight away to add a space before the next word.

### Screen layout

The window is a 480 × 800 portrait (3:5):

| Area | What it shows |
|---|---|
| Top black strip | The Morse chart. While a letter is in progress it dims every character you can no longer reach and highlights the exact match, so you never need to memorise the whole alphabet. |
| Middle | The camera with its original colours, kept at its own aspect ratio so the image is never stretched. The dots and dashes of the letter you are typing appear in the centre, with no box around them, until the letter ends and pops up as a Latin letter. |
| Bottom black strip | The message box at the very bottom, and above it the messages you have already sent, shown as chat bubbles with the time. |

**Sending a message.** Press `Enter` to send what is in the message box. It moves up into the chat as a bubble (newest at the bottom, older ones pushed up) and the box is emptied for the next message. The chat history lasts until the app is closed.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Enter` | Send the message to the chat above the message box |
| `X` | Open or close the camera settings window |
| `H` | Show or hide the Morse chart |
| `M` | Mute or unmute sounds |
| `Backspace` | Delete (same as a pinky tap) |
| `Delete` | Clear all (same as a pinky hold) |
| `Esc` / `Q` | Quit |

### Camera settings window

![Settings window](docs/settings.png)

Press `X` to open it. Changes apply instantly and are saved to `settings.json` when the app closes.

- **Live stats**: the real capture resolution, camera FPS and app FPS. Green is 30 FPS or more, amber is 15 to 30, red is below 15. FPS is shown only here, not on the main screen.
- **Camera**: device number, resolution, target FPS (30 or 60), mirror view, auto exposure, and manual exposure, brightness, contrast and gain.
- **Gestures**: how close the fingers must be to count as a touch, the double-tap window for the space, how long the pinky hold takes, and a switch for cameras that report left and right the wrong way round.
- **Driver settings** (Windows): opens the camera maker's own settings dialog for anything not listed here.

### Troubleshooting

| Problem | Fix |
|---|---|
| "Camera unavailable" | Close other apps using the camera (Teams, Zoom, OBS), or pick another device number in the settings window. |
| FPS below 15 | Choose 640×480 and 60 FPS in settings, turn off auto exposure in a dark room (long exposure halves the frame rate), and plug the laptop in. |
| Touches trigger too easily or not at all | Lower or raise **Touch distance** in the settings window. |
| Right hand is ignored | Turn on **Swap left / right** in the settings window. |
| Model download fails | Download the file from the URL in `morse_hand/config.py` and save it as `models/hand_landmarker.task`. |

---

## Tech stack

| Part | Library | Why |
|---|---|---|
| Hand tracking | [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | Finds 21 points on each hand in real time on a normal CPU |
| Camera | [OpenCV](https://opencv.org/) | Reads the webcam and controls exposure, brightness and so on |
| Window and input | [pygame-ce](https://pyga.me/) | Window, keyboard, text rendering and sound playback |
| Rendering | [ModernGL](https://github.com/moderngl/moderngl) (OpenGL 3.3) | Draws the blurred glass panels and the glow on the graphics card |
| Maths | [NumPy](https://numpy.org/) | Landmark maths, smoothing filter and sound synthesis |
| Fonts | Inter and JetBrains Mono | Bundled in `assets/fonts` under the SIL Open Font License |

No sound files and no image assets are shipped: every sound is synthesised at start-up and every shape is drawn in code.

---

## How it works (non-technical overview)

Think of the app as a small assembly line with five stations.

1. **The camera takes pictures.** About 30 to 60 times a second, the webcam sends a new picture of you.

2. **The hand finder draws a skeleton.** A pre-trained model from Google looks at each picture and places 21 dots on your hand: the wrist, every knuckle and every fingertip. It also says which hand is the left one and which is the right one. The app only listens to the right hand.

3. **The touch detector watches the thumb.** For each finger, the app measures the gap between the tip of that finger and the tip of your thumb. When the gap becomes very small, it counts as a touch. The gap is compared with the size of your palm, so it works the same whether your hand is close to the camera or far from it. To avoid mistakes from a shaky picture, a touch must be seen in two pictures in a row before it counts, and the fingers have to open clearly before a new touch can start.

4. **The Morse translator builds letters.** Each touch is turned into an instruction: dot, dash, finish the letter, space, delete or clear. The dots and dashes are collected until you touch the ring finger, then they are looked up in the international Morse table, exactly like a telegraph operator would do with a code book. If the code does not exist, the app shows a red question mark instead of guessing.

5. **The screen shows the result.** The camera picture keeps its normal colours. The dots and dashes you type glow in the middle of the picture, each finger has its own colour, and a short sound confirms every action. The finished letter pops up in the middle and is added to the message at the bottom. Press Enter and the message moves up into a chat history, like a messaging app.

The app never records or uploads anything. Every picture is processed in memory and thrown away straight after.

---

## How it works (technical deep dive)

### Pipeline

```mermaid
flowchart LR
    A[Camera thread<br/>OpenCV, MJPG] -->|latest frame| B[Mirror + BGR to RGB]
    B --> C[MediaPipe Hand Landmarker<br/>VIDEO mode]
    C -->|21 x 3 landmarks<br/>right hand only| D[PinchDetector<br/>ratio + hysteresis]
    C --> E[One Euro filter<br/>display only]
    D -->|DOWN / HOLD / UP| F[MorseComposer<br/>state machine]
    F --> G[HUD<br/>pygame surfaces]
    E --> G
    G --> H[ModernGL compositor<br/>glass + bloom shader]
    B --> H
    I[Settings process] <-->|Pipe| J[Main loop]
```

### Threads and processes

- **Camera thread** (`camera.py`) blocks on `VideoCapture.read()` and publishes only the newest frame through a `threading.Condition`. The main loop waits on the condition, so it wakes up as soon as a frame arrives and never processes the same frame twice. `CAP_PROP_BUFFERSIZE = 1` and MJPG are requested so frames are fresh and 720p can still reach 30 to 60 FPS. On Windows the backends are tried in the order DirectShow, Media Foundation, default.
- **Main thread** runs detection, gestures, HUD drawing and the OpenGL composition.
- **Settings process** (`settings_window.py`). The main window owns an OpenGL context, and a second SDL window in the same process can steal or invalidate it. Running the settings UI in a `spawn` child process avoids that and keeps slider dragging from stalling the render loop. Messages are small tuples over a `multiprocessing.Pipe`: `("set", section, key, value)`, `("action", name)`, `("stats", {...})`, `("props", {...})`.

### Hand tracking and handedness

MediaPipe's Tasks API runs in `VIDEO` mode, which reuses the previous frame's hand position instead of running the palm detector every time. That is cheaper and steadier than `IMAGE` mode. Timestamps are forced to be strictly increasing because `detect_for_video` rejects duplicates.

MediaPipe labels handedness as if the image were a mirrored selfie. The frame is mirrored before detection when **Mirror view** is on, so the label is used as is. With mirroring off the label is flipped, and the **Swap left / right** option covers cameras whose driver already mirrors the image.

### Touch detection (`gestures.py`)

For every finger `f`:

```
ratio_f = |thumb_tip - tip_f| / |wrist - middle_mcp|
```

- Distance uses x and y in pixels plus half-weighted depth `z`. Monocular depth is noisy, but a small weight still separates fingers that overlap in 2D.
- Normalising by palm length makes the threshold independent of distance to the camera. A unit test checks that a half-size hand gives the same result.
- **Hysteresis**: a touch starts below `touch_ratio` (0.30) and ends above `touch_ratio + release_gap` (0.42).
- **Debounce**: `CONFIRM_FRAMES = 2` and `RELEASE_FRAMES = 2`, plus an 80 ms refractory period after a release.
- **Exclusivity**: only one finger can be active. When several are below the threshold, the smallest ratio wins, and it has to stay the winner for the confirmation frames.
- Events: `DOWN` fires immediately for dot, dash and ring so feedback feels instant. The pinky uses `UP` (delete, when shorter than the hold time) and `HOLD` (clear, fired once). This way a long hold never deletes a character before clearing.
- `closeness` (0 to 1 per finger) is exported for the visuals: fingertip size, glow and the thin tether line that shows which finger is about to fire.

### Morse state machine (`morse.py`)

`MorseComposer` knows nothing about cameras, so it can be tested with plain function calls.

| State | Input | Result |
|---|---|---|
| any | dot / dash | append to `code`, cancel pending space |
| `code` not empty | ring | decode, `LETTER` or `INVALID`, arm double tap |
| armed, within window | ring | `SPACE` (never leading, never doubled) |
| armed, window expired | ring | re-arm (acts as a first tap) |
| `code` longer than 6 | dot / dash | `INVALID` right away, since no code is that long |
| any | pinky tap | delete last symbol, otherwise last character |

The table follows ITU-R M.1677-1 for A to Z, 0 to 9 and common punctuation. `candidates(prefix)` powers the live chart and the prediction preview.

### Smoothing

Drawing uses a vectorised **One Euro filter** (Casiez et al., 2012) on all 63 landmark values: heavy smoothing when the hand is still, almost none when it moves fast. Gesture detection deliberately uses the raw landmarks so a quick tap is never delayed by the filter.

### Rendering (`compositor.py` and `hud.py`)

Each frame uploads three textures and draws one full-screen triangle strip with a single fragment shader:

1. **Camera texture**. The camera is drawn only inside `CAMERA_RECT` (480 × 360, vertically centred in the 480 × 800 window) with its original colours, no filter. A cover crop (like CSS `object-fit: cover`) keeps any camera aspect ratio undistorted inside that rectangle. Everything outside it is black. Hand landmarks are mapped into the same rectangle, so the skeleton lines up with the hand.
2. **Panels**. The HUD sends the chart and message box as rectangles in uniforms. For each pixel the shader computes a rounded-rectangle signed distance field, giving anti-aliased edges, a dark tint, a hairline border that is brighter at the top, and an optional accent colour used for flashes on commit, error or send.
3. **Glow layer**. An opaque black pygame surface where bright shapes are added with `BLEND_RGB_ADD`. The shader adds three mip levels (1.0, 2.6, 4.2) of it, which gives a bloom effect without extra framebuffers or blur passes.
4. **UI layer**. A transparent pygame surface with text, the hand skeleton, the Morse code in progress and the chat bubbles, blended last. It is cleared to a near-white colour with zero alpha so that anti-aliased edges of light text do not get dark fringes.

Chat bubbles are word-wrapped to 330 px, cached per message, and drawn inside a clip rectangle between the camera and the message box, so older messages simply scroll out of view at the top.

Text surfaces and the Morse chart (keyed by the current prefix) are cached, so a typical frame only renders the message line and a few animated shapes.

### Audio (`audio.py`)

Tones are built with NumPy: a sine wave plus a soft second harmonic and a short linear attack and release to prevent clicks. The dash lasts three times as long as the dot, like real Morse timing. If no audio device is found the app simply stays silent.

### Performance budget

Measured on the CPU side (camera texture upload, HUD drawing, layer upload): about 10 ms per frame. MediaPipe adds roughly 8 to 20 ms depending on the CPU. That leaves the app comfortably between 30 and 60 FPS on a typical laptop. VSync is switched off so the camera, not the monitor, sets the pace and no extra frame of latency is added between a touch and its feedback.

---

## Project structure

```
AI_computer_vision_morse_coding/
├── main.py                  entry point
├── requirements.txt
├── morse_hand/
│   ├── app.py               main loop, wires everything together
│   ├── config.py            constants, colours, saved settings
│   ├── camera.py            threaded webcam reader
│   ├── hand_tracker.py      MediaPipe wrapper, model download, handedness
│   ├── gestures.py          thumb-to-finger touch detector
│   ├── morse.py             Morse table and composer state machine
│   ├── filters.py           One Euro smoothing filter
│   ├── hud.py               everything drawn on top of the camera
│   ├── draw.py              anti-aliased shapes, glow sprites, easing
│   ├── compositor.py        ModernGL shader: camera area, panels and bloom
│   ├── audio.py             synthesised feedback sounds
│   └── settings_window.py   camera settings window (separate process)
├── assets/fonts/            Inter and JetBrains Mono (OFL)
├── models/                  hand model, downloaded on first run
├── docs/                    screenshots for this README
└── tests/                   unit tests for Morse logic and gestures
```

## Credits

- Hand tracking model: Google MediaPipe, Apache License 2.0.
- Fonts: Inter by Rasmus Andersson and JetBrains Mono by JetBrains, both under the SIL Open Font License 1.1 (licence files in `assets/fonts`).
