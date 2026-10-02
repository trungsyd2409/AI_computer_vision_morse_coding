"""
Central place for every tunable value in the app.

Keeping constants here means the rest of the code never hard-codes a colour,
a threshold or a file path, and the settings window can change the runtime
values without touching the modules that consume them.
"""

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
MODELS_DIR = ROOT_DIR / "models"

SETTINGS_FILE = ROOT_DIR / "settings.json"   # created on first exit
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

APP_TITLE = "Morse Hand"
WINDOW_SIZE = (480, 800)          # portrait, 3:5
# Where the camera image goes inside the window: full width, vertically
# centred, 4:3 like most webcams. The strips above and below stay black and
# hold the Morse chart and the chat.
CAMERA_RECT = (0, 220, 480, 360)
SETTINGS_TITLE = "Camera Settings"
SETTINGS_SIZE = (400, 720)

# ---------------------------------------------------------------------------
# Visual palette (RGB tuples, 0-255)
# ---------------------------------------------------------------------------

TEXT = (244, 246, 250)
TEXT_MUTED = (150, 160, 178)
TEXT_FAINT = (98, 108, 126)
ACCENT = (103, 232, 249)          # cyan, used for the primary highlight
SUCCESS = (74, 222, 128)
WARNING = (251, 191, 36)
DANGER = (251, 113, 133)

# One colour per action so the line between the fingertips, the typed
# symbols and the feedback bursts all speak the same visual language.
FINGER_COLORS = {
    "index": (103, 232, 249),     # dot
    "middle": (167, 139, 250),    # dash
    "ring": (251, 191, 36),       # end of letter / space
    "pinky": (251, 113, 133),     # clear
}

# ---------------------------------------------------------------------------
# Runtime settings (editable from the settings window)
# ---------------------------------------------------------------------------

RESOLUTION_CHOICES = [(640, 480), (1280, 720), (1920, 1080)]
FPS_CHOICES = [30, 60]
MIN_ACCEPTABLE_FPS = 15


@dataclass
class CameraSettings:
    """Everything the user can change about the capture device."""

    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 60
    mirror: bool = True
    auto_exposure: bool = True
    # Image controls. None means "keep whatever the driver uses", so a fresh
    # install never overrides a camera that is already tuned. Exposure on
    # DirectShow is in log2 seconds, e.g. -6 = 1/64 s.
    exposure: Optional[float] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    gain: Optional[float] = None


@dataclass
class GestureSettings:
    """When the index and middle fingertips count as touching, and timing."""

    # Distance between the index and middle fingertips, divided by the palm
    # length. Below `touch_ratio` the press starts, above
    # `touch_ratio + release_gap` it ends. The gap (hysteresis) stops the
    # state from flickering.
    touch_ratio: float = 0.25
    release_gap: float = 0.10
    # The Morse time unit t, in seconds. Fingers together < 1.5 t = dot,
    # >= 1.5 t = dash; apart 3 t = end of letter, apart 7 t = space.
    time_unit: float = 0.20
    swap_hands: bool = False          # fixes cameras that report the wrong side


@dataclass
class AppSettings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    gesture: GestureSettings = field(default_factory=GestureSettings)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path = SETTINGS_FILE) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> "AppSettings":
        """Read saved settings, ignoring unknown keys and broken files."""
        settings = cls()
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return settings
        for section in (settings.camera, settings.gesture):
            name = "camera" if section is settings.camera else "gesture"
            saved = data.get(name, {})
            for f in fields(section):
                if f.name not in saved:
                    continue
                default = getattr(section, f.name)
                value = saved[f.name]
                # Optional fields default to None, so there is no type to cast to.
                if default is not None and value is not None:
                    value = type(default)(value)
                setattr(section, f.name, value)
        return settings
