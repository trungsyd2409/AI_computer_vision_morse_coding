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
FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

APP_TITLE = "Blink Morse"
WINDOW_SIZE = (800, 600)          # 4:3, matches the native ratio of most webcams
SETTINGS_TITLE = "Settings"
SETTINGS_SIZE = (400, 600)

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

# One colour per action so the legend, the meters, the typed symbols and
# the pause countdown all speak the same visual language.
ACTION_COLORS = {
    "dot": (103, 232, 249),       # right wink
    "dash": (167, 139, 250),      # both eyes
    "letter": (251, 191, 36),     # short pause -> end of letter
    "word": (74, 222, 128),       # long pause  -> space
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
class EyeSettings:
    """Thresholds and timings for reading blinks and winks."""

    # Eye closure is a 0..1 score after removing the user's own resting
    # level. Above `close_threshold` an eye counts as closed.
    close_threshold: float = 0.45
    # The right eye must be closed on its own this long to type a dot. It
    # only has to be long enough to rule out one eye leading a normal blink.
    wink_confirm: float = 0.05
    # Both eyes must stay closed this long to type a dash. 0 = instantly.
    blink_filter: float = 0.0
    # Pause with the eyes open that ends a letter, and that ends a word.
    letter_pause: float = 0.5
    word_pause: float = 2.0
    swap_eyes: bool = False           # fixes cameras that mirror the image


@dataclass
class AppSettings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    eyes: EyeSettings = field(default_factory=EyeSettings)

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
        for name in ("camera", "eyes"):
            section = getattr(settings, name)
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
