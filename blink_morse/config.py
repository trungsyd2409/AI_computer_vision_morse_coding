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
# Hand model used to read the open hand / fist that switches typing on.
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
# Body pose model used to measure the elbow angle (about 5.5 MB).
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

APP_TITLE = "Curl Morse"
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
BONE = (226, 232, 240)            # arm skeleton when the arm is extended
WARNING = (251, 191, 36)
DANGER = (251, 113, 133)

# One colour per action so the legend, the meters, the typed symbols and
# the pause countdown all speak the same visual language.
ACTION_COLORS = {
    "dot": (103, 232, 249),       # short curl
    "dash": (167, 139, 250),      # long curl
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
class ArmSettings:
    """Threshold and timing for reading dumbbell curls."""

    # Curl is a 0..1 score from the elbow angle: 0 = arm straight (about
    # 170 deg), 1 = fully curled (about 45 deg). Above `curl_threshold` the
    # arm counts as curled ("pressed"). Adjustable from 5% to 95%.
    curl_threshold: float = 0.50
    # The Morse time unit t, in seconds. Everything is derived from it:
    # curled < 1.5 t = dot, >= 1.5 t = dash, arm extended 5 t = end of
    # letter, extended 7 t = end of word. A real curl takes about half a
    # second, so t defaults much longer than it did for blinks.
    time_unit: float = 0.50
    swap_arms: bool = False           # fixes cameras that mirror the image
    # Which arm curls to type. The other hand is the switch: fist = typing
    # on, open hand = typing off. S swaps the two roles.
    curl_side: str = "right"

    @property
    def gate_side(self) -> str:
        return "left" if self.curl_side == "right" else "right"

    @property
    def dash_after(self) -> float:
        return 1.5 * self.time_unit

    @property
    def letter_gap(self) -> float:
        return 5.0 * self.time_unit

    @property
    def word_gap(self) -> float:
        return 7.0 * self.time_unit


@dataclass
class AppSettings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    arm: ArmSettings = field(default_factory=ArmSettings)

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
        for name in ("camera", "arm"):
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
