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

APP_TITLE = "Arm Morse"
WINDOW_SIZE = (800, 600)          # 4:3, matches the native ratio of most webcams
SETTINGS_TITLE = "Settings"
SETTINGS_SIZE = (400, 600)

# ---------------------------------------------------------------------------
# Visual palette (RGB tuples, 0-255)
# ---------------------------------------------------------------------------
# Kept deliberately small: off-white for text and bones, one warm accent for
# "this is live", and red for errors. No neon, no glow.

TEXT = (240, 240, 236)
TEXT_MUTED = (170, 170, 164)
TEXT_FAINT = (120, 120, 116)
ACCENT = (255, 196, 64)           # amber: armed hand, newest symbol, matches
SUCCESS = (120, 220, 140)
BONE = (240, 240, 236)
WARNING = (255, 196, 64)
DANGER = (240, 96, 88)
PANEL = (14, 14, 14)              # flat backdrop behind text, used with alpha

# Used by the settings window to label each gesture.
ACTION_COLORS = {
    "dot": (240, 240, 236),
    "dash": (240, 240, 236),
    "letter": (255, 196, 64),
    "word": (255, 196, 64),
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
    """
    Forearm angles that make up the gestures.

    The angle is measured on the camera image, along the forearm (elbow to
    wrist), against the horizontal: 0 = level, +45 = hand raised
    diagonally, -45 = hand lowered diagonally.

    Symbol arm (left by default):   +45 = dash, -45 = dot
    Control arm (right by default): hold a fist to type at all;
                                    +45 = end of letter, -45 = space
    Each arm has to come back to level before its next gesture counts.
    """

    angle_target: float = 45.0        # degrees, the diagonal to aim for
    angle_tolerance: float = 20.0     # +/- around the target that still counts
    neutral_band: float = 15.0        # |angle| below this counts as level
    hold_time: float = 0.10           # seconds a pose must be held to count
    swap_arms: bool = False           # fixes cameras that mirror the image
    # Which arm types dots and dashes. The other one holds the fist and
    # ends letters / adds spaces. S swaps the two roles.
    symbol_side: str = "left"

    @property
    def control_side(self) -> str:
        return "right" if self.symbol_side == "left" else "left"


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
