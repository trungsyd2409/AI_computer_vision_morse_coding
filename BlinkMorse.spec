# PyInstaller spec for Blink Morse. Build with build_exe.bat.
# Set MORSE_CONSOLE=1 to keep a console window that shows errors.
import os
from PyInstaller.utils.hooks import collect_all

datas = [
    ("assets/fonts", "assets/fonts"),
    ("models/face_landmarker.task", "models"),
]
binaries = []
hiddenimports = ["pygrabber", "pygrabber.dshow_graph", "comtypes"]

# These packages ship native libraries or data files that PyInstaller does
# not detect by itself. mediapipe imports matplotlib at start-up, so it must
# be bundled too.
for pkg in ("mediapipe", "matplotlib", "moderngl", "glcontext"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["pytest", "IPython", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BlinkMorse",
    debug=False,
    console=os.environ.get("MORSE_CONSOLE") == "1",
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name="BlinkMorse",
)
