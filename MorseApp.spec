# PyInstaller spec for the Morse webcam app (main.py + blink_morse package).
# Works on every branch: it bundles all models/*.task files it finds.
# Build with build_exe.bat. Set MORSE_CONSOLE=1 to keep a console window.
import glob
import os
from PyInstaller.utils.hooks import collect_all

root = SPECPATH

datas = [(os.path.join(root, "assets", "fonts"), "assets/fonts")]
for model in glob.glob(os.path.join(root, "models", "*.task")):
    datas.append((model, "models"))

binaries = []
hiddenimports = ["pygrabber", "pygrabber.dshow_graph", "comtypes"]

# These packages ship native libraries or data files that PyInstaller does
# not detect by itself. mediapipe imports matplotlib at start-up.
for pkg in ("mediapipe", "matplotlib", "moderngl", "glcontext"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [os.path.join(root, "main.py")],
    pathex=[root],
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
    name="MorseApp",
    debug=False,
    console=os.environ.get("MORSE_CONSOLE") == "1",
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name="MorseApp",
)
