@echo off
rem Builds dist\MorseApp\MorseApp.exe so it runs without Python installed.
rem Usage:  build_exe.bat          normal build (no console window)
rem         build_exe.bat debug    keeps a console window to show errors
cd /d "%~dp0"

if /i "%1"=="debug" set MORSE_CONSOLE=1

if not exist ".venv\Scripts\python.exe" (
    echo Cannot find .venv. Create it first.
    pause
    exit /b 1
)
if not exist "models\face_landmarker.task" (
    echo Missing models\face_landmarker.task - run the app once with Python so it downloads the model.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pyinstaller
if errorlevel 1 goto :fail

".venv\Scripts\python.exe" -m PyInstaller MorseApp.spec --noconfirm --clean
if errorlevel 1 goto :fail

echo.
echo Done. Share the whole folder: dist\MorseApp
echo Run:  dist\MorseApp\MorseApp.exe
pause
exit /b 0

:fail
echo Build failed. See the messages above.
pause
exit /b 1
