@echo off
cd /d "%~dp0"
echo Starting mp4 transcription tool...
echo (First launch may take 10-30 seconds to load libraries)
echo.
python main.py
echo.
echo Done. Press any key to close this window.
pause >nul
