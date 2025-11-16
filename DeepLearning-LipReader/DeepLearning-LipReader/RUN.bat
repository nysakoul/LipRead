@echo off
echo ============================================================
echo Starting Whisper-based Lip Reader
echo ============================================================
echo.
echo This uses Whisper speech-to-text (not actual lip reading)
echo Press 'L' to record and transcribe, 'Q' to quit
echo.
cd /d "%~dp0"
python src\predict.py
pause



