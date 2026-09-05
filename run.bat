@echo off
title Portrait Video Enhancer
cd /d "%~dp0"

:: Temporary directory on Drive H:
if not exist "%~dp0.temp_work" mkdir "%~dp0.temp_work"
set TEMP=%~dp0.temp_work
set TMP=%~dp0.temp_work
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ========================================================
echo        Portrait Video Enhancer (RTX 3050 6GB)
echo ========================================================
echo Starting local server at http://127.0.0.1:7860 ...
echo Memulai server aplikasi di http://127.0.0.1:7860 ...
echo Browser will open automatically / Browser akan terbuka otomatis.
echo Press Ctrl+C to stop / Tekan Ctrl+C untuk berhenti.
echo.

python app.py

if %errorlevel% neq 0 (
    echo.
    echo [WARNING / PERINGATAN] Application stopped with exit code: %errorlevel%
    pause
)
