@echo off
rem Religious Text Matcher. Opens http://localhost:8520 in your browser.
rem First search after launch takes ~30 s while the models load.
rem Phone: on the same home Wi-Fi, open the "Network URL" printed below.
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
    echo No .venv found. Set it up once with:
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
title Religious Text Matcher (close this window to stop)
".venv\Scripts\streamlit.exe" run "app\app.py" --server.port 8520 --browser.gatherUsageStats false
pause
