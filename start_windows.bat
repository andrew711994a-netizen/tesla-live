@echo off
cd /d "%~dp0"
set PY=python
where py >nul 2>nul
if %errorlevel%==0 set PY=py -3

if not exist ".venv\installed.ok" (
    echo First run: installing the app, this takes 1-3 minutes...
    %PY% -m venv .venv
    if errorlevel 1 goto nopython
    ".venv\Scripts\python" -m pip install --upgrade pip
    ".venv\Scripts\python" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
    echo ok> ".venv\installed.ok"
)
echo Starting... the app opens in your browser. Close this window to stop it.
".venv\Scripts\python" -m streamlit run app.py
pause
exit /b

:nopython
echo Python was not found. Install it from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during installation.
pause
exit /b

:failed
echo Installation failed. Check your internet connection and run this file again.
pause
