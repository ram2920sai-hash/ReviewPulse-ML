@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Open Start-ReviewPulse.cmd to launch the app.
pause
exit /b 0
:failed
echo Setup did not finish. Check your Python installation and internet connection.
pause
exit /b 1
