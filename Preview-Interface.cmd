@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 -- --interface-preview
) else if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 -- --interface-preview
) else (
  echo Please install Streamlit, Pandas and Plotly first. See README.md.
  pause
)
