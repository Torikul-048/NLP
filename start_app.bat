@echo off
cd /d "%~dp0"
if exist "..\Scripts\python.exe" (
  "..\Scripts\python.exe" -m streamlit run app.py
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m streamlit run app.py
) else (
  python -m streamlit run app.py
)
pause
