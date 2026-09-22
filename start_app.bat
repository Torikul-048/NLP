@echo off
cd /d "%~dp0"

set "PY_EXE="
if exist "..\Scripts\python.exe" (
  set "PY_EXE=..\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
  set "PY_EXE=.venv\Scripts\python.exe"
) else (
  set "PY_EXE=python"
)

echo ====================================================================
echo  KUET Academic Information Retrieval System
echo ====================================================================
echo [Step 1/2] Generating evaluation plots and confusion matrices into plotting/...
"%PY_EXE%" -m scripts.generate_plots
if %ERRORLEVEL% neq 0 (
  echo [Notice] Plot generation completed with warnings or missing assets. Continuing...
)

echo.
echo [Step 2/2] Launching Streamlit web application...
"%PY_EXE%" -m streamlit run app.py
pause
