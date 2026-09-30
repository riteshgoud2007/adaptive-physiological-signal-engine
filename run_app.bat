@echo off
title Adaptive ECG Signal Intelligence Engine
echo.
echo ============================================================
echo   Adaptive ECG Signal Intelligence Engine
echo ============================================================
echo.
echo Starting Streamlit application...
echo Open http://localhost:8501 in your browser.
echo.

SET PYTHON314=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe

IF EXIST "%PYTHON314%" (
    echo Using Python 3.14...
    "%PYTHON314%" -m streamlit run app.py --server.headless false --server.port 8501
) ELSE (
    echo Using default Python...
    python -m streamlit run app.py --server.headless false --server.port 8501
)
pause
