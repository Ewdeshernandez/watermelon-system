@echo off
REM ============================================================
REM  Watermelon Planta Edition - LAUNCHER
REM  Doble click para abrir la app
REM ============================================================
title Watermelon Planta - App

REM Cambiar al directorio del script
cd /d "%~dp0"

echo.
echo ============================================================
echo   WATERMELON PLANTA EDITION
echo ============================================================
echo.
echo Arrancando la app en localhost...
echo Tu browser default se abrira automaticamente en unos segundos.
echo.
echo Para CERRAR la app, vuelve a esta ventana y presiona Ctrl+C
echo o cierra esta ventana directamente.
echo.

REM Verificar que streamlit este instalado
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Streamlit no instalado. Corre INSTALAR.bat primero.
    pause
    exit /b 1
)

REM Arrancar streamlit con configuracion local
REM --server.headless=false  -> abre browser automaticamente
REM --server.port=8501       -> puerto fijo
REM --server.address=127.0.0.1 -> solo localhost (no expone a la red)
REM --browser.gatherUsageStats=false -> sin telemetria
python -m streamlit run app_planta.py ^
    --server.headless=false ^
    --server.port=8501 ^
    --server.address=127.0.0.1 ^
    --browser.gatherUsageStats=false ^
    --theme.primaryColor="#0f766e" ^
    --theme.backgroundColor="#ffffff"

REM Si streamlit cierra (Ctrl+C o cerrar browser), llegamos aca
echo.
echo App cerrada.
pause
