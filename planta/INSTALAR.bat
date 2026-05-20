@echo off
REM ============================================================
REM  Watermelon Planta Edition - INSTALADOR
REM  Doble click para instalar dependencias Python necesarias
REM ============================================================
title Watermelon Planta - Instalador

echo.
echo ============================================================
echo   WATERMELON PLANTA EDITION - Instalador
echo ============================================================
echo.
echo Este script va a instalar las dependencias Python necesarias
echo para correr Watermelon Planta. Tarda 1-2 minutos.
echo.
echo Pre-requisitos:
echo   - Python 3.10+ instalado y en el PATH
echo   - NI-DAQmx driver instalado (NI MAX debe verse en Start)
echo.
pause

REM Cambiar al directorio del script (importante para rutas relativas)
cd /d "%~dp0"

echo.
echo [1/3] Verificando Python...
python --version
if errorlevel 1 (
    echo.
    echo ERROR: Python no esta instalado o no esta en el PATH.
    echo Instala Python 3.12 de https://www.python.org/downloads/
    echo Asegurate de marcar "Add python.exe to PATH" en el instalador.
    pause
    exit /b 1
)

echo.
echo [2/3] Verificando pip...
python -m pip --version
if errorlevel 1 (
    echo ERROR: pip no esta disponible.
    pause
    exit /b 1
)

echo.
echo [3/3] Instalando dependencias desde requirements-planta.txt...
echo Si hay internet, descarga desde PyPI. Si NO hay, usa wheels/ local.
echo.

REM Primero intentar offline desde ../wheels (si fue copiado en USB)
if exist "..\wheels" (
    echo Encontre carpeta wheels/ - intentando instalacion OFFLINE...
    python -m pip install --no-index --find-links "..\wheels" -r requirements-planta.txt
    if not errorlevel 1 goto :installed_ok
    echo Instalacion offline fallo. Intentando ONLINE...
)

REM Fallback: instalacion online desde PyPI
python -m pip install -r requirements-planta.txt
if errorlevel 1 (
    echo.
    echo ERROR: Instalacion fallo. Verifica internet o wheels/.
    pause
    exit /b 1
)

:installed_ok
echo.
echo ============================================================
echo   INSTALACION COMPLETA
echo ============================================================
echo.
echo Verificando que las librerias cargan correctamente...
python -c "import streamlit; print('  streamlit:', streamlit.__version__)"
python -c "import nidaqmx; print('  nidaqmx: ', nidaqmx.__version__)"
python -c "import nptdms; print('  npTDMS:  ', nptdms.__version__)"
python -c "import numpy; print('  numpy:   ', numpy.__version__)"
python -c "import pandas; print('  pandas:  ', pandas.__version__)"
python -c "import plotly; print('  plotly:  ', plotly.__version__)"

echo.
echo Listo! Ahora corre INICIAR.bat para abrir Watermelon Planta.
echo.
pause
