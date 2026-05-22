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
echo   - Drivers de adquisicion Watermelon instalados
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
echo Verificando integridad del sistema...
python -c "import streamlit" 2>nul && echo   [OK] Interfaz grafica       || echo   [FAIL] Interfaz grafica
python -c "import nidaqmx"  2>nul && echo   [OK] Adquisicion de datos  || echo   [FAIL] Adquisicion de datos
python -c "import nptdms"   2>nul && echo   [OK] Formato de captura    || echo   [FAIL] Formato de captura
python -c "import numpy"    2>nul && echo   [OK] Procesamiento numerico|| echo   [FAIL] Procesamiento numerico
python -c "import pandas"   2>nul && echo   [OK] Manejo de datos       || echo   [FAIL] Manejo de datos
python -c "import plotly"   2>nul && echo   [OK] Visualizacion         || echo   [FAIL] Visualizacion
python -c "import jwt"      2>nul && echo   [OK] Sistema de licencias  || echo   [FAIL] Sistema de licencias

echo.
echo Listo! Ahora corre INICIAR.bat para abrir Watermelon Planta.
echo.
pause
