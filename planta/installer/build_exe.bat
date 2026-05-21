@echo off
REM ==============================================================
REM  Watermelon Planta - Build .exe con PyInstaller
REM  Doble click para generar dist\WatermelonPlanta.exe
REM ==============================================================
title Watermelon Planta - Build EXE

cd /d "%~dp0"

echo.
echo ============================================================
echo   WATERMELON PLANTA EDITION - Build EXE
echo ============================================================
echo.
echo Este script empaqueta Watermelon Planta como un .exe
echo single-file que el cliente puede ejecutar sin Python instalado.
echo.
echo Tarda 5-10 minutos. Tama#o final esperado: ~250 MB.
echo.
pause

echo.
echo [1/4] Verificando PyInstaller...
python -m PyInstaller --version
if errorlevel 1 (
    echo.
    echo PyInstaller no esta instalado. Instalando...
    python -m pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: pip install pyinstaller fallo.
        pause
        exit /b 1
    )
)

echo.
echo [2/4] Limpiando builds anteriores...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__

echo.
echo [3/4] Empaquetando con PyInstaller (esto tarda 5-10 min)...
python -m PyInstaller watermelon-planta.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller fallo. Revisa los mensajes arriba.
    pause
    exit /b 1
)

echo.
echo [4/4] Verificando el .exe generado...
if not exist "dist\WatermelonPlanta.exe" (
    echo ERROR: dist\WatermelonPlanta.exe no se genero.
    pause
    exit /b 1
)

REM Mostrar tamano
for %%I in ("dist\WatermelonPlanta.exe") do echo   Tamano: %%~zI bytes (~%%~zI / 1024 / 1024 MB)

echo.
echo ============================================================
echo   BUILD COMPLETO
echo ============================================================
echo.
echo El .exe esta en:
echo   %CD%\dist\WatermelonPlanta.exe
echo.
echo Probarlo: doble click en ese .exe — debe abrir tu browser
echo en localhost:8501 con la app de Watermelon Planta.
echo.
echo Siguiente paso (opcional): compilar el INSTALADOR profesional
echo con Inno Setup. Ejecuta:
echo   build_installer.bat  (requiere Inno Setup 6 instalado)
echo.
pause
