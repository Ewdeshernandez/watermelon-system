@echo off
REM ==============================================================
REM  Watermelon Planta - Build Installer Profesional con Inno Setup
REM  Doble click para generar WatermelonPlantaSetup-v1.0.exe
REM ==============================================================
title Watermelon Planta - Build Installer

cd /d "%~dp0"

echo.
echo ============================================================
echo   WATERMELON PLANTA - Build Installer Profesional
echo ============================================================
echo.
echo Este script empaqueta WatermelonPlanta.exe + README + assets
echo en un instalador profesional WatermelonPlantaSetup-v1.0.exe
echo.
echo Pre-requisito: tener Inno Setup 6 instalado
echo (descarga free de https://jrsoftware.org/isdl.php)
echo.
pause

REM Detectar Inno Setup
set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
    echo.
    echo ERROR: No encontre Inno Setup en las rutas estandar.
    echo Instala desde https://jrsoftware.org/isdl.php
    echo o edita este .bat para apuntar a tu ruta de ISCC.exe
    pause
    exit /b 1
)

REM Verificar que la carpeta onedir exista (v3.31.436)
if not exist "dist\WatermelonPlanta\WatermelonPlanta.exe" (
    echo.
    echo ERROR: No existe dist\WatermelonPlanta\WatermelonPlanta.exe
    echo Corre primero build_exe.bat para generarlo.
    pause
    exit /b 1
)

REM ==============================================================
REM  v3.31.340 — GUARD paquete COMPLETO para venta. Si el driver del
REM  equipo NO esta empacado, el instalador saldria sin controlador y el
REM  cliente en una maquina NUEVA no podria capturar. No debe pasar en
REM  silencio para un paquete turnkey.
REM ==============================================================
if not exist "dependencies\driver-extracted\Install.exe" (
    echo.
    echo ============================================================
    echo   ATENCION - DRIVER DEL EQUIPO NO EMPACADO
    echo ============================================================
    echo.
    echo No encontre: dependencies\driver-extracted\Install.exe
    echo El instalador saldria SIN el controlador del equipo. En una
    echo maquina NUEVA el cliente NO podria capturar.
    echo.
    echo Para un paquete de venta COMPLETO coloca el runtime del driver en
    echo   dependencies\driver-extracted\   (ver dependencies\README_DRIVER.txt)
    echo.
    set /p _CONT="Continuar igual (solo build interno/pruebas)? [s/N]: "
    if /I not "%_CONT%"=="s" (
        echo Build cancelado. Empaca el driver y reintenta.
        pause
        exit /b 1
    )
)

echo.
echo Inno Setup encontrado: %ISCC%
echo Build empezando...
echo.

"%ISCC%" installer.iss
if errorlevel 1 (
    echo.
    echo ERROR: Inno Setup fallo. Revisa los mensajes arriba.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   INSTALLER COMPLETO
echo ============================================================
echo.
echo El installer profesional esta en:
dir /b dist\WatermelonPlantaSetup*.exe
echo.
echo Este archivo lo puedes mandar a cualquier cliente.
echo Doble click → wizard de instalacion profesional.
echo.
pause
