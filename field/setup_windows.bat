@echo off
REM ============================================================
REM  Watermelon - Setup de campo (PC Windows con cDAQ NI)
REM  Correr UNA vez, con internet (o con wheels offline en \wheels).
REM ============================================================
cd /d "%~dp0\.."
echo Creando entorno virtual...
python -m venv .venv || goto :err
call .venv\Scripts\activate
python -m pip install --upgrade pip
if exist "%~dp0wheels" (
  echo Instalando OFFLINE desde field\wheels ...
  pip install --no-index --find-links "%~dp0wheels" -r requirements.txt nidaqmx || goto :err
) else (
  echo Instalando desde internet ...
  pip install -r requirements.txt || goto :err
  pip install nidaqmx || goto :err
)
echo.
echo ============================================================
echo  OK. Ahora:
echo   1) Copia tu archivo  .streamlit\secrets.toml  (creds Supabase)
echo   2) Instala el driver NI-DAQmx (de ni.com) si no esta
echo   3) Corre:  field\ni_check.bat   (diagnostico hardware)
echo   4) Corre:  field\run_watermelon.bat   (la app)
echo ============================================================
pause
exit /b 0
:err
echo.
echo ERROR en el setup. Revisa que Python 3.11 este instalado y en PATH.
pause
exit /b 1
