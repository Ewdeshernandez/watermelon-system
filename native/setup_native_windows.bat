@echo off
REM  Instala las dependencias del modulo NATIVO en el venv del proyecto.
cd /d "%~dp0\.."
call .venv\Scripts\activate
pip install -r native\requirements.txt
echo.
echo OK. Corre:  native\run_field.bat  (o run_field_sim.bat para demo)
pause
