@echo off
REM  Modulo nativo en modo DEMO (sin hardware) para ver como se ve.
cd /d "%~dp0\.."
call .venv\Scripts\activate
python native\watermelon_field.py --sim
pause
