@echo off
REM  Modulo nativo de campo (NI real). Edita los parametros si hace falta.
cd /d "%~dp0\.."
call .venv\Scripts\activate
python native\watermelon_field.py --sens 100 --fs 5120 --chans 0,1 --names 1YA,1XA
pause
