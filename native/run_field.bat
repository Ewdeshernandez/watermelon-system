@echo off
REM  Watermelon Field NATIVO — campo (NI real).
REM  Tu setup: 2 acelerometros 100 mV/g en Mod1 ai0,ai1. Ajusta si cambia.
cd /d "%~dp0\.."
call .venv\Scripts\activate
python native\watermelon_field.py --chans 0,1 --names 1YA,1XA --sens 100 --fs 5120 --alarm 1.0 --danger 2.0
pause
