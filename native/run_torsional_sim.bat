@echo off
REM  Watermelon Torsional NATIVO — modo DEMO (sin hardware) para ver como se ve.
REM  Telemetria de par Binsfeld TorqueTrak 10K -> NI 9229 (voltaje DC).
cd /d "%~dp0\.."
call .venv\Scripts\activate
python native\watermelon_torsional.py --sim
pause
