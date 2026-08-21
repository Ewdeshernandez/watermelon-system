@echo off
REM  Diagnostico del hardware NI (seguro, sin IEPE)
cd /d "%~dp0\.."
call .venv\Scripts\activate
python tools\ni_check.py
pause
