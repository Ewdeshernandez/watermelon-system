@echo off
REM  Diagnostico con ACELEROMETROS (IEPE ON). Poné la sensib de tu acelerometro
REM  con --sens (ej. 100 mV/g). Conecta el/los acelerometro(s) en el Mod2.
cd /d "%~dp0\.."
call .venv\Scripts\activate
python tools\ni_check.py --iepe --sens 100
pause
