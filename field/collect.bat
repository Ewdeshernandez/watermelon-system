@echo off
REM  COLECTOR de campo (headless). Graba directo, sin navegador -> no se traba.
REM  Por defecto: 2 acelerometros 100 mV/g en Mod1 ai0,ai1, 5120 Hz.
REM  Para cambiar: edita esta linea o pasa argumentos. Ctrl+C para terminar.
cd /d "%~dp0\.."
call .venv\Scripts\activate
python field\collect.py %*
pause
