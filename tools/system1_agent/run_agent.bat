@echo off
REM ============================================================
REM Watermelon System1 Agent — ejecucion horaria (Task Scheduler)
REM Coloca esta carpeta en:  C:\Watermelon\system1_agent\
REM ============================================================
setlocal
cd /d "%~dp0"

REM Ajusta la ruta a tu python si no esta en PATH:
set PY=python

"%PY%" s1_agent.py --once >> logs\run.log 2>&1
exit /b %ERRORLEVEL%
