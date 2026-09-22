@echo off
REM ============================================================
REM Watermelon System1 Agent — ejecucion horaria (Task Scheduler)
REM 1) RPA exporta las ondas a CSV (reemplaza a la persona)
REM 2) El agente convierte esa carpeta y sube a la nube
REM Carpeta destino: C:\Watermelon\system1_agent\
REM ============================================================
setlocal
cd /d "%~dp0"
set PY=python

"%PY%" s1_rpa_export.py --run >> logs\run.log 2>&1
"%PY%" s1_agent.py --csv --once >> logs\run.log 2>&1
exit /b %ERRORLEVEL%
