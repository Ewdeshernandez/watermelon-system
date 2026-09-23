@echo off
REM ============================================================
REM Watermelon System1 Agent — corrida MANUAL de una sola maquina
REM ============================================================
REM OJO: la PRODUCCION NO usa este .bat. El cron real esta partido
REM en 2 maquinas (la VM no tiene internet; el host si):
REM   VM   -> tarea WM_export -> C:\WM_wave\export.bat
REM           = s1_rpa_export.py --expall C:\WM_wave\csv
REM           (activa la hoja "Export csv1" y trae System1 al frente
REM            aunque haya otra hoja/ventana encima)
REM   HOST -> tarea WM_upload -> C:\Watermelon\upload.bat
REM           = s1_agent.py --csv --once  (lee el share de la VM y sube)
REM
REM Este .bat solo sirve si UNA sola maquina tiene System1 + internet
REM (demo / prueba local). Usa --expall (parrilla 4x2), NO el viejo --run.
REM ============================================================
setlocal
cd /d "%~dp0"
set PY=python

"%PY%" s1_rpa_export.py --expall "%~dp0csv" >> logs\run.log 2>&1
"%PY%" s1_agent.py --csv --once >> logs\run.log 2>&1
exit /b %ERRORLEVEL%
