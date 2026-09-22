@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0upload_csv.ps1" >> logs\uploader.log 2>&1
exit /b %ERRORLEVEL%
