@echo off
REM  Lanza la app Watermelon local (unica forma de ver el cDAQ USB)
cd /d "%~dp0\.."
call .venv\Scripts\activate
echo Abriendo Watermelon en http://localhost:8501 ...
streamlit run app.py
pause
