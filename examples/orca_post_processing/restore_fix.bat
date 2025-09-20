@echo off
setlocal
set "F=%~1"
set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" set "PY=C:\Windows\py.exe"

"%PY%" "%~dp0restore_pos_fix.py" "%F%" >"%F%.log" 2>&1
type "%F%.log"
exit /b 0
