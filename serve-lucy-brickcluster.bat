@echo off
cd /d "%~dp0"
set "PY=C:\Users\Azazel\.conda\envs\galatea\python.exe"
if not exist "%PY%" set "PY=C:\Users\Azazel\miniconda3\python.exe"
"%PY%" -B brickcluster_server.py --host 127.0.0.1 --port 8105
pause
