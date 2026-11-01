@echo off
rem Starts Lucy's own API server (which auto-starts her dedicated 27B donor host via
rem server.py's ensure_server() -- separate infra, not Luna's F:\my-agent-vampire) in its own
rem window, then opens an interactive chat with her. Every turn is donor-assisted; after each
rem reply you're asked whether to teach it into her own weights (see talk_to_lucy.py).
rem
rem For BULK, scripted teaching off the host LLM instead of a live conversation, run
rem amelia_teaches_lucy.py directly (e.g. "python -B amelia_teaches_lucy.py 100 50").
cd /d "%~dp0"
set "PY="
for /d %%U in ("C:\Users\*") do (
    if not defined PY if exist "%%U\.conda\envs\galatea\python.exe" set "PY=%%U\.conda\envs\galatea\python.exe"
)
for /d %%U in ("C:\Users\*") do (
    if not defined PY if exist "%%U\miniconda3\python.exe" set "PY=%%U\miniconda3\python.exe"
)
if not defined PY set "PY=python"

echo Starting Lucy's server (this also starts her donor host if not already running)...
start "Lucy BrickCluster Server" /min "%PY%" -B brickcluster_server.py --host 127.0.0.1 --port 8105

echo Waiting for her API to come up (first load can take a few minutes)...
:waitloop
"%PY%" -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8105/health', timeout=2).status==200 else 1)" >nul 2>nul
if errorlevel 1 (
    timeout /t 3 >nul
    goto waitloop
)

"%PY%" -B talk_to_lucy.py 8105
pause
