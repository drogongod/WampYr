@echo off
rem FIXED 2026-10-29: made account-agnostic (wildcard) so this works the same way on any
rem machine, same real fix as the public copy -- see PublicLucyWEST's version of this file for
rem the full note on why `for /d` (not a plain file-matching wildcard) is required here.
cd /d "%~dp0"
set "PY="
for /d %%U in ("C:\Users\*") do (
    if not defined PY if exist "%%U\.conda\envs\galatea\python.exe" set "PY=%%U\.conda\envs\galatea\python.exe"
)
for /d %%U in ("C:\Users\*") do (
    if not defined PY if exist "%%U\miniconda3\python.exe" set "PY=%%U\miniconda3\python.exe"
)
if not defined PY set "PY=python"
"%PY%" -B brickcluster_server.py --host 127.0.0.1 --port 8105
pause
