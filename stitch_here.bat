@echo off
rem Stitch every tile set in this folder (or the folder given as %1).
rem Keep stitch_tool.py next to this file. Requires C:\miniconda3\python.exe.
setlocal
set "PY=C:\miniconda3\python.exe"
if not exist "%PY%" set "PY=python"
set "DIR=%~dp0"
if "%DIR:~-1%"=="\" if not "%DIR:~-2%"==":\" set "DIR=%DIR:~0,-1%"
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=%DIR%"
"%PY%" "%DIR%\stitch_tool.py" "%TARGET%"
if errorlevel 1 pause
endlocal
