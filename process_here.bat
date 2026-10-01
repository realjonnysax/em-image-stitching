@echo off
rem All-in-one: denoise then stitch every tile set in this folder (or %1).
rem Output: <target>\denoised\*.tif  and  <target>\denoised\stitched\<set>_ashlar_v7.ome.tif
setlocal
set "PY=C:\miniconda3\python.exe"
if not exist "%PY%" set "PY=python"
set "DIR=%~dp0"
if "%DIR:~-1%"=="\" if not "%DIR:~-2%"==":\" set "DIR=%DIR:~0,-1%"
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=%DIR%"
set "INVERT="
set /p "ANS=Invert output contrast (white background)? [y/N] "
if /i "%ANS%"=="y" set "INVERT=--invert"
"%PY%" "%DIR%\denoise_tool.py" "%TARGET%" %INVERT%
if errorlevel 1 (pause
    exit /b 1)
"%PY%" "%DIR%\stitch_tool.py" "%TARGET%\denoised"
if errorlevel 1 pause
endlocal
