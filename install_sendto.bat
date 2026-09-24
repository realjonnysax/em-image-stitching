@echo off
rem One-time setup: adds "Stitch with Ashlar" to the right-click Send To menu.
setlocal
set "PY=C:\miniconda3\python.exe"
if not exist "%PY%" set "PY=python"
set "TOOLDIR=%~dp0"
set "LNK=%APPDATA%\Microsoft\Windows\SendTo\Stitch with Ashlar.bat"
> "%LNK%" echo @echo off
>> "%LNK%" echo "%PY%" "%TOOLDIR%stitch_tool.py" %%1
>> "%LNK%" echo if errorlevel 1 pause
echo Installed: %LNK%
echo Right-click any folder of tiles, choose Send To, then "Stitch with Ashlar".
pause
endlocal
