@echo off
rem One-time setup: adds "Denoise with CBI model" to the right-click Send To menu.
setlocal
set "PY=C:\miniconda3\python.exe"
if not exist "%PY%" set "PY=python"
set "TOOLDIR=%~dp0"
set "LNK=%APPDATA%\Microsoft\Windows\SendTo\Denoise with CBI model.bat"
> "%LNK%" echo @echo off
>> "%LNK%" echo "%PY%" "%TOOLDIR%denoise_tool.py" %%1
>> "%LNK%" echo if errorlevel 1 pause
echo Installed: %LNK%
echo Right-click any folder of tiles, choose Send To, then "Denoise with CBI model".
pause
endlocal
