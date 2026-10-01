@echo off
rem One-time setup: adds "Denoise + Stitch with CBI model" to the right-click Send To menu.
setlocal
set "TOOLDIR=%~dp0"
set "LNK=%APPDATA%\Microsoft\Windows\SendTo\Denoise + Stitch with CBI model.bat"
> "%LNK%" echo @echo off
>> "%LNK%" echo call "%TOOLDIR%process_here.bat" %%1
echo Installed: %LNK%
echo Right-click any folder of tiles, choose Send To, then "Denoise + Stitch with CBI model".
pause
endlocal
