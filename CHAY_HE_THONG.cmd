@echo off
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0start_demo.ps1"
if errorlevel 1 pause
