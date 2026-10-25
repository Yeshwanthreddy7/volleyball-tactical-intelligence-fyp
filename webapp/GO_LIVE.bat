@echo off
rem Double-click to turn on video upload at https://volleyballanalyzer.dpdns.org
rem Keep this window open. Press Ctrl+C (then Y) to switch uploads off.
cd /d "%~dp0.."
powershell -ExecutionPolicy Bypass -File webapp\go_live.ps1
pause
