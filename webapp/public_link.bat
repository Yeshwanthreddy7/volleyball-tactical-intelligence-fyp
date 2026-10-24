@echo off
REM Puts the running website (run_website.bat) on the internet.
REM   public_link.bat            -> temporary https://<random>.trycloudflare.com link (no account needed)
REM   public_link.bat <TOKEN>    -> your own domain via a named Cloudflare Tunnel (see DEPLOY.md)
REM The link works only while this PC is on and both windows stay open.
setlocal
if "%VTA_PORT%"=="" set VTA_PORT=80
set "CF=cloudflared"
where cloudflared >nul 2>&1 || set "CF=%USERPROFILE%\cloudflared\cloudflared.exe"
if not exist "%CF%" if not "%CF%"=="cloudflared" (
  echo cloudflared not found. Download it to %USERPROFILE%\cloudflared\cloudflared.exe from
  echo https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
  exit /b 1
)
if "%~1"=="" (
  echo Starting temporary public link - look for https://....trycloudflare.com below
  "%CF%" tunnel --no-autoupdate --protocol http2 --url http://localhost:%VTA_PORT%
) else (
  "%CF%" tunnel --no-autoupdate --protocol http2 run --token %~1
)
