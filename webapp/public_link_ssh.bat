@echo off
REM Temporary public HTTPS link through localhost.run (free, no account).
REM Use this when public_link.bat (Cloudflare) fails because the network blocks port 7844.
REM Needs run_website.bat running. Copy the https://....lhr.life link it prints.
REM The link works while this window stays open; it changes every time you start it.
if "%VTA_PORT%"=="" set VTA_PORT=80
ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -R 80:127.0.0.1:%VTA_PORT% nokey@localhost.run
