@echo off
REM Starts the Volleyball Tactical Analysis website on this PC: http://volleyballtactics.localhost
REM Uploads and results are stored outside OneDrive (OneDrive syncing large videos slows processing).
setlocal
cd /d "%~dp0"
if "%VTA_STORAGE_DIR%"=="" set "VTA_STORAGE_DIR=%USERPROFILE%\vta_storage"
if "%VTA_DEMOS_DIR%"=="" set "VTA_DEMOS_DIR=%~dp0storage\demos"
if "%MONGODB_URI%"=="" set "MONGODB_URI=mongodb://localhost:27017"
set LOKY_MAX_CPU_COUNT=4
REM port 80 gives the plain address http://volleyballtactics.localhost (8000 if 80 is busy)
if "%VTA_PORT%"=="" (
  powershell -NoProfile -Command "exit [int](Test-NetConnection 127.0.0.1 -Port 80 -InformationLevel Quiet -WarningAction SilentlyContinue)"
  if errorlevel 1 (set VTA_PORT=8000) else (set VTA_PORT=80)
)

REM Portable MongoDB in %USERPROFILE%\mongodb (no Docker / admin needed): start it if not running
powershell -NoProfile -Command "if (-not (Test-NetConnection 127.0.0.1 -Port 27017 -InformationLevel Quiet -WarningAction SilentlyContinue)) { $b = Get-ChildItem \"$env:USERPROFILE\mongodb\" -Recurse -Filter mongod.exe -ErrorAction SilentlyContinue | Select-Object -First 1; if ($b) { Start-Process $b.FullName -ArgumentList ('--dbpath \"' + $env:USERPROFILE + '\mongodb\data\" --bind_ip 127.0.0.1 --port 27017 --wiredTigerCacheSizeGB 0.25 --logpath \"' + $env:USERPROFILE + '\mongodb\mongod.log\"') -WindowStyle Hidden; Start-Sleep 4; 'MongoDB started' } else { 'MongoDB not found - using JSON file store' } } else { 'MongoDB already running' }"

if not exist "frontend\dist\index.html" (
  echo Building frontend...
  pushd frontend
  call npm install --include=dev || goto :err
  call npm run build || goto :err
  popd
)
echo Storage : %VTA_STORAGE_DIR%
if "%VTA_PORT%"=="80" (echo Open    : http://volleyballtactics.localhost) else (echo Open    : http://volleyballtactics.localhost:%VTA_PORT%)
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port %VTA_PORT%
goto :eof
:err
echo Frontend build failed.
exit /b 1
