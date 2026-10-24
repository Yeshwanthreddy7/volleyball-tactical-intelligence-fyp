# Run by the installer after the files are copied: creates the private Python
# environment the app runs in. Needs internet (about 3 GB with an NVIDIA GPU,
# about 1 GB without). Safe to re-run: Start menu -> "Repair Volleyball Tactical Analysis".
param([Parameter(Mandatory = $true)][string]$AppDir)
# native tools (uv, nvidia-smi) report on stderr; failures are checked through $LASTEXITCODE
$ErrorActionPreference = "Continue"
$Host.UI.RawUI.WindowTitle = "Volleyball Tactical Analysis - installing AI components"
$log = Join-Path $AppDir "install.log"
Start-Transcript -Path $log -Force | Out-Null

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }

try {
    $uv = Join-Path $AppDir "tools\uv.exe"
    $venv = Join-Path $AppDir "venv"
    $py = Join-Path $venv "Scripts\python.exe"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $AppDir "python"
    $env:UV_CACHE_DIR = Join-Path $AppDir "uv-cache"
    $env:UV_LINK_MODE = "copy"

    Write-Host "Volleyball Tactical Analysis setup" -ForegroundColor Yellow
    Write-Host "Downloading Python and the AI libraries. Keep this window open; it can take 10-40 minutes."

    Step "Python 3.12"
    & $uv venv $venv --python 3.12 --seed
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Python environment." }

    $gpu = $false
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi -and $env:VTA_FORCE_CPU -ne "1") { & nvidia-smi -L 2>$null | Out-Null; $gpu = ($LASTEXITCODE -eq 0) }
    if ($gpu) {
        Step "NVIDIA GPU found: installing PyTorch with CUDA (about 2.5 GB)"
        $index = "https://download.pytorch.org/whl/cu121"
    } else {
        Step "No NVIDIA GPU found: installing CPU PyTorch (analysis will be slow; use Fast mode and short clips)"
        $index = "https://download.pytorch.org/whl/cpu"
    }
    & $uv pip install --python $py "torch==2.5.1" "torchvision==0.20.1" --index-url $index
    if ($LASTEXITCODE -ne 0) { throw "PyTorch download failed." }

    Step "Computer-vision, machine-learning and web libraries"
    & $uv pip install --python $py -r (Join-Path $AppDir "requirements-desktop.txt")
    if ($LASTEXITCODE -ne 0) { throw "Library download failed." }

    Step "Checking the installation"
    & $py -c "import torch, ultralytics, sklearn, cv2, fastapi, webview, pymongo; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "Installed libraries could not be loaded." }

    Remove-Item -Recurse -Force $env:UV_CACHE_DIR -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "All components installed." -ForegroundColor Green
    Stop-Transcript | Out-Null
    exit 0
} catch {
    Write-Host ""
    Write-Host "SETUP FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Check your internet connection, then run 'Repair Volleyball Tactical Analysis' from the Start menu."
    Write-Host "Log: $log"
    Stop-Transcript | Out-Null
    Start-Sleep -Seconds 8
    exit 1
}
