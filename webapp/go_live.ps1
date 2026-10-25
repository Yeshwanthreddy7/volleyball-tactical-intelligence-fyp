# Makes video upload work on https://volleyballanalyzer.dpdns.org (and the
# .vercel.app names): starts the analysis server on this laptop, opens a free
# public tunnel to it (localhost.run over SSH, works on college Wi-Fi), and
# points the website's "live server" at that tunnel. localhost.run changes the
# tunnel address from time to time, so the address is watched and the website
# is re-pointed whenever it changes.
# Keep this window open; press Ctrl+C to go offline again.
#   powershell -ExecutionPolicy Bypass -File webapp\go_live.ps1
param([string]$Project = "volleyballtactics", [int]$Port = 80)
$ErrorActionPreference = "Continue"
$webapp = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $webapp
$vercelDir = Join-Path $env:USERPROFILE "vta_build\vercel"
$tunnelLog = Join-Path $env:TEMP "vta_tunnel.log"
$site = "https://volleyballanalyzer.dpdns.org"
Start-Transcript -Path (Join-Path $env:TEMP "vta_golive.log") -Force | Out-Null

function Healthy { try { (Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 5).status -eq "ok" } catch { $false } }

function Deploy {
    Push-Location $vercelDir
    Set-Content .vercelignore ".env*`n.gitignore"
    npx --yes vercel@65.2.0 link --yes --project $Project 2>&1 | Out-Null
    $out = npx --yes vercel@65.2.0 deploy --prod --yes 2>&1
    Pop-Location
    $ok = [bool]($out | Select-String "Production")
    if (-not $ok) { $out | Select-Object -Last 15 | Out-Host }
    return $ok
}

# fast path: only the live_url inside the exported health.json changes
function Publish([string]$liveUrl) {
    Write-Host "Pointing $site at '$liveUrl' ..." -ForegroundColor Cyan
    $health = Join-Path $vercelDir "static-api\health.json"
    if (-not (Test-Path $health)) {
        Push-Location $repo
        python webapp\scripts\export_static.py | Out-Host
        Pop-Location
    }
    $h = Get-Content $health -Raw | ConvertFrom-Json
    $h.live_url = $liveUrl
    [IO.File]::WriteAllText($health, ($h | ConvertTo-Json -Depth 5))
    for ($i = 1; $i -le 3; $i++) {
        if (Deploy) { Write-Host "Website updated." -ForegroundColor Green; return $true }
        Write-Host "Deploy failed (try $i of 3), retrying..." -ForegroundColor Yellow
        Start-Sleep 5
    }
    return $false
}

function Latest-Url {
    $m = Select-String -Path $tunnelLog -Pattern "tunneled with tls termination, (https://\S+)" -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($m) { return $m.Matches[0].Groups[1].Value.TrimEnd(",") }
    return $null
}

function Start-Tunnel {
    Remove-Item $tunnelLog, "$tunnelLog.err" -ErrorAction SilentlyContinue
    return Start-Process ssh.exe -PassThru -WindowStyle Hidden -RedirectStandardOutput $tunnelLog -RedirectStandardError "$tunnelLog.err" `
        -ArgumentList "-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -R 80:127.0.0.1:$Port nokey@localhost.run"
}

# 1. analysis server
if (-not (Healthy)) {
    Write-Host "Starting the analysis server..." -ForegroundColor Cyan
    Start-Process (Join-Path $webapp "run_website.bat") -WindowStyle Minimized
    for ($i = 0; $i -lt 60 -and -not (Healthy); $i++) { Start-Sleep 2 }
    if (-not (Healthy)) { Write-Host "Server did not start. Run webapp\run_website.bat and check its window." -ForegroundColor Red; exit 1 }
}
Write-Host "Analysis server OK (http://volleyballtactics.localhost)" -ForegroundColor Green

# make sure the exported site carries the latest frontend build
Push-Location $repo
python webapp\scripts\export_static.py | Out-Host
Pop-Location

# 2-3. tunnel + publish; re-done whenever the tunnel drops or changes address
$proc = $null
$published = ""
try {
    while ($true) {
        if (-not $proc -or $proc.HasExited) {
            Write-Host "Opening public tunnel..." -ForegroundColor Cyan
            $proc = Start-Tunnel
            Start-Sleep 5
        }
        $url = Latest-Url
        if ($url -and $url -ne $published) {
            Write-Host "Tunnel: $url" -ForegroundColor Green
            if (Publish $url) {
                $published = $url
                Write-Host ""
                Write-Host "LIVE: video upload now works on $site" -ForegroundColor Green
                Write-Host "Keep this window open and the laptop awake. Ctrl+C = go offline." -ForegroundColor Yellow
            }
        }
        Start-Sleep 5
    }
} finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process $proc -Force }
    # back to the always-on Azure analysis server (webapp/azure/), not offline
    Write-Host "Laptop offline: pointing the website back at the Azure server..." -ForegroundColor Cyan
    Publish "https://20-197-61-218.sslip.io" | Out-Null
    Stop-Transcript | Out-Null
}
