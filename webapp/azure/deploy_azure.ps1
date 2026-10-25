# Pushes the current code to the Azure server and republishes the public website.
#   powershell -ExecutionPolicy Bypass -File webapp\azure\deploy_azure.ps1
# Server: Azure for Students VM "vta-server" (resource group vta-rg, Central India,
# Standard_B2als_v2, Ubuntu 24.04), app + MongoDB + Caddy (HTTPS) via docker compose
# in /srv/vta. Public API: https://20-197-61-218.sslip.io
param([string]$Ip = "20.197.61.218", [string]$Key = "$env:USERPROFILE\.ssh\vta_azure")
$ErrorActionPreference = "Stop"
$webapp = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$repo = Split-Path -Parent $webapp
$stage = Join-Path $env:USERPROFILE "vta_build\hfspace"

Push-Location $webapp\frontend; npm run build | Out-Null; Pop-Location
Push-Location $repo; python webapp\hfspace\build_hf_space.py; Pop-Location
$tgz = Join-Path $env:USERPROFILE "vta_build\vta_app.tgz"
tar -czf $tgz -C $stage .
scp -i $Key $tgz "azureuser@${Ip}:/srv/vta/app/vta_app.tgz"
scp -i $Key "$webapp\azure\docker-compose.yml" "$webapp\azure\Caddyfile" "azureuser@${Ip}:/srv/vta/"
ssh -i $Key "azureuser@$Ip" "cd /srv/vta && rm -rf src && mkdir src && tar -xzf app/vta_app.tgz -C src && sudo docker compose up -d --build"

# website (Vercel + volleyballanalyzer.dpdns.org) -> uploads go to the Azure server
Push-Location $repo; python webapp\scripts\export_static.py; Pop-Location
Push-Location (Join-Path $env:USERPROFILE "vta_build\vercel")
npx --yes vercel@65.2.0 link --yes --project volleyballtactics | Out-Null
npx --yes vercel@65.2.0 deploy --prod --yes
Pop-Location
Write-Host "Deployed. Check https://20-197-61-218.sslip.io/api/health and https://volleyballanalyzer.dpdns.org" -ForegroundColor Green
