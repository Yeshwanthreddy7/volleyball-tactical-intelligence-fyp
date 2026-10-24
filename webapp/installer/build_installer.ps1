# Builds VolleyballTacticalAnalysis-Setup.exe.
#   powershell -ExecutionPolicy Bypass -File webapp\installer\build_installer.ps1
# Needs: built frontend (webapp\frontend\dist), Inno Setup 6 (ISCC.exe), uv.exe and
# a portable MongoDB (mongod.exe). Paths can be overridden with the parameters.
param(
    [string]$Iscc = "$env:USERPROFILE\vta_build_tools\InnoSetup\ISCC.exe",
    [string]$Uv = "$env:USERPROFILE\vta_build_tools\uv\uv.exe",
    [string]$Mongod = (Get-ChildItem "$env:USERPROFILE\mongodb" -Recurse -Filter mongod.exe -ErrorAction SilentlyContinue | Select-Object -First 1).FullName,
    [string]$Work = "$env:USERPROFILE\vta_build"
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$webapp = Split-Path -Parent $here
$repo = Split-Path -Parent $webapp
$stage = Join-Path $Work "stage"
$out = Join-Path $Work "output"

foreach ($p in @($Iscc, $Uv, $Mongod, "$webapp\frontend\dist\index.html")) {
    if (-not $p -or -not (Test-Path $p)) { throw "Missing: $p" }
}
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage, $out | Out-Null

function Copy-Rel([string]$rel, [string[]]$include = @("*")) {
    $src = Join-Path $repo $rel
    $dst = Join-Path $stage "app\repo\$rel"
    if (Test-Path $src -PathType Leaf) {
        New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
        Copy-Item $src $dst
    } elseif (Test-Path $src) {
        New-Item -ItemType Directory -Force $dst | Out-Null
        Get-ChildItem $src -Recurse -File -Include $include |
            Where-Object { $_.FullName -notmatch '\\(__pycache__|node_modules|\.pytest_cache)\\' } |
            ForEach-Object {
                $t = Join-Path $dst $_.FullName.Substring($src.Length).TrimStart('\')
                New-Item -ItemType Directory -Force (Split-Path $t) | Out-Null
                Copy-Item $_.FullName $t
            }
    } else { Write-Warning "skip (missing): $rel" }
}

Write-Host "Staging files..."
# research code + models the analysis engine runs
Copy-Rel "OUTPUTS\bicausal_extract.py"
Copy-Rel "OUTPUTS\bicausal_render.py"
Copy-Rel "OUTPUTS\yolo11m-pose.pt"
Copy-Rel "NEW_OUTPUTS\code" @("*.py")
Get-ChildItem "$repo\fyp\*.py" | ForEach-Object { Copy-Rel ("fyp\" + $_.Name) }
Copy-Rel "fyp\volleyball_best.pt"
Get-ChildItem "$repo\BASEFRAMES\*.py" | ForEach-Object { Copy-Rel ("BASEFRAMES\" + $_.Name) }
Copy-Rel "BASEFRAMES\corner_court_detection\labels"
Copy-Rel "BASEFRAMES\_contact_sheet.png"
Copy-Rel "tactical_ml_v6.joblib"
Copy-Rel "ml_results_v6\ml_results.json"
Copy-Rel "deep_results_v6\deep_results.json"
# documentation portal
Get-ChildItem "$repo\*.md" | ForEach-Object { Copy-Rel $_.Name }
foreach ($r in "ALL_13_MODELS\README.md", "BINARY_ATTACK_DEFENSE\README.md", "FINALE_OUTPUTS\README.md",
               "PANEL_REVIEW\README.md", "PANEL_REVIEW\architecture_diagram.svg", "figures_atlas",
               "figures_perclass_f1", "figures_workflow", "model_comparison", "lovo_metrics_chart.png",
               "train_test_split_chart.png", "PAPER_OUTPUT\action_examples") { Copy-Rel $r }
# web application
Copy-Rel "webapp\backend\app" @("*.py")
Copy-Rel "webapp\backend\requirements.txt"
Copy-Rel "webapp\frontend\dist"
Copy-Rel "webapp\desktop" @("*.py", "*.ico")
Copy-Rel "webapp\storage\demos"
Copy-Rel "webapp\DEPLOY.md"
# runtime tools
New-Item -ItemType Directory -Force "$stage\tools", "$stage\mongodb" | Out-Null
Copy-Item $Uv "$stage\tools\uv.exe"
Copy-Item $Mongod "$stage\mongodb\mongod.exe"
Copy-Item "$here\setup_env.ps1", "$here\requirements-desktop.txt", "$here\README.txt" $stage

$mb = [math]::Round(((Get-ChildItem $stage -Recurse -File | Measure-Object Length -Sum).Sum) / 1MB)
Write-Host "Staged $mb MB -> compiling installer..."
& $Iscc "/DSTAGE=$stage" "/DOUTDIR=$out" "$here\VolleyballTacticalAnalysis.iss"
if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }
$exe = Join-Path $out "VolleyballTacticalAnalysis-Setup.exe"
Write-Host ("Built {0} ({1} MB)" -f $exe, [math]::Round((Get-Item $exe).Length / 1MB))
