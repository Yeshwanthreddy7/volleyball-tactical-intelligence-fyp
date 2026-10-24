# Checks that the website started by run_website.bat is working.
#   powershell -ExecutionPolicy Bypass -File webapp\check_website.ps1 [-Url http://volleyballtactics.localhost]
param([string]$Url = "http://127.0.0.1")
$ok = 0; $bad = 0
function Check($name, $path, $expect) {
    try {
        $r = Invoke-WebRequest "$Url$path" -UseBasicParsing -TimeoutSec 20 -Headers @{ "X-Client-Id" = "check-website-0000000" }
        if ($expect -and $r.Content -notmatch $expect) { throw "unexpected content" }
        Write-Host ("  OK    {0,-28} {1}" -f $name, $path) -ForegroundColor Green; $script:ok++
    } catch {
        Write-Host ("  FAIL  {0,-28} {1}  ({2})" -f $name, $path, $_.Exception.Message) -ForegroundColor Red; $script:bad++
    }
}
Write-Host "Checking $Url" -ForegroundColor Cyan
Check "Home page"            "/"                          "Volleyball Tactical Analysis"
Check "Server health"        "/api/health"                '"status":"ok"'
Check "Analyze page"         "/analyze"                   "<div id=`"root`">"
Check "Model results"        "/api/project/metrics"       "hist_gradient_boosting"
Check "Documentation"        "/api/project/docs/readme"   "markdown"
Check "Figures"              "/api/project/figures"       "path"
Check "Demo videos"          "/api/project/demos"         "mp4"
Check "My analyses (DB)"     "/api/analyses"              ""
try {
    $h = Invoke-RestMethod "$Url/api/health" -TimeoutSec 20
    Write-Host ""
    Write-Host ("  Database : {0}" -f $h.database)
    Write-Host ("  GPU      : {0}" -f $(if ($h.gpu.available) { $h.gpu.name } else { "not available" }))
    Write-Host ("  Models   : {0}" -f $(if ($h.models_present) { "all present" } else { "MISSING" }))
} catch {}
Write-Host ""
if ($bad -eq 0) { Write-Host "Website is working ($ok checks passed). Open http://volleyballtactics.localhost in your browser" -ForegroundColor Green }
else { Write-Host "$bad check(s) failed. Is run_website.bat running?" -ForegroundColor Red; exit 1 }
