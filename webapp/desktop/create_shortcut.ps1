# Creates "Volleyball Tactical Analysis" shortcuts on the Desktop and in the Start menu.
# Run:  powershell -ExecutionPolicy Bypass -File webapp\desktop\create_shortcut.ps1
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = (Get-Command python).Source
$pythonw = Join-Path (Split-Path $python) "pythonw.exe"
if (-not (Test-Path $pythonw)) { $pythonw = $python }
$shell = New-Object -ComObject WScript.Shell
$targets = @(
    [Environment]::GetFolderPath("Desktop"),
    (Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs")
)
foreach ($dir in $targets) {
    $lnk = $shell.CreateShortcut((Join-Path $dir "Volleyball Tactical Analysis.lnk"))
    $lnk.TargetPath = $pythonw
    $lnk.Arguments = "`"$here\desktop_app.py`""
    $lnk.WorkingDirectory = $here
    $lnk.IconLocation = "$here\app.ico"
    $lnk.Description = "Volleyball Tactical Analysis desktop app"
    $lnk.Save()
    Write-Output "created $dir\Volleyball Tactical Analysis.lnk"
}
