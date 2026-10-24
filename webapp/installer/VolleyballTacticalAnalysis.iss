; Volleyball Tactical Analysis - Windows installer (Inno Setup 6).
; Built by build_installer.ps1, which stages the files into STAGE first.
#ifndef STAGE
  #define STAGE "stage"
#endif
#ifndef OUTDIR
  #define OUTDIR "output"
#endif
#define AppName "Volleyball Tactical Analysis"
#define AppVersion "1.0.0"

[Setup]
AppId={{6F1C2B7E-3D4A-4E59-9B1F-5A6C7D8E9F10}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Group 56 - Final Year Project
AppPublisherURL=https://github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp
DefaultDirName={localappdata}\VolleyballTacticalAnalysis
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OUTDIR}
OutputBaseFilename=VolleyballTacticalAnalysis-Setup
SetupIconFile={#STAGE}\app\repo\webapp\desktop\app.ico
UninstallDisplayIcon={app}\app\repo\webapp\desktop\app.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
InfoBeforeFile={#STAGE}\README.txt
ExtraDiskSpaceRequired=5368709120

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#STAGE}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\app\repo\webapp\desktop\desktop_app.py"""; WorkingDir: "{app}\app\repo\webapp\desktop"; IconFilename: "{app}\app\repo\webapp\desktop\app.ico"
Name: "{group}\Repair {#AppName}"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\setup_env.ps1"" -AppDir ""{app}"""; IconFilename: "{app}\app\repo\webapp\desktop\app.ico"
Name: "{group}\Read me"; Filename: "{app}\README.txt"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\app\repo\webapp\desktop\desktop_app.py"""; WorkingDir: "{app}\app\repo\webapp\desktop"; IconFilename: "{app}\app\repo\webapp\desktop\app.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\app\repo\webapp\desktop\desktop_app.py"""; WorkingDir: "{app}\app\repo\webapp\desktop"; Description: "Start {#AppName} now"; Flags: postinstall nowait skipifsilent; Check: EnvReady

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -Command ""Get-Process mongod,pythonw,python -ErrorAction SilentlyContinue | Where-Object {{ $_.Path -like '{app}\*' } | Stop-Process -Force"""; Flags: runhidden; RunOnceId: "StopAppProcesses"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\uv-cache"
Type: filesandordirs; Name: "{app}\data"
Type: files; Name: "{app}\install.log"

[Code]
var
  EnvOk: Boolean;

function EnvReady: Boolean;
begin
  Result := EnvOk;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := 'Downloading Python and AI libraries (see the console window)...';
    EnvOk := Exec('powershell.exe',
      '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\setup_env.ps1') +
      '" -AppDir "' + ExpandConstant('{app}') + '"',
      ExpandConstant('{app}'), SW_SHOW, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
    if not EnvOk then
      MsgBox('The AI components could not be downloaded (internet needed).' + #13#10 +
             'Connect to the internet and run "Repair Volleyball Tactical Analysis" from the Start menu.' + #13#10 +
             'Details: ' + ExpandConstant('{app}\install.log'), mbError, MB_OK);
  end;
end;
