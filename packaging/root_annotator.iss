; Inno Setup script for Root Annotator.
;
; Build the PyInstaller bundle first (packaging/build_windows.ps1), then compile this with:
;   "C:\Program Files\Inno Setup 7\ISCC.exe" packaging\root_annotator.iss
;
; Produces installer\RootAnnotator-Setup.exe -- a normal Windows installer wizard: pick a
; folder, get Start Menu + optional Desktop shortcuts, and an Add/Remove Programs entry.

#define MyAppName "Root Annotator"
#define MyAppVersion "1.0.0"
#define MyAppExeName "root-annotator.exe"

[Setup]
AppId={{9F8F4AED-E97F-4C72-A69C-88D82F23A193}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
SetupIconFile=icon.ico
OutputDir=..\installer
OutputBaseFilename=RootAnnotator-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; The bundle already carries a multi-GB CUDA PyTorch build; skip re-compressing it hard.
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "..\dist\root-annotator\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
