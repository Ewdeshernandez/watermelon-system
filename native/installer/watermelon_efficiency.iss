; Inno Setup — instalador de Watermelon Efficiency (Windows)
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#define MyAppName "Watermelon Efficiency"
#define MyAppPublisher "SIGA"
#define MyAppExeName "WatermelonEfficiency.exe"
#define MyAppURL "https://watermelonsystem.app"

[Setup]
AppId={{3F9D2B64-7A1C-4E85-9C2D-5B0E8A4F1D37}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\Watermelon Efficiency
DefaultGroupName=Watermelon Efficiency
DisableProgramGroupPage=yes
DisableDirPage=auto
OutputDir={#SourcePath}..\..
OutputBaseFilename=WatermelonEfficiency-Setup
SetupIconFile={#SourcePath}..\..\assets\watermelon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
CloseApplications=yes
RestartApplications=no
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourcePath}..\..\out_efficiency\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Watermelon Efficiency"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--sim"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,Watermelon Efficiency}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Watermelon Efficiency"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--sim"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--sim"; Description: "{cm:LaunchProgram,Watermelon Efficiency}"; Flags: nowait postinstall skipifsilent
