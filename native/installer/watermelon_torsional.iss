; Inno Setup — instalador de Watermelon Torsional (Windows)
; Empaqueta out_torsional/ (WatermelonTorsional.exe + lanzador + LEEME) en un
; instalador con logo, menu Inicio, acceso directo de escritorio y desinstalador.
; La version se pasa desde el workflow:  iscc /DMyAppVersion=0.1.0 native/installer/watermelon_torsional.iss
; NOTA: en Inno Setup cada entrada [Files]/[Icons]/[Run] va en UNA sola linea (sin "\").

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#define MyAppName "Watermelon Torsional"
#define MyAppPublisher "SIGA"
#define MyAppExeName "WatermelonTorsional.exe"
#define MyAppURL "https://watermelonsystem.app"

[Setup]
; AppId PROPIO (distinto de Modal / Rotordynamics) -> las apps conviven sin chocar.
AppId={{D1B7F3C4-6E5A-4B8C-9D0E-3F4A5B6C7D8E}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\Watermelon Torsional
DefaultGroupName=Watermelon Torsional
DisableProgramGroupPage=yes
DisableDirPage=auto
OutputDir={#SourcePath}..\..
OutputBaseFilename=WatermelonTorsional-Setup
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
Source: "{#SourcePath}..\..\out_torsional\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Watermelon Torsional"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--sim"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,Watermelon Torsional}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Watermelon Torsional"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--sim"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--sim"; Description: "{cm:LaunchProgram,Watermelon Torsional}"; Flags: nowait postinstall skipifsilent
