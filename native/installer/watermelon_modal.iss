; Inno Setup — instalador de Watermelon Modal (Windows)
; Empaqueta out_modal/ (WatermelonModal.exe + lanzador + LEEME) en un instalador
; con logo, menu Inicio, acceso directo de escritorio y desinstalador.
; La version se pasa desde el workflow:  iscc /DMyAppVersion=0.1.0 native/installer/watermelon_modal.iss
; NOTA: en Inno Setup cada entrada [Files]/[Icons]/[Run] va en UNA sola linea (sin "\").

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#define MyAppName "Watermelon Modal"
#define MyAppPublisher "SIGA"
#define MyAppExeName "WatermelonModal.exe"
#define MyAppURL "https://watermelonsystem.app"

[Setup]
; AppId PROPIO (distinto de Rotordynamics) -> las dos apps conviven sin chocar.
AppId={{C9A4B2D3-5E6F-4A7B-8C9D-2E3F4A5B6C7D}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\Watermelon Modal
DefaultGroupName=Watermelon Modal
DisableProgramGroupPage=yes
DisableDirPage=auto
OutputDir={#SourcePath}..\..
OutputBaseFilename=WatermelonModal-Setup
SetupIconFile={#SourcePath}..\..\assets\watermelon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourcePath}..\..\out_modal\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Watermelon Modal"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--sim"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,Watermelon Modal}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Watermelon Modal"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--sim"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--sim"; Description: "{cm:LaunchProgram,Watermelon Modal}"; Flags: nowait postinstall skipifsilent
