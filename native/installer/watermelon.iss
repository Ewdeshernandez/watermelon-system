; Inno Setup — instalador de Watermelon Rotordynamics (Windows)
; Empaqueta out/ (WatermelonField.exe + lanzadores + Banco_de_Pruebas + LEEME) en un
; instalador con logo, menu Inicio, acceso directo de escritorio y desinstalador.
; La version se pasa desde el workflow:  iscc /DMyAppVersion=0.5.54 native/installer/watermelon.iss
; NOTA: en Inno Setup cada entrada [Files]/[Icons]/[Run] va en UNA sola linea (sin "\").

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#define MyAppName "Watermelon Rotordynamics"
#define MyAppPublisher "SIGA"
#define MyAppExeName "WatermelonField.exe"
#define MyAppURL "https://watermelonsystem.app"

[Setup]
AppId={{B8F3A1C2-4D5E-4F6A-9B7C-1D2E3F4A5B6C}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\Watermelon Rotordynamics
DefaultGroupName=Watermelon Rotordynamics
DisableProgramGroupPage=yes
DisableDirPage=auto
OutputDir={#SourcePath}..\..
OutputBaseFilename=WatermelonRotordynamics-Setup
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
Source: "{#SourcePath}..\..\out\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Watermelon Rotordynamics"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--prox --sens 200 --fs 5120 --chans 22,23 --names 1Y,1X --kph-bnc 21 --machine Rotor_Kit_Field"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\Watermelon Rotordynamics (DEMO)"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--sim"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\Banco de Pruebas"; Filename: "{app}\Banco_de_Pruebas"
Name: "{group}\{cm:UninstallProgram,Watermelon Rotordynamics}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Watermelon Rotordynamics"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--prox --sens 200 --fs 5120 --chans 22,23 --names 1Y,1X --kph-bnc 21 --machine Rotor_Kit_Field"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--prox --sens 200 --fs 5120 --chans 22,23 --names 1Y,1X --kph-bnc 21 --machine Rotor_Kit_Field"; Description: "{cm:LaunchProgram,Watermelon Rotordynamics}"; Flags: nowait postinstall skipifsilent
