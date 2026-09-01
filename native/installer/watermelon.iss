; Inno Setup — instalador de Watermelon Field (Windows)
; Empaqueta out/ (WatermelonField.exe + lanzadores + Banco_de_Pruebas + LEEME) en un
; instalador con logo, menú Inicio, acceso directo de escritorio y desinstalador.
; La versión se pasa desde el workflow:  iscc /DMyAppVersion=0.5.51 native/installer/watermelon.iss

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#define MyAppName "Watermelon Field"
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
DefaultDirName={autopf}\Watermelon Field
DefaultGroupName=Watermelon Field
DisableProgramGroupPage=yes
DisableDirPage=auto
; Rutas ABSOLUTAS vía {#SourcePath} (dir del .iss = native/installer) + ..\.. = raíz del
; repo. Evita la ambigüedad de rutas relativas de Inno Setup.
OutputDir={#SourcePath}..\..
OutputBaseFilename=WatermelonField-Setup
SetupIconFile={#SourcePath}..\..\assets\watermelon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; per-user install → no requiere permisos de administrador (ideal para PC de campo)
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; todo el contenido de out/ (exe + .bat + Banco_de_Pruebas + LEEME.txt)
Source: "{#SourcePath}..\..\out\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
; acceso principal → abre la app (simulador, siempre abre y muestra la UI)
Name: "{group}\Watermelon Field"; Filename: "{app}\{#MyAppExeName}"; \
  Parameters: "--scenario prox_4brg --fs 5120"; WorkingDir: "{app}"; \
  IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\Watermelon Field (DEMO)"; Filename: "{app}\{#MyAppExeName}"; \
  Parameters: "--sim"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\Banco de Pruebas"; Filename: "{app}\Banco_de_Pruebas"
Name: "{group}\{cm:UninstallProgram,Watermelon Field}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Watermelon Field"; Filename: "{app}\{#MyAppExeName}"; \
  Parameters: "--scenario prox_4brg --fs 5120"; WorkingDir: "{app}"; \
  IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--scenario prox_4brg --fs 5120"; \
  Description: "{cm:LaunchProgram,Watermelon Field}"; Flags: nowait postinstall skipifsilent
