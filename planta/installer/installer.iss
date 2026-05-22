; ==============================================================
;  Watermelon Planta Edition - Inno Setup installer script
; ==============================================================
;
; Genera WatermelonPlantaSetup-v1.0.exe — un instalador
; profesional Windows con:
;   - Splash screen + EULA
;   - Selección de directorio de instalación
;   - Shortcuts en Escritorio + Start Menu
;   - Uninstaller automático en Add/Remove Programs
;   - Detección de Python (advertencia si no está; no bloqueante
;     porque el .exe incluye Python embebido)
;
; Pre-requisitos:
;   1. Tener Inno Setup 6+ instalado (free, jrsoftware.org)
;   2. Tener dist\WatermelonPlanta.exe ya buildeado por PyInstaller
;
; Build:
;   doble click en build_installer.bat
;   o desde Inno Setup IDE: File → Open → installer.iss → Compile
;
; Output:
;   Output\WatermelonPlantaSetup-v1.0.0.exe (~280 MB)

#define MyAppName "Watermelon Planta Edition"
; MyAppVersion se lee dinámicamente del archivo VERSION del repo (raíz del proyecto)
; Ej: si VERSION dice "v3.31.230", MyAppVersion queda "3.31.230" (sin el "v")
#define VersionFile FileOpen(SourcePath + "..\..\VERSION")
#define VersionRaw Trim(FileRead(VersionFile))
#expr FileClose(VersionFile)
#define MyAppVersion StringChange(VersionRaw, "v", "")
#define MyAppPublisher "SIGA GROUP S.A.S"
#define MyAppURL "https://watermelonsys.net"
#define MyAppExeName "WatermelonPlanta.exe"
#define MyAppCopyright "© 2026 SIGA GROUP — Todos los derechos reservados"

[Setup]
; AppId DEBE ser único — usar GUID generado en https://www.guidgenerator.com/
; Cambiarlo regenera el app como "nuevo" en Windows
AppId={{A47B3F12-9B5E-4F8A-8D3C-2A1C8E9F1B5D}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
AppCopyright={#MyAppCopyright}

DefaultDirName={autopf}\WatermelonPlanta
DefaultGroupName=Watermelon
DisableProgramGroupPage=yes

; License agreement mostrada en el wizard
LicenseFile=assets\license.txt

; Pre-install splash (opcional) — comentado: en Inno Setup 6.4+ ya no
; existen los archivos WizModernImage-IS.bmp. El wizard usa el default moderno
; automáticamente cuando WizardStyle=modern (línea de abajo).
; WizardImageFile=compiler:WizModernImage-IS.bmp
; WizardSmallImageFile=compiler:WizModernSmallImage-IS.bmp

; Estilo del wizard
WizardStyle=modern
DisableWelcomePage=no

; Compression
Compression=lzma2/ultra64
SolidCompression=yes

; Output — guardamos en installer\dist (al lado del .exe portable de PyInstaller)
OutputDir=dist
OutputBaseFilename=WatermelonPlantaSetup-v{#MyAppVersion}
; SetupIconFile comentado — se usará el icono default de Inno Setup hasta
; que generemos un .ico profesional desde el SVG. El .exe instalado igual
; tiene el icono de WatermelonPlanta.exe (también default por ahora).
; SetupIconFile=assets\watermelon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

; Requires Windows 10+
MinVersion=10.0.17763

; Privileges
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Architecture
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; \
    Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
; El .exe principal (generado por PyInstaller en installer\dist\)
Source: "dist\WatermelonPlanta.exe"; DestDir: "{app}"; Flags: ignoreversion

; README operativo
Source: "..\README_PLANTA.txt"; DestDir: "{app}"; Flags: ignoreversion isreadme

; Template del secrets.toml para que el técnico lo edite post-install
Source: "..\.streamlit\secrets.toml.example"; \
    DestDir: "{app}\.streamlit"; \
    DestName: "secrets.toml.example"; Flags: ignoreversion

; Instrucciones de activación de licencia — el cliente lee esto primero
Source: "assets\LICENCIA_README.txt"; \
    DestDir: "{app}\data"; Flags: ignoreversion

[Dirs]
; Crear carpeta de capturas durante la instalación
Name: "{app}\data\captures"; Permissions: users-modify
; Carpeta data/ donde el cliente pega su license.token
Name: "{app}\data"; Permissions: users-modify

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; \
    Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    Tasks: desktopicon

[Run]
; Mostrar instrucciones de licencia al terminar la instalación
Filename: "notepad.exe"; Parameters: """{app}\data\LICENCIA_README.txt"""; \
    Description: "Ver instrucciones para activar tu licencia"; \
    Flags: postinstall skipifsilent shellexec

; Mostrar README operativo
Filename: "notepad.exe"; Parameters: """{app}\README_PLANTA.txt"""; \
    Description: "Ver README de instalación"; \
    Flags: postinstall skipifsilent shellexec

; Opción de abrir la app inmediatamente
Filename: "{app}\{#MyAppExeName}"; \
    Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; \
    Flags: nowait postinstall skipifsilent shellexec

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  // Aquí podemos agregar checks pre-install: NI-DAQmx driver, Python, etc.
  // Por ahora solo retornamos True.
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    // Aquí podríamos copiar secrets.toml.example → secrets.toml
    // si no existe ya, para que el user solo tenga que editarlo.
    // Por ahora se deja al user hacerlo manual.
  end;
end;
