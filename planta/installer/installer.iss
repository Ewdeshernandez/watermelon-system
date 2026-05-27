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
; Icono del Setup.exe (wizard) + del shortcut en el escritorio
SetupIconFile=assets\watermelon.ico
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

; ---------------------------------------------------------------------
; v3.31.247 — Bundled driver del fabricante (descargado por GitHub
; Action desde Supabase Storage y extraído del ISO en CI). Va a la
; carpeta temporal {tmp} del instalador, NO se queda en el disco del
; cliente — se ejecuta durante el install y después Inno Setup lo borra.
; ---------------------------------------------------------------------
Source: "dependencies\driver-extracted\*"; \
    DestDir: "{tmp}\driver-installer"; \
    Flags: ignoreversion recursesubdirs createallsubdirs deleteafterinstall \
    skipifsourcedoesntexist

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
; ---------------------------------------------------------------------
; v3.31.247 — Instalar driver del fabricante PRIMERO (silencioso).
; Solo si NO está ya instalado (detección en [Code] abajo).
; Flags /qb = barra de progreso visible pero sin preguntas
;       /norestart = NO reiniciar Windows en este punto (lo manejamos al
;                    final del wizard de Watermelon para que el cliente
;                    entienda por qué).
; ---------------------------------------------------------------------
Filename: "{tmp}\driver-installer\Install.exe"; \
    Parameters: "/qb /norestart /ACCEPTEULAS"; \
    StatusMsg: "Instalando driver de adquisición (10-15 min)..."; \
    Check: NeedsDriverInstall and DriverInstallerExists; \
    Flags: waituntilterminated

; ---------------------------------------------------------------------
; v3.31.255 — Silenciar el popup "Device Monitor" del fabricante.
; Cliente nuevo NO debe ver branding externo al conectar el hardware.
; Detenemos el servicio + lo deshabilitamos del autostart. La app
; Watermelon habla con el driver directo via DAQmx — no necesita el
; Device Monitor (que solo abre el popup de "configurar/explorar").
; ---------------------------------------------------------------------
Filename: "{sys}\sc.exe"; Parameters: "stop ""NIDeviceMonitor"""; \
    StatusMsg: "Configurando servicio de adquisición..."; \
    Flags: runhidden waituntilterminated; \
    Check: NeedsDriverInstall and DriverInstallerExists
Filename: "{sys}\sc.exe"; Parameters: "config ""NIDeviceMonitor"" start= disabled"; \
    Flags: runhidden waituntilterminated; \
    Check: NeedsDriverInstall and DriverInstallerExists
; Por si quedó el proceso colgado
Filename: "{sys}\taskkill.exe"; Parameters: "/F /IM nidevmon.exe /T"; \
    Flags: runhidden waituntilterminated; \
    Check: NeedsDriverInstall and DriverInstallerExists

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
// =====================================================================
// v3.31.247 — Detección del driver del fabricante.
// Si ya está instalado (registry key del fabricante presente),
// skipear la instalación del driver y solo instalar Watermelon.
// =====================================================================
function NeedsDriverInstall(): Boolean;
var
  Installed: Boolean;
begin
  // El driver del fabricante deja una key en el registry de Windows.
  // Chequeamos esa key — si existe, ya está instalado.
  // Path típico: HKLM\SOFTWARE\National Instruments\NI-DAQmx\CurrentVersion
  Installed := RegKeyExists(HKLM, 'SOFTWARE\National Instruments\NI-DAQmx\CurrentVersion') or
               RegKeyExists(HKLM, 'SOFTWARE\WOW6432Node\National Instruments\NI-DAQmx\CurrentVersion');
  if Installed then begin
    Log('Driver del fabricante ya instalado — skipeando instalación.');
    Result := False;
  end else begin
    Log('Driver del fabricante NO detectado — se instalará.');
    Result := True;
  end;
end;

// Helper: chequear que el installer del driver EXISTE en el bundle.
// (Si CI no lo descargó por alguna razón, no fallar el wizard entero —
//  solo se skipea la instalación del driver con un warning al final.)
function DriverInstallerExists(): Boolean;
begin
  Result := FileExists(ExpandConstant('{tmp}\driver-installer\Install.exe'));
  if not Result then begin
    Log('WARNING: Install.exe del driver no está en el bundle — skipeando.');
  end;
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    // v3.31.247 — Si instalamos el driver, recordar al user reiniciar
    // Windows. El driver del fabricante es de kernel, requiere reboot
    // para que Windows reconozca la maleta cuando se conecte.
    if NeedsDriverInstall() then begin
      MsgBox(
        'IMPORTANTE: Reiniciá Windows antes de conectar la maleta de ' +
        'adquisición.' + #13#10 + #13#10 +
        'El driver acaba de instalarse y necesita reinicio para que ' +
        'Windows reconozca el hardware.' + #13#10 + #13#10 +
        'Después del reinicio:' + #13#10 +
        '  1. Conectá la maleta por USB' + #13#10 +
        '  2. Esperá que Windows la detecte (LED verde)' + #13#10 +
        '  3. Abrí Watermelon Planta desde el escritorio',
        mbInformation,
        MB_OK
      );
    end;
  end;
end;
