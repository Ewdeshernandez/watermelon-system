============================================================
  WATERMELON PLANTA EDITION - GUÍA DE BUILD
============================================================

Esta guía explica cómo generar el instalador profesional
WatermelonPlantaSetup-v1.0.exe que recibe el cliente final.

Tiempo total: ~30 min la primera vez (incluye instalar tools).
Siguientes builds: ~10 min.


============================================================
  ARQUITECTURA DEL BUILD
============================================================

  Source code (planta/)
        ↓
  PyInstaller (build_exe.bat)
        ↓
  dist\WatermelonPlanta.exe  (~250 MB - el .exe single-file)
        ↓
  Inno Setup (build_installer.bat)
        ↓
  dist\WatermelonPlantaSetup-v1.0.0.exe (~280 MB - installer profesional)
        ↓
  ESTE archivo es el que mandas a clientes
        ↓
  Cliente doble click → wizard de instalación → ícono en escritorio →
  abre Watermelon Planta sin necesidad de tener Python instalado


============================================================
  PRE-REQUISITOS (UNA SOLA VEZ EN EL PC BUILD)
============================================================

1. Windows 10/11 64-bit con admin
2. Python 3.10-3.12 con "Add to PATH"
3. Driver de adquisición del fabricante (para que los imports nativos no fallen al buildear)
4. Inno Setup 6+ (free: https://jrsoftware.org/isdl.php)
5. Todas las dependencias del repo:
     cd planta
     pip install -r requirements-planta.txt
     pip install pyinstaller


============================================================
  ASSETS REQUERIDOS (UNA SOLA VEZ)
============================================================

Generar el icono del .exe (ver assets/README_ASSETS.txt):

  1. Ir a https://convertio.co/svg-ico/
  2. Subir assets/watermelon-logo.svg
  3. Descargar el .ico generado
  4. Guardarlo como  assets/watermelon.ico

Sin este archivo, build_exe.bat falla con "icon file not found".


============================================================
  PASO 1 - GENERAR EL .exe SINGLE-FILE (~10 min)
============================================================

En PowerShell:

  cd C:\path\to\watermelon-system\planta\installer
  .\build_exe.bat

Lo que pasa:
  - PyInstaller analiza launcher.py
  - Empaqueta Python + libs + nuestro código
  - Genera dist\WatermelonPlanta.exe

Tamaño esperado: 200-300 MB

Probarlo:
  Doble click en dist\WatermelonPlanta.exe
  Debe abrir tu browser default en localhost:8501
  Con Watermelon Planta funcionando exactamente igual que INICIAR.bat

Si falla, revisar build\WatermelonPlanta\warn-WatermelonPlanta.txt
para ver qué imports faltan.


============================================================
  PASO 2 - GENERAR EL INSTALLER PROFESIONAL (~5 min)
============================================================

En PowerShell:

  .\build_installer.bat

Lo que pasa:
  - Inno Setup compila installer.iss
  - Empaqueta WatermelonPlanta.exe + README + license + assets
  - Genera dist\WatermelonPlantaSetup-v1.0.0.exe

Tamaño esperado: ~280 MB

Probarlo:
  Doble click en dist\WatermelonPlantaSetup-v1.0.0.exe
  Debe abrir el wizard de instalación con:
    1. Welcome screen
    2. License agreement (license.txt)
    3. Selección de directorio
    4. Opciones de shortcuts
    5. Instalación
    6. Opción de abrir la app


============================================================
  DISTRIBUCIÓN A CLIENTES
============================================================

El archivo final dist\WatermelonPlantaSetup-v1.0.0.exe es el que mandas
a clientes. Puede ir por:

  - Email (si <25 MB después de compresión — improbable)
  - WeTransfer (free para <2 GB)
  - Google Drive / OneDrive (link)
  - USB stick directo
  - Servidor SIGA (futuro)

El cliente:
  1. Recibe el .exe
  2. Doble click → wizard de instalación
  3. Acepta licencia, elige carpeta
  4. Instalador crea shortcuts en Escritorio + Start Menu
  5. Doble click al ícono → Watermelon Planta arranca

NO requiere instalar Python, ni dependencias, ni nada extra.
SÍ requiere driver de adquisición del fabricante (eso es separado, una vez por PC).


============================================================
  TROUBLESHOOTING
============================================================

Error: "watermelon.ico no encontrado"
  → Generar el .ico como dice en assets/README_ASSETS.txt

Error: "ModuleNotFoundError" al ejecutar el .exe
  → Agregar el módulo faltante a hiddenimports en watermelon-planta.spec
  → Rebuild con build_exe.bat

Error: "build folder permission denied"
  → Cerrar todas las ventanas que tengan dist/ o build/ abiertas
  → Reintentar build_exe.bat

Error: "Inno Setup no encontrado"
  → Instalar de https://jrsoftware.org/isdl.php
  → O editar build_installer.bat con tu ruta de ISCC.exe

.exe muy grande (>400 MB)
  → Revisar excludes en watermelon-planta.spec
  → Quitar libs que no usemos (matplotlib, kaleido, etc.)


============================================================
  VERSIONAMIENTO
============================================================

Cada vez que cambies el código del repo y quieras generar un nuevo
installer:

  1. Bumpea VERSION (e.g. de 1.0.0 a 1.0.1) en installer.iss:
       #define MyAppVersion "1.0.1"
  2. Corre build_exe.bat (regenera .exe con código nuevo)
  3. Corre build_installer.bat (regenera installer con la nueva versión)
  4. El installer output sale como WatermelonPlantaSetup-v1.0.1.exe

Inno Setup detecta automáticamente que es upgrade y mantiene la
configuración del cliente (incluyendo secrets.toml).


============================================================
  SOPORTE
============================================================

Para dudas sobre el build, contactar a Ewdes Hernández.
Repositorio: https://github.com/Ewdeshernandez/watermelon-system
