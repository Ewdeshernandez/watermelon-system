# -*- mode: python ; coding: utf-8 -*-
"""
watermelon-planta.spec — Receta PyInstaller para Watermelon Planta Edition
============================================================================

Genera WatermelonPlanta.exe single-file que el cliente recibe e instala.

Build:
    cd planta\\installer
    pyinstaller watermelon-planta.spec --clean --noconfirm

Output:
    dist/WatermelonPlanta.exe (~250 MB)

Después de buildear, ese .exe se wrappea con Inno Setup (installer.iss)
para generar el installer profesional WatermelonPlantaSetup-v1.0.exe.

Tamaño:
PyInstaller incluye Python interpreter + todas las libs en el .exe.
Streamlit + plotly + pandas son pesadas → 200-300 MB es esperado.
Vale la pena: cliente no necesita Python instalado.

Hidden imports:
PyInstaller analiza imports estáticos. Streamlit + supabase + nidaqmx
usan imports dinámicos (importlib, lazy) → hay que listarlos manual
para que se incluyan en el bundle.
"""
import sys
from pathlib import Path
from PyInstaller.utils.hooks import (  # type: ignore
    collect_data_files, collect_submodules, copy_metadata,
)

# Path relativo desde el .spec hasta el root del repo
_SPEC_DIR = Path(SPECPATH).resolve()      # planta/installer/
_PLANTA_DIR = _SPEC_DIR.parent             # planta/
_REPO_ROOT = _PLANTA_DIR.parent            # repo root

block_cipher = None

# ============================================================
# Datas — archivos que hay que incluir en el bundle
# ============================================================
datas = []

# El app principal y la página de captura
datas += [
    (str(_PLANTA_DIR / "app_planta.py"),                "planta"),
    (str(_PLANTA_DIR / "auth_planta.py"),                "planta"),
    (str(_PLANTA_DIR / "sync_uploader.py"),              "planta"),
    (str(_PLANTA_DIR / "license_manager.py"),            "planta"),
    (str(_PLANTA_DIR / "updater.py"),                    "planta"),
    # Icono para el tray icon en runtime
    (str(_SPEC_DIR / "assets" / "watermelon.ico"),
        "planta/installer/assets"),
    (str(_PLANTA_DIR / "pages" / "01_Captura_Modal.py"), "planta/pages"),
    # README y assets visibles al cliente
    (str(_PLANTA_DIR / "README_PLANTA.txt"),             "planta"),
    # Template del secrets.toml
    (str(_PLANTA_DIR / ".streamlit" / "secrets.toml.example"),
        "planta/.streamlit"),
]

# Core modal (reusable desde planta/)
_CORE_MODAL = _REPO_ROOT / "core" / "modal"
if _CORE_MODAL.exists():
    for py in _CORE_MODAL.glob("*.py"):
        datas.append((str(py), "core/modal"))

# Streamlit static files (HTML/JS del frontend)
datas += collect_data_files("streamlit")

# Plotly static (matplotlib data + plotly.js)
datas += collect_data_files("plotly")

# Importer metadata (necesario para que Streamlit detecte sus extensiones)
datas += copy_metadata("streamlit")
datas += copy_metadata("supabase")

# v3.31.339 — FIX bug "Componente de adquisición pendiente": el paquete de
# adquisición y el de TDMS tienen un ÁRBOL de submódulos. Antes se listaban
# 3 submódulos a mano → el .exe congelado quedaba incompleto y `import` fallaba
# en campo. Ahora se colectan TODOS sus submódulos + data files + metadata.
for _pkg in ("nidaqmx", "nptdms"):
    try:
        datas += collect_data_files(_pkg)
    except Exception:  # noqa: BLE001
        pass
    try:
        datas += copy_metadata(_pkg)
    except Exception:  # noqa: BLE001
        pass

# ============================================================
# Hidden imports — módulos que PyInstaller no detecta automático
# ============================================================
hiddenimports = []
hiddenimports += collect_submodules("streamlit")
hiddenimports += collect_submodules("plotly")
hiddenimports += collect_submodules("pandas")
hiddenimports += collect_submodules("supabase")
hiddenimports += collect_submodules("storage3")
hiddenimports += collect_submodules("postgrest")
hiddenimports += collect_submodules("realtime")
hiddenimports += collect_submodules("supabase_auth")

# v3.31.339 — Colectar el ÁRBOL COMPLETO de submódulos del paquete de
# adquisición y de TDMS (antes solo 3 a mano → bundle incompleto → ImportError
# en campo = "Componente de adquisición pendiente").
hiddenimports += collect_submodules("nidaqmx")
hiddenimports += collect_submodules("nptdms")

hiddenimports += [
    "numpy", "scipy", "scipy.signal",
    "toml", "pathlib", "sqlite3",
    "httpx", "websockets", "h2", "hpack", "hyperframe",
    "pydantic", "pydantic_core",
    "jiter", "rich", "cryptography",
    "core.modal.acq_backend",
    "core.modal.tdms_importer",
    "core.modal.frf_compute",
    "core.modal.iso7626_validator",
    "core.modal.ema_engine",
    "core.modal.oma_engine",
    "core.modal.geometry_3d",
    "core.modal.ui_components",
    "auth_planta",
    "sync_uploader",
    # Licencias (FASE D v3.31.215)
    "license_manager",
    "jwt",                       # PyJWT runtime
    "jwt.algorithms",
    "jwt.exceptions",
    "cryptography.hazmat.primitives.asymmetric.rsa",
    "cryptography.hazmat.primitives.asymmetric.padding",
    "cryptography.hazmat.primitives.serialization",
    "cryptography.hazmat.primitives.hashes",
    "cryptography.hazmat.backends.openssl",
    # Auto-updater (FASE F v3.31.216)
    "updater",
    "urllib.request", "urllib.error", "urllib.parse",
    # Tray icon (FASE L v3.31.233 — sin pantalla negra)
    "pystray", "pystray._base", "pystray._win32",
    "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont",
]

# ============================================================
# Excludes — librerías pesadas que NO necesitamos
# ============================================================
excludes = [
    "tkinter",       # GUI vieja, no la usamos
    "matplotlib",    # solo plotly
    "PyQt5", "PyQt6", "PySide2", "PySide6",
    "IPython", "jupyter", "notebook",
    "test", "tests", "unittest",
    "sphinx", "pytest",
    "anthropic",     # solo lo usa Watermelon Cloud, no planta
    "openai",
    "reportlab",     # PDF reports — los hace Cloud, no planta
    "kaleido",       # PNG export — Cloud, no planta
    "imageio", "imageio_ffmpeg",  # MP4 export — Cloud, no planta
    "imageio_ffmpeg.binaries",
]

# ============================================================
# Análisis + bundle
# ============================================================
a = Analysis(
    [str(_SPEC_DIR / "launcher.py")],
    pathex=[str(_REPO_ROOT), str(_PLANTA_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="WatermelonPlanta",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,           # comprime el .exe (~30% reducción)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,      # FASE L v3.31.233: sin ventana negra cmd.exe
                        # Los logs van a data\logs\watermelon-YYYYMMDD.log
                        # El user controla la app desde el tray icon
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Icon custom — solo se aplica si existe el .ico, sino usa el default de Windows
    icon=(
        str(_SPEC_DIR / "assets" / "watermelon.ico")
        if (_SPEC_DIR / "assets" / "watermelon.ico").exists()
        else None
    ),
)
