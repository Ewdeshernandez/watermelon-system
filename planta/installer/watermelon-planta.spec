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
    collect_data_files, collect_submodules, copy_metadata, collect_all,
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

# ============================================================
# secrets.toml REAL con las claves PÚBLICAS del proyecto (url + anon_key).
# En CI lo escribe el step "Generar secrets.toml" ANTES del build a partir de
# los GH secrets SUPABASE_URL + SUPABASE_ANON_KEY. Empaquetarlo es lo que hace
# funcionar el login OTP en el .exe (sin él, no hay env ni st.secrets y el
# login truena con "SUPABASE_URL/ANON_KEY no configurados").
# NUNCA debe contener el service_key (solo url + anon_key, ambos públicos).
# Si no existe (dev local sin claves), el build sigue igual y el login usa
# env/st.secrets como antes.
# ============================================================
_REAL_SECRETS = _PLANTA_DIR / ".streamlit" / "secrets.toml"
if _REAL_SECRETS.exists():
    _txt = _REAL_SECRETS.read_text(encoding="utf-8", errors="ignore")
    # Chequear SOLO líneas que NO son comentarios, para no dar falso positivo
    # cuando un comentario menciona la palabra 'service_key'/'service_role'.
    # Detectamos una asignación real de esas claves.
    import re as _re
    _body = "\n".join(
        _l for _l in _txt.splitlines() if not _l.lstrip().startswith("#")
    )
    if _re.search(r"service_(role|key)", _body):
        raise SystemExit(
            "\n\n[watermelon-planta.spec] ABORTANDO BUILD: "
            "planta/.streamlit/secrets.toml contiene 'service_key'/'service_role'. "
            "El .exe del cliente SOLO puede llevar url + anon_key (públicos). "
            "Quitá el service_key de ese archivo antes de buildear.\n"
        )
    datas.append((str(_REAL_SECRETS), "planta/.streamlit"))
    print("[watermelon-planta.spec] OK incluido secrets.toml real (url + anon_key).")
else:
    print("[watermelon-planta.spec] AVISO: no hay secrets.toml real; el .exe "
          "dependerá de env/st.secrets para el login (OTP no funcionará en campo).")

# Core modal (reusable desde planta/)
_CORE_MODAL = _REPO_ROOT / "core" / "modal"
if _CORE_MODAL.exists():
    for py in _CORE_MODAL.glob("*.py"):
        datas.append((str(py), "core/modal"))

# Streamlit static files (HTML/JS del frontend)
datas += collect_data_files("streamlit")

# ============================================================
# v3.31.387 — FIX "Unable to preload CSS for .../DataFrame.*.css".
# collect_data_files("streamlit") NO incluye de forma confiable todo el
# árbol del frontend de Streamlit (carpeta static/, con los chunks por
# componente como static/css/DataFrame.*.css que usa st.data_editor /
# st.dataframe). Sin esos chunks, la grilla de canales (st.data_editor en
# 01_Captura_Modal.py) rompe con "Unable to preload CSS" en el .exe.
# Agregamos la carpeta static (y runtime) EXPLÍCITA. Aborta si no existe.
# ============================================================
import streamlit as _st_mod  # noqa: E402
_ST_DIR = Path(_st_mod.__file__).resolve().parent
# SOLO 'static' (frontend, son datos). NO bundlear 'runtime' como datos: es
# un paquete de CÓDIGO (ya va en el PYZ vía collect_submodules) y meterlo
# como datos sueltos puede romper el arranque del servidor Streamlit
# ("Streamlit server is not responding"). v3.31.388 revierte ese exceso.
_ST_STATIC = _ST_DIR / "static"
if not _ST_STATIC.exists():
    raise SystemExit(
        f"\n\n[watermelon-planta.spec] ABORTANDO BUILD: no se encontró "
        f"'streamlit/static' en el entorno de build ({_ST_STATIC}).\n"
    )
datas.append((str(_ST_STATIC), "streamlit/static"))
print("[watermelon-planta.spec] OK incluida carpeta 'streamlit/static'.")

# Plotly static (matplotlib data + plotly.js)
datas += collect_data_files("plotly")

# Importer metadata (necesario para que Streamlit detecte sus extensiones)
datas += copy_metadata("streamlit")
datas += copy_metadata("supabase")

# ============================================================
# v3.31.385 — FIX DE RAÍZ del bug recurrente "Componente de captura
# pendiente". Causa: el bloque anterior usaba try/except: pass y
# collect_submodules, que DEVUELVEN VACÍO EN SILENCIO si el paquete no
# importa/no tiene metadata al congelar → el .exe salía SIN nidaqmx y el
# build NO fallaba. En campo, `from nidaqmx.system import System` reventaba
# (nidaqmx lee su versión vía importlib.metadata al importar → necesita su
# dist-info empaquetado) y la app mostraba "componente de captura pendiente".
#
# Ahora se usa collect_all() (submódulos + datas + BINARIOS/DLL + metadata
# en una sola llamada) y la colección es OBLIGATORIA: si recoge 0 módulos,
# el build ABORTA RUIDOSAMENTE en vez de generar un .exe roto.
# ============================================================
binaries = []
_acq_hiddenimports = []
for _pkg in ("nidaqmx", "nptdms"):
    _d, _b, _h = collect_all(_pkg)
    if not _h and not _d:
        raise SystemExit(
            f"\n\n[watermelon-planta.spec] ABORTANDO BUILD: el paquete '{_pkg}' "
            f"no se pudo colectar (0 módulos / 0 datas).\n"
            f"Causa típica: '{_pkg}' no está instalado en el entorno de build, "
            f"o no importa al congelar.\n"
            f"Solución: en la máquina/CI de build ejecutá  "
            f"`pip install -r planta/requirements-planta.txt`  y verificá  "
            f"`python -c \"import {_pkg}\"`  ANTES de buildear.\n"
            f"Sin esto el .exe sale sin el componente de captura "
            f"(bug 'componente de captura pendiente').\n"
        )
    datas += _d
    binaries += _b
    _acq_hiddenimports += _h
    print(f"[watermelon-planta.spec] OK colectado '{_pkg}': "
          f"{len(_h)} submódulos, {len(_d)} datas, {len(_b)} binarios.")

# ============================================================
# v3.31.386 — FIX REAL del "componente de captura pendiente".
# Causa: nidaqmx/__init__.py hace  `__version__ = version("nidaqmx")`
# (importlib.metadata) AL IMPORTAR. collect_all incluye la metadata solo
# "best-effort"; si la dist-info NO quedó en el bundle, `import nidaqmx`
# lanza PackageNotFoundError y muere → la app muestra "componente
# pendiente" aunque los .py SÍ estén empaquetados (caso 385: archivos OK
# pero import roto). Forzamos la metadata explícita + recursiva (también
# la de las dependencias que leen su versión). OBLIGATORIO: si falla,
# aborta el build en vez de generar un .exe roto.
# ============================================================
for _pkg in ("nidaqmx", "nptdms"):
    try:
        _md = copy_metadata(_pkg, recursive=True)
    except Exception as _exc:
        raise SystemExit(
            f"\n\n[watermelon-planta.spec] ABORTANDO BUILD: no se pudo copiar "
            f"la metadata de '{_pkg}' ({type(_exc).__name__}: {_exc}).\n"
            f"Sin la metadata, `import {_pkg}` revienta en runtime con "
            f"PackageNotFoundError → bug 'componente de captura pendiente'.\n"
        )
    if not _md:
        raise SystemExit(
            f"\n\n[watermelon-planta.spec] ABORTANDO BUILD: copy_metadata('{_pkg}') "
            f"devolvió VACÍO. La dist-info no está instalada en el entorno de build. "
            f"Ejecutá `pip install -r planta/requirements-planta.txt` antes de buildear.\n"
        )
    datas += _md
    print(f"[watermelon-planta.spec] OK metadata '{_pkg}': {len(_md)} dist-info.")

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

# v3.31.385 — submódulos de adquisición/TDMS ya colectados arriba con
# collect_all() (obligatorio, aborta si falta). Solo los sumamos aquí.
hiddenimports += _acq_hiddenimports

hiddenimports += [
    "numpy", "scipy", "scipy.signal",
    # v3.31.386 — deps DURAS de nidaqmx que se importan a nivel módulo al
    # cargar System (errors.py→deprecation, system.py→hightime, tzlocal).
    # Si alguna falta, `from nidaqmx.system import System` revienta.
    "hightime", "tzlocal", "deprecation",
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
    binaries=binaries,
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
