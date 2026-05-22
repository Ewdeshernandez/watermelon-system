"""
planta/installer/launcher.py — Entry point para el .exe empaquetado
=====================================================================

Este script es el entry point que PyInstaller convierte en .exe.
Su trabajo:
  1. Arrancar Streamlit programáticamente (sin ventana de consola visible)
  2. Crear un tray icon en la bandeja del sistema para Abrir/Salir
  3. Abrir el browser default del cliente apuntando a la app
  4. Logear todo a archivo (sin consola) para diagnóstico

Estructura:
  · Sin ventana negra cmd.exe (console=False en el spec)
  · Tray icon con menú "Abrir Watermelon Planta" / "Salir"
  · Logs en data\logs\watermelon-YYYYMMDD.log
  · MessageBox de Windows si hay error fatal al boot
"""
from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path


# ============================================================================
# HELPERS
# ============================================================================

def _is_frozen() -> bool:
    """True si corremos dentro del .exe de PyInstaller, False si dev."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def _resource_base() -> Path:
    """Base path para encontrar recursos (bundle o dev)."""
    if _is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2]


def _get_external_data_dir() -> Path:
    """
    Directorio data/ externo al .exe (writable por el usuario).
    En bundle: al lado del .exe (C:\\Program Files\\WatermelonPlanta\\data\\)
    En dev: planta/data/
    """
    if _is_frozen():
        return Path(sys.executable).parent / "data"
    return _resource_base() / "planta" / "data"


def _find_app_planta_py() -> Path:
    """Encuentra app_planta.py en el bundle o en el repo."""
    base = _resource_base()
    candidates = [
        base / "planta" / "app_planta.py",
        base / "app_planta.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"app_planta.py no encontrado en {base}. "
        f"Reinstala Watermelon Planta — el archivo está corrupto."
    )


def _find_free_port(start: int = 8501) -> int:
    """Si 8501 está ocupado, busca uno libre."""
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def _setup_logging(data_dir: Path) -> Path:
    """Configura logging a archivo (no console). Retorna el path del log."""
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"watermelon-{datetime.now():%Y%m%d}.log"

    # Configurar root logger a archivo
    handler = logging.FileHandler(str(log_file), encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    # Redirigir stdout/stderr a archivo también (PyInstaller console=False
    # los hace None y eso rompe Streamlit que asume que stdout existe)
    sys.stdout = open(log_file, "a", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout

    return log_file


def _show_error_dialog(title: str, msg: str) -> None:
    """Muestra un MessageBox de Windows (solo Windows)."""
    try:
        import ctypes
        # MB_ICONERROR (0x10) | MB_OK (0x0) = 0x10
        ctypes.windll.user32.MessageBoxW(0, msg, title, 0x10)
    except Exception:  # noqa: BLE001
        pass  # no es Windows o falló — el log queda como respaldo


# ============================================================================
# BROWSER OPENER
# ============================================================================

def _open_browser_when_ready(port: int, max_wait_s: int = 30) -> None:
    """Espera a que el server esté listo y abre el browser default."""
    url = f"http://127.0.0.1:{port}"
    start = time.time()
    while time.time() - start < max_wait_s:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(("127.0.0.1", port))
                webbrowser.open(url)
                logging.info(f"Browser abierto en {url}")
                return
            except OSError:
                time.sleep(0.5)
    logging.warning(f"Server tardó >{max_wait_s}s en estar listo")
    webbrowser.open(url)  # abrir igual


# ============================================================================
# TRAY ICON
# ============================================================================

def _create_tray_icon(port: int):
    """
    Crea el ícono en la bandeja del sistema con menú Abrir / Salir.
    Si pystray no está disponible, retorna None (la app sigue corriendo
    pero sin tray icon — el user debe cerrar via Task Manager).
    """
    try:
        import pystray
        from PIL import Image
    except ImportError as e:
        logging.warning(f"pystray no disponible: {e} — sin tray icon")
        return None

    # Buscar el .ico embebido
    icon_path = _resource_base() / "planta" / "installer" / "assets" / "watermelon.ico"
    if not icon_path.exists():
        # Fallback: buscar en data/ (improbable pero seguro)
        icon_path = _resource_base() / "watermelon.ico"
    if not icon_path.exists():
        logging.warning("watermelon.ico no encontrado — tray icon usará default")
        # Pillow puede crear una imagen blanca como fallback
        image = Image.new("RGBA", (64, 64), (15, 118, 110, 255))
    else:
        try:
            image = Image.open(str(icon_path))
        except Exception as e:  # noqa: BLE001
            logging.warning(f"No se pudo abrir {icon_path}: {e}")
            image = Image.new("RGBA", (64, 64), (15, 118, 110, 255))

    url = f"http://127.0.0.1:{port}"

    def on_open(icon, item):
        webbrowser.open(url)
        logging.info(f"User abrió ventana desde tray icon")

    def on_quit(icon, item):
        logging.info("User cerró Watermelon Planta desde tray icon")
        icon.stop()
        # Forzar cierre del proceso (Streamlit no termina solo)
        os._exit(0)

    menu = pystray.Menu(
        pystray.MenuItem("Abrir Watermelon Planta", on_open, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Salir", on_quit),
    )

    icon = pystray.Icon(
        "watermelon_planta",
        image,
        "Watermelon Planta Edition",
        menu,
    )

    # Correr el tray icon en thread separado (no bloquea Streamlit)
    threading.Thread(target=icon.run, daemon=True).start()
    logging.info("Tray icon creado y corriendo")
    return icon


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    """Entry point del .exe."""
    # 1. Setup logging primero (sin consola, todo a archivo)
    data_dir = _get_external_data_dir()
    try:
        log_file = _setup_logging(data_dir)
    except Exception as e:  # noqa: BLE001
        # Si no podemos logear, mostrar error y salir
        _show_error_dialog(
            "Watermelon Planta — Error",
            f"No se pudo crear el directorio de logs.\n\n"
            f"Verifica que tengas permisos de escritura en:\n"
            f"{data_dir}\n\n"
            f"Error: {e}",
        )
        return 1

    logging.info("=" * 60)
    logging.info("WATERMELON PLANTA EDITION — iniciando")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Frozen: {_is_frozen()}")
    logging.info(f"Data dir: {data_dir}")
    logging.info("=" * 60)

    # 2. Localizar app_planta.py
    try:
        app_path = _find_app_planta_py()
    except FileNotFoundError as e:
        logging.exception("No se encontró app_planta.py")
        _show_error_dialog(
            "Watermelon Planta — Instalación corrupta",
            f"{e}\n\nReinstala Watermelon Planta desde el installer original.",
        )
        return 1

    logging.info(f"app_planta.py: {app_path}")

    # 3. Port libre
    port = _find_free_port(8501)
    logging.info(f"Port: {port}")

    # 4. Tray icon (en background)
    tray = _create_tray_icon(port)

    # 5. Browser opener (en background, espera a que server esté listo)
    threading.Thread(
        target=_open_browser_when_ready,
        args=(port,),
        daemon=True,
    ).start()

    # 6. Cambiar cwd para que imports relativos funcionen
    os.chdir(app_path.parent)

    # 7. Arrancar Streamlit (BLOQUEA hasta que se cierre)
    try:
        from streamlit.web import bootstrap
        from streamlit import config as st_config

        st_config.set_option("server.headless", True)
        st_config.set_option("server.port", port)
        st_config.set_option("server.address", "127.0.0.1")
        st_config.set_option("server.enableCORS", False)
        st_config.set_option("server.enableXsrfProtection", False)
        st_config.set_option("browser.gatherUsageStats", False)
        st_config.set_option("theme.primaryColor", "#0f766e")
        st_config.set_option("theme.backgroundColor", "#ffffff")
        st_config.set_option("global.developmentMode", False)

        logging.info("Lanzando Streamlit bootstrap...")
        bootstrap.run(
            str(app_path),
            is_hello=False,
            args=[],
            flag_options={},
        )
    except SystemExit:
        # Streamlit normalmente termina con SystemExit cuando lo matan
        logging.info("Streamlit terminó normalmente")
        return 0
    except Exception as e:  # noqa: BLE001
        logging.exception("Error fatal arrancando Streamlit")
        tb = traceback.format_exc()
        _show_error_dialog(
            "Watermelon Planta — Error al iniciar",
            f"No se pudo arrancar la aplicación.\n\n"
            f"Error: {e}\n\n"
            f"Detalles completos en el log:\n"
            f"{log_file}\n\n"
            f"Si el problema persiste, contacta a SIGA:\n"
            f"ehernandez@sigasas.com",
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
