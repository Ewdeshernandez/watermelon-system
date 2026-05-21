"""
planta/installer/launcher.py — Entry point para el .exe empaquetado
=====================================================================

Este script es el entry point que PyInstaller convierte en .exe.
Su trabajo: arrancar Streamlit programáticamente (sin terminal visible)
y abrir el browser default del cliente apuntando a localhost:8501.

Por qué no usar streamlit run app_planta.py directamente:
PyInstaller empaqueta Python + libs + nuestro código en un ejecutable
auto-extraíble. Streamlit normalmente se invoca via CLI (`streamlit run`)
que necesita Python en el PATH del sistema — eso NO existe en el .exe.

Solución: usar streamlit.web.bootstrap.run() — la API interna de Streamlit
para arrancar el servidor desde código Python.

Estructura del .exe:
Cuando PyInstaller arma el .exe, todos los archivos data se extraen a un
temp folder (sys._MEIPASS). El launcher tiene que:
1. Detectar si estamos corriendo desde el .exe (sys._MEIPASS exists)
2. Construir la ruta correcta a app_planta.py (extraído en MEIPASS/planta/)
3. Arrancar Streamlit apuntando a esa ruta
4. Abrir browser apuntando a http://localhost:8501
"""
from __future__ import annotations

import os
import sys
import time
import socket
import threading
import webbrowser
from pathlib import Path


def _is_frozen() -> bool:
    """True si corremos dentro del .exe de PyInstaller, False si dev."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def _resource_base() -> Path:
    """
    Base path para encontrar recursos.
    - En .exe: sys._MEIPASS (temp folder donde PyInstaller extrae)
    - En dev: directorio del repo root (parent del planta/)
    """
    if _is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2]


def _find_app_planta_py() -> Path:
    """Encuentra app_planta.py en el bundle o en el repo."""
    base = _resource_base()
    candidates = [
        base / "planta" / "app_planta.py",  # bundle structure
        base / "app_planta.py",              # si planta es root del bundle
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No encontré app_planta.py en {base}. "
        f"Reinstala Watermelon Planta — el archivo se corrompió."
    )


def _find_free_port(start: int = 8501) -> int:
    """Si 8501 está ocupado, busca uno libre desde 8501."""
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start  # fallback — Streamlit dirá error y user verá


def _open_browser_when_ready(port: int, max_wait_s: int = 15) -> None:
    """Espera a que el server esté listo y abre el browser default."""
    url = f"http://127.0.0.1:{port}"
    start = time.time()
    while time.time() - start < max_wait_s:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(("127.0.0.1", port))
                webbrowser.open(url)
                return
            except OSError:
                time.sleep(0.5)
    # Si el server no arrancó en max_wait_s, igual abrir el browser
    # — al user le aparece el error de "site can't be reached" pero
    # al menos sabe a qué URL ir cuando arranque
    webbrowser.open(url)


def main() -> int:
    """Entry point del .exe."""
    # Forzar UTF-8 en stdout/stderr (Windows console por default es cp1252)
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    print("=" * 60, flush=True)
    print("  WATERMELON PLANTA EDITION", flush=True)
    print("  Iniciando captura modal offline...", flush=True)
    print("=" * 60, flush=True)

    # Localizar app_planta.py
    try:
        app_path = _find_app_planta_py()
    except FileNotFoundError as exc:
        print(f"\n✗ ERROR FATAL: {exc}", file=sys.stderr, flush=True)
        input("\nPresiona Enter para cerrar...")
        return 1

    print(f"  App path: {app_path}", flush=True)

    # Port libre
    port = _find_free_port(8501)
    print(f"  Port:     {port}", flush=True)
    print(f"  URL:      http://localhost:{port}", flush=True)
    print("", flush=True)
    print("  Tu browser default va a abrir en unos segundos.", flush=True)
    print("  Para cerrar la app, cierra esta ventana negra.", flush=True)
    print("=" * 60, flush=True)

    # Browser opener en background — espera a que server esté listo
    t = threading.Thread(target=_open_browser_when_ready, args=(port,),
                          daemon=True)
    t.start()

    # Cambiar cwd al directorio del app_planta para que los imports relativos
    # y los pages/ se encuentren correctamente
    os.chdir(app_path.parent)

    # Arrancar Streamlit programáticamente
    try:
        from streamlit.web import bootstrap
        from streamlit import config as st_config

        # Config Streamlit para modo desktop
        st_config.set_option("server.headless", True)
        st_config.set_option("server.port", port)
        st_config.set_option("server.address", "127.0.0.1")
        st_config.set_option("server.enableCORS", False)
        st_config.set_option("server.enableXsrfProtection", False)
        st_config.set_option("browser.gatherUsageStats", False)
        st_config.set_option("theme.primaryColor", "#0f766e")
        st_config.set_option("theme.backgroundColor", "#ffffff")
        st_config.set_option("global.developmentMode", False)

        bootstrap.run(
            str(app_path),
            is_hello=False,
            args=[],
            flag_options={},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n✗ ERROR arrancando Streamlit: {exc}",
              file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        input("\nPresiona Enter para cerrar...")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
