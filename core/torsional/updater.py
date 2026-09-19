"""
core/torsional/updater.py — Auto-actualizador del módulo NATIVO (Watermelon Torsional)
==============================================================================

Cuando el PC de campo se conecta a internet, la app consulta los Releases de
GitHub (tags `torsional-v*`), compara con su versión local y, si hay una más nueva,
avisa y permite actualizar de una: descarga el instalador `WatermelonTorsional-Setup.exe`
del Release y lo lanza (Inno Setup actualiza la instalación existente). Sin red
o sin release nuevo: no hace nada (silencioso, offline-first).

No requiere dependencias extra (usa urllib de la stdlib).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.request
from typing import Any, Dict, Optional

REPO = os.environ.get("WM_TORSIONAL_REPO", "Ewdeshernandez/watermelon-system")
_UA = {"User-Agent": "WatermelonTorsional-Updater", "Accept": "application/vnd.github+json"}


def _ssl_context():
    """Contexto SSL con CA de certifi. En el .exe empaquetado (PyInstaller) el
    almacén de certificados del sistema no siempre está disponible → sin esto
    falla con CERTIFICATE_VERIFY_FAILED. certifi trae su propio cacert.pem."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        try:
            return ssl.create_default_context()
        except Exception:  # noqa: BLE001
            return None


def _parse_ver(s: str):
    nums = re.findall(r"\d+", s or "")
    nums = [int(x) for x in nums[:3]]
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def check_for_update(current_version: str, timeout: float = 6.0) -> Optional[Dict[str, Any]]:
    """Devuelve info del release más nuevo (`torsional-v*`) si supera a current_version,
    o None (sin red / sin actualización / error). No lanza excepciones."""
    try:
        # per_page=100 (máximo de la API): los releases de las apps se intercalan,
        # así que con 30 el torsional-v* más nuevo podía caer fuera de la primera página.
        url = f"https://api.github.com/repos/{REPO}/releases?per_page=100"
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
            rels = json.load(r)
    except Exception:  # noqa: BLE001
        return None
    best = None
    for rel in rels or []:
        tag = rel.get("tag_name", "") or ""
        if not tag.startswith("torsional-v") or rel.get("draft"):
            continue
        v = _parse_ver(tag)
        if best is None or v > best[0]:
            best = (v, rel)
    if not best:
        return None
    v, rel = best
    if v <= _parse_ver(current_version):
        return None
    setup_url = zip_url = None
    for a in rel.get("assets", []) or []:
        n = (a.get("name", "") or "")
        if n.lower().endswith("setup.exe"):
            setup_url = a.get("browser_download_url")
        elif n.lower().endswith(".zip"):
            zip_url = a.get("browser_download_url")
    return {"version": ".".join(str(x) for x in v), "tag": rel.get("tag_name", ""),
            "notes": (rel.get("body", "") or "")[:2000],
            "published": (rel.get("published_at", "") or "")[:10],
            "setup_url": setup_url, "zip_url": zip_url, "html_url": rel.get("html_url", "")}


def diagnose(current_version: str, timeout: float = 6.0):
    """Chequeo MANUAL con diagnóstico legible (para el botón del campo). Devuelve
    (info_or_None, mensaje). Sirve para ver por qué no aparece la actualización
    (red bloqueada, proxy, al día, etc.)."""
    url = f"https://api.github.com/repos/{REPO}/releases?per_page=100"
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
            code = r.getcode()
            rels = json.load(r)
    except Exception as e:  # noqa: BLE001
        return None, (f"Could not reach the update server.\n{type(e).__name__}: {e}\n\n"
                      f"URL: {url}\nDoes this PC's network block the update server or require a proxy?")
    tags = [x.get("tag_name") for x in (rels or []) if str(x.get("tag_name", "")).startswith("torsional-v")]
    info = check_for_update(current_version, timeout)
    if info:
        return info, (f"A newer version is available: v{info['version']} (you have v{current_version}).\n"
                      f"HTTP {code}.")
    return None, (f"You are up to date (v{current_version}).\nHTTP {code}.")


def download_file(url: str, dest: Optional[str] = None, timeout: float = 60.0,
                  on_progress=None, retries: int = 6) -> Optional[str]:
    """Descarga `url` a `dest` (o a temp) de forma RESISTENTE: si la red de campo
    corta a mitad, REANUDA desde donde iba (HTTP Range) y REINTENTA varias veces.
    Devuelve la ruta local (verificada por tamaño) o None."""
    if not url:
        return None
    dest = dest or os.path.join(tempfile.gettempdir(), url.split("/")[-1].split("?")[0])
    part = dest + ".part"
    total = 0
    for attempt in range(max(1, retries)):
        got = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"User-Agent": "WatermelonTorsional-Updater"}
        if got > 0:
            headers["Range"] = f"bytes={got}-"          # reanuda desde lo ya bajado
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
                code = r.getcode()
                if got > 0 and code != 206:             # el server ignoró el Range → empezar de cero
                    got = 0
                clen = int(r.headers.get("Content-Length", 0) or 0)
                if code == 206:                          # Range: total = fin del Content-Range
                    cr = r.headers.get("Content-Range", "")
                    if "/" in cr:
                        try: total = int(cr.rsplit("/", 1)[1])
                        except Exception: total = got + clen  # noqa: BLE001
                else:
                    total = clen
                mode = "ab" if got > 0 else "wb"
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(262144)
                        if not chunk:
                            break
                        f.write(chunk); got += len(chunk)
                        if on_progress and total:
                            try: on_progress(min(got / total, 1.0))
                            except Exception: pass  # noqa: BLE001
            if total and os.path.getsize(part) < total:  # incompleto → reintenta (reanuda)
                continue
            os.replace(part, dest)                        # completo → nombre final
            return dest
        except Exception:  # noqa: BLE001 — corte de red / timeout → reintenta reanudando
            continue
    return None


def launch_installer(setup_path: str) -> bool:
    """Lanza el instalador descargado (Inno Setup actualiza sobre la instalación
    existente). La app debe cerrarse tras llamar esto. True si arrancó."""
    try:
        import subprocess
        if os.name == "nt":
            os.startfile(setup_path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen([setup_path])
        return True
    except Exception:  # noqa: BLE001
        return False
