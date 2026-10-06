"""
core/efficiency/updater.py — Auto-actualizador del módulo NATIVO (Watermelon Efficiency)
=========================================================================================

Consulta los Releases de GitHub (tags `efficiency-v*`), compara con la versión
local y, si hay una más nueva, descarga el instalador `WatermelonEfficiency-Setup.exe`
y lo lanza (Inno Setup actualiza sobre la instalación existente). Resistente a
cortes de red (HTTP Range + reintentos). Offline-first (silencioso sin red).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.request
from typing import Any, Dict, Optional

REPO = os.environ.get("WM_TORSIONAL_REPO", "Ewdeshernandez/watermelon-system")
_UA = {"User-Agent": "WatermelonEfficiency-Updater", "Accept": "application/vnd.github+json"}


def _ssl_context():
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
    """Info del release más nuevo (`efficiency-v*`) si supera a current_version, o None."""
    try:
        url = f"https://api.github.com/repos/{REPO}/releases?per_page=100"
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
            rels = json.load(r)
    except Exception:  # noqa: BLE001
        return None
    best = None
    for rel in rels or []:
        tag = rel.get("tag_name", "") or ""
        if not tag.startswith("efficiency-v") or rel.get("draft"):
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
    """Chequeo MANUAL con diagnóstico legible (botón del campo)."""
    url = f"https://api.github.com/repos/{REPO}/releases?per_page=100"
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
            code = r.getcode()
            rels = json.load(r)
    except Exception as e:  # noqa: BLE001
        return None, (f"Could not reach the update server.\n{type(e).__name__}: {e}\n\n"
                      f"URL: {url}\nDoes this PC's network block the update server or require a proxy?")
    info = check_for_update(current_version, timeout)
    if info:
        return info, (f"A newer version is available: v{info['version']} (you have v{current_version}).\n"
                      f"HTTP {code}.")
    return None, (f"You are up to date (v{current_version}).\nHTTP {code}.")


def download_file(url: str, dest: Optional[str] = None, timeout: float = 60.0,
                  on_progress=None, retries: int = 6) -> Optional[str]:
    """Descarga `url` de forma RESISTENTE (reanuda con HTTP Range + reintenta)."""
    if not url:
        return None
    dest = dest or os.path.join(tempfile.gettempdir(), url.split("/")[-1].split("?")[0])
    part = dest + ".part"
    total = 0
    for attempt in range(max(1, retries)):
        got = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"User-Agent": "WatermelonEfficiency-Updater"}
        if got > 0:
            headers["Range"] = f"bytes={got}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
                code = r.getcode()
                if got > 0 and code != 206:
                    got = 0
                clen = int(r.headers.get("Content-Length", 0) or 0)
                if code == 206:
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
            if total and os.path.getsize(part) < total:
                continue
            os.replace(part, dest)
            return dest
        except Exception:  # noqa: BLE001
            continue
    return None


def launch_installer(setup_path: str) -> bool:
    """Lanza el instalador descargado. La app debe cerrarse tras llamar esto."""
    try:
        import subprocess
        if os.name == "nt":
            os.startfile(setup_path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen([setup_path])
        return True
    except Exception:  # noqa: BLE001
        return False
