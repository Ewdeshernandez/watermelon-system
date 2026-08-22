"""
core.reports_ext.drafts — Autoguardado + borradores para los módulos nuevos
===========================================================================

Persistencia LIGERA y REUTILIZABLE del estado de un formulario (Calibración,
reportes de campo, etc.) para que NO se pierda el trabajo si:
  - la página se cae / da error,
  - se corta la red o se reconecta la sesión,
  - se hace un redeploy / reinicio del servidor.

Diseño:
  - Un archivo JSON por borrador, por usuario y por módulo, en el disco
    persistente (WM_PERSIST_DIR, ej. /var/data) — sobrevive redeploys.
  - Slot especial "_autosave" para la recuperación automática tras caída.
  - Escritura ATÓMICA y TOLERANTE A DISCO LLENO (ENOSPC): si no hay espacio,
    NO crashea (devuelve False).
  - Se descartan bytes (imágenes) — no se serializan; el usuario re-sube fotos
    si hubo caída. Lo valioso (datos tecleados, lazos, resultados) sí se guarda.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

AUTOSAVE_SLOT = "_autosave"


def _root() -> Path:
    d = os.environ.get("WM_PERSIST_DIR") or os.path.join(
        os.path.expanduser("~"), ".watermelon_state")
    return Path(d)


def _user_slug() -> str:
    try:
        from core.auth import get_current_user
        u = get_current_user() or {}
        em = (u.get("email") or "anon").strip().lower()
    except Exception:
        em = "anon"
    return re.sub(r"[^a-z0-9]+", "_", em).strip("_") or "anon"


def _module_dir(module: str) -> Path:
    d = _root() / "form_drafts" / _user_slug() / re.sub(r"[^a-z0-9_]+", "", module.lower())
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_") or "draft"


def _strip_bytes(obj: Any) -> Any:
    """Quita bytes/bytearray (imágenes) recursivamente — no van al borrador."""
    if isinstance(obj, dict):
        return {k: _strip_bytes(v) for k, v in obj.items()
                if not isinstance(v, (bytes, bytearray))}
    if isinstance(obj, (list, tuple)):
        return [_strip_bytes(v) for v in obj if not isinstance(v, (bytes, bytearray))]
    if isinstance(obj, (bytes, bytearray)):
        return None
    return obj


def _safe_write(path: Path, payload: Dict[str, Any]) -> bool:
    """Escritura atómica tolerante a ENOSPC. True si guardó."""
    tmp = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, default=str)
        os.replace(str(tmp), str(path))
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------
def save_draft(module: str, name: str, state: Dict[str, Any]) -> bool:
    payload = {"name": name, "saved_at": time.time(), "state": _strip_bytes(state)}
    return _safe_write(_module_dir(module) / f"{_slug(name)}.json", payload)


def autosave(module: str, state: Dict[str, Any]) -> bool:
    return save_draft(module, AUTOSAVE_SLOT, state)


def load_draft(module: str, name: str) -> Optional[Dict[str, Any]]:
    d = _read(_module_dir(module) / f"{_slug(name)}.json")
    return d.get("state") if d else None


def load_autosave(module: str) -> Optional[Dict[str, Any]]:
    return load_draft(module, AUTOSAVE_SLOT)


def list_drafts(module: str) -> List[str]:
    out: List[str] = []
    for p in _module_dir(module).glob("*.json"):
        if p.stem == _slug(AUTOSAVE_SLOT):
            continue
        d = _read(p)
        if d:
            out.append(d.get("name") or p.stem)
    return sorted(out)


def delete_draft(module: str, name: str) -> bool:
    try:
        (_module_dir(module) / f"{_slug(name)}.json").unlink()
        return True
    except Exception:
        return False


__all__ = [
    "save_draft", "autosave", "load_draft", "load_autosave", "list_drafts",
    "delete_draft", "AUTOSAVE_SLOT",
]
