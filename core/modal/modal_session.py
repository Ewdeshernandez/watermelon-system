"""
core/modal/modal_session.py — Persistencia de la selección de sesión Modal
==========================================================================

Guarda a disco QUÉ activo (o metadata ad-hoc) estaba seleccionado en el módulo
Modal, para restaurarlo automáticamente cuando la app se reinicia, se actualiza
o el usuario vuelve al Tab Setup. Antes, la selección vivía solo en
session_state y se perdía en cada recarga → había que re-seleccionar el activo
desde Machinery Library una y otra vez (feedback de campo v3.31.438).

Persiste en el mismo disco durable que las geometrías (WM_PERSIST_DIR en Render,
junto al .exe en Planta, relativo en dev).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _modal_dir() -> Path:
    _pd = os.environ.get("WM_PERSIST_DIR")
    if _pd:
        return Path(_pd) / "modal"
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "data" / "modal"
    return Path("data/modal")


_SESSION_FILE = "last_session.json"


def save_last_selection(setup_mode: str = "",
                        asset_id: str = "",
                        adhoc: Optional[Dict[str, Any]] = None) -> None:
    """Persiste la última selección (modo + activo + metadata ad-hoc).
    Silencioso — nunca rompe la UI."""
    try:
        d = _modal_dir()
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "setup_mode": setup_mode or "",
            "asset_id": asset_id or "",
            "adhoc": adhoc or {},
        }
        with open(d / _SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001
        pass


def load_last_selection() -> Dict[str, Any]:
    """Devuelve {'setup_mode', 'asset_id', 'adhoc'} o dict vacío si no hay."""
    try:
        p = _modal_dir() / _SESSION_FILE
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001
        pass
    return {}
