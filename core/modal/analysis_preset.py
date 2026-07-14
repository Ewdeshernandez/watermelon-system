"""
core/modal/analysis_preset.py — Presets COMPLETOS de análisis modal
====================================================================

Guarda/recarga una configuración COMPLETA del ensayo como un archivo
reutilizable de un clic: geometría (bloques + sensores + DOF) + parámetros de
adquisición (fs, duración, f_min/f_max, prominencia, RPM, promedios) + el mapa
de canales de la maleta. Antes solo se podía guardar la geometría; el usuario
tenía que reconfigurar sensores y parámetros en cada ensayo (feedback de campo
v3.31.440).

Un preset es un JSON con:
    {
      "name": "...",
      "geometry": "<ModalGeometry.to_json()>"  (o ""),
      "params": { <clave de session_state>: <valor JSON> , ... }
    }

Persiste en el mismo disco durable que las geometrías (WM_PERSIST_DIR en
Render, junto al .exe en Planta, relativo en dev).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _presets_dir() -> Path:
    _pd = os.environ.get("WM_PERSIST_DIR")
    if _pd:
        return Path(_pd) / "modal" / "presets"
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "data" / "modal" / "presets"
    return Path("data/modal/presets")


# Claves de session_state que forman la "configuración del ensayo" (parámetros
# de adquisición + análisis + grillas de canales OMA/EMA). Todas serializables.
PRESET_PARAM_KEYS: List[str] = [
    "ni_mode", "ni_fs", "ni_fn_low_oma", "ni_dur_oma", "ni_dur_ema",
    "ni_avg_ema", "ni_channel_grid_oma", "ni_channel_grid_ema",
    "oma_fmin", "oma_fmax", "oma_prom", "oma_rpm",
    "ema_fmin", "ema_fmax", "ema_prom", "ema_dist",
    "ti_fmin", "ti_fmax",
]


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_"
                   for c in (name or "").strip().lower())[:60].strip("_")


def _path_for(name: str) -> Path:
    return _presets_dir() / f"{_slug(name)}.json"


def save_preset(name: str, geometry_json: str = "",
                params: Optional[Dict[str, Any]] = None) -> Path:
    """Guarda un preset completo. Devuelve el path. Lanza si name vacío."""
    if not (name or "").strip():
        raise ValueError("El nombre del preset no puede estar vacío.")
    d = _presets_dir()
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name.strip(),
        "geometry": geometry_json or "",
        "params": params or {},
    }
    p = _path_for(name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def load_preset(name: str) -> Optional[Dict[str, Any]]:
    """Devuelve el dict del preset {name, geometry, params} o None."""
    p = _path_for(name)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("geometry", "")
            data.setdefault("params", {})
            return data
    except Exception:  # noqa: BLE001
        pass
    return None


def list_presets() -> List[str]:
    """Nombres legibles de los presets guardados (ordenados)."""
    d = _presets_dir()
    if not d.exists():
        return []
    names = []
    for p in sorted(d.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                nm = (json.load(f) or {}).get("name") or p.stem
        except Exception:  # noqa: BLE001
            nm = p.stem
        names.append(nm)
    return names


def delete_preset(name: str) -> bool:
    p = _path_for(name)
    if p.exists():
        try:
            p.unlink()
            return True
        except Exception:  # noqa: BLE001
            return False
    return False
