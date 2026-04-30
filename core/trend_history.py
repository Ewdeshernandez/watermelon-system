"""
core.trend_history
==================

Persistencia histórica de **CSVs completos** del módulo Trend, por
instancia (Ciclo 17.5).

A diferencia de Polar/Bode/SCL que snapshotean métricas derivadas
(amp, fase, eccentricity), Trend NECESITA preservar los CSV crudos
porque el valor del módulo es ver la **serie temporal completa** de
las mediciones a lo largo de meses/años, y eso requiere las muestras
originales — no solo agregados.

Cada "corrida" del Trend = un upload de N CSVs (vibración +
operacional). El sistema guarda los archivos enteros bajo la
instancia, y al volver a abrir el módulo el usuario puede:

  - Ver solo los CSVs de la corrida ACTUAL (default)
  - INCLUIR corridas históricas en el análisis → las series se
    concatenan cronológicamente, dando un trend largo de varios
    meses/años aunque cada corrida individual sea de pocas semanas.

Beneficio operativo: el ingeniero NO tiene que conservar los CSVs
viejos en su máquina. Cada vez que sube una corrida nueva, la
sistema la archiva. Para reportes anuales o post-mantenimiento el
sistema reconstruye el trend largo automáticamente.

Storage:

    {INSTANCES_DIR}/{instance_id}/trend_history/{corrida_id}/
        metadata.json
        files/
            {original_csv_name_1}
            {original_csv_name_2}
            ...

Metadata.json:
    {
      "corrida_id": "corrida_20260430_153022",
      "instance_id": "...",
      "timestamp": "ISO8601 cuando se guardó",
      "corrida_label": "etiqueta humana opcional",
      "notes": "observaciones",
      "n_files": 5,
      "files": ["a.csv", "b.csv", ...],
      "time_range": {
          "min": "ISO8601 muestra más vieja entre todos los CSVs",
          "max": "ISO8601 muestra más nueva"
      }
    }

Funciones públicas:

  - save_trend_corrida(instance_id, files, label, notes) -> corrida_id
  - list_trend_corridas(instance_id) -> list[dict] (sin payload, lite)
  - load_trend_corrida_files(instance_id, corrida_id) -> list[bytes]
        Cada elemento es (file_name, csv_bytes) para reparse.
  - delete_trend_corrida(instance_id, corrida_id) -> bool
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.instance_repository import INSTANCES_DIR


MAX_TREND_CORRIDAS_PER_INSTANCE = 36  # 3 años de mensual


# ============================================================
# PATHS
# ============================================================

def _trend_history_root(instance_id: str) -> Path:
    p = INSTANCES_DIR / instance_id / "trend_history"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _corrida_dir(instance_id: str, corrida_id: str) -> Path:
    return _trend_history_root(instance_id) / corrida_id


def _new_corrida_id() -> str:
    return "corrida_" + datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# SAVE / LOAD / LIST / DELETE
# ============================================================

def save_trend_corrida(
    instance_id: str,
    files: List[Tuple[str, bytes]],
    *,
    corrida_label: str = "",
    notes: str = "",
    detected_time_range: Optional[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = None,
) -> str:
    """
    Persiste una corrida (= conjunto de CSVs Trend) bajo la instancia.

    Args:
        instance_id: ID de la Asset Instance.
        files: lista de tuplas ``(file_name, csv_bytes)``. Cada tupla
            corresponde a un CSV (vibration trend o operational data).
        corrida_label: etiqueta humana opcional (default = ISO timestamp).
        notes: observaciones del usuario.
        detected_time_range: opcionalmente (min_ts, max_ts) detectado
            por el caller después de parsear los CSVs. Si no se pasa,
            queda en blanco — el caller puede actualizarlo después.

    Returns:
        corrida_id (string).
    """
    if not instance_id:
        raise ValueError("instance_id requerido")
    if not files:
        raise ValueError("No hay archivos para guardar")

    corrida_id = _new_corrida_id()
    corrida_path = _corrida_dir(instance_id, corrida_id)
    files_path = corrida_path / "files"
    files_path.mkdir(parents=True, exist_ok=True)

    saved_names: List[str] = []
    for file_name, csv_bytes in files:
        clean_name = Path(str(file_name)).name
        if not clean_name:
            continue
        target = files_path / clean_name
        # Si ya existe (caso raro de duplicado en el mismo upload),
        # agregar un suffix.
        if target.exists():
            base = target.stem
            ext = target.suffix
            counter = 2
            while target.exists():
                target = files_path / f"{base}_{counter}{ext}"
                counter += 1
        try:
            with open(target, "wb") as f:
                f.write(csv_bytes)
            saved_names.append(target.name)
        except Exception:
            continue

    if not saved_names:
        # Si no se pudo guardar nada, limpiar el folder creado.
        try:
            shutil.rmtree(corrida_path, ignore_errors=True)
        except Exception:
            pass
        raise ValueError("No se pudo guardar ningún archivo")

    ts_iso = datetime.now().isoformat(timespec="seconds")
    label = (corrida_label or "").strip() or ts_iso

    time_range = {"min": "", "max": ""}
    if detected_time_range:
        ts_min, ts_max = detected_time_range
        if ts_min is not None:
            try:
                time_range["min"] = pd.Timestamp(ts_min).isoformat()
            except Exception:
                pass
        if ts_max is not None:
            try:
                time_range["max"] = pd.Timestamp(ts_max).isoformat()
            except Exception:
                pass

    metadata = {
        "corrida_id": corrida_id,
        "instance_id": instance_id,
        "timestamp": ts_iso,
        "corrida_label": label,
        "notes": notes or "",
        "n_files": len(saved_names),
        "files": saved_names,
        "time_range": time_range,
    }

    with open(corrida_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    _enforce_max_corridas(instance_id)
    return corrida_id


def list_trend_corridas(
    instance_id: str, limit: int = MAX_TREND_CORRIDAS_PER_INSTANCE,
) -> List[Dict[str, Any]]:
    """Lista corridas Trend de una instancia, más recientes primero.
    Devuelve metadata sin payload (no carga los CSVs)."""
    root = _trend_history_root(instance_id)
    if not root.exists():
        return []
    items = []
    for d in sorted(root.iterdir(), reverse=True):
        if not d.is_dir() or not d.name.startswith("corrida_"):
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            items.append({
                "corrida_id": meta.get("corrida_id", d.name),
                "timestamp": meta.get("timestamp", ""),
                "corrida_label": meta.get("corrida_label", ""),
                "notes": meta.get("notes", ""),
                "n_files": meta.get("n_files", 0),
                "files": meta.get("files", []),
                "time_range": meta.get("time_range", {"min": "", "max": ""}),
            })
            if len(items) >= limit:
                break
        except Exception:
            continue
    return items


def load_trend_corrida_files(
    instance_id: str, corrida_id: str,
) -> List[Tuple[str, bytes]]:
    """Devuelve lista de ``(file_name, csv_bytes)`` para una corrida."""
    files_path = _corrida_dir(instance_id, corrida_id) / "files"
    if not files_path.exists():
        return []
    out = []
    for f in sorted(files_path.iterdir()):
        if not f.is_file():
            continue
        try:
            with open(f, "rb") as fh:
                out.append((f.name, fh.read()))
        except Exception:
            continue
    return out


def delete_trend_corrida(instance_id: str, corrida_id: str) -> bool:
    """Borra una corrida completa (todos los CSVs + metadata)."""
    p = _corrida_dir(instance_id, corrida_id)
    if not p.exists():
        return False
    try:
        shutil.rmtree(p)
        return True
    except Exception:
        return False


def get_corrida_metadata(
    instance_id: str, corrida_id: str,
) -> Optional[Dict[str, Any]]:
    """Carga solo el metadata.json de una corrida."""
    meta_path = _corrida_dir(instance_id, corrida_id) / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def update_corrida_time_range(
    instance_id: str,
    corrida_id: str,
    ts_min: Optional[pd.Timestamp],
    ts_max: Optional[pd.Timestamp],
) -> bool:
    """
    Actualiza el time_range del metadata después de parsear los CSVs.
    Pensado para que el caller (la página) pueda llamarlo después de
    save_trend_corrida si el detected_time_range no se conocía al
    momento de guardar.
    """
    meta = get_corrida_metadata(instance_id, corrida_id)
    if meta is None:
        return False
    try:
        meta["time_range"] = {
            "min": pd.Timestamp(ts_min).isoformat() if ts_min is not None else "",
            "max": pd.Timestamp(ts_max).isoformat() if ts_max is not None else "",
        }
        meta_path = _corrida_dir(instance_id, corrida_id) / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _enforce_max_corridas(instance_id: str) -> None:
    """Mantiene solo las últimas MAX_TREND_CORRIDAS_PER_INSTANCE."""
    root = _trend_history_root(instance_id)
    dirs = sorted(
        [d for d in root.iterdir()
         if d.is_dir() and d.name.startswith("corrida_")],
        reverse=True,
    )
    for old in dirs[MAX_TREND_CORRIDAS_PER_INSTANCE:]:
        try:
            shutil.rmtree(old)
        except Exception:
            pass


# ============================================================
# HELPERS DE COMBINACIÓN
# ============================================================

def list_corridas_summary(instance_id: str) -> Dict[str, Any]:
    """Resumen ultra-corto: cuántas corridas + rango temporal global."""
    corridas = list_trend_corridas(instance_id)
    n = len(corridas)
    if n == 0:
        return {"n_corridas": 0, "earliest": "", "latest": "", "total_files": 0}
    times_min, times_max = [], []
    total_files = 0
    for c in corridas:
        tr = c.get("time_range", {}) or {}
        if tr.get("min"):
            times_min.append(tr["min"])
        if tr.get("max"):
            times_max.append(tr["max"])
        total_files += int(c.get("n_files", 0) or 0)
    return {
        "n_corridas": n,
        "earliest": min(times_min) if times_min else "",
        "latest": max(times_max) if times_max else "",
        "total_files": total_files,
    }


__all__ = [
    "MAX_TREND_CORRIDAS_PER_INSTANCE",
    "save_trend_corrida",
    "list_trend_corridas",
    "load_trend_corrida_files",
    "delete_trend_corrida",
    "get_corrida_metadata",
    "update_corrida_time_range",
    "list_corridas_summary",
]
