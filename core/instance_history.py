"""
core.instance_history
=====================

Persistencia y comparativo de snapshots históricos del Tabular DataFrame
por instancia (Ciclo 16.2).

El sistema guarda automáticamente o a pedido del usuario un snapshot
del estado de cada sensor del Sensor Map en una corrida dada
(Overall, Status, Alarm, Danger, Unit, signal source, RPM, timestamp).
Esto permite que en corridas posteriores el sistema compare la
condición actual contra una corrida anterior y muestre tendencia
(↑ subiendo, ↓ bajando, → estable) con el delta numérico.

Beneficio operativo: el ingeniero NO necesita conservar los CSVs
viejos — el sistema mantiene los readings consolidados en JSON
liviano (~5 KB por corrida). Para una unidad crítica monitoreada
mensual, 12 snapshots ocupan ~60 KB.

Estructura en disco (backend local):

    {INSTANCES_DIR}/{instance_id}/history/snapshot_{ISO8601}.json

Cada snapshot es un dict JSON con:

  - snapshot_id    str    ID único (timestamp-based).
  - instance_id    str    Ref de la instancia.
  - timestamp      str    ISO8601 cuando se tomó el snapshot.
  - corrida_label  str    Etiqueta humana (default = timestamp).
  - notes          str    Observaciones del usuario (opcional).
  - rpm_operativa  float  RPM promedio detectada de los signals (si la hay).
  - readings       list   Por sensor:
        sensor_label, plane, plane_label, type, family,
        unit, alarm, danger, overall, status, source_signal.

Funciones públicas:

  - save_snapshot(instance_id, df, ...)        → snapshot_id
  - list_snapshots(instance_id, limit=12)      → list[dict]
  - load_snapshot(instance_id, snapshot_id)    → dict | None
  - delete_snapshot(instance_id, snapshot_id)  → bool
  - get_previous_snapshot(instance_id, before_ts=None) → dict | None
  - compare_to_previous(current_df, previous_snapshot) → DataFrame
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from core.instance_repository import INSTANCES_DIR


# Cuántos snapshots conservamos por instancia. Más viejos se eliminan
# automáticamente al guardar uno nuevo.
MAX_SNAPSHOTS_PER_INSTANCE = 24


# ============================================================
# PATHS
# ============================================================

def _history_dir(instance_id: str) -> Path:
    """Directorio de historial para una instancia. Crea si no existe."""
    p = INSTANCES_DIR / instance_id / "history"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _snapshot_path(instance_id: str, snapshot_id: str) -> Path:
    return _history_dir(instance_id) / f"{snapshot_id}.json"


def _new_snapshot_id() -> str:
    """ID basado en timestamp ISO8601 sanitizado."""
    now = datetime.now()
    return "snapshot_" + now.strftime("%Y%m%d_%H%M%S")


# ============================================================
# SAVE / LOAD / LIST / DELETE
# ============================================================

def save_snapshot(
    instance_id: str,
    df: pd.DataFrame,
    *,
    corrida_label: str = "",
    notes: str = "",
    rpm_operativa: Optional[float] = None,
    signals_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Guarda un snapshot del DataFrame de severidad para una instancia.

    Args:
        instance_id: ID de la Asset Instance.
        df: DataFrame con columnas Label, Plane, Plane Label, Type,
            Family, Unit, Alarm, Danger, Overall, Status, Source.
            Mismo formato que devuelve build_severity_table.
        corrida_label: Etiqueta humana opcional. Default = timestamp.
        notes: Observaciones del usuario.
        rpm_operativa: RPM promedio (opcional, para metadata).
        signals_metadata: Snapshot de metadata de signals (opcional).

    Returns:
        snapshot_id (string).
    """
    if not instance_id:
        raise ValueError("instance_id requerido")
    if df is None or df.empty:
        raise ValueError("df vacío — nada que snapshotear")

    sid = _new_snapshot_id()
    ts_iso = datetime.now().isoformat(timespec="seconds")
    label = (corrida_label or "").strip() or ts_iso

    readings = []
    for _, r in df.iterrows():
        try:
            readings.append({
                "sensor_label": str(r.get("Label", "")),
                "plane": int(r.get("Plane", 0) or 0),
                "plane_label": str(r.get("Plane Label", "") or ""),
                "type": str(r.get("Type", "") or ""),
                "family": str(r.get("Family", "") or ""),
                "unit": str(r.get("Unit", "") or ""),
                "alarm": _safe_float(r.get("Alarm")),
                "danger": _safe_float(r.get("Danger")),
                "overall": _safe_float(r.get("Overall")),
                "status": str(r.get("Status", "") or ""),
                "source_signal": str(r.get("Source", "") or ""),
            })
        except Exception:
            continue

    payload = {
        "snapshot_id": sid,
        "instance_id": instance_id,
        "timestamp": ts_iso,
        "corrida_label": label,
        "notes": notes or "",
        "rpm_operativa": rpm_operativa if rpm_operativa is not None else None,
        "readings": readings,
        "signals_metadata": signals_metadata or {},
    }

    path = _snapshot_path(instance_id, sid)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Limitar cantidad de snapshots
    _enforce_max_snapshots(instance_id)
    return sid


def list_snapshots(
    instance_id: str, limit: int = MAX_SNAPSHOTS_PER_INSTANCE
) -> List[Dict[str, Any]]:
    """
    Lista los snapshots de una instancia, más recientes primero.
    Devuelve dicts con snapshot_id, timestamp, corrida_label, notes,
    rpm_operativa, n_readings (sin cargar el detalle de readings para
    eficiencia).
    """
    h = _history_dir(instance_id)
    if not h.exists():
        return []
    items = []
    for p in sorted(h.glob("snapshot_*.json"), reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            items.append({
                "snapshot_id": d.get("snapshot_id", p.stem),
                "timestamp": d.get("timestamp", ""),
                "corrida_label": d.get("corrida_label", ""),
                "notes": d.get("notes", ""),
                "rpm_operativa": d.get("rpm_operativa"),
                "n_readings": len(d.get("readings", [])),
            })
            if len(items) >= limit:
                break
        except Exception:
            continue
    return items


def load_snapshot(instance_id: str, snapshot_id: str) -> Optional[Dict[str, Any]]:
    """Carga el snapshot completo (con readings)."""
    p = _snapshot_path(instance_id, snapshot_id)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_snapshot(instance_id: str, snapshot_id: str) -> bool:
    """Borra un snapshot. Devuelve True si lo borró."""
    p = _snapshot_path(instance_id, snapshot_id)
    if not p.exists():
        return False
    try:
        p.unlink()
        return True
    except Exception:
        return False


def get_previous_snapshot(
    instance_id: str, before_ts: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Devuelve el snapshot más reciente anterior a ``before_ts``.
    Si before_ts es None, devuelve el más reciente sin filtro.
    """
    snaps = list_snapshots(instance_id, limit=MAX_SNAPSHOTS_PER_INSTANCE)
    if not snaps:
        return None
    if before_ts:
        snaps = [s for s in snaps if s["timestamp"] < before_ts]
    if not snaps:
        return None
    return load_snapshot(instance_id, snaps[0]["snapshot_id"])


# ============================================================
# COMPARATIVO
# ============================================================

def compare_to_previous(
    current_df: pd.DataFrame,
    previous_snapshot: Dict[str, Any],
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con columnas extras de comparativo:

      Label, Plane, Plane Label, Type, Unit,
      Overall (current), Overall_prev, Delta, Delta_pct,
      Status (current), Status_prev, Trend.

    Trend toma:
      - "up_critical"   delta % >= +20 ó cambio Normal→Alarm/Danger
      - "up"            +5 a +20 %
      - "stable"        |delta %| < 5 %
      - "down"          -5 a -20 %
      - "down_good"     <= -20 % o cambio Alarm/Danger→Normal
      - "no_prev"       sin lectura anterior para este sensor
      - "no_curr"       sensor que existia antes y ahora no (raro)
    """
    if current_df is None or current_df.empty:
        return pd.DataFrame()

    prev_by_label: Dict[str, Dict[str, Any]] = {}
    for r in (previous_snapshot or {}).get("readings", []):
        lbl = str(r.get("sensor_label", "")).strip()
        if lbl:
            prev_by_label[lbl] = r

    rows = []
    for _, r in current_df.iterrows():
        lbl = str(r.get("Label", ""))
        ov_curr = _safe_float(r.get("Overall"))
        st_curr = str(r.get("Status", "") or "")

        prev = prev_by_label.get(lbl)
        if prev is None:
            ov_prev = None
            st_prev = ""
            delta = None
            delta_pct = None
            trend = "no_prev"
        else:
            ov_prev = _safe_float(prev.get("overall"))
            st_prev = str(prev.get("status", "") or "")
            delta = ov_curr - ov_prev
            if ov_prev > 0:
                delta_pct = (delta / ov_prev) * 100.0
            else:
                delta_pct = None
            trend = _classify_trend(delta_pct, st_prev, st_curr)

        rows.append({
            "Label": lbl,
            "Plane": r.get("Plane", 0),
            "Plane Label": r.get("Plane Label", ""),
            "Type": r.get("Type", ""),
            "Unit": r.get("Unit", ""),
            "Overall": ov_curr,
            "Overall_prev": ov_prev,
            "Delta": delta,
            "Delta_pct": delta_pct,
            "Status": st_curr,
            "Status_prev": st_prev,
            "Trend": trend,
        })

    return pd.DataFrame(rows)


def _classify_trend(
    delta_pct: Optional[float],
    status_prev: str,
    status_curr: str,
) -> str:
    """Categoría de tendencia para colorear/iconear el comparativo."""
    # Cambios de status fuertes priman
    rank = {"Normal": 1, "Alarm": 2, "Danger": 3, "No Data": 0, "": 0}
    rp = rank.get(status_prev, 0)
    rc = rank.get(status_curr, 0)
    if rc > rp and rc >= 2:
        return "up_critical"
    if rp >= 2 and rc == 1:
        return "down_good"

    if delta_pct is None:
        return "no_prev"
    if delta_pct >= 20.0:
        return "up_critical"
    if delta_pct >= 5.0:
        return "up"
    if delta_pct <= -20.0:
        return "down_good"
    if delta_pct <= -5.0:
        return "down"
    return "stable"


def trend_arrow(trend: str) -> str:
    """Símbolo unicode para representar la tendencia."""
    return {
        "up_critical": "▲",
        "up": "↑",
        "stable": "→",
        "down": "↓",
        "down_good": "▼",
        "no_prev": "—",
        "no_curr": "?",
    }.get(trend, "—")


def trend_color(trend: str) -> str:
    """Color hex para chip de tendencia."""
    return {
        "up_critical": "#dc2626",  # rojo
        "up": "#f59e0b",           # ámbar
        "stable": "#64748b",       # slate
        "down": "#16a34a",         # verde claro
        "down_good": "#059669",    # verde más fuerte
        "no_prev": "#94a3b8",      # gris
        "no_curr": "#94a3b8",
    }.get(trend, "#64748b")


# ============================================================
# UTILS
# ============================================================

def _safe_float(v) -> float:
    try:
        f = float(v)
        if f != f:  # NaN
            return 0.0
        return f
    except Exception:
        return 0.0


def _enforce_max_snapshots(instance_id: str) -> None:
    """Mantiene solo los últimos MAX_SNAPSHOTS_PER_INSTANCE."""
    h = _history_dir(instance_id)
    files = sorted(h.glob("snapshot_*.json"), reverse=True)
    for old in files[MAX_SNAPSHOTS_PER_INSTANCE:]:
        try:
            old.unlink()
        except Exception:
            pass


__all__ = [
    "MAX_SNAPSHOTS_PER_INSTANCE",
    "save_snapshot",
    "list_snapshots",
    "load_snapshot",
    "delete_snapshot",
    "get_previous_snapshot",
    "compare_to_previous",
    "trend_arrow",
    "trend_color",
]
