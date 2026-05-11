"""
core.tabular_history
====================

Persistencia de snapshots históricos de Tabular List (Ciclo 23.80).

El módulo Tabular es la vista resumen "channel list" estilo Bently
System1: una fila por canal/sensor con sus métricas Direct, 1X, 2X,
Gap, vector, severity zone, y umbrales ISO/API.

A diferencia de los otros history modules, Tabular guarda un
**resumen agregado**, no time-series. Es info densa pero liviana
(~5-10 KB por snapshot).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core import history_storage

_SNAPSHOT_TYPE = "tabular"
MAX_TABULAR_SNAPSHOTS_PER_INSTANCE = history_storage.MAX_SNAPSHOTS_PER_TYPE


def _new_tabular_snapshot_id() -> str:
    return history_storage.new_snapshot_id(_SNAPSHOT_TYPE)


def _safe_float(v) -> float:
    try:
        f = float(v)
        if f != f:
            return 0.0
        return f
    except Exception:
        return 0.0


def save_tabular_snapshot(
    instance_id: str,
    *,
    channels_data: List[Dict[str, Any]],
    corrida_label: str = "",
    notes: str = "",
    operating_speed_rpm: Optional[float] = None,
) -> str:
    """Guarda snapshot de la tabla de canales.

    channels_data: lista por canal/sensor:
        - sensor_label, csv_file
        - direct, direct_unit
        - vector_1x_amp, vector_1x_phase, vector_1x_unit
        - vector_2x_amp, vector_2x_phase, vector_2x_unit
        - gap_voltage (V DC, opcional)
        - severity zone ("Normal", "Alarma", "Danger"), threshold_alarm, threshold_danger
        - iso_zone ("A", "B", "C", "D" según ISO 20816)
        - api_compliance bool
        - csv_timestamp
    """
    if not instance_id:
        raise ValueError("instance_id requerido")
    if not channels_data:
        raise ValueError("channels_data vacío")

    sid = _new_tabular_snapshot_id()
    ts_iso = datetime.now().isoformat(timespec="seconds")
    label = (corrida_label or "").strip() or ts_iso

    cleaned = []
    for c in channels_data:
        try:
            entry = {
                "sensor_label": str(c.get("sensor_label", "")),
                "csv_file": str(c.get("csv_file", "")),
                "csv_timestamp": str(c.get("csv_timestamp", "") or ""),

                "direct": _safe_float(c.get("direct")),
                "direct_unit": str(c.get("direct_unit", "") or ""),

                "vector_1x_amp": _safe_float(c.get("vector_1x_amp")),
                "vector_1x_phase": _safe_float(c.get("vector_1x_phase")) % 360.0,
                "vector_1x_unit": str(c.get("vector_1x_unit", "") or ""),

                "vector_2x_amp": _safe_float(c.get("vector_2x_amp")),
                "vector_2x_phase": _safe_float(c.get("vector_2x_phase")) % 360.0,
                "vector_2x_unit": str(c.get("vector_2x_unit", "") or ""),

                "gap_voltage": _safe_float(c.get("gap_voltage")) if c.get("gap_voltage") is not None else None,

                "severity": str(c.get("severity", "") or ""),
                "threshold_alarm": _safe_float(c.get("threshold_alarm")),
                "threshold_danger": _safe_float(c.get("threshold_danger")),

                "iso_zone": str(c.get("iso_zone", "") or ""),
                "api_compliance": bool(c.get("api_compliance", False)),
            }
            cleaned.append(entry)
        except Exception:
            continue

    payload = {
        "snapshot_id": sid,
        "instance_id": instance_id,
        "timestamp": ts_iso,
        "corrida_label": label,
        "notes": notes or "",
        "operating_speed_rpm": _safe_float(operating_speed_rpm) if operating_speed_rpm else None,
        "channels": cleaned,
    }

    history_storage.save_snapshot(instance_id, _SNAPSHOT_TYPE, sid, payload)
    return sid


def list_tabular_snapshots(
    instance_id: str,
    limit: int = MAX_TABULAR_SNAPSHOTS_PER_INSTANCE,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    snaps = history_storage.list_snapshots(instance_id, _SNAPSHOT_TYPE)
    for snap in snaps[:limit]:
        d = history_storage.load_snapshot(instance_id, _SNAPSHOT_TYPE, snap["snapshot_id"])
        if d is None:
            continue
        # Compute summary statistics for the card
        channels = d.get("channels", [])
        severities = [c.get("severity", "") for c in channels]
        n_normal = sum(1 for s in severities if s == "Normal")
        n_alarm = sum(1 for s in severities if s == "Alarma")
        n_danger = sum(1 for s in severities if s == "Danger")
        items.append({
            "snapshot_id": d.get("snapshot_id", snap["snapshot_id"]),
            "timestamp": d.get("timestamp", ""),
            "corrida_label": d.get("corrida_label", ""),
            "notes": d.get("notes", ""),
            "operating_speed_rpm": d.get("operating_speed_rpm"),
            "n_channels": len(channels),
            "n_normal": n_normal,
            "n_alarm": n_alarm,
            "n_danger": n_danger,
            "worst_severity": "Danger" if n_danger else ("Alarma" if n_alarm else "Normal"),
        })
    return items


def load_tabular_snapshot(instance_id: str, snapshot_id: str) -> Optional[Dict[str, Any]]:
    return history_storage.load_snapshot(instance_id, _SNAPSHOT_TYPE, snapshot_id)


def delete_tabular_snapshot(instance_id: str, snapshot_id: str) -> bool:
    return history_storage.delete_snapshot(instance_id, _SNAPSHOT_TYPE, snapshot_id)


__all__ = [
    "MAX_TABULAR_SNAPSHOTS_PER_INSTANCE",
    "save_tabular_snapshot",
    "list_tabular_snapshots",
    "load_tabular_snapshot",
    "delete_tabular_snapshot",
]
