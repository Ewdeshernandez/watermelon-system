"""
core.orbit_history
==================

Persistencia de snapshots históricos de Orbit Analysis (Ciclo 23.80).

Por par de proximity probes ortogonales (X/Y) en un bearing,
captura la órbita filtrada (típicamente Direct AC component) +
las componentes 1X y 2X (amplitudes y fases) para diagnóstico.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core import history_storage

_SNAPSHOT_TYPE = "orbit"
MAX_ORBIT_SNAPSHOTS_PER_INSTANCE = history_storage.MAX_SNAPSHOTS_PER_TYPE
SNAPSHOT_MAX_SAMPLES = 4_096


def _new_orbit_snapshot_id() -> str:
    return history_storage.new_snapshot_id(_SNAPSHOT_TYPE)


def _safe_float(v) -> float:
    try:
        f = float(v)
        if f != f:
            return 0.0
        return f
    except Exception:
        return 0.0


def _downsample(values: List[float], target: int = SNAPSHOT_MAX_SAMPLES) -> List[float]:
    n = len(values)
    if n <= target:
        return [_safe_float(v) for v in values]
    stride = max(1, n // target)
    return [_safe_float(values[i]) for i in range(0, n, stride)]


def save_orbit_snapshot(
    instance_id: str,
    *,
    bearings_data: List[Dict[str, Any]],
    corrida_label: str = "",
    notes: str = "",
    operating_speed_rpm: Optional[float] = None,
) -> str:
    """Guarda snapshot de orbits por bearing.

    bearings_data: lista por bearing:
        - bearing_label (ej. "BRG #1", "GEN DE")
        - x_sensor_label, y_sensor_label
        - x_csv_file, y_csv_file
        - x_values (list[float]), y_values (list[float]) — time series
          ortogonales sincronizadas, downsampleadas
        - amp_unit (ej. "mil pp")
        - vector_1x: {"amp_x", "amp_y", "phase_x_deg", "phase_y_deg"}
        - vector_2x: idem
        - orbit_shape: ej. "elíptica", "8-figura", "espiral", etc (opcional)
        - severity, csv_timestamp
    """
    if not instance_id:
        raise ValueError("instance_id requerido")
    if not bearings_data:
        raise ValueError("bearings_data vacío")

    sid = _new_orbit_snapshot_id()
    ts_iso = datetime.now().isoformat(timespec="seconds")
    label = (corrida_label or "").strip() or ts_iso

    cleaned = []
    for b in bearings_data:
        try:
            entry = {
                "bearing_label": str(b.get("bearing_label", "")),
                "x_sensor_label": str(b.get("x_sensor_label", "")),
                "y_sensor_label": str(b.get("y_sensor_label", "")),
                "x_csv_file": str(b.get("x_csv_file", "")),
                "y_csv_file": str(b.get("y_csv_file", "")),
                "amp_unit": str(b.get("amp_unit", "") or ""),
                "orbit_shape": str(b.get("orbit_shape", "") or ""),
                "severity": str(b.get("severity", "") or ""),
                "csv_timestamp": str(b.get("csv_timestamp", "") or ""),
            }

            x_values = b.get("x_values")
            y_values = b.get("y_values")
            if x_values is not None and y_values is not None:
                entry["x_values"] = _downsample(x_values)
                entry["y_values"] = _downsample(y_values)
                entry["n_samples_snapshot"] = len(entry["x_values"])

            v1x = b.get("vector_1x") or {}
            if v1x:
                entry["vector_1x"] = {
                    "amp_x": _safe_float(v1x.get("amp_x")),
                    "amp_y": _safe_float(v1x.get("amp_y")),
                    "phase_x_deg": _safe_float(v1x.get("phase_x_deg")) % 360.0,
                    "phase_y_deg": _safe_float(v1x.get("phase_y_deg")) % 360.0,
                }

            v2x = b.get("vector_2x") or {}
            if v2x:
                entry["vector_2x"] = {
                    "amp_x": _safe_float(v2x.get("amp_x")),
                    "amp_y": _safe_float(v2x.get("amp_y")),
                    "phase_x_deg": _safe_float(v2x.get("phase_x_deg")) % 360.0,
                    "phase_y_deg": _safe_float(v2x.get("phase_y_deg")) % 360.0,
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
        "bearings": cleaned,
    }

    history_storage.save_snapshot(instance_id, _SNAPSHOT_TYPE, sid, payload)
    return sid


def list_orbit_snapshots(
    instance_id: str,
    limit: int = MAX_ORBIT_SNAPSHOTS_PER_INSTANCE,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    snaps = history_storage.list_snapshots(instance_id, _SNAPSHOT_TYPE)
    for snap in snaps[:limit]:
        d = history_storage.load_snapshot(instance_id, _SNAPSHOT_TYPE, snap["snapshot_id"])
        if d is None:
            continue
        items.append({
            "snapshot_id": d.get("snapshot_id", snap["snapshot_id"]),
            "timestamp": d.get("timestamp", ""),
            "corrida_label": d.get("corrida_label", ""),
            "notes": d.get("notes", ""),
            "operating_speed_rpm": d.get("operating_speed_rpm"),
            "n_bearings": len(d.get("bearings", [])),
            "bearing_labels": [b.get("bearing_label", "") for b in d.get("bearings", [])],
        })
    return items


def load_orbit_snapshot(instance_id: str, snapshot_id: str) -> Optional[Dict[str, Any]]:
    return history_storage.load_snapshot(instance_id, _SNAPSHOT_TYPE, snapshot_id)


def delete_orbit_snapshot(instance_id: str, snapshot_id: str) -> bool:
    return history_storage.delete_snapshot(instance_id, _SNAPSHOT_TYPE, snapshot_id)


__all__ = [
    "MAX_ORBIT_SNAPSHOTS_PER_INSTANCE",
    "SNAPSHOT_MAX_SAMPLES",
    "save_orbit_snapshot",
    "list_orbit_snapshots",
    "load_orbit_snapshot",
    "delete_orbit_snapshot",
]
