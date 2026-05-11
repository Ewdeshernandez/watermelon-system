"""
core.waveform_history
=====================

Persistencia de snapshots históricos de Time Waveforms (Ciclo 23.80).

Permite que el especialista guarde una corrida de waveforms procesados
y que aparezcan después en Live Monitoring para consumo del cliente
("último análisis de waveform — 3 días atrás").

Storage: delegado a ``core.history_storage`` (Supabase Storage + gzip + LRU).

Cada snapshot captura, por sensor:

  - sensor_label, csv_file
  - sampling_rate_hz, duration_sec, n_samples (raw)
  - time/value downsampleadas a 16k puntos para que el snapshot pese ~80 KB
    gzipped. Para análisis fino el especialista vuelve a Load Data con CSV.
  - métricas: peak, peak_to_peak, rms, crest_factor, kurtosis
  - severity y umbrales ISO/API si aplican

API pública:

  save_waveform_snapshot(instance_id, *, sensors_data, corrida_label, notes) → sid
  list_waveform_snapshots(instance_id, limit=10) → list[dict]
  load_waveform_snapshot(instance_id, snapshot_id) → dict | None
  delete_waveform_snapshot(instance_id, snapshot_id) → bool
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core import history_storage

_SNAPSHOT_TYPE = "waveform"
MAX_WAVEFORM_SNAPSHOTS_PER_INSTANCE = history_storage.MAX_SNAPSHOTS_PER_TYPE

# Si el waveform crudo es más largo que esto, se downsamplea para el snapshot
SNAPSHOT_MAX_SAMPLES = 16_384


def _new_waveform_snapshot_id() -> str:
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
    """Decimación uniforme (no anti-aliasing — para snapshot visual basta)."""
    n = len(values)
    if n <= target:
        return [_safe_float(v) for v in values]
    stride = max(1, n // target)
    return [_safe_float(values[i]) for i in range(0, n, stride)]


def save_waveform_snapshot(
    instance_id: str,
    *,
    sensors_data: List[Dict[str, Any]],
    corrida_label: str = "",
    notes: str = "",
    operating_speed_rpm: Optional[float] = None,
) -> str:
    """Guarda un snapshot de waveforms.

    Args:
        instance_id: ID de la instancia activa.
        sensors_data: lista de dicts, uno por sensor:
            - sensor_label, csv_file
            - sampling_rate_hz, duration_sec, n_samples_raw
            - time (list[float]), values (list[float]) — se downsamplean
            - unit (ej. "g pk", "in/s pk", "mil pp")
            - metrics: dict con peak/p2p/rms/crest/kurtosis
            - severity (Normal/Alarma/Danger), thresholds (alarm/danger)
        corrida_label: etiqueta humana
        notes: observaciones
        operating_speed_rpm: velocidad operativa opcional
    """
    if not instance_id:
        raise ValueError("instance_id requerido")
    if not sensors_data:
        raise ValueError("sensors_data vacío")

    sid = _new_waveform_snapshot_id()
    ts_iso = datetime.now().isoformat(timespec="seconds")
    label = (corrida_label or "").strip() or ts_iso

    cleaned = []
    for s in sensors_data:
        try:
            entry = {
                "sensor_label": str(s.get("sensor_label", "")),
                "csv_file": str(s.get("csv_file", "")),
                "sampling_rate_hz": _safe_float(s.get("sampling_rate_hz")),
                "duration_sec": _safe_float(s.get("duration_sec")),
                "n_samples_raw": int(s.get("n_samples_raw") or 0),
                "unit": str(s.get("unit", "") or ""),
                "severity": str(s.get("severity", "") or ""),
                "csv_timestamp": str(s.get("csv_timestamp", "") or ""),
            }

            # Downsample time/values para snapshot
            time_arr = s.get("time")
            value_arr = s.get("values")
            if time_arr is not None and value_arr is not None:
                entry["time"] = _downsample(time_arr)
                entry["values"] = _downsample(value_arr)
                entry["n_samples_snapshot"] = len(entry["values"])

            # Métricas
            metrics = s.get("metrics") or {}
            if metrics:
                entry["metrics"] = {
                    "peak": _safe_float(metrics.get("peak")),
                    "peak_to_peak": _safe_float(metrics.get("peak_to_peak")),
                    "rms": _safe_float(metrics.get("rms")),
                    "crest_factor": _safe_float(metrics.get("crest_factor")),
                    "kurtosis": _safe_float(metrics.get("kurtosis")),
                }

            # Thresholds
            thresholds = s.get("thresholds") or {}
            if thresholds:
                entry["thresholds"] = {
                    "alarm": _safe_float(thresholds.get("alarm")),
                    "danger": _safe_float(thresholds.get("danger")),
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
        "sensors": cleaned,
    }

    history_storage.save_snapshot(instance_id, _SNAPSHOT_TYPE, sid, payload)
    return sid


def list_waveform_snapshots(
    instance_id: str,
    limit: int = MAX_WAVEFORM_SNAPSHOTS_PER_INSTANCE,
) -> List[Dict[str, Any]]:
    """Lista snapshots de waveform más recientes primero."""
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
            "n_sensors": len(d.get("sensors", [])),
            "sensor_labels": [s.get("sensor_label", "") for s in d.get("sensors", [])],
        })
    return items


def load_waveform_snapshot(instance_id: str, snapshot_id: str) -> Optional[Dict[str, Any]]:
    return history_storage.load_snapshot(instance_id, _SNAPSHOT_TYPE, snapshot_id)


def delete_waveform_snapshot(instance_id: str, snapshot_id: str) -> bool:
    return history_storage.delete_snapshot(instance_id, _SNAPSHOT_TYPE, snapshot_id)


__all__ = [
    "MAX_WAVEFORM_SNAPSHOTS_PER_INSTANCE",
    "SNAPSHOT_MAX_SAMPLES",
    "save_waveform_snapshot",
    "list_waveform_snapshots",
    "load_waveform_snapshot",
    "delete_waveform_snapshot",
]
