"""
core.spectrum_history
=====================

Persistencia de snapshots históricos de Spectrum FFT (Ciclo 23.80).

Cada snapshot captura, por sensor:
  - frecuencias + amplitudes (espectro completo o downsampleado a ~8k bins)
  - sampling_rate, window (Hanning/Flat-top), n_avg, overlap
  - peaks identificados (top N) con freq + amp + label opcional
  - lines_of_resolution, freq_span_hz
  - unidades (amp_unit, freq_unit)
  - severity y thresholds si aplican
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core import history_storage

_SNAPSHOT_TYPE = "spectrum"
MAX_SPECTRUM_SNAPSHOTS_PER_INSTANCE = history_storage.MAX_SNAPSHOTS_PER_TYPE
SNAPSHOT_MAX_BINS = 8_192  # típico es 6.4k, da margen


def _new_spectrum_snapshot_id(sensor_id: str = "") -> str:
    return history_storage.new_snapshot_id(_SNAPSHOT_TYPE, sensor_id=sensor_id)


def _safe_float(v) -> float:
    try:
        f = float(v)
        if f != f:
            return 0.0
        return f
    except Exception:
        return 0.0


def _downsample(values: List[float], target: int = SNAPSHOT_MAX_BINS) -> List[float]:
    n = len(values)
    if n <= target:
        return [_safe_float(v) for v in values]
    stride = max(1, n // target)
    return [_safe_float(values[i]) for i in range(0, n, stride)]


def save_spectrum_snapshot(
    instance_id: str,
    *,
    sensors_data: List[Dict[str, Any]],
    corrida_label: str = "",
    notes: str = "",
    operating_speed_rpm: Optional[float] = None,
    sensor_id: str = "",
) -> str:
    """Guarda un snapshot de spectra FFT.

    sensors_data: lista de dicts por sensor:
        - sensor_label, csv_file
        - freqs (list[float]), amps (list[float])
        - amp_unit (ej. "g pk", "in/s rms")
        - freq_unit ("Hz" o "CPM")
        - sampling_rate_hz, freq_span_hz, lines_of_resolution
        - window, n_avg, overlap_pct
        - peaks: list of {freq, amp, label}
        - severity, thresholds
    """
    if not instance_id:
        raise ValueError("instance_id requerido")
    if not sensors_data:
        raise ValueError("sensors_data vacío")

    # Ciclo 17.34 (v3.31.239) — sensor_id en sid.
    _inferred_sensor = sensor_id
    if not _inferred_sensor and len(sensors_data) == 1:
        _inferred_sensor = str(sensors_data[0].get("sensor_label") or "")
    sid = _new_spectrum_snapshot_id(_inferred_sensor)
    ts_iso = datetime.now().isoformat(timespec="seconds")
    label = (corrida_label or "").strip() or ts_iso

    cleaned = []
    for s in sensors_data:
        try:
            entry = {
                "sensor_label": str(s.get("sensor_label", "")),
                "csv_file": str(s.get("csv_file", "")),
                "amp_unit": str(s.get("amp_unit", "") or ""),
                "freq_unit": str(s.get("freq_unit", "Hz") or "Hz"),
                "sampling_rate_hz": _safe_float(s.get("sampling_rate_hz")),
                "freq_span_hz": _safe_float(s.get("freq_span_hz")),
                "lines_of_resolution": int(s.get("lines_of_resolution") or 0),
                "window": str(s.get("window", "") or ""),
                "n_avg": int(s.get("n_avg") or 1),
                "overlap_pct": _safe_float(s.get("overlap_pct")),
                "severity": str(s.get("severity", "") or ""),
                "csv_timestamp": str(s.get("csv_timestamp", "") or ""),
            }

            freqs = s.get("freqs")
            amps = s.get("amps")
            if freqs is not None and amps is not None:
                entry["freqs"] = _downsample(freqs)
                entry["amps"] = _downsample(amps)
                entry["n_bins_snapshot"] = len(entry["amps"])

            peaks = s.get("peaks") or []
            if peaks:
                entry["peaks"] = [
                    {
                        "freq": _safe_float(p.get("freq")),
                        "amp": _safe_float(p.get("amp")),
                        "label": str(p.get("label", "") or ""),
                    }
                    for p in peaks[:20]  # top 20 peaks max
                ]

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


def list_spectrum_snapshots(
    instance_id: str,
    limit: int = MAX_SPECTRUM_SNAPSHOTS_PER_INSTANCE,
    *,
    sensor_id: str = "",
) -> List[Dict[str, Any]]:
    """sensor_id (v3.31.239+): filtra solo del sensor indicado."""
    items: List[Dict[str, Any]] = []
    snaps = history_storage.list_snapshots(
        instance_id, _SNAPSHOT_TYPE, sensor_id=sensor_id,
    )
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


def load_spectrum_snapshot(instance_id: str, snapshot_id: str) -> Optional[Dict[str, Any]]:
    return history_storage.load_snapshot(instance_id, _SNAPSHOT_TYPE, snapshot_id)


def delete_spectrum_snapshot(instance_id: str, snapshot_id: str) -> bool:
    return history_storage.delete_snapshot(instance_id, _SNAPSHOT_TYPE, snapshot_id)


__all__ = [
    "MAX_SPECTRUM_SNAPSHOTS_PER_INSTANCE",
    "SNAPSHOT_MAX_BINS",
    "save_spectrum_snapshot",
    "list_spectrum_snapshots",
    "load_spectrum_snapshot",
    "delete_spectrum_snapshot",
]
