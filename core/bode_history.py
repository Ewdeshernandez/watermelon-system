"""
core.bode_history
=================

Persistencia y comparativo de snapshots históricos del módulo Bode
(Ciclo 17.2). Para cada corrida snapshoteada captura, por sensor del
Sensor Map matched:

  - amp + fase a la velocidad operativa
  - trayectoria completa (speed, amp, phase) downsampleada a ~80 puntos
  - velocidad crítica detectada + amp/fase del peak + Q factor
  - separation margin contra la operativa

El render del Bode superpone las curvas amp vs RPM y phase vs RPM
de cada snapshot histórico en gradiente cronológico, permitiendo
ver migración del modo (peak en distinto RPM), cambios del Q
(amplitud del peak crece o baja) y deriva de fase a través del
modo entre corridas.

Storage: ``{INSTANCES_DIR}/{instance_id}/history/bode_{ISO8601}.json``

Reusa los mismos clasificadores de polar_history (phase_shift_classifier,
amplitude_change_classifier, shortest_arc_phase_diff) porque son
módulo-agnósticos.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.instance_repository import INSTANCES_DIR

# Reusar clasificadores ya validados — son agnosticos al modulo
from core.polar_history import (
    phase_shift_classifier,
    amplitude_change_classifier,
    shortest_arc_phase_diff,
)


MAX_BODE_SNAPSHOTS_PER_INSTANCE = 24


def _bode_history_dir(instance_id: str) -> Path:
    p = INSTANCES_DIR / instance_id / "history"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _bode_snapshot_path(instance_id: str, snapshot_id: str) -> Path:
    return _bode_history_dir(instance_id) / f"{snapshot_id}.json"


def _new_bode_snapshot_id() -> str:
    return "bode_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float(v) -> float:
    try:
        f = float(v)
        if f != f:
            return 0.0
        return f
    except Exception:
        return 0.0


# ============================================================
# SAVE / LOAD / LIST / DELETE
# ============================================================

def save_bode_snapshot(
    instance_id: str,
    *,
    operating_speed_rpm: float,
    sensors_data: List[Dict[str, Any]],
    corrida_label: str = "",
    notes: str = "",
) -> str:
    """
    Guarda un snapshot del estado Bode de la instancia.

    Args:
        instance_id: ID de la Asset Instance.
        operating_speed_rpm: Velocidad operativa nominal/elegida.
        sensors_data: lista de dicts por sensor con:
            - sensor_label, csv_file
            - amp_at_op, phase_at_op, amp_unit, phase_unit
            - csv_timestamp
            - trajectory_speed/amp/phase (listas downsampleadas)
            - critical_speed_rpm, critical_speed_amp,
              critical_speed_phase_delta, q_factor (opcionales)
        corrida_label: etiqueta humana
        notes: observaciones del usuario
    """
    if not instance_id:
        raise ValueError("instance_id requerido")
    if not sensors_data:
        raise ValueError("sensors_data vacío")

    sid = _new_bode_snapshot_id()
    ts_iso = datetime.now().isoformat(timespec="seconds")
    label = (corrida_label or "").strip() or ts_iso

    cleaned = []
    for s in sensors_data:
        try:
            entry = {
                "sensor_label": str(s.get("sensor_label", "")),
                "csv_file": str(s.get("csv_file", "")),
                "amp_at_op": _safe_float(s.get("amp_at_op")),
                "phase_at_op": _safe_float(s.get("phase_at_op")),
                "amp_unit": str(s.get("amp_unit", "") or ""),
                "phase_unit": str(s.get("phase_unit", "deg") or "deg"),
                "csv_timestamp": str(s.get("csv_timestamp", "") or ""),
            }
            traj_speed = s.get("trajectory_speed")
            traj_amp = s.get("trajectory_amp")
            traj_phase = s.get("trajectory_phase")
            if (
                traj_speed is not None and traj_amp is not None
                and traj_phase is not None
                and len(traj_speed) == len(traj_amp) == len(traj_phase)
                and len(traj_speed) > 1
            ):
                entry["trajectory_speed"] = [
                    round(_safe_float(v), 2) for v in traj_speed
                ]
                entry["trajectory_amp"] = [
                    round(_safe_float(v), 4) for v in traj_amp
                ]
                entry["trajectory_phase"] = [
                    round(_safe_float(v) % 360.0, 2) for v in traj_phase
                ]
            cs_rpm = s.get("critical_speed_rpm")
            if cs_rpm is not None:
                entry["critical_speed_rpm"] = _safe_float(cs_rpm)
                entry["critical_speed_amp"] = _safe_float(
                    s.get("critical_speed_amp"))
                entry["critical_speed_phase_delta"] = _safe_float(
                    s.get("critical_speed_phase_delta"))
                entry["q_factor"] = _safe_float(s.get("q_factor"))
            cleaned.append(entry)
        except Exception:
            continue

    payload = {
        "snapshot_id": sid,
        "instance_id": instance_id,
        "timestamp": ts_iso,
        "corrida_label": label,
        "notes": notes or "",
        "operating_speed_rpm": _safe_float(operating_speed_rpm),
        "sensors": cleaned,
    }

    with open(_bode_snapshot_path(instance_id, sid), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _enforce_max_bode_snapshots(instance_id)
    return sid


def list_bode_snapshots(
    instance_id: str, limit: int = MAX_BODE_SNAPSHOTS_PER_INSTANCE,
) -> List[Dict[str, Any]]:
    """Lista snapshots Bode más recientes primero."""
    h = _bode_history_dir(instance_id)
    if not h.exists():
        return []
    items = []
    for p in sorted(h.glob("bode_*.json"), reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            items.append({
                "snapshot_id": d.get("snapshot_id", p.stem),
                "timestamp": d.get("timestamp", ""),
                "corrida_label": d.get("corrida_label", ""),
                "notes": d.get("notes", ""),
                "operating_speed_rpm": d.get("operating_speed_rpm"),
                "n_sensors": len(d.get("sensors", [])),
            })
            if len(items) >= limit:
                break
        except Exception:
            continue
    return items


def load_bode_snapshot(instance_id: str, snapshot_id: str) -> Optional[Dict[str, Any]]:
    p = _bode_snapshot_path(instance_id, snapshot_id)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_bode_snapshot(instance_id: str, snapshot_id: str) -> bool:
    p = _bode_snapshot_path(instance_id, snapshot_id)
    if not p.exists():
        return False
    try:
        p.unlink()
        return True
    except Exception:
        return False


def _enforce_max_bode_snapshots(instance_id: str) -> None:
    h = _bode_history_dir(instance_id)
    files = sorted(h.glob("bode_*.json"), reverse=True)
    for old in files[MAX_BODE_SNAPSHOTS_PER_INSTANCE:]:
        try:
            old.unlink()
        except Exception:
            pass


# ============================================================
# COMPARATIVO + SKIP IDENTICAL
# ============================================================

def _bode_snapshot_is_identical_to(
    snapshot: Dict[str, Any],
    current_by_label: Dict[str, Dict[str, float]],
    amp_tol_pct: float = 1.0,
    phase_tol_deg: float = 1.0,
) -> bool:
    """True si el snapshot es esencialmente la misma corrida que current."""
    if not current_by_label:
        return False
    snap_by_label = {
        str(s.get("sensor_label", "")): s
        for s in snapshot.get("sensors", [])
    }
    if not snap_by_label:
        return False
    matched = 0
    for lbl, cur in current_by_label.items():
        if lbl not in snap_by_label:
            return False
        snap_s = snap_by_label[lbl]
        amp_curr = _safe_float(cur.get("amp"))
        amp_snap = _safe_float(snap_s.get("amp_at_op"))
        ph_curr = _safe_float(cur.get("phase"))
        ph_snap = _safe_float(snap_s.get("phase_at_op"))

        if amp_snap == 0:
            if amp_curr != 0:
                return False
        else:
            if abs(amp_curr - amp_snap) / abs(amp_snap) * 100.0 > amp_tol_pct:
                return False
        diff = abs((ph_curr - ph_snap + 540) % 360 - 180)
        if diff > phase_tol_deg:
            return False
        matched += 1
    return matched > 0


def get_previous_bode_snapshot(
    instance_id: str,
    skip_identical_to_sensors: Optional[Dict[str, Dict[str, float]]] = None,
    identical_amp_tol_pct: float = 1.0,
    identical_phase_tol_deg: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """
    Devuelve el snapshot Bode más reciente que NO es la corrida actual.
    Mismo patrón que get_previous_polar_snapshot.
    """
    snaps = list_bode_snapshots(instance_id)
    if not snaps:
        return None
    for s in snaps:
        snap = load_bode_snapshot(instance_id, s["snapshot_id"])
        if snap is None:
            continue
        if skip_identical_to_sensors is None:
            return snap
        if not _bode_snapshot_is_identical_to(
            snap, skip_identical_to_sensors,
            identical_amp_tol_pct, identical_phase_tol_deg,
        ):
            return snap
    return None


def get_bode_history_for_sensor(
    instance_id: str,
    sensor_label: str,
    max_snapshots: int = 8,
    current_reading: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Devuelve los puntos históricos de un sensor a través de los
    snapshots Bode, ordenados cronológicamente (asc).
    """
    snaps = list_bode_snapshots(instance_id, limit=max_snapshots)
    points: List[Dict[str, Any]] = []
    for s in snaps:
        snap = load_bode_snapshot(instance_id, s["snapshot_id"])
        if snap is None:
            continue
        for sens in snap.get("sensors", []):
            if str(sens.get("sensor_label", "")) == sensor_label:
                points.append({
                    "timestamp": snap.get("timestamp", ""),
                    "corrida_label": snap.get("corrida_label", ""),
                    "op_speed_rpm": _safe_float(snap.get("operating_speed_rpm")),
                    "amp": _safe_float(sens.get("amp_at_op")),
                    "phase": _safe_float(sens.get("phase_at_op")),
                    "unit": str(sens.get("amp_unit", "") or ""),
                    "critical_speed_rpm": _safe_float(sens.get("critical_speed_rpm")),
                    "q_factor": _safe_float(sens.get("q_factor")),
                })
                break

    points.sort(key=lambda p: p["timestamp"])

    if current_reading is not None:
        cur_amp = _safe_float(current_reading.get("amp_at_op"))
        cur_phase = _safe_float(current_reading.get("phase_at_op"))
        already = False
        if points:
            last = points[-1]
            if abs(last["amp"] - cur_amp) < (max(abs(last["amp"]), 0.001) * 0.01) \
               and abs((last["phase"] - cur_phase + 540) % 360 - 180) < 1.0:
                already = True
        if not already:
            points.append({
                "timestamp": current_reading.get("timestamp")
                    or datetime.now().isoformat(timespec="seconds"),
                "corrida_label": current_reading.get("corrida_label", "Actual"),
                "op_speed_rpm": _safe_float(current_reading.get("op_speed_rpm")),
                "amp": cur_amp,
                "phase": cur_phase,
                "unit": str(current_reading.get("amp_unit", "") or ""),
                "critical_speed_rpm": _safe_float(current_reading.get("critical_speed_rpm")),
                "q_factor": _safe_float(current_reading.get("q_factor")),
            })

    return points


__all__ = [
    "MAX_BODE_SNAPSHOTS_PER_INSTANCE",
    "save_bode_snapshot",
    "list_bode_snapshots",
    "load_bode_snapshot",
    "delete_bode_snapshot",
    "get_bode_history_for_sensor",
    "get_previous_bode_snapshot",
    "phase_shift_classifier",
    "amplitude_change_classifier",
    "shortest_arc_phase_diff",
]
