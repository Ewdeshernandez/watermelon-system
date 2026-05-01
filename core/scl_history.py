"""
core.scl_history
================

Persistencia y comparativo de snapshots históricos del módulo Shaft
Centerline (Ciclo 17.3). Para cada corrida snapshoteada captura, por
**par de sondas X-Y** del cojinete:

  - x_gap, y_gap (posición del muñón a velocidad operativa, en mil)
  - eccentricity_ratio (0-1) — distancia desde centro / clearance
  - attitude_angle (deg) — ángulo de la posición respecto a load
  - bearing clearance utilizado para los cálculos
  - trayectoria completa speed/x_gap/y_gap (lift-off curve)

Permite ver superpuestas las trayectorias del muñón entre corridas:

  - **Migración del centerline**: posición media en operativa cambió
    entre corridas → desgaste asimétrico, asentamiento, deformación
    térmica del soporte.
  - **Cambio de eccentricity ratio**: el muñón está más cerca o más
    lejos de la pared del cojinete → cambio de carga, de viscosidad
    del aceite, o pérdida de clearance.
  - **Shift del attitude angle**: dirección de la fuerza
    hidrodinámica cambió → cambio de balance, alineación o
    distribución de carga entre cojinetes.
  - **Lift-off speed evolution**: si la velocidad a la que el muñón
    se separa del fondo del cojinete cambia → degradación del
    soporte hidrodinámico (API 670 §6.7).

Storage: ``{INSTANCES_DIR}/{instance_id}/history/scl_{ISO8601}.json``

Reusa los clasificadores de polar_history para amplitude/phase
classifiers cuando aplique (eccentricity ratio change uses similar
percentage thresholds).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.instance_repository import INSTANCES_DIR

MAX_SCL_SNAPSHOTS_PER_INSTANCE = 24


def _scl_history_dir(instance_id: str) -> Path:
    p = INSTANCES_DIR / instance_id / "history"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _scl_snapshot_path(instance_id: str, snapshot_id: str) -> Path:
    return _scl_history_dir(instance_id) / f"{snapshot_id}.json"


def _new_scl_snapshot_id() -> str:
    return "scl_" + datetime.now().strftime("%Y%m%d_%H%M%S")


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

def save_scl_snapshot(
    instance_id: str,
    *,
    operating_speed_rpm: float,
    bearings_data: List[Dict[str, Any]],
    corrida_label: str = "",
    notes: str = "",
) -> str:
    """
    Guarda un snapshot del estado SCL de la instancia.

    Args:
        instance_id: ID de la Asset Instance.
        operating_speed_rpm: Velocidad operativa nominal/elegida.
        bearings_data: lista de dicts por par X-Y con:
            - bearing_label (ej. "GEN DE", "GEN NDE")
            - csv_file
            - x_gap_at_op, y_gap_at_op (mil — posición del muñón)
            - gap_unit (default "mil")
            - eccentricity_ratio (0-1)
            - attitude_angle (deg)
            - clearance_radial (mil — radio de clearance usado)
            - lift_off_speed (rpm — opcional, si se detectó)
            - trajectory_speed/x_gap/y_gap (listas downsampleadas)
        corrida_label: etiqueta humana
        notes: observaciones del usuario
    """
    if not instance_id:
        raise ValueError("instance_id requerido")
    if not bearings_data:
        raise ValueError("bearings_data vacío")

    sid = _new_scl_snapshot_id()
    ts_iso = datetime.now().isoformat(timespec="seconds")
    label = (corrida_label or "").strip() or ts_iso

    cleaned = []
    for b in bearings_data:
        try:
            entry = {
                "bearing_label": str(b.get("bearing_label", "")),
                "csv_file": str(b.get("csv_file", "")),
                "x_gap_at_op": _safe_float(b.get("x_gap_at_op")),
                "y_gap_at_op": _safe_float(b.get("y_gap_at_op")),
                "gap_unit": str(b.get("gap_unit", "mil") or "mil"),
                "eccentricity_ratio": _safe_float(b.get("eccentricity_ratio")),
                "attitude_angle": _safe_float(b.get("attitude_angle")),
                "clearance_radial": _safe_float(b.get("clearance_radial")),
                "lift_off_speed": _safe_float(b.get("lift_off_speed")),
                "csv_timestamp": str(b.get("csv_timestamp", "") or ""),
            }
            traj_speed = b.get("trajectory_speed")
            traj_x = b.get("trajectory_x_gap")
            traj_y = b.get("trajectory_y_gap")
            if (
                traj_speed is not None and traj_x is not None
                and traj_y is not None
                and len(traj_speed) == len(traj_x) == len(traj_y)
                and len(traj_speed) > 1
            ):
                entry["trajectory_speed"] = [
                    round(_safe_float(v), 2) for v in traj_speed
                ]
                entry["trajectory_x_gap"] = [
                    round(_safe_float(v), 4) for v in traj_x
                ]
                entry["trajectory_y_gap"] = [
                    round(_safe_float(v), 4) for v in traj_y
                ]
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
        "bearings": cleaned,
    }

    with open(_scl_snapshot_path(instance_id, sid), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _enforce_max_scl_snapshots(instance_id)
    return sid


def list_scl_snapshots(
    instance_id: str, limit: int = MAX_SCL_SNAPSHOTS_PER_INSTANCE,
) -> List[Dict[str, Any]]:
    """Lista snapshots SCL más recientes primero."""
    h = _scl_history_dir(instance_id)
    if not h.exists():
        return []
    items = []
    for p in sorted(h.glob("scl_*.json"), reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            items.append({
                "snapshot_id": d.get("snapshot_id", p.stem),
                "timestamp": d.get("timestamp", ""),
                "corrida_label": d.get("corrida_label", ""),
                "notes": d.get("notes", ""),
                "operating_speed_rpm": d.get("operating_speed_rpm"),
                "n_bearings": len(d.get("bearings", [])),
            })
            if len(items) >= limit:
                break
        except Exception:
            continue
    return items


def load_scl_snapshot(instance_id: str, snapshot_id: str) -> Optional[Dict[str, Any]]:
    p = _scl_snapshot_path(instance_id, snapshot_id)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_scl_snapshot(instance_id: str, snapshot_id: str) -> bool:
    p = _scl_snapshot_path(instance_id, snapshot_id)
    if not p.exists():
        return False
    try:
        p.unlink()
        return True
    except Exception:
        return False


def _enforce_max_scl_snapshots(instance_id: str) -> None:
    h = _scl_history_dir(instance_id)
    files = sorted(h.glob("scl_*.json"), reverse=True)
    for old in files[MAX_SCL_SNAPSHOTS_PER_INSTANCE:]:
        try:
            old.unlink()
        except Exception:
            pass


# ============================================================
# COMPARATIVO + SKIP IDENTICAL
# ============================================================

def _scl_snapshot_is_identical_to(
    snapshot: Dict[str, Any],
    current_by_label: Dict[str, Dict[str, float]],
    pos_tol_mil: float = 0.05,
    eccentricity_tol: float = 0.02,
) -> bool:
    """True si snapshot es esencialmente la misma corrida que current.
    Considera identicas posiciones x/y dentro de 0.05 mil y eccentricity
    dentro de 0.02 (2%)."""
    if not current_by_label:
        return False
    snap_by_label = {
        str(b.get("bearing_label", "")): b
        for b in snapshot.get("bearings", [])
    }
    if not snap_by_label:
        return False
    matched = 0
    for lbl, cur in current_by_label.items():
        if lbl not in snap_by_label:
            return False
        snap_b = snap_by_label[lbl]
        x_curr = _safe_float(cur.get("x_gap"))
        y_curr = _safe_float(cur.get("y_gap"))
        x_snap = _safe_float(snap_b.get("x_gap_at_op"))
        y_snap = _safe_float(snap_b.get("y_gap_at_op"))

        if abs(x_curr - x_snap) > pos_tol_mil:
            return False
        if abs(y_curr - y_snap) > pos_tol_mil:
            return False
        ecc_curr = _safe_float(cur.get("eccentricity_ratio"))
        ecc_snap = _safe_float(snap_b.get("eccentricity_ratio"))
        if abs(ecc_curr - ecc_snap) > eccentricity_tol:
            return False
        matched += 1
    return matched > 0


def get_previous_scl_snapshot(
    instance_id: str,
    skip_identical_to_bearings: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[Dict[str, Any]]:
    """Snapshot SCL más reciente que NO es la corrida actual."""
    snaps = list_scl_snapshots(instance_id)
    if not snaps:
        return None
    for s in snaps:
        snap = load_scl_snapshot(instance_id, s["snapshot_id"])
        if snap is None:
            continue
        if skip_identical_to_bearings is None:
            return snap
        if not _scl_snapshot_is_identical_to(snap, skip_identical_to_bearings):
            return snap
    return None


def get_scl_history_for_bearing(
    instance_id: str,
    bearing_label: str,
    max_snapshots: int = 8,
    current_reading: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Histórico de un bearing a través de snapshots SCL ordenados asc."""
    snaps = list_scl_snapshots(instance_id, limit=max_snapshots)
    points: List[Dict[str, Any]] = []
    for s in snaps:
        snap = load_scl_snapshot(instance_id, s["snapshot_id"])
        if snap is None:
            continue
        for bear in snap.get("bearings", []):
            if str(bear.get("bearing_label", "")) == bearing_label:
                points.append({
                    "timestamp": snap.get("timestamp", ""),
                    "corrida_label": snap.get("corrida_label", ""),
                    "op_speed_rpm": _safe_float(snap.get("operating_speed_rpm")),
                    "x_gap": _safe_float(bear.get("x_gap_at_op")),
                    "y_gap": _safe_float(bear.get("y_gap_at_op")),
                    "eccentricity_ratio": _safe_float(bear.get("eccentricity_ratio")),
                    "attitude_angle": _safe_float(bear.get("attitude_angle")),
                    "clearance_radial": _safe_float(bear.get("clearance_radial")),
                    "lift_off_speed": _safe_float(bear.get("lift_off_speed")),
                    "gap_unit": str(bear.get("gap_unit", "mil") or "mil"),
                })
                break

    points.sort(key=lambda p: p["timestamp"])

    if current_reading is not None:
        x_curr = _safe_float(current_reading.get("x_gap"))
        y_curr = _safe_float(current_reading.get("y_gap"))
        already = False
        if points:
            last = points[-1]
            if abs(last["x_gap"] - x_curr) < 0.01 and abs(last["y_gap"] - y_curr) < 0.01:
                already = True
        if not already:
            points.append({
                "timestamp": current_reading.get("timestamp")
                    or datetime.now().isoformat(timespec="seconds"),
                "corrida_label": current_reading.get("corrida_label", "Actual"),
                "op_speed_rpm": _safe_float(current_reading.get("op_speed_rpm")),
                "x_gap": x_curr,
                "y_gap": y_curr,
                "eccentricity_ratio": _safe_float(
                    current_reading.get("eccentricity_ratio")),
                "attitude_angle": _safe_float(
                    current_reading.get("attitude_angle")),
                "clearance_radial": _safe_float(
                    current_reading.get("clearance_radial")),
                "lift_off_speed": _safe_float(current_reading.get("lift_off_speed")),
                "gap_unit": str(current_reading.get("gap_unit", "mil") or "mil"),
            })

    return points


# ============================================================
# CLASIFICADORES DIAGNOSTICOS SCL
# ============================================================

def eccentricity_change_classifier(delta_ratio: float) -> str:
    """
    Clasifica un cambio de eccentricity ratio entre corridas:
      - stable           |Δe/c| < 0.05 (5% del clearance)
      - migration_minor  0.05 ≤ |Δe/c| < 0.15
      - migration_major  0.15 ≤ |Δe/c| < 0.25
      - migration_critical |Δe/c| >= 0.25 (cambio severo de carga
                                           o pérdida de clearance)
    """
    if delta_ratio is None:
        return "no_prev"
    try:
        d = abs(float(delta_ratio))
    except Exception:
        return "no_prev"
    if d < 0.05:
        return "stable"
    if d < 0.15:
        return "migration_minor"
    if d < 0.25:
        return "migration_major"
    return "migration_critical"


def attitude_shift_classifier(delta_deg: float) -> str:
    """
    Clasifica shift del attitude angle entre corridas:
      - stable        |Δ| < 5°
      - shift_minor   5° ≤ |Δ| < 15°
      - shift_major   15° ≤ |Δ| < 30°
      - shift_critical |Δ| ≥ 30° (cambio severo de distribución
                                  de carga / posible misalignment)
    """
    if delta_deg is None:
        return "no_prev"
    try:
        d = abs(float(delta_deg))
    except Exception:
        return "no_prev"
    d = abs((d + 540) % 360 - 180)
    if d < 5.0:
        return "stable"
    if d < 15.0:
        return "shift_minor"
    if d < 30.0:
        return "shift_major"
    return "shift_critical"


__all__ = [
    "MAX_SCL_SNAPSHOTS_PER_INSTANCE",
    "save_scl_snapshot",
    "list_scl_snapshots",
    "load_scl_snapshot",
    "delete_scl_snapshot",
    "get_scl_history_for_bearing",
    "get_previous_scl_snapshot",
    "eccentricity_change_classifier",
    "attitude_shift_classifier",
]
