"""
core.scl_history
================

Persistencia y comparativo de snapshots históricos del módulo Shaft
Centerline (Ciclo 17.3, refactorizado en 23.78).

Para cada corrida snapshoteada captura, por **par de sondas X-Y** del
cojinete:

  - x_gap, y_gap (posición del muñón a velocidad operativa, en mil)
  - eccentricity_ratio (0-1) — distancia desde centro / clearance
  - attitude_angle (deg) — ángulo de la posición respecto a load
  - bearing clearance utilizado para los cálculos
  - trayectoria completa speed/x_gap/y_gap (lift-off curve)

Permite ver superpuestas las trayectorias del muñón entre corridas:

  - **Migración del centerline**
  - **Cambio de eccentricity ratio**
  - **Shift del attitude angle**
  - **Lift-off speed evolution** (API 670 §6.7)

Storage (Ciclo 23.78): delegado a ``core.history_storage`` que abstrae
Supabase Storage (backend principal) + fallback a disco local. La API
pública de este módulo NO cambia — los callers (`pages/09_Shaft_Centerline.py`)
siguen funcionando idénticos.

Reusa los clasificadores de polar_history para amplitude/phase
classifiers cuando aplique.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core import history_storage

# Compat: la app vieja exportaba esta constante. Mantenemos el símbolo
# pero ahora el LRU real lo aplica history_storage (default 10).
MAX_SCL_SNAPSHOTS_PER_INSTANCE = history_storage.MAX_SNAPSHOTS_PER_TYPE

_SNAPSHOT_TYPE = "scl"


# ============================================================
# HELPERS internos (preservados de la versión anterior)
# ============================================================

def _new_scl_snapshot_id() -> str:
    """Compat: el caller histórico usa este nombre. Delega a history_storage."""
    return history_storage.new_snapshot_id(_SNAPSHOT_TYPE)


def _safe_float(v) -> float:
    try:
        f = float(v)
        if f != f:  # NaN check
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

    Returns:
        snapshot_id (str)
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

    # Ciclo 23.78: delegar al backend. LRU rotation se aplica solo.
    history_storage.save_snapshot(instance_id, _SNAPSHOT_TYPE, sid, payload)
    return sid


def list_scl_snapshots(
    instance_id: str,
    limit: int = MAX_SCL_SNAPSHOTS_PER_INSTANCE,
) -> List[Dict[str, Any]]:
    """Lista snapshots SCL más recientes primero.

    Devuelve metadata resumida (sin el payload completo) para que la
    UI pueda armar el listado sin descargar TODAS las trayectorias.
    """
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
        })
    return items


def load_scl_snapshot(instance_id: str, snapshot_id: str) -> Optional[Dict[str, Any]]:
    return history_storage.load_snapshot(instance_id, _SNAPSHOT_TYPE, snapshot_id)


def delete_scl_snapshot(instance_id: str, snapshot_id: str) -> bool:
    return history_storage.delete_snapshot(instance_id, _SNAPSHOT_TYPE, snapshot_id)


# ============================================================
# COMPARATIVO + SKIP IDENTICAL
# (preservado idéntico — no toca storage, solo lógica de comparación)
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
    """Histórico de un bearing a través de snapshots SCL ordenados asc.

    Args:
        instance_id: Asset Instance ID.
        bearing_label: ej. "GEN DE".
        max_snapshots: máximo histórico a devolver (incluye current si pasa).
        current_reading: lectura actual opcional con keys x_gap, y_gap,
            eccentricity_ratio, attitude_angle, csv_timestamp. Si se pasa,
            se incluye como último elemento del retorno.

    Returns:
        Lista de dicts ordenados de más antiguo a más reciente:
            [
                {snapshot_id, timestamp, corrida_label, op_speed,
                 x_gap, y_gap, ecc, attitude, clearance, csv_timestamp,
                 is_current (solo si current_reading provisto)},
                ...
            ]
    """
    snaps_meta = list_scl_snapshots(instance_id, limit=max_snapshots)
    if not snaps_meta and current_reading is None:
        return []

    items: List[Dict[str, Any]] = []
    # Construir histórico desde snapshots (orden inverso para acabar asc)
    for s in reversed(snaps_meta):
        snap = load_scl_snapshot(instance_id, s["snapshot_id"])
        if snap is None:
            continue
        bearing_data = next(
            (b for b in snap.get("bearings", [])
             if str(b.get("bearing_label", "")) == bearing_label),
            None,
        )
        if bearing_data is None:
            continue
        items.append({
            "snapshot_id": snap.get("snapshot_id"),
            "timestamp": snap.get("timestamp", ""),
            "corrida_label": snap.get("corrida_label", ""),
            "op_speed": snap.get("operating_speed_rpm"),
            "x_gap": bearing_data.get("x_gap_at_op"),
            "y_gap": bearing_data.get("y_gap_at_op"),
            "ecc": bearing_data.get("eccentricity_ratio"),
            "attitude": bearing_data.get("attitude_angle"),
            "clearance": bearing_data.get("clearance_radial"),
            "csv_timestamp": bearing_data.get("csv_timestamp", ""),
            "is_current": False,
        })

    # Agregar current_reading al final si existe
    if current_reading is not None:
        items.append({
            "snapshot_id": "_current",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "corrida_label": "Corrida actual",
            "op_speed": current_reading.get("op_speed"),
            "x_gap": _safe_float(current_reading.get("x_gap")),
            "y_gap": _safe_float(current_reading.get("y_gap")),
            "ecc": _safe_float(current_reading.get("eccentricity_ratio")),
            "attitude": _safe_float(current_reading.get("attitude_angle")),
            "clearance": _safe_float(current_reading.get("clearance_radial")),
            "csv_timestamp": str(current_reading.get("csv_timestamp", "")),
            "is_current": True,
        })

    return items
