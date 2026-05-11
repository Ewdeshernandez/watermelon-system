"""
core.polar_history
==================

Persistencia y comparativo de snapshots históricos del módulo Polar
(Ciclo 17.1). Para cada corrida snapshoteada captura, por sensor del
Sensor Map matched, el 1X amplitude + fase a la velocidad operativa
seleccionada por el usuario.

Por qué importa: el shift de fase 1X entre corridas es **el** indicador
clásico de cambio de balance del rotor. Un cambio >30° entre dos
corridas a la misma velocidad operativa es síntoma de masa perdida o
crack en el rotor (API 684, ISO 21940-12). El crecimiento de amplitud
1X sin shift de fase apunta a desbalance progresivo.

Estructura en disco (backend local):

    {INSTANCES_DIR}/{instance_id}/history/polar_{ISO8601}.json

Cada snapshot:

  - snapshot_id          str    polar_{timestamp}
  - instance_id          str
  - timestamp            str    ISO8601 cuando se snapshoteó
  - corrida_label        str    Etiqueta humana
  - notes                str    Observaciones
  - operating_speed_rpm  float  Velocidad operativa elegida
  - sensors              list   Por sensor matched:
        sensor_label, csv_file, amp_at_op, phase_at_op,
        amp_unit, phase_unit, csv_timestamp.

Funciones:

  - save_polar_snapshot(...)        → snapshot_id
  - list_polar_snapshots(...)       → list[dict]
  - load_polar_snapshot(...)        → dict | None
  - delete_polar_snapshot(...)      → bool
  - get_polar_history_for_sensor(instance_id, sensor_label, max=8,
        current_reading=None) → list[{timestamp, corrida_label,
        op_speed, amp, phase, unit}]
  - phase_shift_classifier(delta_deg) → "stable"/"shift_minor"/
        "shift_major"/"shift_critical"
  - amplitude_change_classifier(delta_pct) → similar a Tabular trend
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ciclo 23.79: storage delegado a history_storage (Supabase Storage + gzip
# + LRU). API pública preservada al 100%.
from core import history_storage

_SNAPSHOT_TYPE = "polar"

# Compat exportada — el LRU real lo aplica history_storage (default 10).
MAX_POLAR_SNAPSHOTS_PER_INSTANCE = history_storage.MAX_SNAPSHOTS_PER_TYPE


def _new_polar_snapshot_id() -> str:
    return history_storage.new_snapshot_id(_SNAPSHOT_TYPE)


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

def save_polar_snapshot(
    instance_id: str,
    *,
    operating_speed_rpm: float,
    sensors_data: List[Dict[str, Any]],
    corrida_label: str = "",
    notes: str = "",
) -> str:
    """
    Guarda un snapshot del estado Polar de la instancia.

    Args:
        instance_id: ID de la Asset Instance.
        operating_speed_rpm: Velocidad operativa nominal/elegida.
        sensors_data: lista de dicts con:
            - sensor_label  (ej. "1X_D") — del Sensor Map matched
            - csv_file      (nombre del CSV original)
            - amp_at_op     (float)
            - phase_at_op   (float, en grados)
            - amp_unit      (ej. "mil pp")
            - phase_unit    (ej. "deg")
            - csv_timestamp (ISO opcional, del metadata del CSV)
        corrida_label: etiqueta humana, default = ISO timestamp.
        notes: observaciones del usuario.

    Returns:
        snapshot_id (string).
    """
    if not instance_id:
        raise ValueError("instance_id requerido")
    if not sensors_data:
        raise ValueError("sensors_data vacío")

    sid = _new_polar_snapshot_id()
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
            # Ciclo 17.1.2 — trayectoria completa downsampleada
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
            # Velocidad crítica + Q de la corrida (si la hay)
            cs_rpm = s.get("critical_speed_rpm")
            if cs_rpm is not None:
                entry["critical_speed_rpm"] = _safe_float(cs_rpm)
                entry["critical_speed_amp"] = _safe_float(
                    s.get("critical_speed_amp"))
                entry["critical_speed_phase"] = _safe_float(
                    s.get("critical_speed_phase"))
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

    # Ciclo 23.79: storage delegado al backend (LRU automático).
    history_storage.save_snapshot(instance_id, _SNAPSHOT_TYPE, sid, payload)
    return sid


def list_polar_snapshots(
    instance_id: str, limit: int = MAX_POLAR_SNAPSHOTS_PER_INSTANCE,
) -> List[Dict[str, Any]]:
    """Lista snapshots Polar más recientes primero, con metadata
    resumida (sin sensors detallados, para eficiencia)."""
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
        })
    return items


def load_polar_snapshot(instance_id: str, snapshot_id: str) -> Optional[Dict[str, Any]]:
    return history_storage.load_snapshot(instance_id, _SNAPSHOT_TYPE, snapshot_id)


def delete_polar_snapshot(instance_id: str, snapshot_id: str) -> bool:
    return history_storage.delete_snapshot(instance_id, _SNAPSHOT_TYPE, snapshot_id)


# ============================================================
# POR SENSOR — para charts y comparativos
# ============================================================

def get_polar_history_for_sensor(
    instance_id: str,
    sensor_label: str,
    max_snapshots: int = 8,
    current_reading: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Devuelve los puntos históricos de un sensor a través de los
    snapshots Polar, ordenados cronológicamente (asc).

    Si se pasa current_reading (dict con amp_at_op, phase_at_op,
    op_speed, amp_unit), se anexa al final como punto "actual" si
    no es identico al último snapshot.

    Cada punto:
      timestamp, corrida_label, op_speed_rpm, amp, phase, unit.
    """
    snaps = list_polar_snapshots(instance_id, limit=max_snapshots)
    points: List[Dict[str, Any]] = []
    for s in snaps:
        snap = load_polar_snapshot(instance_id, s["snapshot_id"])
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
                })
                break

    points.sort(key=lambda p: p["timestamp"])

    if current_reading is not None:
        cur_amp = _safe_float(current_reading.get("amp_at_op"))
        cur_phase = _safe_float(current_reading.get("phase_at_op"))
        already = False
        if points:
            last = points[-1]
            # Considera identico si amp diff <1% y phase diff <1°
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
            })

    return points


def get_previous_polar_snapshot(
    instance_id: str,
    skip_identical_to_sensors: Optional[Dict[str, Dict[str, float]]] = None,
    identical_amp_tol_pct: float = 1.0,
    identical_phase_tol_deg: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """
    Como get_previous_snapshot del Tabular: devuelve el snapshot Polar
    más reciente que NO es esencialmente la corrida actual.

    Args:
        skip_identical_to_sensors: dict ``{sensor_label: {"amp": x,
            "phase": y}}`` con los valores actuales. El snapshot se
            considera identico si TODOS los sensores tienen amp diff
            <= identical_amp_tol_pct% y phase diff <= identical_phase_tol_deg.
    """
    snaps = list_polar_snapshots(instance_id)
    if not snaps:
        return None
    for s in snaps:
        snap = load_polar_snapshot(instance_id, s["snapshot_id"])
        if snap is None:
            continue
        if skip_identical_to_sensors is None:
            return snap
        if not _polar_snapshot_is_identical_to(
            snap, skip_identical_to_sensors,
            identical_amp_tol_pct, identical_phase_tol_deg,
        ):
            return snap
    return None


def _polar_snapshot_is_identical_to(
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
        # circular phase diff
        diff = abs((ph_curr - ph_snap + 540) % 360 - 180)
        if diff > phase_tol_deg:
            return False
        matched += 1
    return matched > 0


# ============================================================
# CLASIFICADORES DIAGNOSTICOS
# ============================================================

def phase_shift_classifier(delta_deg: float) -> str:
    """
    Clasifica un shift de fase 1X entre corridas según severidad
    diagnóstica:
      - stable        |Δ| <  10° (variación normal)
      - shift_minor   10° ≤ |Δ| < 30° (cambio que vale la pena vigilar)
      - shift_major   30° ≤ |Δ| < 60° (síntoma probable de cambio
                                      de balance / masa perdida)
      - shift_critical |Δ| >= 60° (degradación severa o crack)

    Usa shortest-arc circular distance (siempre ≤ 180°).
    """
    if delta_deg is None:
        return "no_prev"
    try:
        d = abs(float(delta_deg))
    except Exception:
        return "no_prev"
    # Normalizar a [0, 180]
    d = abs((d + 540) % 360 - 180)
    if d < 10.0:
        return "stable"
    if d < 30.0:
        return "shift_minor"
    if d < 60.0:
        return "shift_major"
    return "shift_critical"


def amplitude_change_classifier(delta_pct: Optional[float]) -> str:
    """Como el de Tabular pero con strings específicos del Polar."""
    if delta_pct is None:
        return "no_prev"
    if delta_pct >= 50.0:
        return "amp_critical"
    if delta_pct >= 20.0:
        return "amp_high"
    if delta_pct >= 5.0:
        return "amp_up"
    if delta_pct <= -20.0:
        return "amp_down_strong"
    if delta_pct <= -5.0:
        return "amp_down"
    return "amp_stable"


def shortest_arc_phase_diff(p1: float, p2: float) -> float:
    """Diferencia de fase circular en grados (firma signed). Resultado
    en [-180, 180] indicando dirección del shift."""
    diff = (p2 - p1 + 540) % 360 - 180
    return diff


__all__ = [
    "MAX_POLAR_SNAPSHOTS_PER_INSTANCE",
    "save_polar_snapshot",
    "list_polar_snapshots",
    "load_polar_snapshot",
    "delete_polar_snapshot",
    "get_polar_history_for_sensor",
    "get_previous_polar_snapshot",
    "phase_shift_classifier",
    "amplitude_change_classifier",
    "shortest_arc_phase_diff",
]
