"""
core/remote_monitoring/instance_sync.py
=======================================

Puente de TRANSICIÓN Machines (AcqSetup) ↔ Machinery Library (`instances`).

Live Monitoring, el Machine Map y los reportes todavía leen los sensores de
`instances`. Mientras se migran esos lectores, Machines escribe también ahí.
Reglas (nunca romper lo que hoy funciona):

  * El match es por label industrial (`1Y_D`) — `sensor_map.sensor_label()`.
  * `enrich_setup_from_sensors`: trae a la config lo que solo vivía en la
    Machinery Library (límites alarma/peligro, ángulo/lado de montaje). Solo
    rellena valores VACÍOS (0 / "") de la config.
  * `sync_sensors_from_setup`: al publicar, actualiza límites y ángulos de las
    sondas que ya existen (solo con valores > 0 de la config) y AGREGA las
    nuevas. NUNCA borra una sonda de `instances`.

Lógica pura sobre listas de dicts (testeable); `sync_instance` hace el I/O.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from core.remote_monitoring.config import AcqSetup
from core.sensor_map import new_sensor, sensor_label

_TO_INST_TYPE = {"proximity": "proximity", "velometer": "velocity",
                 "accelerometer": "accelerometer", "keyphasor": "keyphasor"}


def _is_speed(ch) -> bool:
    return ch.sensor_type == "keyphasor" and ch.point_label.upper().startswith("SPEED")


def _index(sensors: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, s in enumerate(sensors):
        out.setdefault(sensor_label(s).upper(), i)
    return out


def enrich_setup_from_sensors(setup: AcqSetup, sensors: List[Dict[str, Any]]
                              ) -> Tuple[AcqSetup, Dict[str, List[str]]]:
    new = copy.deepcopy(setup)
    idx = _index(sensors)
    enriched, only_setup = [], []
    for ch in new.channels:
        i = idx.get(ch.point_label.upper())
        if i is None:
            if not _is_speed(ch):
                only_setup.append(ch.point_label)
            continue
        s = sensors[i]
        before = (ch.alarm, ch.danger, ch.angle_deg, ch.side, ch.location)
        if not ch.alarm and float(s.get("alarm") or 0) > 0:
            ch.alarm = float(s["alarm"])
        if not ch.danger and float(s.get("danger") or 0) > 0:
            ch.danger = float(s["danger"])
        side = str(s.get("side") or "").strip().upper()
        if not ch.angle_deg and float(s.get("angle_deg") or 0) > 0 and side in ("L", "R"):
            ch.angle_deg, ch.side = float(s["angle_deg"]), side
        if not ch.location and s.get("plane_label"):
            ch.location = str(s["plane_label"]).strip()
        if (ch.alarm, ch.danger, ch.angle_deg, ch.side, ch.location) != before:
            enriched.append(ch.point_label)
    labels = {c.point_label.upper() for c in new.channels}
    only_inst = [sensor_label(s) for s in sensors if sensor_label(s).upper() not in labels
                 and str(s.get("sensor_type") or "") != "keyphasor"]
    return new, {"enriched": enriched, "only_in_config": only_setup, "only_in_library": only_inst}


def sync_sensors_from_setup(sensors: List[Dict[str, Any]], setup: AcqSetup
                            ) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    out = copy.deepcopy(sensors or [])
    idx = _index(out)
    updated, added, label_mismatch = [], [], []
    for ch in setup.channels:
        if _is_speed(ch):
            continue
        i = idx.get(ch.point_label.upper())
        if i is not None:
            s = out[i]
            before = dict(s)
            if ch.alarm > 0:
                s["alarm"] = float(ch.alarm)
            if ch.danger > 0:
                s["danger"] = float(ch.danger)
            if ch.angle_deg > 0 and ch.side in ("L", "R"):
                s["angle_deg"], s["side"] = float(ch.angle_deg), ch.side
            if s != before:
                updated.append(ch.point_label)
            continue
        disp = ch.point_label.replace("_", "")
        sens = new_sensor(
            plane=int(ch.plane or 0),
            plane_label=f"{disp} {ch.location}".strip(),
            side=ch.side or "",
            angle_deg=float(ch.angle_deg or 0),
            direction=(ch.direction or "").upper(),
            sensor_type=_TO_INST_TYPE.get(ch.sensor_type, ch.sensor_type),
            unit_native=ch.unit_native,
            alarm=float(ch.alarm or 0),
            danger=float(ch.danger or 0),
            csv_match_pattern=disp,
            sensitivity_mv_per_eu=float(ch.sensitivity_mv_per_eu or 0) or None,
            coupling=ch.coupling or "",
        )
        if sensor_label(sens).upper() != ch.point_label.upper():
            label_mismatch.append(f"{ch.point_label}→{sensor_label(sens)}")
        out.append(sens)
        idx[sensor_label(sens).upper()] = len(out) - 1
        added.append(ch.point_label)
    return out, {"updated": updated, "added": added, "label_mismatch": label_mismatch}


# Ficha del activo: Machines → instances. Live (título y chips), reporte PDF
# (subtítulo cliente · sitio), app móvil (flota por cliente) y correos la leen
# de `instances`. Solo valores NO vacíos (nunca borra lo que ya hay).
_HEADER_MAP = (("client", "client"), ("location", "site"), ("tag", "tag"),
               ("iso_norm", "iso_norm_code"), ("iso_class", "iso_norm_class"))


def header_changes(inst: Any, setup: AcqSetup) -> Dict[str, Tuple[str, str]]:
    """{campo_instance: (antes, después)} que el publish aplicaría."""
    out: Dict[str, Tuple[str, str]] = {}
    for src, dst in _HEADER_MAP:
        new = (getattr(setup.machine, src, "") or "").strip()
        old = (getattr(inst, dst, "") or "") if inst is not None else ""
        if new and new != old:
            out[dst] = (old, new)
    # Operación: rpm nominal (Machines = fuente única)
    rpm = float(getattr(setup.machine, "rpm_nominal", 0) or 0)
    old_rpm = float(getattr(inst, "nominal_rpm", 0) or 0) if inst is not None else 0.0
    if rpm > 0 and abs(rpm - old_rpm) > 0.5:
        out["nominal_rpm"] = (f"{old_rpm:g}" if old_rpm else "", f"{rpm:g}")
    return out


def sync_instance(setup: AcqSetup, create_if_missing: bool = True) -> Dict[str, Any]:
    """Escribe la config publicada en `instances` (transición). Nunca borra."""
    if not setup.instance_id:
        return {"ok": False, "reason": "sin instance_id"}
    from core.instance_state import _save_instance, create_instance, get_instance
    inst = get_instance(setup.instance_id)
    created = False
    if inst is None:
        if not create_if_missing:
            return {"ok": False, "reason": "no existe en Machinery Library"}
        inst = create_instance(instance_id=setup.instance_id, profile_key="custom_manual",
                               tag=setup.machine.tag or setup.machine.name,
                               seed_from_profile=False)
        created = True
    sensors, rep = sync_sensors_from_setup(inst.sensors, setup)
    hdr = header_changes(inst, setup)
    if not created and not rep["updated"] and not rep["added"] and not hdr:
        return {"ok": True, "created": False, "header": {}, **rep}
    inst.sensors = sensors
    for field_name, (_old, new) in hdr.items():
        setattr(inst, field_name, float(new) if field_name == "nominal_rpm" else new)
    _save_instance(inst)
    return {"ok": True, "created": created, "header": {k: v[1] for k, v in hdr.items()}, **rep}
