"""
core/live_train.py
==================

Hero de Live Monitoring = el TREN dibujado en Machines (builder.html modo #live).

Si la máquina tiene tren dibujado en Machines (rm_setups.metadata.layout con
máquinas), Live muestra ese tren con cada sonda en vivo (valor, ángulo, % de la
alerta) y los auxiliares en su caja. Si no, Live sigue con su dibujo actual
(migración máquina por máquina).

`live_payload` es puro (testeable): latest_for_instance → {probes, speeds}.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.remote_monitoring.config import AcqSetup

_KIND = {"proximity": "prox", "velometer": "vel", "accelerometer": "accel"}


def machines_train_for(instance_id: str, client: Any = None) -> Optional[Tuple[AcqSetup, Dict[str, Any]]]:
    """(setup, layout) publicado en Machines para esta instancia, o None si no
    tiene tren dibujado."""
    from core.remote_monitoring.config import find_setup_cloud_by_instance
    found = find_setup_cloud_by_instance(instance_id, client=client)
    if not found:
        return None
    setup = found["setup"]
    layout = setup.layout or {}
    if not layout.get("machines"):
        return None
    return setup, layout


def _sev(v: Optional[float], alarm: float, danger: float) -> str:
    if v is None:
        return "off"
    if danger and v >= danger:
        return "danger"
    if alarm and v >= alarm:
        return "alert"
    return "ok"


def live_payload(setup: AcqSetup, latest: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Últimas lecturas (live_readings) → datos para builder.html#live."""
    by_label: Dict[Tuple[str, str], float] = {}
    by_var: Dict[str, float] = {}
    for r in latest or []:
        if r.get("value") is None:
            continue
        lbl, met = r.get("sensor_label"), r.get("metric") or "Direct"
        if lbl:
            by_label[(str(lbl), met)] = float(r["value"])
        elif met == "Direct" and r.get("variable"):
            by_var[str(r["variable"])] = float(r["value"])
    probes, speeds = [], []
    for ch in setup.channels:
        if ch.sensor_type == "keyphasor":
            v = by_var.get(ch.variable)
            speeds.append({"label": ch.variable or ch.point_label,
                           "v": round(v) if v is not None else None, "u": "rpm"})
            continue
        v = by_label.get((ch.point_label, "Direct"))
        x1 = by_label.get((ch.point_label, "1X_Ampl"))
        ph = by_label.get((ch.point_label, "1X_Phase"))
        probes.append({
            "n": ch.point_label.replace("_", ""), "brg": ch.plane,
            "k": _KIND.get(ch.sensor_type, "accel"),
            "v": round(v, 3) if v is not None else None, "u": ch.unit_native,
            "sev": _sev(v, ch.alarm, ch.danger), "alarm": ch.alarm, "danger": ch.danger,
            "ang": f"{ch.angle_deg:g}°{ch.side}" if ch.side else "",
            "x1": round(x1, 2) if x1 is not None else None,
            "ph": round(ph) if ph is not None else None,
            "derived_from": ch.derived_from.replace("_", "") if ch.derived_from else "",
        })
    return {"probes": probes, "speeds": speeds}
