"""
core/remote_monitoring/config_bridge.py
=======================================

Puente host-agnóstico de `native/field_ui/config.html` — la configuración de
máquina de 7 pasos que corre IGUAL en el campo (PySide6/QWebEngine) y en la web
(Streamlit, pages/00_Machines.py).

El HTML solo dibuja y edita; TODA la lógica vive aquí (una sola vez):
  * defaults por tipo de sensor y unidades válidas
  * construir (merge del constructor de tren, sin pisar lo que alimenta Live)
  * validación API 670 / ISO 20816 (validate_setup)
  * formato de versiones para el Summary

Mensajes host → página:  {wm:'init'|'setup'|'findings'|'versions'|'toast', ...}
Mensajes página → host:  {wm:'action', name, setup, ...}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.remote_monitoring.builder_spec import _SDEF, merge_builder_spec
from core.remote_monitoring.config import (
    UNITS_BY_TYPE,
    AcqSetup,
    MachineConfig,
    defaults_for_type,
    setup_from_dict,
    setup_to_dict,
    validate_setup,
)

STEPS = ["Machine", "Constructor", "Sensors & layout", "Channel editor",
         "Acquisition", "Validation", "Summary"]

# Banda de MEDICIÓN del Direct por familia (API 670 / ISO 7919 / ISO 20816).
MEASURE_BAND_DEFAULTS = {"prox": [1.0, 1000.0], "vel": [10.0, 1000.0], "accel": [10.0, 1000.0]}


def _norms() -> List[Dict[str, Any]]:
    """Catálogo de normas ISO/API con sus clases (para el selector de Machines)."""
    try:
        from core.iso_thresholds import list_classes_for_norm, list_norms
        out = []
        for n in list_norms():
            code = n.get("code")
            out.append({"code": code, "name": n.get("name") or n.get("title") or code,
                        "classes": [{"code": c.get("code"), "label": c.get("label"),
                                     "warning": c.get("warning"), "danger": c.get("danger")}
                                    for c in (list_classes_for_norm(code) or [])]})
        return out
    except Exception:  # noqa: BLE001
        return []


def defaults_payload() -> Dict[str, Any]:
    out: Dict[str, Any] = {"types": {}, "units": {k: list(v) for k, v in UNITS_BY_TYPE.items()},
                           "norms": _norms()}
    for t, d in _SDEF.items():
        base = defaults_for_type(t)
        out["types"][t] = {"sens": d["sens"], "gap": d["gap"], "alarm": d["alarm"],
                           "danger": d["danger"], "unit": base["unit_native"],
                           "coupling": base["coupling"]}
    return out


def normalize(setup: AcqSetup) -> Dict[str, Any]:
    """AcqSetup → dict para la página (con bandas por defecto si faltan)."""
    d = setup_to_dict(setup)
    mb = dict(MEASURE_BAND_DEFAULTS)
    mb.update(d.get("measure_band") or {})
    d["measure_band"] = mb
    return d


def blank_setup(name: str = "") -> AcqSetup:
    return AcqSetup(machine=MachineConfig(name=name, rpm_nominal=3000.0), channels=[])


def is_built(setup: AcqSetup) -> bool:
    return bool(setup.layout) or bool(setup.channels)


def init_message(setup: AcqSetup, *, host: str, library: List[Dict[str, Any]],
                 versions: List[Dict[str, Any]], current_id: str = "",
                 unlocked: Optional[int] = None, step: Optional[int] = None) -> Dict[str, Any]:
    built = is_built(setup)
    if unlocked is None:
        unlocked = len(STEPS) - 1 if built else 0
    return {"wm": "init", "host": host, "setup": normalize(setup), "library": library,
            "versions": versions_view(versions), "current": current_id,
            "defaults": defaults_payload(), "unlocked": unlocked, "built": built,
            "step": (0 if not built else 1) if step is None else step}


def handle_build(setup_dict: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    new, rep = merge_builder_spec(setup_from_dict(setup_dict), spec or {})
    return {"wm": "setup", "setup": normalize(new), "report": rep}


def findings(setup_dict: Dict[str, Any]) -> List[Dict[str, str]]:
    setup = setup_from_dict(setup_dict)
    out = [{"level": f.level, "code": f.code, "message": f.message} for f in validate_setup(setup)]
    # Migración Modbus → NI: una sonda que hoy alimenta Live por Modbus y queda
    # sin variable al pasar a NI corta su tendencia.
    if setup.source == "ni":
        for ch in setup.channels:
            if ch.modbus and not ch.variable:
                out.insert(0, {"level": "warn", "code": "live_variable",
                               "message": f"{ch.point_label.replace('_', '')}: no Live variable — "
                                          f"its history from the rack will not continue."})
    return out


def n_errors(fs: List[Dict[str, str]]) -> int:
    return sum(1 for f in fs if f.get("level") == "error")


def versions_view(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows or []:
        when = str(r.get("published_at") or r.get("created_at") or "")[:16].replace("T", " ")
        node = r.get("node_status") or ""
        if node and r.get("node_at"):
            node += " " + str(r["node_at"])[11:16]
        out.append({"v": r.get("version"), "st": r.get("status"), "org": r.get("origin"),
                    "by": r.get("published_by") or r.get("author") or "", "when": when,
                    "note": r.get("note") or "", "node": node,
                    "base": r.get("base_version")})
    return out
