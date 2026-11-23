"""
core/remote_monitoring/builder_spec.py
======================================

Constructor de tren (native/field_ui/builder.html) → config de máquina.

El MISMO builder.html corre en el campo (PySide6/QWebEngine, piel oscura) y en
la web (Machines, componente Streamlit, piel clara). Ambos emiten el mismo spec:

    {machines:[{type,name,bearings,bearing_type,axial,axial_brg,casing,kph,bno}],
     keyphasors, aux, sensors:[{n,k,ang,comp,rack}]}

Este módulo es la traducción ÚNICA spec → ChannelRow (antes vivía solo dentro
de la GUI de campo) y el MERGE con una máquina existente, que NUNCA pisa lo
que ya alimenta Live (variable, origen Modbus, límites): solo agrega sondas
nuevas y reporta las que el tren ya no tiene.

Convención de labels (CLAUDE.md): en datos con underscore (`1X_D`), en display
sin underscore (`1XD`). El builder emite el display → aquí se canoniza.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

from core.remote_monitoring.config import AcqSetup, ChannelRow, defaults_for_type

KIND_TO_TYPE = {"prox": "proximity", "accel": "accelerometer",
                "vel": "velometer", "kph": "keyphasor"}
# Defaults de campo (mismos que native/watermelon_field.py _SDEF):
# prox 200 mV/mil, accel/vel 100 mV/unidad; límites API 670 / ISO 20816.
_SDEF = {
    "proximity":     dict(sens=200.0, gap=-9.5, alarm=2.5, danger=4.0),
    "accelerometer": dict(sens=100.0, gap=0.0, alarm=4.5, danger=7.1),
    "velometer":     dict(sens=100.0, gap=0.0, alarm=4.5, danger=7.1),
    "keyphasor":     dict(sens=1.0, gap=0.0, alarm=0.0, danger=0.0),
}

_RE_BEARING = re.compile(r"^(\d+)([XYHVZ])D([ab]?)$", re.IGNORECASE)
# Sondas de carcasa/aux por tipo: 4YA (acel), 4YV (vel) → 4Y_A, 4Y_V (labels del rack)
_RE_TYPED = re.compile(r"^(\d+)([XYHVZ])([AV])$", re.IGNORECASE)


def canonical_label(display: str) -> str:
    """'1XD' → '1X_D', '2ZDa' → '2Z_Da'; el resto queda igual (KPH, 1A, aux)."""
    s = (display or "").strip()
    m = _RE_BEARING.match(s)
    if m:
        return f"{m.group(1)}{m.group(2).upper()}_D{m.group(3).lower()}"
    m = _RE_TYPED.match(s)
    if m:
        return f"{m.group(1)}{m.group(2).upper()}_{m.group(3).upper()}"
    return s


def display_label(label: str) -> str:
    """'1X_D' → '1XD' (display sin underscore)."""
    return (label or "").replace("_", "")


def _angle(ang: str) -> Tuple[float, str]:
    a = str(ang or "")
    side = "R" if "R" in a else ("L" if "L" in a else "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*°", a)
    return (float(m.group(1)) if (m and side) else 0.0), side


def _direction(label: str) -> str:
    m = _RE_BEARING.match(display_label(label))
    return m.group(2).upper() if m else ""


def _plane(label: str) -> int:
    m = re.match(r"^(\d+)", label or "")
    return int(m.group(1)) if m else 0


def spec_to_channels(spec: Dict[str, Any]) -> List[ChannelRow]:
    """Sondas del builder → ChannelRow en orden canónico (keyphasor → cojinetes
    1..N Y/X → auxiliares), BNC secuencial 1..N (igual que el campo)."""
    rows: List[Tuple[Tuple, ChannelRow]] = []
    for i, s in enumerate(spec.get("sensors") or []):
        st = KIND_TO_TYPE.get(str(s.get("k") or "accel"), "accelerometer")
        label = canonical_label(str(s.get("n") or f"CH{i + 1}"))
        deg, side = _angle(s.get("ang", ""))
        d = _SDEF[st]
        base = defaults_for_type(st)
        ch = ChannelRow(bnc_port=0, point_label=label, plane=_plane(label), sensor_type=st,
                        sensitivity_mv_per_eu=d["sens"], unit_native=base["unit_native"],
                        coupling=base["coupling"], angle_deg=deg, side=side,
                        alarm=d["alarm"], danger=d["danger"], gap_bias_v=d["gap"],
                        location=str(s.get("comp") or ""), direction=_direction(label))
        disp = display_label(label).upper()
        if st == "keyphasor":
            key = (0, i)
        elif disp[:1].isdigit():
            axis = 0 if "Y" in disp[1:3] else (1 if "X" in disp[1:3] else 2)
            key = (1, _plane(label), axis, disp)
        else:
            key = (2, i)
        rows.append((key, ch))
    out = [ch for _, ch in sorted(rows, key=lambda t: t[0])]
    for n, ch in enumerate(out, start=1):
        ch.bnc_port = n
    return out


def merge_builder_spec(setup: AcqSetup, spec: Dict[str, Any],
                       add_new: Optional[bool] = None) -> Tuple[AcqSetup, Dict[str, List[str]]]:
    """Aplica un build del constructor a una máquina existente SIN pisar lo que
    alimenta Live. Devuelve (setup nuevo, reporte {added, matched, not_in_train}).

    - Sonda del tren que ya existe (mismo label canónico) → se conserva TAL CUAL
      (variable, origen Modbus, límites, ángulos reales de montaje).
    - Sonda nueva → se agrega con defaults de campo, BNC libre siguiente.
    - Canal existente que el tren ya no tiene → se conserva y se reporta (nunca
      se borra en silencio; quitarlo es decisión explícita en la tabla).
    - Máquina que lee del RACK (canales con origen Modbus): el rack define qué
      sondas existen → las sondas nuevas del tren NO se agregan, se reportan como
      `suggested` (el analista las agrega a mano si de verdad existen).
      `add_new` fuerza el comportamiento.
    """
    if add_new is None:
        add_new = not any(c.modbus for c in setup.channels)
    new = copy.deepcopy(setup)
    built = spec_to_channels(spec)
    have = {c.point_label.upper(): c for c in new.channels}
    used_bnc = {c.bnc_port for c in new.channels}
    nxt = (max(used_bnc) if used_bnc else 0) + 1
    added, matched, suggested = [], [], []
    for ch in built:
        if ch.point_label.upper() in have:
            matched.append(ch.point_label)
            continue
        if not add_new:
            if ch.sensor_type != "keyphasor":
                suggested.append(ch.point_label)
            continue
        ch.bnc_port = nxt
        nxt += 1
        new.channels.append(ch)
        added.append(ch.point_label)
    in_train = {c.point_label.upper() for c in built}
    # Velocidad Modbus (SPEEDn) = keyphasor del rack: cuenta como presente si el
    # tren trae algún keyphasor.
    has_kph = any(c.sensor_type == "keyphasor" for c in built)
    not_in_train = [c.point_label for c in setup.channels
                    if c.point_label.upper() not in in_train
                    and not (has_kph and c.sensor_type == "keyphasor")]
    new.layout = dict(spec)
    if not new.machine.name and spec.get("machines"):
        new.machine.name = spec["machines"][0].get("name") or new.machine.name
    bts = {m.get("bearing_type") for m in spec.get("machines") or []}
    if bts:
        new.machine.bearing_type = ("mixed" if len(bts) > 1 else
                                    "rolling" if "rolling" in bts else "plain")
    new.machine.n_bearings = sum(int(m.get("bearings") or 0) for m in spec.get("machines") or []) \
        or new.machine.n_bearings
    return new, {"added": added, "matched": matched, "not_in_train": not_in_train,
                 "suggested": suggested}
