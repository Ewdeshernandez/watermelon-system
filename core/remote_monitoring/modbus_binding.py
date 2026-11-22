"""
core/remote_monitoring/modbus_binding.py
========================================

ORIGEN Modbus de una máquina (rack de protección existente, ej. Bently 3500/92).

La máquina se define UNA vez como `AcqSetup` (sensores, ángulos, límites).
Cada canal puede leerse de dos orígenes intercambiables:

    * NI      → bnc_port / módulo (nodo Watermelon, onda cruda)
    * Modbus  → ChannelRow.modbus = {metric: {"register": 6005, ...}}

El mapa JSON que consume `collector/wm_collector.py` deja de escribirse a mano:
se GENERA con `modbus_map_from_setup()`. Para migrar los mapas existentes
(data/modbus_maps/*.json) está `setup_from_modbus_map()`; el par es
round-trip sin pérdida (tests/test_modbus_binding.py).

Contrato que NO cambia al pasar de Modbus a NI (continuidad del histórico):
    instance_id + sensor_label (= point_label) + metric + variable

Core puro: sin Streamlit, sin red.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.remote_monitoring.config import AcqSetup, ChannelRow, MachineConfig

# sensor_type del mapa Modbus ↔ sensor_type del AcqSetup
_MAP_TO_SETUP_TYPE = {"proximity": "proximity", "velocity": "velometer",
                      "velometer": "velometer", "accelerometer": "accelerometer",
                      "keyphasor": "keyphasor"}
_SETUP_TO_MAP_TYPE = {"proximity": "proximity", "velometer": "velocity",
                      "accelerometer": "accelerometer", "keyphasor": "keyphasor"}

# Campos de registro que se elevan al canal (no se repiten por registro).
_CHANNEL_FIELDS = ("sensor_label", "plane", "direction", "sensor_type", "location", "variable")
# Claves de primer nivel del mapa que NO van al gateway.
_TOP_RESERVED = ("instance_id", "registers")

SPEED_PREFIX = "SPEED"


def _speed_label(i: int) -> str:
    return f"{SPEED_PREFIX}{i}"


def setup_from_modbus_map(raw: Dict[str, Any], name: str = "",
                          mode: str = "online") -> AcqSetup:
    """Mapa Modbus del collector (dict) → AcqSetup con origen Modbus.

    - Registros con `sensor_label` se agrupan en un canal por sensor.
    - Registros `kind=speed` (sin sensor_label) → canal keyphasor (en NI la
      velocidad sale del keyphasor; en Modbus, del registro de velocidad).
    - Todo lo demás del mapa (gateway, intervalos, notas) → modbus_gateway.
    """
    regs: Dict[str, Dict[str, Any]] = raw.get("registers") or {}
    channels: Dict[str, ChannelRow] = {}
    order: List[str] = []
    n_speed = 0

    def _addr_key(a: str) -> Tuple[int, str]:
        try:
            return (int(a), a)
        except ValueError:
            return (1 << 30, a)

    for addr in sorted(regs, key=_addr_key):
        spec = dict(regs[addr])
        label = spec.get("sensor_label")
        metric = str(spec.get("metric") or "Direct")
        if not label:
            # Velocidad (o registro suelto sin sensor): canal propio para no
            # perderlo en el round-trip; `kind` original se conserva.
            n_speed += 1
            label = _speed_label(n_speed)
            ch = ChannelRow(bnc_port=len(order) + 1, point_label=label,
                            plane=0, sensor_type="keyphasor",
                            unit_native=str(spec.get("unit") or "rpm"),
                            variable=str(spec.get("variable") or label))
            channels[label] = ch
            order.append(label)
            entry = {k: v for k, v in spec.items() if k not in _CHANNEL_FIELDS and k != "metric"}
            entry["register"] = int(addr)
            entry["_no_label"] = True
            ch.modbus[metric] = entry
            continue

        ch = channels.get(label)
        if ch is None:
            ch = ChannelRow(bnc_port=len(order) + 1, point_label=str(label),
                            variable=str(spec.get("variable") or ""))
            channels[label] = ch
            order.append(label)
        # Campos de canal: el registro 'sensor' (Direct) es la autoridad.
        if spec.get("kind") == "sensor" or "sensor_type" in spec:
            st = _MAP_TO_SETUP_TYPE.get(str(spec.get("sensor_type") or ""), "")
            if st:
                ch.sensor_type = st
            if spec.get("plane") is not None:
                try:
                    ch.plane = int(spec["plane"])
                except (TypeError, ValueError):
                    pass
            ch.direction = str(spec.get("direction") or ch.direction)
            ch.location = str(spec.get("location") or ch.location)
            if metric == "Direct" and spec.get("unit"):
                ch.unit_native = str(spec["unit"])
        if not ch.variable and spec.get("variable"):
            ch.variable = str(spec["variable"])

        entry = {k: v for k, v in spec.items()
                 if k not in _CHANNEL_FIELDS and k != "metric"}
        # Variable distinta a la del canal → override por registro.
        if spec.get("variable") is not None and str(spec["variable"]) != ch.variable:
            entry["variable"] = spec["variable"]
        # Campos de canal presentes en el registro (para reconstruir igual).
        entry["_fields"] = [k for k in _CHANNEL_FIELDS if k in spec and k != "variable"]
        if "variable" not in spec:
            entry["_no_variable"] = True
        entry["register"] = int(addr)
        if metric in ch.modbus:
            # Mismo sensor+métrica en 2 registros: clave única para no perderlo.
            metric = f"{metric}@{addr}"
        ch.modbus[metric] = entry

    gateway = {k: v for k, v in raw.items() if k not in _TOP_RESERVED}
    instance_id = str(raw.get("instance_id") or "")
    machine = MachineConfig(name=name or instance_id or "Machine")
    return AcqSetup(machine=machine, channels=[channels[k] for k in order],
                    instance_id=instance_id, mode=mode if mode in ("spot", "online") else "online",
                    source="modbus", modbus_gateway=gateway)


def modbus_map_from_setup(setup: AcqSetup) -> Dict[str, Any]:
    """AcqSetup → mapa Modbus que consume el collector (mismo formato que
    data/modbus_maps/*.json). Los campos de canal (label, plano, dirección,
    tipo, ubicación, variable) salen del CANAL: editar el sensor en Machines
    cambia lo que postea el collector."""
    out: Dict[str, Any] = {"instance_id": setup.instance_id}
    gw = dict(setup.modbus_gateway or {})
    out.update(gw)
    registers: List[Tuple[int, Dict[str, Any]]] = []
    for ch in setup.channels:
        for metric_key, entry in (ch.modbus or {}).items():
            e = dict(entry)
            addr = int(e.pop("register"))
            metric = metric_key.split("@", 1)[0]
            no_label = e.pop("_no_label", False)
            fields_present = e.pop("_fields", None)
            no_variable = e.pop("_no_variable", False)
            var_override = e.pop("variable", None)
            spec: Dict[str, Any] = {}
            if no_label:
                kind = e.pop("kind", "speed")
                spec["kind"] = kind
                spec["variable"] = var_override if var_override is not None else (ch.variable or ch.point_label)
            else:
                if "kind" in e:
                    spec["kind"] = e.pop("kind")
                if fields_present is None:
                    fields_present = ["sensor_label"]
                    if spec.get("kind") == "sensor":
                        fields_present += ["plane", "direction", "sensor_type", "location"]
                vals = {
                    "sensor_label": ch.point_label,
                    "plane": ch.plane,
                    "direction": ch.direction,
                    "sensor_type": _SETUP_TO_MAP_TYPE.get(ch.sensor_type, ch.sensor_type),
                    "location": ch.location,
                }
                for k in ("sensor_label", "plane", "direction", "sensor_type", "location"):
                    if k in fields_present:
                        spec[k] = vals[k]
                if not no_variable:
                    spec["variable"] = var_override if var_override is not None else (ch.variable or ch.point_label)
            spec["metric"] = metric
            spec.update(e)
            registers.append((addr, spec))
    registers.sort(key=lambda t: t[0])
    out["registers"] = {str(a): s for a, s in registers}
    return out


def modbus_channels(setup: AcqSetup) -> List[ChannelRow]:
    """Canales con origen Modbus configurado."""
    return [c for c in setup.channels if c.modbus]


def diff_modbus_maps(old: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, List[str]]:
    """Diferencias legibles entre 2 mapas (para 'Publicar vN+1' y logs del
    collector): registros agregados / quitados / cambiados + gateway."""
    old = old or {}
    o_regs = old.get("registers") or {}
    n_regs = new.get("registers") or {}
    added = [f"{a} {n_regs[a].get('sensor_label') or n_regs[a].get('variable')} {n_regs[a].get('metric')}"
             for a in n_regs if a not in o_regs]
    removed = [f"{a} {o_regs[a].get('sensor_label') or o_regs[a].get('variable')} {o_regs[a].get('metric')}"
               for a in o_regs if a not in n_regs]
    changed = []
    for a in n_regs:
        if a in o_regs and o_regs[a] != n_regs[a]:
            keys = sorted(k for k in set(o_regs[a]) | set(n_regs[a])
                          if o_regs[a].get(k) != n_regs[a].get(k))
            changed.append(f"{a}: " + ", ".join(
                f"{k} {o_regs[a].get(k)!r}→{n_regs[a].get(k)!r}" for k in keys))
    gw_old = {k: v for k, v in old.items() if k not in _TOP_RESERVED}
    gw_new = {k: v for k, v in new.items() if k not in _TOP_RESERVED}
    gateway = [f"{k}: {gw_old.get(k)!r}→{gw_new.get(k)!r}"
               for k in sorted(set(gw_old) | set(gw_new)) if gw_old.get(k) != gw_new.get(k)]
    return {"added": added, "removed": removed, "changed": changed, "gateway": gateway}
