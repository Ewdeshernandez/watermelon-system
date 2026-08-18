"""
core/remote_monitoring/config.py — Config amigable de máquina + canales
=======================================================================

Modelo de configuración estilo ADRE 408 / System1 pero simple:

  MachineConfig  — propiedades del tren (rpm, sentido de giro, speed control,
                   cojinete). Equivale a "Machine Properties" de System1
                   (Tarea 2 del taller T00336).
  ChannelRow     — una fila del grid de canales: BNC físico → punto de
                   medición (tipo, sensib, unidad, coupling, ángulo, lado,
                   Alert/Danger). Equivale al "Mapping" de System1 (Tarea 3)
                   fundido con el punto de medición.
  AcqSetup       — máquina + canales. La unidad que se guarda y desde la que
                   arranca el stream.

Fuente única de verdad: `setup_to_sensor_map()` produce dicts compatibles
con core.sensor_map.new_sensor(), y `setup_to_channel_configs()` produce
los ChannelConfig de adquisición. Así esta UI amigable NO crea un modelo
paralelo — escribe al mismo modelo que consume el resto del sistema.

Marco: API 670 (protección/transductores), API 684 (rotordinámica),
ISO 20816 (evaluación). Las validaciones siguen esos criterios.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =====================================================================
# Catálogos (rangos físicos típicos para validación API 670)
# =====================================================================
SENSOR_TYPES = ["proximity", "velometer", "accelerometer", "keyphasor"]
COUPLINGS = ["AC", "DC", "IEPE"]
ROTATIONS = ["CW", "CCW"]
SPEED_CONTROLS = ["constant", "variable"]
BEARING_TYPES = ["plain", "tilting_pad", "rolling", "mixed"]

# Unidad nativa por defecto según tipo de sensor
_DEFAULT_UNIT = {
    "proximity": "mil pp",
    "velometer": "mm/s rms",
    "accelerometer": "g rms",
    "keyphasor": "pulses/rev",
}
# Coupling por defecto según tipo
_DEFAULT_COUPLING = {
    "proximity": "AC",
    "velometer": "AC",
    "accelerometer": "IEPE",
    "keyphasor": "DC",
}
# Sensibilidad por defecto (mV/EU)
_DEFAULT_SENS = {
    "proximity": 200.0,     # mV/mil (Bently 3300/3500)
    "velometer": 100.0,     # mV/(mm/s) — depende, valor razonable
    "accelerometer": 100.0, # mV/g
    "keyphasor": 1.0,
}
# Rango físico razonable de sensibilidad por tipo (para warning)
_SENS_RANGE = {
    "proximity": (100.0, 300.0),
    "velometer": (50.0, 600.0),
    "accelerometer": (10.0, 1000.0),
    "keyphasor": (0.0, 1e9),
}

# Tokens de keyphasor (mismo criterio que stream_source)
_KPH_TOKENS = ("kph", "keyphasor", "keyph", "tach", "tacho", "trigger")


# =====================================================================
# Dataclasses
# =====================================================================
@dataclass
class MachineConfig:
    name: str = "Máquina ad-hoc"
    template_id: str = ""
    rpm_nominal: float = 3600.0
    rpm_min: float = 0.0
    rpm_max: float = 0.0
    rotation: str = "CCW"            # CW | CCW
    speed_control: str = "constant"  # constant | variable
    bearing_type: str = "plain"      # plain | tilting_pad | rolling | mixed
    n_bearings: int = 2
    iso_norm: str = ""

    def __post_init__(self) -> None:
        if self.rotation not in ROTATIONS:
            self.rotation = "CCW"
        if self.speed_control not in SPEED_CONTROLS:
            self.speed_control = "constant"


@dataclass
class ChannelRow:
    bnc_port: int
    point_label: str
    plane: int = 1
    sensor_type: str = "proximity"
    sensitivity_mv_per_eu: float = 200.0
    unit_native: str = "mil pp"
    coupling: str = "AC"
    angle_deg: float = 0.0
    side: str = ""
    alarm: float = 0.0
    danger: float = 0.0

    def is_keyphasor(self) -> bool:
        if self.sensor_type == "keyphasor":
            return True
        nm = (self.point_label or "").lower()
        return any(tok in nm for tok in _KPH_TOKENS)


@dataclass
class AcqSetup:
    machine: MachineConfig = field(default_factory=MachineConfig)
    channels: List[ChannelRow] = field(default_factory=list)

    def keyphasor_row(self) -> Optional[ChannelRow]:
        for ch in self.channels:
            if ch.is_keyphasor():
                return ch
        return None


# =====================================================================
# Auto-layout — genera el mapa estándar desde la plantilla
# =====================================================================
def auto_layout(machine: MachineConfig, with_keyphasor: bool = True) -> List[ChannelRow]:
    """Genera pares de proximidad X/Y por cojinete + keyphasor, estilo
    layout estándar Bently. Convención: Y=0° (TDC), X=90° (lado R).

    El usuario luego ajusta en el grid. Esto mata la densidad del wizard:
    un click y tenés el layout base.
    """
    rows: List[ChannelRow] = []
    bnc = 1
    n = max(1, int(machine.n_bearings))
    for brg in range(1, n + 1):
        rows.append(ChannelRow(bnc_port=bnc, point_label=f"{brg}Y", plane=brg,
                               sensor_type="proximity", sensitivity_mv_per_eu=200.0,
                               unit_native="mil pp", coupling="AC",
                               angle_deg=0.0, side="", alarm=2.5, danger=4.0))
        bnc += 1
        rows.append(ChannelRow(bnc_port=bnc, point_label=f"{brg}X", plane=brg,
                               sensor_type="proximity", sensitivity_mv_per_eu=200.0,
                               unit_native="mil pp", coupling="AC",
                               angle_deg=90.0, side="R", alarm=2.5, danger=4.0))
        bnc += 1
    if with_keyphasor:
        rows.append(ChannelRow(bnc_port=bnc, point_label="KPH", plane=0,
                               sensor_type="keyphasor", sensitivity_mv_per_eu=1.0,
                               unit_native="pulses/rev", coupling="DC",
                               angle_deg=0.0, side="", alarm=0.0, danger=0.0))
    return rows


def defaults_for_type(sensor_type: str) -> Dict[str, Any]:
    """Devuelve unidad/coupling/sensib por defecto para un tipo de sensor.
    Útil para que el grid auto-complete al cambiar el tipo."""
    return {
        "unit_native": _DEFAULT_UNIT.get(sensor_type, ""),
        "coupling": _DEFAULT_COUPLING.get(sensor_type, "AC"),
        "sensitivity_mv_per_eu": _DEFAULT_SENS.get(sensor_type, 100.0),
    }


# =====================================================================
# Validación API 670 / ISO 20816
# =====================================================================
@dataclass
class Finding:
    level: str   # "error" | "warn" | "ok"
    code: str
    message: str


def validate_setup(setup: AcqSetup) -> List[Finding]:
    """Valida el setup contra criterios API 670 / ISO. Devuelve findings
    ordenados error → warn → ok."""
    out: List[Finding] = []
    chans = setup.channels

    if not chans:
        return [Finding("error", "no_channels", "No hay canales configurados.")]

    # BNC único
    seen: Dict[int, str] = {}
    for ch in chans:
        if ch.bnc_port in seen:
            out.append(Finding("error", "dup_bnc",
                               f"BNC {ch.bnc_port} duplicado ({seen[ch.bnc_port]} y {ch.point_label})."))
        else:
            seen[ch.bnc_port] = ch.point_label

    # Keyphasor
    kph = [c for c in chans if c.is_keyphasor()]
    if not kph:
        out.append(Finding("warn", "no_keyphasor",
                           "Sin keyphasor: no habrá gráficos síncronos (bode/polar/órbita/1X)."))
    elif len(kph) > 1:
        out.append(Finding("warn", "multi_keyphasor",
                           f"{len(kph)} canales keyphasor — normalmente basta 1 por tren."))

    # Pares X/Y ortogonales por cojinete (API 670: sondas radiales a 90°)
    by_plane: Dict[int, List[ChannelRow]] = {}
    for c in chans:
        if c.is_keyphasor():
            continue
        by_plane.setdefault(c.plane, []).append(c)
    for plane, group in by_plane.items():
        radials = [c for c in group if c.sensor_type in ("proximity", "velometer", "accelerometer")]
        if len(radials) >= 2:
            a0, a1 = radials[0].angle_deg, radials[1].angle_deg
            sep = abs((a0 - a1) % 180.0)
            sep = min(sep, 180.0 - sep)  # separación no orientada
            if abs(sep - 90.0) > 5.0:
                out.append(Finding("warn", "xy_not_orthogonal",
                                   f"Cojinete {plane}: sondas a {a0:.0f}°/{a1:.0f}° "
                                   f"(separación {sep:.0f}°, se esperan 90°±5° para órbita)."))

    # Alert < Danger; sensibilidad en rango físico; unidades por tipo
    for ch in chans:
        if ch.is_keyphasor():
            continue
        if ch.danger > 0 and ch.alarm > 0 and ch.alarm >= ch.danger:
            out.append(Finding("error", "alert_ge_danger",
                               f"{ch.point_label}: Alert ({ch.alarm}) ≥ Danger ({ch.danger})."))
        lo, hi = _SENS_RANGE.get(ch.sensor_type, (0.0, 1e9))
        if not (lo <= ch.sensitivity_mv_per_eu <= hi):
            out.append(Finding("warn", "sens_out_of_range",
                               f"{ch.point_label}: sensib {ch.sensitivity_mv_per_eu} mV/EU fuera "
                               f"del rango típico [{lo:.0f}–{hi:.0f}] para {ch.sensor_type}."))

    errors = [f for f in out if f.level == "error"]
    warns = [f for f in out if f.level == "warn"]
    if not errors and not warns:
        out.append(Finding("ok", "all_good", "Configuración válida — sin observaciones."))
    return sorted(out, key=lambda f: {"error": 0, "warn": 1, "ok": 2}[f.level])


def is_setup_valid(setup: AcqSetup) -> bool:
    """True si no hay findings de nivel error."""
    return not any(f.level == "error" for f in validate_setup(setup))


# =====================================================================
# Puentes a los modelos existentes (fuente única de verdad)
# =====================================================================
def setup_to_channel_configs(setup: AcqSetup) -> List:
    """AcqSetup → List[ChannelConfig] de core.modal.acq_backend (adquisición)."""
    from core.modal.acq_backend import ChannelConfig
    out = []
    for ch in setup.channels:
        coup = ch.coupling if ch.coupling in COUPLINGS else "AC"
        out.append(ChannelConfig(
            name=ch.point_label,
            coupling=coup,
            sensitivity_mv_per_eu=float(ch.sensitivity_mv_per_eu or 1.0),
            bnc_port=int(ch.bnc_port),
            units=ch.unit_native or "",
        ))
    return out


def setup_to_sensor_map(setup: AcqSetup) -> List[Dict[str, Any]]:
    """AcqSetup → lista de dicts compatibles con core.sensor_map.new_sensor().
    Este es el puente para persistir al modelo del activo (fuente única)."""
    from core.sensor_map import new_sensor
    sensors = []
    for ch in setup.channels:
        # Inferir direction (Y/X/RAD/AX/keyphasor) del label del punto
        lbl = (ch.point_label or "").upper()
        if ch.is_keyphasor():
            direction = "keyphasor"
        elif lbl.endswith("Y") or "Y" in lbl[-2:]:
            direction = "Y"
        elif lbl.endswith("X") or "X" in lbl[-2:]:
            direction = "X"
        elif lbl.endswith("A") or "AX" in lbl:
            direction = "AX"
        else:
            direction = "RAD"
        coup = ch.coupling if ch.coupling in COUPLINGS else ""
        sensors.append(new_sensor(
            plane=int(ch.plane),
            plane_label=ch.point_label,
            side=ch.side or "L",
            angle_deg=float(ch.angle_deg),
            direction=direction,
            sensor_type=ch.sensor_type,
            unit_native=ch.unit_native,
            coupling=coup,
            sensitivity_mv_per_eu=float(ch.sensitivity_mv_per_eu or 0.0),
            alarm=float(ch.alarm or 0.0),
            danger=float(ch.danger or 0.0),
        ))
    return sensors


# =====================================================================
# Persistencia local (JSON, durable-dir aware) — Fase 1
# =====================================================================
def _setups_dir() -> Path:
    pd = os.environ.get("WM_PERSIST_DIR")
    if pd:
        return Path(pd) / "remote_monitoring" / "setups"
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "data" / "remote_monitoring" / "setups"
    return Path("data/remote_monitoring/setups")


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (name or "setup")).strip("_") or "setup"


def save_setup(setup: AcqSetup) -> Path:
    d = _setups_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{_slug(setup.machine.name)}.json"
    payload = {"machine": asdict(setup.machine),
               "channels": [asdict(c) for c in setup.channels]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_setup(name: str) -> Optional[AcqSetup]:
    path = _setups_dir() / f"{_slug(name)}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    machine = MachineConfig(**data.get("machine", {}))
    channels = [ChannelRow(**c) for c in data.get("channels", [])]
    return AcqSetup(machine=machine, channels=channels)


def list_setups() -> List[str]:
    d = _setups_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))
