"""
core/remote_monitoring/sim_machine.py — Máquina simulada configurable
=====================================================================

Modelo editable de una "máquina real" para el simulador del módulo nativo:
la defines (sensores, geometría), le inyectas FENÓMENOS (fallas) y la corres en
un MODO de operación (estable / arranque / parada / arranque-parada). Se
guarda/carga como JSON en una biblioteca local.

    m = SimMachine.plantilla_motor_bomba()
    m.phenomena = {"accel": "bearing_bpfo", "prox": "oil_whirl"}
    m.mode = "arranque"
    save_to_library(m)
    cfg = m.to_stream_config()      # -> StreamConfig para SimulatedStreamSource
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from core.modal.acq_backend import ChannelConfig
from core.remote_monitoring.stream_source import StreamConfig


# Modo de operación → perfil de velocidad del stream
MODE_TO_PROFILE = {
    "estable": "constant",
    "arranque": "runup",
    "parada": "coastdown",
    "arranque_parada": "runup_coastdown",
}
MODES = list(MODE_TO_PROFILE.keys())

# Fenómenos disponibles por tipo de sensor (para los combos del editor)
PHENOMENA = {
    "prox": ["none", "unbalance", "misalignment", "looseness", "rub", "oil_whirl"],
    "vel": ["none", "unbalance", "misalignment", "looseness"],
    "accel": ["none", "bearing_bpfo", "bearing_bpfi", "bearing_bsf", "gear_mesh"],
}

# Unidades y acoplamiento por tipo de sensor
_UNITS = {"prox": "mil pp", "vel": "mm/s rms", "accel": "g rms", "keyphasor": "pulses/rev"}
_COUP = {"prox": "DC", "vel": "AC", "accel": "IEPE", "keyphasor": "DC"}


@dataclass
class SensorSpec:
    name: str
    kind: str            # prox | vel | accel | keyphasor
    bnc: int
    sensitivity: float = 100.0
    angle: float = 0.0   # ángulo de sonda desde TDC (grados)
    side: str = ""       # "R" (horario) | "L" (antihorario) | "" — convención API 670
    gap: float = 0.0     # voltaje gap/bias (V) — sondas de proximidad DC
    alarm: float = 0.0   # nivel de alarma (unid. del canal, ISO 20816)
    danger: float = 0.0  # nivel de peligro

    def units(self) -> str:
        return _UNITS.get(self.kind, "EU")

    def coupling(self) -> str:
        return _COUP.get(self.kind, "DC")

    def abs_angle(self) -> float:
        """Ángulo ABSOLUTO desde TDC: R = horario (=ángulo), L = antihorario
        (=-ángulo). 45°R→45°, 45°L→315° (convención API 670)."""
        a = float(self.angle) % 360.0
        return (-a) % 360.0 if (self.side or "").upper() == "L" else a

    def to_channel(self) -> ChannelConfig:
        return ChannelConfig(name=self.name, coupling=self.coupling(),
                             sensitivity_mv_per_eu=self.sensitivity, bnc_port=self.bnc,
                             units=self.units())


@dataclass
class SimMachine:
    name: str = "Maquina_1"
    fs: float = 5120.0
    sensors: List[SensorSpec] = field(default_factory=list)
    rotation: str = "CCW"            # CW | CCW (sentido de giro, para la órbita)
    bearing_type: str = "plain"      # plain | tilting_pad | rolling | mixed
    # operación
    mode: str = "estable"            # estable | arranque | parada | arranque_parada
    rpm: float = 3000.0
    rpm_start: float = 300.0
    rpm_end: float = 6000.0
    ramp_s: float = 90.0
    crit1: float = 0.0
    crit2: float = 0.0
    zeta: float = 0.06
    # fenómenos por tipo de sensor + severidad global
    phenomena: Dict[str, str] = field(default_factory=dict)
    severity: float = 1.0
    # geometría (frecuencias de falla / engrane)
    n_balls: int = 8
    bd_pd: float = 0.34
    gear_teeth: int = 22

    # --- canales / stream ---------------------------------------------------
    def channels(self) -> List[ChannelConfig]:
        return [s.to_channel() for s in self.sensors]

    def to_stream_config(self, fs: Optional[float] = None) -> StreamConfig:
        fs = float(fs or self.fs)
        dbk = {k: v for k, v in (self.phenomena or {}).items() if v and v != "none"}
        prof = MODE_TO_PROFILE.get(self.mode, "constant")
        return StreamConfig(
            sample_rate_hz=fs, channels=self.channels(),
            block_seconds=0.1, buffer_seconds=12.0,
            rpm=self.rpm, speed_profile=prof,
            rpm_start=self.rpm_start, rpm_end=self.rpm_end, ramp_seconds=self.ramp_s,
            sim_critical_rpm=self.crit1, sim_critical_rpm2=self.crit2, sim_zeta=self.zeta,
            defect_by_kind=dbk, sim_severity=self.severity,
            sim_n_balls=self.n_balls, sim_bd_pd=self.bd_pd, sim_gear_teeth=self.gear_teeth)

    # --- persistencia -------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SimMachine":
        # Tolerante a drift de esquema: ignora claves desconocidas y usa defaults
        # para las que falten (una máquina guardada con otra versión igual carga).
        from dataclasses import fields as _fields
        d = dict(d or {})
        _ss_ok = {f.name for f in _fields(SensorSpec)}
        d["sensors"] = [SensorSpec(**{k: v for k, v in (s or {}).items() if k in _ss_ok})
                        for s in d.get("sensors", [])]
        _sm_ok = {f.name for f in _fields(SimMachine)}
        return SimMachine(**{k: v for k, v in d.items() if k in _sm_ok})

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "SimMachine":
        with open(path, "r", encoding="utf-8") as f:
            return SimMachine.from_dict(json.load(f))

    # --- plantillas ---------------------------------------------------------
    @staticmethod
    def plantilla_prox_train(n_bearings: int = 4, name: str = "Tren_proximidad") -> "SimMachine":
        s = [SensorSpec("KPH", "keyphasor", 1, 1.0)]
        b = 2
        for brg in range(1, n_bearings + 1):
            for ax in ("Y", "X"):
                side = "L" if ax == "Y" else "R"       # Y=45°L, X=45°R (90° entre sí)
                s.append(SensorSpec(f"{brg}{ax}", "prox", b, 200.0, angle=45.0, side=side,
                                    gap=-9.5, alarm=2.5, danger=4.0)); b += 1
        return SimMachine(name=name, sensors=s, rpm=3000.0, crit1=2000.0)

    @staticmethod
    def plantilla_motor_bomba(name: str = "Motor_Bomba") -> "SimMachine":
        """Motor eléctrico (rodamientos, 4 acelerómetros) + bomba (cojinetes
        planos, 4 proximidades), keyphasor común."""
        s = [SensorSpec("KPH", "keyphasor", 1, 1.0)]
        b = 2
        for brg in (1, 2):                      # motor: rodamientos
            for ax in ("H", "V"):
                s.append(SensorSpec(f"{brg}{ax}", "accel", b, 100.0, alarm=4.5, danger=7.1)); b += 1
        for brg in (3, 4):                      # bomba: cojinetes planos
            for ax in ("Y", "X"):
                side = "L" if ax == "Y" else "R"
                s.append(SensorSpec(f"{brg}{ax}", "prox", b, 200.0, angle=45.0, side=side,
                                    gap=-9.5, alarm=2.5, danger=4.0)); b += 1
        return SimMachine(name=name, fs=25600.0, sensors=s, rpm=3000.0, crit1=1500.0,
                          rotation="CCW", bearing_type="mixed",
                          rpm_start=300.0, rpm_end=6000.0, ramp_s=90.0,
                          phenomena={"accel": "bearing_bpfo", "prox": "oil_whirl"})


# --- biblioteca local -------------------------------------------------------
def library_dir() -> str:
    d = os.environ.get("WM_MACHINES_DIR") or os.path.join(
        os.path.expanduser("~"), ".watermelon", "machines")
    os.makedirs(d, exist_ok=True)
    return d


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (name or "maquina")).strip("_")


def save_to_library(m: SimMachine) -> str:
    return m.save(os.path.join(library_dir(), f"{_slug(m.name)}.json"))


def list_machines() -> List[str]:
    d = library_dir()
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))


def load_from_library(name_or_slug: str) -> SimMachine:
    return SimMachine.load(os.path.join(library_dir(), f"{_slug(name_or_slug)}.json"))
