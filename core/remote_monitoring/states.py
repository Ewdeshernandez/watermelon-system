"""
core/remote_monitoring/states.py — Máquina de estados por velocidad
===================================================================

Clasifica el estado operativo del tren a partir de la velocidad (rpm),
igual que System1 (Startup/Operating/Slow Roll/Off, taller T00336 Tarea 4).
El estado decide CUÁNDO capturar en modo transitorio (bode/cascade se
llenan durante arranque/parada).

Estados:
  OFF        — máquina detenida (rpm ~ 0)
  SLOW_ROLL  — giro lento (turning gear)
  STARTUP    — acelerando (rpm subiendo)
  COASTDOWN  — desacelerando (rpm bajando)
  STEADY     — velocidad estable (estado estacionario)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

OFF = "OFF"
SLOW_ROLL = "SLOW_ROLL"
STARTUP = "STARTUP"
COASTDOWN = "COASTDOWN"
STEADY = "STEADY"

TRANSIENT_STATES = (STARTUP, COASTDOWN)

_LABELS = {
    OFF: ("Detenida", "#94a3b8"),
    SLOW_ROLL: ("Giro lento", "#38bdf8"),
    STARTUP: ("Arranque ▲", "#f59e0b"),
    COASTDOWN: ("Parada ▼", "#f59e0b"),
    STEADY: ("Estable", "#10b981"),
}


@dataclass
class StateConfig:
    off_rpm: float = 10.0          # por debajo → OFF
    slow_roll_rpm: float = 200.0   # por debajo → SLOW_ROLL
    delta_rpm: float = 15.0        # cambio para considerar transitorio


def classify_state(rpm: Optional[float], prev_rpm: Optional[float],
                   cfg: Optional[StateConfig] = None) -> str:
    cfg = cfg or StateConfig()
    if rpm is None or rpm < cfg.off_rpm:
        return OFF
    if rpm < cfg.slow_roll_rpm:
        return SLOW_ROLL
    if prev_rpm is not None:
        d = rpm - prev_rpm
        if d > cfg.delta_rpm:
            return STARTUP
        if d < -cfg.delta_rpm:
            return COASTDOWN
    return STEADY


def is_transient(state: str) -> bool:
    return state in TRANSIENT_STATES


def state_label(state: str) -> str:
    return _LABELS.get(state, (state, "#94a3b8"))[0]


def state_color(state: str) -> str:
    return _LABELS.get(state, (state, "#94a3b8"))[1]
