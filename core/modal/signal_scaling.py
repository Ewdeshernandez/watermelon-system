"""
core/modal/signal_scaling.py — Escalado de señales crudas a unidades de ingeniería
==================================================================================

Convierte voltajes crudos del NI-9234 (Volts) a unidades de ingeniería
correspondientes al sensor (g, mil, in/s, N, etc.) aplicando la
sensibilidad nominal del transductor.

Sensores típicos en SIGA
------------------------
· Wilcoxon acelerómetro: 100 mV/g (IEPE)
  V_raw → g: g = V / 0.100  =  V × 10

· Bently proximity probe 3300/3500: 200 mV/mil
  V_raw → mil: mil = V / 0.200  =  V × 5
  (probetas instaladas con gap de 50 mil — bias DC de ~10 V)

· PCB martillo modal (e.g. 086C03): 2.4 mV/N
  V_raw → N: N = V / 0.0024  =  V × 416.67

Notas
-----
· Para proximidad Bently: la señal viene con bias DC (~10 V por gap nominal
  de 50 mil). El bias se debe filtrar (DC blocker o substracción de la media)
  antes de aplicar la sensitivity para obtener desplazamiento dinámico.

· Para acelerómetros IEPE: el NI-9234 suministra la corriente IEPE (2 mA) y
  el bias de ~12 V se filtra internamente. La señal es directa.

Norma aplicable
---------------
ISO 7626-1 §4 — Definición de movilidad mecánica. La señal debe estar en
unidades de ingeniería antes de cualquier cálculo de FRF o parámetro modal.
"""

from __future__ import annotations

import numpy as np


def voltage_to_eu(
    voltage_signal: np.ndarray,
    sensitivity_mv_per_eu: float,
    remove_dc_bias: bool = False,
) -> np.ndarray:
    """
    Convierte un vector de voltajes (V) a unidades de ingeniería.

    Args:
        voltage_signal: Array de voltajes crudos del DAQ
        sensitivity_mv_per_eu: Sensibilidad del transductor en mV/EU
          (e.g. 100.0 para Wilcoxon 100 mV/g, 200.0 para Bently 200 mV/mil)
        remove_dc_bias: Si True, resta la media (uso típico en proximidad
          para quitar el offset del gap)

    Returns:
        Array en unidades de ingeniería (g, mil, N, etc.)
    """
    if sensitivity_mv_per_eu <= 0:
        raise ValueError(f"Sensitivity debe ser positiva, se recibió {sensitivity_mv_per_eu}")

    # mV/EU → V/EU: dividir por 1000
    v_per_eu = sensitivity_mv_per_eu / 1000.0

    signal = np.asarray(voltage_signal, dtype=float)
    if remove_dc_bias:
        signal = signal - np.mean(signal)

    return signal / v_per_eu


def sensor_eu_from_kind(sensor_kind: str) -> str:
    """
    Devuelve la unidad de ingeniería estándar para un tipo de sensor.

    Used para autocompletar la unidad en UI / reports.
    """
    mapping = {
        "acceleration": "g",
        "velocity": "in/s",
        "displacement": "mil",
        "force": "N",
    }
    return mapping.get(sensor_kind, "")


def default_sensitivity(sensor_kind: str) -> float:
    """
    Devuelve la sensitivity nominal típica para sensores estándar SIGA.

    Útil como default en UI cuando no se conoce el sensor exacto.
    """
    defaults = {
        "acceleration": 100.0,  # Wilcoxon
        "displacement": 200.0,  # Bently
        "force": 2.4,           # PCB martillo
    }
    return defaults.get(sensor_kind, 100.0)


def default_coupling(sensor_kind: str) -> str:
    """
    Devuelve el coupling típico según tipo de sensor.
    """
    if sensor_kind == "acceleration":
        return "IEPE"
    if sensor_kind == "displacement":
        return "AC"  # Bently con DC blocker
    if sensor_kind == "force":
        return "IEPE"  # martillo PCB
    return "AC"
