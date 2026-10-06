"""
core/efficiency/engine.py — Motor de eficiencia de máquinas rotatorias
======================================================================

Mide la eficiencia operativa de máquinas rotatorias (ventiladores, bombas,
compresores, turbinas, motores, sopladores, …) combinando:

  · Potencia MECÁNICA en el eje:  P_mec = T·ω/1000  (T del TorqueTrak 10K, ω=rpm·π/30)
  · Potencia ELÉCTRICA del motor: P_elec = √3·V·I·cosφ/1000  (analizador de red)
  · η_motor = P_mec/P_elec ·100   · η_operativa = P_mec/P_diseño ·100
  · Eficiencia por TIPO de máquina (hidráulica/aire/isentrópica).

Normas: IEC 60034-2 (motor) · ISO 5801 (ventiladores) · ISO 9906 (bombas) ·
ASME PTC 10 (compresores) · IEC 60041 (turbinas hidráulicas) · ISO 20816 (vib).

Numpy/py puro (sin Qt/Streamlit/hardware): lo comparten campo y web.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

G = 9.80665                      # gravedad [m/s²]
RHO_WATER = 1000.0               # densidad agua [kg/m³]
RHO_AIR = 1.204                  # densidad aire a 20°C [kg/m³]

# Tipos de máquina soportados (código, etiqueta EN, etiqueta ES, norma).
MACHINE_TYPES = [
    ("motor",       "Motor (generic drive)",   "Motor (accionamiento)",     "IEC 60034-2"),
    ("fan",         "Fan / blower",            "Ventilador / soplador",     "ISO 5801"),
    ("pump",        "Centrifugal pump",        "Bomba centrífuga",          "ISO 9906"),
    ("compressor",  "Centrifugal compressor",  "Compresor centrífugo",      "ASME PTC 10"),
    ("hydro",       "Hydraulic turbine",       "Turbina hidráulica",        "IEC 60041"),
    ("steam_gas",   "Steam / gas turbine",     "Turbina de vapor / gas",    "ASME PTC 6 / PTC 22"),
    ("generic",     "Generic rotating machine","Máquina rotatoria genérica","ISO 20816"),
]


# =====================================================================
# Potencias base
# =====================================================================
def omega_rad_s(rpm: float) -> float:
    """Velocidad angular [rad/s] = rpm·π/30."""
    return float(rpm) * math.pi / 30.0


def mechanical_power_kw(torque_nm: float, rpm: float) -> float:
    """Potencia mecánica en el eje [kW] = T·ω/1000."""
    return float(torque_nm) * omega_rad_s(rpm) / 1000.0


def torque_from_power_nm(p_mec_kw: float, rpm: float) -> float:
    """Par [N·m] a partir de la potencia mecánica (inverso de mechanical_power_kw)."""
    w = omega_rad_s(rpm)
    return (float(p_mec_kw) * 1000.0 / w) if w > 0 else 0.0


def electrical_power_kw(voltage_v: float, current_a: float, power_factor: float,
                        phases: int = 3) -> float:
    """Potencia eléctrica activa [kW]. Trifásico: √3·V_LL·I·cosφ/1000; mono: V·I·cosφ/1000."""
    k = math.sqrt(3.0) if phases == 3 else 1.0
    return k * float(voltage_v) * float(current_a) * float(power_factor) / 1000.0


# =====================================================================
# Eficiencias
# =====================================================================
def motor_efficiency_pct(p_mec_kw: float, p_elec_kw: float) -> float:
    """η_motor [%] = P_mec/P_elec ·100 (IEC 60034-2)."""
    return (float(p_mec_kw) / float(p_elec_kw) * 100.0) if p_elec_kw > 1e-9 else 0.0


def operational_efficiency_pct(p_mec_kw: float, p_design_kw: float) -> float:
    """η_operativa [%] = P_mec_real/P_diseño ·100."""
    return (float(p_mec_kw) / float(p_design_kw) * 100.0) if p_design_kw > 1e-9 else 0.0


@dataclass
class Diagnosis:
    code: str            # "overload" | "normal" | "degradation" | "failure"
    label_en: str
    label_es: str
    color: str           # "green" | "amber" | "red"


def diagnose_operational(pct: float) -> Diagnosis:
    """Semáforo de la eficiencia operativa (bandas del módulo):
    >105 sobrecarga (amber) · 95–105 normal (green) · 85–95 degradación (amber) · <85 falla (red)."""
    p = float(pct)
    if p > 105.0:
        return Diagnosis("overload", "Overload", "Sobrecarga", "amber")
    if p >= 95.0:
        return Diagnosis("normal", "Normal", "Normal", "green")
    if p >= 85.0:
        return Diagnosis("degradation", "Degradation", "Degradación", "amber")
    return Diagnosis("failure", "Imminent failure", "Falla inminente", "red")


# =====================================================================
# Eficiencia por tipo de máquina
# =====================================================================
def hydraulic_power_kw(flow_m3s: float, head_m: float, rho: float = RHO_WATER) -> float:
    """Potencia hidráulica [kW] = ρ·g·Q·H/1000 (bombas/turbinas)."""
    return float(rho) * G * float(flow_m3s) * float(head_m) / 1000.0


def air_power_kw(flow_m3s: float, dp_pa: float) -> float:
    """Potencia de aire [kW] = Q·Δp/1000 (ventiladores/sopladores, ISO 5801)."""
    return float(flow_m3s) * float(dp_pa) / 1000.0


def pump_efficiency_pct(p_hydraulic_kw: float, p_mec_kw: float) -> float:
    """η_bomba [%] = P_hidráulica/P_mecánica ·100 (ISO 9906)."""
    return (float(p_hydraulic_kw) / float(p_mec_kw) * 100.0) if p_mec_kw > 1e-9 else 0.0


def fan_efficiency_pct(p_air_kw: float, p_mec_kw: float) -> float:
    """η_ventilador [%] = P_aire/P_mecánica ·100 (ISO 5801)."""
    return (float(p_air_kw) / float(p_mec_kw) * 100.0) if p_mec_kw > 1e-9 else 0.0


def hydro_turbine_efficiency_pct(p_elec_kw: float, flow_m3s: float, head_m: float,
                                 rho: float = RHO_WATER) -> float:
    """η_turbina [%] = P_eléctrica / P_hidráulica_disponible ·100 (IEC 60041)."""
    p_hyd = hydraulic_power_kw(flow_m3s, head_m, rho)
    return (float(p_elec_kw) / p_hyd * 100.0) if p_hyd > 1e-9 else 0.0


def compressor_isentropic_efficiency_pct(mdot_kg_s: float, cp_kj_kgk: float, t_in_k: float,
                                         pressure_ratio: float, k_ratio: float,
                                         p_mec_kw: float) -> float:
    """η_isentrópica [%] (compresor centrífugo, ASME PTC 10):
        W_isen = m·Cp·T1·(π^((k-1)/k) − 1)   [kW, con Cp en kJ/kg·K]
        η = W_isen / W_real(=P_mec) ·100."""
    try:
        w_isen = (float(mdot_kg_s) * float(cp_kj_kgk) * float(t_in_k)
                  * (float(pressure_ratio) ** ((float(k_ratio) - 1.0) / float(k_ratio)) - 1.0))
    except Exception:  # noqa: BLE001
        return 0.0
    return (w_isen / float(p_mec_kw) * 100.0) if p_mec_kw > 1e-9 else 0.0


# =====================================================================
# Cálculo integral
# =====================================================================
@dataclass
class EfficiencyInputs:
    machine_type: str = "motor"
    # Mecánico (del TorqueTrak / manual)
    torque_nm: float = 0.0
    rpm: float = 0.0
    # Eléctrico (analizador de red / manual)
    voltage_v: float = 0.0
    current_a: float = 0.0
    power_factor: float = 0.92
    phases: int = 3
    # Diseño / placa
    design_power_kw: float = 0.0
    # Proceso (según tipo)
    flow_m3s: float = 0.0           # caudal (bomba/vent/turbina)
    head_m: float = 0.0             # altura manométrica (bomba/turbina)
    dp_pa: float = 0.0              # presión total (ventilador)
    rho: float = RHO_WATER          # densidad del fluido
    # Compresor
    mdot_kg_s: float = 0.0
    cp_kj_kgk: float = 1.005
    t_in_k: float = 293.15
    pressure_ratio: float = 1.0
    k_ratio: float = 1.4


@dataclass
class EfficiencyResult:
    machine_type: str
    p_mec_kw: float
    p_elec_kw: float
    eta_motor_pct: float
    eta_operational_pct: float
    diagnosis: Diagnosis
    process_power_kw: float = 0.0   # hidráulica / aire / isentrópica según tipo
    eta_process_pct: float = 0.0    # η específica del tipo (bomba/vent/turbina/compresor)
    process_label: str = ""


def compute(inp: EfficiencyInputs) -> EfficiencyResult:
    """Cálculo integral de eficiencia para cualquier tipo de máquina."""
    p_mec = mechanical_power_kw(inp.torque_nm, inp.rpm)
    p_elec = electrical_power_kw(inp.voltage_v, inp.current_a, inp.power_factor, inp.phases)
    eta_m = motor_efficiency_pct(p_mec, p_elec)
    eta_op = operational_efficiency_pct(p_mec, inp.design_power_kw)
    diag = diagnose_operational(eta_op if inp.design_power_kw > 0 else 100.0)

    proc_kw, proc_eta, proc_lbl = 0.0, 0.0, ""
    mt = inp.machine_type
    if mt == "pump":
        proc_kw = hydraulic_power_kw(inp.flow_m3s, inp.head_m, inp.rho)
        proc_eta = pump_efficiency_pct(proc_kw, p_mec); proc_lbl = "pump"
    elif mt in ("fan",):
        proc_kw = air_power_kw(inp.flow_m3s, inp.dp_pa)
        proc_eta = fan_efficiency_pct(proc_kw, p_mec); proc_lbl = "fan"
    elif mt == "hydro":
        proc_kw = hydraulic_power_kw(inp.flow_m3s, inp.head_m, inp.rho)
        proc_eta = hydro_turbine_efficiency_pct(p_elec, inp.flow_m3s, inp.head_m, inp.rho); proc_lbl = "hydro"
    elif mt == "compressor":
        proc_eta = compressor_isentropic_efficiency_pct(inp.mdot_kg_s, inp.cp_kj_kgk, inp.t_in_k,
                                                        inp.pressure_ratio, inp.k_ratio, p_mec)
        proc_lbl = "compressor"
    return EfficiencyResult(machine_type=mt, p_mec_kw=p_mec, p_elec_kw=p_elec,
                            eta_motor_pct=eta_m, eta_operational_pct=eta_op, diagnosis=diag,
                            process_power_kw=proc_kw, eta_process_pct=proc_eta, process_label=proc_lbl)
