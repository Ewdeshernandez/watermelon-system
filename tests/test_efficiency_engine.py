"""Tests del motor de eficiencia de máquinas rotatorias."""
import math

from core.efficiency.engine import (
    omega_rad_s, mechanical_power_kw, torque_from_power_nm, electrical_power_kw,
    motor_efficiency_pct, operational_efficiency_pct, diagnose_operational,
    hydraulic_power_kw, air_power_kw, pump_efficiency_pct, fan_efficiency_pct,
    hydro_turbine_efficiency_pct, compressor_isentropic_efficiency_pct,
    EfficiencyInputs, compute, MACHINE_TYPES,
)


def test_mechanical_power():
    assert abs(omega_rad_s(1785) - 1785 * math.pi / 30) < 1e-9
    p = mechanical_power_kw(11626.0, 1785.0)
    assert abs(p - 11626.0 * (1785 * math.pi / 30) / 1000.0) < 1e-6
    assert 2150 < p < 2200                       # ~2173 kW


def test_torque_roundtrip():
    p = mechanical_power_kw(5000.0, 1500.0)
    assert abs(torque_from_power_nm(p, 1500.0) - 5000.0) < 1e-6


def test_electrical_power_3ph():
    p = electrical_power_kw(4160, 438, 0.92, phases=3)
    assert abs(p - math.sqrt(3) * 4160 * 438 * 0.92 / 1000.0) < 1e-6
    assert 2890 < p < 2910                        # ~2903 kW


def test_electrical_power_1ph():
    assert abs(electrical_power_kw(230, 10, 0.9, phases=1) - 230 * 10 * 0.9 / 1000.0) < 1e-9


def test_motor_and_operational_efficiency():
    assert abs(motor_efficiency_pct(2173, 2650) - 2173 / 2650 * 100) < 1e-6
    assert abs(operational_efficiency_pct(2173, 2140) - 2173 / 2140 * 100) < 1e-6
    assert motor_efficiency_pct(100, 0) == 0.0    # sin división por cero


def test_diagnose_bands():
    assert diagnose_operational(110).code == "overload"
    assert diagnose_operational(101.5).code == "normal"
    assert diagnose_operational(95).code == "normal"
    assert diagnose_operational(90).code == "degradation"
    assert diagnose_operational(80).code == "failure"
    assert diagnose_operational(101.5).color == "green"


def test_pump_hydraulic_and_efficiency():
    ph = hydraulic_power_kw(0.1, 50.0, rho=1000.0)
    assert abs(ph - 1000 * 9.80665 * 0.1 * 50 / 1000.0) < 1e-6   # ~49.03 kW
    assert abs(pump_efficiency_pct(49.0, 70.0) - 70.0) < 1e-6


def test_fan_air_power():
    pa = air_power_kw(185.0, 4850.0)
    assert abs(pa - 185 * 4850 / 1000.0) < 1e-6                   # 897.25 kW
    assert abs(fan_efficiency_pct(897.25, 2173.0) - 897.25 / 2173 * 100) < 1e-6


def test_hydro_turbine_efficiency():
    e = hydro_turbine_efficiency_pct(4000.0, 10.0, 45.0, rho=1000.0)
    ph = hydraulic_power_kw(10.0, 45.0)
    assert abs(e - 4000.0 / ph * 100.0) < 1e-6


def test_compressor_isentropic():
    # aire: m=5 kg/s, Cp=1.005, T1=293.15K, π=2.0, k=1.4, P_mec=600 kW
    e = compressor_isentropic_efficiency_pct(5.0, 1.005, 293.15, 2.0, 1.4, 600.0)
    w_isen = 5.0 * 1.005 * 293.15 * (2.0 ** ((1.4 - 1) / 1.4) - 1)
    assert abs(e - w_isen / 600.0 * 100.0) < 1e-6
    assert 0 < e < 100


def test_compute_fan_integral():
    inp = EfficiencyInputs(machine_type="fan", torque_nm=11626, rpm=1785,
                           voltage_v=4160, current_a=438, power_factor=0.92,
                           design_power_kw=2140, flow_m3s=185, dp_pa=4850)
    r = compute(inp)
    assert 2150 < r.p_mec_kw < 2200
    assert 2890 < r.p_elec_kw < 2910
    assert 70 < r.eta_motor_pct < 80                           # ~74.9%
    assert r.diagnosis.code == "normal"                        # 101.5%
    assert r.process_label == "fan" and r.eta_process_pct > 0


def test_machine_types_present():
    codes = {c for c, *_ in MACHINE_TYPES}
    assert {"motor", "fan", "pump", "compressor", "hydro", "steam_gas", "generic"} <= codes
