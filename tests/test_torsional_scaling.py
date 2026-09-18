"""
tests.test_torsional_scaling
============================

Valida el núcleo de escalado torsional (core.torsional) contra los ejemplos
numéricos publicados en el User's Guide del Binsfeld TorqueTrak 10K
(Appendix B) — la fuente de verdad del fabricante.

Casos de referencia del manual:
  · B1 Torque:   Do=3.000", GF=2.08,  GXMT=4000 → TFS = 4,901 ft-lb
  · B2 Axial:    Do=2.25",  GF=2.045, GXMT=4000 → PFS = 89,739 lb
  · B3 ¼ puente: GF=2.045,  GXMT=4000           → εFS = 1956 µε
  · Shunt Ref1:  εs=100, εFS=500, Z=0.9802       → Vs  = 1.9604 V
  · Gain table:  GF=2.0, GXMT=4000               → εFS = 500 µε (torque)
"""
from __future__ import annotations

import numpy as np
import pytest

from core.torsional.scaling import (
    ShaftGeometry,
    GageConfig,
    BridgeType,
    TorqueScaling,
    full_scale_torque,
    full_scale_force_axial,
    full_scale_strain_quarter,
    full_scale_strain_torque,
    voltage_to_torque,
)
from core.torsional.shunt_cal import (
    REF1_100UE,
    REF2_500UE,
    expected_shunt_voltage,
    simulated_strain_from_shunt,
    verify_shunt,
)


# -----------------------------------------------------------------
# Appendix B1 — Torque en eje sólido de acero
# -----------------------------------------------------------------
def test_full_scale_torque_matches_manual_example():
    """Do=3", GF=2.08, GXMT=4000 → 4,901 ft-lb (manual pág. 31)."""
    shaft = ShaftGeometry(outer_diameter_in=3.000)
    gage = GageConfig(gage_factor=2.08, transmitter_gain=4000)
    tfs = full_scale_torque(shaft, gage, units="ftlb")
    assert tfs == pytest.approx(4901.0, rel=1e-3)


def test_full_scale_torque_simplified_solid_steel_constant():
    """La ecuación general debe reducirse a 1510.38e3·Do³/(GF·GXMT) para acero sólido."""
    shaft = ShaftGeometry(outer_diameter_in=2.5)
    gage = GageConfig(gage_factor=2.0, transmitter_gain=4000)
    simplified = 1510.38e3 * shaft.outer_diameter_in**3 / (
        gage.gage_factor * gage.transmitter_gain)
    tfs = full_scale_torque(shaft, gage, units="ftlb")
    assert tfs == pytest.approx(simplified, rel=1e-3)


def test_full_scale_torque_nm_conversion():
    shaft = ShaftGeometry(outer_diameter_in=3.000)
    gage = GageConfig(gage_factor=2.08, transmitter_gain=4000)
    tfs_ftlb = full_scale_torque(shaft, gage, units="ftlb")
    tfs_nm = full_scale_torque(shaft, gage, units="nm")
    assert tfs_nm == pytest.approx(tfs_ftlb / 0.737562149, rel=1e-6)


def test_hollow_shaft_less_than_solid():
    """Un eje hueco tiene menor TFS que uno sólido del mismo Do."""
    gage = GageConfig(gage_factor=2.0, transmitter_gain=4000)
    solid = full_scale_torque(ShaftGeometry(3.0, 0.0), gage)
    hollow = full_scale_torque(ShaftGeometry(3.0, 1.5), gage)
    assert hollow < solid


# -----------------------------------------------------------------
# Appendix B2 — Fuerza axial
# -----------------------------------------------------------------
def test_full_scale_force_axial_matches_manual():
    """Do=2.25", GF=2.045, GXMT=4000 → 89,739 lb (manual pág. 36)."""
    shaft = ShaftGeometry(outer_diameter_in=2.25)
    gage = GageConfig(gage_factor=2.045, transmitter_gain=4000, bridge=BridgeType.AXIAL)
    pfs = full_scale_force_axial(shaft, gage)
    assert pfs == pytest.approx(89_739.0, rel=2e-3)


# -----------------------------------------------------------------
# Appendix B3 — ¼ puente (galga simple)
# -----------------------------------------------------------------
def test_full_scale_strain_quarter_matches_manual():
    """GF=2.045, GXMT=4000 → 1956 µε (manual pág. 37)."""
    gage = GageConfig(gage_factor=2.045, transmitter_gain=4000, bridge=BridgeType.QUARTER)
    eps = full_scale_strain_quarter(gage)
    assert eps == pytest.approx(1956.0, rel=1e-3)


def test_full_scale_strain_torque_gain_table():
    """Tabla de ganancia: GF=2.0, GXMT=4000 → 500 µε (datasheet pág. 2)."""
    gage = GageConfig(gage_factor=2.0, transmitter_gain=4000)
    assert full_scale_strain_torque(gage) == pytest.approx(500.0, rel=1e-6)


@pytest.mark.parametrize("gxmt,expected_ue", [
    (500, 4000.0), (1000, 2000.0), (2000, 1000.0),
    (4000, 500.0), (8000, 250.0), (16000, 125.0),
])
def test_full_scale_strain_torque_full_gain_table(gxmt, expected_ue):
    gage = GageConfig(gage_factor=2.0, transmitter_gain=gxmt)
    assert full_scale_strain_torque(gage) == pytest.approx(expected_ue, rel=1e-6)


# -----------------------------------------------------------------
# TorqueScaling y conversión voltaje → torque
# -----------------------------------------------------------------
def test_scaling_from_geometry_10v_equals_tfs():
    """Con Z=1, 10 V debe corresponder al TFS calculado."""
    shaft = ShaftGeometry(outer_diameter_in=3.000)
    gage = GageConfig(gage_factor=2.08, transmitter_gain=4000)
    sc = TorqueScaling.from_geometry(shaft, gage, units="ftlb")
    torque = voltage_to_torque(np.array([10.0]), sc)
    assert torque[0] == pytest.approx(sc.full_scale_torque, rel=1e-9)
    assert torque[0] == pytest.approx(4901.0, rel=1e-3)


def test_scaling_scale_factor_z():
    """Con Z=0.9802, 10 V ↔ TFS/Z (≈5000 ft-lb en el ejemplo del manual)."""
    shaft = ShaftGeometry(outer_diameter_in=3.000)
    gage = GageConfig(gage_factor=2.08, transmitter_gain=4000)
    sc = TorqueScaling.from_geometry(shaft, gage, units="ftlb", scale_factor_z=0.9802)
    torque = voltage_to_torque(np.array([10.0]), sc)
    assert torque[0] == pytest.approx(5000.0, rel=2e-3)


def test_scaling_linear_and_signed():
    shaft = ShaftGeometry(outer_diameter_in=2.0)
    gage = GageConfig(gage_factor=2.0, transmitter_gain=4000)
    sc = TorqueScaling.from_geometry(shaft, gage, units="nm")
    v = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    t = voltage_to_torque(v, sc)
    assert t[2] == pytest.approx(0.0, abs=1e-9)
    assert t[0] == pytest.approx(-t[4], rel=1e-9)
    assert t[3] == pytest.approx(t[4] / 2.0, rel=1e-9)


def test_voltage_to_torque_remove_dc():
    """remove_dc deja sólo el torque dinámico (media cero)."""
    shaft = ShaftGeometry(outer_diameter_in=2.0)
    gage = GageConfig(gage_factor=2.0, transmitter_gain=4000)
    sc = TorqueScaling.from_geometry(shaft, gage, units="nm")
    v = 5.0 + np.sin(np.linspace(0, 2 * np.pi, 100))  # bias DC 5 V + rizado
    t = voltage_to_torque(v, sc, remove_dc=True)
    assert np.mean(t) == pytest.approx(0.0, abs=1e-9)


def test_deadweight_calibration():
    """La calibración por peso muerto fija eu_per_volt directamente."""
    sc = TorqueScaling.from_deadweight(applied_torque=500.0, measured_volts=1.0, units="ftlb")
    assert sc.eu_per_volt == pytest.approx(500.0)
    assert voltage_to_torque(np.array([2.0]), sc)[0] == pytest.approx(1000.0)


# -----------------------------------------------------------------
# Shunt calibration
# -----------------------------------------------------------------
def test_expected_shunt_voltage_manual_example():
    """εs=100, εFS=500, Z=0.9802 → 1.9604 V (manual pág. 34)."""
    vs = expected_shunt_voltage(simulated_ue=100.0, full_scale_ue=500.0,
                                scale_factor_z=0.9802)
    assert vs == pytest.approx(1.9604, rel=1e-4)


def test_simulated_strain_from_shunt_ref_values():
    """Los shunts de fábrica dan ~100 y ~500 µε con 350 Ω / GF 2.0, N=4 (puente torque)."""
    eps1 = simulated_strain_from_shunt(REF1_100UE.resistance_ohm, 350.0, 4, 2.0)
    eps2 = simulated_strain_from_shunt(REF2_500UE.resistance_ohm, 350.0, 4, 2.0)
    assert eps1 == pytest.approx(100.0, rel=5e-3)
    assert eps2 == pytest.approx(500.0, rel=5e-3)


def test_verify_shunt_pass():
    """Lectura igual al esperado → aprueba, error ~0, Z sugerido ~ actual."""
    gage = GageConfig(gage_factor=2.0, transmitter_gain=4000)  # εFS=500
    expected = expected_shunt_voltage(100.0, 500.0, 1.0)       # 2.0 V
    chk = verify_shunt(measured_v=expected, shunt=REF1_100UE, gage=gage)
    assert chk.passed
    assert chk.error_pct == pytest.approx(0.0, abs=1e-9)
    assert chk.suggested_z == pytest.approx(1.0, rel=1e-6)


def test_verify_shunt_fail_suggests_correction():
    """Lectura desviada → falla y sugiere un Z que corrige la desviación."""
    gage = GageConfig(gage_factor=2.0, transmitter_gain=4000)
    chk = verify_shunt(measured_v=2.10, shunt=REF1_100UE, gage=gage, tolerance_pct=0.5)
    assert not chk.passed
    assert chk.expected_v == pytest.approx(2.0, rel=1e-9)
    # El Z efectivo sugerido reproduce la lectura medida (2.10 V).
    assert expected_shunt_voltage(100.0, 500.0, chk.suggested_z) == pytest.approx(2.10, rel=1e-6)


# -----------------------------------------------------------------
# Validaciones de entrada
# -----------------------------------------------------------------
def test_invalid_geometry_raises():
    with pytest.raises(ValueError):
        ShaftGeometry(outer_diameter_in=0.0)
    with pytest.raises(ValueError):
        ShaftGeometry(outer_diameter_in=2.0, inner_diameter_in=2.0)


def test_invalid_scale_factor_raises():
    shaft = ShaftGeometry(2.0)
    gage = GageConfig()
    with pytest.raises(ValueError):
        TorqueScaling.from_geometry(shaft, gage, scale_factor_z=5.0)
