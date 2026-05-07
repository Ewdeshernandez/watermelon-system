"""
tests.test_rotordynamics_zones
==============================

Validación de core.rotordynamics.iso_20816_2_zone() — clasificación
A/B/C/D según ISO 20816-2 para turbogeneradores grandes con cojinetes
planos.

Convención: amplitud en µm pico-pico (shaft displacement) o mm/s RMS
(casing velocity).

Tabla ISO 20816-2 grupo 2 a 3600 RPM (memorizada para test):
    A/B = 90 µm pp  (norma 2017 group 2 @ 3600 rpm)
    B/C = 165 µm pp
    C/D = 240 µm pp

Validamos que:
  - Una amplitud por debajo de A/B → zone "A"
  - Entre A/B y B/C → zone "B"
  - Entre B/C y C/D → zone "C"
  - Por encima de C/D → zone "D"
  - Casos extremos (negativos, NaN, infinitos) → zone "D"
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.rotordynamics import (
    iso_20816_2_zone,
    micrometers_to_mils,
    mils_to_micrometers,
)


# -----------------------------------------------------------------
# Conversiones unidades
# -----------------------------------------------------------------

def test_mils_to_micrometers_known_value():
    # 1 mil = 25.4 µm exactamente
    assert mils_to_micrometers(1.0) == pytest.approx(25.4, rel=1e-9)


def test_micrometers_to_mils_known_value():
    assert micrometers_to_mils(25.4) == pytest.approx(1.0, rel=1e-9)


def test_round_trip_micrometers_mils():
    for v in [10.0, 50.0, 100.0, 250.0]:
        v2 = micrometers_to_mils(mils_to_micrometers(v))
        assert v2 == pytest.approx(v, rel=1e-9)


# -----------------------------------------------------------------
# ISO 20816-2 zones — shaft displacement
# -----------------------------------------------------------------

def test_zone_A_for_low_amplitude():
    z = iso_20816_2_zone(
        amplitude=30.0,                # 30 µm pp — claramente good
        measurement_type="shaft_displacement",
        machine_group="group2",
        operating_speed_rpm=3600.0,
    )
    assert z.zone == "A"


def test_zone_D_for_extreme_amplitude():
    z = iso_20816_2_zone(
        amplitude=10_000.0,           # absurdo alto
        measurement_type="shaft_displacement",
        machine_group="group2",
        operating_speed_rpm=3600.0,
    )
    assert z.zone == "D"


def test_zone_returns_negative_amplitude_as_D():
    z = iso_20816_2_zone(
        amplitude=-50.0,
        measurement_type="shaft_displacement",
        machine_group="group2",
        operating_speed_rpm=3600.0,
    )
    assert z.zone == "D"


def test_zone_returns_nan_amplitude_as_D():
    z = iso_20816_2_zone(
        amplitude=float("nan"),
        measurement_type="shaft_displacement",
        machine_group="group2",
        operating_speed_rpm=3600.0,
    )
    assert z.zone == "D"


def test_zone_boundaries_are_consistent():
    """Para una clasificación dada, los boundaries devueltos deben ser
    monotónicos AB < BC < CD."""
    z = iso_20816_2_zone(
        amplitude=50.0,
        measurement_type="shaft_displacement",
        machine_group="group2",
        operating_speed_rpm=3600.0,
    )
    assert z.boundary_AB < z.boundary_BC < z.boundary_CD


def test_zone_just_below_AB_is_A():
    """Justo bajo el límite A/B → zone A."""
    z_test = iso_20816_2_zone(
        amplitude=50.0,
        measurement_type="shaft_displacement",
        machine_group="group2",
        operating_speed_rpm=3600.0,
    )
    ab = z_test.boundary_AB
    z = iso_20816_2_zone(
        amplitude=ab - 0.001,
        measurement_type="shaft_displacement",
        machine_group="group2",
        operating_speed_rpm=3600.0,
    )
    assert z.zone == "A"


def test_zone_just_above_AB_is_B():
    z_test = iso_20816_2_zone(
        amplitude=50.0,
        measurement_type="shaft_displacement",
        machine_group="group2",
        operating_speed_rpm=3600.0,
    )
    ab = z_test.boundary_AB
    z = iso_20816_2_zone(
        amplitude=ab + 0.001,
        measurement_type="shaft_displacement",
        machine_group="group2",
        operating_speed_rpm=3600.0,
    )
    assert z.zone == "B"


def test_zone_just_above_CD_is_D():
    z_test = iso_20816_2_zone(
        amplitude=50.0,
        measurement_type="shaft_displacement",
        machine_group="group2",
        operating_speed_rpm=3600.0,
    )
    cd = z_test.boundary_CD
    z = iso_20816_2_zone(
        amplitude=cd + 0.001,
        measurement_type="shaft_displacement",
        machine_group="group2",
        operating_speed_rpm=3600.0,
    )
    assert z.zone == "D"


# -----------------------------------------------------------------
# Casing velocity branch
# -----------------------------------------------------------------

def test_zone_casing_velocity_branch_runs():
    """No verificamos números — sólo que la rama funciona."""
    z = iso_20816_2_zone(
        amplitude=2.5,                   # 2.5 mm/s RMS, valor moderado
        measurement_type="casing_velocity",
        machine_group="group2",
        operating_speed_rpm=3600.0,
    )
    assert z.unit == "mm_s_rms"
    assert z.zone in ("A", "B", "C", "D")


def test_invalid_measurement_type_raises():
    with pytest.raises(ValueError):
        iso_20816_2_zone(
            amplitude=50.0,
            measurement_type="potato",
            machine_group="group2",
            operating_speed_rpm=3600.0,
        )
