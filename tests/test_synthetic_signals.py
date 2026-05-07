"""
tests.test_synthetic_signals
============================

Sanity check de los generadores definidos en conftest.py.

Si estos tests fallan, los demás tests no son confiables porque su
ground-truth está roto. Por eso es la primera barrera: garantizar que
los datos de entrada tienen las propiedades que documentamos.
"""

from __future__ import annotations

import numpy as np

from tests.conftest import (
    DEFAULT_FS,
    DEFAULT_RPM,
    hz_from_rpm,
    make_bearing_impacts,
    make_bode_curve,
    make_looseness,
    make_multi_harmonic,
    make_orbit_xy,
    make_pure_sine,
    make_subsynchronous,
    make_time,
)


# -----------------------------------------------------------------
# Time vector
# -----------------------------------------------------------------

def test_make_time_length_and_dt():
    t = make_time(fs=1000.0, duration=2.0)
    assert t.size == 2000
    dt = np.diff(t)
    np.testing.assert_allclose(dt, 1e-3, rtol=1e-12)


def test_hz_from_rpm():
    assert hz_from_rpm(3600.0) == 60.0
    assert hz_from_rpm(1800.0) == 30.0
    assert hz_from_rpm(0.0) == 0.0


# -----------------------------------------------------------------
# Pure sine
# -----------------------------------------------------------------

def test_pure_sine_shape_and_no_dc(sine_clean):
    y = sine_clean["y"]
    assert y.size == int(DEFAULT_FS * sine_clean["duration"])
    # Sin DC offset declarado, la media debe estar cerca de 0.
    assert abs(float(np.mean(y))) < 1e-3


def test_pure_sine_peak_within_5pct(sine_clean):
    expected_peak = sine_clean["expected"]["peak"]
    measured_peak = float(np.max(np.abs(sine_clean["y"])))
    assert abs(measured_peak - expected_peak) / expected_peak < 0.05


def test_pure_sine_rms_close_to_A_over_sqrt2(sine_clean):
    expected = sine_clean["expected"]["rms"]
    measured = float(np.sqrt(np.mean(sine_clean["y"] ** 2)))
    np.testing.assert_allclose(measured, expected, rtol=2e-3)


def test_pure_sine_with_dc_offset_changes_mean():
    sig = make_pure_sine(dc_offset=2.5)
    assert abs(float(np.mean(sig["y"])) - 2.5) < 1e-3


def test_pure_sine_noise_seed_is_reproducible():
    a = make_pure_sine(noise_rms=0.1, seed=123)["y"]
    b = make_pure_sine(noise_rms=0.1, seed=123)["y"]
    np.testing.assert_allclose(a, b)


# -----------------------------------------------------------------
# Multi-harmonic
# -----------------------------------------------------------------

def test_multi_harmonic_default_orders(multi_harmonic_clean):
    assert set(multi_harmonic_clean["harmonics"].keys()) == {1, 2, 3}
    assert multi_harmonic_clean["y"].size > 0


def test_multi_harmonic_misalignment_2x_dominant(multi_harmonic_misalignment):
    """En misalignment paralelo, 2X tiene amplitud mayor que 1X."""
    h = multi_harmonic_misalignment["harmonics"]
    assert h[2] > h[1]


# -----------------------------------------------------------------
# Subsynchronous
# -----------------------------------------------------------------

def test_subsync_carries_two_components(subsync_oil_whirl):
    assert subsync_oil_whirl["sub_order"] < 0.5  # oil whirl < 0.5X
    assert subsync_oil_whirl["sub_amp"] > 0
    assert subsync_oil_whirl["one_x_amp"] > 0


# -----------------------------------------------------------------
# Bearing impacts
# -----------------------------------------------------------------

def test_bearing_bpfo_frequency_correct(bearing_bpfo):
    expected = bearing_bpfo["bpfo_factor"] * hz_from_rpm(bearing_bpfo["rpm"])
    assert abs(bearing_bpfo["bpfo_hz"] - expected) < 1e-9


def test_bearing_signal_is_finite(bearing_bpfo):
    y = bearing_bpfo["y"]
    assert np.all(np.isfinite(y))
    assert y.size > 0


# -----------------------------------------------------------------
# Looseness
# -----------------------------------------------------------------

def test_looseness_default_six_harmonics(looseness_pattern):
    assert looseness_pattern["n_harmonics"] == 6


# -----------------------------------------------------------------
# Bode
# -----------------------------------------------------------------

def test_bode_amplitude_peaks_near_critical(bode_clean):
    rpm = bode_clean["rpm"]
    amp = bode_clean["amp"]
    idx_max = int(np.argmax(amp))
    rpm_at_peak = float(rpm[idx_max])
    expected_critical = bode_clean["expected"]["critical_rpm"]
    # Debe coincidir dentro de la resolución del barrido (Δrpm ≈ 37.7 con n=200)
    drpm = float(rpm[1] - rpm[0])
    assert abs(rpm_at_peak - expected_critical) <= 2 * drpm


def test_bode_phase_in_valid_range(bode_clean):
    phase = bode_clean["phase"]
    assert np.all(phase >= 0.0)
    assert np.all(phase < 360.0 + 1e-6)


def test_bode_phase_swings_at_least_120deg(bode_clean):
    """En una crítica clara, la fase debe cambiar > 120° entre rpm_min y rpm_max."""
    phase = bode_clean["phase"]
    swing = float(np.max(phase) - np.min(phase))
    # Convertimos a swing efectivo considerando wrapping
    # (forma cruda — verificamos al menos que rebasa los 120°)
    assert swing > 120.0


# -----------------------------------------------------------------
# Orbit
# -----------------------------------------------------------------

def test_orbit_circular_size_consistent(orbit_circular):
    assert orbit_circular["x"].size == orbit_circular["y"].size
    expected_n = orbit_circular["samples_per_rev"] * orbit_circular["n_revs"]
    assert orbit_circular["x"].size == expected_n


def test_orbit_circular_amplitude_close(orbit_circular):
    """Para órbita circular, amp_x ≈ amp_y (dentro de ruido)."""
    amp_x_meas = float(np.max(orbit_circular["x"]) - np.min(orbit_circular["x"]))
    amp_y_meas = float(np.max(orbit_circular["y"]) - np.min(orbit_circular["y"]))
    assert abs(amp_x_meas - amp_y_meas) < 0.1 * orbit_circular["amp_x_pp"]
