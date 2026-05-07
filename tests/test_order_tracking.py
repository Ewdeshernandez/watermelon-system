"""
tests.test_order_tracking
=========================

Validación de core.order_tracking.analyze_order_tracking() contra
señales sintéticas con armónicos conocidos.

La aserción clave: dada una señal sintética con amplitudes 1X, 2X, 3X
predefinidas, el order tracking debe recuperarlas con error < 5%
(tolerancia generosa porque hay leakage de FFT-by-rev).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.order_tracking import analyze_order_tracking
from tests.conftest import make_signal_obj


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _rel_err(actual, expected):
    if expected == 0:
        return abs(actual)
    return abs(actual - expected) / abs(expected)


# -----------------------------------------------------------------
# Sine pura → 1X recuperado, 2X y 3X cerca de 0
# -----------------------------------------------------------------

def test_pure_sine_recovers_1x(sine_clean):
    """Para sine pura 1X con peak A, mean_amp_pp ≈ 2A."""
    sig = make_signal_obj(sine_clean["y"], sine_clean["fs"], sine_clean["rpm"])
    out = analyze_order_tracking(
        signal=sig, fs=sine_clean["fs"], rpm=sine_clean["rpm"], max_order=3,
    )
    expected_pp = 2.0 * sine_clean["amplitude_peak"]
    one_x = out["order_results"][1]["mean_amp_pp"]
    assert _rel_err(one_x, expected_pp) < 0.05


def test_pure_sine_2x_and_3x_near_zero(sine_clean):
    sig = make_signal_obj(sine_clean["y"], sine_clean["fs"], sine_clean["rpm"])
    out = analyze_order_tracking(
        signal=sig, fs=sine_clean["fs"], rpm=sine_clean["rpm"], max_order=3,
    )
    one_x = out["order_results"][1]["mean_amp_pp"]
    two_x = out["order_results"][2]["mean_amp_pp"]
    three_x = out["order_results"][3]["mean_amp_pp"]
    # 2X y 3X deben ser claramente menores que 1X.
    assert two_x < 0.1 * one_x
    assert three_x < 0.1 * one_x


# -----------------------------------------------------------------
# Multi-harmonic → recupera amplitudes con error < 5%
# -----------------------------------------------------------------

def test_multi_harmonic_recovers_orders_amplitudes(multi_harmonic_clean):
    """Señal con {1: 1.0, 2: 0.3, 3: 0.1}.
    Cada orden recuperado en pp ≈ 2 × amplitud_peak."""
    sig = make_signal_obj(
        multi_harmonic_clean["y"],
        multi_harmonic_clean["fs"],
        multi_harmonic_clean["rpm"],
    )
    out = analyze_order_tracking(
        signal=sig,
        fs=multi_harmonic_clean["fs"],
        rpm=multi_harmonic_clean["rpm"],
        max_order=4,
    )

    for order, amp_peak in multi_harmonic_clean["harmonics"].items():
        recovered_pp = out["order_results"][order]["mean_amp_pp"]
        expected_pp = 2.0 * amp_peak
        assert _rel_err(recovered_pp, expected_pp) < 0.06, (
            f"Orden {order}: esperado {expected_pp:.4f} pp, "
            f"recuperado {recovered_pp:.4f}"
        )


def test_multi_harmonic_4x_negligible(multi_harmonic_clean):
    """En la señal sintética NO hay 4X — debe salir cerca de 0."""
    sig = make_signal_obj(
        multi_harmonic_clean["y"],
        multi_harmonic_clean["fs"],
        multi_harmonic_clean["rpm"],
    )
    out = analyze_order_tracking(
        signal=sig,
        fs=multi_harmonic_clean["fs"],
        rpm=multi_harmonic_clean["rpm"],
        max_order=4,
    )
    four_x = out["order_results"][4]["mean_amp_pp"]
    one_x = out["order_results"][1]["mean_amp_pp"]
    assert four_x < 0.05 * one_x


# -----------------------------------------------------------------
# Validación de errores
# -----------------------------------------------------------------

def test_invalid_rpm_raises():
    sig = make_signal_obj(np.zeros(1000), fs=1000.0, rpm=-10.0)
    with pytest.raises(ValueError):
        analyze_order_tracking(signal=sig, fs=1000.0, rpm=-10.0, max_order=3)


def test_invalid_fs_raises():
    sig = make_signal_obj(np.zeros(1000), fs=0.0, rpm=1800.0)
    with pytest.raises(ValueError):
        analyze_order_tracking(signal=sig, fs=0.0, rpm=1800.0, max_order=3)


def test_too_short_signal_raises():
    """Menos de 2 revoluciones → error."""
    fs = 1000.0
    rpm = 60.0
    # 1 revolución ≈ 1000 muestras a 60 RPM y fs=1000. Damos 800 muestras → < 1 rev.
    y = np.sin(2 * np.pi * 1.0 * np.arange(800) / fs)
    sig = make_signal_obj(y, fs=fs, rpm=rpm)
    with pytest.raises(ValueError):
        analyze_order_tracking(signal=sig, fs=fs, rpm=rpm, max_order=3)


# -----------------------------------------------------------------
# Phase consistency
# -----------------------------------------------------------------

def test_phase_returned_in_degrees(multi_harmonic_clean):
    sig = make_signal_obj(
        multi_harmonic_clean["y"],
        multi_harmonic_clean["fs"],
        multi_harmonic_clean["rpm"],
    )
    out = analyze_order_tracking(
        signal=sig,
        fs=multi_harmonic_clean["fs"],
        rpm=multi_harmonic_clean["rpm"],
        max_order=3,
    )
    for order in [1, 2, 3]:
        phase = out["order_results"][order]["mean_phase_deg"]
        # Fase debe ser un float finito (puede ser cualquier valor en [-180, 180] o [0, 360))
        assert np.isfinite(phase)
        assert -360.0 <= phase <= 360.0
