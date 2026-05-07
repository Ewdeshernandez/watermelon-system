"""
tests.test_waveform_metrics
===========================

Validación de core.waveform_metrics.compute_waveform_metrics() contra
señales sintéticas con valores cerrados conocidos.

Cierres analíticos para una sinusoide pura A·sin(ωt):
    RMS         = A / sqrt(2)
    peak        = A
    peak-to-pk  = 2A
    crest       = sqrt(2) ≈ 1.4142
    skewness    = 0
    kurtosis    = 1.5  (momento4/std^4 sin restar 3)

Si una nueva versión de waveform_metrics rompe estas igualdades, el test
falla y obliga a revisar la convención (kurtosis raw vs excess, RMS vs
peak, etc.) antes de mergear.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.waveform_metrics import compute_waveform_metrics


# -----------------------------------------------------------------
# Casos triviales
# -----------------------------------------------------------------

def test_compute_metrics_empty_returns_dict():
    m = compute_waveform_metrics(np.array([]))
    assert isinstance(m, dict)


def test_compute_metrics_none_returns_empty():
    m = compute_waveform_metrics(None)
    assert m == {}


def test_compute_metrics_constant_signal_zero_rms_zero_std():
    y = np.full(1000, 5.0)
    m = compute_waveform_metrics(y)
    # RMS de constante 5 → 5
    assert m["rms"] == pytest.approx(5.0, rel=1e-6)
    assert m["std"] == pytest.approx(0.0, abs=1e-9)
    assert m["mean"] == pytest.approx(5.0, rel=1e-9)


# -----------------------------------------------------------------
# Sinusoide pura — golden case
# -----------------------------------------------------------------

def test_pure_sine_rms(sine_clean):
    m = compute_waveform_metrics(sine_clean["y"])
    expected_rms = sine_clean["expected"]["rms"]
    assert m["rms"] == pytest.approx(expected_rms, rel=2e-3)


def test_pure_sine_peak(sine_clean):
    m = compute_waveform_metrics(sine_clean["y"])
    expected_peak = sine_clean["expected"]["peak"]
    assert m["peak"] == pytest.approx(expected_peak, rel=5e-3)


def test_pure_sine_peak_to_peak(sine_clean):
    m = compute_waveform_metrics(sine_clean["y"])
    expected_pp = sine_clean["expected"]["peak_to_peak"]
    assert m["peak_to_peak"] == pytest.approx(expected_pp, rel=5e-3)


def test_pure_sine_crest_factor(sine_clean):
    m = compute_waveform_metrics(sine_clean["y"])
    assert m["crest_factor"] == pytest.approx(math.sqrt(2.0), rel=1e-2)


def test_pure_sine_skewness_near_zero(sine_clean):
    m = compute_waveform_metrics(sine_clean["y"])
    assert abs(m["skewness"]) < 1e-2


# -----------------------------------------------------------------
# Crest factor sube con impactos (bearing fault)
# -----------------------------------------------------------------

def test_crest_factor_higher_for_impacts_than_for_sine(sine_clean, bearing_bpfo):
    m_sine = compute_waveform_metrics(sine_clean["y"])
    m_bear = compute_waveform_metrics(bearing_bpfo["y"])
    # Una señal con impactos debe tener crest_factor mayor que una sine pura.
    # Tolerancia generosa para no depender de tunings finos.
    assert m_bear["crest_factor"] > m_sine["crest_factor"]


def test_kurtosis_higher_for_impacts_than_for_sine(sine_clean, bearing_bpfo):
    m_sine = compute_waveform_metrics(sine_clean["y"])
    m_bear = compute_waveform_metrics(bearing_bpfo["y"])
    # Kurtosis debe ser claramente mayor en impactos.
    assert m_bear["kurtosis"] > m_sine["kurtosis"]


# -----------------------------------------------------------------
# Robustez: NaN/Inf removal
# -----------------------------------------------------------------

def test_metrics_filter_non_finite():
    y = np.array([0.0, 1.0, np.nan, np.inf, -np.inf, -1.0, 0.5])
    m = compute_waveform_metrics(y)
    # Sólo deben quedar 4 muestras finitas, samples=4.
    assert m["samples"] == 4
    assert math.isfinite(m["rms"])


# -----------------------------------------------------------------
# Lista vs ndarray
# -----------------------------------------------------------------

def test_metrics_accepts_list_input():
    m_list = compute_waveform_metrics([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    m_np = compute_waveform_metrics(np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]))
    assert m_list["rms"] == pytest.approx(m_np["rms"], rel=1e-9)
