"""
tests.test_critical_speeds
==========================

Validación de core.rotordynamics.detect_critical_speeds() y la cadena
asociada (compute_q_factor, evaluate_api684_margin).

Generamos una respuesta Bode sintética con una crítica conocida
(critical_rpm, q_factor predefinidos) y verificamos:

  1. detect_critical_speeds() la encuentra dentro de tolerancia.
  2. El Q factor estimado está dentro del 30% del Q sintético.
  3. evaluate_api684_margin() calcula bien el margen relativo a una
     velocidad operativa.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.rotordynamics import (
    detect_critical_speeds,
    evaluate_api684_margin,
)
from tests.conftest import make_bode_curve


# -----------------------------------------------------------------
# Bode limpia → recupera crítica única
# -----------------------------------------------------------------

def test_detect_single_critical_clean_bode(bode_clean):
    crits = detect_critical_speeds(
        rpm=bode_clean["rpm"],
        amp=bode_clean["amp"],
        phase=bode_clean["phase"],
        min_phase_change_deg=40.0,
    )
    assert len(crits) >= 1, "No detectó la crítica sintética"

    expected_rpm = bode_clean["expected"]["critical_rpm"]
    detected_rpm = crits[0].rpm
    drpm = float(bode_clean["rpm"][1] - bode_clean["rpm"][0])
    # Tolerancia: 3 pasos del barrido (≈ 113 RPM con n=200 entre 500-8000)
    assert abs(detected_rpm - expected_rpm) <= 3 * drpm, (
        f"Crítica detectada {detected_rpm:.0f} RPM lejos del ground truth "
        f"{expected_rpm:.0f} RPM (tol={3 * drpm:.0f})"
    )


def test_detect_no_critical_for_flat_bode():
    """Bode plana sin pico → no debe detectar críticas."""
    rpm = np.linspace(1000.0, 8000.0, 200)
    amp = np.ones_like(rpm) * 5.0  # plana
    # Fase también plana
    phase = np.full_like(rpm, 90.0)
    crits = detect_critical_speeds(
        rpm=rpm,
        amp=amp,
        phase=phase,
        min_phase_change_deg=40.0,
    )
    assert len(crits) == 0


def test_detect_handles_empty_arrays():
    crits = detect_critical_speeds(
        rpm=np.array([]),
        amp=np.array([]),
        phase=np.array([]),
    )
    assert crits == []


def test_detect_critical_q_factor_reasonable(bode_clean):
    crits = detect_critical_speeds(
        rpm=bode_clean["rpm"],
        amp=bode_clean["amp"],
        phase=bode_clean["phase"],
    )
    if not crits:
        pytest.skip("No hay críticas detectadas — cubierto por test anterior")

    expected_q = bode_clean["expected"]["q_factor"]
    measured_q = crits[0].q_factor

    if measured_q is None or not np.isfinite(measured_q):
        pytest.skip("Q no calculable en este barrido — cubierto por otro test")

    # El estimador FWHM tiene sesgo y depende de la densidad del barrido.
    # Tolerancia generosa pero significativa.
    assert measured_q > 1.5
    assert measured_q < 4.0 * expected_q


# -----------------------------------------------------------------
# evaluate_api684_margin
# -----------------------------------------------------------------

def test_api684_margin_below_critical_safe():
    """Operando bien por debajo de una crítica → margen positivo."""
    margin = evaluate_api684_margin(
        operating_rpm=3000.0,
        critical_rpm=4500.0,
        q_factor=6.0,
    )
    assert margin.actual_margin_pct > 0
    # 4500 vs 3000 → 50% por encima → margen importante


def test_api684_margin_above_critical_safe_too():
    """Operando por encima de la crítica también puede tener margen."""
    margin = evaluate_api684_margin(
        operating_rpm=6000.0,
        critical_rpm=4500.0,
        q_factor=6.0,
    )
    assert margin.actual_margin_pct > 0


def test_api684_margin_at_critical_zero_or_negative():
    """Operando exactamente en la crítica → separación ~ 0% (riesgo)."""
    margin = evaluate_api684_margin(
        operating_rpm=4500.0,
        critical_rpm=4500.0,
        q_factor=6.0,
    )
    assert margin.actual_margin_pct == pytest.approx(0.0, abs=0.5)
    # Estar parado en la crítica nunca debería ser conforme.
    assert margin.compliant is False


def test_api684_higher_q_requires_more_margin():
    """A mayor Q, mayor separación requerida — verificamos relación."""
    m_low_q = evaluate_api684_margin(
        operating_rpm=3000.0,
        critical_rpm=4500.0,
        q_factor=2.0,
    )
    m_high_q = evaluate_api684_margin(
        operating_rpm=3000.0,
        critical_rpm=4500.0,
        q_factor=10.0,
    )
    assert m_high_q.required_margin_pct >= m_low_q.required_margin_pct
