"""Tests del motor de balanceo por coeficientes de influencia."""
import numpy as np
import pytest

from core.balancing.influence import (
    phasor, to_polar, Weight, single_plane, two_plane,
    iso21940_permissible, combine_weights, split_to_holes,
)


def _close(z1, z2, tol=1e-6):
    return abs(complex(z1) - complex(z2)) < tol


# ---------------------------------------------------------------- fasores
def test_phasor_roundtrip():
    z = phasor(8.6, 63.0)
    a, p = to_polar(z)
    assert abs(a - 8.6) < 1e-9 and abs(p - 63.0) < 1e-9


# ---------------------------------------------------------------- un plano
def test_single_plane_recovers_correction():
    """Con influencia y desbalance conocidos, el solver da W_c = −U."""
    alpha_true = phasor(0.5, 30.0)          # vib por gramo
    U = phasor(12.0, 80.0)                   # desbalance real (gramos ∠°)
    v0 = alpha_true * U                       # vibración inicial
    Wt = Weight(10.0, 0.0)                    # peso de prueba
    vt = alpha_true * (U + Wt.vector)         # vibración con peso de prueba
    res = single_plane(v0, vt, Wt)
    assert _close(res.influence[0], alpha_true, 1e-6)
    # corrección debe ser −U (misma magnitud, 180° opuesto al desbalance)
    assert _close(res.corrections[0].vector, -U, 1e-6)
    assert abs(res.v_residual[0]) < 1e-6          # residual ≈ 0
    assert res.effectiveness_pct > 99.9


def test_single_plane_zero_trial_raises():
    with pytest.raises(ValueError):
        single_plane(phasor(5, 0), phasor(6, 10), Weight(0.0, 0.0))


# ---------------------------------------------------------------- dos planos
def test_two_plane_recovers_corrections():
    a = np.array([[phasor(0.4, 20), phasor(0.15, 200)],
                  [phasor(0.12, 160), phasor(0.5, -30)]], dtype=complex)
    U1, U2 = phasor(9.0, 45.0), phasor(6.0, 300.0)
    va0 = a[0, 0] * U1 + a[0, 1] * U2
    vb0 = a[1, 0] * U1 + a[1, 1] * U2
    Wt1, Wt2 = Weight(8.0, 0.0), Weight(8.0, 0.0)
    va1 = va0 + a[0, 0] * Wt1.vector; vb1 = vb0 + a[1, 0] * Wt1.vector
    va2 = va0 + a[0, 1] * Wt2.vector; vb2 = vb0 + a[1, 1] * Wt2.vector
    res = two_plane([va0, vb0], (Wt1, [va1, vb1]), (Wt2, [va2, vb2]))
    assert _close(res.corrections[0].vector, -U1, 1e-5)
    assert _close(res.corrections[1].vector, -U2, 1e-5)
    assert abs(res.v_residual[0]) < 1e-5 and abs(res.v_residual[1]) < 1e-5
    assert res.effectiveness_pct > 99.9


def test_two_plane_singular_raises():
    """Pesos de prueba que dan columnas dependientes → error claro."""
    a = np.array([[phasor(0.4, 20), phasor(0.4, 20)],
                  [phasor(0.4, 20), phasor(0.4, 20)]], dtype=complex)
    U1, U2 = phasor(9.0, 45.0), phasor(6.0, 300.0)
    va0 = a[0, 0] * U1 + a[0, 1] * U2; vb0 = a[1, 0] * U1 + a[1, 1] * U2
    Wt = Weight(8.0, 0.0)
    va1 = va0 + a[0, 0] * Wt.vector; vb1 = vb0 + a[1, 0] * Wt.vector
    va2 = va0 + a[0, 1] * Wt.vector; vb2 = vb0 + a[1, 1] * Wt.vector
    with pytest.raises(ValueError):
        two_plane([va0, vb0], (Wt, [va1, vb1]), (Wt, [va2, vb2]))


# ---------------------------------------------------------------- ISO 21940
def test_iso21940_formula():
    chk = iso21940_permissible(2.5, 100.0, 3000.0)
    assert abs(chk.u_per_g_mm - 9549 * 2.5 * 100 / 3000) < 1e-6      # ≈ 795.75 g·mm
    assert abs(chk.e_per_um - chk.u_per_g_mm / 100.0) < 1e-9         # ≈ 7.96 µm


def test_iso21940_pass_fail():
    ok = iso21940_permissible(6.3, 50.0, 1800.0, residual_g_mm=100.0)
    assert ok.passed is (100.0 <= ok.u_per_g_mm)
    bad = iso21940_permissible(1.0, 50.0, 1800.0, residual_g_mm=1e6)
    assert bad.passed is False


# ---------------------------------------------------------------- utilidades
def test_split_to_holes_reconstructs_vector():
    w = Weight(10.0, 37.0)
    h0, h1 = split_to_holes(w, 8)                # agujeros cada 45°
    assert _close(h0.vector + h1.vector, w.vector, 1e-6)
    assert h0.angle_deg in (0.0, 45.0) and h1.angle_deg in (0.0, 45.0)


def test_combine_weights_vector_sum():
    a = Weight(5.0, 0.0); b = Weight(5.0, 90.0)
    c = combine_weights(a, b)
    amp, ang = to_polar(c.vector)
    assert abs(amp - np.hypot(5, 5)) < 1e-6 and abs(ang - 45.0) < 1e-6
