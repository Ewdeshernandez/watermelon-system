"""
tests.test_balance_engine
=========================

Regresión del motor de balanceo (core.balance.engine), extraído de ROTORIX
y validado en campo. Bloquea las fórmulas ISO 21940-11/12 + API 684 y prueba
los solvers con CASOS ANALÍTICOS EXACTOS (respuesta cerrada conocida):

  - 1 plano: con coeficiente de influencia H = 1∠0, la corrección debe ser
    -V0 y la vibración predicha 0.
  - 2 planos: con matriz H = identidad, las correcciones deben ser -A0 / -B0
    y las vibraciones después 0.

Además valida los permisibles ISO, el peso de prueba API 684, la evaluación
por grados y el diagnóstico estático/couple.

Corre con pytest (repo) o directo:  python tests/test_balance_engine.py
"""
from __future__ import annotations

import math

from core.balance.engine import (
    to_complex, to_polar, polar_to_complex, complex_to_polar,
    umax_api684_gmm, recommend_trial_weight_g,
    calc_e_per, calc_U_per, calc_U_trial, pct_reduction, status_from_ratio,
    ISO_GRADES, evaluate_iso_grades,
    solve_1plane, solve_2plane,
    status_level, diagnose_static_couple,
)


def _close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# Helpers complejos / polar
# ---------------------------------------------------------------------------
def test_polar_roundtrip():
    mag, ang = to_polar(to_complex(5.0, 30.0))
    assert _close(mag, 5.0)
    assert _close(ang, 30.0)


def test_aliases_are_same_functions():
    # La migración de la UI de ROTORIX depende de estos alias.
    assert polar_to_complex is to_complex
    assert complex_to_polar is to_polar


# ---------------------------------------------------------------------------
# 1 PLANO — caso analítico exacto (H = 1∠0)
# ---------------------------------------------------------------------------
def test_solve_1plane_analytical_identity():
    # V0 = 2∠0, trial = 1g∠0. Con H=1, Vt = V0 + H·Wt = 3∠0.
    r = solve_1plane(V0_mag=2.0, V0_ang=0.0,
                     Vt_mag=3.0, Vt_ang=0.0,
                     trial_mass_g=1.0, trial_ang_deg=0.0)
    assert _close(abs(r["H"]), 1.0)
    assert _close(r["corr_mass_g"], 2.0)        # Wcorr = -V0/H = 2∠180
    assert _close(r["corr_ang_deg"], 180.0)
    assert _close(r["pred_mag"], 0.0, tol=1e-9)  # vibración residual ~0
    assert r["quality"] == "GOOD"


def test_solve_1plane_matches_closed_form():
    # Caso arbitrario: se compara contra la fórmula cerrada re-derivada aquí.
    V0 = to_complex(5.0, 30.0)
    Vt = to_complex(8.0, 70.0)
    Wt = to_complex(10.0, 0.0)
    H = (Vt - V0) / Wt
    Wcorr = -V0 / H
    exp_mag, exp_ang = to_polar(Wcorr)

    r = solve_1plane(5.0, 30.0, 8.0, 70.0, 10.0, 0.0)
    assert _close(r["corr_mass_g"], exp_mag)
    assert _close(r["corr_ang_deg"], exp_ang)


def test_solve_1plane_zero_trial_raises():
    try:
        solve_1plane(2.0, 0.0, 3.0, 0.0, 0.0, 0.0)
    except ValueError:
        return
    raise AssertionError("Debe lanzar ValueError con peso de prueba cero")


# ---------------------------------------------------------------------------
# 2 PLANOS — caso analítico exacto (matriz H = identidad)
# ---------------------------------------------------------------------------
def test_solve_2plane_analytical_identity():
    A0 = to_complex(2.0, 0.0)
    B0 = to_complex(3.0, 0.0)
    WA = to_complex(1.0, 0.0)
    WB = to_complex(1.0, 0.0)
    # H = I  ->  run A: A1=A0+1, B1=B0 ; run B: A2=A0, B2=B0+1
    A1, B1 = A0 + WA, B0
    A2, B2 = A0, B0 + WB

    r = solve_2plane(A0, B0, A1, B1, A2, B2, WA, WB)
    wa_mag, wa_ang = to_polar(r["WA_corr"])
    wb_mag, wb_ang = to_polar(r["WB_corr"])

    assert _close(wa_mag, 2.0) and _close(wa_ang, 180.0)   # -A0
    assert _close(wb_mag, 3.0) and _close(wb_ang, 180.0)   # -B0
    assert _close(abs(r["A_after"]), 0.0, tol=1e-9)
    assert _close(abs(r["B_after"]), 0.0, tol=1e-9)
    assert r["quality"] == "GOOD"


def test_solve_2plane_zero_trial_raises():
    z = to_complex(1.0, 0.0)
    try:
        solve_2plane(z, z, z, z, z, z, to_complex(0.0, 0.0), z)
    except ValueError:
        return
    raise AssertionError("Debe lanzar ValueError con trial cero")


def test_solve_2plane_singular_raises_valueerror():
    # Corridas 1 y 2 con respuesta IDÉNTICA → columnas de M paralelas →
    # matriz singular. Debe lanzar ValueError limpio (no LinAlgError cruda).
    A0 = to_complex(1.0, 0.0); B0 = to_complex(1.0, 0.0)
    A1 = to_complex(1.1, 0.0); B1 = to_complex(1.05, 0.0)
    A2 = to_complex(1.1, 0.0); B2 = to_complex(1.05, 0.0)   # = corrida 1
    W = to_complex(1.0, 0.0)
    try:
        solve_2plane(A0, B0, A1, B1, A2, B2, W, W)
    except ValueError:
        return
    raise AssertionError("Matriz singular debe lanzar ValueError")


# ---------------------------------------------------------------------------
# ISO 21940-11 — permisibles
# ---------------------------------------------------------------------------
def test_iso_e_per_and_u_per():
    # e_per = 9549·G/N ; U_per = e_per·W
    assert _close(calc_e_per(2.5, 3600.0), 9549.0 * 2.5 / 3600.0)
    e = calc_e_per(2.5, 3600.0)
    assert _close(calc_U_per(e, 11000.0), e * 11000.0)
    assert _close(calc_e_per(2.5, 0.0), 0.0)          # rpm<=0 -> 0
    assert _close(calc_U_trial(50.0, 420.0), 21000.0)  # masa·radio


def test_pct_reduction_and_status():
    assert _close(pct_reduction(5.0, 1.0), 80.0)
    assert status_from_ratio(0.80) == ("CUMPLE", "ok")
    assert status_from_ratio(0.95) == ("LÍMITE", "warn")
    assert status_from_ratio(1.20) == ("NO CUMPLE", "bad")


# ---------------------------------------------------------------------------
# API 684 — peso de prueba
# ---------------------------------------------------------------------------
def test_api684_trial_weight():
    assert _close(umax_api684_gmm(3500.0, 3600.0), 6350.0 * 3500.0 / 3600.0)
    Wtrial, Utrial = recommend_trial_weight_g(3500.0, 3600.0, 420.0, 1.25)
    umax = umax_api684_gmm(3500.0, 3600.0)
    assert _close(Utrial, umax * 1.25)
    assert _close(Wtrial, Utrial / 420.0)
    # radio<=0 -> (0,0)
    assert recommend_trial_weight_g(3500.0, 3600.0, 0.0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Grados ISO 21940-12
# ---------------------------------------------------------------------------
def test_evaluate_iso_grades_pass_and_fail():
    # U_res minúsculo -> cumple hasta el grado más fino (0.4)
    ok = evaluate_iso_grades(11000.0, 3600.0, U_res=1.0)
    assert ok["best_grade"] == 0.4
    assert ok["status_code"] == "EXCELLENT"
    # U_res enorme -> no cumple ninguno
    bad = evaluate_iso_grades(11000.0, 3600.0, U_res=1e12)
    assert bad["best_grade"] is None
    assert bad["status_code"] == "FAIL"
    assert ISO_GRADES == [0.4, 1.0, 2.5, 6.3, 16.0]


# ---------------------------------------------------------------------------
# Diagnóstico estático / couple + niveles
# ---------------------------------------------------------------------------
def test_diagnose_static_couple_kinds():
    assert diagnose_static_couple(10.0, 1.0, 1.0)["kind"] == "STATIC"
    assert diagnose_static_couple(1.0, 10.0, 10.0)["kind"] == "COUPLE"
    mixed = diagnose_static_couple(10.0, 10.0, 10.0)
    assert mixed["kind"] == "MIXED"
    assert mixed["dominant"] == "COUPLE"   # Cavg >= S


def test_status_level():
    assert status_level(2.0, 3.0, alarm=2.5, trip=4.0) == "ALARM"
    assert status_level(5.0, 1.0, alarm=2.5, trip=4.0) == "TRIP"
    assert status_level(1.0, 1.0, alarm=2.5, trip=4.0) == "PASS"


# ---------------------------------------------------------------------------
# Runner standalone (sin pytest)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
