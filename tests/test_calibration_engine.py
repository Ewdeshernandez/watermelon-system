"""
tests.test_calibration_engine
=============================

Regresión del motor de calibración (core.calibration.engine). Bloquea:

  - PROXIMIDAD contra el archivo real del cliente (lazo 1YD del calibrador
    portátil Bently/GE v3.2.1): best-fit slope 0.19852 V/mil, intercept
    0.10528 V, ISF por tramo y máx. linealidad 0.89 %.
  - Criterios API 670: ISF ±5 %, DSL ±1 mil, rango ≥ 80 mil.
  - Casos analíticos exactos (recta perfecta → r2=1, desviación 0).
  - Respuesta en frecuencia: −6 dB ⇒ 20·log10(0.5) = −6.02 dB.
  - check_principal_sensitivity 100 mV/g ± 5 %.
  - Errores esperados (menos de 2 puntos, x constante).

Corre con pytest o directo:  python tests/test_calibration_engine.py
"""
from __future__ import annotations

import math

from core.calibration.engine import (
    best_fit_line, incremental_scale_factors,
    analyze_proximity_linearity, analyze_amplitude_linearity,
    analyze_frequency_response, check_principal_sensitivity,
)

# Datos reales del Excel "Curva de sensibilidad 1YD.xlsm" (mil p-p, V).
_YD_X = [10, 20, 30, 40, 50, 60, 70, 80, 90]
_YD_Y = [1.97, 4.05, 6.11, 8.11, 10.12, 12.08, 14.01, 15.95, 17.88]


def _close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# Best-fit
# ---------------------------------------------------------------------------
def test_best_fit_exact_line():
    # y = 3x + 2 exacto → slope 3, intercept 2, r2 = 1.
    x = [0, 1, 2, 3, 4]
    y = [2, 5, 8, 11, 14]
    m, b, r2 = best_fit_line(x, y)
    assert _close(m, 3.0)
    assert _close(b, 2.0)
    assert _close(r2, 1.0)


def test_best_fit_matches_excel_1yd():
    m, b, r2 = best_fit_line(_YD_X, _YD_Y)
    assert _close(m, 0.19852, tol=1e-4)      # Excel SLOPE
    assert _close(b, 0.10528, tol=1e-4)      # Excel INTERCEPT
    assert r2 > 0.999


# ---------------------------------------------------------------------------
# ISF
# ---------------------------------------------------------------------------
def test_incremental_scale_factors():
    xmid, isf = incremental_scale_factors(_YD_X, _YD_Y)
    # primer tramo (4.05-1.97)/(20-10) = 0.208 V/mil
    assert _close(isf[0], 0.208, tol=1e-9)
    assert _close(xmid[0], 15.0)
    assert len(isf) == len(_YD_X) - 1


# ---------------------------------------------------------------------------
# Proximidad — validación completa vs Excel + API 670
# ---------------------------------------------------------------------------
def test_proximity_1yd_full():
    a = analyze_proximity_linearity(_YD_X, _YD_Y, x_unit="mil")
    assert _close(a["slope_v_per_x"], 0.19852, tol=1e-4)
    assert _close(a["asf_mv_per_x"], 198.52, tol=1e-1)
    # Máx. linealidad del Excel = 0.89 % (excluye starting point)
    assert _close(a["max_linearity_pct"], 0.89, tol=0.02)
    # ISF por tramo: primero 208 mV/mil, error 4 %
    assert _close(a["isf_mv_per_x"][0], 208.0, tol=1e-6)
    assert _close(a["max_isf_err_pct"], 4.0, tol=1e-6)
    # cumple API 670
    assert a["pass_isf"] and a["pass_dsl"] and a["pass_range"]
    assert a["verdict"] == "PASA"


def test_proximity_isf_fail():
    # Curva con un tramo fuera de ±5 % (salto grande) → FALLA ISF.
    x = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    y = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0]  # último tramo 400 mV/mil
    a = analyze_proximity_linearity(x, y, x_unit="mil")
    assert not a["pass_isf"]
    assert a["verdict"] == "FALLA"


def test_proximity_range_fail():
    # Rango 40 mil < 80 mil → FALLA rango.
    x = [10, 20, 30, 40, 50]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    a = analyze_proximity_linearity(x, y, x_unit="mil")
    assert not a["pass_range"]
    assert a["verdict"] == "FALLA"


def test_proximity_units_um():
    # Recta perfecta a la pendiente nominal API 670 (7.874015748 mV/µm) →
    # DSL e ISF exactamente 0.
    nominal_v_per_um = 7.874015748 / 1000.0
    x = [250 * i for i in range(1, 10)]
    y = [nominal_v_per_um * xi + 0.1 for xi in x]
    a = analyze_proximity_linearity(x, y, x_unit="um")
    assert a["x_unit"] == "um"
    assert _close(a["max_dsl_x"], 0.0, tol=1e-6)
    assert _close(a["max_isf_err_pct"], 0.0, tol=1e-6)
    assert a["pass"]


# ---------------------------------------------------------------------------
# Amplitud
# ---------------------------------------------------------------------------
def test_amplitude_perfect_line():
    x = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
    y = [100 * xi for xi in x]   # exactamente 100 mV/g
    a = analyze_amplitude_linearity(x, y, nominal_sensitivity=100, tol_pct=1.0)
    assert _close(a["sensitivity"], 100.0, tol=1e-6)
    assert _close(a["max_dev_pct_fs"], 0.0, tol=1e-6)
    assert _close(a["sens_err_pct"], 0.0, tol=1e-6)
    assert a["pass"]


def test_amplitude_dev_fail():
    x = [1, 2, 5, 10, 20, 50]
    y = [100, 200, 500, 1000, 2000, 4000]   # último punto muy bajo (no lineal)
    a = analyze_amplitude_linearity(x, y, nominal_sensitivity=100, tol_pct=1.0)
    assert not a["pass"]


# ---------------------------------------------------------------------------
# Respuesta en frecuencia
# ---------------------------------------------------------------------------
def test_frequency_flat_pass():
    x = [10, 50, 100, 500, 1000, 5000, 10000]
    y = [100.0] * len(x)
    a = analyze_frequency_response(x, y, ref_freq_hz=100, tol_db=3.0)
    assert _close(a["max_dev_db"], 0.0, tol=1e-9)
    assert a["pass"]


def test_frequency_minus_6db():
    # Sensibilidad a la mitad → 20·log10(0.5) = -6.0206 dB.
    x = [100, 1000]
    y = [100.0, 50.0]
    a = analyze_frequency_response(x, y, ref_freq_hz=100, tol_db=3.0)
    assert _close(a["dev_db"][1], -6.0206, tol=1e-3)
    assert not a["pass"]      # 6.02 dB > 3 dB


def test_frequency_out_of_band_ignored():
    # Punto fuera de banda con caída fuerte no debe reprobar.
    x = [100, 1000, 20000]     # 20 kHz fuera de (10, 10000)
    y = [100.0, 100.0, 10.0]
    a = analyze_frequency_response(x, y, ref_freq_hz=100, tol_db=3.0,
                                   band_hz=(10.0, 10000.0))
    assert a["pass"]


# ---------------------------------------------------------------------------
# Sensibilidad de eje principal
# ---------------------------------------------------------------------------
def test_principal_sensitivity():
    assert check_principal_sensitivity(102.0, 100.0, 5.0)["pass"]      # +2 %
    assert not check_principal_sensitivity(110.0, 100.0, 5.0)["pass"]  # +10 %


# ---------------------------------------------------------------------------
# Errores esperados
# ---------------------------------------------------------------------------
def test_errors():
    for fn, args in (
        (analyze_proximity_linearity, ([10], [1.0])),
        (best_fit_line, ([1, 1, 1], [1, 2, 3])),
    ):
        try:
            fn(*args)
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__} debió lanzar ValueError")


# ---------------------------------------------------------------------------
# Runner standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
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
