"""
core.calibration.engine
========================

Motor de cálculo de calibración / curvas de linealidad de sensores de
vibración. Matemática PURA (solo numpy) — sin Streamlit, sin I/O, sin estado
global. Testeable y reutilizable.

Cubre los tres ensayos de API 670 (5.ª ed.) + manuales de fabricante:

  1. Linealidad estática de PROXIMIDAD (gap → voltaje):
     - Best-fit (mínimos cuadrados) → sensibilidad promedio (ASF).
     - Incremental Scale Factor (ISF): sensibilidad de cada tramo vs nominal
       7.87 mV/µm (200 mV/mil). API 670 exige ISF dentro de ±5 %.
     - Deviation from Straight Line (DSL): error de gap (mil/µm) respecto a la
       recta best-fit a pendiente nominal. API 670 exige DSL dentro de ±1 mil
       (±25.4 µm) en banco.
     - Linealidad % (estilo fabricante/Bently): |V − V_bestfit| / V_bestfit.
     - Rango lineal mínimo: 2 mm (80 mil).

  2. Linealidad de AMPLITUD (acelerómetro / velomitor, shaker):
     nivel de excitación → salida. Best-fit, sensibilidad, desviación %FS.
     API 670 (acelerómetro): 1 % de 0.1 g a 50 g pico.

  3. Respuesta en FRECUENCIA (acelerómetro / velomitor):
     frecuencia → sensibilidad. Desviación en dB respecto a la referencia.
     API 670 (acelerómetro): ±3 dB de 10 Hz a 10 kHz.

Convención de unidades: el motor trabaja en las unidades nativas de los datos
(x = gap en mil ó µm; y = salida en V). El scale factor nominal se pasa en
**mV por unidad-x** (mV/mil ó mV/µm) para comparar directo contra API 670.

Referencia normativa: API STANDARD 670, 5th ed., Tabla 1 (Machinery Protection
System Accuracy Requirements) y Figura 4 (curvas ISF/DSL del sistema de
proximidad). **No modificar los criterios sin re-validar contra la norma.**
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =========================================================
# Best-fit lineal (mínimos cuadrados)
# =========================================================
def best_fit_line(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float]:
    """Recta best-fit y = m·x + b por mínimos cuadrados.

    Devuelve (slope, intercept, r2). r2 = coeficiente de determinación.
    Requiere al menos 2 puntos con x no todos iguales.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size < 2 or ya.size < 2 or xa.size != ya.size:
        raise ValueError("Se requieren al menos 2 puntos (x, y) del mismo tamaño.")
    if np.ptp(xa) < 1e-12:
        raise ValueError("Todos los valores de x son iguales: no hay recta.")
    slope, intercept = np.polyfit(xa, ya, 1)
    y_hat = slope * xa + intercept
    ss_res = float(np.sum((ya - y_hat) ** 2))
    ss_tot = float(np.sum((ya - np.mean(ya)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
    return float(slope), float(intercept), float(r2)


def fixed_slope_intercept(x: Sequence[float], y: Sequence[float],
                          slope: float) -> float:
    """Mejor intercepto b para una recta de pendiente FIJA (best-fit con
    pendiente impuesta): b = mean(y) − slope·mean(x)."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    return float(np.mean(ya) - float(slope) * np.mean(xa))


# =========================================================
# Incremental Scale Factor (ISF) — API 670
# =========================================================
def incremental_scale_factors(
    x: Sequence[float], y: Sequence[float]
) -> Tuple[List[float], List[float]]:
    """Sensibilidad de cada tramo (ISF) entre puntos consecutivos.

    Devuelve (x_mid, isf) donde isf[i] = (y[i+1]−y[i]) / (x[i+1]−x[i]) en
    unidades y/x, y x_mid[i] es el punto medio del tramo. API 670 mide el ISF
    a incrementos especificados (típ. 250 µm / 10 mil) a lo largo del rango.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    x_mid: List[float] = []
    isf: List[float] = []
    for i in range(len(xa) - 1):
        dx = xa[i + 1] - xa[i]
        if abs(dx) < 1e-12:
            continue
        isf.append(float((ya[i + 1] - ya[i]) / dx))
        x_mid.append(float((xa[i + 1] + xa[i]) / 2.0))
    return x_mid, isf


# =========================================================
# Análisis de linealidad de PROXIMIDAD (API 670)
# =========================================================
# Criterios API 670 5th ed. (banco, 0–45 °C) para sistema de proximidad.
API670_PROX_NOMINAL_MV_PER_MIL = 200.0        # 7.87 mV/µm
API670_PROX_NOMINAL_MV_PER_UM = 7.874015748   # 200 mV/mil ÷ 25.4
API670_PROX_ISF_TOL_PCT = 5.0                 # ±5 % del nominal
API670_PROX_DSL_TOL_MIL = 1.0                 # ±1 mil del best-fit
API670_PROX_DSL_TOL_UM = 25.4                 # ±25.4 µm
API670_PROX_MIN_RANGE_MIL = 80.0              # 2 mm
API670_PROX_MIN_RANGE_UM = 2000.0

MIL_TO_UM = 25.4


def analyze_proximity_linearity(
    displacement: Sequence[float],
    output_v: Sequence[float],
    *,
    x_unit: str = "mil",
    nominal_mv_per_x: Optional[float] = None,
    isf_tol_pct: float = API670_PROX_ISF_TOL_PCT,
    dsl_tol_x: Optional[float] = None,
    min_range_x: Optional[float] = None,
    starting_point_is_reference: bool = True,
) -> Dict[str, Any]:
    """Analiza una curva de linealidad estática de un lazo de proximidad.

    Parámetros
    ----------
    displacement : gap en `x_unit` (mil ó µm). Suele ir 10,20,...,90.
    output_v     : salida del oscilador-demodulador en Voltios (magnitud, no
                   necesariamente negativa; se usa |ΔV|).
    x_unit       : "mil" ó "um".
    nominal_mv_per_x : sensibilidad nominal en mV por unidad-x. Si None usa el
                   nominal API 670 según la unidad (200 mV/mil ó 7.874 mV/µm).
    isf_tol_pct  : tolerancia del ISF vs nominal (API 670: ±5 %).
    dsl_tol_x    : tolerancia DSL en unidad-x (API 670: ±1 mil ó ±25.4 µm).
    min_range_x  : rango lineal mínimo requerido (API 670: 80 mil ó 2000 µm).
    starting_point_is_reference : si True (convención del calibrador portátil
                   Bently/GE), el primer punto es el ancla de referencia y se
                   excluye del máximo de linealidad % (cerca del origen la
                   razón |V−Vfit|/Vfit se dispara y no es representativa).

    Devuelve un dict con best-fit, ASF, ISF (por tramo, mV/x y error %),
    DSL (por punto, en unidad-x), linealidad %, rango, y banderas de
    cumplimiento API 670.
    """
    xa = np.asarray(displacement, dtype=float)
    ya = np.asarray(output_v, dtype=float)
    if xa.size < 2:
        raise ValueError("Se requieren al menos 2 puntos de la curva.")

    unit = "um" if str(x_unit).lower() in ("um", "µm", "micron", "micrones") else "mil"
    if nominal_mv_per_x is None:
        nominal_mv_per_x = (API670_PROX_NOMINAL_MV_PER_UM if unit == "um"
                            else API670_PROX_NOMINAL_MV_PER_MIL)
    if dsl_tol_x is None:
        dsl_tol_x = API670_PROX_DSL_TOL_UM if unit == "um" else API670_PROX_DSL_TOL_MIL
    if min_range_x is None:
        min_range_x = API670_PROX_MIN_RANGE_UM if unit == "um" else API670_PROX_MIN_RANGE_MIL

    # --- Best-fit libre → sensibilidad promedio (ASF) ---
    slope, intercept, r2 = best_fit_line(xa, ya)  # V/x, V
    asf_mv_per_x = slope * 1000.0                  # mV/x
    y_fit = slope * xa + intercept

    # --- ISF por tramo (mV/x) y error % vs nominal ---
    x_mid, isf_v = incremental_scale_factors(xa, ya)
    isf_mv = [v * 1000.0 for v in isf_v]
    isf_err_pct = [(m - nominal_mv_per_x) / nominal_mv_per_x * 100.0 for m in isf_mv]
    max_isf_err_pct = max((abs(e) for e in isf_err_pct), default=0.0)

    # --- DSL: recta de referencia a pendiente NOMINAL, best intercept ---
    slope_ref = nominal_mv_per_x / 1000.0          # V/x
    b_ref = fixed_slope_intercept(xa, ya, slope_ref)
    # error de gap (unidad-x) = (V_medido − V_recta_ref) / pendiente_ref
    dsl_x = [float((ya[i] - (slope_ref * xa[i] + b_ref)) / slope_ref)
             for i in range(len(xa))]
    max_dsl_x = max((abs(d) for d in dsl_x), default=0.0)

    # --- Linealidad % estilo fabricante: |V − V_bestfit| / V_bestfit ---
    lin_pct = []
    for i in range(len(xa)):
        denom = y_fit[i] if abs(y_fit[i]) > 1e-12 else np.nan
        lin_pct.append(float(abs(ya[i] - y_fit[i]) / denom * 100.0)
                       if denom == denom else 0.0)
    # El primer punto es el ancla de referencia (convención del calibrador
    # portátil): se excluye del máximo para no contaminar con el efecto de
    # origen. El valor por punto sí se conserva en la lista completa.
    _lin_for_max = lin_pct[1:] if (starting_point_is_reference and len(lin_pct) > 1) \
        else lin_pct
    max_lin_pct = max((v for v in _lin_for_max if v == v), default=0.0)

    # --- Rango lineal cubierto ---
    span_x = float(np.ptp(xa))

    # --- Cumplimiento API 670 ---
    pass_isf = max_isf_err_pct <= isf_tol_pct + 1e-9
    pass_dsl = max_dsl_x <= dsl_tol_x + 1e-9
    pass_range = span_x >= min_range_x - 1e-9
    passed = bool(pass_isf and pass_dsl and pass_range)

    return {
        "x_unit": unit,
        "n_points": int(xa.size),
        "slope_v_per_x": float(slope),
        "intercept_v": float(intercept),
        "r2": float(r2),
        "asf_mv_per_x": float(asf_mv_per_x),
        "nominal_mv_per_x": float(nominal_mv_per_x),
        "asf_err_pct": float((asf_mv_per_x - nominal_mv_per_x) / nominal_mv_per_x * 100.0),
        "x_mid": x_mid,
        "isf_mv_per_x": isf_mv,
        "isf_err_pct": isf_err_pct,
        "max_isf_err_pct": float(max_isf_err_pct),
        "dsl_x": dsl_x,
        "max_dsl_x": float(max_dsl_x),
        "dsl_ref_intercept_v": float(b_ref),
        "linearity_pct": lin_pct,
        "max_linearity_pct": float(max_lin_pct),
        "starting_point_is_reference": bool(starting_point_is_reference),
        "span_x": span_x,
        "min_range_x": float(min_range_x),
        "isf_tol_pct": float(isf_tol_pct),
        "dsl_tol_x": float(dsl_tol_x),
        "pass_isf": bool(pass_isf),
        "pass_dsl": bool(pass_dsl),
        "pass_range": bool(pass_range),
        "pass": passed,
        "verdict": "PASA" if passed else "FALLA",
        "x": [float(v) for v in xa],
        "y": [float(v) for v in ya],
        "y_fit": [float(v) for v in y_fit],
    }


# =========================================================
# Linealidad de AMPLITUD (acelerómetro / velomitor, shaker)
# =========================================================
def analyze_amplitude_linearity(
    level: Sequence[float],
    output: Sequence[float],
    *,
    nominal_sensitivity: Optional[float] = None,
    tol_pct: float = 1.0,
    level_unit: str = "g pk",
    output_unit: str = "mV",
) -> Dict[str, Any]:
    """Linealidad de amplitud: nivel de excitación → salida (shaker).

    Best-fit → sensibilidad (pendiente). Desviación de cada punto respecto al
    best-fit, expresada como % del fondo de escala (%FS). API 670 acelerómetro:
    1 % de 0.1 g a 50 g pico.

    nominal_sensitivity: en output_unit/level_unit (ej. 100 mV/g). Si None, se
    usa la sensibilidad best-fit como referencia (auto).
    """
    xa = np.asarray(level, dtype=float)
    ya = np.asarray(output, dtype=float)
    if xa.size < 2:
        raise ValueError("Se requieren al menos 2 niveles de excitación.")

    slope, intercept, r2 = best_fit_line(xa, ya)
    sensitivity = float(slope)
    y_fit = slope * xa + intercept
    fs = float(np.max(np.abs(ya))) if ya.size else 0.0

    dev_pct_fs = []
    for i in range(len(xa)):
        dev_pct_fs.append(float((ya[i] - y_fit[i]) / fs * 100.0) if fs > 1e-12 else 0.0)
    max_dev_pct_fs = max((abs(d) for d in dev_pct_fs), default=0.0)

    ref = nominal_sensitivity if nominal_sensitivity else sensitivity
    sens_err_pct = float((sensitivity - ref) / ref * 100.0) if ref else 0.0

    passed = bool(max_dev_pct_fs <= tol_pct + 1e-9)
    return {
        "sensitivity": sensitivity,
        "sensitivity_unit": f"{output_unit}/{level_unit}",
        "nominal_sensitivity": (float(nominal_sensitivity)
                                if nominal_sensitivity else None),
        "sens_err_pct": sens_err_pct,
        "intercept": float(intercept),
        "r2": float(r2),
        "dev_pct_fs": dev_pct_fs,
        "max_dev_pct_fs": float(max_dev_pct_fs),
        "tol_pct": float(tol_pct),
        "fs": fs,
        "level_unit": level_unit,
        "output_unit": output_unit,
        "pass": passed,
        "verdict": "PASA" if passed else "FALLA",
        "x": [float(v) for v in xa],
        "y": [float(v) for v in ya],
        "y_fit": [float(v) for v in y_fit],
    }


# =========================================================
# Respuesta en FRECUENCIA (acelerómetro / velomitor)
# =========================================================
def analyze_frequency_response(
    freq_hz: Sequence[float],
    sensitivity: Sequence[float],
    *,
    ref_freq_hz: float = 100.0,
    tol_db: float = 3.0,
    band_hz: Optional[Tuple[float, float]] = (10.0, 10000.0),
    sens_unit: str = "mV/g",
) -> Dict[str, Any]:
    """Respuesta en frecuencia: desviación en dB respecto a la referencia.

    dev_dB(f) = 20·log10( S(f) / S_ref ). S_ref = sensibilidad a `ref_freq_hz`
    (o la más cercana). API 670 acelerómetro: ±3 dB de 10 Hz a 10 kHz,
    referido a la sensibilidad medida del eje principal.

    band_hz: banda (fmin, fmax) donde se aplica el criterio; puntos fuera de la
    banda se reportan pero no marcan FALLA.
    """
    fa = np.asarray(freq_hz, dtype=float)
    sa = np.asarray(sensitivity, dtype=float)
    if fa.size < 2:
        raise ValueError("Se requieren al menos 2 puntos de frecuencia.")

    idx_ref = int(np.argmin(np.abs(fa - float(ref_freq_hz))))
    s_ref = float(sa[idx_ref])
    if abs(s_ref) < 1e-12:
        raise ValueError("Sensibilidad de referencia nula.")

    dev_db = [float(20.0 * np.log10(abs(s) / abs(s_ref))) if abs(s) > 1e-12
              else float("-inf") for s in sa]

    in_band = []
    for i in range(len(fa)):
        if band_hz is None:
            in_band.append(True)
        else:
            in_band.append(bool(band_hz[0] - 1e-9 <= fa[i] <= band_hz[1] + 1e-9))

    max_dev_db = max((abs(dev_db[i]) for i in range(len(fa))
                      if in_band[i] and dev_db[i] != float("-inf")), default=0.0)
    passed = bool(max_dev_db <= tol_db + 1e-9)

    return {
        "ref_freq_hz": float(fa[idx_ref]),
        "ref_sensitivity": s_ref,
        "sens_unit": sens_unit,
        "dev_db": dev_db,
        "in_band": in_band,
        "max_dev_db": float(max_dev_db),
        "tol_db": float(tol_db),
        "band_hz": band_hz,
        "pass": passed,
        "verdict": "PASA" if passed else "FALLA",
        "x": [float(v) for v in fa],
        "y": [float(v) for v in sa],
    }


# =========================================================
# Sensibilidad de eje principal (punto único, API 670)
# =========================================================
def check_principal_sensitivity(
    measured: float, nominal: float, tol_pct: float = 5.0
) -> Dict[str, Any]:
    """Compara sensibilidad de eje principal contra nominal ± tol %.

    API 670 acelerómetro (banco): 100 mV/g ± 5 %.
    """
    err = (float(measured) - float(nominal)) / float(nominal) * 100.0 if nominal else 0.0
    passed = bool(abs(err) <= tol_pct + 1e-9)
    return {
        "measured": float(measured),
        "nominal": float(nominal),
        "err_pct": float(err),
        "tol_pct": float(tol_pct),
        "pass": passed,
        "verdict": "PASA" if passed else "FALLA",
    }


__all__ = [
    "best_fit_line", "fixed_slope_intercept", "incremental_scale_factors",
    "analyze_proximity_linearity", "analyze_amplitude_linearity",
    "analyze_frequency_response", "check_principal_sensitivity",
    "MIL_TO_UM",
    "API670_PROX_NOMINAL_MV_PER_MIL", "API670_PROX_NOMINAL_MV_PER_UM",
    "API670_PROX_ISF_TOL_PCT", "API670_PROX_DSL_TOL_MIL",
    "API670_PROX_DSL_TOL_UM", "API670_PROX_MIN_RANGE_MIL",
    "API670_PROX_MIN_RANGE_UM",
]
