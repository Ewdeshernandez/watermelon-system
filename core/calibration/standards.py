"""
core.calibration.standards
===========================

Criterios normativos (API 670 5.ª ed.) y valores nominales por fabricante para
los tres tipos de sensor: proximidad, acelerómetro y velomitor.

Diseño: los CRITERIOS DE ACEPTACIÓN provienen de API 670 (columna vertebral,
no se tocan sin re-validar la norma). Los VALORES NOMINALES por fabricante
(sensibilidad, unidades típicas, grilla de puntos sugerida) son *defaults*
editables — el técnico siempre puede ajustar según la hoja de datos del modelo
exacto. Nunca se afirma un número de fabricante como inamovible.

Fuente normativa: API STANDARD 670, 5th ed., Tabla 1 y Figura 4.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Criterios API 670 (banco, rango de prueba 0–45 °C). Re-exportados del motor
# para tener un único punto de verdad.
from core.calibration.engine import (
    API670_PROX_NOMINAL_MV_PER_MIL, API670_PROX_NOMINAL_MV_PER_UM,
    API670_PROX_ISF_TOL_PCT, API670_PROX_DSL_TOL_MIL, API670_PROX_DSL_TOL_UM,
    API670_PROX_MIN_RANGE_MIL, API670_PROX_MIN_RANGE_UM,
)

# API 670 — acelerómetro
API670_ACCEL_NOMINAL_MV_PER_G = 100.0     # sensibilidad eje principal
API670_ACCEL_SENS_TOL_PCT = 5.0           # ±5 % (banco); ±20 % (operación)
API670_ACCEL_AMPL_LIN_TOL_PCT = 1.0       # 1 % de 0.1 g a 50 g pico
API670_ACCEL_AMPL_RANGE_G = (0.1, 50.0)
API670_ACCEL_FREQ_TOL_DB = 3.0            # ±3 dB
API670_ACCEL_FREQ_BAND_HZ = (10.0, 10000.0)
API670_ACCEL_FREQ_REF_HZ = 100.0

SENSOR_TYPES = ["proximity", "accelerometer", "velomitor"]
SENSOR_TYPE_LABELS = {
    "proximity": "Lazo de proximidad (eddy current)",
    "accelerometer": "Acelerómetro",
    "velomitor": "Velomitor (velocidad sísmica)",
}

MANUFACTURERS = [
    "Bently Nevada", "Emerson", "SKF", "Metrix", "PCB Piezotronics",
    "Wilcoxon", "Otro / genérico",
]


def prox_default_grid(x_unit: str = "mil") -> List[float]:
    """Grilla sugerida de gap para la curva de proximidad (10 mil / 250 µm
    de incremento, típico API 670). 10..90 mil ó 250..2250 µm."""
    if str(x_unit).lower() in ("um", "µm"):
        return [250.0 * i for i in range(1, 10)]      # 250..2250 µm
    return [10.0 * i for i in range(1, 10)]           # 10..90 mil


def accel_default_levels() -> List[float]:
    """Niveles de excitación sugeridos para linealidad de amplitud (g pk)."""
    return [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def freq_default_points() -> List[float]:
    """Frecuencias sugeridas para respuesta en frecuencia (Hz)."""
    return [10, 30, 50, 100, 200, 500, 1000, 2000, 5000, 8000, 10000]


def get_default_spec(sensor_type: str, manufacturer: str = "Bently Nevada",
                     x_unit: str = "mil") -> Dict[str, Any]:
    """Devuelve el spec por defecto (nominal + tolerancias + unidades + grilla)
    para un tipo de sensor y fabricante. Todo es editable en la UI.

    proximity → nominal 200 mV/mil (7.874 mV/µm), ISF ±5 %, DSL ±1 mil,
                rango ≥ 80 mil. AISI 4140 como blanco de referencia.
    accelerometer → 100 mV/g ± 5 %, linealidad amplitud 1 % (0.1–50 g),
                respuesta ±3 dB (10 Hz–10 kHz).
    velomitor → sensibilidad dependiente del modelo (default ≈ 3.94 mV/(mm/s),
                equiv. 100 mV/(in/s) tipo Bently Velomitor); linealidad y
                respuesta con el mismo esquema, banda editable.
    """
    st = str(sensor_type).lower()
    unit = "um" if str(x_unit).lower() in ("um", "µm") else "mil"

    if st == "proximity":
        return {
            "sensor_type": "proximity",
            "manufacturer": manufacturer,
            "x_unit": unit,
            "y_unit": "V",
            "nominal_mv_per_x": (API670_PROX_NOMINAL_MV_PER_UM if unit == "um"
                                 else API670_PROX_NOMINAL_MV_PER_MIL),
            "nominal_label": ("7.874 mV/µm" if unit == "um" else "200 mV/mil"),
            "isf_tol_pct": API670_PROX_ISF_TOL_PCT,
            "dsl_tol_x": (API670_PROX_DSL_TOL_UM if unit == "um"
                          else API670_PROX_DSL_TOL_MIL),
            "dsl_tol_label": ("±25.4 µm" if unit == "um" else "±1 mil"),
            "min_range_x": (API670_PROX_MIN_RANGE_UM if unit == "um"
                            else API670_PROX_MIN_RANGE_MIL),
            "min_range_label": ("2 mm" if unit == "um" else "80 mil"),
            "grid": prox_default_grid(unit),
            "target_material": "AISI 4140 (blanco de referencia API 670)",
            "norm": "API 670 5th ed. · Tabla 1 / Fig. 4",
        }

    if st == "accelerometer":
        return {
            "sensor_type": "accelerometer",
            "manufacturer": manufacturer,
            "nominal_sensitivity": API670_ACCEL_NOMINAL_MV_PER_G,
            "sensitivity_unit": "mV/g",
            "sens_tol_pct": API670_ACCEL_SENS_TOL_PCT,
            "ampl_tol_pct": API670_ACCEL_AMPL_LIN_TOL_PCT,
            "ampl_range_g": API670_ACCEL_AMPL_RANGE_G,
            "levels": accel_default_levels(),
            "level_unit": "g pk",
            "output_unit": "mV",
            "freq_tol_db": API670_ACCEL_FREQ_TOL_DB,
            "freq_band_hz": API670_ACCEL_FREQ_BAND_HZ,
            "freq_ref_hz": API670_ACCEL_FREQ_REF_HZ,
            "freq_points": freq_default_points(),
            "norm": "API 670 5th ed. · Tabla 1",
        }

    # velomitor
    return {
        "sensor_type": "velomitor",
        "manufacturer": manufacturer,
        # Sensibilidad dependiente del modelo — default tipo Bently Velomitor.
        "nominal_sensitivity": 3.94,
        "sensitivity_unit": "mV/(mm/s)",
        "nominal_note": "≈ 100 mV/(in/s) · verificar hoja de datos del modelo",
        "sens_tol_pct": 5.0,
        "ampl_tol_pct": 1.0,
        "levels": [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        "level_unit": "mm/s pk",
        "output_unit": "mV",
        "freq_tol_db": 3.0,
        # Velomitores típicos: banda útil más angosta que un acelerómetro.
        "freq_band_hz": (10.0, 1000.0),
        "freq_ref_hz": 100.0,
        "freq_points": [4.5, 10, 20, 50, 100, 200, 500, 1000, 2000],
        "norm": "API 670 (marco) + manual del fabricante",
    }


__all__ = [
    "SENSOR_TYPES", "SENSOR_TYPE_LABELS", "MANUFACTURERS",
    "get_default_spec", "prox_default_grid", "accel_default_levels",
    "freq_default_points",
    "API670_ACCEL_NOMINAL_MV_PER_G", "API670_ACCEL_SENS_TOL_PCT",
    "API670_ACCEL_AMPL_LIN_TOL_PCT", "API670_ACCEL_AMPL_RANGE_G",
    "API670_ACCEL_FREQ_TOL_DB", "API670_ACCEL_FREQ_BAND_HZ",
    "API670_ACCEL_FREQ_REF_HZ",
]
