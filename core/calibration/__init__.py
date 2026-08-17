"""
core.calibration — Módulo de calibración / curvas de linealidad de sensores.

Motor puro (engine), criterios y defaults (standards), curvas (curve) y reporte
PDF (report). Basado en API 670 5.ª ed. + manuales de fabricante.
"""
from __future__ import annotations

from core.calibration.engine import (
    best_fit_line, fixed_slope_intercept, incremental_scale_factors,
    analyze_proximity_linearity, analyze_amplitude_linearity,
    analyze_frequency_response, check_principal_sensitivity, MIL_TO_UM,
)
from core.calibration.standards import (
    SENSOR_TYPES, SENSOR_TYPE_LABELS, MANUFACTURERS, get_default_spec,
    prox_default_grid, accel_default_levels, freq_default_points,
)

__all__ = [
    "best_fit_line", "fixed_slope_intercept", "incremental_scale_factors",
    "analyze_proximity_linearity", "analyze_amplitude_linearity",
    "analyze_frequency_response", "check_principal_sensitivity", "MIL_TO_UM",
    "SENSOR_TYPES", "SENSOR_TYPE_LABELS", "MANUFACTURERS", "get_default_spec",
    "prox_default_grid", "accel_default_levels", "freq_default_points",
]
