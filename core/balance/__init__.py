"""
core.balance
============

Motor de balanceo de rotores (extraído de ROTORIX, validado en campo).

Balanceo por coeficiente de influencia en 1 y 2 planos, cálculo de peso de
prueba (API 684), desbalance residual permisible (ISO 21940-11), evaluación
por grados ISO (ISO 21940-12) y diagnóstico estático/couple.

Este paquete NO depende de Streamlit — es matemática pura y testeable. La UI
(páginas Watermelon) y los adaptadores de datos (manual / Live Monitoring)
importan de aquí. No modificar las fórmulas sin re-validar contra campo.
"""
from core.balance.engine import (  # noqa: F401
    # Complejos / polar
    to_complex,
    to_polar,
    norm360,
    polar_to_complex,
    complex_to_polar,
    # API 684 — peso de prueba
    umax_api684_gmm,
    recommend_trial_weight_g,
    # ISO 21940 — permisibles y grados
    calc_e_per,
    calc_U_per,
    calc_U_trial,
    calc_U_res_auto,
    pct_reduction,
    ISO_GRADES,
    evaluate_iso_grades,
    status_from_ratio,
    # Solvers
    solve_1plane,
    solve_2plane,
    # Diagnóstico
    status_level,
    diagnose_static_couple,
)

__all__ = [
    "to_complex", "to_polar", "norm360", "polar_to_complex", "complex_to_polar",
    "umax_api684_gmm", "recommend_trial_weight_g",
    "calc_e_per", "calc_U_per", "calc_U_trial", "calc_U_res_auto",
    "pct_reduction", "ISO_GRADES", "evaluate_iso_grades", "status_from_ratio",
    "solve_1plane", "solve_2plane",
    "status_level", "diagnose_static_couple",
]
