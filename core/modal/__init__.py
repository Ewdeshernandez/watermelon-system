"""
core/modal — Modal Analysis Module
===================================

Módulo de análisis modal experimental (EMA) y operacional (OMA) para
Watermelon System. Sustituye dependencia del software externo Artemis
Modal con stack open-source nativo (pyEMA + PyOMA2 + scipy).

Marco normativo aplicable
-------------------------
EMA — Experimental Modal Analysis:
  · ISO 7626-1: Definiciones y fundamentos de movilidad mecánica
  · ISO 7626-2: Excitación translacional con shaker
  · ISO 7626-3: Excitación rotacional
  · ISO 7626-4: Matriz completa de movilidad (FRF)
  · ISO 7626-5: Excitación con martillo modal (impact hammer) ← principal
  · ISO 7626-6: Formatos de datos e identificación modal

OMA — Operational Modal Analysis:
  · ISO 20816: Evaluación de vibraciones en máquinas en operación
  · API 684: Rotor dynamics, validación de modelos, velocidades críticas

FEA — Finite Element Analysis:
  · Sin norma única para iteración de modelo
  · Correlación EMA/OMA ↔ FEA vía MAC (Modal Assurance Criterion)
  · API 618 §7.9.4.2.5.3.2 — criterio separación modal (≥10% real, ≥20% diseño)

Hardware estandarizado
----------------------
· Maleta de adquisición Watermelon — multi-canal, 24-bit, hasta 51.2 kHz, IEPE built-in
· Acelerómetros IEPE estándar — 100 mV/g
· Probetas de proximidad estándar — 200 mV/mil (requiere PS -24 VDC)
· Martillo modal

Estructura del módulo
---------------------
geometry_3d       — Wireframe 3D (nodes + edges + faces) en JSON nativo
sensor_3d_mapping — Mapea sensor_map.plane_label → nodo geometría + DOF
tdms_importer     — Lee archivos .tdms nativos del hardware de captura (npTDMS)
artemis_importer  — Lee exports legacy de Artemis Modal (.txt) — compat
acq_backend       — Adquisición live del hardware (EMA triggered + OMA continuous)
signal_scaling    — Aplica sensitivities (mV/g, mV/mil) → engineering units
frf_compute       — Cálculo FRF: H1, H2 estimators + coherencia (scipy.signal)
ema_engine        — Curve fitting LSCF + stability diagram (pyEMA wrapper)
oma_engine        — FDD + SSI-COV + SSI-DATA (PyOMA2 wrapper)
modal_animator    — Animación 3D mode shapes (Plotly Mesh3d frames)
modal_report      — Generador PDF integrado al sistema Reports
modal_history     — Persistencia de runs modales en Supabase Storage
"""

__version__ = "0.1.0-scaffold"
__all__ = [
    "geometry_3d",
    "sensor_3d_mapping",
    "tdms_importer",
    "artemis_importer",
    "acq_backend",
    "signal_scaling",
    "frf_compute",
    "ema_engine",
    "oma_engine",
    "modal_animator",
    "modal_report",
    "modal_history",
]
