"""
core/modal/ema_engine.py — Motor de Análisis Modal Experimental (EMA)
======================================================================

Extrae parámetros modales (frecuencias naturales, damping, mode shapes)
de FRFs medidas con martillo modal usando algoritmo LSCF.

Algoritmo
---------
LSCF — Least-Squares Complex Frequency-domain
  · Ajusta un modelo de fracciones racionales a las FRFs medidas
  · Más robusto y rápido que métodos clásicos (LSCE, Rational Fraction Polynomial)
  · Implementado en pyEMA (open-source)
  · También conocido como polyMAX (versión comercial del mismo método)

Outputs por modo:
  · Frecuencia natural fn (Hz)
  · Damping ratio ζ (%)
  · Vector mode shape φ (complejo, N puntos)
  · Modal complexity (% — qué tan complejo es el modo)
  · Pole stability flag (estable / inestable a través de model orders)

Stability Diagram
-----------------
Para cada model order n = 1..N_max:
  Encuentra los polos del modelo LSCF
  Compara con el orden anterior: si fn, ζ, mode shape son similares → "stable"

Modos "estables" son los reales. Modos "inestables" son artefactos numéricos.

Dependencias
------------
pyEMA — Open-source, MIT license
  pip install pyEMA

Norma aplicable
---------------
ISO 7626-6 §6.3 — Identificación de parámetros modales por curve fitting
ISO 7626-6 §6.5 — Validación con MAC entre orders consecutivos
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ModalMode:
    """Un modo natural identificado."""
    index: int
    natural_frequency_hz: float
    damping_ratio_pct: float
    mode_shape: np.ndarray  # vector complejo de N DOFs
    complexity_pct: float
    is_stable: bool = True


@dataclass
class StabilityDiagramPoint:
    """Punto del diagrama de estabilidad."""
    model_order: int
    frequency_hz: float
    damping_pct: float
    flag: str  # "stable", "freq_stable", "unstable", "spurious"


@dataclass
class EMAResult:
    """Resultado completo del análisis EMA."""
    modes: List[ModalMode]
    stability_points: List[StabilityDiagramPoint] = field(default_factory=list)
    frequency_band: tuple = (0.0, 1000.0)
    n_inputs: int = 1
    n_outputs: int = 0
    algorithm: str = "LSCF"


def run_lscf(
    frfs: np.ndarray,
    frequencies_hz: np.ndarray,
    pol_order: int = 50,
    band: Optional[tuple] = None,
) -> EMAResult:
    """
    Ejecuta el algoritmo LSCF sobre un set de FRFs.

    Args:
        frfs: Matriz (N_freq, N_outputs) — FRFs complejas
        frequencies_hz: Vector de frecuencias correspondiente
        pol_order: Orden máximo del polinomio (típico 30-80)
        band: Banda de frecuencia de interés (f_low, f_high). Si None, usa toda.

    Returns:
        EMAResult con modos identificados y stability diagram
    """
    try:
        import pyEMA  # noqa
    except ImportError:
        raise ImportError("pyEMA no instalado. Ejecuta: pip install pyEMA")

    # TODO: implementar wrapper
    raise NotImplementedError("Fase scaffolding — implementación próximo sprint")


def half_power_damping(
    magnitude: np.ndarray,
    frequencies_hz: np.ndarray,
    peak_freq_hz: float,
) -> float:
    """
    Método half-power para estimación rápida de damping de un modo aislado.

    Útil como sanity check vs el LSCF.

    Args:
        magnitude: |H(f)| de la FRF
        frequencies_hz: eje de frecuencia
        peak_freq_hz: frecuencia del pico identificado

    Returns:
        Damping ratio en %
    """
    raise NotImplementedError("Fase scaffolding")
