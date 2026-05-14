"""
core/modal/oma_engine.py — Motor de Análisis Modal Operacional (OMA)
=====================================================================

Identifica modos naturales SIN necesidad de excitación controlada (martillo),
usando solo las señales de respuesta durante operación normal de la máquina.

Casos de uso
------------
· Máquinas que NO se pueden detener para hacer EMA (turbinas en producción)
· Estructuras grandes donde no es práctico golpear con martillo
· Validación in-situ de modos identificados previamente con EMA
· Monitoreo modal continuo (frecuencias modales como indicador de daño)

Algoritmos soportados
---------------------
FDD — Frequency Domain Decomposition
  · Aplica SVD a la matriz de PSD cruzados
  · Los modos aparecen como picos en el primer singular value
  · Rápido pero menos preciso para damping
  · Bueno como primera pasada

SSI-COV — Stochastic Subspace Identification con covarianza
  · Trabaja en el dominio del tiempo
  · Extrae modelo state-space del sistema
  · Da fn, ζ y mode shape complejos
  · Más preciso para damping que FDD

SSI-DATA — SSI directo sobre data (sin covarianza intermedia)
  · Como SSI-COV pero numéricamente más estable
  · Computacionalmente más caro
  · Recomendado para records cortos (< 60 seg)

Requerimientos de datos
-----------------------
· Duración mínima: 30 segundos (recomendado 60-300 seg)
· Velocidad constante durante todo el record
· Múltiples canales sincronizados (mínimo 4 para buena cobertura espacial)
· Sample rate: 5-10 kHz típico (Nyquist 2-5 kHz cubre la mayoría de modos)

⚠ Caveat — Modos forzados (running speed, 2×, etc) aparecen también como
   "modos" en OMA pero NO son modos naturales — son excitaciones armónicas
   de la operación. Hay que filtrarlos manualmente o usar harmonic detection.

Dependencias
------------
PyOMA2 — Open-source, LGPL
  pip install pyOMA2

Norma aplicable
---------------
ISO 20816 — Evaluación de vibraciones en máquinas en operación. Define
los niveles de vibración aceptables que sirven de baseline para los datos
operacionales usados como input al OMA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class OMAMode:
    """Modo identificado por OMA."""
    natural_frequency_hz: float
    damping_ratio_pct: float
    mode_shape: np.ndarray  # vector complejo
    stability_score: float  # 0-1, qué tan estable es a través de orders
    is_harmonic: bool = False  # True si parece ser modo forzado (no natural)


@dataclass
class OMAResult:
    """Resultado del análisis OMA."""
    modes: List[OMAMode]
    algorithm: str  # "FDD", "SSI-COV", "SSI-DATA"
    record_duration_s: float
    sample_rate_hz: float
    n_channels: int


def run_fdd(
    time_data: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 4096,
) -> OMAResult:
    """
    Frequency Domain Decomposition.

    Args:
        time_data: Matriz (N_samples, N_channels) con señales temporales
        sample_rate_hz: Frecuencia de muestreo
        nperseg: Tamaño del segmento para PSD (típico 2048-8192)

    Returns:
        OMAResult con modos detectados como picos del primer singular value
    """
    try:
        import pyOMA2  # noqa
    except ImportError:
        raise ImportError("pyOMA2 no instalado. Ejecuta: pip install pyOMA2")

    # TODO: implementar wrapper FDD
    raise NotImplementedError("Fase scaffolding — implementación próximo sprint")


def run_ssi_cov(
    time_data: np.ndarray,
    sample_rate_hz: float,
    max_order: int = 50,
) -> OMAResult:
    """
    Stochastic Subspace Identification con covarianza.

    Args:
        time_data: Matriz (N_samples, N_channels)
        sample_rate_hz: Frecuencia de muestreo
        max_order: Orden máximo del modelo state-space

    Returns:
        OMAResult con modos y stability info
    """
    raise NotImplementedError("Fase scaffolding")


def run_ssi_data(
    time_data: np.ndarray,
    sample_rate_hz: float,
    max_order: int = 50,
) -> OMAResult:
    """
    SSI directo sobre data (sin covarianza intermedia).

    Más estable numéricamente, especialmente para records cortos.
    """
    raise NotImplementedError("Fase scaffolding")


def detect_harmonic_modes(
    modes: List[OMAMode],
    operating_rpm: float,
    tolerance_pct: float = 1.0,
) -> List[OMAMode]:
    """
    Marca como `is_harmonic=True` los modos cuya frecuencia coincide con
    armónicas de la velocidad de operación (1×, 2×, 3×, ...).

    Estos son excitaciones forzadas, NO modos naturales.
    """
    raise NotImplementedError("Fase scaffolding")
