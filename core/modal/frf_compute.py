"""
core/modal/frf_compute.py — Cálculo de FRF (Frequency Response Function)
=========================================================================

Computa funciones de respuesta en frecuencia (FRF) a partir de señales
temporales sincronizadas de input (martillo / shaker) y output (sensor).

Estimadores soportados
----------------------
H1 — Asume ruido solo en output: H1 = Sxy / Sxx
  Mejor cuando hay ruido en la respuesta (uso típico EMA)

H2 — Asume ruido solo en input: H2 = Syy / Syx
  Mejor cuando hay ruido en la excitación

Hv — Promedio robusto entre H1 y H2 (uso avanzado, no incluido en V1)

Coherencia
----------
γ²(f) = |Sxy(f)|² / (Sxx(f) · Syy(f))

Indica qué tanto del output es causado por el input. Valores < 0.8 en una
banda indican mal acoplamiento o ruido excesivo — esos modos no son
confiables.

Norma aplicable
---------------
ISO 7626-2 §6.2 — Cálculo de movilidad por estimadores H1/H2
ISO 7626-5 §7.3 — Promediado de FRFs en ensayos con martillo
ISO 7626-5 §7.4 — Validación con coherencia (mínimo 0.8 en banda de interés)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class FRFResult:
    """Resultado del cálculo FRF."""
    frequencies_hz: np.ndarray
    frf_complex: np.ndarray       # Compleja: real + j·imag
    coherence: np.ndarray         # γ² en [0, 1]
    estimator: str                # "H1" o "H2"
    n_averages: int               # Número de impactos promediados
    window: str                   # "rectangular", "hanning", "force_expo"

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.frf_complex)

    @property
    def magnitude_db(self) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(self.magnitude, 1e-30))

    @property
    def phase_deg(self) -> np.ndarray:
        return np.degrees(np.angle(self.frf_complex))

    def is_band_reliable(self, f_low: float, f_high: float, threshold: float = 0.8) -> bool:
        """Verifica si la coherencia es ≥ threshold en una banda de frecuencia."""
        mask = (self.frequencies_hz >= f_low) & (self.frequencies_hz <= f_high)
        if not np.any(mask):
            return False
        return bool(np.min(self.coherence[mask]) >= threshold)


def compute_frf_h1(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 1024,
    noverlap: Optional[int] = None,
    window: str = "hanning",
) -> FRFResult:
    """
    Calcula FRF usando estimador H1 = Sxy / Sxx.

    Args:
        input_signal: Vector temporal del input (e.g. fuerza del martillo)
        output_signal: Vector temporal de la respuesta (e.g. acelerómetro)
        sample_rate_hz: Frecuencia de muestreo (Hz)
        nperseg: Tamaño de segmento para Welch (default 1024)
        noverlap: Solape entre segmentos (default nperseg/2)
        window: Función de ventana ("hanning", "hamming", "rectangular")

    Returns:
        FRFResult con magnitude, phase y coherence
    """
    try:
        from scipy.signal import csd, welch, coherence  # noqa
    except ImportError:
        raise ImportError("scipy es requerido para cálculo FRF")

    # TODO: implementar usando scipy.signal:
    #   freq, Sxx = welch(input, fs, nperseg, noverlap, window)
    #   freq, Sxy = csd(input, output, fs, nperseg, noverlap, window)
    #   freq, gamma2 = coherence(input, output, fs, nperseg, noverlap, window)
    #   H1 = Sxy / Sxx
    raise NotImplementedError("Fase scaffolding — implementación próximo sprint")


def compute_frf_h2(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 1024,
    noverlap: Optional[int] = None,
    window: str = "hanning",
) -> FRFResult:
    """
    Calcula FRF usando estimador H2 = Syy / Syx.

    Útil cuando se sospecha ruido alto en la excitación.
    """
    raise NotImplementedError("Fase scaffolding")


def average_frf(frfs: list) -> FRFResult:
    """
    Promedia múltiples FRFs (típico en ensayos con martillo, 5-10 impactos).

    Asume que todas tienen el mismo eje de frecuencia.
    """
    raise NotImplementedError("Fase scaffolding")
