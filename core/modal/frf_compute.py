"""
core/modal/frf_compute.py — Cálculo de FRF (Frequency Response Function)
=========================================================================

Computa funciones de respuesta en frecuencia (FRF) a partir de señales
temporales sincronizadas de input (martillo / shaker) y output (sensor).
Además incluye detección de picos y damping por half-power method para
identificación modal rápida sin requerir pyEMA.

Estimadores soportados
----------------------
H1 — Asume ruido solo en output: H1 = Sxy / Sxx
  Mejor cuando hay ruido en la respuesta (uso típico EMA)

H2 — Asume ruido solo en input: H2 = Syy / Syx
  Mejor cuando hay ruido en la excitación

Coherencia
----------
γ²(f) = |Sxy(f)|² / (Sxx(f) · Syy(f))
Valores < 0.8 en una banda indican mal acoplamiento — modos no confiables.

Half-power damping
------------------
Para un modo aislado con factor de calidad alto:
  ζ ≈ (f₂ - f₁) / (2 × fn)
donde f₁, f₂ son las frecuencias donde la magnitud cae a |H_peak| / √2
(= -3 dB), y fn es la frecuencia del pico.

Norma aplicable
---------------
ISO 7626-2 secc. 6.2 — Cálculo de movilidad por estimadores H1/H2
ISO 7626-5 secc. 7.3 — Promediado de FRFs en ensayos con martillo
ISO 7626-5 secc. 7.4 — Validación con coherencia (mínimo 0.8 en banda de interés)
ISO 7626-6 secc. 6.3.2 — Half-power method para damping
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math
import numpy as np


@dataclass
class FRFResult:
    """Resultado del cálculo FRF."""
    frequencies_hz: np.ndarray
    frf_complex: np.ndarray       # Compleja: real + j·imag
    coherence: np.ndarray         # γ² en [0, 1]
    estimator: str                # "H1" o "H2"
    n_averages: int               # Número de segmentos promediados
    window: str                   # "hanning", "hamming", "rectangular"

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.frf_complex)

    @property
    def magnitude_db(self) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(self.magnitude, 1e-30))

    @property
    def phase_deg(self) -> np.ndarray:
        return np.degrees(np.angle(self.frf_complex))

    def is_band_reliable(self, f_low: float, f_high: float,
                         threshold: float = 0.8) -> bool:
        mask = (self.frequencies_hz >= f_low) & (self.frequencies_hz <= f_high)
        if not np.any(mask):
            return False
        return bool(np.min(self.coherence[mask]) >= threshold)


@dataclass
class ModalPeak:
    """Modo natural detectado por half-power."""
    frequency_hz: float
    damping_ratio_pct: float
    magnitude_peak: float
    half_power_f1_hz: float
    half_power_f2_hz: float
    bandwidth_hz: float
    quality_factor: float
    coherence_at_peak: Optional[float] = None
    is_reliable: bool = True


# =====================================================================
# Cálculo de FRF
# =====================================================================

def compute_frf_h1(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 1024,
    noverlap: Optional[int] = None,
    window: str = "hann",
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
        from scipy.signal import csd, welch, coherence
    except ImportError as exc:
        raise ImportError("scipy es requerido para cálculo FRF") from exc

    x = np.asarray(input_signal, dtype=float)
    y = np.asarray(output_signal, dtype=float)
    if x.size != y.size:
        raise ValueError(f"input y output deben tener mismo tamaño: {x.size} vs {y.size}")
    if x.size < nperseg:
        nperseg = max(64, x.size // 2)

    if noverlap is None:
        noverlap = nperseg // 2

    # Auto-spectrum del input
    freq, Sxx = welch(x, fs=sample_rate_hz, nperseg=nperseg,
                       noverlap=noverlap, window=window)
    # Cross-spectrum
    _, Sxy = csd(x, y, fs=sample_rate_hz, nperseg=nperseg,
                  noverlap=noverlap, window=window)
    # Coherencia
    _, gamma2 = coherence(x, y, fs=sample_rate_hz, nperseg=nperseg,
                           noverlap=noverlap, window=window)

    # H1 = Sxy / Sxx (evitar división por 0)
    Sxx_safe = np.where(Sxx > 1e-30, Sxx, 1e-30)
    H1 = Sxy / Sxx_safe

    # Número aproximado de promedios de Welch
    n_avg = max(1, int(np.ceil((x.size - noverlap) / (nperseg - noverlap))))

    return FRFResult(
        frequencies_hz=freq,
        frf_complex=H1,
        coherence=gamma2,
        estimator="H1",
        n_averages=n_avg,
        window=window,
    )


def compute_frf_h2(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 1024,
    noverlap: Optional[int] = None,
    window: str = "hann",
) -> FRFResult:
    """
    Calcula FRF usando estimador H2 = Syy / Syx*.

    Útil cuando se sospecha ruido alto en la excitación.
    """
    try:
        from scipy.signal import csd, welch, coherence
    except ImportError as exc:
        raise ImportError("scipy es requerido") from exc

    x = np.asarray(input_signal, dtype=float)
    y = np.asarray(output_signal, dtype=float)
    if x.size < nperseg:
        nperseg = max(64, x.size // 2)
    if noverlap is None:
        noverlap = nperseg // 2

    _, Syy = welch(y, fs=sample_rate_hz, nperseg=nperseg,
                    noverlap=noverlap, window=window)
    freq, Syx = csd(y, x, fs=sample_rate_hz, nperseg=nperseg,
                     noverlap=noverlap, window=window)
    _, gamma2 = coherence(x, y, fs=sample_rate_hz, nperseg=nperseg,
                           noverlap=noverlap, window=window)

    Syx_safe = np.where(np.abs(Syx) > 1e-30, Syx, 1e-30 + 0j)
    H2 = Syy / Syx_safe
    n_avg = max(1, int(np.ceil((x.size - noverlap) / (nperseg - noverlap))))

    return FRFResult(
        frequencies_hz=freq, frf_complex=H2, coherence=gamma2,
        estimator="H2", n_averages=n_avg, window=window,
    )


def average_frf(frfs: List[FRFResult]) -> FRFResult:
    """
    Promedia múltiples FRFs (típico en ensayos con martillo, 5-10 impactos).

    Asume que todas tienen el mismo eje de frecuencia (mismo nperseg + fs).
    """
    if not frfs:
        raise ValueError("Lista de FRFs vacía")
    if len(frfs) == 1:
        return frfs[0]

    ref = frfs[0]
    for f in frfs[1:]:
        if f.frequencies_hz.size != ref.frequencies_hz.size:
            raise ValueError("Las FRFs deben tener el mismo eje de frecuencia")

    avg_complex = np.mean([f.frf_complex for f in frfs], axis=0)
    avg_coh = np.mean([f.coherence for f in frfs], axis=0)
    return FRFResult(
        frequencies_hz=ref.frequencies_hz,
        frf_complex=avg_complex,
        coherence=avg_coh,
        estimator=ref.estimator,
        n_averages=sum(f.n_averages for f in frfs),
        window=ref.window,
    )


# =====================================================================
# Detección de modos por half-power method
# =====================================================================

def detect_modal_peaks(
    frequencies_hz: np.ndarray,
    magnitude: np.ndarray,
    coherence: Optional[np.ndarray] = None,
    f_min_hz: float = 5.0,
    f_max_hz: Optional[float] = None,
    prominence_db: float = 6.0,
    min_distance_hz: float = 2.0,
    coherence_threshold: float = 0.7,
) -> List[ModalPeak]:
    """
    Detecta picos modales en una FRF y calcula damping por half-power.

    Args:
        frequencies_hz: Eje de frecuencia
        magnitude: |H(f)| (linear, no dB)
        coherence: γ² opcional para evaluar confiabilidad
        f_min_hz: Mínima frecuencia donde buscar picos
        f_max_hz: Máxima frecuencia. Si None, usa Nyquist
        prominence_db: Mínima prominencia del pico en dB
        min_distance_hz: Mínima separación entre picos (Hz)
        coherence_threshold: Picos con coherencia < threshold se marcan unreliable

    Returns:
        Lista de ModalPeak ordenados por frecuencia ascendente
    """
    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise ImportError("scipy es requerido para detect_modal_peaks") from exc

    f = np.asarray(frequencies_hz, dtype=float)
    mag = np.asarray(magnitude, dtype=float)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-30))

    if f_max_hz is None:
        f_max_hz = float(f[-1])
    band = (f >= f_min_hz) & (f <= f_max_hz)

    df = float(f[1] - f[0]) if len(f) > 1 else 1.0
    distance_samples = max(1, int(round(min_distance_hz / df)))

    # Buscar picos en la banda de interés
    f_band = f[band]
    mag_band = mag[band]
    mag_db_band = mag_db[band]

    peak_indices, _ = find_peaks(
        mag_db_band,
        prominence=prominence_db,
        distance=distance_samples,
    )

    peaks: List[ModalPeak] = []
    for idx in peak_indices:
        fn = float(f_band[idx])
        mag_peak = float(mag_band[idx])
        target = mag_peak / math.sqrt(2.0)  # half-power: -3 dB

        # Hacia la izquierda: encontrar f1 donde mag cae al target
        f1 = fn
        for i in range(idx, -1, -1):
            if mag_band[i] <= target:
                # Interpolar linealmente entre i e i+1
                if i + 1 <= idx:
                    m0, m1 = mag_band[i], mag_band[i + 1]
                    if m1 > m0:
                        frac = (target - m0) / (m1 - m0)
                        f1 = float(f_band[i] + frac * (f_band[i + 1] - f_band[i]))
                    else:
                        f1 = float(f_band[i])
                break

        # Hacia la derecha: encontrar f2
        f2 = fn
        for i in range(idx, len(mag_band)):
            if mag_band[i] <= target:
                if i - 1 >= idx:
                    m0, m1 = mag_band[i - 1], mag_band[i]
                    if m0 > m1:
                        frac = (target - m1) / (m0 - m1)
                        f2 = float(f_band[i] - frac * (f_band[i] - f_band[i - 1]))
                    else:
                        f2 = float(f_band[i])
                break

        bw = max(f2 - f1, 1e-9)
        damping = bw / (2.0 * fn) * 100.0  # porcentaje
        q_factor = fn / bw if bw > 0 else float("inf")

        # Coherencia en el pico
        coh_at_peak = None
        is_reliable = True
        if coherence is not None:
            coh = np.asarray(coherence, dtype=float)
            # idx corresponde a f_band, mapear a el array completo
            full_idx = int(np.argmin(np.abs(f - fn)))
            coh_at_peak = float(coh[full_idx])
            if coh_at_peak < coherence_threshold:
                is_reliable = False

        peaks.append(ModalPeak(
            frequency_hz=fn,
            damping_ratio_pct=damping,
            magnitude_peak=mag_peak,
            half_power_f1_hz=f1,
            half_power_f2_hz=f2,
            bandwidth_hz=bw,
            quality_factor=q_factor,
            coherence_at_peak=coh_at_peak,
            is_reliable=is_reliable,
        ))

    peaks.sort(key=lambda p: p.frequency_hz)
    return peaks


def half_power_damping(
    magnitude: np.ndarray,
    frequencies_hz: np.ndarray,
    peak_freq_hz: float,
) -> float:
    """
    Estimación rápida de damping para un solo modo conocido.

    Args:
        magnitude: |H(f)| de la FRF
        frequencies_hz: eje de frecuencia
        peak_freq_hz: frecuencia del pico identificado

    Returns:
        Damping ratio en %
    """
    f = np.asarray(frequencies_hz, dtype=float)
    mag = np.asarray(magnitude, dtype=float)
    idx = int(np.argmin(np.abs(f - peak_freq_hz)))
    target = mag[idx] / math.sqrt(2.0)

    f1 = peak_freq_hz
    for i in range(idx, -1, -1):
        if mag[i] <= target:
            f1 = float(f[i])
            break

    f2 = peak_freq_hz
    for i in range(idx, len(mag)):
        if mag[i] <= target:
            f2 = float(f[i])
            break

    bw = max(f2 - f1, 1e-9)
    return bw / (2.0 * peak_freq_hz) * 100.0
