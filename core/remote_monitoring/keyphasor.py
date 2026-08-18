"""
core/remote_monitoring/keyphasor.py — Referencia de fase (tacómetro)
====================================================================

El keyphasor es un pulso once-per-rev. De él derivamos:
  · rpm  — de los intervalos entre pulsos.
  · vector 1X (amplitud + fase) de cada canal de vibración, referenciado
    al keyphasor → habilita Bode, Polar, Orbita compensada y 1X phase.

Sin keyphasor solo hay espectro/waveform/tendencia (nada síncrono).

Detección polaridad-independiente: se mide la desviación respecto a la
mediana y se marca "pulso" cuando |desv| supera un umbral adaptativo. Así
funciona igual con pulsos positivos (típico) o negativos (nuestro sim /
algunos notch de Bently).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class KeyphasorResult:
    rpm: Optional[float]
    f1_hz: Optional[float]                 # rpm / 60
    n_pulses: int
    pulse_sample_indices: np.ndarray       # índices de los flancos de entrada
    ref_sample: Optional[int]              # primer flanco → t=0 de fase


def detect_keyphasor(
    kph: np.ndarray,
    fs: float,
    min_rpm: float = 60.0,
    max_rpm: float = 60000.0,
    threshold_frac: float = 0.5,
) -> KeyphasorResult:
    """Detecta pulsos once-per-rev y estima rpm.

    threshold_frac: fracción del pico de desviación para marcar pulso (0.5
    = mitad de la altura del pulso).
    """
    kph = np.asarray(kph, dtype=float)
    n = kph.size
    if n < 2:
        return KeyphasorResult(None, None, 0, np.zeros(0, dtype=int), None)

    base = np.median(kph)
    dev = np.abs(kph - base)
    peak = float(np.max(dev))
    if peak <= 1e-9:
        return KeyphasorResult(None, None, 0, np.zeros(0, dtype=int), None)

    thr = threshold_frac * peak
    active = dev > thr
    # flancos de entrada: pasa de inactivo a activo
    rising = np.flatnonzero((~active[:-1]) & (active[1:])) + 1
    if rising.size < 2:
        return KeyphasorResult(None, None, rising.size,
                               rising.astype(int), int(rising[0]) if rising.size else None)

    intervals = np.diff(rising) / fs           # segundos por revolución
    # descartar intervalos fuera de rango físico
    lo = 60.0 / max_rpm
    hi = 60.0 / min_rpm
    good = intervals[(intervals >= lo) & (intervals <= hi)]
    if good.size == 0:
        return KeyphasorResult(None, None, rising.size, rising.astype(int), int(rising[0]))

    rev_period = float(np.median(good))
    rpm = 60.0 / rev_period
    return KeyphasorResult(
        rpm=rpm,
        f1_hz=rpm / 60.0,
        n_pulses=int(rising.size),
        pulse_sample_indices=rising.astype(int),
        ref_sample=int(rising[0]),
    )


def one_x_vector(
    vib: np.ndarray,
    fs: float,
    f1_hz: float,
    ref_sample: int = 0,
) -> Tuple[float, float]:
    """Amplitud (0-pk, en EU) y fase (grados) del 1X, referenciado al
    keyphasor (ref_sample = t=0).

    Fase en convención "phase lag" positiva creciente. La amplitud usa
    corrección de ventana Hanning para que sea comparable a un pico real.
    """
    vib = np.asarray(vib, dtype=float)
    n = vib.size
    if n == 0 or f1_hz <= 0:
        return 0.0, 0.0

    x = vib - np.mean(vib)
    w = np.hanning(n)
    # corrección de amplitud de Hanning (ganancia coherente = 0.5)
    win_gain = np.sum(w) / n
    t = (np.arange(n) - ref_sample) / fs
    ref = np.exp(-1j * 2.0 * np.pi * f1_hz * t)
    c = np.sum(x * w * ref) / n / win_gain
    amp = 2.0 * np.abs(c)
    phase = np.degrees(np.angle(c))
    # normaliza a [0, 360)
    phase = float(phase % 360.0)
    return float(amp), phase
