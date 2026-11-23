"""
core/remote_monitoring/derived.py
=================================

Mediciones DERIVADAS de un mismo transductor (sin sensor ni BNC extra).

Caso típico: en campo se instala UN acelerómetro (ej. 4YA, bomba de lubricación)
y se quiere también la VELOCIDAD (4YV). El rack Bently 3500 lo hace por dentro
(integra y publica 2 registros Modbus). Con el nodo NI lo hacemos igual sobre la
onda cruda del 9234: integración en frecuencia ("omega arithmetic") dentro de la
banda de severidad ISO 20816 (10–1000 Hz por defecto).

En la config (Machines) la medición derivada es un ChannelRow con
`derived_from = "4Y_A"` y `measure = "velocity"`: conserva su sensor_label y su
variable de Live (continuidad Modbus → NI) y no ocupa BNC.

Core puro (numpy).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

G_MM_S2 = 9806.65                     # 1 g en mm/s²
MEASURES = ("velocity",)              # derivaciones soportadas


def accel_to_velocity(signal_g: np.ndarray, fs: float, fmin: float = 10.0,
                      fmax: float = 1000.0) -> np.ndarray:
    """Onda de aceleración (g) → onda de velocidad (mm/s) por integración en
    frecuencia, limitada a [fmin, fmax] (quita deriva DC y ruido de alta)."""
    x = np.asarray(signal_g, dtype=float)
    n = x.size
    if n < 4 or fs <= 0:
        return np.zeros_like(x)
    x = x - x.mean()
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (f >= max(fmin, f[1] if f.size > 1 else 0.0)) & (f <= min(fmax, fs / 2.0))
    out = np.zeros_like(spec)
    out[band] = spec[band] * G_MM_S2 / (1j * 2.0 * np.pi * f[band])
    return np.fft.irfft(out, n=n)


def velocity_rms_from_accel(signal_g: np.ndarray, fs: float, fmin: float = 10.0,
                            fmax: float = 1000.0) -> float:
    """Velocidad RMS (mm/s) en la banda ISO 20816 a partir de la onda de aceleración (g)."""
    v = accel_to_velocity(signal_g, fs, fmin, fmax)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else 0.0


def derived_value(measure: str, parent_signal: np.ndarray, fs: float,
                  band: Optional[Tuple[float, float]] = None) -> Optional[float]:
    """Valor escalar de una medición derivada (para el Direct de Live)."""
    if measure == "velocity":
        lo, hi = band or (10.0, 1000.0)
        return velocity_rms_from_accel(parent_signal, fs, lo, hi)
    return None
