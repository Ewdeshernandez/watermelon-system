"""
tests.test_tsa
==============

Validación de core.tsa.analyze_tsa() — Time Synchronous Average.

Idea física: el TSA promedia muchas revoluciones alineadas, lo cual
elimina componentes asíncronos (ruido + frecuencias no enteras del rpm)
y deja sólo lo síncrono. Por tanto:

  - Para una señal pura 1X + ruido blanco, el TSA debe converger a 1X
    casi sin ruido, conservando amplitud y forma.
  - Para una señal con componentes asíncronos (subsync 0.45X), el TSA
    debe atenuarlos significativamente.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.tsa import analyze_tsa
from tests.conftest import make_pure_sine, make_signal_obj, make_subsynchronous


# -----------------------------------------------------------------
# Sine + ruido → TSA limpia el ruido
# -----------------------------------------------------------------

def test_tsa_reduces_white_noise_for_pure_sine():
    sig = make_pure_sine(amplitude_peak=1.0, noise_rms=0.3, duration=4.0)
    signal_obj = make_signal_obj(sig["y"], sig["fs"], sig["rpm"])

    out = analyze_tsa(signal=signal_obj, fs=sig["fs"], rpm=sig["rpm"])

    avg = np.asarray(out["tsa_mean"], dtype=float)
    # Amplitud conservada dentro del 10%
    avg_pp = float(np.max(avg) - np.min(avg))
    expected_pp = 2.0 * sig["amplitude_peak"]
    assert abs(avg_pp - expected_pp) / expected_pp < 0.10

    # El SNR del promedio debe ser claramente mayor que el de la señal cruda.
    snr_raw = sig["amplitude_peak"] / sig["noise_rms"]
    # Estimamos noise residual del promedio comparando con sine ideal teórica.
    n = avg.size
    t_rev = np.arange(n) / float(n)  # un período normalizado
    ideal = sig["amplitude_peak"] * np.sin(2 * np.pi * t_rev + (avg_pp - avg_pp))
    # Como la fase del promedio puede no ser 0, comparamos amplitud de la
    # mejor sinusoide ajustada:
    # proyectar avg sobre cos y sin → amplitud del armónico fundamental
    c = float(np.mean(avg * np.cos(2 * np.pi * t_rev))) * 2.0
    s = float(np.mean(avg * np.sin(2 * np.pi * t_rev))) * 2.0
    fundamental_amp = float(np.hypot(c, s))
    residual = avg - (c * np.cos(2 * np.pi * t_rev) + s * np.sin(2 * np.pi * t_rev))
    residual_rms = float(np.sqrt(np.mean(residual ** 2)))
    snr_avg = fundamental_amp / max(residual_rms, 1e-12)
    assert snr_avg > snr_raw, (
        f"TSA debería mejorar SNR. raw={snr_raw:.2f}, avg={snr_avg:.2f}"
    )


# -----------------------------------------------------------------
# TSA atenúa componentes asíncronos (subsync 0.45X)
# -----------------------------------------------------------------

def test_tsa_attenuates_subsynchronous():
    sig = make_subsynchronous(
        rpm=3600.0,
        sub_order=0.45,
        sub_amp=0.7,
        one_x_amp=1.0,
        duration=4.0,
        noise_rms=0.0,
    )
    signal_obj = make_signal_obj(sig["y"], sig["fs"], sig["rpm"])
    out = analyze_tsa(signal=signal_obj, fs=sig["fs"], rpm=sig["rpm"])
    avg = np.asarray(out["tsa_mean"], dtype=float)

    # Energía total del promedio vs energía total de la señal cruda
    raw_rms = float(np.sqrt(np.mean(sig["y"] ** 2)))
    avg_rms = float(np.sqrt(np.mean(avg ** 2)))

    # El RMS del promedio debe ser MENOR que el de la señal cruda porque
    # la componente subsync se cancela. Toleramos 0.85 como umbral
    # (cancelación parcial es esperable en pocas revoluciones).
    assert avg_rms < 0.85 * raw_rms, (
        f"TSA no atenúa subsync: raw_rms={raw_rms:.4f} avg_rms={avg_rms:.4f}"
    )


# -----------------------------------------------------------------
# Robustez
# -----------------------------------------------------------------

def test_tsa_invalid_rpm_raises():
    signal_obj = make_signal_obj(np.zeros(2000), fs=1000.0, rpm=0.0)
    with pytest.raises((ValueError, ZeroDivisionError, OverflowError)):
        analyze_tsa(signal=signal_obj, fs=1000.0, rpm=0.0)


def test_tsa_returns_one_revolution_array():
    sig = make_pure_sine(rpm=1800.0, duration=2.0)
    signal_obj = make_signal_obj(sig["y"], sig["fs"], sig["rpm"])
    out = analyze_tsa(signal=signal_obj, fs=sig["fs"], rpm=sig["rpm"])
    samples_per_rev = int(round(sig["fs"] / (sig["rpm"] / 60.0)))
    avg = np.asarray(out["tsa_mean"], dtype=float)
    # El promedio debe tener exactamente 1 revolución de muestras
    assert avg.size == samples_per_rev
