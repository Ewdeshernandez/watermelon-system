"""
tests.conftest
==============

Fixtures globales y generadores de señales sintéticas para la suite de
pruebas de Watermelon System.

Toda señal sintética se construye con parámetros conocidos (amplitudes,
frecuencias, RPM, fs) para que las aserciones de los tests puedan
comparar contra el ground-truth analítico, no contra una corrida
arbitraria. Esto convierte la suite en un golden dataset que permite
refactorizar core/ sin miedo.

Convenciones:
    - Tiempo en segundos.
    - Frecuencia en Hz (no CPM).
    - RPM en rev/min (RPM = freq_Hz * 60).
    - Amplitudes:
        * pico (peak)        — para señales tipo sine A·sin(2πft).
        * pico-pico (pp)     — para desplazamiento (vibration probes).
        * RMS                — para velocity/acceleration (ISO).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest


# Asegurar que el paquete del proyecto sea importable cuando se corre
# pytest desde la raíz, sin depender de instalación con setup.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================
# Constantes utilitarias
# =============================================================

DEFAULT_FS = 25_600.0  # Hz, típico de tarjeta DAQ industrial
DEFAULT_RPM = 3600.0   # rev/min — turbomáquina 60 Hz 2 polos
DEFAULT_DURATION = 1.0  # s


def hz_from_rpm(rpm: float) -> float:
    """Convierte RPM a Hz."""
    return float(rpm) / 60.0


def rpm_from_hz(hz: float) -> float:
    """Convierte Hz a RPM."""
    return float(hz) * 60.0


def make_time(fs: float = DEFAULT_FS, duration: float = DEFAULT_DURATION) -> np.ndarray:
    """Genera vector de tiempo uniforme."""
    n = int(round(fs * duration))
    return np.arange(n, dtype=float) / float(fs)


def make_signal_obj(
    y: np.ndarray,
    fs: float,
    rpm: float,
    time: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SimpleNamespace:
    """
    Construye un objeto signal compatible con la API real de Watermelon
    (analyze_order_tracking, analyze_tsa, etc.) que espera atributo .x.

    Estos consumidores usan hasattr(signal, "x") y, si encuentran un dict
    sin esa propiedad, caen a la rama .values (que en dicts es un método),
    de modo que el camino seguro y documentado es entregarles un objeto
    con .x y .metadata. Esto reproduce la interfaz real (Signal Watermelon).
    """
    md = {"rpm": float(rpm), "RPM": float(rpm)}
    if metadata:
        md.update(metadata)
    obj = SimpleNamespace(x=np.asarray(y, dtype=float), metadata=md)
    if time is not None:
        obj.time = np.asarray(time, dtype=float)
    return obj


# =============================================================
# Generadores de señales sintéticas
# =============================================================

def make_pure_sine(
    rpm: float = DEFAULT_RPM,
    amplitude_peak: float = 1.0,
    fs: float = DEFAULT_FS,
    duration: float = DEFAULT_DURATION,
    phase_deg: float = 0.0,
    dc_offset: float = 0.0,
    noise_rms: float = 0.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Genera una sinusoide pura a la frecuencia rotacional 1X.

    Returns:
        dict con 'time', 'y', 'fs', 'rpm', 'amplitude_peak', 'expected'.
        'expected' lleva los valores cerrados que un test debe poder
        recuperar:
            - rms_expected = A / sqrt(2)
            - peak_expected = A
            - peak_to_peak_expected = 2A
            - crest_expected = sqrt(2)
            - kurtosis_expected ≈ 1.5  (kurtosis de una sinusoide pura
              en convención momento_4/std^4 es 1.5; 'excess kurtosis'
              -1.5).
    """
    t = make_time(fs, duration)
    f = hz_from_rpm(rpm)
    phase = np.deg2rad(phase_deg)
    y = amplitude_peak * np.sin(2 * np.pi * f * t + phase) + dc_offset

    if noise_rms > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(loc=0.0, scale=noise_rms, size=y.shape)

    A = float(amplitude_peak)
    expected = {
        "rms": A / np.sqrt(2.0),
        "peak": A,
        "peak_to_peak": 2.0 * A,
        "crest_factor": np.sqrt(2.0),
        # kurtosis "raw" (momento4/std^4); para señales con DC=0 y sine pura → 1.5
        "kurtosis_raw": 1.5,
        "kurtosis_excess": -1.5,
        "skewness": 0.0,
        "f_hz": f,
        "rpm": float(rpm),
    }

    return {
        "time": t,
        "y": y,
        "fs": float(fs),
        "rpm": float(rpm),
        "amplitude_peak": A,
        "noise_rms": float(noise_rms),
        "duration": float(duration),
        "expected": expected,
    }


def make_multi_harmonic(
    rpm: float = DEFAULT_RPM,
    harmonics: Dict[int, float] = None,
    fs: float = DEFAULT_FS,
    duration: float = DEFAULT_DURATION,
    noise_rms: float = 0.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Genera una señal compuesta por varias órdenes (1X, 2X, 3X, ...).

    Args:
        harmonics: {orden: amplitud_pico}. Por defecto {1: 1.0, 2: 0.3, 3: 0.1}.

    Patrones típicos en máquinas reales:
        - {1: 1.0}                      → sólo desbalance
        - {1: 0.6, 2: 0.8}              → misalignment paralelo (2X domina)
        - {1: 0.5, 2: 0.4, 3: 0.3}      → misalignment angular + bent shaft
        - {1: 1.0, 2: 0.05}             → 1X limpio sin misalignment
    """
    if harmonics is None:
        harmonics = {1: 1.0, 2: 0.3, 3: 0.1}

    t = make_time(fs, duration)
    f1 = hz_from_rpm(rpm)
    y = np.zeros_like(t)
    for order, amp in harmonics.items():
        y = y + float(amp) * np.sin(2 * np.pi * float(order) * f1 * t)

    if noise_rms > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(loc=0.0, scale=noise_rms, size=y.shape)

    return {
        "time": t,
        "y": y,
        "fs": float(fs),
        "rpm": float(rpm),
        "harmonics": dict(harmonics),
        "noise_rms": float(noise_rms),
        "duration": float(duration),
    }


def make_subsynchronous(
    rpm: float = DEFAULT_RPM,
    sub_order: float = 0.45,
    sub_amp: float = 0.7,
    one_x_amp: float = 1.0,
    fs: float = DEFAULT_FS,
    duration: float = DEFAULT_DURATION,
    noise_rms: float = 0.02,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Señal con componente subsíncrona (0.45X típico de oil whirl, ~0.5X
    de oil whip). Usa esto para validar detect_subsynchronous().
    """
    t = make_time(fs, duration)
    f1 = hz_from_rpm(rpm)
    y = (
        one_x_amp * np.sin(2 * np.pi * f1 * t)
        + sub_amp * np.sin(2 * np.pi * sub_order * f1 * t)
    )

    if noise_rms > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(loc=0.0, scale=noise_rms, size=y.shape)

    return {
        "time": t,
        "y": y,
        "fs": float(fs),
        "rpm": float(rpm),
        "sub_order": float(sub_order),
        "sub_amp": float(sub_amp),
        "one_x_amp": float(one_x_amp),
        "duration": float(duration),
    }


def make_bearing_impacts(
    rpm: float = DEFAULT_RPM,
    bpfo_factor: float = 3.572,
    impact_amp: float = 1.0,
    one_x_amp: float = 0.2,
    fs: float = DEFAULT_FS,
    duration: float = DEFAULT_DURATION,
    decay_tau: float = 1e-3,
    ring_freq_hz: float = 3500.0,
    noise_rms: float = 0.05,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Genera impactos periódicos a la frecuencia BPFO (típico de defecto
    en pista externa de un rodamiento). Cada impacto es un decaimiento
    exponencial modulado por la frecuencia de resonancia (ring_freq_hz)
    del soporte. Esto reproduce la firma clásica que envelope/Hilbert
    debería extraer.

    Default factor 3.572 = aprox SKF 6309 BPFO factor.

    Returns:
        dict con campos esperados (BPFO_hz, BPFI_hz, etc. para asserts).
    """
    rng = np.random.default_rng(seed)
    t = make_time(fs, duration)
    f1 = hz_from_rpm(rpm)
    bpfo_hz = bpfo_factor * f1

    # Base: 1X de fondo (desbalance leve siempre presente)
    y = one_x_amp * np.sin(2 * np.pi * f1 * t)

    # Impactos periódicos
    impact_period = 1.0 / bpfo_hz
    impact_times = np.arange(impact_period, duration, impact_period)
    for ti in impact_times:
        # Cada impacto = ring decay
        delta = t - ti
        mask = delta >= 0
        ring = np.zeros_like(t)
        ring[mask] = (
            impact_amp
            * np.exp(-delta[mask] / decay_tau)
            * np.sin(2 * np.pi * ring_freq_hz * delta[mask])
        )
        y = y + ring

    if noise_rms > 0:
        y = y + rng.normal(loc=0.0, scale=noise_rms, size=y.shape)

    return {
        "time": t,
        "y": y,
        "fs": float(fs),
        "rpm": float(rpm),
        "bpfo_hz": float(bpfo_hz),
        "bpfo_factor": float(bpfo_factor),
        "impact_amp": float(impact_amp),
        "ring_freq_hz": float(ring_freq_hz),
        "duration": float(duration),
    }


def make_looseness(
    rpm: float = DEFAULT_RPM,
    n_harmonics: int = 6,
    decay: float = 0.55,
    fs: float = DEFAULT_FS,
    duration: float = DEFAULT_DURATION,
    noise_rms: float = 0.05,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Patrón de holgura mecánica: serie de armónicos enteros 1X..NX con
    decaimiento moderado. Detector espera ver muchos armónicos.
    """
    t = make_time(fs, duration)
    f1 = hz_from_rpm(rpm)

    y = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        y = y + (decay ** (k - 1)) * np.sin(2 * np.pi * k * f1 * t)

    if noise_rms > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(loc=0.0, scale=noise_rms, size=y.shape)

    return {
        "time": t,
        "y": y,
        "fs": float(fs),
        "rpm": float(rpm),
        "n_harmonics": int(n_harmonics),
        "decay": float(decay),
        "duration": float(duration),
    }


def make_bode_curve(
    critical_rpm: float = 4000.0,
    q_factor: float = 6.0,
    rpm_min: float = 500.0,
    rpm_max: float = 8000.0,
    n_points: int = 200,
    base_amp: float = 5.0,
    response_amp: float = 50.0,
    add_noise_pct: float = 0.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Genera curva Bode amplitud + fase ideal para una crítica simple.
    Modelo: respuesta de un sistema de un grado de libertad subamortiguado
    barrido contra RPM. Útil para validar detect_critical_speeds().

    La amplitud se modela como un Lorentziano (transfer function 2do orden
    cerca de la resonancia), la fase salta ~180° pasando por -90° en la
    crítica.

    Returns:
        dict con 'rpm', 'amp', 'phase' + 'expected' (critical_rpm, q).
    """
    rpm = np.linspace(rpm_min, rpm_max, n_points)
    f = rpm / critical_rpm  # razón de frecuencia
    zeta = 1.0 / (2.0 * q_factor)

    # Magnitud transfer function 2do orden: 1 / sqrt((1-f^2)^2 + (2 ζ f)^2)
    denom = np.sqrt((1 - f ** 2) ** 2 + (2 * zeta * f) ** 2)
    mag = 1.0 / denom

    # Normalizar para que en la crítica el amp valga base + response
    mag_at_critical = 1.0 / (2.0 * zeta)
    amp = base_amp + (response_amp / mag_at_critical) * mag

    # Fase: -arctan(2 ζ f / (1 - f^2)) en grados, ajustada a [0, 360)
    phase_rad = -np.arctan2(2 * zeta * f, 1 - f ** 2)
    phase_deg = np.rad2deg(phase_rad)
    phase_deg = (phase_deg + 360.0) % 360.0

    if add_noise_pct > 0:
        rng = np.random.default_rng(seed)
        amp = amp * (1.0 + rng.normal(0.0, add_noise_pct, size=amp.shape))
        phase_deg = phase_deg + rng.normal(0.0, 2.0, size=phase_deg.shape)
        phase_deg = (phase_deg + 360.0) % 360.0

    return {
        "rpm": rpm,
        "amp": amp,
        "phase": phase_deg,
        "expected": {
            "critical_rpm": float(critical_rpm),
            "q_factor": float(q_factor),
            "amp_at_critical_min": base_amp + response_amp * 0.7,
        },
    }


def make_orbit_xy(
    rpm: float = DEFAULT_RPM,
    amp_x: float = 50.0,
    amp_y: float = 50.0,
    phase_y_lag_deg: float = 90.0,
    samples_per_rev: int = 256,
    n_revs: int = 8,
    noise_rms: float = 0.5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Genera par X/Y para órbita. Por defecto círculo perfecto (90° lag),
    útil para validar precesión y forward whirl.

    amp en µm pp (escala típica de probes de proximidad).
    """
    f1 = hz_from_rpm(rpm)
    fs = samples_per_rev * f1
    duration = n_revs / f1
    t = make_time(fs, duration)

    x = (amp_x / 2.0) * np.sin(2 * np.pi * f1 * t)
    y = (amp_y / 2.0) * np.sin(2 * np.pi * f1 * t - np.deg2rad(phase_y_lag_deg))

    if noise_rms > 0:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0.0, noise_rms, size=x.shape)
        y = y + rng.normal(0.0, noise_rms, size=y.shape)

    return {
        "time": t,
        "x": x,
        "y": y,
        "fs": float(fs),
        "rpm": float(rpm),
        "samples_per_rev": int(samples_per_rev),
        "n_revs": int(n_revs),
        "amp_x_pp": float(amp_x),
        "amp_y_pp": float(amp_y),
        "phase_y_lag_deg": float(phase_y_lag_deg),
    }


# =============================================================
# Fixtures pytest reutilizables
# =============================================================

@pytest.fixture
def sine_clean():
    """Sinusoide pura 3600 RPM, 1.0 peak, 1 s a 25.6 kHz."""
    return make_pure_sine()


@pytest.fixture
def sine_noisy():
    """Sinusoide pura 3600 RPM, ruido 5% RMS — validar tolerancia."""
    return make_pure_sine(noise_rms=0.05)


@pytest.fixture
def multi_harmonic_clean():
    """Misalignment-like: 1X dominante + 2X + 3X."""
    return make_multi_harmonic(harmonics={1: 1.0, 2: 0.3, 3: 0.1})


@pytest.fixture
def multi_harmonic_misalignment():
    """Misalignment paralelo: 2X domina."""
    return make_multi_harmonic(harmonics={1: 0.6, 2: 0.8})


@pytest.fixture
def subsync_oil_whirl():
    """Oil whirl clásico: 0.45X amplitud comparable a 1X."""
    return make_subsynchronous()


@pytest.fixture
def bearing_bpfo():
    """Defecto pista externa SKF 6309 (factor 3.572)."""
    return make_bearing_impacts(bpfo_factor=3.572)


@pytest.fixture
def looseness_pattern():
    """Holgura mecánica: 6 armónicos enteros."""
    return make_looseness()


@pytest.fixture
def bode_clean():
    """Bode con crítica clara en 4000 RPM, Q=6."""
    return make_bode_curve()


@pytest.fixture
def orbit_circular():
    """Órbita circular perfecta forward 50 µm pp."""
    return make_orbit_xy()
