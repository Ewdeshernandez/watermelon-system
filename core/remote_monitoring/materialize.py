"""
core/remote_monitoring/materialize.py — Ventana del buffer → Signal
===================================================================

Puente entre el streaming en vivo y los motores de gráficos existentes.

Toma un snapshot del RingBuffer (n_channels, n_samples) + los ChannelConfig
y produce objetos `core.signal_registry.Signal` (vía LoadedSignal), que es
EXACTAMENTE lo que consumen Spectrum, Time Waveforms, Orbit, Bode, Polar,
Trends, Shaft Centerline, etc. Cero reimplementación de gráficos.

Nota sobre escalado: el snapshot viene en Volts crudos. Acá aplicamos la
sensitivity (mV/EU) para pasar a unidades de ingeniería (g, mil, mm/s...).
La misma convención que usa el módulo modal en signal_scaling.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.loaders.base import LoadedSignal, loaded_to_signal
from core.modal.acq_backend import ChannelConfig
from core.remote_monitoring.stream_source import is_keyphasor_channel


def _volts_to_eu(volts: np.ndarray, ch: ChannelConfig) -> np.ndarray:
    """Convierte Volts crudos a unidades de ingeniería usando sensitivity.

    sensitivity_mv_per_eu = mV por unidad de ingeniería.
    EU = V / (mV/EU / 1000) = V * 1000 / sensitivity_mv_per_eu.
    Si sensitivity es 0/inválida, devuelve Volts sin escalar (fallback seguro).
    """
    s = float(ch.sensitivity_mv_per_eu or 0.0)
    if s <= 0:
        return np.asarray(volts, dtype=float)
    return np.asarray(volts, dtype=float) * 1000.0 / s


def window_to_loaded_signals(
    snapshot: np.ndarray,
    channels: List[ChannelConfig],
    fs: float,
    rpm: Optional[float] = None,
    captured_at: Optional[str] = None,
    tag_prefix: str = "LIVE",
    keyphasor_name: Optional[str] = None,
    apply_scaling: bool = True,
    include_keyphasor: bool = False,
) -> List[LoadedSignal]:
    """Convierte una ventana del buffer en LoadedSignal por canal.

    Por defecto EXCLUYE el keyphasor (no es una señal de vibración a
    graficar, es la referencia de fase). Se puede incluir con
    include_keyphasor=True (útil para debug del tacómetro).
    """
    if snapshot.ndim != 2:
        raise ValueError(f"snapshot debe ser 2D (n_ch, n); recibí {snapshot.shape}")
    if snapshot.shape[0] != len(channels):
        raise ValueError(
            f"snapshot tiene {snapshot.shape[0]} canales pero channels tiene "
            f"{len(channels)}"
        )

    n = snapshot.shape[1]
    time = np.arange(n, dtype=float) / float(fs) if n else np.zeros(0)

    out: List[LoadedSignal] = []
    for ci, ch in enumerate(channels):
        is_kph = is_keyphasor_channel(ch, keyphasor_name)
        if is_kph and not include_keyphasor:
            continue

        raw = snapshot[ci]
        x = raw if is_kph or not apply_scaling else _volts_to_eu(raw, ch)
        units = "V" if is_kph else ch.units

        ls = LoadedSignal(
            file_name=f"{tag_prefix}:{ch.name}",
            x=np.asarray(x, dtype=float),
            time=time.copy(),
            y=None,
            fs=float(fs),
            rpm=float(rpm) if rpm is not None else None,
            units=units,
            domain="time",
            vendor="watermelon",
            metadata={
                "sensor_label": ch.name,
                "bnc_port": ch.bnc_port,
                "coupling": ch.coupling,
                "sensitivity_mv_per_eu": float(ch.sensitivity_mv_per_eu or 0.0),
                "role": "keyphasor" if is_kph else "vibration",
                "source": "remote_monitoring.live",
                "captured_at": captured_at,
            },
        )
        ls.validate()
        out.append(ls)
    return out


def window_to_signals(
    snapshot: np.ndarray,
    channels: List[ChannelConfig],
    fs: float,
    **kwargs,
) -> List:
    """Igual que window_to_loaded_signals pero devuelve objetos Signal
    (`core.signal_registry.Signal`) listos para SignalRegistry / gráficos."""
    loaded = window_to_loaded_signals(snapshot, channels, fs, **kwargs)
    return [loaded_to_signal(ls) for ls in loaded]
