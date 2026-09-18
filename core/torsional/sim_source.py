"""
core/torsional/sim_source.py — Fuente simulada de torque (TorqueTrak 10K)
=========================================================================

Genera data de **par torsional** sin hardware, para desarrollo en Mac y tests.
Implementa la misma interfaz `StreamSource` del módulo de monitoreo, así que
reusa tal cual el `AcqAgent`, el `RingBuffer` y el grabador de transitorio
(`core.remote_monitoring`), y el `on_block` de la UI de campo.

La fuente entrega **volts crudos** — exactamente lo que produciría la salida
±10 V del RX10K leída por la NI 9229/9215 — y aguas abajo se aplica
`core.torsional.scaling.voltage_to_torque` para volver a unidades de ingeniería.
El sim codifica torque→volts con el MISMO `TorqueScaling` con que el análisis
lo decodifica, de modo que el round-trip es exacto.

Física simulada del par
-----------------------
  T(t) = T_medio
       + Σ_k  A_k · SDOF(k·f1, f_res, ζ) · sin(k·φ + θ_k)     (órdenes 1×,2×,…)
       + A_gm · sin(Z·φ)                                       (engrane, Z dientes)
       + ruido

donde φ es el ángulo 1× acumulado (continuo entre bloques, con velocidad
variable), f1 = rpm/60, y SDOF amplifica cada orden al cruzar la frecuencia
natural torsional `torsional_res_hz` durante un runup → esto produce el
Campbell / order-tracking torsional (el "money shot" del análisis).

Un canal keyphasor entrega un pulso once-per-rev (tacómetro) para order
tracking, igual que la fuente de vibración.

Techo de banda del equipo real: 500 Hz (-3 dB). El sim no impone ese filtro
(deja pasar lo que pongas); mantén los fenómenos < 500 Hz para realismo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from core.modal.acq_backend import ChannelConfig
from core.remote_monitoring.stream_source import (
    StreamConfig,
    StreamSource,
    is_keyphasor_channel,
)
from core.torsional.scaling import (
    ShaftGeometry,
    GageConfig,
    TorqueScaling,
)

# Una componente de orden: (orden, amplitud_EU, fase_grados)
#   orden 1.0 = síncrono (1×), 2.0 = 2×, 0.5 = sub-síncrono, etc.
OrderComponent = Tuple[float, float, float]


def make_torsional_channels(
    torque_bnc: int = 1,
    tach_bnc: int = 2,
    torque_name: str = "Torque",
    tach_name: str = "KPH",
    units: str = "nm",
) -> List[ChannelConfig]:
    """
    Construye los dos canales típicos de un ensayo torsional:
    un canal de torque (voltaje DC de la salida del RX10K) y un keyphasor.

    Reusa `ChannelConfig` (mismo modelo de canal que modal/monitoreo). El
    canal de torque va como `coupling="DC"`, `sensitivity_mv_per_eu=1000`
    (identidad V→V; la conversión a torque la hace `TorqueScaling`, no la
    sensibilidad lineal del canal).
    """
    return [
        ChannelConfig(bnc_port=torque_bnc, name=torque_name, coupling="DC",
                      sensitivity_mv_per_eu=1000.0, units=units, voltage_range=10.0),
        ChannelConfig(bnc_port=tach_bnc, name=tach_name, coupling="DC",
                      sensitivity_mv_per_eu=1000.0, units="pulse", voltage_range=10.0),
    ]


@dataclass
class TorsionalStreamConfig(StreamConfig):
    """
    Configuración de streaming torsional. Extiende `StreamConfig` (hereda
    sample_rate, canales, perfiles de velocidad runup/coastdown, keyphasor,
    tamaños de bloque/buffer) y añade los parámetros del par.

    Attributes:
        mean_torque: Par medio (estático) en unidades de `torque_units`.
        orders: Componentes de orden [(orden, amplitud_EU, fase_deg), …].
            Default: 1× y 2× (desbalance de par / desalineación torsional).
        gear_teeth: Nº de dientes → orden de engrane (GMF = Z·f1). 0 = sin engrane.
        gear_amp_eu: Amplitud de la componente de engrane [EU].
        torsional_res_hz: Frecuencia natural torsional [Hz]. >0 activa la
            amplificación SDOF de las órdenes al cruzarla (resonancia torsional).
        torsional_zeta: Amortiguamiento del modo torsional (ζ).
        torque_noise_rms_eu: Ruido RMS del par [EU].
        torque_units: "nm" o "ftlb".
        scaling: `TorqueScaling` usado para codificar EU→volts (y decodificar).
            Si None, se construye uno canónico (eje sólido 3", GF 2.0, GXMT 4000).
    """
    mean_torque: float = 0.0
    orders: Tuple[OrderComponent, ...] = ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    gear_teeth: int = 0
    gear_amp_eu: float = 0.0
    torsional_res_hz: float = 0.0
    torsional_zeta: float = 0.03
    torque_noise_rms_eu: float = 0.0
    torque_units: str = "nm"
    scaling: Optional[TorqueScaling] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scaling is None:
            self.scaling = TorqueScaling.from_geometry(
                ShaftGeometry(outer_diameter_in=3.0),
                GageConfig(gage_factor=2.0, transmitter_gain=4000),
                units=self.torque_units,
            )


class SimulatedTorsionalSource(StreamSource):
    """
    Fuente sintética de par torsional. Corre en cualquier plataforma (Mac/tests).

    Emite bloques (n_channels, block_samples) en **Volts**. El canal keyphasor
    da un pulso negativo once-per-rev; los demás canales llevan la señal de par
    codificada a volts vía `config.scaling`.

    La continuidad de fase entre bloques se garantiza con un cursor global de
    muestra y una fase 1× acumulada (sin saltos en las fronteras de bloque).
    """

    def __init__(self, config: TorsionalStreamConfig) -> None:
        super().__init__(config)
        self.config: TorsionalStreamConfig = config
        self._cursor = 0
        self._phase = 0.0
        self._rng: Optional[np.random.Generator] = None
        self._kph_idx = config.keyphasor_index()

    # --- ciclo de vida ---
    def start(self) -> None:
        self._cursor = 0
        self._phase = 0.0
        self._rng = np.random.default_rng(self.config.seed)
        self._running = True

    def stop(self) -> None:
        self._running = False

    def rewind(self) -> None:
        """Reinicia el reloj del rotor sin parar el stream (cambio de modo)."""
        self._cursor = 0
        self._phase = 0.0

    @staticmethod
    def _sdof_gain(freq: np.ndarray, fn: float, zeta: float) -> np.ndarray:
        """Amplificación SDOF |H| = 1/√((1-r²)²+(2ζr)²), r = freq/fn.

        Pico ≈ 1/(2ζ) en la resonancia; ~1 muy por debajo; →0 muy por encima.
        fn<=0 → ganancia plana (1)."""
        if fn <= 0:
            return np.ones_like(freq)
        r = freq / fn
        denom = np.sqrt((1.0 - r ** 2) ** 2 + (2.0 * zeta * r) ** 2)
        denom = np.where(denom < 1e-9, 1e-9, denom)
        return 1.0 / denom

    def read_block(self) -> np.ndarray:
        if not self._running or self._rng is None:
            raise RuntimeError("SimulatedTorsionalSource no está corriendo (llama start())")

        cfg = self.config
        n = cfg.block_samples
        fs = cfg.sample_rate_hz
        idx = np.arange(self._cursor, self._cursor + n)
        t = idx / fs

        # Velocidad instantánea → fase 1× acumulada (continua entre bloques)
        rpm = cfg.rpm_at(t)
        f1 = rpm / 60.0
        dphi = 2.0 * math.pi * f1 / fs
        phase = self._phase + np.cumsum(dphi)

        eu_per_volt = cfg.scaling.eu_per_volt

        # --- Señal de par en unidades de ingeniería ---
        torque_eu = np.full(n, float(cfg.mean_torque))
        for order, amp_eu, ph_deg in cfg.orders:
            if amp_eu == 0.0:
                continue
            gain = self._sdof_gain(order * f1, cfg.torsional_res_hz, cfg.torsional_zeta)
            torque_eu = torque_eu + amp_eu * gain * np.sin(order * phase + math.radians(ph_deg))
        if cfg.gear_teeth > 0 and cfg.gear_amp_eu != 0.0:
            torque_eu = torque_eu + cfg.gear_amp_eu * np.sin(cfg.gear_teeth * phase)
        if cfg.torque_noise_rms_eu > 0.0:
            torque_eu = torque_eu + cfg.torque_noise_rms_eu * self._rng.standard_normal(n)

        torque_volts = torque_eu / eu_per_volt

        out = np.empty((cfg.n_channels, n), dtype=float)
        for ci, ch in enumerate(cfg.channels):
            if ci == self._kph_idx or is_keyphasor_channel(ch, cfg.keyphasor_name):
                frac = np.mod(phase / (2.0 * math.pi), 1.0)   # once-per-rev
                out[ci] = np.where(frac < 0.02, -5.0, 0.0)
            else:
                # Ligera variación por canal si hubiese varios ejes.
                scale = 1.0 + 0.05 * (((ch.bnc_port or 1) - 1) % 4)
                out[ci] = torque_volts * scale

        self._cursor += n
        self._phase = float(phase[-1])
        return out
