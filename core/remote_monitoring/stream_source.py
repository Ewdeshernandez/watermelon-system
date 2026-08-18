"""
core/remote_monitoring/stream_source.py — Fuentes de streaming continuo
=======================================================================

Una `StreamSource` entrega bloques de muestras multicanal SIN FIN. El
ACQ Agent la consume en loop: read_block() → RingBuffer → materialize.

Dos implementaciones:
  · SimulatedStreamSource — data sintética de rotor (1X/2X/0.5X + ruido +
    keyphasor once-per-rev). Corre en CUALQUIER plataforma (Mac dev).
  · NIStreamSource — nidaqmx en modo continuo (Windows/Linux + driver).
    Import lazy; no rompe en Mac/Cloud si no está el driver. (Se implementa
    cuando conectemos el hardware de campo.)

La convención de canales reusa `core.modal.acq_backend.ChannelConfig`
(bnc_port 1..32, name, coupling, sensitivity, units) para no duplicar el
modelo de canal entre el módulo modal y este.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.modal.acq_backend import ChannelConfig, _nearest_valid_rate


# Nombres que, por convención, identifican el canal de referencia de fase
# (keyphasor / tacómetro). Case-insensitive, substring.
_KEYPHASOR_TOKENS = ("kph", "keyphasor", "keyph", "tach", "tacho", "trigger", "kp")


def is_keyphasor_channel(ch: ChannelConfig, explicit_name: Optional[str] = None) -> bool:
    """True si el canal es el keyphasor (referencia de fase 1X).

    Se identifica por nombre explícito (StreamConfig.keyphasor_name) o por
    tokens convencionales en el nombre del canal.
    """
    nm = (ch.name or "").strip().lower()
    if explicit_name and nm == explicit_name.strip().lower():
        return True
    return any(tok in nm for tok in _KEYPHASOR_TOKENS)


@dataclass
class StreamConfig:
    """Configuración de una sesión de streaming continuo.

    A diferencia de `AcquisitionConfig` (modal, tiene duration_s), acá NO
    hay duración: el stream corre hasta que el Agent lo detiene.
    """
    sample_rate_hz: float
    channels: List[ChannelConfig] = field(default_factory=list)

    # Tamaño del bloque que entrega read_block(), en segundos. 0.25s es un
    # buen balance: refresco fluido sin saturar CPU con reruns.
    block_seconds: float = 0.25

    # Profundidad del buffer rodante en RAM, en segundos. Debe cubrir la
    # ventana de análisis más larga (waterfall/cascade necesitan más).
    buffer_seconds: float = 10.0

    # Nombre del canal keyphasor (si None, se autodetecta por tokens).
    keyphasor_name: Optional[str] = None

    chassis_name: str = "cDAQ1"

    # --- Solo simulación ---
    rpm: float = 3600.0                 # velocidad del rotor simulado
    defect: str = "none"                # none | unbalance | misalignment
    noise_rms: float = 0.02
    seed: int = 7

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz debe ser > 0")
        if not self.channels:
            raise ValueError("StreamConfig requiere al menos 1 canal")
        if self.block_seconds <= 0:
            raise ValueError("block_seconds debe ser > 0")
        if self.buffer_seconds < self.block_seconds:
            raise ValueError("buffer_seconds no puede ser menor que block_seconds")

    # Derivados ----------------------------------------------------------
    @property
    def block_samples(self) -> int:
        return max(1, int(round(self.block_seconds * self.sample_rate_hz)))

    @property
    def buffer_samples(self) -> int:
        return max(self.block_samples, int(round(self.buffer_seconds * self.sample_rate_hz)))

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def keyphasor_index(self) -> Optional[int]:
        """Índice del canal keyphasor en la lista, o None si no hay."""
        for i, ch in enumerate(self.channels):
            if is_keyphasor_channel(ch, self.keyphasor_name):
                return i
        return None


class StreamSource:
    """Interfaz base de una fuente de streaming continuo."""

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self._running = False

    # --- ciclo de vida ---
    def start(self) -> None:
        raise NotImplementedError

    def read_block(self) -> np.ndarray:
        """Devuelve un bloque (n_channels, block_samples) de muestras en Volts.

        Bloqueante en hardware real (espera a que el DAQ llene el bloque).
        """
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    # --- introspección ---
    @property
    def sample_rate_hz(self) -> float:
        return self.config.sample_rate_hz

    @property
    def channels(self) -> List[ChannelConfig]:
        return self.config.channels

    def is_running(self) -> bool:
        return self._running

    def __enter__(self) -> "StreamSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class SimulatedStreamSource(StreamSource):
    """Fuente sintética de rotor — corre sin hardware (Mac dev / tests).

    Genera, por canal de vibración:
      1X   a f1 = rpm/60         (desbalance dominante)
      2X   a 2·f1                (desalineación)
      0.5X a f1/2                (whirl/rub, sabor sub-síncrono)
      + ruido banda ancha.

    Cada canal tiene un offset de fase distinto (múltiplos de 90°) para que
    los pares X/Y produzcan órbitas reales y la fase 1X sea significativa.

    El canal keyphasor entrega un pulso negativo once-per-rev.

    La continuidad entre bloques se garantiza con un cursor de muestra
    global (`_cursor`): no hay saltos de fase en las fronteras de bloque.
    """

    def __init__(self, config: StreamConfig) -> None:
        super().__init__(config)
        self._cursor = 0
        self._rng: Optional[np.random.Generator] = None
        self._phase_offsets = np.array(
            [((i * math.pi / 2.0) % (2 * math.pi)) for i in range(config.n_channels)]
        )
        self._kph_idx = config.keyphasor_index()

    def start(self) -> None:
        self._cursor = 0
        self._rng = np.random.default_rng(self.config.seed)
        self._running = True

    def stop(self) -> None:
        self._running = False

    def _keyphasor(self, t: np.ndarray, f1: float) -> np.ndarray:
        """Pulso negativo once-per-rev (~2% duty), como un keyphasor real."""
        period = 1.0 / f1
        phase = np.mod(t, period) / period
        return np.where(phase < 0.02, -5.0, 0.0)

    def read_block(self) -> np.ndarray:
        if not self._running or self._rng is None:
            raise RuntimeError("SimulatedStreamSource no está corriendo (llama start())")

        cfg = self.config
        n = cfg.block_samples
        fs = cfg.sample_rate_hz
        idx = np.arange(self._cursor, self._cursor + n)
        t = idx / fs
        f1 = cfg.rpm / 60.0

        out = np.empty((cfg.n_channels, n), dtype=float)
        for ci, ch in enumerate(cfg.channels):
            if ci == self._kph_idx:
                out[ci] = self._keyphasor(t, f1)
                continue

            base = 0.3 + 0.15 * (((ch.bnc_port or 1) - 1) % 4)
            ph = self._phase_offsets[ci]

            sig = base * np.sin(2 * math.pi * f1 * t + ph)              # 1X
            sig += 0.40 * base * np.sin(2 * math.pi * 2 * f1 * t + ph / 2)  # 2X
            sig += 0.15 * base * np.sin(2 * math.pi * 0.5 * f1 * t)     # 0.5X

            if cfg.defect == "unbalance":
                sig += 1.20 * base * np.sin(2 * math.pi * f1 * t + ph)      # boost 1X
            elif cfg.defect == "misalignment":
                sig += 0.90 * base * np.sin(2 * math.pi * 2 * f1 * t + ph)  # boost 2X

            sig += cfg.noise_rms * self._rng.standard_normal(n)
            out[ci] = sig

        self._cursor += n
        return out
