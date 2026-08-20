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
    rpm: float = 3600.0                 # velocidad del rotor simulado (constante)
    defect: str = "none"                # none | unbalance | misalignment
    noise_rms: float = 0.02
    seed: int = 7

    # Perfil de velocidad (solo simulación) — habilita TRANSITORIOS (bode/cascade).
    #   constant         → rpm fija (comportamiento por defecto)
    #   runup            → rampa rpm_start → rpm_end en ramp_seconds
    #   coastdown        → rampa rpm_start → rpm_end (start alto, end bajo)
    #   runup_coastdown  → sube y baja (arranque + parada)
    speed_profile: str = "constant"
    rpm_start: float = 0.0
    rpm_end: float = 0.0
    ramp_seconds: float = 30.0
    # Velocidad crítica del rotor simulado (resonancia SDOF). Si > 0, la
    # respuesta 1X se amplifica al pasar por la crítica y la fase gira 0→180°,
    # produciendo un Bode físicamente correcto. 0 = sin resonancia (plano).
    sim_critical_rpm: float = 0.0
    sim_critical_rpm2: float = 0.0     # 2ª crítica opcional (2 resonancias)
    sim_zeta: float = 0.05

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

    def rpm_at(self, t: "np.ndarray") -> "np.ndarray":
        """Velocidad instantánea (rpm) en el tiempo t (segundos desde el inicio
        del stream). Perfil según speed_profile. Continuo — t es absoluto."""
        prof = self.speed_profile
        start = self.rpm_start if self.rpm_start > 0 else self.rpm
        end = self.rpm_end if self.rpm_end > 0 else self.rpm
        ramp = max(self.ramp_seconds, 1e-6)
        if prof == "constant" or (self.rpm_start <= 0 and self.rpm_end <= 0):
            return np.full_like(t, float(self.rpm))
        lo, hi = min(start, end), max(start, end)
        if prof in ("runup", "coastdown"):
            return np.clip(start + (end - start) * (t / ramp), lo, hi)
        if prof == "runup_coastdown":
            up = start + (end - start) * (t / ramp)
            down = end + (start - end) * ((t - ramp) / ramp)
            return np.clip(np.where(t <= ramp, up, down), lo, hi)
        return np.full_like(t, float(self.rpm))

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
        self._phase = 0.0   # fase 1X acumulada (rad) — continuidad con velocidad variable
        self._rng: Optional[np.random.Generator] = None
        # Offsets de fase 1X por canal. Para que la FORMA DEFLECTADA (unir los
        # keyphasor de cada cojinete) salga como una curva SUAVE de lado-libre a
        # lado-acople: la fase base avanza ~28° por cojinete y, dentro del par,
        # X va en cuadratura (+90°) respecto a Y → órbita elíptica bien formada.
        offs = []
        for ch in config.channels:
            label = (getattr(ch, "name", "") or "").upper()
            digits = "".join(c for c in label if c.isdigit())
            brg = int(digits) if digits else 0
            if "X" in label and "KPH" not in label:
                quad = math.pi / 2.0
            else:
                quad = 0.0
            theta = math.radians(28.0 * max(0, brg - 1))
            offs.append((theta + quad) % (2 * math.pi))
        self._phase_offsets = np.array(offs) if offs else np.zeros(config.n_channels)
        self._kph_idx = config.keyphasor_index()

    def start(self) -> None:
        self._cursor = 0
        self._phase = 0.0
        self._rng = np.random.default_rng(self.config.seed)
        self._running = True

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _sdof(f1: np.ndarray, fn: float, zeta: float):
        """Respuesta síncrona de rotor (SDOF): amplificación + fase 1X.

        Devuelve (A, phi) con A = r²/√((1-r²)²+(2ζr)²) y phi = atan2(2ζr, 1-r²)
        (0 en baja velocidad → 90° en la crítica → 180° por encima), donde
        r = f1/fn. Si fn<=0, respuesta plana (A=1, phi=0)."""
        if fn <= 0:
            return np.ones_like(f1), np.zeros_like(f1)
        r = f1 / fn
        denom = np.sqrt((1.0 - r ** 2) ** 2 + (2.0 * zeta * r) ** 2)
        denom = np.where(denom < 1e-9, 1e-9, denom)
        A = r ** 2 / denom
        phi = np.arctan2(2.0 * zeta * r, 1.0 - r ** 2)
        return A, phi

    def read_block(self) -> np.ndarray:
        if not self._running or self._rng is None:
            raise RuntimeError("SimulatedStreamSource no está corriendo (llama start())")

        cfg = self.config
        n = cfg.block_samples
        fs = cfg.sample_rate_hz
        idx = np.arange(self._cursor, self._cursor + n)
        t = idx / fs

        # Velocidad instantánea → fase 1X acumulada (continua entre bloques)
        rpm = cfg.rpm_at(t)
        f1 = rpm / 60.0
        dphi = 2.0 * math.pi * f1 / fs
        phase = self._phase + np.cumsum(dphi)   # ángulo 1X en rad

        # Amplificación síncrona por velocidad(es) crítica(s) (Bode físico).
        # Suma vectorial de una o dos resonancias SDOF; sin críticas → plano.
        crits = [c / 60.0 for c in (cfg.sim_critical_rpm, cfg.sim_critical_rpm2) if c > 0]
        if crits:
            z = np.zeros_like(f1, dtype=complex)
            for fn in crits:
                Ai, phii = self._sdof(f1, fn, cfg.sim_zeta)
                z = z + Ai * np.exp(1j * phii)
            A, phi_res = np.abs(z), np.angle(z)
        else:
            A, phi_res = np.ones_like(f1), np.zeros_like(f1)

        out = np.empty((cfg.n_channels, n), dtype=float)
        for ci, ch in enumerate(cfg.channels):
            if ci == self._kph_idx:
                # Pulso once-per-rev (~2% duty) desde la fase acumulada
                frac = np.mod(phase / (2.0 * math.pi), 1.0)
                out[ci] = np.where(frac < 0.02, -5.0, 0.0)
                continue

            base = 0.3 + 0.15 * (((ch.bnc_port or 1) - 1) % 4)
            ph = self._phase_offsets[ci]

            sig = base * A * np.sin(phase + ph + phi_res)       # 1X (amplificado)
            sig += 0.40 * base * np.sin(2.0 * phase + ph / 2)   # 2X
            sig += 0.15 * base * np.sin(0.5 * phase)            # 0.5X

            if cfg.defect == "unbalance":
                sig += 1.20 * base * A * np.sin(phase + ph + phi_res)  # boost 1X
            elif cfg.defect == "misalignment":
                sig += 0.90 * base * np.sin(2.0 * phase + ph)          # boost 2X
            elif cfg.defect == "oil_whirl" and cfg.sim_critical_rpm > 0:
                # Inestabilidad de película: oil WHIRL (~0.45X, sigue el rpm) que
                # al pasar ~2.2× la 1ª crítica se ENGANCHA a la natural = oil WHIP
                # (frecuencia fija). Aparece por encima de ~1.8× la crítica.
                fn1 = cfg.sim_critical_rpm / 60.0
                aw = 0.7 * base
                whirl = aw * np.sin(0.45 * phase + ph)
                whip = 1.4 * aw * np.sin(2.0 * math.pi * fn1 * (idx / fs))
                on = rpm > 1.8 * cfg.sim_critical_rpm
                lock = rpm > 2.2 * cfg.sim_critical_rpm
                sig += np.where(on, np.where(lock, whip, whirl), 0.0)

            sig += cfg.noise_rms * self._rng.standard_normal(n)
            out[ci] = sig

        self._cursor += n
        self._phase = float(phase[-1])
        return out
