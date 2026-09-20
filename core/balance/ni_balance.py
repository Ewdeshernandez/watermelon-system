"""
core/balance/ni_balance.py — Adquisición NI para BALANCEO (9229 / 9234)
======================================================================

Extrae de la maleta NI lo que el balanceo necesita:
  · **Vibración directa** (la señal cruda del sensor, en su unidad).
  · **1× filtrado**: amplitud + FASE referenciada al **keyphasor** (lo esencial
    del balanceo por coeficientes de influencia).
  · **Velocidad** (mm/s): para vibración de carcasa/absoluta con acelerómetro
    (se integra la aceleración → velocidad, ISO 20816).

Reglas fijas (acordadas):
  · **El keyphasor SIEMPRE es el canal 0 (AI0)** — de ahí parte todo.
  · Keyphasor = **Bently Nevada 3300 XL 8 mm + Proximitor** (pulso NEGATIVO, DC,
    requiere −24 VDC) **o** **foto-tacómetro con cinta reflectiva** (pulso
    POSITIVO). Se elige uno; cambia flanco/umbral/alimentación.

Sensores de vibración:
  · **Proximidad (NI 9229, DC)** → desplazamiento µm pk-pk (relativo de eje).
    El proximitor requiere −24 VDC externos.
  · **Acelerómetro IEPE (NI 9234, AC, 2 mA)** → se integra a **velocidad mm/s**.

Las funciones de extracción son numpy PURO (testeables sin hardware). La fuente
`NIBalanceSource` usa nidaqmx perezoso y NUNCA simula en silencio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from core.torsional.ni_source import KeyphasorSensor, rpm_from_keyphasor, nidaqmx_available


# =====================================================================
# 1× filtrado: amplitud + fase referenciada al keyphasor
# =====================================================================
def extract_1x(vibration: np.ndarray, keyphasor: np.ndarray, fs: float,
               sensor: KeyphasorSensor, to_pp: bool = True
               ) -> Tuple[float, float, float]:
    """
    Componente 1× (síncrona) de la vibración, con FASE referenciada al keyphasor.

    Método: se detecta el pulso del keyphasor (una vez por vuelta) → frecuencia
    de giro f1. Se proyecta la vibración sobre el fundamental con t=0 en el pulso
    del keyphasor → la fase queda referida al keyphasor (como Bently/IRD).

    Convención: `phase_lag_deg` = retardo del pico 1× respecto al keyphasor,
    medido CONTRA el sentido de giro (0–360°). Para el coeficiente de influencia
    la convención se cancela mientras V0 y Vt usen la MISMA (aquí sí).

    Returns:
        (amplitud_1x, phase_lag_deg, rpm). amplitud en pk-pk si to_pp=True
        (2·pico), o en 0-pico si False. Ceros si no hay ≥2 pulsos.
    """
    v = np.asarray(vibration, dtype=float)
    kph = np.asarray(keyphasor, dtype=float)
    if v.size < 4:
        return 0.0, 0.0, 0.0
    # Flancos del keyphasor (una vez por vuelta / por pulso).
    if sensor.edge == "rising":
        above = kph > sensor.trigger_level_v
        edges = np.where(above[1:] & ~above[:-1])[0] + 1
    else:
        below = kph < sensor.trigger_level_v
        edges = np.where(below[1:] & ~below[:-1])[0] + 1
    if edges.size < 2:
        return 0.0, 0.0, 0.0
    # f1 preciso del SPAN primer↔último flanco (promedia el jitter de muestreo).
    ppr = max(1, int(sensor.pulses_per_rev))
    n_rev = (edges.size - 1) / ppr
    span_s = (edges[-1] - edges[0]) / fs
    if span_s <= 0:
        return 0.0, 0.0, 0.0
    f1 = n_rev / span_s
    rpm = f1 * 60.0
    if f1 <= 0:
        return 0.0, 0.0, 0.0
    t0 = edges[0] / fs                               # primer pulso = origen de fase
    n = v.size
    t = np.arange(n) / fs - t0
    vac = v - float(np.mean(v))                      # quita DC/gap
    # Proyección DFT al fundamental (ventana Hann, amplitud-correcta).
    w = np.hanning(n); sw = np.sum(w)
    if sw <= 0:
        return 0.0, 0.0, rpm
    proj = np.sum(vac * w * np.exp(-1j * 2.0 * np.pi * f1 * t))
    amp0pk = 2.0 * abs(proj) / sw                    # 0-pico
    phase_lag = (-np.degrees(np.angle(proj))) % 360.0
    amp = amp0pk * (2.0 if to_pp else 1.0)
    return float(amp), float(phase_lag), rpm


# =====================================================================
# Aceleración → velocidad (ISO 20816)
# =====================================================================
def accel_to_velocity(accel_ms2: np.ndarray, fs: float, hp_hz: float = 2.0) -> np.ndarray:
    """Integra aceleración [m/s²] → velocidad [mm/s] en el dominio de frecuencia
    (V(f)=A(f)/(j2πf)), con paso-alto a `hp_hz` para quitar la deriva de la
    integración. Devuelve la serie de velocidad en mm/s."""
    a = np.asarray(accel_ms2, dtype=float)
    n = a.size
    if n < 4:
        return np.zeros_like(a)
    A = np.fft.rfft(a - float(np.mean(a)))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    jw = 1j * 2.0 * np.pi * f
    V = np.zeros_like(A)
    m = f >= max(hp_hz, 1e-6)
    V[m] = A[m] / jw[m]                               # integración
    v = np.fft.irfft(V, n=n) * 1000.0                 # m/s → mm/s
    return v


def velocity_rms_mm_s(accel_ms2: np.ndarray, fs: float, hp_hz: float = 2.0) -> float:
    """Velocidad RMS global [mm/s] (banda ISO 20816) desde la aceleración [m/s²]."""
    v = accel_to_velocity(accel_ms2, fs, hp_hz)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else 0.0


def one_x_accel_to_velocity(accel_amp: float, phase_deg: float, rpm: float
                            ) -> Tuple[float, float]:
    """Convierte una componente 1× de ACELERACIÓN (amp [m/s²], fase) a VELOCIDAD
    (mm/s, fase). v = a/ω, con desfase −90°.  Devuelve (amp_mm_s, phase_deg)."""
    f1 = rpm / 60.0
    if f1 <= 0:
        return 0.0, phase_deg
    w = 2.0 * np.pi * f1
    amp_mm_s = (float(accel_amp) / w) * 1000.0        # m/s → mm/s
    return amp_mm_s, (float(phase_deg) - 90.0) % 360.0


# =====================================================================
# Canal de vibración + configuración de adquisición
# =====================================================================
@dataclass
class VibChannel:
    """Un canal de vibración para balanceo.

    kind: "proximity_9229" (desplazamiento µm, DC, prox −24V) |
          "accel_9234" (acelerómetro IEPE, se integra a velocidad mm/s).
    label: nombre del punto/plano (p.ej. "1Y", "A", "B").
    sensitivity_mv_per_unit: mV por unidad física (200 mV/mil prox; 100 mV/g accel).
    """
    label: str
    kind: str = "accel_9234"
    device: str = "cDAQ1Mod2"
    ai: int = 0
    sensitivity_mv_per_unit: float = 100.0

    @property
    def coupling(self) -> str:
        return "IEPE" if self.kind == "accel_9234" else "DC"

    @property
    def unit(self) -> str:
        return "mm/s RMS" if self.kind == "accel_9234" else "µm pk-pk"


@dataclass
class NIBalanceConfig:
    """Adquisición NI para balanceo. **AI0 del módulo del keyphasor = keyphasor.**"""
    keyphasor: KeyphasorSensor = field(default_factory=KeyphasorSensor.phototach_reflective)
    kph_device: str = "cDAQ1Mod1"
    kph_ai: int = 0                                   # SIEMPRE 0 (de aquí parte todo)
    vib_channels: List[VibChannel] = field(default_factory=list)
    sample_rate_hz: float = 5120.0
    block_seconds: float = 0.5
    voltage_range: float = 10.0

    @property
    def block_samples(self) -> int:
        return max(1, int(round(self.sample_rate_hz * self.block_seconds)))


def keyphasor_power_note(sensor: KeyphasorSensor) -> str:
    """Recordatorio de ALIMENTACIÓN según el sensor de keyphasor elegido."""
    if sensor.kind == "proximitor_3300xl":
        return ("Bently 3300 XL 8mm: el Proximitor requiere alimentación −24 VDC. "
                "Señal DC negativa; keyway = pulso negativo (flanco de bajada).")
    if sensor.kind == "phototach_reflective":
        return ("Foto-tacómetro: alimentar el sensor (5–24 VDC según modelo). "
                "Pulso positivo por cinta reflectiva (flanco de subida).")
    return "Keyphasor simulado."


class NIBalanceSource:
    """
    Fuente REAL NI para balanceo. Una tarea AI con reloj compartido; **canal 0 =
    keyphasor** y luego los canales de vibración. `read_block()` → (n_ch, block)
    en Volts (fila 0 = keyphasor). nidaqmx perezoso; NUNCA simula en silencio.
    """

    def __init__(self, config: NIBalanceConfig) -> None:
        self.config = config
        self._task = None
        self._running = False

    @property
    def n_channels(self) -> int:
        return 1 + len(self.config.vib_channels)

    def keyphasor_index(self) -> int:
        return 0

    def start(self) -> None:
        try:
            import nidaqmx
            from nidaqmx.constants import AcquisitionType, TerminalConfiguration
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "NI-DAQmx no está disponible. La adquisición real requiere Windows "
                "con el driver NI-DAQmx y la maleta (9229/9234) conectada."
            ) from exc
        cfg = self.config
        vmax = float(cfg.voltage_range); vmin = -vmax
        task = nidaqmx.Task()
        try:
            # Canal 0 = KEYPHASOR (DC). Siempre primero.
            task.ai_channels.add_ai_voltage_chan(
                f"{cfg.kph_device}/ai{cfg.kph_ai}", name_to_assign_to_channel="KPH",
                terminal_config=TerminalConfiguration.DIFF, min_val=vmin, max_val=vmax)
            # Canales de vibración.
            for ch in cfg.vib_channels:
                if ch.kind == "accel_9234":
                    from nidaqmx.constants import ExcitationSource
                    task.ai_channels.add_ai_accel_chan(
                        f"{ch.device}/ai{ch.ai}", name_to_assign_to_channel=ch.label,
                        sensitivity=ch.sensitivity_mv_per_unit,
                        current_excit_source=ExcitationSource.INTERNAL, current_excit_val=0.002)
                else:  # proximidad → voltaje DC (µm por sensibilidad, se escala en SW)
                    task.ai_channels.add_ai_voltage_chan(
                        f"{ch.device}/ai{ch.ai}", name_to_assign_to_channel=ch.label,
                        terminal_config=TerminalConfiguration.DIFF, min_val=vmin, max_val=vmax)
            task.timing.cfg_samp_clk_timing(
                rate=float(cfg.sample_rate_hz), sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=max(cfg.block_samples * 8, 8192))
            task.start()
        except Exception:
            try: task.close()
            except Exception: pass  # noqa: BLE001
            raise
        self._task = task
        self._running = True

    def read_block(self) -> np.ndarray:
        if not self._running or self._task is None:
            raise RuntimeError("NIBalanceSource no iniciado (llama a start()).")
        data = self._task.read(number_of_samples_per_channel=self.config.block_samples)
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(self.n_channels, -1)
        return arr

    def stop(self) -> None:
        self._running = False
        if self._task is not None:
            try:
                self._task.stop(); self._task.close()
            except Exception: pass  # noqa: BLE001
            self._task = None
