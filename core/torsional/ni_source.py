"""
core/torsional/ni_source.py — Adquisición REAL con NI 9229 (P1)
================================================================

Estructura de adquisición por hardware para el ensayo torsional. Espejo en
forma de `SimulatedTorsionalSource` (mismos métodos `start/read_block/stop`,
mismos bloques en **Volts**), pero leyendo de una tarjeta **NI 9229** (BNC,
±60 V, 24 bit, muestreo SIMULTÁNEO en una sola tarea AI con reloj compartido).

Cableado (ver memoria de hardware):
  · Par:       salida ±10 V del RX10K Binsfeld (bananas) → BNC AI0.
  · Keyphasor: el tacómetro → BNC AI1 (NO pasa por el OpDAQ azul).

Dos sensores de keyphasor soportados (`KeyphasorSensor`):
  1. **Bently Nevada 3300 XL 8 mm + Proximitor**: salida DC negativa; el keyway
     da un pulso NEGATIVO → flanco de BAJADA, threshold negativo.
  2. **Foto-tacómetro con cinta reflectiva**: pulso POSITIVO al ver la cinta →
     flanco de SUBIDA, threshold positivo. `pulses_per_rev` = nº de cintas.

`nidaqmx` es un import perezoso: en Mac/tests el módulo se importa igual; solo
`start()` requiere el driver. NUNCA cae a simulado en silencio — si el hardware
no está, `start()` lanza `RuntimeError` con un mensaje claro (lo maneja la UI).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from core.torsional.analysis import keyphasor_to_rpm


# =====================================================================
# Sensor de keyphasor (tacómetro) — dos tipos, con presets
# =====================================================================
@dataclass
class KeyphasorSensor:
    """
    Describe el sensor de velocidad y cómo detectar su pulso once/n-per-rev.

    kind:            id del sensor ("proximitor_3300xl" | "phototach_reflective").
    label:           nombre para UI/reporte.
    edge:            "falling" (proximidad) | "rising" (foto-tacómetro).
    trigger_level_v: nivel de disparo [V] para detectar el pulso.
    pulses_per_rev:  keyways (proximidad) o cintas reflectivas (foto-tacómetro).
    coupling:        acople del canal ("DC" para ambos).
    note:            recordatorio de cableado/alimentación para el operador.
    """
    kind: str
    label: str
    edge: str
    trigger_level_v: float
    pulses_per_rev: int = 1
    coupling: str = "DC"
    note: str = ""

    @staticmethod
    def bently_3300xl_8mm(keyways: int = 1, gap_bias_v: float = -10.0) -> "KeyphasorSensor":
        """Sonda de proximidad Bently Nevada 3300 XL 8 mm + Proximitor.
        Salida ~-24 V alim., señal DC negativa (200 mV/mil); el keyway se ve como
        un pulso MÁS negativo → flanco de bajada. threshold ~ 3 V bajo el bias de gap."""
        return KeyphasorSensor(
            kind="proximitor_3300xl", label="Bently 3300 XL 8mm + Proximitor",
            edge="falling", trigger_level_v=gap_bias_v - 3.0,
            pulses_per_rev=max(1, int(keyways)), coupling="DC",
            note="Proximitor -24 VDC; keyway/notch once-per-rev; señal DC negativa a AI del 9229.")

    @staticmethod
    def phototach_reflective(strips: int = 1, level_v: float = 2.5) -> "KeyphasorSensor":
        """Foto-tacómetro que lee cinta reflectiva. Pulso POSITIVO (0→+V) al pasar
        la cinta → flanco de subida. pulses_per_rev = nº de cintas en el eje."""
        return KeyphasorSensor(
            kind="phototach_reflective", label="Photo-tach (reflective tape)",
            edge="rising", trigger_level_v=abs(level_v),
            pulses_per_rev=max(1, int(strips)), coupling="DC",
            note="Alimentar el foto-tacómetro; pulso positivo por cinta reflectiva a AI del 9229.")

    @staticmethod
    def simulated() -> "KeyphasorSensor":
        """Keyphasor del simulador (pulso negativo once-per-rev)."""
        return KeyphasorSensor(kind="sim", label="Simulated keyphasor",
                               edge="falling", trigger_level_v=-1.0, pulses_per_rev=1)


# Registro de sensores disponibles (para poblar selectores de UI).
KEYPHASOR_SENSORS = {
    "proximitor_3300xl": KeyphasorSensor.bently_3300xl_8mm,
    "phototach_reflective": KeyphasorSensor.phototach_reflective,
    "sim": KeyphasorSensor.simulated,
}


def rpm_from_keyphasor(keyphasor: np.ndarray, fs: float, sensor: KeyphasorSensor
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """rpm instantánea usando el flanco/umbral/ppr del sensor elegido."""
    return keyphasor_to_rpm(keyphasor, fs, threshold=sensor.trigger_level_v,
                            pulses_per_rev=sensor.pulses_per_rev, edge=sensor.edge)


# =====================================================================
# Configuración + fuente NI real
# =====================================================================
@dataclass
class NITorsionalConfig:
    """Configuración de adquisición NI 9229 para el ensayo torsional.

    device:         nombre del módulo NI en el driver (p.ej. "cDAQ1Mod1").
    torque_ai:      índice AI del canal de par (0 = AI0).
    kph_ai:         índice AI del canal de keyphasor (1 = AI1).
    sample_rate_hz: fs (≥1280 para 500 Hz de ancho de banda del RX10K).
    block_seconds:  tamaño de bloque de lectura.
    voltage_range:  ±rango de entrada [V] (10 para el ±10 V del RX10K).
    keyphasor:      sensor de keyphasor (proximidad / foto-tacómetro).
    """
    device: str = "cDAQ1Mod1"
    torque_ai: int = 0
    kph_ai: int = 1
    sample_rate_hz: float = 2560.0
    block_seconds: float = 0.1
    voltage_range: float = 10.0
    keyphasor: KeyphasorSensor = field(default_factory=KeyphasorSensor.phototach_reflective)

    @property
    def n_channels(self) -> int:
        return 2

    def keyphasor_index(self) -> int:
        return 1     # orden fijo: [Torque(AI0), Keyphasor(AI1)]

    @property
    def block_samples(self) -> int:
        return max(1, int(round(self.sample_rate_hz * self.block_seconds)))


def nidaqmx_available() -> bool:
    """True si el driver NI-DAQmx está instalado (Windows con la maleta)."""
    try:
        import nidaqmx  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


class NITorsionalSource:
    """
    Fuente REAL NI 9229. Misma interfaz que `SimulatedTorsionalSource`:
    `start()`, `read_block() -> (n_channels, block) en Volts`, `stop()`.

    Una sola tarea AI con reloj compartido → par y keyphasor muestreados
    SIMULTÁNEAMENTE (regla de oro rotodinámica). Torque en AI0, keyphasor en AI1.
    """

    def __init__(self, config: NITorsionalConfig) -> None:
        self.config = config
        self._task = None
        self._running = False

    def _phys(self, ai: int) -> str:
        return f"{self.config.device}/ai{ai}"

    def start(self) -> None:
        try:
            import nidaqmx
            from nidaqmx.constants import AcquisitionType, TerminalConfiguration
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "NI-DAQmx no está disponible en este equipo. La adquisición real "
                "requiere Windows con el driver NI-DAQmx y la NI 9229 conectada."
            ) from exc
        cfg = self.config
        vmax = float(cfg.voltage_range); vmin = -vmax
        task = nidaqmx.Task()
        try:
            # Canal de PAR (AI0) — DC, ±10 V (salida del RX10K).
            task.ai_channels.add_ai_voltage_chan(
                self._phys(cfg.torque_ai), name_to_assign_to_channel="Torque",
                terminal_config=TerminalConfiguration.DIFF, min_val=vmin, max_val=vmax)
            # Canal KEYPHASOR (AI1) — DC (proximidad negativa o pulso positivo).
            task.ai_channels.add_ai_voltage_chan(
                self._phys(cfg.kph_ai), name_to_assign_to_channel="KPH",
                terminal_config=TerminalConfiguration.DIFF, min_val=vmin, max_val=vmax)
            # Reloj compartido → muestreo SIMULTÁNEO, continuo.
            task.timing.cfg_samp_clk_timing(
                rate=float(cfg.sample_rate_hz), sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=max(cfg.block_samples * 8, 4096))
            task.start()
        except Exception:
            try:
                task.close()
            except Exception:  # noqa: BLE001
                pass
            raise
        self._task = task
        self._running = True

    def read_block(self) -> np.ndarray:
        if not self._running or self._task is None:
            raise RuntimeError("NITorsionalSource no iniciado (llama a start()).")
        n = self.config.block_samples
        data = self._task.read(number_of_samples_per_channel=n)
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:                      # una sola muestra/canal degenerada
            arr = arr.reshape(self.config.n_channels, -1)
        return arr                              # (n_channels, block) en Volts

    def stop(self) -> None:
        self._running = False
        if self._task is not None:
            try:
                self._task.stop(); self._task.close()
            except Exception:  # noqa: BLE001
                pass
            self._task = None
