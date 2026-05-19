"""
core/modal/ni_daq.py — Adquisición con maleta cDAQ-9178 + NI-9234
==================================================================

Wrapper sobre `nidaqmx` (driver oficial NI Python) para capturar datos
de una maleta con chasis NI cDAQ-9178 (8 slots USB 2.0) poblada con
hasta 8 módulos NI-9234 → **32 canales simultáneos** numerados como
puertos BNC 1..32 en el frente de la maleta.

Hardware soportado (v3.31.201+)
-------------------------------
· Chasis: NI cDAQ-9178 (8 slots, USB)
· Módulos: 1 a 8× NI-9234 (4 ch IEPE/AC/DC c/u, 24-bit, ±5V)
· Total: hasta 32 canales simultáneos muestreados
· Sincronización: sample clock compartido del chasis (auto)
· Cumple ATEX Ex II 3G / UL Class I Div 2 → apto rotating equipment

Naming convention NI-DAQmx
--------------------------
El driver NI nombra los canales físicos como `{chassis}Mod{slot}/ai{idx}`:
  cDAQ1Mod1/ai0..ai3   (slot 1 → BNC 1..4)
  cDAQ1Mod2/ai0..ai3   (slot 2 → BNC 5..8)
  ...
  cDAQ1Mod8/ai0..ai3   (slot 8 → BNC 29..32)

El operador piensa en **BNC port (1..32)** — el número impreso en el
frente de la maleta. La conversión a (slot, channel_index) es interna:
  slot  = (bnc_port - 1) // 4 + 1   →  1..8
  idx   = (bnc_port - 1) % 4         →  0..3

Modo EMA — Impact Hammer Test (ISO 7626-5)
-------------------------------------------
Captura sincronizada de impacto + respuesta:
  · 1 canal martillo (trigger)
  · N canales acelerómetros respuesta (1..31)
  · Trigger: por nivel en canal de fuerza (e.g. > 0.5 N)
  · Duración: 1-2 segundos típico
  · Promediado: 5-10 impactos para reducir ruido aleatorio

Modo OMA — Operational Modal Analysis (ISO 20816)
--------------------------------------------------
Captura continua durante operación normal:
  · Hasta 32 canales sincronizados (acelerómetros / proximidad / mix)
  · Sin trigger — adquisición continua streaming
  · Duración: 60-300+ segundos a velocidad constante
  · Sample rate: 5-10 kHz (configurable hasta 51.2 kHz)
  · TDMS escrito **chunk-por-chunk** (RAM constante ~5 MB para evitar
    OOM en captura 32ch × 300s × 5120 Hz que serían ~390 MB en RAM)

Modo Simulated (para development sin hardware)
-----------------------------------------------
Genera data sintética que imita lo que devolvería la maleta real. Útil
para probar el pipeline modal end-to-end sin chasis conectado. Activable
con `AcquisitionConfig.mode = "simulated_ema"` o `"simulated_oma"`
o flag `--simulated` en el companion.

NI-9234 specs (por módulo)
--------------------------
· 4 canales analógicos simultáneos
· 24-bit resolución
· Sample rate: 1.652 kHz to 51.2 kHz (16 valores discretos)
· Built-in IEPE excitation (configurable per canal)
· Rango: ±5 V

Sensor coupling
---------------
· IEPE (acelerómetros Wilcoxon 100 mV/g): NI-9234 alimenta excitación 2 mA
· AC con bias (Bently proximity 200 mV/mil): requiere PS externa -24 VDC,
  conexión BNC → NI-9234 en modo AC coupled
· DC: rara vez usado en vibración

Dependencias
------------
nidaqmx — Driver Python oficial NI (requiere NI-DAQmx driver instalado en
laptop de captura). Import lazy — el módulo se importa OK sin nidaqmx
disponible; solo falla al llamar las funciones de captura real.

  pip install nidaqmx npTDMS

Marco normativo
---------------
ISO 7626-5 §6 — Configuración de canales para impact testing
ISO 7626-5 §7 — Adquisición sincronizada input/output
ISO 20816-1 §5.3 — Requisitos de instrumentación para mediciones operacionales
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import math
import time

# Lazy imports: nidaqmx y nptdms se importan dentro de las funciones que
# los necesitan. Esto permite que el módulo se importe OK en Streamlit
# Cloud (donde NO está el driver NI-DAQmx) y solo falle al intentar
# capturar real.

# NI-9234 valid sample rates (Hz) — la tarjeta solo acepta valores
# discretos por su sigma-delta ADC. Si el usuario pide otro, se redondea
# al más cercano permitido.
_NI9234_VALID_RATES = [
    1652, 2000, 2048, 2500, 3200, 4000, 4096, 5120,
    6400, 8000, 8192, 10240, 12800, 16000, 16384, 20480,
    25600, 32000, 32768, 40960, 51200,
]


@dataclass
class ChannelConfig:
    """
    Configuración de un canal del NI-9234 dentro de la maleta cDAQ-9178.

    Identificación del canal físico
    -------------------------------
    Hay dos maneras de identificar un canal y son mutuamente convertibles:

    1. **bnc_port (1..32)** — el número impreso en el frente de la maleta.
       Es lo que el operador ve. RECOMENDADO.

    2. **module_slot (1..8) + channel_index (0..3)** — direccionamiento
       NI-DAQmx nativo. El driver lo necesita así internamente.

    Si pasas `bnc_port`, los otros dos se computan auto. Si pasas
    `channel_index` solo (sin bnc_port) — modo legacy v3.31.200-, asume
    `module_slot=1` (backward compat para single-module 4-canal).

    Ejemplo:
        ChannelConfig(bnc_port=5, name="1YA", coupling="IEPE",
                      sensitivity_mv_per_eu=100.0, units="g")
        # → module_slot=2, channel_index=0 (Mod2/ai0)

        ChannelConfig(bnc_port=32, name="2YV", coupling="IEPE",
                      sensitivity_mv_per_eu=100.0)
        # → module_slot=8, channel_index=3 (Mod8/ai3)
    """
    name: str           # Etiqueta del sensor (e.g. "1YA", "Hammer")
    coupling: str       # "IEPE", "AC", "DC"
    sensitivity_mv_per_eu: float  # 100.0 para Wilcoxon, 200.0 para Bently

    # Identificación del canal físico — pasa bnc_port o channel_index
    bnc_port: Optional[int] = None      # 1..32 (RECOMENDADO)
    channel_index: int = 0              # 0..3 dentro del módulo (legacy)
    module_slot: int = 1                # 1..8 dentro del chasis

    units: str = "g"    # "g", "mil", "N"
    voltage_range: float = 5.0  # ±V

    def __post_init__(self) -> None:
        """Normaliza bnc_port ↔ (module_slot, channel_index) post-init."""
        if self.bnc_port is not None:
            if not (1 <= self.bnc_port <= 32):
                raise ValueError(
                    f"bnc_port={self.bnc_port} fuera de rango [1..32]. "
                    f"Maleta cDAQ-9178 + 8× NI-9234 soporta máximo 32 canales."
                )
            # Computa slot e idx desde bnc_port
            self.module_slot = (self.bnc_port - 1) // 4 + 1
            self.channel_index = (self.bnc_port - 1) % 4
        else:
            # Modo legacy: solo channel_index dado → asume Mod1
            if not (0 <= self.channel_index <= 3):
                raise ValueError(
                    f"channel_index={self.channel_index} fuera de rango [0..3]."
                )
            if not (1 <= self.module_slot <= 8):
                raise ValueError(
                    f"module_slot={self.module_slot} fuera de rango [1..8]."
                )
            # Computa bnc_port para que esté siempre disponible
            self.bnc_port = (self.module_slot - 1) * 4 + self.channel_index + 1


@dataclass
class AcquisitionConfig:
    """Configuración de una sesión de captura."""
    mode: str  # "ema_triggered" | "oma_continuous" | "simulated_ema" | "simulated_oma"
    sample_rate_hz: float  # típico 5120 Hz para EMA, 10240 Hz para OMA
    duration_s: float       # 1-2 seg EMA, 60-300+ seg OMA
    channels: List[ChannelConfig] = field(default_factory=list)

    # Chasis del NI-DAQmx (típicamente "cDAQ1" para una sola maleta
    # conectada). Si hay varias maletas, usar "cDAQ2", "cDAQ3", etc.
    # discover_ni9234_modules() ayuda a identificar cuál está conectado.
    chassis_name: str = "cDAQ1"

    # DEPRECATED v3.31.201: usar chassis_name. Mantenido por backward
    # compat — si se da, se interpreta como chassis_name (extrayendo el
    # prefix hasta antes de "Mod" si tiene formato legacy "cDAQ1Mod1").
    device_name: Optional[str] = None

    # Solo para modo EMA:
    trigger_channel: Optional[int] = None  # bnc_port del martillo (1..32)
    trigger_level_V: float = 0.5
    pre_trigger_samples: int = 100
    n_averages: int = 5  # número de impactos a promediar

    # Streaming OMA: chunk size en segundos. Default 1s = buen balance
    # entre overhead de I/O y RAM. Bajar a 0.5s si la captura tiene
    # muchísimos canales (>16) y el disco es lento.
    oma_chunk_seconds: float = 1.0

    # Output path del .tdms generado
    output_tdms_path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Normaliza device_name legacy → chassis_name moderno."""
        if self.device_name and self.chassis_name == "cDAQ1":
            # Extrae el chasis del formato legacy "cDAQ1Mod1" → "cDAQ1"
            legacy = self.device_name
            if "Mod" in legacy:
                self.chassis_name = legacy.split("Mod")[0]
            else:
                self.chassis_name = legacy


def _nearest_valid_rate(requested: float) -> int:
    """Redondea al sample rate válido más cercano del NI-9234."""
    return min(_NI9234_VALID_RATES, key=lambda r: abs(r - requested))


def _expected_samples(config: AcquisitionConfig) -> int:
    """Calcula el número total de samples por canal."""
    return int(round(config.sample_rate_hz * config.duration_s))


def _build_phys_channel(chassis_name: str, ch: ChannelConfig) -> str:
    """
    Construye el nombre físico NI-DAQmx para un ChannelConfig.

    Ejemplo:
        chassis_name="cDAQ1", ch.bnc_port=5
        → "cDAQ1Mod2/ai0"  (slot 2, channel index 0 dentro del módulo)

        chassis_name="cDAQ1", ch.bnc_port=32
        → "cDAQ1Mod8/ai3"  (slot 8, channel index 3)
    """
    return f"{chassis_name}Mod{ch.module_slot}/ai{ch.channel_index}"


def _resolve_trigger_phys(chassis_name: str,
                            trigger_channel: int,
                            channels: List[ChannelConfig]) -> str:
    """
    Construye el nombre físico del canal trigger para EMA.

    `trigger_channel` puede ser:
      · Un bnc_port (1..32) — preferido
      · Un channel_index legacy (0..3) — para backward compat, asume Mod1
    """
    if trigger_channel >= 1 and trigger_channel <= 32:
        # Asumimos bnc_port. Computa slot e idx directo.
        slot = (trigger_channel - 1) // 4 + 1
        idx = (trigger_channel - 1) % 4
        return f"{chassis_name}Mod{slot}/ai{idx}"
    elif 0 <= trigger_channel <= 3:
        # Legacy: índice dentro de Mod1
        return f"{chassis_name}Mod1/ai{trigger_channel}"
    else:
        raise ValueError(
            f"trigger_channel={trigger_channel} fuera de rango. "
            f"Usa bnc_port (1..32) o channel_index legacy (0..3)."
        )


def list_available_devices() -> List[Dict[str, str]]:
    """
    Lista los chassis y módulos NI conectados al sistema.

    Returns:
        Lista de dicts {"name": str, "product_type": str, "serial": str}

    Raises:
        ImportError si nidaqmx no está disponible
    """
    try:
        from nidaqmx.system import System
    except ImportError as exc:
        raise ImportError(
            "nidaqmx no está instalado. Ejecuta: pip install nidaqmx\n"
            "Adicionalmente, el driver NI-DAQmx debe estar instalado en el sistema "
            "(descarga gratuita en ni.com)."
        ) from exc

    system = System.local()
    return [
        {
            "name": dev.name,
            "product_type": dev.product_type,
            "serial": str(dev.serial_num),
        }
        for dev in system.devices
    ]


def discover_ni9234_modules(chassis_name: str = "cDAQ1") -> List[Dict]:
    """
    Detecta cuántos módulos NI-9234 hay instalados en el chasis y en qué slots.

    Útil para validar que la maleta esté completa antes de configurar una
    captura de 32 canales. Si solo hay 4 módulos instalados (slots 1-4),
    el operador puede usar máximo BNC 1..16.

    Args:
        chassis_name: nombre del chasis (e.g. "cDAQ1"). Default "cDAQ1".

    Returns:
        Lista ordenada por slot:
          [{"slot": 1, "device_name": "cDAQ1Mod1", "serial": "0x12345",
            "bnc_range": (1, 4)},
           {"slot": 2, "device_name": "cDAQ1Mod2", "serial": "0x12346",
            "bnc_range": (5, 8)},
           ...]
        Vacío si no hay módulos 9234 detectados.

    Raises:
        ImportError si nidaqmx no está disponible
    """
    try:
        from nidaqmx.system import System
    except ImportError as exc:
        raise ImportError(
            "nidaqmx no está instalado. discover_ni9234_modules() solo "
            "funciona en el laptop de captura con NI-DAQmx driver."
        ) from exc

    system = System.local()
    modules = []
    for dev in system.devices:
        product = (dev.product_type or "").upper()
        if "9234" not in product:
            continue
        # El driver NI nombra módulos como "cDAQ1Mod1", "cDAQ1Mod2"...
        name = dev.name
        if not name.startswith(chassis_name) or "Mod" not in name:
            continue
        try:
            slot = int(name.split("Mod")[-1])
        except (ValueError, IndexError):
            continue
        if not (1 <= slot <= 8):
            continue
        bnc_start = (slot - 1) * 4 + 1
        bnc_end = bnc_start + 3
        modules.append({
            "slot": slot,
            "device_name": name,
            "product_type": dev.product_type,
            "serial": str(dev.serial_num),
            "bnc_range": (bnc_start, bnc_end),
        })
    return sorted(modules, key=lambda m: m["slot"])


def validate_channels_against_hardware(
    config: AcquisitionConfig,
) -> Tuple[bool, List[str]]:
    """
    Verifica que todos los bnc_port del config tengan módulo físico instalado.

    Útil pre-captura para evitar el error críptico de NI-DAQmx cuando se
    pide un canal en un slot vacío.

    Returns:
        (ok, problems) — ok=True si todos los canales tienen hardware.
        problems es lista de strings descriptivos de qué falta.
    """
    try:
        modules = discover_ni9234_modules(config.chassis_name)
    except ImportError:
        # Sin driver no podemos validar. Asumimos OK (probablemente modo
        # simulated en una máquina sin hardware).
        return True, []

    installed_slots = {m["slot"] for m in modules}
    problems: List[str] = []
    for ch in config.channels:
        if ch.module_slot not in installed_slots:
            problems.append(
                f"Canal '{ch.name}' (BNC {ch.bnc_port}) requiere slot "
                f"{ch.module_slot} pero ese slot está vacío. "
                f"Slots con NI-9234: {sorted(installed_slots) or 'ninguno'}"
            )
    return len(problems) == 0, problems


def self_test_channel(channel: ChannelConfig, sample_rate_hz: float = 5120,
                       duration_s: float = 1.0,
                       chassis_name: str = "cDAQ1") -> Dict:
    """
    Prueba rápida de un canal: captura 1 seg, devuelve estadísticas.

    Útil para validar conexión y sensibilidades antes de captura modal.

    Args:
        channel: ChannelConfig con bnc_port o (module_slot, channel_index)
        sample_rate_hz: típico 5120 Hz
        duration_s: típico 1 s
        chassis_name: nombre del chasis (default "cDAQ1")

    Returns:
        dict {"mean_V": float, "rms_V": float, "peak_V": float,
              "n_samples": int, "saturated": bool, "phys_channel": str}
    """
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType
    except ImportError as exc:
        raise ImportError("nidaqmx requerido para self_test_channel") from exc

    n_samples = int(sample_rate_hz * duration_s)
    phys_chan = _build_phys_channel(chassis_name, channel)

    with nidaqmx.Task() as task:
        if channel.coupling.upper() == "IEPE":
            task.ai_channels.add_ai_accel_chan(
                phys_chan,
                sensitivity=channel.sensitivity_mv_per_eu,
                max_val=channel.voltage_range,
                min_val=-channel.voltage_range,
            )
        else:
            task.ai_channels.add_ai_voltage_chan(
                phys_chan,
                max_val=channel.voltage_range,
                min_val=-channel.voltage_range,
            )

        task.timing.cfg_samp_clk_timing(
            rate=sample_rate_hz,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=n_samples,
        )
        data = task.read(number_of_samples_per_channel=n_samples)

    try:
        import numpy as np
    except ImportError:
        # Sin numpy, retornamos stats básicos
        data_list = list(data)
        mean_v = sum(data_list) / max(len(data_list), 1)
        peak_v = max(abs(x) for x in data_list)
        rms_v = math.sqrt(sum(x ** 2 for x in data_list) / max(len(data_list), 1))
        saturated = peak_v >= channel.voltage_range * 0.95
        return {"mean_V": mean_v, "rms_V": rms_v, "peak_V": peak_v,
                "n_samples": len(data_list), "saturated": saturated}

    arr = np.asarray(data, dtype=float)
    return {
        "mean_V": float(arr.mean()),
        "rms_V": float(np.sqrt((arr ** 2).mean())),
        "peak_V": float(np.abs(arr).max()),
        "n_samples": arr.size,
        "saturated": bool(np.abs(arr).max() >= channel.voltage_range * 0.95),
        "phys_channel": phys_chan,
        "bnc_port": channel.bnc_port,
    }


def capture(
    config: AcquisitionConfig,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """
    Ejecuta la captura según configuración.

    Args:
        config: AcquisitionConfig validada
        on_progress: callback opcional (progress_0_to_1, status_text)

    Returns:
        Path al archivo .tdms generado

    Raises:
        ImportError si nidaqmx no está disponible (modo real)
        RuntimeError si la captura falla
    """
    if config.output_tdms_path is None:
        raise ValueError("output_tdms_path es requerido")

    # Validar y normalizar
    config.sample_rate_hz = _nearest_valid_rate(config.sample_rate_hz)

    if not config.channels:
        raise ValueError("Al menos 1 canal debe configurarse")

    progress = on_progress or (lambda f, s: None)
    progress(0.0, "Iniciando captura")

    # Ciclo 23.158 — Aceptar simulated_ema / simulated_oma (variantes que
    # preservan el sub-modo para el TDMS metadata, sin romper compat con
    # "simulated" plano legacy).
    if config.mode == "simulated" or config.mode.startswith("simulated_"):
        return _capture_simulated(config, progress)
    if config.mode == "ema_triggered":
        return _capture_ema(config, progress)
    if config.mode == "oma_continuous":
        return _capture_oma(config, progress)
    raise ValueError(f"Modo desconocido: {config.mode}")


# =====================================================================
# EMA — Impact Hammer Triggered Capture
# =====================================================================

def _capture_ema(config: AcquisitionConfig, progress: Callable) -> Path:
    """
    Captura EMA con martillo modal.

    Workflow:
      1. Configura todos los canales
      2. Configura trigger por nivel en el canal del martillo
      3. Para cada uno de N_averages impactos:
         - Espera trigger
         - Captura pre_trigger + duration samples
         - Acumula
      4. Promedia los N_averages
      5. Escribe TDMS con time-series promediado
    """
    try:
        import nidaqmx
        from nidaqmx.constants import (
            AcquisitionType, TerminalConfiguration, TriggerType, Slope,
        )
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "nidaqmx + numpy requeridos para captura EMA real. "
            "Usa mode='simulated' para development sin hardware."
        ) from exc

    if config.trigger_channel is None:
        raise ValueError("trigger_channel requerido para modo ema_triggered")

    chassis = config.chassis_name
    n_samples = _expected_samples(config)
    accumulated: List[List[float]] = [[0.0] * n_samples for _ in config.channels]

    for avg_idx in range(config.n_averages):
        progress(avg_idx / config.n_averages,
                 f"Impacto {avg_idx + 1}/{config.n_averages} — esperando trigger...")

        with nidaqmx.Task() as task:
            for ch in config.channels:
                phys = _build_phys_channel(chassis, ch)
                if ch.coupling.upper() == "IEPE":
                    task.ai_channels.add_ai_accel_chan(
                        phys, sensitivity=ch.sensitivity_mv_per_eu,
                        max_val=ch.voltage_range, min_val=-ch.voltage_range,
                    )
                else:
                    task.ai_channels.add_ai_voltage_chan(
                        phys, max_val=ch.voltage_range,
                        min_val=-ch.voltage_range,
                    )

            task.timing.cfg_samp_clk_timing(
                rate=config.sample_rate_hz,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=n_samples,
            )

            # Trigger analógico por nivel en el canal del martillo
            trig_phys = _resolve_trigger_phys(
                chassis, config.trigger_channel, config.channels,
            )
            task.triggers.start_trigger.cfg_anlg_edge_start_trig(
                trigger_source=trig_phys,
                trigger_level=config.trigger_level_V,
                trigger_slope=Slope.RISING,
            )
            task.triggers.start_trigger.pretrigger_samples = config.pre_trigger_samples

            data = task.read(number_of_samples_per_channel=n_samples, timeout=30.0)
            if not isinstance(data[0], list):
                data = [data]  # un solo canal

            for ch_idx, samples in enumerate(data):
                for i, v in enumerate(samples):
                    accumulated[ch_idx][i] += v

    # Promediar
    for ch_idx in range(len(accumulated)):
        for i in range(n_samples):
            accumulated[ch_idx][i] /= max(config.n_averages, 1)

    progress(0.95, "Escribiendo TDMS...")
    _write_tdms(config, accumulated)
    progress(1.0, f"Listo · {n_samples} samples × {len(config.channels)} ch")
    return config.output_tdms_path


# =====================================================================
# OMA — Continuous Acquisition
# =====================================================================

def _capture_oma(config: AcquisitionConfig, progress: Callable) -> Path:
    """
    Captura continua streaming para OMA con escritura TDMS incremental.

    Diferencias vs versión legacy (4-canal, RAM completa):
      · Usa TdmsWriter en context manager abierto durante toda la captura
      · Escribe write_segment() cada chunk de N segundos (config.oma_chunk_seconds)
      · RAM constante ~5 MB (solo el chunk actual), no crece con duration
      · Soporta hasta 32 canales × cualquier duración sin OOM
      · Si la captura falla a mitad, el TDMS queda parcial pero válido
        (TdmsWriter cierra segmentos atómicamente — npTDMS puede leerlo)

    Math de RAM:
      Legacy: 32 ch × 300 s × 5120 Hz × 8 bytes = ~390 MB en RAM
      Stream:  32 ch ×   1 s × 5120 Hz × 8 bytes = ~1.3 MB en RAM (constante)
    """
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType
        import numpy as np
        from nptdms import TdmsWriter, ChannelObject, GroupObject, RootObject
    except ImportError as exc:
        raise ImportError(
            "nidaqmx + numpy + npTDMS requeridos para captura OMA real. "
            "Usa mode='simulated_oma' para development sin hardware."
        ) from exc

    chassis = config.chassis_name
    fs = int(config.sample_rate_hz)
    total_samples = _expected_samples(config)
    chunk_samples = max(int(config.oma_chunk_seconds * fs), 1)

    # Validar hardware presente antes de empezar (modo real)
    ok, problems = validate_channels_against_hardware(config)
    if not ok:
        raise RuntimeError(
            "Canales pedidos no tienen hardware instalado:\n  - "
            + "\n  - ".join(problems)
        )

    output_path = Path(config.output_tdms_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Metadata root + group escrito en el primer segmento
    timestamp = datetime.now(timezone.utc).isoformat()
    root = RootObject(properties={
        "sample_rate_hz": float(config.sample_rate_hz),
        "mode": config.mode,
        "duration_s": float(config.duration_s),
        "n_averages": int(config.n_averages),
        "captured_at_utc": timestamp,
        "chassis_name": chassis,
        "n_channels": len(config.channels),
        "streaming": True,
        "chunk_seconds": float(config.oma_chunk_seconds),
    })
    group = GroupObject("Acquisition")

    with nidaqmx.Task() as task, TdmsWriter(str(output_path)) as writer:
        for ch in config.channels:
            phys = _build_phys_channel(chassis, ch)
            if ch.coupling.upper() == "IEPE":
                task.ai_channels.add_ai_accel_chan(
                    phys, sensitivity=ch.sensitivity_mv_per_eu,
                    max_val=ch.voltage_range, min_val=-ch.voltage_range,
                )
            else:
                task.ai_channels.add_ai_voltage_chan(
                    phys, max_val=ch.voltage_range,
                    min_val=-ch.voltage_range,
                )

        task.timing.cfg_samp_clk_timing(
            rate=fs,
            sample_mode=AcquisitionType.CONTINUOUS,
            # Buffer interno NI grande (4× chunk) para evitar overruns
            # durante el write_segment() a disco
            samps_per_chan=chunk_samples * 4,
        )

        task.start()
        collected = 0
        first_segment = True
        while collected < total_samples:
            this_chunk = min(chunk_samples, total_samples - collected)
            data = task.read(number_of_samples_per_channel=this_chunk, timeout=30.0)
            if not isinstance(data[0], list):
                data = [data]  # caso 1-canal: nidaqmx devuelve flat list

            # Construir ChannelObjects para este segmento
            ch_objs = []
            for ch_cfg, samples in zip(config.channels, data):
                arr = np.asarray(samples, dtype=np.float32)  # float32 = 4 bytes
                props = (
                    {  # primer segmento lleva metadata del canal
                        "module_slot": ch_cfg.module_slot,
                        "channel_index": ch_cfg.channel_index,
                        "bnc_port": ch_cfg.bnc_port,
                        "coupling": ch_cfg.coupling,
                        "sensitivity_mv_per_eu": float(ch_cfg.sensitivity_mv_per_eu),
                        "units": ch_cfg.units,
                        "voltage_range": float(ch_cfg.voltage_range),
                        "wf_increment": 1.0 / float(config.sample_rate_hz),
                        "wf_start_offset": 0.0,
                    } if first_segment else {}
                )
                ch_objs.append(
                    ChannelObject("Acquisition", ch_cfg.name, arr,
                                   properties=props)
                )

            if first_segment:
                writer.write_segment([root, group, *ch_objs])
                first_segment = False
            else:
                writer.write_segment(ch_objs)

            collected += this_chunk
            progress(
                collected / total_samples,
                f"OMA · {collected/fs:.1f}/{config.duration_s:.1f} s "
                f"({len(config.channels)} ch streaming)"
            )
        task.stop()

    progress(1.0, f"Listo · {total_samples} samples × {len(config.channels)} ch (TDMS streamed)")
    return output_path


# =====================================================================
# Simulated — Synthetic Data Generator
# =====================================================================

def _capture_simulated(config: AcquisitionConfig, progress: Callable) -> Path:
    """
    Genera data sintética que imita una captura real del NI-9234.

    Para EMA: input = impulso decaído + ruido, output = respuesta modal
    de un sistema 2-DOF con frecuencias 50 Hz y 120 Hz.

    Para OMA: respuesta a ruido blanco filtrado por sistema 3-DOF con
    frecuencias 30 Hz, 75 Hz, 140 Hz.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy requerido para modo simulated") from exc

    fs = float(config.sample_rate_hz)
    n_samples = _expected_samples(config)
    t = np.arange(n_samples) / fs

    buffers: List[List[float]] = []

    rng = np.random.default_rng(seed=42)

    # Ciclo 23.158 — Bifurcación clara EMA vs OMA en simulado.
    # · simulated_ema o trigger_channel is not None → simular impacto + respuesta
    # · simulated_oma o sin trigger → simular respuesta a ruido blanco (sistema 3-DOF)
    _is_ema_sim = (
        config.mode == "simulated_ema"
        or (config.mode == "simulated" and config.trigger_channel is not None)
    )
    # Determinar qué canal es el martillo. trigger_channel puede ser:
    #   · bnc_port (1..32) — comparar contra ch.bnc_port
    #   · channel_index legacy (0..3) — comparar contra posición en lista
    def _is_trigger_channel(ch_idx: int, ch: ChannelConfig) -> bool:
        tc = config.trigger_channel
        if tc is None:
            return False
        if 1 <= tc <= 32:
            return ch.bnc_port == tc
        return ch_idx == tc  # legacy

    if _is_ema_sim:
        # Simular EMA: martillo + respuesta
        modes = [(50.0, 0.02), (120.0, 0.015)]
        for ch_idx, ch in enumerate(config.channels):
            if _is_trigger_channel(ch_idx, ch):
                # Martillo: impulso al t=0.05 seg con decaimiento rápido
                impulse = np.zeros(n_samples)
                t_impact = int(0.05 * fs)
                tau_impact = 0.001  # 1 ms decay
                for i in range(t_impact, min(t_impact + int(0.01 * fs), n_samples)):
                    impulse[i] = math.exp(-(i - t_impact) / (tau_impact * fs)) * 3.0
                buffers.append((impulse + 0.01 * rng.standard_normal(n_samples)).tolist())
            else:
                # Respuesta modal: suma de senoidales decaídas + ruido
                resp = np.zeros(n_samples)
                t_start = int(0.05 * fs)
                for fn, zeta in modes:
                    wn = 2 * math.pi * fn
                    wd = wn * math.sqrt(1 - zeta ** 2)
                    decay = np.exp(-zeta * wn * (t - t[t_start])) * (t >= t[t_start])
                    resp += decay * np.sin(wd * (t - t[t_start])) * 0.5
                resp += 0.02 * rng.standard_normal(n_samples)
                buffers.append(resp.tolist())
            progress((ch_idx + 1) / len(config.channels),
                     f"Simulando canal {ch.name}")
    else:
        # Simular OMA: ruido filtrado por sistema 3-DOF
        modes = [(30.0, 0.03), (75.0, 0.025), (140.0, 0.018)]
        for ch_idx, ch in enumerate(config.channels):
            white = rng.standard_normal(n_samples)
            resp = np.zeros(n_samples)
            for fn, zeta in modes:
                wn = 2 * math.pi * fn
                # Aproximación simple: filtro IIR resonante
                a1 = -2 * math.cos(wn / fs) * math.exp(-zeta * wn / fs)
                a2 = math.exp(-2 * zeta * wn / fs)
                filtered = np.zeros(n_samples)
                for i in range(2, n_samples):
                    filtered[i] = white[i] - a1 * filtered[i - 1] - a2 * filtered[i - 2]
                resp += filtered * 0.3
            buffers.append(resp.tolist())
            progress((ch_idx + 1) / len(config.channels),
                     f"Simulando canal {ch.name}")

    progress(0.95, "Escribiendo TDMS simulado...")
    _write_tdms(config, buffers)
    progress(1.0, f"Listo · {n_samples} samples × {len(config.channels)} ch (simulado)")
    return config.output_tdms_path


# =====================================================================
# TDMS Writer
# =====================================================================

def _write_tdms(config: AcquisitionConfig, data: List[List[float]]) -> None:
    """
    Escribe los datos capturados a un archivo TDMS portable.

    Estructura:
      File-level properties: sample_rate, mode, n_averages, timestamp
      Group: "Acquisition"
      Channels: uno por sensor, con properties: name, coupling,
                sensitivity, units, channel_index
    """
    try:
        from nptdms import TdmsWriter, ChannelObject, GroupObject, RootObject
    except ImportError as exc:
        raise ImportError(
            "npTDMS requerido para escribir archivos TDMS. "
            "Ejecuta: pip install npTDMS"
        ) from exc

    timestamp = datetime.now(timezone.utc).isoformat()
    root = RootObject(properties={
        "sample_rate_hz": float(config.sample_rate_hz),
        "mode": config.mode,
        "duration_s": float(config.duration_s),
        "n_averages": int(config.n_averages),
        "captured_at_utc": timestamp,
        "chassis_name": config.chassis_name,
        "device_name": config.device_name or "",  # legacy field
        "n_channels": len(config.channels),
        "streaming": False,
    })
    group = GroupObject("Acquisition")
    channel_objs = []
    for ch_cfg, samples in zip(config.channels, data):
        ch_obj = ChannelObject(
            "Acquisition", ch_cfg.name, samples,
            properties={
                "module_slot": ch_cfg.module_slot,
                "channel_index": ch_cfg.channel_index,
                "bnc_port": ch_cfg.bnc_port,
                "coupling": ch_cfg.coupling,
                "sensitivity_mv_per_eu": float(ch_cfg.sensitivity_mv_per_eu),
                "units": ch_cfg.units,
                "voltage_range": float(ch_cfg.voltage_range),
                "wf_increment": 1.0 / float(config.sample_rate_hz),
                "wf_start_offset": 0.0,
            },
        )
        channel_objs.append(ch_obj)

    output_path = config.output_tdms_path
    if output_path is None:
        raise ValueError("output_tdms_path no definido")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TdmsWriter(str(output_path)) as writer:
        writer.write_segment([root, group, *channel_objs])
