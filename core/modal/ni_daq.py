"""
core/modal/ni_daq.py — Adquisición con tarjeta NI-9234
=======================================================

Wrapper sobre `nidaqmx` (driver oficial NI Python) para capturar datos
del módulo NI-9234 en dos modos:

Modo EMA — Impact Hammer Test (ISO 7626-5)
-------------------------------------------
Captura sincronizada de impacto + respuesta:
  · Canal 0: martillo modal (input, trigger)
  · Canales 1-3: acelerómetros respuesta (output)
  · Trigger: por nivel en canal de fuerza (e.g. > 0.5 N)
  · Ventana: rectangular en input (force window), exponencial en output
  · Duración: 1-2 segundos típico
  · Promediado: 5-10 impactos para reducir ruido aleatorio

Modo OMA — Operational Modal Analysis (ISO 20816)
--------------------------------------------------
Captura continua durante operación normal:
  · 4 canales sincronizados (acelerómetros o proximidad)
  · Sin trigger — adquisición continua streaming
  · Duración: 60-300 segundos a velocidad constante
  · Sample rate: 5-10 kHz (configurable hasta 51.2 kHz)
  · Output: archivo .tdms para procesamiento posterior con SSI/FDD

Modo Simulated (para development sin hardware)
-----------------------------------------------
Genera data sintética que imita lo que devolvería el NI-9234. Útil para
probar el pipeline modal end-to-end sin tarjeta conectada. Activable con
`AcquisitionConfig.mode = "simulated"` o flag `--simulated` en el companion.

NI-9234 specs
-------------
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
    """Configuración de un canal del NI-9234."""
    channel_index: int  # 0..3 para NI-9234
    name: str           # Etiqueta del sensor (e.g. "1YA", "Hammer")
    coupling: str       # "IEPE", "AC", "DC"
    sensitivity_mv_per_eu: float  # 100.0 para Wilcoxon, 200.0 para Bently
    units: str = "g"    # "g", "mil", "N"
    voltage_range: float = 5.0  # ±V


@dataclass
class AcquisitionConfig:
    """Configuración de una sesión de captura."""
    mode: str  # "ema_triggered" | "oma_continuous" | "simulated"
    sample_rate_hz: float  # típico 5120 Hz para EMA, 10240 Hz para OMA
    duration_s: float       # 1-2 seg EMA, 60-300 seg OMA
    channels: List[ChannelConfig] = field(default_factory=list)

    # Device chassis (e.g. "cDAQ1Mod1"). Si None, se autodetecta el primer
    # NI-9234 disponible.
    device_name: Optional[str] = None

    # Solo para modo EMA:
    trigger_channel: Optional[int] = None  # canal del martillo (e.g. 0)
    trigger_level_V: float = 0.5
    pre_trigger_samples: int = 100
    n_averages: int = 5  # número de impactos a promediar

    # Output path del .tdms generado
    output_tdms_path: Optional[Path] = None


def _nearest_valid_rate(requested: float) -> int:
    """Redondea al sample rate válido más cercano del NI-9234."""
    return min(_NI9234_VALID_RATES, key=lambda r: abs(r - requested))


def _expected_samples(config: AcquisitionConfig) -> int:
    """Calcula el número total de samples por canal."""
    return int(round(config.sample_rate_hz * config.duration_s))


def list_available_devices() -> List[Dict[str, str]]:
    """
    Lista los chassis NI conectados al sistema con sus módulos.

    Returns:
        Lista de dicts {"name": str, "product_type": str, "serial": str}

    Raises:
        ImportError si nidaqmx no está disponible
    """
    try:
        import nidaqmx
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


def self_test_channel(channel: ChannelConfig, sample_rate_hz: float = 5120,
                       duration_s: float = 1.0) -> Dict:
    """
    Prueba rápida de un canal: captura 1 seg, devuelve estadísticas.

    Útil para validar conexión y sensibilidades antes de captura modal.

    Returns:
        dict {"mean_V": float, "rms_V": float, "peak_V": float,
              "n_samples": int, "saturated": bool}
    """
    try:
        import nidaqmx
        from nidaqmx.constants import TerminalConfiguration, AcquisitionType
    except ImportError as exc:
        raise ImportError("nidaqmx requerido para self_test_channel") from exc

    n_samples = int(sample_rate_hz * duration_s)
    device_name = "cDAQ1Mod1"  # asume primer módulo
    phys_chan = f"{device_name}/ai{channel.channel_index}"

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

    device_name = config.device_name or "cDAQ1Mod1"
    n_samples = _expected_samples(config)
    accumulated: List[List[float]] = [[0.0] * n_samples for _ in config.channels]

    for avg_idx in range(config.n_averages):
        progress(avg_idx / config.n_averages,
                 f"Impacto {avg_idx + 1}/{config.n_averages} — esperando trigger...")

        with nidaqmx.Task() as task:
            for ch in config.channels:
                phys = f"{device_name}/ai{ch.channel_index}"
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
            trig_phys = f"{device_name}/ai{config.trigger_channel}"
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
    Captura continua streaming para OMA.

    Workflow:
      1. Configura todos los canales en modo continuous
      2. Lee chunks de 1 segundo
      3. Acumula en buffer + actualiza progreso
      4. Al cumplir duration_s total, escribe TDMS
    """
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "nidaqmx + numpy requeridos para captura OMA real. "
            "Usa mode='simulated' para development sin hardware."
        ) from exc

    device_name = config.device_name or "cDAQ1Mod1"
    fs = int(config.sample_rate_hz)
    total_samples = _expected_samples(config)
    chunk_samples = fs  # 1 segundo por chunk

    buffers: List[List[float]] = [[] for _ in config.channels]

    with nidaqmx.Task() as task:
        for ch in config.channels:
            phys = f"{device_name}/ai{ch.channel_index}"
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
            samps_per_chan=chunk_samples * 4,  # buffer interno generoso
        )

        task.start()
        collected = 0
        while collected < total_samples:
            this_chunk = min(chunk_samples, total_samples - collected)
            data = task.read(number_of_samples_per_channel=this_chunk, timeout=30.0)
            if not isinstance(data[0], list):
                data = [data]
            for ch_idx, samples in enumerate(data):
                buffers[ch_idx].extend(samples)
            collected += this_chunk
            progress(collected / total_samples,
                     f"OMA · {collected/fs:.1f} / {config.duration_s:.1f} s")
        task.stop()

    progress(0.95, "Escribiendo TDMS...")
    _write_tdms(config, buffers)
    progress(1.0, f"Listo · {total_samples} samples × {len(config.channels)} ch")
    return config.output_tdms_path


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
    if _is_ema_sim:
        # Simular EMA: martillo + respuesta
        modes = [(50.0, 0.02), (120.0, 0.015)]
        for ch_idx, ch in enumerate(config.channels):
            if ch_idx == config.trigger_channel:
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
        "device_name": config.device_name or "",
        "n_channels": len(config.channels),
    })
    group = GroupObject("Acquisition")
    channel_objs = []
    for ch_cfg, samples in zip(config.channels, data):
        ch_obj = ChannelObject(
            "Acquisition", ch_cfg.name, samples,
            properties={
                "channel_index": ch_cfg.channel_index,
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
