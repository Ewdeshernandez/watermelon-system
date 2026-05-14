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
laptop de captura). Solo necesario en el companion script, NO en Streamlit
Cloud que solo procesa los .tdms ya capturados.

  pip install nidaqmx
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Callable


@dataclass
class ChannelConfig:
    """Configuración de un canal del NI-9234."""
    channel_index: int  # 0..3 para NI-9234
    name: str           # Etiqueta del sensor (e.g. "1YA", "Hammer")
    coupling: str       # "IEPE", "AC", "DC"
    sensitivity_mv_per_eu: float  # 100.0 para Wilcoxon, 200.0 para Bently
    units: str          # "g", "mil", "N"
    voltage_range: float = 5.0  # ±V


@dataclass
class AcquisitionConfig:
    """Configuración de una sesión de captura."""
    mode: str  # "ema_triggered" o "oma_continuous"
    sample_rate_hz: float  # típico 5120 Hz para EMA, 10240 Hz para OMA
    duration_s: float       # 1-2 seg EMA, 60-300 seg OMA
    channels: List[ChannelConfig]

    # Solo para modo EMA:
    trigger_channel: Optional[int] = None  # canal del martillo
    trigger_level_V: float = 0.5
    pre_trigger_samples: int = 100
    n_averages: int = 5  # número de impactos

    # Output
    output_tdms_path: Optional[str] = None


def capture(
    config: AcquisitionConfig,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Ejecuta la captura según configuración.

    Args:
        config: AcquisitionConfig validada
        on_progress: callback opcional (progress_0_to_1, status_text)

    Returns:
        Path al archivo .tdms generado

    Raises:
        ImportError si nidaqmx no está disponible
        RuntimeError si la captura falla
    """
    try:
        import nidaqmx  # noqa
    except ImportError:
        raise ImportError(
            "nidaqmx no está instalado. Ejecuta: pip install nidaqmx\n"
            "Adicionalmente, el driver NI-DAQmx debe estar instalado en el sistema "
            "(descarga gratuita en ni.com)."
        )

    # TODO: implementar lógica completa para ambos modos
    raise NotImplementedError("Fase scaffolding — implementación en sprint NI-DAQ")


def list_available_devices() -> List[str]:
    """
    Lista los chassis NI conectados al sistema.

    Útil para verificación previa antes de configurar una captura.
    """
    raise NotImplementedError("Fase scaffolding")


def self_test_channel(channel: ChannelConfig) -> dict:
    """
    Prueba rápida de un canal: lee 1 segundo de data, devuelve estadísticas.

    Útil para validar conexión y sensibilidades antes de la captura modal.
    """
    raise NotImplementedError("Fase scaffolding")
