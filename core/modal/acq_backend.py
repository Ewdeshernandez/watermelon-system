"""
core/modal/acq_backend.py — Backend de adquisición Watermelon
=========================================================

Wrapper de la maleta Watermelon de adquisición multicanal — hasta
**32 canales simultáneos** numerados como puertos BNC 1..32 en el
frente de la maleta.

Capacidad del sistema
---------------------
· Hasta 32 canales analógicos simultáneos
· Resolución 24-bit por canal
· Sample rate hasta 51.2 kHz (16 valores discretos válidos)
· Excitación IEPE / AC / DC configurable por canal
· Rango ±5 V por canal
· Sincronización via sample clock compartido del chasis
· Apto para equipos rotativos (ATEX Ex II 3G / UL Class I Div 2)

Naming convention interno
-------------------------
Cada canal físico se direcciona como `{chassis}Mod{slot}/ai{idx}`:
  Mod1/ai0..ai3   (slot 1 → BNC 1..4)
  Mod2/ai0..ai3   (slot 2 → BNC 5..8)
  ...
  Mod8/ai0..ai3   (slot 8 → BNC 29..32)

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
  · Archivo escrito **chunk-por-chunk** (RAM constante ~5 MB para evitar
    OOM en captura 32ch × 300s × 5120 Hz que serían ~390 MB en RAM)

Modo Simulated (para development sin hardware)
-----------------------------------------------
Genera data sintética que imita lo que devolvería la maleta real. Útil
para probar el pipeline modal end-to-end sin maleta conectada. Activable
con `AcquisitionConfig.mode = "simulated_ema"` o `"simulated_oma"`
o flag `--simulated` en el companion.

Sensor coupling
---------------
· IEPE (acelerómetros 100 mV/g): la maleta suministra excitación 2 mA
· AC con bias (sondas de proximidad 200 mV/mil): requiere alimentación
  externa, conexión BNC → maleta en modo AC coupled
· DC: rara vez usado en vibración

Dependencias internas
---------------------
Import lazy del backend driver — el módulo se importa OK sin drivers
disponibles; solo falla al llamar las funciones de captura real (útil
para entornos sin maleta como Cloud).

Marco normativo
---------------
ISO 7626-5 secc. 6 — Configuración de canales para impact testing
ISO 7626-5 secc. 7 — Adquisición sincronizada input/output
ISO 20816-1 secc. 5.3 — Requisitos de instrumentación para mediciones operacionales
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
# Cloud (donde NO está el driver driver del fabricante) y solo falle al intentar
# capturar real.

# módulo de adquisición valid sample rates (Hz) — la tarjeta solo acepta valores
# discretos por su sigma-delta ADC. Si el usuario pide otro, se redondea
# al más cercano permitido.
_ACQ_VALID_RATES = [
    1652, 2000, 2048, 2500, 3200, 4000, 4096, 5120,
    6400, 8000, 8192, 10240, 12800, 16000, 16384, 20480,
    25600, 32000, 32768, 40960, 51200,
]


@dataclass
class ChannelConfig:
    """
    Configuración de un canal del módulo de adquisición dentro de la maleta maleta de adquisición.

    Identificación del canal físico
    -------------------------------
    Hay dos maneras de identificar un canal y son mutuamente convertibles:

    1. **bnc_port (1..32)** — el número impreso en el frente de la maleta.
       Es lo que el operador ve. RECOMENDADO.

    2. **module_slot (1..8) + channel_index (0..3)** — direccionamiento
       interno que el backend driver necesita.

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
    sensitivity_mv_per_eu: float  # 100.0 acelerómetro IEPE, 200.0 sonda proximidad

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
                    f"La maleta Watermelon soporta máximo 32 canales."
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

    # Chasis del driver del fabricante (típicamente "cDAQ1" para una sola maleta
    # conectada). Si hay varias maletas, usar "cDAQ2", "cDAQ3", etc.
    # discover_acq_modules() ayuda a identificar cuál está conectado.
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
    """Redondea al sample rate válido más cercano del módulo de adquisición."""
    return min(_ACQ_VALID_RATES, key=lambda r: abs(r - requested))


def _expected_samples(config: AcquisitionConfig) -> int:
    """Calcula el número total de samples por canal."""
    return int(round(config.sample_rate_hz * config.duration_s))


def _build_phys_channel(chassis_name: str, ch: ChannelConfig) -> str:
    """
    Construye el nombre físico driver del fabricante para un ChannelConfig.

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
            "Drivers de adquisición Watermelon no disponibles en este equipo. "
            "Esto es esperado en entornos sin maleta conectada (ej: modo Cloud). "
            "Para captura local, corre INSTALAR.bat de Watermelon Planta."
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


def discover_acq_modules(chassis_name: str = "cDAQ1") -> List[Dict]:
    """
    Detecta cuántos módulos módulo de adquisición hay instalados en el chasis y en qué slots.

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
            "nidaqmx no está instalado. discover_acq_modules() solo "
            "funciona en el laptop de captura con driver del fabricante driver."
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


def diagnose_acquisition(chassis_name: str = "cDAQ1") -> Dict[str, object]:
    """v3.31.339 — Autodiagnóstico NO-lanzante de la cadena de adquisición.

    Distingue las 3 capas para que el operador sepa EXACTAMENTE qué falló,
    en vez de un genérico "reinstala":

      1. software_module: el paquete de captura quedó incluido en la app
         (si False → el build del .exe está incompleto, hay que reinstalar
         con el instalador completo).
      2. equipment_driver: el controlador del equipo está instalado y carga
         (si False con software_module True → falta instalar el driver del
         equipo desde el instalador / NI MAX).
      3. devices: módulos físicos detectados (si vacío con las dos anteriores
         OK → el equipo no está conectado o encendido).

    Mensajes SANITIZADOS (sin marcas del fabricante) — la app de planta es
    visible al cliente.
    """
    out: Dict[str, object] = {
        "software_module": False,
        "equipment_driver": False,
        "devices": [],
        "detail": "",
    }
    # Capa 1 — paquete de software de captura presente en el bundle
    try:
        from nidaqmx.system import System  # noqa: F401
        out["software_module"] = True
    except Exception as exc:  # ImportError u otro al congelar incompleto
        out["detail"] = f"software_module: {type(exc).__name__}"
        return out
    # Capa 2 — controlador del equipo instalado y cargable
    try:
        system = System.local()
        _ = list(system.devices)  # fuerza la carga del driver runtime
        out["equipment_driver"] = True
    except Exception as exc:
        out["detail"] = f"equipment_driver: {type(exc).__name__}"
        return out
    # Capa 3 — módulos físicos detectados
    try:
        out["devices"] = discover_acq_modules(chassis_name)
    except Exception as exc:
        out["detail"] = f"devices: {type(exc).__name__}"
    return out


def validate_channels_against_hardware(
    config: AcquisitionConfig,
) -> Tuple[bool, List[str]]:
    """
    Verifica que todos los bnc_port del config tengan módulo físico instalado.

    Útil pre-captura para evitar el error críptico de driver del fabricante cuando se
    pide un canal en un slot vacío.

    Returns:
        (ok, problems) — ok=True si todos los canales tienen hardware.
        problems es lista de strings descriptivos de qué falta.
    """
    try:
        modules = discover_acq_modules(config.chassis_name)
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
                f"Slots con módulo de adquisición: {sorted(installed_slots) or 'ninguno'}"
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
        from nidaqmx.constants import AcquisitionType, CurrentExcitSource
    except ImportError as exc:
        raise ImportError("nidaqmx requerido para self_test_channel") from exc

    n_samples = int(sample_rate_hz * duration_s)
    phys_chan = _build_phys_channel(chassis_name, channel)

    with nidaqmx.Task() as task:
        if channel.coupling.upper() == "IEPE":
            _g = channel.voltage_range * 1000.0 / (channel.sensitivity_mv_per_eu or 100.0)
            task.ai_channels.add_ai_accel_chan(
                phys_chan,
                sensitivity=channel.sensitivity_mv_per_eu,
                max_val=_g, min_val=-_g,
                current_excit_source=CurrentExcitSource.INTERNAL,
                current_excit_val=0.002,
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
    Captura EMA con martillo modal usando SOFTWARE TRIGGER.

    Por qué software trigger:
    Los módulo de adquisición son ADCs sigma-delta — el hardware NO soporta analog
    reference trigger (Status Code -200265 si se intenta). Es limitación
    física del módulo. Solución estándar para EMA con módulo de adquisición: capturar
    continuamente y detectar el impacto en software.

    Workflow (v3.31.205):
      1. Configura todos los canales (sin trigger hardware)
      2. Para cada uno de N_averages impactos:
         - Captura FINITE de capture_window_factor × duration segundos
           (3 segundos por default → el operador tiene tiempo de golpear)
         - Lee todos los samples
         - En software: busca el primer cruce ascendente del trigger_level
           en el canal del martillo (identificado por bnc_port o índice)
         - Extrae slice de [trigger_idx - pre_samples : + n_samples]
         - Acumula
      3. Promedia solo sobre impactos exitosamente detectados
      4. Si NINGÚN impacto se detectó → RuntimeError con hint diagnóstico
      5. Escribe TDMS con time-series promediado
    """
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, CurrentExcitSource
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "nidaqmx + numpy requeridos para captura EMA real. "
            "Usa mode='simulated' para development sin hardware."
        ) from exc

    if config.trigger_channel is None:
        raise ValueError("trigger_channel requerido para modo ema_triggered")

    chassis = config.chassis_name
    n_samples_per_impact = _expected_samples(config)

    # Capturamos 3× duration para dar ventana al operador para golpear
    capture_window_factor = 3
    n_samples_window = n_samples_per_impact * capture_window_factor

    # Encontrar el índice del canal del martillo en config.channels
    # trigger_channel puede ser bnc_port (1..32) o channel_index legacy (0..3)
    trigger_idx_in_list: Optional[int] = None
    tc = config.trigger_channel
    for i, ch in enumerate(config.channels):
        if tc is not None and 1 <= tc <= 32 and ch.bnc_port == tc:
            trigger_idx_in_list = i
            break
        if tc is not None and 0 <= tc <= 3 and ch.channel_index == tc and ch.module_slot == 1:
            trigger_idx_in_list = i
            break
    if trigger_idx_in_list is None:
        raise ValueError(
            f"trigger_channel={tc} no corresponde a ningún canal en --channels. "
            f"Asegúrate de incluir el martillo en --channels con el mismo BNC."
        )

    pre_samples = max(int(config.pre_trigger_samples), 100)
    accumulated = [np.zeros(n_samples_per_impact, dtype=np.float64)
                   for _ in config.channels]
    successful_impacts = 0
    failed_attempts: List[str] = []

    for avg_idx in range(config.n_averages):
        progress(
            avg_idx / config.n_averages,
            f"Impacto {avg_idx + 1}/{config.n_averages} — "
            f"golpea el martillo en los próximos "
            f"{capture_window_factor * config.duration_s:.0f} s...",
        )

        with nidaqmx.Task() as task:
            for ch in config.channels:
                phys = _build_phys_channel(chassis, ch)
                if ch.coupling.upper() == "IEPE":
                    _g = ch.voltage_range * 1000.0 / (ch.sensitivity_mv_per_eu or 100.0)
                    task.ai_channels.add_ai_accel_chan(
                        phys, sensitivity=ch.sensitivity_mv_per_eu,
                        max_val=_g, min_val=-_g,
                        current_excit_source=CurrentExcitSource.INTERNAL,
                        current_excit_val=0.002,
                    )
                else:
                    task.ai_channels.add_ai_voltage_chan(
                        phys, max_val=ch.voltage_range,
                        min_val=-ch.voltage_range,
                    )

            task.timing.cfg_samp_clk_timing(
                rate=config.sample_rate_hz,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=n_samples_window,
            )

            # SIN trigger hardware. Captura inmediatamente.
            timeout = (n_samples_window / config.sample_rate_hz) + 10.0
            raw = task.read(
                number_of_samples_per_channel=n_samples_window, timeout=timeout,
            )
            if not isinstance(raw[0], list):
                raw = [raw]  # un solo canal

            data_arr = np.array(
                [np.asarray(d, dtype=np.float64) for d in raw]
            )  # shape (n_channels, n_samples_window)

        # Software trigger en el canal del martillo.
        # v3.31.206 — Refactor: usar PEAK DETECTION en lugar de primer cruce
        # de threshold. Razón: el martillo IEPE puede tener noise floor que
        # cruza el threshold por ruido (offset DC, vibración ambiental) ANTES
        # del impacto real, contaminando la alineación → el slice quedaba con
        # el impacto al final, respuesta sin tiempo de decaer.
        #
        # Approach correcto: encontrar el sample donde |señal| es máximo →
        # ese es el peak del impacto físico. Centrar el slice 4ms antes del
        # peak garantiza que el impacto quede SIEMPRE al inicio del slice y
        # la respuesta tenga toda la duration_s para decaer.
        hammer_signal = data_arr[trigger_idx_in_list]
        threshold = config.trigger_level_V
        peak_value = float(np.abs(hammer_signal).max())

        # Validar que sí hubo un golpe (peak > threshold). El threshold se
        # interpreta en las EU del canal del martillo (N para PCB hammer).
        if peak_value < threshold:
            failed_attempts.append(
                f"Impacto {avg_idx + 1}: martillo NO superó {threshold} "
                f"(peak observado: {peak_value:.3f})"
            )
            continue

        # Encontrar el sample del peak (impacto real)
        trigger_idx = int(np.argmax(np.abs(hammer_signal)))
        start_idx = trigger_idx - pre_samples
        end_idx = start_idx + n_samples_per_impact

        if start_idx < 0:
            failed_attempts.append(
                f"Impacto {avg_idx + 1}: golpe demasiado pronto "
                f"(antes de pre_trigger {pre_samples} samples). "
                f"Espera un poquito antes de golpear."
            )
            continue
        if end_idx > n_samples_window:
            failed_attempts.append(
                f"Impacto {avg_idx + 1}: golpe demasiado tarde "
                f"(no cabe duration completa después del trigger). "
                f"Golpea más temprano en la ventana."
            )
            continue

        # Acumular slice alineado al impacto
        for ch_idx in range(len(config.channels)):
            accumulated[ch_idx] += data_arr[ch_idx][start_idx:end_idx]
        successful_impacts += 1

    if successful_impacts == 0:
        diag = "\n  - " + "\n  - ".join(failed_attempts) if failed_attempts else ""
        raise RuntimeError(
            f"No se detectaron impactos válidos en {config.n_averages} intentos.\n"
            f"Posibles causas:\n"
            f"  - Martillo no superó threshold={config.trigger_level_V}. "
            f"Baja --trigger-level (ej: 0.1) o pega más fuerte.\n"
            f"  - Sensibilidad del martillo mal configurada (--channels Hammer:BNC:IEPE:SENS)\n"
            f"  - Martillo no conectado al BNC configurado en --trigger-bnc\n"
            f"  - Tip del martillo muy blando para la estructura ensayada"
            + diag
        )

    # Promediar solo sobre impactos exitosos
    for ch_idx in range(len(config.channels)):
        accumulated[ch_idx] /= float(successful_impacts)

    if successful_impacts < config.n_averages:
        progress(
            0.92,
            f"⚠ {successful_impacts}/{config.n_averages} impactos válidos. "
            f"Continuando con promedios disponibles.",
        )

    progress(0.95, "Escribiendo TDMS...")
    # accumulated es lista de np.ndarray ahora (v3.31.205 software trigger)
    # Convertir a lista de listas para _write_tdms (que acepta ambos por compat)
    _write_tdms(config, [arr.tolist() for arr in accumulated])
    progress(
        1.0,
        f"Listo · {n_samples_per_impact} samples × {len(config.channels)} ch · "
        f"{successful_impacts} promedios válidos"
    )
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
        from nidaqmx.constants import AcquisitionType, CurrentExcitSource, Coupling
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
                # Rango en unidades de ACELERACIÓN (g), no en voltios:
                #   g_range = Vrange · 1000 / sensibilidad(mV/g)   (ej. 5V·1000/100 = ±50 g)
                _g = ch.voltage_range * 1000.0 / (ch.sensitivity_mv_per_eu or 100.0)
                task.ai_channels.add_ai_accel_chan(
                    phys, sensitivity=ch.sensitivity_mv_per_eu,
                    max_val=_g, min_val=-_g,
                    current_excit_source=CurrentExcitSource.INTERNAL,
                    current_excit_val=0.002,          # NI 9234: IEPE 2 mA
                )
            else:
                _vc = task.ai_channels.add_ai_voltage_chan(
                    phys, max_val=ch.voltage_range,
                    min_val=-ch.voltage_range,
                )
                if ch.coupling.upper() == "AC":     # proximidad: quita el DC del gap (−bias)
                    try:
                        _vc.ai_coupling = Coupling.AC
                    except Exception:  # noqa: BLE001
                        pass

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
    Genera data sintética que imita una captura real del módulo de adquisición.

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
