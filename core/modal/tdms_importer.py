"""
core/modal/tdms_importer.py — Lector de archivos .tdms del módulo de adquisición
===================================================================

Importa archivos TDMS nativos generados por LabVIEW SignalExpress o
nuestro propio companion script (`scripts/capture_companion/capture.py`) que
captura data del módulo de adquisición.

Implementación
--------------
Usa `npTDMS` (open-source MIT) para parseo. La librería NO requiere driver
driver del fabricante — el TDMS es un formato binario portable que cualquier sistema
puede leer/escribir.

Estructura TDMS esperada
------------------------
File-level properties:
  · sample_rate_hz, mode (ema_triggered|oma_continuous|simulated)
  · duration_s, n_averages, captured_at_utc
  · device_name, n_channels

Group: "Acquisition"

Channels (uno por sensor) properties:
  · channel_index, coupling, sensitivity_mv_per_eu, units
  · voltage_range, wf_increment (= 1/fs), wf_start_offset

Detección automática del canal martillo
----------------------------------------
ISO 7626-5 requiere identificar el input (martillo) vs outputs (sensores).
Heurísticas en orden de prioridad:
  1. Property `name` contiene "hammer" o "martillo"
  2. Sensitivity < 10 mV/EU (martillos: 2.4 mV/N típico vs accel 100 mV/g)
  3. Kurtosis > 10 (impacto puntual tiene distribución muy peaked)

Norma aplicable
---------------
ISO 7626-6 §5 — Formatos de intercambio de datos modales
ISO 7626-5 §7.3 — Identificación de canal de excitación (input)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TDMSChannel:
    """Un canal del archivo TDMS con time series + metadata."""
    name: str
    group_name: str
    time_s: np.ndarray            # Vector temporal en segundos
    data: np.ndarray              # Amplitud en EU (después de aplicar sensitivity)
    raw_voltage: np.ndarray       # Señal cruda en Volts (antes de scaling)
    sample_rate_hz: float
    sensitivity_mv_per_eu: Optional[float] = None
    units: str = "V"              # "g", "mil", "N", "V"
    coupling: str = "AC"          # "IEPE", "AC", "DC"
    channel_index: int = 0
    iepe_enabled: bool = False
    properties: Dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.data)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz if self.sample_rate_hz > 0 else 0.0

    @property
    def kurtosis(self) -> float:
        """Kurtosis Fisher-Pearson (3 = Gaussian, >>3 = peaked → impacto)."""
        x = np.asarray(self.data, dtype=float)
        if x.size < 4:
            return 0.0
        m = x.mean()
        std = x.std()
        if std < 1e-12:
            return 0.0
        return float(((x - m) ** 4).mean() / std ** 4)

    @property
    def peak_to_rms(self) -> float:
        """Ratio peak / RMS — útil para detectar impactos (alto) vs ruido (bajo)."""
        x = np.asarray(self.data, dtype=float)
        rms = float(np.sqrt((x ** 2).mean()))
        if rms < 1e-12:
            return 0.0
        return float(np.abs(x).max() / rms)


@dataclass
class TDMSFile:
    """Archivo TDMS completo con todos sus canales y metadata global."""
    file_path: Path
    channels: List[TDMSChannel]
    file_properties: Dict = field(default_factory=dict)
    sample_rate_hz: float = 0.0
    mode: str = ""  # "ema_triggered" | "oma_continuous" | "simulated"
    captured_at_utc: str = ""
    n_averages: int = 0

    def channel_by_name(self, name: str) -> Optional[TDMSChannel]:
        for ch in self.channels:
            if ch.name == name:
                return ch
        return None

    def channel_by_index(self, idx: int) -> Optional[TDMSChannel]:
        for ch in self.channels:
            if ch.channel_index == idx:
                return ch
        return None

    def detect_hammer_channel(self) -> Optional[TDMSChannel]:
        """
        Detecta automáticamente el canal del martillo modal.

        Heurísticas en orden de confianza:
          1. Nombre contiene "hammer" o "martillo" (case-insensitive)
          2. Sensitivity baja típica de martillo (< 10 mV/EU)
          3. Kurtosis alta (>10) — impacto puntual

        Returns None si no se detecta claramente.
        """
        # 1. Por nombre
        for ch in self.channels:
            name_low = ch.name.lower()
            if "hammer" in name_low or "martillo" in name_low or name_low == "input":
                return ch

        # 2. Por sensitivity baja
        low_sens = [ch for ch in self.channels
                    if ch.sensitivity_mv_per_eu is not None
                    and 0 < ch.sensitivity_mv_per_eu < 10]
        if len(low_sens) == 1:
            return low_sens[0]

        # 3. Por kurtosis (si EMA mode)
        if self.mode == "ema_triggered" and len(self.channels) > 1:
            ranked = sorted(self.channels, key=lambda c: c.kurtosis, reverse=True)
            top = ranked[0]
            second = ranked[1]
            if top.kurtosis > 10 and top.kurtosis > 2 * second.kurtosis:
                return top

        return None

    def response_channels(self) -> List[TDMSChannel]:
        """Canales de respuesta (output) = todos menos el martillo."""
        hammer = self.detect_hammer_channel()
        if hammer is None:
            return list(self.channels)
        return [ch for ch in self.channels if ch.name != hammer.name]


def load_tdms(path: Path) -> TDMSFile:
    """
    Carga un archivo TDMS completo y devuelve estructura tipada.

    Aplica scaling automático: si la property `sensitivity_mv_per_eu` está
    presente en el canal, convierte voltage → engineering units. Si no,
    devuelve la data cruda (assumed to be in Volts o ya escalada).
    """
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise ImportError(
            "npTDMS no está instalado. Ejecuta: pip install npTDMS"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TDMS file not found: {path}")

    tdms = TdmsFile.read(str(path))

    # File-level properties
    file_props = dict(tdms.properties or {})
    fs = float(file_props.get("sample_rate_hz", 0.0))
    mode = str(file_props.get("mode", "") or "")
    n_avg = int(file_props.get("n_averages", 0) or 0)
    captured_at = str(file_props.get("captured_at_utc", "") or "")

    channels: List[TDMSChannel] = []
    for group in tdms.groups():
        for ch_obj in group.channels():
            raw = np.asarray(ch_obj.data, dtype=float)
            ch_props = dict(ch_obj.properties or {})

            # Recover sample rate per-channel si está en wf_increment
            wf_inc = ch_props.get("wf_increment")
            ch_fs = (1.0 / float(wf_inc)) if wf_inc else fs
            if ch_fs <= 0:
                ch_fs = 1.0  # fallback para evitar div by zero

            n = raw.size
            time_s = np.arange(n) / ch_fs

            sens = ch_props.get("sensitivity_mv_per_eu")
            sens_f = float(sens) if sens is not None else None
            units = str(ch_props.get("units", "V") or "V")
            coupling = str(ch_props.get("coupling", "AC") or "AC").upper()

            # Apply scaling si tenemos sensitivity y la unidad NO es V
            # (esto significa que la data ya está en EU si units != V)
            if units == "V" and sens_f and sens_f > 0:
                # raw está en V → convertir a EU
                data_eu = raw / (sens_f / 1000.0)
                # Adivinar unit por coupling+sensitivity
                if coupling == "IEPE" and 50 <= sens_f <= 200:
                    units = "g"
                elif coupling == "AC" and 100 <= sens_f <= 250:
                    units = "mil"
                elif sens_f < 10:
                    units = "N"
            else:
                # Data ya está escalada (modo simulated o sin sensitivity)
                data_eu = raw

            channels.append(TDMSChannel(
                name=ch_obj.name,
                group_name=group.name,
                time_s=time_s,
                data=data_eu,
                raw_voltage=raw,
                sample_rate_hz=ch_fs,
                sensitivity_mv_per_eu=sens_f,
                units=units,
                coupling=coupling,
                channel_index=int(ch_props.get("channel_index", 0) or 0),
                iepe_enabled=(coupling == "IEPE"),
                properties=ch_props,
            ))

    return TDMSFile(
        file_path=path,
        channels=channels,
        file_properties=file_props,
        sample_rate_hz=fs if fs > 0 else (channels[0].sample_rate_hz if channels else 0.0),
        mode=mode,
        captured_at_utc=captured_at,
        n_averages=n_avg,
    )


def load_tdms_summary(path: Path) -> Dict:
    """
    Vista rápida del archivo TDMS sin cargar toda la data en memoria.

    Útil para mostrar al usuario qué canales tiene el archivo antes
    de procesarlo completo.
    """
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise ImportError("npTDMS no instalado") from exc

    tdms = TdmsFile.read(str(path))
    file_props = dict(tdms.properties or {})

    channels_summary = []
    for group in tdms.groups():
        for ch_obj in group.channels():
            ch_props = dict(ch_obj.properties or {})
            channels_summary.append({
                "name": ch_obj.name,
                "group": group.name,
                "n_samples": len(ch_obj),
                "channel_index": ch_props.get("channel_index"),
                "coupling": ch_props.get("coupling", ""),
                "sensitivity_mv_per_eu": ch_props.get("sensitivity_mv_per_eu"),
                "units": ch_props.get("units", "V"),
            })

    return {
        "file_path": str(path),
        "sample_rate_hz": file_props.get("sample_rate_hz"),
        "mode": file_props.get("mode"),
        "captured_at_utc": file_props.get("captured_at_utc"),
        "n_averages": file_props.get("n_averages"),
        "n_channels": len(channels_summary),
        "channels": channels_summary,
    }
