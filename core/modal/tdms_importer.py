"""
core/modal/tdms_importer.py — Lector de archivos .tdms del NI-9234
===================================================================

Importa archivos TDMS nativos generados por LabVIEW SignalExpress o
nuestro propio companion script (`scripts/ni_companion/`) que captura
data del NI-9234.

Dependencias
------------
npTDMS — Lectura nativa del formato TDMS de NI (MIT license, maduro)
  pip install npTDMS

Estructura TDMS
---------------
Un archivo .tdms tiene:
  · File-level properties (metadata global)
  · Groups (típicamente uno por test setup)
  · Channels (un canal por sensor, con time + amplitude + properties)

Las properties típicas que extraemos:
  · "Sample Rate" o "wf_increment" (Δt entre muestras)
  · "Channel Name", "Channel Description"
  · "Sensitivity" si fue configurada en LabVIEW (mV/EU)
  · "IEPE Enabled" (true/false)
  · "Start Time" (timestamp UTC)

Norma aplicable
---------------
ISO 7626-6 — Recomienda formato UFF/UNV como intercambio universal. TDMS es
el formato nativo del NI y es directamente leído sin conversión intermedia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


@dataclass
class TDMSChannel:
    """Un canal del archivo TDMS con time series + metadata."""
    name: str
    group_name: str
    time_s: np.ndarray            # Vector temporal en segundos
    data: np.ndarray              # Amplitud cruda (Volts si no se escaló)
    sample_rate_hz: float
    sensitivity_mv_per_eu: Optional[float] = None  # Si fue configurado en LabVIEW
    units: Optional[str] = None   # "V", "g", "mil", etc.
    iepe_enabled: bool = False
    properties: Dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.data)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz if self.sample_rate_hz > 0 else 0.0


@dataclass
class TDMSFile:
    """Archivo TDMS completo con todos sus canales."""
    file_path: Path
    channels: List[TDMSChannel]
    file_properties: Dict = field(default_factory=dict)


def load_tdms(path: Path) -> TDMSFile:
    """
    Carga un archivo TDMS y devuelve estructura tipada.

    Requires:
        npTDMS package (`pip install npTDMS`)

    Args:
        path: Ruta al archivo .tdms

    Returns:
        TDMSFile con channels poblados
    """
    try:
        from nptdms import TdmsFile  # noqa
    except ImportError:
        raise ImportError(
            "npTDMS no está instalado. Ejecuta: pip install npTDMS"
        )

    # TODO: implementar lectura completa con extracción de:
    #   - sample rate desde wf_increment
    #   - sensitivities por canal
    #   - units
    #   - IEPE flag
    raise NotImplementedError("Fase scaffolding — implementación en próximo sprint")


def load_tdms_summary(path: Path) -> Dict:
    """
    Vista rápida del archivo TDMS sin cargar toda la data.

    Útil para mostrar al usuario qué canales tiene el archivo antes
    de procesarlo completo.
    """
    raise NotImplementedError("Fase scaffolding")
