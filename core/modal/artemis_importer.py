"""
core/modal/artemis_importer.py — Lector de exports legacy de Artemis Modal
==========================================================================

Permite reusar datos modales ya capturados en Artemis Modal (software externo)
durante la transición a Watermelon Modal nativo.

Formatos soportados
-------------------
Artemis exporta archivos .txt con valores numéricos sin headers. Detectamos
automáticamente el tipo de archivo por número de columnas y rango de valores:

  · 1 columna, valores negativos (-20 a -90): espectro magnitud en dB
    Ejemplos: Hammer.txt, SENSOR 1.txt, SENSOR 2.txt

  · 2 columnas, valores con signo: FRF compleja (Real / Imag)
    Ejemplos: All_frequency_response_functions.txt, DATA1.txt

Caveat importante
-----------------
Artemis NO incluye el eje de frecuencia en estos exports. El usuario debe
proveer fs (sample rate) y bw (bandwidth) — el eje se reconstruye:
  Δf = bw / (N - 1)
  freq[i] = i × Δf  para i en [0, N-1]

Norma aplicable
---------------
ISO 7626-6 §5 — Formatos de intercambio de datos modales. Este importer es
fallback para datos previos a la adopción del formato UFF/UNV recomendado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ArtemisFRF:
    """Función de respuesta en frecuencia importada de Artemis."""
    frequencies_hz: np.ndarray  # eje de frecuencia reconstruido
    magnitude_db: Optional[np.ndarray] = None  # para archivos 1-columna
    real: Optional[np.ndarray] = None  # para archivos 2-columnas
    imag: Optional[np.ndarray] = None
    source_file: str = ""
    channel_label: str = ""

    @property
    def n_bins(self) -> int:
        return len(self.frequencies_hz)

    @property
    def df(self) -> float:
        if self.n_bins < 2:
            return 0.0
        return float(self.frequencies_hz[1] - self.frequencies_hz[0])

    @property
    def is_complex_frf(self) -> bool:
        return self.real is not None and self.imag is not None

    def magnitude_linear(self) -> np.ndarray:
        """Devuelve magnitud lineal (no dB)."""
        if self.is_complex_frf:
            return np.sqrt(self.real**2 + self.imag**2)
        if self.magnitude_db is not None:
            return 10.0 ** (self.magnitude_db / 20.0)
        return np.array([])

    def phase_deg(self) -> Optional[np.ndarray]:
        """Devuelve fase en grados (solo si es FRF compleja)."""
        if not self.is_complex_frf:
            return None
        return np.degrees(np.arctan2(self.imag, self.real))


def detect_file_type(path: Path) -> str:
    """
    Detecta tipo de archivo Artemis por número de columnas.

    Returns:
        "spectrum_db" para 1-columna (espectro en dB)
        "frf_complex" para 2-columnas (Real + Imag)
        "unknown" si no coincide
    """
    with open(path, "r") as f:
        first_lines = [f.readline().strip() for _ in range(5)]

    # Contar columnas en las primeras líneas no vacías
    col_counts = []
    for line in first_lines:
        if line:
            # Artemis usa espacios o tabs como separador
            parts = line.split()
            col_counts.append(len(parts))

    if not col_counts:
        return "unknown"

    most_common = max(set(col_counts), key=col_counts.count)
    if most_common == 1:
        return "spectrum_db"
    if most_common == 2:
        return "frf_complex"
    return "unknown"


def load_artemis_file(
    path: Path,
    sample_rate_hz: float,
    bandwidth_hz: Optional[float] = None,
    channel_label: str = "",
) -> ArtemisFRF:
    """
    Carga un archivo .txt exportado de Artemis.

    Args:
        path: Ruta al archivo
        sample_rate_hz: Frecuencia de muestreo original (requerida para reconstruir eje f)
        bandwidth_hz: Ancho de banda del análisis. Si None, asume fs/2 (Nyquist).
        channel_label: Etiqueta opcional del canal (e.g. "1YA", "Hammer")

    Returns:
        ArtemisFRF con datos cargados
    """
    file_type = detect_file_type(path)
    if file_type == "unknown":
        raise ValueError(f"No se puede detectar el tipo de archivo: {path}")

    # Leer matriz numérica
    data = np.loadtxt(path)

    if data.ndim == 1:
        n_bins = len(data)
    else:
        n_bins = data.shape[0]

    # Reconstruir eje de frecuencia
    bw = bandwidth_hz if bandwidth_hz is not None else (sample_rate_hz / 2.0)
    df = bw / (n_bins - 1) if n_bins > 1 else 0.0
    frequencies_hz = np.arange(n_bins) * df

    if file_type == "spectrum_db":
        return ArtemisFRF(
            frequencies_hz=frequencies_hz,
            magnitude_db=data.flatten(),
            source_file=str(path.name),
            channel_label=channel_label,
        )
    else:  # frf_complex
        return ArtemisFRF(
            frequencies_hz=frequencies_hz,
            real=data[:, 0],
            imag=data[:, 1],
            source_file=str(path.name),
            channel_label=channel_label,
        )


def load_artemis_batch(
    folder: Path,
    sample_rate_hz: float,
    bandwidth_hz: Optional[float] = None,
) -> List[ArtemisFRF]:
    """
    Carga todos los archivos .txt de una carpeta como ArtemisFRFs.

    Usa el nombre del archivo (sin extensión) como channel_label.
    """
    results = []
    for path in sorted(folder.glob("*.txt")):
        label = path.stem
        try:
            frf = load_artemis_file(path, sample_rate_hz, bandwidth_hz, channel_label=label)
            results.append(frf)
        except (ValueError, OSError) as e:
            # Skip archivos no parseables, log para diagnóstico
            print(f"[artemis_importer] Skip {path.name}: {e}")
    return results
