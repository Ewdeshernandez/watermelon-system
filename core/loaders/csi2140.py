"""
core.loaders.csi2140
====================

Parser de exports CSV/TXT del Emerson CSI 2140 Machinery Health Analyzer.

Formato típico (varía levemente según versión del firmware AMS):

    Route Name,Bombas Centrífugas
    Equipment,Pump 21A
    Point,Bearing 1 Vertical
    Direction,Vertical
    Date,2026-04-15
    Time,10:23:00
    Sample Rate,5120 Hz
    RPM,1780
    Number of Lines,800
    Sensitivity,100 mV/g
    Fmax,400 Hz
    Window,Hanning
    Averaging,4 Linear
    Units,g pk

    [DATA]
    Time(s),Acceleration(g)
    0.000000,0.0234
    0.000195,-0.0156
    ...

O (modo espectro):

    Frequency(Hz),Amplitude(g pk)
    0.0,0.0001
    0.5,0.0023
    ...

Estrategia:
  - Localizar el header tabular (Time/Frequency,Amplitude/Acceleration)
    como punto pivote; todo lo de arriba es metadata.
  - Detectar dominio (time vs spectrum) por la primera columna.
  - Tolerar separadores `,`, `;` o `\t`. Tolerar formato europeo
    con decimal `,` (ya cubierto por _try_float).
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.loaders.base import LoadedSignal, _read_text_input, _try_float


# Heurísticas para detectar la línea tabular
_TIME_HEADERS = {"time", "time(s)", "time (s)", "tiempo", "tiempo(s)", "t"}
_FREQ_HEADERS = {"freq", "frequency", "frequency(hz)", "frecuencia", "f", "f(hz)"}
_AMP_HEADERS = {
    "amplitude", "amp", "acceleration", "velocity", "displacement",
    "amplitud", "g", "mm/s", "um", "mil",
}


def _detect_separator(sample_lines: List[str]) -> str:
    """Elige `,`, `;` o `\\t` según ocurrencia mayoritaria."""
    counts = {",": 0, ";": 0, "\t": 0}
    for line in sample_lines:
        for sep in counts:
            counts[sep] += line.count(sep)
    return max(counts, key=counts.get)


def _normalize(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("_", "")


def _is_header_line(parts: List[str]) -> Tuple[bool, str]:
    """
    Returns (is_header, domain). domain ∈ {"time", "spectrum", ""}.
    """
    if len(parts) < 2:
        return (False, "")
    p0 = _normalize(parts[0])
    p1 = _normalize(parts[1])
    p0_is_time = any(h.replace(" ", "") == p0 for h in _TIME_HEADERS) or p0.startswith("time")
    p0_is_freq = any(h.replace(" ", "") == p0 for h in _FREQ_HEADERS) or p0.startswith("freq")
    p1_is_amp = (
        any(h.replace(" ", "") in p1 for h in _AMP_HEADERS)
        or "amp" in p1 or "accel" in p1 or "veloc" in p1 or "displ" in p1
    )
    if p0_is_time and (p1_is_amp or _try_float(parts[1]) is None):
        return (True, "time")
    if p0_is_freq and (p1_is_amp or _try_float(parts[1]) is None):
        return (True, "spectrum")
    return (False, "")


def _parse_metadata_line(line: str, sep: str) -> Optional[Tuple[str, str]]:
    """Tolera 'key,value' y 'Key: value' como entradas de metadata."""
    line = line.strip()
    if not line:
        return None
    # marker [DATA]
    if line.upper() in ("[DATA]", "DATA", "---"):
        return None

    # 'Sample Rate: 5120 Hz'  |  'Sample Rate=5120'
    for kv_sep in (":", "="):
        if kv_sep in line and (sep == "," or kv_sep not in line.split(sep)[0]):
            k, v = line.split(kv_sep, 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                return (k, v)
    # 'Sample Rate,5120 Hz'
    parts = [p.strip() for p in line.split(sep)]
    if len(parts) >= 2:
        if any(c.isalpha() for c in parts[0]):
            return (parts[0], sep.join(parts[1:]))
    return None


def _extract_units_from_header(header: List[str]) -> str:
    """A partir de 'Acceleration(g)' o 'Amplitude(mm/s)' extrae 'g' o 'mm/s'."""
    if len(header) < 2:
        return ""
    second = header[1]
    m = re.search(r"\(([^)]+)\)", second)
    if m:
        return m.group(1).strip()
    return ""


def parse_csi2140(source: Any, file_name: str = "csi2140_export") -> LoadedSignal:
    """
    Parsea un export CSV/TXT del Emerson CSI 2140 y devuelve LoadedSignal.

    Args:
        source: path | str | bytes | file-like.
        file_name: etiqueta opcional para el LoadedSignal.

    Raises:
        ValueError si no se encuentra una sección tabular reconocible.
    """
    text = _read_text_input(source)
    if not text.strip():
        raise ValueError("CSI 2140: archivo vacío")

    raw_lines = text.splitlines()
    sample = [ln for ln in raw_lines if ln.strip()][:20]
    sep = _detect_separator(sample)

    # Localizar header tabular
    header_idx = -1
    domain = ""
    header_parts: List[str] = []
    for i, line in enumerate(raw_lines):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(sep)]
        ok, d = _is_header_line(parts)
        if ok:
            header_idx = i
            domain = d
            header_parts = parts
            break

    if header_idx < 0:
        raise ValueError(
            "CSI 2140: no se encontró header de columnas (Time/Frequency, Amplitude). "
            "¿Es realmente un export de CSI 2140?"
        )

    # Metadata (todo lo previo al header tabular)
    metadata: Dict[str, Any] = {}
    for line in raw_lines[:header_idx]:
        kv = _parse_metadata_line(line, sep)
        if kv:
            metadata[kv[0]] = kv[1]

    # Extraer fs / rpm / units crudos para campos top-level
    fs = _coerce_float_with_keys(metadata, ["Sample Rate", "sample_rate", "Fs", "fs", "Sampling Rate"])
    rpm = _coerce_float_with_keys(metadata, ["RPM", "rpm", "Speed", "Sample Speed", "Machine Speed"])
    units = _extract_units_from_header(header_parts) or metadata.get("Units", "") or ""

    # Datos numéricos
    col0: List[float] = []
    col1: List[float] = []
    for line in raw_lines[header_idx + 1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(sep)]
        if len(parts) < 2:
            continue
        v0 = _try_float(parts[0])
        v1 = _try_float(parts[1])
        if v0 is None or v1 is None:
            # Línea no numérica → fin de la sección de datos
            continue
        col0.append(v0)
        col1.append(v1)

    if not col1:
        raise ValueError("CSI 2140: no se encontraron filas numéricas tras el header.")

    x = np.asarray(col1, dtype=float)
    axis = np.asarray(col0, dtype=float)

    # Construir LoadedSignal
    if domain == "time":
        time = axis
        # Si no hay fs explícito, inferirlo del time vector
        if fs is None and time.size >= 2:
            dt = float(np.median(np.diff(time)))
            if dt > 0:
                fs = 1.0 / dt
        return LoadedSignal(
            file_name=file_name,
            x=x,
            time=time,
            fs=fs,
            rpm=rpm,
            units=units,
            domain="time",
            vendor="csi2140",
            metadata=metadata,
        )

    # spectrum
    metadata.setdefault("frequency_hz", axis.tolist())
    return LoadedSignal(
        file_name=file_name,
        x=x,
        time=None,
        fs=fs,
        rpm=rpm,
        units=units,
        domain="spectrum",
        vendor="csi2140",
        metadata={**metadata, "axis_freq_hz": axis.tolist()},
    )


def _coerce_float_with_keys(meta: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in meta:
            v = _try_float(meta[k])
            if v is not None:
                return v
    return None


__all__ = ["parse_csi2140"]
