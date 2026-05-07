"""
core.loaders.adre408
====================

Parser de exports CSV/TXT del Bently Nevada ADRE 408 / ADRE Sxp.

ADRE 408 es el predecesor de System1; sigue muy presente en plantas
heredadas (refinerías Pemex, Ecopetrol, Petrobras subcontratistas).
Su software ADREsoftware exporta a CSV con formato de cabecera por
metadata key-value y bloque tabular al final.

Formato típico de export:

    "Header"
    "Machine","Compressor 21B"
    "Point","Bearing A Vertical"
    "Probe","8mm proximity"
    "Date","2026-04-15 10:23:00"
    "Sample Rate","2560"
    "RPM","3600"
    "Units","mils pp"
    ""
    "Time","Amplitude"
    0.000000,0.0234
    0.000391,-0.0156
    ...

Variaciones cubiertas:
  - Comillas dobles en cabecera (estilo Excel SaveAs CSV)
  - Comillas simples
  - Sin comillas
  - Separador `,` (default), `;` (export europeo) o tab
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from core.loaders.base import LoadedSignal, _read_text_input, _try_float


_DATA_HEADER_PATTERNS = [
    r'^"?\s*time\s*"?\s*[,;\t]\s*"?\s*amplitude\s*"?',
    r'^"?\s*time\s*"?\s*[,;\t]\s*"?\s*displacement\s*"?',
    r'^"?\s*time\s*"?\s*[,;\t]\s*"?\s*velocity\s*"?',
    r'^"?\s*time\s*\(s\)\s*"?\s*[,;\t]\s*"?\s*amplitude\s*"?',
    r'^"?\s*frequency\s*"?\s*[,;\t]\s*"?\s*amplitude\s*"?',
]


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _detect_separator(sample_lines: List[str]) -> str:
    counts = {",": 0, ";": 0, "\t": 0}
    for line in sample_lines:
        for sep in counts:
            counts[sep] += line.count(sep)
    return max(counts, key=counts.get) if any(counts.values()) else ","


def _is_data_header(line: str) -> Optional[str]:
    """
    Devuelve "time" o "spectrum" si la línea es el header tabular,
    None en caso contrario.
    """
    s = line.strip().lower()
    s_norm = re.sub(r"\s+", " ", s)
    for pat in _DATA_HEADER_PATTERNS:
        if re.match(pat, s_norm):
            if "frequency" in s_norm or "freq" in s_norm:
                return "spectrum"
            return "time"
    # Heurística fallback: dos columnas, primera contiene "time" o "freq"
    parts = [_strip_quotes(p) for p in re.split(r"[,;\t]", s_norm)]
    if len(parts) >= 2:
        if parts[0].startswith("time") or parts[0] == "t":
            return "time"
        if parts[0].startswith("freq") or parts[0] == "f":
            return "spectrum"
    return None


def parse_adre408(source: Any, file_name: str = "adre408_export") -> LoadedSignal:
    """
    Parsea un export CSV del Bently Nevada ADRE 408 y devuelve LoadedSignal.

    Args:
        source: path | str | bytes | file-like.
        file_name: etiqueta opcional.

    Raises:
        ValueError si no se encuentra el header de datos esperado.
    """
    text = _read_text_input(source)
    if not text.strip():
        raise ValueError("ADRE 408: archivo vacío")

    lines = text.splitlines()
    non_empty = [ln for ln in lines if ln.strip()][:30]
    sep = _detect_separator(non_empty)

    metadata: Dict[str, Any] = {}
    header_idx = -1
    domain = "time"

    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if not line.strip():
            continue

        d = _is_data_header(line)
        if d is not None:
            header_idx = i
            domain = d
            break

        # Metadata key-value
        parts = [_strip_quotes(p) for p in line.split(sep, 1)]
        if len(parts) == 2:
            k, v = parts[0].strip(), parts[1].strip()
            if k and v and k.lower() not in ("header", "metadata"):
                metadata[k] = v

    if header_idx < 0:
        raise ValueError(
            "ADRE 408: no se encontró header tabular ('Time,Amplitude' o similar). "
            "¿El archivo viene de ADREsoftware export?"
        )

    fs = _coerce_float_with_keys(metadata, ["Sample Rate", "sample_rate", "Fs", "Sampling Rate"])
    rpm = _coerce_float_with_keys(metadata, ["RPM", "rpm", "Sample Speed", "Speed"])
    units = metadata.get("Units", "")

    col0: List[float] = []
    col1: List[float] = []
    for raw in lines[header_idx + 1:]:
        if not raw.strip():
            continue
        parts = [_strip_quotes(p) for p in re.split(r"[,;\t]", raw)]
        if len(parts) < 2:
            continue
        v0 = _try_float(parts[0])
        v1 = _try_float(parts[1])
        if v0 is None or v1 is None:
            continue
        col0.append(v0)
        col1.append(v1)

    if not col1:
        raise ValueError("ADRE 408: no se encontraron filas numéricas tras el header.")

    x = np.asarray(col1, dtype=float)
    axis = np.asarray(col0, dtype=float)

    if domain == "time":
        time = axis
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
            vendor="adre408",
            metadata=metadata,
        )

    return LoadedSignal(
        file_name=file_name,
        x=x,
        time=None,
        fs=fs,
        rpm=rpm,
        units=units,
        domain="spectrum",
        vendor="adre408",
        metadata={**metadata, "axis_freq_hz": axis.tolist()},
    )


def _coerce_float_with_keys(meta: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in meta:
            v = _try_float(meta[k])
            if v is not None:
                return v
    return None


__all__ = ["parse_adre408"]
