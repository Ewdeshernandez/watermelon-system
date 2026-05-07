"""
core.loaders.base
=================

Tipo canónico que producen todos los loaders + helper de conversión a
`core.signal_registry.Signal` (objeto que el resto del sistema consume).

Filosofía:
  - LoadedSignal es un dataclass simple, sin dependencias de Streamlit
    ni numpy fancy. Usable en tests y en CLI.
  - loaded_to_signal() lo convierte al objeto Signal histórico cuando
    haya que entregárselo a páginas/core/diagnostics. Esto hace
    completamente compatible el nuevo loader con el código existente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class LoadedSignal:
    """
    Resultado canónico de cualquier loader.

    Campos:
        file_name: nombre fuente (file path o etiqueta lógica).
        time:      vector de tiempo (s). None si la data es espectral.
        x:         vector primario (waveform o magnitud espectral).
        y:         vector secundario opcional (segunda probe, fase, etc.).
        fs:        frecuencia de muestreo Hz (para waveform).
        rpm:       RPM nominal (puede venir del header).
        units:     unidades del eje principal (g, mm/s, µm pp, mil, ...).
        domain:    "time" | "spectrum" | "polar" | "trend" | "bode".
        vendor:    "csi2140" | "adre408" | "uff" | "watermelon" | ...
        metadata:  dict libre con TODA la metadata cruda parseada.

    Invariantes:
      - len(time) == len(x) si time no es None.
      - len(y) == len(x) si y no es None.
      - x debe ser 1D y finito (NaN/Inf se filtran o se reportan en metadata).
    """
    file_name: str
    x: np.ndarray
    time: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    fs: Optional[float] = None
    rpm: Optional[float] = None
    units: str = ""
    domain: str = "time"
    vendor: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Lanza ValueError si la estructura es inconsistente."""
        if self.x is None or len(self.x) == 0:
            raise ValueError(f"{self.file_name}: x array vacío")
        if self.time is not None and len(self.time) != len(self.x):
            raise ValueError(
                f"{self.file_name}: len(time)={len(self.time)} != len(x)={len(self.x)}"
            )
        if self.y is not None and len(self.y) != len(self.x):
            raise ValueError(
                f"{self.file_name}: len(y)={len(self.y)} != len(x)={len(self.x)}"
            )

    def as_dict(self) -> Dict[str, Any]:
        """Versión serializable a JSON (numpy arrays → listas)."""
        return {
            "file_name": self.file_name,
            "time": self.time.tolist() if self.time is not None else None,
            "x": self.x.tolist(),
            "y": self.y.tolist() if self.y is not None else None,
            "fs": self.fs,
            "rpm": self.rpm,
            "units": self.units,
            "domain": self.domain,
            "vendor": self.vendor,
            "metadata": dict(self.metadata),
        }


def loaded_to_signal(loaded: LoadedSignal):
    """
    Convierte LoadedSignal al objeto Signal histórico
    (`core.signal_registry.Signal`). Importación local para no crear
    dependencia circular si alguien importa loaders sin signal_registry.
    """
    from core.signal_registry import Signal

    md = dict(loaded.metadata)
    if loaded.fs is not None:
        md.setdefault("fs", float(loaded.fs))
        md.setdefault("Sample Rate", float(loaded.fs))
    if loaded.rpm is not None:
        md.setdefault("rpm", float(loaded.rpm))
        md.setdefault("RPM", float(loaded.rpm))
    if loaded.units:
        md.setdefault("units", loaded.units)
    if loaded.domain:
        md.setdefault("domain", loaded.domain)
    if loaded.vendor:
        md.setdefault("vendor", loaded.vendor)

    time_vector = (
        loaded.time if loaded.time is not None else np.arange(len(loaded.x), dtype=float)
    )

    return Signal(
        file_name=loaded.file_name,
        time_vector=time_vector,
        x_signal=loaded.x,
        y_signal=loaded.y,
        metadata=md,
    )


# =============================================================
# Helpers compartidos para parsers
# =============================================================

def _read_text_input(source: Any, encoding_hints=("utf-8-sig", "utf-8", "latin-1")) -> str:
    """
    Acepta path-like, str (texto crudo), bytes, file-like con .read().
    Devuelve un string decodificado.

    Esta función es la única forma autorizada para que los loaders
    consuman su entrada — así toda la suite trata BOMs, encoding,
    bytes vs str, etc. de forma idéntica.
    """
    import io
    import os
    from pathlib import Path

    if source is None:
        raise ValueError("source es None")

    # path-like
    if isinstance(source, (str, Path)) and os.path.exists(str(source)):
        for enc in encoding_hints:
            try:
                with open(str(source), "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        # último recurso: leer bytes y decode con replace
        with open(str(source), "rb") as f:
            data = f.read()
        return data.decode("utf-8", errors="replace")

    # raw text
    if isinstance(source, str):
        return source

    # bytes
    if isinstance(source, (bytes, bytearray)):
        for enc in encoding_hints:
            try:
                return bytes(source).decode(enc)
            except UnicodeDecodeError:
                continue
        return bytes(source).decode("utf-8", errors="replace")

    # file-like
    if hasattr(source, "read"):
        try:
            source.seek(0)
        except Exception:
            pass
        data = source.read()
        if isinstance(data, bytes):
            for enc in encoding_hints:
                try:
                    return data.decode(enc)
                except UnicodeDecodeError:
                    continue
            return data.decode("utf-8", errors="replace")
        return str(data)

    raise TypeError(f"source de tipo no soportado: {type(source).__name__}")


def _try_float(s: Any) -> Optional[float]:
    """Intenta convertir a float; devuelve None si no se puede."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().replace(",", ".")
    if not s:
        return None
    # Quitar unidades comunes pegadas (3600 RPM, 100 mV/g)
    head = []
    for ch in s:
        if ch.isdigit() or ch in "+-.eE":
            head.append(ch)
        else:
            break
    head_s = "".join(head)
    try:
        return float(head_s)
    except ValueError:
        return None


__all__ = [
    "LoadedSignal",
    "loaded_to_signal",
    "_read_text_input",
    "_try_float",
]
