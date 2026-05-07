"""
core.loaders
============

Importadores universales para Watermelon System. Cada loader convierte
el formato propietario de un vendor en el shape canónico
(`LoadedSignal`, compat con `core.signal_registry.Signal`).

Argumento de venta directo: el cliente que viene de System1 / AMS / @ptitude
no tiene que pelearse con su data — Watermelon la lee y la analiza igual.

Loaders incluidos:
  - csi2140  : Emerson CSI 2140 CSV exports
  - adre408  : Bently Nevada ADRE 408 CSV exports
  - uff      : Universal File Format (UFF/UNV) dataset 58 — estándar SDRC

Diseño:
  - Sin dependencia de Streamlit ni I/O bloqueante en imports.
  - Todos los parsers aceptan path | str | bytes | file-like.
  - El shape de salida es estable y fácil de extender (LoadedSignal).
  - Robustos: parser inválido → ValueError con mensaje legible, NUNCA
    cuelga ni lanza errores oscuros.
"""

from __future__ import annotations

from core.loaders.base import LoadedSignal, loaded_to_signal


__all__ = [
    "LoadedSignal",
    "loaded_to_signal",
]
