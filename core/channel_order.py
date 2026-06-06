"""
core.channel_order
==================

Orden canónico de canales para las vistas multi-sensor (overview apilado de
Spectrum / Time Waveforms). Convención de operaciones (Ewdes, jun-2026):

    1. Velocidad sísmica   (1YV, 2YV — VL/VEL CRF, TRF)
    2. Aceleración         (1YA, 2YA — ACEL CRF, TRF)
    3. Proximidad          (3XD, 3YD, 4XD, 4YD — sondas por número)

Dentro de cada grupo: por plano/número natural (1 antes que 2, 5807 antes que
5810) y alfabético como desempate (CRF antes que TRF).

Detección en dos niveles:
  a) Token canónico en el nombre/punto/variable (ej. "1Y_V" → 1YV).
  b) Si no hay token, familia por unidad (in/s|mm/s → vel, g → acel,
     mil|µm → prox) o por palabras (VEL/VL, ACEL/ACC), y número natural.
"""
from __future__ import annotations

import re

_CANON = [
    "1XV", "1YV", "2XV", "2YV",          # velocidad sísmica
    "1XA", "1YA", "2XA", "2YA",          # aceleración
    "3XD", "3YD", "4XD", "4YD",          # proximidad
]


def _norm(text: str) -> str:
    return re.sub(r"[\s_\-()./]+", "", (text or "").upper())


def _family_rank(unit: str, text_norm: str) -> int:
    u = (unit or "").lower()
    if "mm/s" in u or "in/s" in u or "ips" in u:
        return 0
    if u.strip().startswith("g") or "m/s2" in u or "m/s²" in u:
        return 1
    if "mil" in u or "µm" in u or "um" in u:
        return 2
    # Fallback por tokens del nombre (CSV sin unidad)
    if "VEL" in text_norm or "VL" in text_norm:
        return 0
    if "ACEL" in text_norm or "ACC" in text_norm:
        return 1
    return 3


def channel_sort_key(name: str, unit: str = "", extra: str = "") -> tuple:
    """Key de orden canónico. Usar:
        sorted(records, key=lambda r: channel_sort_key(r.name, r.amplitude_unit,
                                                       f"{r.point} {r.variable}"))
    """
    text = _norm(f"{name} {extra}")
    for i, tok in enumerate(_CANON):
        if tok in text:
            return (0, i, 0, name or "")
    fam = _family_rank(unit, text)
    nums = re.findall(r"\d+", name or "")
    num = int(nums[0]) if nums else 999999
    return (1, fam, num, name or "")


__all__ = ["channel_sort_key"]
