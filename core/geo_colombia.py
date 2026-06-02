"""
core.geo_colombia
=================

Geocodificación OFFLINE de ubicaciones colombianas (campo `location` legacy
de los activos, ej. "Tame, Arauca") a coordenadas (lat, lon), para pintar el
mapa de flota del Home.

Sin red: usa diccionarios estáticos de municipios conocidos de la flota +
capitales de los 32 departamentos como fallback. `geocode("Ciudad, Depto")`
intenta primero por municipio, luego por departamento, luego None.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional, Tuple

# Municipios específicos vistos en la flota (y comunes). lat, lon.
_MUNICIPIOS: Dict[str, Tuple[float, float]] = {
    "tame": (6.4607, -71.7355),                 # Arauca
    "villavicencio": (4.1420, -73.6266),        # Meta
    "plato": (9.7889, -74.7836),                # Magdalena
    "villanueva": (4.6094, -72.2986),           # Casanare (default)
    "yopal": (5.3378, -72.3959),
    "arauca": (7.0844, -70.7591),
    "bogota": (4.7110, -74.0721),
    "barrancabermeja": (7.0653, -73.8547),
    "monteria": (8.7479, -75.8814),
    "cartagena": (10.3910, -75.4794),
    "santa marta": (11.2408, -74.1990),
    "neiva": (2.9273, -75.2819),
    "cucuta": (7.8939, -72.5078),
}

# Capitales de los 32 departamentos (fallback por departamento). lat, lon.
_DEPARTAMENTOS: Dict[str, Tuple[float, float]] = {
    "amazonas": (-4.2150, -69.9406),
    "antioquia": (6.2442, -75.5736),
    "arauca": (7.0844, -70.7591),
    "atlantico": (10.9685, -74.7813),
    "bolivar": (10.3910, -75.4794),
    "boyaca": (5.5353, -73.3678),
    "caldas": (5.0703, -75.5138),
    "caqueta": (1.6144, -75.6062),
    "casanare": (5.3378, -72.3959),
    "cauca": (2.4448, -76.6147),
    "cesar": (10.4631, -73.2532),
    "choco": (5.6919, -76.6583),
    "cordoba": (8.7479, -75.8814),
    "cundinamarca": (4.7110, -74.0721),
    "bogota": (4.7110, -74.0721),
    "guainia": (3.8653, -67.9239),
    "guaviare": (2.5664, -72.6402),
    "huila": (2.9273, -75.2819),
    "la guajira": (11.5444, -72.9072),
    "guajira": (11.5444, -72.9072),
    "magdalena": (11.2408, -74.1990),
    "meta": (4.1420, -73.6266),
    "narino": (1.2136, -77.2811),
    "norte de santander": (7.8939, -72.5078),
    "putumayo": (1.1500, -76.6483),
    "quindio": (4.5339, -75.6811),
    "risaralda": (4.8143, -75.6946),
    "san andres": (12.5833, -81.7000),
    "santander": (7.1193, -73.1227),
    "sucre": (9.3047, -75.3978),
    "tolima": (4.4389, -75.2322),
    "valle del cauca": (3.4516, -76.5320),
    "valle": (3.4516, -76.5320),
    "vaupes": (1.2530, -70.2340),
    "vichada": (6.1890, -67.4860),
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip().lower()


def geocode(location: str) -> Optional[Tuple[float, float]]:
    """Devuelve (lat, lon) para una ubicación 'Ciudad, Departamento' o None.

    Intenta: municipio exacto → departamento → None.
    """
    if not location:
        return None
    raw = _norm(location)
    parts = [p.strip() for p in re.split(r"[,/·|-]", raw) if p.strip()]

    # 1) municipio (cualquier parte que matchee)
    for p in parts:
        if p in _MUNICIPIOS:
            return _MUNICIPIOS[p]
    if raw in _MUNICIPIOS:
        return _MUNICIPIOS[raw]

    # 2) departamento (cualquier parte)
    for p in parts:
        if p in _DEPARTAMENTOS:
            return _DEPARTAMENTOS[p]

    # 3) substring match contra municipios/departamentos conocidos
    for name, coord in _MUNICIPIOS.items():
        if name in raw:
            return coord
    for name, coord in _DEPARTAMENTOS.items():
        if name in raw:
            return coord
    return None


__all__ = ["geocode"]
