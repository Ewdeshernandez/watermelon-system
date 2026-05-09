"""
core.asset_library.catalog
==========================

Registry central de la library de iconografía. Cada entrada:
  - key: identificador único usado por wizard / composer / API
  - role: 'driver' | 'driven'
  - default_label: texto que aparece arriba del icono si el caller no
                   pasa uno custom
  - oem_examples: ejemplos de OEMs típicos (para autocomplete del wizard)
  - support_type: rolling_element | fluid_film | mixed (define modo de
                  instrumentación default)
  - typical_planes: cantidad de planos esperados (para sensor map seed)
  - builder: función generadora del SVG
  - tags: lista de tags para filtrado/búsqueda
  - category: agrupación visual en el wizard
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from core.asset_library import drivers as _d
from core.asset_library import driven as _dn


ASSET_CATALOG: Dict[str, Dict[str, Any]] = {
    # ============================================================
    # DRIVERS
    # ============================================================
    "electric_motor_sleeve": {
        "role": "driver",
        "category": "Motores eléctricos",
        "default_label": "Motor AC (sleeve)",
        "oem_examples": ["WEG W22 Sleeve", "Siemens SIMOTICS HV", "ABB AMA"],
        "support_type": "fluid_film",
        "typical_planes": 2,
        "builder": _d.electric_motor_sleeve,
        "tags": ["motor", "electric", "sleeve", "hydrodynamic"],
    },
    "electric_motor_rolling": {
        "role": "driver",
        "category": "Motores eléctricos",
        "default_label": "Motor AC (rolling)",
        "oem_examples": ["TECO MAX-E", "US Motors RGZ", "WEG W22 Rolling"],
        "support_type": "rolling_element",
        "typical_planes": 2,
        "builder": _d.electric_motor_rolling,
        "tags": ["motor", "electric", "rolling", "ball-bearing"],
    },
    "gas_turbine_aero": {
        "role": "driver",
        "category": "Turbinas a gas",
        "default_label": "Turbina aero-derivativa",
        "oem_examples": ["GE LM6000 PD/PG", "GE LM2500", "GE LM5000",
                          "Rolls-Royce Trent 60", "Siemens SGT-A65"],
        "support_type": "rolling_element",  # squeeze film + rolling
        "typical_planes": 2,
        "builder": _d.gas_turbine_aero,
        "tags": ["turbine", "aero", "gas", "lm6000", "trent"],
    },
    "gas_turbine_industrial": {
        "role": "driver",
        "category": "Turbinas a gas",
        "default_label": "Turbina industrial",
        "oem_examples": ["Siemens SGT-300", "Siemens SGT-400",
                          "GE Frame 9E", "GE Frame 7FA",
                          "Mitsubishi M501"],
        "support_type": "fluid_film",
        "typical_planes": 2,
        "builder": _d.gas_turbine_industrial,
        "tags": ["turbine", "industrial", "gas", "heavy-duty", "sgt"],
    },
    "steam_turbine": {
        "role": "driver",
        "category": "Turbinas a vapor",
        "default_label": "Turbina a vapor",
        "oem_examples": ["Siemens SST-300", "Siemens SST-700",
                          "GE BB-series", "Mitsubishi MTHT",
                          "Elliott YR"],
        "support_type": "fluid_film",
        "typical_planes": 2,
        "builder": _d.steam_turbine,
        "tags": ["turbine", "steam", "condensing", "back-pressure"],
    },
    "recip_engine_8cyl_inline": {
        "role": "driver",
        "category": "Motores reciprocantes",
        "default_label": "Recip engine 8-cyl",
        "oem_examples": ["Caterpillar G3508", "Waukesha 8L-AT27GL",
                          "MWM TCG 2020 V8"],
        "support_type": "rolling_element",  # rolling main bearings común
        "typical_planes": 2,
        "builder": _d.recip_engine_8cyl_inline,
        "tags": ["engine", "reciprocating", "gas", "inline", "8cyl"],
    },
    "recip_engine_16cyl_inline": {
        "role": "driver",
        "category": "Motores reciprocantes",
        "default_label": "Recip engine 16-cyl",
        "oem_examples": ["Cooper-Bessemer GMVH-16", "Waukesha 16V-AT27GL",
                          "MWM TCG 2032 V16"],
        "support_type": "fluid_film",  # main bearings hidrodinámicos
        "typical_planes": 2,
        "builder": _d.recip_engine_16cyl_inline,
        "tags": ["engine", "reciprocating", "gas", "inline", "16cyl",
                  "integral"],
    },

    # ============================================================
    # DRIVEN
    # ============================================================
    "generator_synchronous": {
        "role": "driven",
        "category": "Generadores",
        "default_label": "Generador síncrono",
        "oem_examples": ["Brush BDAX 7-290ER", "GE 7FH2", "Siemens TLRI",
                          "ABB AMG", "Westinghouse"],
        "support_type": "fluid_film",
        "typical_planes": 2,
        "builder": _dn.generator_synchronous,
        "tags": ["generator", "synchronous", "electric"],
    },
    "centrifugal_compressor": {
        "role": "driven",
        "category": "Compresores",
        "default_label": "Compresor centrífugo",
        "oem_examples": ["Solar Mars", "MAN RB Centrifugal",
                          "Atlas Copco CB", "Dresser-Rand DATUM"],
        "support_type": "fluid_film",
        "typical_planes": 2,
        "builder": _dn.centrifugal_compressor,
        "tags": ["compressor", "centrifugal", "single-stage"],
    },
    "centrifugal_pump_single": {
        "role": "driven",
        "category": "Bombas",
        "default_label": "Bomba centrífuga",
        "oem_examples": ["KSB Etanorm", "Sulzer OHH", "Goulds 3196",
                          "Flowserve LR/LL"],
        "support_type": "rolling_element",
        "typical_planes": 2,
        "builder": _dn.centrifugal_pump_single,
        "tags": ["pump", "centrifugal", "single-stage", "API 610",
                  "OH"],
    },
    "centrifugal_pump_multistage": {
        "role": "driven",
        "category": "Bombas",
        "default_label": "Bomba multietapa",
        "oem_examples": ["Flowserve DMX", "Flowserve DDM",
                          "Sulzer MSD", "Goulds 3700"],
        "support_type": "fluid_film",  # bombas multietapa grandes suelen tener fluid film
        "typical_planes": 2,
        "builder": _dn.centrifugal_pump_multistage,
        "tags": ["pump", "centrifugal", "multistage", "BB3", "API 610",
                  "flowserve"],
    },
    "gearbox_parallel": {
        "role": "driven",  # también puede ir entre driver y driven
        "category": "Reductores",
        "default_label": "Gearbox parallel",
        "oem_examples": ["Lufkin", "Voith Vorecon", "Renk PSC",
                          "Flender Sip"],
        "support_type": "fluid_film",
        "typical_planes": 4,  # 2 input + 2 output
        "builder": _dn.gearbox_parallel,
        "tags": ["gearbox", "parallel-shaft", "API 613"],
    },
    "recip_compressor_boxer_2cyl": {
        "role": "driven",
        "category": "Compresores reciprocantes",
        "default_label": "Recip compresor 2-cyl",
        "oem_examples": ["Ariel JGE-2", "Burckhardt 2P",
                          "Dresser-Rand 2HOS"],
        "support_type": "fluid_film",
        "typical_planes": 2,
        "builder": _dn.recip_compressor_boxer_2cyl,
        "tags": ["compressor", "reciprocating", "boxer", "2cyl",
                  "API 618"],
    },
    "recip_compressor_boxer_4cyl": {
        "role": "driven",
        "category": "Compresores reciprocantes",
        "default_label": "Recip compresor 4-cyl",
        "oem_examples": ["Ariel JGM-4", "Burckhardt 4P",
                          "Dresser-Rand 4HOS"],
        "support_type": "fluid_film",
        "typical_planes": 2,
        "builder": _dn.recip_compressor_boxer_4cyl,
        "tags": ["compressor", "reciprocating", "boxer", "4cyl",
                  "API 618"],
    },
}


def list_drivers() -> List[Dict[str, Any]]:
    """Lista todos los activos role='driver' con su metadata pública."""
    return [
        {**v, "key": k} for k, v in ASSET_CATALOG.items()
        if v["role"] == "driver"
    ]


def list_driven() -> List[Dict[str, Any]]:
    """Lista todos los activos role='driven' con su metadata pública."""
    return [
        {**v, "key": k} for k, v in ASSET_CATALOG.items()
        if v["role"] == "driven"
    ]


def list_by_category(role: str = "driver") -> Dict[str, List[Dict[str, Any]]]:
    """Agrupa los activos por categoría visual para el wizard."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for k, v in ASSET_CATALOG.items():
        if v["role"] != role:
            continue
        cat = v.get("category", "Otros")
        out.setdefault(cat, []).append({**v, "key": k})
    return out


def get_asset_meta(icon_key: str) -> Optional[Dict[str, Any]]:
    """Devuelve la metadata completa o None si no existe."""
    return ASSET_CATALOG.get(icon_key)


__all__ = [
    "ASSET_CATALOG",
    "list_drivers",
    "list_driven",
    "list_by_category",
    "get_asset_meta",
]
