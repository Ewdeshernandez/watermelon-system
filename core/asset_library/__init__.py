"""
core.asset_library
==================

Library curada de iconografía industrial 2D para activos rotativos.

Diseño inspirado en el estándar de los referentes internacionales del
mercado (Bently System1, Emerson AMS Machine Works) que mantienen un
catálogo de tipos de equipo (motor AC, turbina aero, generador,
compresor reciprocante boxer, etc.) y NO usan fotografías 3D.

Cada icono es un SVG vectorial puro generado en Python por una función
parametrizable. Devuelve:

  (svg_group_string, anchors_dict)

donde anchors_dict tiene las coordenadas de los puntos relevantes
(DE, NDE, axial, coupling, etc.) usados por el composer para colocar
sensor dots automáticamente.

API pública:

    from core.asset_library import get_icon, list_drivers, list_driven
    svg, anchors = get_icon("gas_turbine_aero", label="GE LM6000")

    from core.asset_library.composer import compose_train
    full_svg = compose_train(driver_key="gas_turbine_aero",
                              driven_key="generator_synchronous",
                              coupling="flexible")
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.asset_library.catalog import (
    ASSET_CATALOG,
    list_drivers,
    list_driven,
    get_asset_meta,
)


def get_icon(
    icon_key: str,
    label: str = "",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Devuelve (svg_group_string, anchors_dict) para el icono solicitado.

    icon_key debe ser una key del ASSET_CATALOG. Si no existe, raises
    KeyError.

    label es el texto opcional a mostrar arriba del icono (ej. "GE LM6000").
    Si está vacío, se usa el nombre default del catálogo.
    """
    meta = get_asset_meta(icon_key)
    if not meta:
        raise KeyError(f"Asset icon '{icon_key}' no existe en el catálogo.")

    builder = meta["builder"]
    if label == "":
        label = meta.get("default_label", icon_key)
    return builder(label=label, x_offset=x_offset, y_offset=y_offset)


__all__ = [
    "ASSET_CATALOG",
    "list_drivers",
    "list_driven",
    "get_asset_meta",
    "get_icon",
]
