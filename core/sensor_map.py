"""
core.sensor_map
===============

Mapa de sensores de vibración por máquina (Ciclo 14c.1).

Cada Asset Instance tiene una lista de sensores configurados en
Machinery Library, donde cada sensor describe:

  - **Plano**: número correlativo desde el conductor al conducido,
    siguiendo la convención API 670 / ISO 20816-1.
    Para un tren motor + bomba: planos 1-2 = motor (DE/NDE),
    3-4 = bomba (DE/NDE).
  - **Lado y ángulo**: convención polar dividida en hemisferio L y R
    vista desde el extremo conductor del eje, con 0° arriba.
    Las sondas X-Y típicas API 670 van a +45° R (X) y +45° L (Y).
  - **Dirección**: X / Y / radial / axial.
  - **Tipo de sensor**: proximity (Desplazamiento, mil pp / µm pp),
    velocity (V, mm/s RMS), accelerometer (A, g RMS).
  - **Unidad nativa** y **setpoints individuales** (alarm / danger).
  - **Patrón de match al Point del CSV**: glob estilo ``*5807*Y*``
    que asocia los CSVs cargados al sensor correspondiente.

El label de sensor sigue convención naming industrial: ``1Y_D``
(plano 1, dirección Y, Desplazamiento), ``3X_A`` (plano 3, X,
Aceleración), ``2_RAD_V`` (plano 2, radial, Velocidad).

Ciclo 14c.1 — exporta:

  - :func:`generate_standard_sensor_map` — pre-llena 8 sensores típicos
  - :func:`resolve_sensor_for_point` — encuentra el sensor que matchea
    un Point del CSV
  - :func:`sensor_label` — formatea label estilo industrial
  - :func:`sensor_unit_family` — Proximity / Velocity / Acceleration
  - :func:`new_sensor` — constructor con defaults
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Dict, List, Optional


# Mapeo tipo de sensor → familia de medida (para Tabular List)
_TYPE_TO_FAMILY = {
    "proximity": "Proximity",
    "velocity": "Velocity",
    "accelerometer": "Acceleration",
}

# Mapeo tipo de sensor → letra de unidad (para naming convention)
_TYPE_TO_LETTER = {
    "proximity": "D",       # Desplazamiento
    "velocity": "V",
    "accelerometer": "A",
}

# Unidades nativas por defecto según tipo
_DEFAULT_UNIT_BY_TYPE = {
    "proximity": "mil pp",
    "velocity": "mm/s RMS",
    "accelerometer": "g RMS",
}


def new_sensor(
    *,
    plane: int = 1,
    plane_label: str = "",
    side: str = "L",
    angle_deg: float = 45.0,
    direction: str = "Y",
    sensor_type: str = "proximity",
    unit_native: str = "",
    alarm: float = 0.0,
    danger: float = 0.0,
    csv_match_pattern: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    """Constructor de un sensor con defaults razonables."""
    if not unit_native:
        unit_native = _DEFAULT_UNIT_BY_TYPE.get(sensor_type, "")
    return {
        "plane": int(plane),
        "plane_label": plane_label,
        "side": side,
        "angle_deg": float(angle_deg),
        "direction": direction,
        "sensor_type": sensor_type,
        "unit_native": unit_native,
        "alarm": float(alarm),
        "danger": float(danger),
        "csv_match_pattern": csv_match_pattern,
        "notes": notes,
    }


def sensor_label(sensor: Dict[str, Any]) -> str:
    """
    Devuelve label industrial corto: ``1Y_D``, ``3X_A``, ``2_RAD_V``.

    Formato: ``{plane}{direction}_{type_letter}`` cuando direction
    es X / Y. Cuando es radial o axial, formato ``{plane}_{DIR}_{letter}``.
    """
    plane = int(sensor.get("plane", 0) or 0)
    direction = str(sensor.get("direction", "") or "").strip().upper()
    sensor_type = str(sensor.get("sensor_type", "") or "").lower()
    letter = _TYPE_TO_LETTER.get(sensor_type, "?")

    if direction in ("X", "Y"):
        return f"{plane}{direction}_{letter}"
    if direction in ("RADIAL", "RAD"):
        return f"{plane}_RAD_{letter}"
    if direction in ("AXIAL", "AX"):
        return f"{plane}_AX_{letter}"
    return f"{plane}_{direction or '?'}_{letter}"


def sensor_unit_family(sensor: Dict[str, Any]) -> str:
    """Devuelve la familia de medida en el lenguaje de Tabular List."""
    return _TYPE_TO_FAMILY.get(
        str(sensor.get("sensor_type", "") or "").lower(),
        "Auto",
    )


def _normalize_for_match(text: str) -> str:
    """Normaliza un string para comparación case-insensitive sin espacios extra."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def resolve_sensor_for_point(
    sensors: List[Dict[str, Any]],
    csv_point: str,
    csv_variable: str = "",
    csv_unit: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Encuentra el sensor del mapa que matchea un Point del CSV.

    Estrategia de match (en orden de prioridad):
      1. ``csv_match_pattern`` del sensor (glob, case-insensitive)
         contra ``csv_point`` o ``csv_variable``.
      2. Match heurístico: si el csv_point contiene "(X)" o "(Y)"
         busca un sensor con misma direction. Si csv_unit indica
         g/m/s² busca accelerometer; mil/µm busca proximity; mm/s
         o in/s busca velocity. (Solo si exactamente 1 sensor matchea.)

    Args:
        sensors: lista de sensores del Instance.sensors.
        csv_point: campo Point del CSV (ej. "VE5807 (Y)").
        csv_variable: campo Variable del CSV (ej. "Disp Wf").
        csv_unit: unidad de amplitud (ej. "mil pp", "g").

    Returns:
        El sensor que matcheó (dict) o None si no encontró match.
    """
    if not sensors:
        return None

    point_norm = _normalize_for_match(csv_point)
    variable_norm = _normalize_for_match(csv_variable)
    unit_norm = _normalize_for_match(csv_unit)

    # 1. Match por csv_match_pattern (más confiable)
    for sensor in sensors:
        pattern = (sensor.get("csv_match_pattern") or "").strip()
        if not pattern:
            continue
        pattern_norm = pattern.lower()
        if (
            fnmatch.fnmatch(point_norm, pattern_norm)
            or fnmatch.fnmatch(variable_norm, pattern_norm)
        ):
            return sensor

    # 2. Match heurístico por dirección X/Y + unidad
    direction_hint = ""
    if "(x)" in point_norm or " x " in f" {point_norm} " or point_norm.endswith(" x"):
        direction_hint = "X"
    elif "(y)" in point_norm or " y " in f" {point_norm} " or point_norm.endswith(" y"):
        direction_hint = "Y"

    type_hint = ""
    if any(tok in unit_norm for tok in ("g rms", "g pk", "g p", "m/s", "m/s²", "m/s2")):
        type_hint = "accelerometer"
    elif any(tok in unit_norm for tok in ("mil", "µm", "um")):
        type_hint = "proximity"
    elif any(tok in unit_norm for tok in ("mm/s", "in/s", "ips")):
        type_hint = "velocity"

    if not type_hint and ("acell" in point_norm or "accel" in point_norm or "ace" in variable_norm):
        type_hint = "accelerometer"

    candidates = sensors
    if direction_hint:
        candidates = [s for s in candidates if str(s.get("direction", "")).upper() == direction_hint]
    if type_hint:
        candidates = [s for s in candidates if str(s.get("sensor_type", "")).lower() == type_hint]

    if len(candidates) == 1:
        return candidates[0]

    # Si quedan varios, intentar tie-break por substring del label
    for sensor in candidates:
        lbl = sensor_label(sensor).lower()
        if lbl in point_norm:
            return sensor

    return None


def generate_standard_sensor_map(
    *,
    nominal_planes_driver: int = 2,
    nominal_planes_driven: int = 2,
    include_axial_accelerometer: bool = True,
    proximity_alarm_mil_pp: float = 4.0,
    proximity_danger_mil_pp: float = 6.0,
    accel_alarm_g_rms: float = 4.5,
    accel_danger_g_rms: float = 9.0,
) -> List[Dict[str, Any]]:
    """
    Genera un mapa estándar de sensores para un tren acoplado típico:
    cada plano tiene par X-Y de proxímetros a +45° R / +45° L (API 670),
    y opcionalmente 1 acelerómetro radial en cada cojinete del driven.

    Args:
        nominal_planes_driver: cantidad de cojinetes del driver (típico 2).
        nominal_planes_driven: cantidad de cojinetes del driven (típico 2).
        include_axial_accelerometer: agrega 1 acelerómetro radial por cojinete del driven.
        proximity_alarm_mil_pp / proximity_danger_mil_pp: setpoints proximity.
        accel_alarm_g_rms / accel_danger_g_rms: setpoints acelerómetro.

    Returns:
        Lista de sensores (dicts) lista para asignar a Instance.sensors.
    """
    sensors: List[Dict[str, Any]] = []
    plane_idx = 0

    # Driver: planos 1..N
    for i in range(nominal_planes_driver):
        plane_idx += 1
        plane_lbl = "DE driver" if i == 0 else (
            "NDE driver" if i == 1 else f"Driver bearing {i + 1}"
        )
        sensors.append(new_sensor(
            plane=plane_idx, plane_label=plane_lbl, side="R", angle_deg=45.0,
            direction="X", sensor_type="proximity",
            alarm=proximity_alarm_mil_pp, danger=proximity_danger_mil_pp,
            csv_match_pattern=f"*{plane_idx}*x*",
        ))
        sensors.append(new_sensor(
            plane=plane_idx, plane_label=plane_lbl, side="L", angle_deg=45.0,
            direction="Y", sensor_type="proximity",
            alarm=proximity_alarm_mil_pp, danger=proximity_danger_mil_pp,
            csv_match_pattern=f"*{plane_idx}*y*",
        ))

    # Driven: planos siguientes
    for i in range(nominal_planes_driven):
        plane_idx += 1
        plane_lbl = "DE driven" if i == 0 else (
            "NDE driven" if i == 1 else f"Driven bearing {i + 1}"
        )
        sensors.append(new_sensor(
            plane=plane_idx, plane_label=plane_lbl, side="R", angle_deg=45.0,
            direction="X", sensor_type="proximity",
            alarm=proximity_alarm_mil_pp, danger=proximity_danger_mil_pp,
            csv_match_pattern=f"*{plane_idx}*x*",
        ))
        sensors.append(new_sensor(
            plane=plane_idx, plane_label=plane_lbl, side="L", angle_deg=45.0,
            direction="Y", sensor_type="proximity",
            alarm=proximity_alarm_mil_pp, danger=proximity_danger_mil_pp,
            csv_match_pattern=f"*{plane_idx}*y*",
        ))

        if include_axial_accelerometer:
            sensors.append(new_sensor(
                plane=plane_idx, plane_label=plane_lbl, side="top", angle_deg=0.0,
                direction="radial", sensor_type="accelerometer",
                alarm=accel_alarm_g_rms, danger=accel_danger_g_rms,
                csv_match_pattern=f"*{plane_idx}*acell*",
            ))

    return sensors


__all__ = [
    "new_sensor",
    "sensor_label",
    "sensor_unit_family",
    "resolve_sensor_for_point",
    "generate_standard_sensor_map",
]
