"""
core.severity
=============

Cálculo unificado de severidad para Live Monitoring (Tier 0 A).

Jerarquía de fuentes de threshold (en orden de prioridad):

  1. **Sensor-specific** — `sensor.alarm` y `sensor.danger` configurados
     en el Sensor Map del activo (Machinery Library). Es la fuente de
     mayor confianza porque es valor real del cliente.

  2. **Asset-class default** — defaults por familia de activo:
     - Aero turbine (LM6000, LM5000, TM2500, Trent): thresholds del
       fabricante GE (alarm=1.0 in/s pk, danger=1.5 in/s pk).
     - Industrial turbine (SGT, MS5/6/7, Frame 9): thresholds OEM Siemens/GE.
     - Reciprocating compressor: API 618.
     - Generic rotating: ISO 20816-3 Class IV (large machines).

  3. **ISO 20816 / API 670 generic fallback** — último recurso.

  4. **Sin Norma** — no hay forma de evaluar; el módulo lo informa
     transparentemente al UI ("Sin Norma" status) en lugar de inventar.

Esto es lo que diferencia a un sistema profesional de uno amateur:
**transparencia de la fuente del threshold**. El operador puede saber si
está mirando un threshold que el cliente firmó o si es genérico.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


# ============================================================
# Thresholds por clase de activo
# ============================================================

# Cada entrada es (family, unit_lower) → (alarm, danger).
# Las unidades se normalizan a lowercase y trim antes de buscar.

_AERO_TURBINE = {
    # GE LM6000 / LM5000 / LM2500 / TM2500 / Trent — service manuals OEM
    ("Velocity",     "in/s pk"):  (1.0,  1.5),    # 25 mm/s pk alert / 38 mm/s pk trip
    ("Velocity",     "mm/s pk"):  (25.0, 38.0),
    ("Velocity",     "mm/s rms"): (10.0, 15.0),
    ("Velocity",     "in/s rms"): (0.39, 0.59),
    ("Acceleration", "g pk"):     (5.0,  8.0),
    ("Acceleration", "g rms"):    (3.5,  5.5),
    # Proximity probes en generador acoplado (Brush, Westinghouse)
    ("Proximity",    "mil pp"):   (4.0,  6.0),
    ("Proximity",    "µm pp"):    (100.0, 150.0),
    ("Proximity",    "um pp"):    (100.0, 150.0),
}

_INDUSTRIAL_TURBINE = {
    # Siemens SGT, GE Frame, Mitsubishi M501 etc.
    ("Velocity",     "in/s pk"):  (0.71, 1.0),
    ("Velocity",     "mm/s pk"):  (18.0, 25.0),
    ("Velocity",     "mm/s rms"): (7.1,  11.0),
    ("Acceleration", "g pk"):     (3.5,  5.5),
    ("Proximity",    "mil pp"):   (3.0,  4.5),
    ("Proximity",    "µm pp"):    (75.0, 110.0),
    ("Proximity",    "um pp"):    (75.0, 110.0),
}

_RECIP_COMPRESSOR = {
    # API 618 / API 11P, compresores reciprocantes alternativos
    ("Velocity",     "in/s pk"):  (0.39, 0.71),
    ("Velocity",     "mm/s rms"): (4.5,  7.1),
    ("Acceleration", "g pk"):     (5.0,  8.0),
    ("Acceleration", "g rms"):    (3.5,  5.5),
}

_ROTATING_GENERAL = {
    # ISO 20816-3 Class IV (large rigid foundation, 300+ kW)
    ("Velocity",     "in/s pk"):  (0.39, 0.61),
    ("Velocity",     "mm/s pk"):  (10.0, 15.5),
    ("Velocity",     "mm/s rms"): (4.5,  7.1),
    ("Velocity",     "in/s rms"): (0.18, 0.28),
    ("Acceleration", "g pk"):     (2.0,  5.0),
    ("Acceleration", "g rms"):    (1.4,  3.5),
    ("Proximity",    "mil pp"):   (2.5,  4.0),
    ("Proximity",    "µm pp"):    (63.0, 100.0),
    ("Proximity",    "um pp"):    (63.0, 100.0),
}

CLASS_TABLES: Dict[str, Dict[Tuple[str, str], Tuple[float, float]]] = {
    "aero_turbine":        _AERO_TURBINE,
    "industrial_turbine":  _INDUSTRIAL_TURBINE,
    "recip_compressor":    _RECIP_COMPRESSOR,
    "rotating_general":    _ROTATING_GENERAL,
}

CLASS_LABELS: Dict[str, str] = {
    "aero_turbine":       "Aero turbine (OEM)",
    "industrial_turbine": "Industrial turbine (OEM)",
    "recip_compressor":   "Recip compressor (API 618)",
    "rotating_general":   "Generic (ISO 20816-3)",
}


# ============================================================
# Detección automática de clase de activo
# ============================================================

def detect_asset_class(instance_obj: Any) -> str:
    """
    Detecta la clase de activo desde el modelo del driver/driven y el
    profile_key. Devuelve uno de:
        aero_turbine | industrial_turbine | recip_compressor | rotating_general
    """
    if instance_obj is None:
        return "rotating_general"

    driver = (getattr(instance_obj, "driver_model", "") or "").lower()
    driven = (getattr(instance_obj, "driven_model", "") or "").lower()
    profile_key = (getattr(instance_obj, "profile_key", "") or "").lower()
    asset_class = (getattr(instance_obj, "asset_class", "") or "").lower()

    haystack = " ".join([driver, driven, profile_key, asset_class])

    aero_keywords = ("lm6000", "lm5000", "lm2500", "tm2500", "trent",
                     "aero", "lm9000", "lms100")
    if any(k in haystack for k in aero_keywords):
        return "aero_turbine"

    industrial_turbine_keywords = (
        "sgt", "frame 9", "frame 7", "frame 6", "frame 5",
        "ms5", "ms6", "ms7", "ms9", "9e", "9f", "7ea", "7fa",
        "industrial turbine", "heavy duty", "ge 9e", "ge 9f",
        "gas turbine"
    )
    if any(k in haystack for k in industrial_turbine_keywords):
        return "industrial_turbine"

    recip_keywords = ("recip", "reciprocating", "compr recip", "hnp", "ariel")
    if any(k in haystack for k in recip_keywords):
        return "recip_compressor"

    return "rotating_general"


# ============================================================
# API pública
# ============================================================

def thresholds_for(
    family: str,
    unit: str,
    instance_obj: Any = None,
) -> Tuple[float, float, str]:
    """
    Devuelve (alarm, danger, source_label) para una familia + unidad
    según la clase de activo detectada.

    Si no hay match en la tabla de clase, retorna (0, 0, "no_match").
    El caller decide qué hacer (mostrar 'Sin Norma' es lo correcto).
    """
    asset_class = detect_asset_class(instance_obj)
    table = CLASS_TABLES.get(asset_class, _ROTATING_GENERAL)
    u_norm = (unit or "").lower().strip()

    if (family, u_norm) in table:
        a, d = table[(family, u_norm)]
        return (a, d, asset_class)

    return (0.0, 0.0, "no_match")


def family_from(sensor_type: str, unit: str) -> str:
    """Infiere familia de medida desde sensor_type y/o unidad."""
    s = (sensor_type or "").lower()
    u = (unit or "").lower()
    if s == "velocity" or "mm/s" in u or "in/s" in u:
        return "Velocity"
    if s == "accelerometer":
        return "Acceleration"
    if u.strip() in ("g", "g pk", "g rms") or "g pk" in u or "g rms" in u or "m/s²" in u:
        return "Acceleration"
    if s == "proximity" or "mil" in u or "µm" in u or "um pp" in u:
        return "Proximity"
    return ""


def compute_severity(
    value: Optional[float],
    sensor_match: Optional[Dict[str, Any]],
    unit: str,
    instance_obj: Any = None,
    sensor_type_hint: str = "",
) -> Dict[str, Any]:
    """
    Devuelve un dict con:
        status:      Normal | Alarma | Danger | Sin Norma | No Data
        fg:          color de texto del badge
        bg:          color de fondo del badge
        alarm:       valor de alarm efectivo usado
        danger:      valor de danger efectivo usado
        source:      'sensor_map' | 'aero_turbine' | 'industrial_turbine' |
                     'recip_compressor' | 'rotating_general' | 'no_match'

    El campo `source` permite al UI decirle al usuario de dónde viene
    el threshold — diferenciador clave vs sistemas opacos.
    """
    if value is None:
        return {
            "status": "No Data",
            "fg": "#475569", "bg": "#f1f5f9",
            "alarm": 0.0, "danger": 0.0,
            "source": "no_data",
        }

    # 1) Sensor-specific (más alta prioridad)
    sm = sensor_match or {}
    alarm_sm = float(sm.get("alarm", 0) or 0)
    danger_sm = float(sm.get("danger", 0) or 0)
    if alarm_sm > 0 and danger_sm > 0:
        alarm = alarm_sm
        danger = danger_sm
        source = "sensor_map"
    else:
        # 2) Asset-class default
        family = family_from(
            sensor_type_hint or sm.get("sensor_type", ""),
            unit,
        )
        a, d, src = thresholds_for(family, unit, instance_obj)

        # Si el sensor map tiene algun valor pero no ambos, lo respetamos parcial
        alarm = alarm_sm if alarm_sm > 0 else a
        danger = danger_sm if danger_sm > 0 else d
        source = src if src != "no_match" else (
            "sensor_map_partial" if (alarm_sm > 0 or danger_sm > 0) else "no_match"
        )

    # No hay norma aplicable
    if alarm <= 0 and danger <= 0:
        return {
            "status": "Sin Norma",
            "fg": "#92400e", "bg": "#fef3c7",
            "alarm": 0.0, "danger": 0.0,
            "source": source,
        }

    try:
        v = float(value)
    except Exception:
        return {
            "status": "No Data",
            "fg": "#475569", "bg": "#f1f5f9",
            "alarm": alarm, "danger": danger,
            "source": source,
        }

    if danger > 0 and v >= danger:
        status, fg, bg = "Danger", "#991b1b", "#fee2e2"
    elif alarm > 0 and v >= alarm:
        status, fg, bg = "Alarma", "#92400e", "#fef3c7"
    else:
        status, fg, bg = "Normal", "#166534", "#dcfce7"

    return {
        "status": status, "fg": fg, "bg": bg,
        "alarm": alarm, "danger": danger,
        "source": source,
    }


# ============================================================
# Display-only English translation of the severity taxonomy
# ------------------------------------------------------------
# The canonical severity/status strings stay Spanish everywhere in the logic and
# in persisted data (last_executive_severity, DB rows, comparisons, dict keys).
# These maps translate ONLY at the render point, so the English web UI shows
# English labels without desyncing history or breaking any comparison.
# ============================================================
STATUS_DISPLAY_EN: Dict[str, str] = {
    "Normal": "Normal",
    "Alarma": "Alarm",
    "Danger": "Danger",
    "Peligro": "Danger",
    "Sin Norma": "No Standard",
    "Sin norma": "No Standard",
    "No Data": "No Data",
    "Sin Datos": "No Data",
}

EXEC_SEVERITY_DISPLAY_EN: Dict[str, str] = {
    "CRÍTICA": "CRITICAL",
    "ACCIÓN REQUERIDA": "ACTION REQUIRED",
    "ATENCIÓN": "WARNING",
    "VIGILANCIA": "MONITOR",
    "CONDICIÓN ACEPTABLE": "ACCEPTABLE",
    "SIN ANÁLISIS": "NO ANALYSIS",
    "SIN NORMA": "NO STANDARD",
    # title / mixed case fallbacks
    "Sin análisis": "No analysis",
    "Sin analisis": "No analysis",
}


def status_display_en(status: Optional[str]) -> str:
    """English DISPLAY label for a sensor status (Normal/Alarma/Danger/Sin Norma…).
    Returns the input unchanged if unknown. Never use for comparisons."""
    if not status:
        return status or ""
    return STATUS_DISPLAY_EN.get(status, STATUS_DISPLAY_EN.get(status.strip(), status))


def exec_severity_display_en(severity: Optional[str]) -> str:
    """English DISPLAY label for an executive severity class (CRÍTICA/ATENCIÓN…).
    Case/space-insensitive; returns the input unchanged if unknown."""
    if not severity:
        return severity or ""
    s = severity.strip()
    if s in EXEC_SEVERITY_DISPLAY_EN:
        return EXEC_SEVERITY_DISPLAY_EN[s]
    up = s.upper()
    return EXEC_SEVERITY_DISPLAY_EN.get(up, severity)


__all__ = [
    "CLASS_TABLES", "CLASS_LABELS",
    "detect_asset_class", "thresholds_for",
    "family_from", "compute_severity",
    "STATUS_DISPLAY_EN", "EXEC_SEVERITY_DISPLAY_EN",
    "status_display_en", "exec_severity_display_en",
]
