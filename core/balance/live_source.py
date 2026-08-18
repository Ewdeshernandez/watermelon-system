"""
core.balance.live_source
========================

Adaptador Live Monitoring → Balanceo.

Toma el mapa de sensores de un activo y sus lecturas en vivo (tabla
`live_readings`) y expone lo que el módulo de balanceo necesita:

  - Los PLANOS de balanceo (cojinetes) con sus sondas de proximidad X/Y,
    agrupados por sección (Driver / Driven), para que el analista elija.
  - El vector síncrono 1X (magnitud + fase) por sonda, leído de las lecturas
    live (`metric` = "1X_Ampl" y "1X_Phase"), que es justo el vector que
    consume el balanceo por coeficiente de influencia.

Diseño
------
- La UNIDAD de selección es la sonda/plano (no "la parte de la máquina"): el
  balanceo se hace en planos = cojinetes. La sección (turbina / generador /
  compresor / motor / bomba) es solo el agrupador para navegar el picker.
- Un vector por plano (práctica de campo estándar). En 2 planos se usa la
  MISMA dirección (X o Y) en ambos planos por defecto.

Las funciones "puras" (group_planes_from_sensors, parse_1x_rows, pick_sensor_
for_plane) no tocan red y son testeables. Los wrappers (list_balance_planes,
latest_1x_by_sensor, capture_1x) sí leen de Supabase vía las funciones ya
existentes de Watermelon.

CONVENCIÓN DE FASE: el `1X_Phase` de live_readings debe estar en la misma
convención que el balanceo validado (referencia del keyphasor + signo del
ángulo). Validar contra una medición real antes de confiar en el auto-import;
la entrada manual queda siempre como respaldo.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _f(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


# =====================================================================
# Helpers PUROS (sin red) — testeables
# =====================================================================
# Palabras que indican la sección del tren en el plane_label. Cubre tanto la
# convención del wizard ("DE driven") como las descriptivas de máquinas
# configuradas a mano ("5YD DE generador", "compresor centrífugo", "bomba").
_DRIVEN_KW = ("driven", "generador", "generator", "alternador", "compresor",
              "compressor", "bomba", "pump", "carga", "load", "driven equipment")
_DRIVER_KW = ("driver", "turbina", "turbine", "motor", "engine",
              "gas producer", "power turbine", " gp ", " pt ")
_GEARBOX_KW = ("gearbox", "reductor", "multiplicador", "caja")


def _section_from_label(plane_label: str) -> str:
    """Clasifica un plano en Driver / Driven / Gearbox según su etiqueta.
    Robusto a etiquetas descriptivas (no solo la convención 'driven')."""
    pl = f" {str(plane_label or '').lower()} "
    if any(k in pl for k in _GEARBOX_KW):
        return "Gearbox"
    if any(k in pl for k in _DRIVEN_KW):
        return "Driven"
    if any(k in pl for k in _DRIVER_KW):
        return "Driver"
    return "Driver"  # default conservador


def group_planes_from_sensors(sensors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Agrupa las sondas de proximidad radiales (X/Y) por plano.

    Devuelve una lista ordenada por número de plano; cada entrada:
        {
          "plane": int,
          "plane_label": str,
          "section": "Driver" | "Driven",
          "sensors": [ {"label", "direction", "side", "angle_deg"}, ... ],
        }

    Solo incluye sondas proximity con dirección X/Y (las que sirven para
    balancear). Axiales, keyphasor, acelerómetros y velocímetros se omiten.
    """
    from core.sensor_map import sensor_label

    planes: Dict[int, Dict[str, Any]] = {}
    for s in (sensors or []):
        stype = str(s.get("sensor_type") or "").lower()
        direction = str(s.get("direction") or "").upper()
        if stype != "proximity":
            continue
        if direction not in ("X", "Y"):
            continue
        plane = int(s.get("plane") or 0)
        plabel = s.get("plane_label") or f"Plano {plane}"
        section = _section_from_label(plabel)
        entry = planes.setdefault(plane, {
            "plane": plane,
            "plane_label": plabel,
            "section": section,
            "sensors": [],
        })
        entry["sensors"].append({
            "label": sensor_label(s),
            "direction": direction,
            "side": s.get("side") or "",
            "angle_deg": s.get("angle_deg"),
        })
    return [planes[k] for k in sorted(planes)]


def pick_sensor_for_plane(plane: Dict[str, Any], direction: str) -> Optional[str]:
    """Elige el label de la sonda de un plano en la dirección pedida (X/Y).

    Si no existe esa dirección, cae a la otra sonda disponible del plano.
    Devuelve el sensor_label o None si el plano no tiene sondas.
    """
    direction = str(direction or "").upper()
    sensors = plane.get("sensors") or []
    for s in sensors:
        if str(s.get("direction") or "").upper() == direction:
            return s.get("label")
    return sensors[0]["label"] if sensors else None


def parse_1x_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Parsea las filas de `latest_for_instance` a un dict por sensor_label:

        { sensor_label: {"mag", "phase", "unit", "captured_at"} }

    Toma `metric` == "1X_Ampl" (magnitud) y "1X_Phase" (fase, grados).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for r in (rows or []):
        lbl = r.get("sensor_label")
        if not lbl:
            continue
        metric = str(r.get("metric") or "")
        d = out.setdefault(lbl, {})
        if metric == "1X_Ampl":
            d["mag"] = _f(r.get("value"))
            d["unit"] = r.get("unit")
            d["captured_at"] = r.get("captured_at")
        elif metric == "1X_Phase":
            d["phase"] = _f(r.get("value"))
    return out


# =====================================================================
# Wrappers con acceso a datos (Supabase / instancia)
# =====================================================================
def list_balance_planes(instance_id: str) -> List[Dict[str, Any]]:
    """Planos de balanceo del activo (agrupados por sección). [] si no existe."""
    from core.instance_state import get_instance
    inst = get_instance(instance_id)
    if inst is None:
        return []
    return group_planes_from_sensors(getattr(inst, "sensors", []) or [])


def latest_1x_by_sensor(instance_id: str) -> Dict[str, Dict[str, Any]]:
    """1X (mag+fase) más reciente por sensor_label del activo."""
    from core.live_readings import latest_for_instance
    rows = latest_for_instance(instance_id) or []
    return parse_1x_rows(rows)


def capture_1x(
    instance_id: str,
    sensor_labels: List[str],
) -> Dict[str, Optional[Tuple[float, float, Optional[str], Optional[str]]]]:
    """Captura el 1X actual de las sondas pedidas.

    Devuelve { label: (mag, phase, unit, captured_at) } o { label: None } si
    falta la magnitud o la fase 1X de esa sonda en las lecturas live.
    """
    live = latest_1x_by_sensor(instance_id)
    out: Dict[str, Optional[Tuple[float, float, Optional[str], Optional[str]]]] = {}
    for lbl in sensor_labels:
        d = live.get(lbl) or {}
        mag, ph = d.get("mag"), d.get("phase")
        if mag is None or ph is None:
            out[lbl] = None
        else:
            out[lbl] = (float(mag), float(ph), d.get("unit"), d.get("captured_at"))
    return out


__all__ = [
    "group_planes_from_sensors", "pick_sensor_for_plane", "parse_1x_rows",
    "list_balance_planes", "latest_1x_by_sensor", "capture_1x",
]
