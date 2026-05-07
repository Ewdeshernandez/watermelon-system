"""
core.machine_templates
======================

Loader y consultor del catálogo extendido de plantillas de máquinas
(`data/machine_templates.json`).

Este módulo es **complementario** a `core/machine_profiles.py`. La idea:

  - `machine_profiles.py` mantiene los profiles internos hardcodeados
    en Python que ya usan las páginas en producción (pages/00_Machinery_Library,
    pages/01b_Machine_Map, etc.). Eso NO se toca.
  - `machine_templates.py` (este archivo) lee un catálogo JSON externo,
    fácil de editar, con metadata más rica (sensor_layout, common_bearings,
    múltiples normas, RPM range, etc.). Pensado para:
        * pre-cargar set-points cuando el usuario crea un activo nuevo
        * sugerir esquema de sensores en Machine Map
        * sugerir ISO/API recommendations en Reports
        * exportar a YAML / OpenAPI en el futuro

API pública:
    list_templates()                      -> List[Template]
    get_template(template_id)             -> Template | None
    list_template_ids()                   -> List[str]
    list_categories()                     -> List[str]
    list_templates_by_category(cat)       -> List[Template]
    list_templates_by_manufacturer(name)  -> List[Template]
    suggest_norm_for_template(tid)        -> Tuple[norm_code, class_code] | (None, None)
    template_to_legacy_profile(tid)       -> dict   # bridge a machine_profiles

Robustez:
    - Si el JSON no existe o es inválido → API devuelve listas vacías o
      None, NUNCA lanza al import. Esto garantiza que un mal release del
      catálogo no rompe la app.

No hay efectos secundarios. No imports streamlit. Pensado para tests
y CI sin entorno UI.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


log = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "machine_templates.json"


# =============================================================
# Modelo
# =============================================================

@dataclass(frozen=True)
class MachineTemplate:
    """
    Una plantilla de máquina. Todos los campos opcionales para
    tolerar entradas parciales del JSON sin romper.
    """
    id: str
    label: str
    manufacturer: str = ""
    model: str = ""
    category: str = ""
    application: List[str] = field(default_factory=list)
    rated_power_kw: List[float] = field(default_factory=list)
    operating_rpm_nominal: float = 0.0
    operating_rpm_range: List[float] = field(default_factory=list)
    bearing_type: str = ""
    iso_norm_recommended: Optional[str] = None
    iso_class_recommended: Optional[str] = None
    api_norm_recommended: Optional[str] = None
    common_bearings: List[str] = field(default_factory=list)
    sensor_layout: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Serializable a dict (útil para PDF / JSON exports)."""
        return asdict(self)


# =============================================================
# Loader (cacheado)
# =============================================================

@lru_cache(maxsize=1)
def _load_raw_catalog() -> Dict[str, Any]:
    """
    Lee el JSON una vez por proceso. lru_cache lo hace idempotente y
    barato.
    """
    if not DEFAULT_CATALOG_PATH.exists():
        log.warning("machine_templates.json no encontrado en %s", DEFAULT_CATALOG_PATH)
        return {"templates": [], "_meta": {}}

    try:
        with DEFAULT_CATALOG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.error("machine_templates.json inválido (%s). Devolviendo vacío.", e)
        return {"templates": [], "_meta": {}}

    if not isinstance(data, dict):
        log.error("machine_templates.json: root no es un dict.")
        return {"templates": [], "_meta": {}}

    if "templates" not in data or not isinstance(data["templates"], list):
        log.error("machine_templates.json: falta lista 'templates'.")
        data["templates"] = []

    return data


def reload_catalog() -> None:
    """
    Limpia la cache. Útil en dev cuando cambian el JSON sin reiniciar
    el server de Streamlit. En producción no se llama.
    """
    _load_raw_catalog.cache_clear()


def _coerce_template(entry: Dict[str, Any]) -> Optional[MachineTemplate]:
    """Convierte un dict del JSON en MachineTemplate, ignorando entradas
    sin id/label (consideradas inválidas)."""
    if not isinstance(entry, dict):
        return None
    tid = entry.get("id")
    label = entry.get("label")
    if not tid or not label:
        return None

    return MachineTemplate(
        id=str(tid),
        label=str(label),
        manufacturer=str(entry.get("manufacturer", "") or ""),
        model=str(entry.get("model", "") or ""),
        category=str(entry.get("category", "") or ""),
        application=list(entry.get("application", []) or []),
        rated_power_kw=[float(v) for v in (entry.get("rated_power_kw") or [])],
        operating_rpm_nominal=float(entry.get("operating_rpm_nominal", 0) or 0),
        operating_rpm_range=[float(v) for v in (entry.get("operating_rpm_range") or [])],
        bearing_type=str(entry.get("bearing_type", "") or ""),
        iso_norm_recommended=entry.get("iso_norm_recommended"),
        iso_class_recommended=entry.get("iso_class_recommended"),
        api_norm_recommended=entry.get("api_norm_recommended"),
        common_bearings=list(entry.get("common_bearings", []) or []),
        sensor_layout=dict(entry.get("sensor_layout", {}) or {}),
        notes=str(entry.get("notes", "") or ""),
    )


# =============================================================
# API pública
# =============================================================

def list_templates() -> List[MachineTemplate]:
    """Devuelve todas las plantillas válidas del catálogo."""
    raw = _load_raw_catalog()
    out: List[MachineTemplate] = []
    for entry in raw.get("templates", []):
        t = _coerce_template(entry)
        if t is not None:
            out.append(t)
    return out


def list_template_ids() -> List[str]:
    """IDs disponibles, ordenados alfabéticamente."""
    return sorted(t.id for t in list_templates())


def get_template(template_id: str) -> Optional[MachineTemplate]:
    """
    Recupera una plantilla por id. Devuelve None si no existe (no
    lanza, para no romper UI cuando se eliminan plantillas obsoletas).
    """
    if not template_id:
        return None
    target = str(template_id).strip()
    for t in list_templates():
        if t.id == target:
            return t
    return None


def list_categories() -> List[str]:
    """Categorías únicas presentes en el catálogo."""
    seen = set()
    out: List[str] = []
    for t in list_templates():
        if t.category and t.category not in seen:
            seen.add(t.category)
            out.append(t.category)
    return sorted(out)


def list_templates_by_category(category: str) -> List[MachineTemplate]:
    """Templates filtradas por categoría exacta."""
    cat = str(category or "").strip()
    if not cat:
        return []
    return [t for t in list_templates() if t.category == cat]


def list_templates_by_manufacturer(manufacturer: str) -> List[MachineTemplate]:
    """Templates filtradas por fabricante (case-insensitive, contains)."""
    mfg = str(manufacturer or "").strip().lower()
    if not mfg:
        return []
    return [t for t in list_templates() if mfg in t.manufacturer.lower()]


def suggest_norm_for_template(template_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Devuelve (norm_code, class_code) recomendado para la plantilla.
    Útil para autoseleccionar la norma en la UI cuando el usuario crea
    un activo a partir de una plantilla.
    """
    t = get_template(template_id)
    if t is None:
        return (None, None)
    return (t.iso_norm_recommended, t.iso_class_recommended)


def template_to_legacy_profile(template_id: str) -> Optional[Dict[str, Any]]:
    """
    Bridge: convierte una MachineTemplate al shape que usa
    `core.machine_profiles.MachineProfile` (versión legacy).

    Esto permite que el resto del sistema use el catálogo extendido sin
    cambiar machine_profiles.py. Si en el futuro se quiere migrar todo
    a templates, este puente sigue siendo útil para retrocompatibilidad.

    Devuelve None si la plantilla no existe.
    """
    t = get_template(template_id)
    if t is None:
        return None

    # Mapeo de campos extendidos → legacy. Cuando el campo legacy no
    # tiene equivalente directo, dejamos un valor sensato.
    iso_part = ""
    if t.iso_norm_recommended and t.iso_norm_recommended.startswith("ISO_"):
        iso_part = t.iso_norm_recommended.replace("ISO_", "").replace("_", "-")

    rated_min = float(t.rated_power_kw[0]) / 1000.0 if t.rated_power_kw else 0.0
    rated_max = float(t.rated_power_kw[-1]) / 1000.0 if t.rated_power_kw else 0.0

    return {
        "key": t.id,
        "label": t.label,
        "category": t.category,
        "iso_part": iso_part,
        "machine_group": t.iso_class_recommended or "",
        "operating_rpm": float(t.operating_rpm_nominal),
        "bearing_type": t.bearing_type,
        "rated_power_mw_min": rated_min,
        "rated_power_mw_max": rated_max,
        "applicable_modules": [],   # no inferimos módulos automáticamente
        "threshold_strategy": "iso",
        "oem_thresholds_um_pp": None,
        "notes": t.notes,
    }


def get_catalog_metadata() -> Dict[str, Any]:
    """Metadata del catálogo (versión, fecha, etc.)."""
    raw = _load_raw_catalog()
    meta = raw.get("_meta", {})
    return dict(meta) if isinstance(meta, dict) else {}


__all__ = [
    "MachineTemplate",
    "list_templates",
    "list_template_ids",
    "get_template",
    "list_categories",
    "list_templates_by_category",
    "list_templates_by_manufacturer",
    "suggest_norm_for_template",
    "template_to_legacy_profile",
    "get_catalog_metadata",
    "reload_catalog",
]
