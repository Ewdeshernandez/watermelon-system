"""
api.services
============

Capa de servicios pura. Sin FastAPI, sin Streamlit, sin Pydantic.

Cada función:
  - Recibe parámetros simples (str, int, dict) o nada.
  - Devuelve estructuras serializables a JSON (dict / list de dicts).
  - Es completamente testeable con pytest.

Si en el futuro reemplazás FastAPI por Flask, Starlite o gRPC, esta
capa NO se toca.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.iso_thresholds import (
    get_norm_metadata,
    get_thresholds,
    list_classes_for_norm,
    list_norm_groups,
    list_norms,
)
from core.machine_templates import (
    get_template,
    list_categories,
    list_template_ids,
    list_templates,
    list_templates_by_category,
    list_templates_by_manufacturer,
    suggest_norm_for_template,
    template_to_legacy_profile,
)


log = logging.getLogger(__name__)


# =============================================================
# Health
# =============================================================

def get_health() -> Dict[str, Any]:
    """Endpoint de salud. Devuelve siempre 200 con metadata de versión."""
    try:
        from core.version import get_version_info
        version = get_version_info()
    except Exception:
        version = {"version": "unknown", "environment": "unknown"}

    return {
        "status": "ok",
        "service": "watermelon-system-api",
        "api_version": "v1",
        "build": {
            "version": version.get("version"),
            "environment": version.get("environment"),
            "commit": version.get("commit"),
        },
    }


# =============================================================
# Templates (catálogo extendido)
# =============================================================

def list_machine_templates_summary(
    category: Optional[str] = None,
    manufacturer: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Lista plantillas disponibles. Por defecto todas; filtrable por
    categoría o fabricante.
    """
    if category:
        templates = list_templates_by_category(category)
    elif manufacturer:
        templates = list_templates_by_manufacturer(manufacturer)
    else:
        templates = list_templates()

    return [
        {
            "id": t.id,
            "label": t.label,
            "manufacturer": t.manufacturer,
            "model": t.model,
            "category": t.category,
            "operating_rpm_nominal": t.operating_rpm_nominal,
            "iso_norm_recommended": t.iso_norm_recommended,
        }
        for t in templates
    ]


def get_machine_template_detail(template_id: str) -> Optional[Dict[str, Any]]:
    """Detalle completo de una plantilla. None si no existe."""
    t = get_template(template_id)
    if t is None:
        return None
    return t.as_dict()


def list_template_categories() -> List[str]:
    """Categorías disponibles en el catálogo."""
    return list_categories()


def list_template_ids_only() -> List[str]:
    """Sólo los IDs (útil para autocompletes)."""
    return list_template_ids()


def get_norm_recommendation_for_template(template_id: str) -> Dict[str, Any]:
    """
    Norma + clase recomendada para una plantilla. Útil para que un
    cliente API sepa qué thresholds aplicar al ingresar una nueva
    medición.
    """
    norm, cls = suggest_norm_for_template(template_id)
    norm_meta = get_norm_metadata(norm) if norm else None
    return {
        "template_id": template_id,
        "iso_norm_code": norm,
        "iso_class_code": cls,
        "norm_metadata": dict(norm_meta) if norm_meta else None,
    }


def get_legacy_profile_for_template(template_id: str) -> Optional[Dict[str, Any]]:
    """Bridge: devuelve la plantilla en shape MachineProfile legacy."""
    return template_to_legacy_profile(template_id)


# =============================================================
# Norms (ISO/API thresholds)
# =============================================================

def list_norms_summary() -> List[Dict[str, Any]]:
    """Lista todas las normas registradas (ISO/API/IEC/...)."""
    out: List[Dict[str, Any]] = []
    for n in list_norms():
        if isinstance(n, dict):
            out.append(dict(n))
    return out


def list_norm_groups_summary() -> Dict[str, List[Dict[str, Any]]]:
    """Normas agrupadas por familia (ISO/API/etc.)."""
    raw = list_norm_groups()
    return {
        k: [dict(item) if isinstance(item, dict) else item for item in v]
        for k, v in raw.items()
    }


def get_norm_detail(norm_code: str) -> Optional[Dict[str, Any]]:
    """Metadata + lista de clases para una norma."""
    meta = get_norm_metadata(norm_code)
    if meta is None:
        return None
    classes = list_classes_for_norm(norm_code) or []
    return {
        "code": norm_code,
        "metadata": dict(meta),
        "classes": [dict(c) if isinstance(c, dict) else {"raw": c} for c in classes],
    }


def get_norm_class_thresholds(norm_code: str, class_code: str) -> Optional[Dict[str, Any]]:
    """Thresholds de una clase específica de una norma."""
    info = get_thresholds(norm_code, class_code)
    if info is None:
        return None
    return dict(info)


# =============================================================
# Loaders capability advertisement
# =============================================================

def list_supported_loaders() -> List[Dict[str, Any]]:
    """
    Devuelve la lista de formatos de import soportados por la
    instalación. Útil para que un cliente API sepa qué archivo enviar.
    """
    return [
        {
            "vendor": "watermelon",
            "name": "Watermelon native CSV",
            "format": "csv",
            "domain": ["time", "spectrum", "polar", "bode", "trend"],
            "notes": "Formato canónico Watermelon System (Bently Nevada style).",
        },
        {
            "vendor": "csi2140",
            "name": "Emerson CSI 2140",
            "format": "csv",
            "domain": ["time", "spectrum"],
            "notes": "Exports del Machinery Health Analyzer AMS Suite.",
        },
        {
            "vendor": "adre408",
            "name": "Bently Nevada ADRE 408",
            "format": "csv",
            "domain": ["time", "spectrum"],
            "notes": "Exports de ADREsoftware (precursor System1).",
        },
        {
            "vendor": "uff",
            "name": "Universal File Format",
            "format": "uff/unv",
            "domain": ["time", "spectrum"],
            "notes": "Estándar SDRC/IDEAS dataset 58. ASCII soportado; binary próximamente.",
        },
    ]


# =============================================================
# Bearings catalog (resumen público)
# =============================================================

def list_bearings_summary(limit: int = 200) -> List[Dict[str, Any]]:
    """
    Lista hasta `limit` rodamientos del catálogo. La API NO expone
    factores BPFO/BPFI que un competidor podría scrapear masivamente
    sin contrato — sólo identificación. Para el cálculo, el cliente
    debe usar `get_bearing_overlay`.
    """
    from core.bearing_catalog import load_bearing_catalog

    df = load_bearing_catalog()
    if df is None or df.empty:
        return []

    out: List[Dict[str, Any]] = []
    for _, row in df.head(limit).iterrows():
        out.append({
            "manufacturer": str(row.get("manufacturer", "") or ""),
            "model": str(row.get("model", "") or ""),
            "aliases": [
                a for a in (
                    str(row.get("alias1", "") or ""),
                    str(row.get("alias2", "") or ""),
                    str(row.get("alias3", "") or ""),
                ) if a
            ],
        })
    return out


def get_bearing_overlay(model: str, rpm: float, harmonics: int = 3) -> Dict[str, Any]:
    """
    Cálculo de frecuencias de falla para un rodamiento dado a una RPM.
    Endpoint clave de venta — un cliente puede pedirle a la API
    'dame BPFO/BPFI/BSF/FTF de SKF 6319 a 1780 RPM' y obtener una
    respuesta firmada sin necesidad de instalar SKF @ptitude.
    """
    from core.bearing_fault_frequencies import build_bearing_fault_overlay

    overlay = build_bearing_fault_overlay(
        selected_name=model,
        rpm=float(rpm),
        harmonic_count=int(harmonics),
    )
    return dict(overlay)
