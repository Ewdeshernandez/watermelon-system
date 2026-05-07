"""
tests.test_machine_templates
============================

Validación del catálogo extendido y su loader (`core.machine_templates`).

Reglas que validamos:
  - El JSON existe y es válido.
  - Cada plantilla tiene id único, label no vacío, RPM > 0.
  - Cada plantilla referencia normas que el catálogo iso_thresholds
    reconoce (cuando hay norma declarada).
  - get_template / list_categories / etc. devuelven shapes correctos.
  - El bridge a perfil legacy no rompe (devuelve dict válido).
"""

from __future__ import annotations

from typing import List

import pytest

from core.iso_thresholds import get_norm_metadata
from core.machine_templates import (
    MachineTemplate,
    get_catalog_metadata,
    get_template,
    list_categories,
    list_template_ids,
    list_templates,
    list_templates_by_category,
    list_templates_by_manufacturer,
    reload_catalog,
    suggest_norm_for_template,
    template_to_legacy_profile,
)


# -----------------------------------------------------------------
# Catálogo carga sin errores
# -----------------------------------------------------------------

def test_catalog_loads_at_least_15_templates():
    reload_catalog()
    templates = list_templates()
    assert len(templates) >= 15, (
        f"Esperaba >=15 plantillas en data/machine_templates.json, hay {len(templates)}"
    )


def test_catalog_metadata_has_version():
    meta = get_catalog_metadata()
    assert "version" in meta, "machine_templates.json debe tener _meta.version"


# -----------------------------------------------------------------
# Invariantes por plantilla
# -----------------------------------------------------------------

def test_each_template_has_unique_id():
    templates = list_templates()
    ids = [t.id for t in templates]
    assert len(ids) == len(set(ids)), "IDs duplicados en machine_templates.json"


def test_each_template_id_lowercase_no_spaces():
    """Disciplina de naming: ids en snake_case."""
    for t in list_templates():
        assert " " not in t.id, f"ID con espacio: '{t.id}'"
        assert t.id == t.id.lower(), f"ID no lowercase: '{t.id}'"


def test_each_template_has_positive_rpm():
    for t in list_templates():
        assert t.operating_rpm_nominal > 0, (
            f"RPM nominal inválido en {t.id}: {t.operating_rpm_nominal}"
        )


def test_each_template_rpm_range_consistent_with_nominal():
    """El nominal debe estar dentro del rango si el rango está definido."""
    for t in list_templates():
        if not t.operating_rpm_range:
            continue
        assert len(t.operating_rpm_range) == 2, (
            f"operating_rpm_range debe ser [min, max] en {t.id}"
        )
        rmin, rmax = t.operating_rpm_range
        assert rmin <= rmax, f"rango invertido en {t.id}: {t.operating_rpm_range}"
        # Tolerancia 10% — algunas plantillas listan rango "operación normal"
        # pero el nominal puede ser justo el extremo.
        rmin_tol = rmin * 0.9
        rmax_tol = rmax * 1.1
        assert rmin_tol <= t.operating_rpm_nominal <= rmax_tol, (
            f"{t.id}: nominal {t.operating_rpm_nominal} fuera de rango "
            f"{t.operating_rpm_range} (con tolerancia 10%)"
        )


def test_each_template_label_not_empty():
    for t in list_templates():
        assert t.label, f"label vacío en {t.id}"


# -----------------------------------------------------------------
# Norma referenciada existe en iso_thresholds
# -----------------------------------------------------------------

def test_each_iso_norm_is_known_to_catalog():
    """Si una plantilla declara iso_norm_recommended, ese código debe
    existir en core.iso_thresholds. Esto previene drift entre los dos
    catálogos."""
    for t in list_templates():
        if not t.iso_norm_recommended:
            continue
        meta = get_norm_metadata(t.iso_norm_recommended)
        assert meta is not None, (
            f"{t.id} referencia norma desconocida: {t.iso_norm_recommended}"
        )


# -----------------------------------------------------------------
# API queries
# -----------------------------------------------------------------

def test_list_template_ids_sorted_unique():
    ids = list_template_ids()
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_get_template_returns_object():
    ids = list_template_ids()
    if not ids:
        pytest.skip("Catálogo vacío")
    t = get_template(ids[0])
    assert isinstance(t, MachineTemplate)
    assert t.id == ids[0]


def test_get_template_unknown_returns_none():
    assert get_template("definitely-not-a-real-id-xyz-9999") is None
    assert get_template("") is None
    assert get_template(None) is None


def test_list_categories_non_empty():
    cats = list_categories()
    assert isinstance(cats, list)
    assert len(cats) > 0


def test_list_templates_by_category_filters_correctly():
    cats = list_categories()
    if not cats:
        pytest.skip("Catálogo sin categorías")
    cat = cats[0]
    filtered = list_templates_by_category(cat)
    assert all(t.category == cat for t in filtered)
    assert len(filtered) > 0


def test_list_templates_by_manufacturer_case_insensitive():
    """SOLAR / solar / Solar deben encontrar lo mismo."""
    a = list_templates_by_manufacturer("Solar")
    b = list_templates_by_manufacturer("solar")
    c = list_templates_by_manufacturer("SOLAR")
    assert {t.id for t in a} == {t.id for t in b} == {t.id for t in c}


# -----------------------------------------------------------------
# Suggestions
# -----------------------------------------------------------------

def test_suggest_norm_for_known_template():
    ids = list_template_ids()
    if not ids:
        pytest.skip("Catálogo vacío")
    norm, cls = suggest_norm_for_template(ids[0])
    # Puede ser None si la plantilla no declara norma — pero el shape es correcto
    assert (norm is None) or isinstance(norm, str)
    assert (cls is None) or isinstance(cls, str)


def test_suggest_norm_for_unknown_returns_none_pair():
    norm, cls = suggest_norm_for_template("does-not-exist")
    assert norm is None and cls is None


# -----------------------------------------------------------------
# Bridge a profile legacy
# -----------------------------------------------------------------

def test_template_to_legacy_profile_returns_dict():
    ids = list_template_ids()
    if not ids:
        pytest.skip("Catálogo vacío")
    profile = template_to_legacy_profile(ids[0])
    assert isinstance(profile, dict)
    # Campos legacy esperados (subset)
    assert "key" in profile
    assert "label" in profile
    assert "category" in profile
    assert "operating_rpm" in profile
    assert profile["operating_rpm"] > 0


def test_template_to_legacy_profile_unknown_is_none():
    assert template_to_legacy_profile("xxx-no-existe") is None


# -----------------------------------------------------------------
# Smoke: plantillas críticas que esperamos siempre presentes
# -----------------------------------------------------------------

@pytest.mark.parametrize("expected_id", [
    "brush_turbogen_54mw",
    "solar_mars_100",
    "siemens_sgt_700",
])
def test_critical_templates_present(expected_id):
    """Plantillas seed que el resto del proyecto referencia.
    Si alguien las renombra sin actualizar consumidores, el test falla."""
    t = get_template(expected_id)
    assert t is not None, f"Falta plantilla seed: {expected_id}"
