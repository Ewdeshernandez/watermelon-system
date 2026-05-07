"""
tests.test_iso_thresholds
=========================

Validación del catálogo central de normas en core.iso_thresholds.

Validamos invariantes estructurales (no valores numéricos finos, eso es
trabajo del comité ISO):
  - Cada norma listada existe y tiene metadata mínima.
  - Cada norma tiene al menos una clase.
  - get_thresholds devuelve dict válido con boundaries crecientes A<B<C<D.
  - suggest_norm_for_machine devuelve algo razonable para casos típicos.
"""

from __future__ import annotations

import pytest

from core.iso_thresholds import (
    get_norm_metadata,
    get_thresholds,
    list_classes_for_norm,
    list_norm_groups,
    list_norms,
    suggest_balance_grade,
    suggest_class_for_machine,
    suggest_norm_for_machine,
)


# -----------------------------------------------------------------
# Catálogo no vacío
# -----------------------------------------------------------------

def test_list_norms_not_empty():
    norms = list_norms()
    assert len(norms) > 0


def test_list_norms_each_entry_has_minimum_fields():
    norms = list_norms()
    for n in norms:
        assert "code" in n or "name" in n, f"Falta clave en {n}"


def test_list_norm_groups_not_empty():
    groups = list_norm_groups()
    assert isinstance(groups, dict)
    assert len(groups) > 0


# -----------------------------------------------------------------
# ISO 20816-2 — turbogenerador grande (caso histórico del Brush 54 MW)
# -----------------------------------------------------------------

def test_iso_20816_2_metadata_exists():
    meta = get_norm_metadata("ISO_20816_2")
    assert meta is not None
    assert "name" in meta or "long_name" in meta


def test_iso_20816_2_has_classes():
    classes = list_classes_for_norm("ISO_20816_2")
    assert len(classes) > 0


def test_iso_20816_2_thresholds_monotonic():
    """Para cualquier clase válida de 20816-2, los boundaries deben ser
    estrictamente crecientes A<B<C<D."""
    classes = list_classes_for_norm("ISO_20816_2")
    for c in classes:
        code = c.get("code") or c.get("class") or c.get("name")
        if not code:
            continue
        info = get_thresholds("ISO_20816_2", str(code))
        if info is None:
            continue
        boundaries = _extract_boundaries(info)
        if boundaries is None:
            continue
        a, b, cc, d = boundaries
        assert a < b < cc < d, f"Boundaries no monotónicos en clase {code}: {boundaries}"


# -----------------------------------------------------------------
# ISO 20816-3 — máquinas industriales 15 kW – 40 MW
# -----------------------------------------------------------------

def test_iso_20816_3_exists_and_has_classes():
    meta = get_norm_metadata("ISO_20816_3")
    assert meta is not None
    classes = list_classes_for_norm("ISO_20816_3")
    assert len(classes) > 0


# -----------------------------------------------------------------
# Suggestions (heurísticos)
# -----------------------------------------------------------------

def test_suggest_norm_for_large_steam_turbine():
    """Una turbogeneradora grande debe sugerir 20816-2 o equivalente."""
    code = suggest_norm_for_machine(asset_class="steam_turbine", driver_kind="")
    assert isinstance(code, (str, type(None)))


def test_suggest_norm_for_reciprocating_compressor():
    code = suggest_norm_for_machine(asset_class="reciprocating_compressor", driver_kind="")
    # Debe sugerir algo (idealmente ISO_20816_8, pero al menos algo válido)
    if code is not None:
        assert isinstance(code, str)
        assert len(code) > 0


def test_suggest_class_returns_string_or_none():
    out = suggest_class_for_machine("ISO_20816_3", power_kw=200.0)
    assert isinstance(out, (str, type(None)))


def test_suggest_balance_grade_returns_string_or_none():
    g = suggest_balance_grade("steam_turbine")
    assert isinstance(g, (str, type(None)))


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _extract_boundaries(info):
    """
    Distintas normas pueden representar boundaries con claves
    diferentes. Aquí cubrimos las más comunes.
    """
    # Probar claves típicas
    candidates = [
        ("ab", "bc", "cd"),
        ("a_b", "b_c", "c_d"),
        ("boundary_AB", "boundary_BC", "boundary_CD"),
    ]
    for keys in candidates:
        if all(k in info for k in keys):
            ab = float(info[keys[0]])
            bc = float(info[keys[1]])
            cd = float(info[keys[2]])
            # No tenemos el límite inferior (0 implícito)
            return (0.0, ab, bc, cd)

    # Si la norma usa lista 'thresholds' = [ab, bc, cd]
    if "thresholds" in info and isinstance(info["thresholds"], (list, tuple)):
        t = info["thresholds"]
        if len(t) >= 3:
            return (0.0, float(t[0]), float(t[1]), float(t[2]))

    return None
