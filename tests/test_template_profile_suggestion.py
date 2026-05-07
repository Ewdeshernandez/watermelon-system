"""
tests.test_template_profile_suggestion
======================================

Validación de suggest_profile_key_for_template() — la función que sugiere
qué profile_key de PROFILES usar al crear un activo desde plantilla LATAM.
"""

from __future__ import annotations

import pytest

from core.machine_profiles import PROFILES
from core.machine_templates import (
    get_template,
    list_templates,
    suggest_profile_key_for_template,
)


def test_unknown_template_returns_none():
    assert suggest_profile_key_for_template("does-not-exist") is None
    assert suggest_profile_key_for_template("") is None


def test_all_known_templates_get_a_suggestion():
    """Toda plantilla del catálogo debe recibir un profile_key sugerido
    (aunque sea custom_manual)."""
    for t in list_templates():
        suggested = suggest_profile_key_for_template(t.id)
        assert suggested is not None, f"Sin sugerencia para {t.id}"
        assert isinstance(suggested, str)


def test_suggested_profile_keys_exist_in_PROFILES():
    """Cada sugerencia debe ser una key real del catálogo legacy."""
    for t in list_templates():
        suggested = suggest_profile_key_for_template(t.id)
        if suggested is None:
            continue
        assert suggested in PROFILES, (
            f"Plantilla {t.id} sugiere {suggested} que NO existe en PROFILES"
        )


@pytest.mark.parametrize("template_id,expected_profile", [
    ("brush_turbogen_54mw", "brush_turbogenerator_54mw_3600"),
    ("ariel_kbb_recip", "reciprocating_compressor"),
    ("burckhardt_recip", "reciprocating_compressor"),
])
def test_specific_mappings(template_id, expected_profile):
    suggested = suggest_profile_key_for_template(template_id)
    assert suggested == expected_profile, (
        f"{template_id}: esperado {expected_profile}, recibido {suggested}"
    )


def test_motor_suggestions_match_rpm():
    """Motor 1780 RPM (4 polos) → motor_4pole_60hz; etc."""
    weg = suggest_profile_key_for_template("weg_w22_motor_lv")
    assert weg == "motor_4pole_60hz"  # 1780 rpm

    abb = suggest_profile_key_for_template("abb_ami_motor_hv")
    assert abb == "motor_4pole_60hz"  # 1780 rpm también


def test_pump_suggestions():
    sulzer = suggest_profile_key_for_template("sulzer_zsk_pump")
    goulds = suggest_profile_key_for_template("goulds_3700_pump")
    # Ambas son centrifugal_pump → pump_horizontal_multistage por default
    assert sulzer in ("pump_horizontal_multistage", "pump_vertical_multistage")
    assert goulds in ("pump_horizontal_multistage", "pump_vertical_multistage")
