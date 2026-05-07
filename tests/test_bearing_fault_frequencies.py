"""
tests.test_bearing_fault_frequencies
====================================

Validación de core.bearing_fault_frequencies.build_bearing_fault_overlay()
y comportamiento del catálogo de rodamientos.

Reglas físicas que validamos:
  - BPFO_freq = factor_BPFO × RPM   (en CPM)
  - BPFI_freq > BPFO_freq           (siempre, geometría estándar)
  - FTF_freq < 0.5 × RPM            (cage frequency siempre < shaft/2)
  - Armónicos 1x, 2x, 3x se generan correctamente.
  - Sin RPM válido → available=False con mensaje claro.
"""

from __future__ import annotations

import pytest

from core.bearing_catalog import list_bearing_catalog_options, load_bearing_catalog
from core.bearing_fault_frequencies import build_bearing_fault_overlay


# -----------------------------------------------------------------
# Catálogo cargado correctamente
# -----------------------------------------------------------------

def test_catalog_not_empty():
    df = load_bearing_catalog()
    assert not df.empty, "data/bearing_catalog.csv parece vacío o ausente"


def test_catalog_has_required_columns():
    df = load_bearing_catalog()
    required = {
        "manufacturer", "model",
        "bpfo_factor", "bpfi_factor", "bsf_factor", "ftf_factor",
    }
    assert required.issubset(df.columns)


def test_options_list_contains_skf_6319():
    """SKF 6319 está en el catálogo seed — usado en el smoke test."""
    options = list_bearing_catalog_options()
    assert any("6319" in o for o in options), f"6319 no está. Opciones: {options}"


# -----------------------------------------------------------------
# Cálculo de frecuencias para un rodamiento conocido
# -----------------------------------------------------------------

def test_overlay_skf_6319_frequencies_at_3600rpm():
    """
    SKF 6319 en seed:
      BPFO = 3.0960
      BPFI = 4.9040
      BSF  = 4.1980
      FTF  = 0.3870

    @3600 RPM ⇒
      BPFO ≈ 11146 CPM
      BPFI ≈ 17654 CPM
      BSF  ≈ 15113 CPM
      FTF  ≈ 1393  CPM
    """
    overlay = build_bearing_fault_overlay(
        selected_name="SKF 6319",
        rpm=3600.0,
        harmonic_count=3,
    )
    assert overlay["available"] is True

    families_by_name = {f["family"]: f for f in overlay["families"]}

    assert families_by_name["BPFO"]["base_freq_cpm"] == pytest.approx(3.0960 * 3600.0, rel=1e-6)
    assert families_by_name["BPFI"]["base_freq_cpm"] == pytest.approx(4.9040 * 3600.0, rel=1e-6)
    assert families_by_name["BSF"]["base_freq_cpm"] == pytest.approx(4.1980 * 3600.0, rel=1e-6)
    assert families_by_name["FTF"]["base_freq_cpm"] == pytest.approx(0.3870 * 3600.0, rel=1e-6)


def test_overlay_harmonics_count_matches():
    overlay = build_bearing_fault_overlay(
        selected_name="SKF 6319",
        rpm=1800.0,
        harmonic_count=4,
    )
    assert overlay["available"]
    for fam in overlay["families"]:
        assert len(fam["lines"]) == 4
        # Cada línea n debe tener freq = base × n
        base = fam["base_freq_cpm"]
        for line in fam["lines"]:
            assert line["freq_cpm"] == pytest.approx(base * line["harmonic"], rel=1e-9)


# -----------------------------------------------------------------
# Reglas físicas universales
# -----------------------------------------------------------------

@pytest.mark.parametrize("rpm", [600.0, 1800.0, 3600.0, 14000.0])
def test_bpfi_greater_than_bpfo_for_all_rpm(rpm):
    overlay = build_bearing_fault_overlay(
        selected_name="SKF 6319",
        rpm=rpm,
        harmonic_count=1,
    )
    assert overlay["available"]
    families = {f["family"]: f["base_freq_cpm"] for f in overlay["families"]}
    assert families["BPFI"] > families["BPFO"]


@pytest.mark.parametrize("rpm", [600.0, 1800.0, 3600.0])
def test_ftf_less_than_half_rpm(rpm):
    """Cage (FTF) siempre debe ser < shaft_freq/2."""
    overlay = build_bearing_fault_overlay(
        selected_name="SKF 6319",
        rpm=rpm,
        harmonic_count=1,
    )
    assert overlay["available"]
    ftf = next(f["base_freq_cpm"] for f in overlay["families"] if f["family"] == "FTF")
    assert ftf < 0.5 * rpm


# -----------------------------------------------------------------
# Casos de error / borde
# -----------------------------------------------------------------

def test_no_rpm_returns_unavailable():
    overlay = build_bearing_fault_overlay(
        selected_name="SKF 6319",
        rpm=None,
        harmonic_count=3,
    )
    assert overlay["available"] is False
    assert "RPM" in overlay["message"]


def test_zero_rpm_returns_unavailable():
    overlay = build_bearing_fault_overlay(
        selected_name="SKF 6319",
        rpm=0.0,
        harmonic_count=3,
    )
    assert overlay["available"] is False


def test_unknown_bearing_returns_unavailable():
    overlay = build_bearing_fault_overlay(
        selected_name="MARCA-INEXISTENTE-XYZ-9999",
        rpm=3600.0,
        harmonic_count=3,
    )
    assert overlay["available"] is False
    assert "catálogo" in overlay["message"].lower() or "catalog" in overlay["message"].lower()
