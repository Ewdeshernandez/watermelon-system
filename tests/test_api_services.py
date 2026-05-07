"""
tests.test_api_services
=======================

Validación de la capa de servicios pura (`api.services`) — sin
FastAPI / uvicorn instalados.

Estos tests garantizan que los endpoints REST devuelven shapes
consistentes y serializables a JSON aunque los archivos de datos
del catálogo cambien.
"""

from __future__ import annotations

import json

import pytest

from api import services
from api.auth import hash_for_log, is_valid_api_key


# =============================================================
# Health
# =============================================================

def test_health_returns_status_ok():
    out = services.get_health()
    assert out["status"] == "ok"
    assert out["service"] == "watermelon-system-api"
    assert "build" in out


def test_health_is_json_serializable():
    out = services.get_health()
    s = json.dumps(out)
    assert isinstance(s, str)


# =============================================================
# Templates
# =============================================================

def test_list_machine_templates_summary_non_empty():
    out = services.list_machine_templates_summary()
    assert isinstance(out, list)
    assert len(out) >= 15


def test_list_machine_templates_summary_required_fields():
    out = services.list_machine_templates_summary()
    for t in out:
        for key in ["id", "label", "manufacturer", "category", "operating_rpm_nominal"]:
            assert key in t, f"Falta clave {key} en {t}"


def test_list_machine_templates_filter_by_category():
    cats = services.list_template_categories()
    if not cats:
        pytest.skip("Sin categorías")
    for cat in cats:
        filtered = services.list_machine_templates_summary(category=cat)
        assert all(t["category"] == cat for t in filtered)


def test_list_machine_templates_filter_by_manufacturer():
    out = services.list_machine_templates_summary(manufacturer="Solar")
    # Si hay plantillas Solar (deberían), todas son Solar
    assert all("solar" in t["manufacturer"].lower() for t in out)


def test_get_machine_template_detail_known():
    out = services.get_machine_template_detail("brush_turbogen_54mw")
    assert out is not None
    assert out["id"] == "brush_turbogen_54mw"
    # JSON serializable
    json.dumps(out)


def test_get_machine_template_detail_unknown():
    assert services.get_machine_template_detail("does-not-exist") is None


def test_get_norm_recommendation_for_known_template():
    out = services.get_norm_recommendation_for_template("brush_turbogen_54mw")
    assert "iso_norm_code" in out
    assert out["iso_norm_code"] == "ISO_20816_2"


def test_get_norm_recommendation_for_unknown_template():
    out = services.get_norm_recommendation_for_template("xxx")
    assert out["iso_norm_code"] is None
    assert out["iso_class_code"] is None


def test_get_legacy_profile_for_template_returns_dict():
    out = services.get_legacy_profile_for_template("brush_turbogen_54mw")
    assert isinstance(out, dict)
    assert out["operating_rpm"] > 0


# =============================================================
# Norms
# =============================================================

def test_list_norms_summary_non_empty():
    out = services.list_norms_summary()
    assert isinstance(out, list)
    assert len(out) > 0


def test_list_norm_groups_summary_is_dict():
    out = services.list_norm_groups_summary()
    assert isinstance(out, dict)
    assert len(out) > 0


def test_get_norm_detail_known():
    out = services.get_norm_detail("ISO_20816_2")
    assert out is not None
    assert out["code"] == "ISO_20816_2"
    assert "metadata" in out
    assert "classes" in out


def test_get_norm_detail_unknown():
    assert services.get_norm_detail("FAKE_NORM_999") is None


# =============================================================
# Loaders advertisement
# =============================================================

def test_list_supported_loaders_advertises_all_three():
    """v1.0 debe anunciar al menos: csi2140, adre408, uff."""
    out = services.list_supported_loaders()
    vendors = {item["vendor"] for item in out}
    assert {"csi2140", "adre408", "uff"}.issubset(vendors)


def test_loaders_response_is_json_serializable():
    json.dumps(services.list_supported_loaders())


# =============================================================
# Bearings
# =============================================================

def test_list_bearings_summary_non_empty():
    out = services.list_bearings_summary(limit=10)
    assert isinstance(out, list)
    assert len(out) > 0
    # No exponer factores BPFO/BPFI en summary (eso es valor)
    for b in out:
        assert "bpfo_factor" not in b
        assert "bpfi_factor" not in b


def test_get_bearing_overlay_skf_6319_known():
    overlay = services.get_bearing_overlay(model="SKF 6319", rpm=3600.0, harmonics=3)
    assert overlay["available"] is True
    assert len(overlay["families"]) == 4


def test_get_bearing_overlay_unknown_returns_unavailable():
    overlay = services.get_bearing_overlay(model="FAKE-123", rpm=3600.0)
    assert overlay["available"] is False


# =============================================================
# Auth
# =============================================================

def test_is_valid_api_key_rejects_empty():
    assert is_valid_api_key("") is False
    assert is_valid_api_key(None) is False


def test_is_valid_api_key_rejects_when_no_keys_configured(monkeypatch=None):
    """Sin WATERMELON_API_KEYS configurada, ninguna key es válida."""
    import os
    saved = os.environ.pop("WATERMELON_API_KEYS", None)
    try:
        assert is_valid_api_key("anything") is False
    finally:
        if saved is not None:
            os.environ["WATERMELON_API_KEYS"] = saved


def test_is_valid_api_key_accepts_configured():
    import os
    os.environ["WATERMELON_API_KEYS"] = "secret-abc-123,secret-def-456"
    try:
        assert is_valid_api_key("secret-abc-123") is True
        assert is_valid_api_key("secret-def-456") is True
        assert is_valid_api_key("not-in-list") is False
    finally:
        del os.environ["WATERMELON_API_KEYS"]


def test_hash_for_log_is_short_and_stable():
    h1 = hash_for_log("my-secret-key")
    h2 = hash_for_log("my-secret-key")
    assert h1 == h2
    assert len(h1) == 12
    # Distinct keys → distinct hashes
    h3 = hash_for_log("other-key")
    assert h1 != h3


def test_hash_for_log_handles_empty():
    assert hash_for_log("") == ""
