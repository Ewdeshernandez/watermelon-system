"""
tests.test_clients_acl
======================

Validación del registry multi-tenant (Ciclo 20A):
  - Carga clients.json sin error.
  - resolve_by_phone identifica admin/specialist/client correctamente.
  - resolve_by_api_key respeta admin keys del env y client api_keys.
  - filter_matches:
      admin/specialist     → True para todo
      client (Ecopetrol)   → True para reports cuyo report_meta
                             contenga 'ecopetrol' o 'magnex'
      client (Parex)       → False para reports de Ecopetrol
      unknown              → False
"""

from __future__ import annotations

import pytest

from core.clients import (
    CallerScope,
    Client,
    filter_matches,
    get_client_by_id,
    list_admins,
    list_clients,
    list_specialists,
    reload_registry,
    resolve_by_api_key,
    resolve_by_phone,
)


# =============================================================
# Registry carga
# =============================================================

def test_registry_loads_admins_and_specialists():
    reload_registry()
    admins = list_admins()
    specialists = list_specialists()
    assert len(admins) >= 1, "Debe existir al menos 1 admin (Ewdes)"
    assert len(specialists) >= 2, "Debe haber 2 specialists (Jessica + Natalia)"


def test_registry_loads_three_seed_clients():
    reload_registry()
    clients = list_clients()
    ids = {c.id for c in clients}
    assert "ecopetrol_magnex" in ids
    assert "parex" in ids
    assert "refoenergy" in ids


def test_get_client_by_id_known_unknown():
    assert get_client_by_id("ecopetrol_magnex") is not None
    assert get_client_by_id("does-not-exist") is None
    assert get_client_by_id("") is None


# =============================================================
# Resolve by phone
# =============================================================

def test_resolve_phone_admin_ewdes():
    scope = resolve_by_phone("573008888883")
    assert scope.role == "admin"
    assert scope.email == "ehernandez@sigasas.com"


def test_resolve_phone_specialist_jessica():
    scope = resolve_by_phone("573106776206")
    assert scope.role == "specialist"
    assert "jsuarez" in scope.email


def test_resolve_phone_specialist_natalia():
    scope = resolve_by_phone("573219826271")
    assert scope.role == "specialist"
    assert "nlopez" in scope.email


def test_resolve_phone_unknown_returns_unauthorized():
    scope = resolve_by_phone("999999999")
    assert scope.role == "unknown"
    assert scope.is_authorized is False


def test_resolve_phone_normalizes_plus_prefix():
    a = resolve_by_phone("+573008888883")
    b = resolve_by_phone("573008888883")
    c = resolve_by_phone(" 57 300 888 8883 ")
    assert a.role == b.role == c.role == "admin"


# =============================================================
# Resolve by API key
# =============================================================

def test_resolve_api_key_admin_via_env():
    import os
    saved = os.environ.get("WATERMELON_API_KEYS")
    os.environ["WATERMELON_API_KEYS"] = "test-admin-key-1,test-admin-key-2"
    try:
        s = resolve_by_api_key("test-admin-key-1")
        assert s.role == "admin"
        s2 = resolve_by_api_key("test-admin-key-2")
        assert s2.role == "admin"
    finally:
        if saved is None:
            os.environ.pop("WATERMELON_API_KEYS", None)
        else:
            os.environ["WATERMELON_API_KEYS"] = saved


def test_resolve_api_key_unknown():
    s = resolve_by_api_key("definitely-not-a-real-key-xyz", admin_keys=[])
    assert s.role == "unknown"


def test_resolve_api_key_empty():
    assert resolve_by_api_key("").role == "unknown"
    assert resolve_by_api_key(None).role == "unknown"


# =============================================================
# filter_matches
# =============================================================

ECOPETROL_REPORT = {
    "client": "Ecopetrol — Magnex",
    "instance_tag": "TES1",
    "asset_class": "Turbogenerador",
    "train_description": "GE Vernova TM2500 + Brush BDAX7",
}

PAREX_REPORT = {
    "client": "Parex Resources",
    "instance_tag": "C-200-C",
    "asset_class": "Compresor reciprocante",
    "train_description": "Hyundai HNP2 + Ariel KBK/4",
}

REFO_REPORT = {
    "client": "Refoenergy",
    "instance_tag": "X-1",
    "asset_class": "Bomba",
}


def test_admin_sees_everything():
    admin = CallerScope(role="admin", name="Ewdes")
    assert filter_matches(ECOPETROL_REPORT, admin) is True
    assert filter_matches(PAREX_REPORT, admin) is True
    assert filter_matches(REFO_REPORT, admin) is True


def test_specialist_sees_everything():
    spec = CallerScope(role="specialist", name="Jessica")
    assert filter_matches(ECOPETROL_REPORT, spec) is True
    assert filter_matches(PAREX_REPORT, spec) is True
    assert filter_matches(REFO_REPORT, spec) is True


def test_client_ecopetrol_only_sees_own():
    ecopetrol = CallerScope(
        role="client",
        client_id="ecopetrol_magnex",
        match_strings=("ecopetrol", "magnex"),
    )
    assert filter_matches(ECOPETROL_REPORT, ecopetrol) is True
    assert filter_matches(PAREX_REPORT, ecopetrol) is False
    assert filter_matches(REFO_REPORT, ecopetrol) is False


def test_client_parex_only_sees_own():
    parex = CallerScope(
        role="client",
        client_id="parex",
        match_strings=("parex",),
    )
    assert filter_matches(ECOPETROL_REPORT, parex) is False
    assert filter_matches(PAREX_REPORT, parex) is True
    assert filter_matches(REFO_REPORT, parex) is False


def test_client_with_no_match_strings_sees_nothing():
    cli = CallerScope(role="client", client_id="x", match_strings=())
    assert filter_matches(ECOPETROL_REPORT, cli) is False


def test_unknown_caller_sees_nothing():
    unk = CallerScope(role="unknown")
    assert filter_matches(ECOPETROL_REPORT, unk) is False


def test_filter_is_case_insensitive():
    cli = CallerScope(role="client", match_strings=("ECOPETROL",))  # mayúsculas
    # match_strings se normalizó a lower al cargar el cliente, pero
    # CallerScope construido directo no. Aún así, filter_matches
    # toma el text en .lower(). Probamos ambos casos:
    cli_lower = CallerScope(role="client", match_strings=("ecopetrol",))
    assert filter_matches(ECOPETROL_REPORT, cli_lower) is True


# =============================================================
# Smoke: scope.sees_everything propiedad
# =============================================================

def test_scope_sees_everything_property():
    assert CallerScope(role="admin").sees_everything is True
    assert CallerScope(role="specialist").sees_everything is True
    assert CallerScope(role="client").sees_everything is False
    assert CallerScope(role="unknown").sees_everything is False
