"""
tests.test_save_registry
========================

Validación del save_registry() (Ciclo 20B Admin UI).
Usa un registry path temporal para no contaminar el repo.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core import clients as clients_mod


def _swap_registry_path(new_path):
    """Reemplaza temporalmente REGISTRY_PATH del módulo y limpia cache."""
    original = clients_mod.REGISTRY_PATH
    clients_mod.REGISTRY_PATH = new_path
    clients_mod.reload_registry()
    return original


def _restore_path(original):
    clients_mod.REGISTRY_PATH = original
    clients_mod.reload_registry()


def test_save_registry_writes_json_atomically():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "clients.json"
        original = _swap_registry_path(path)
        try:
            data = {
                "admins": [{"name": "test_admin", "email": "a@x.com",
                            "whatsapp_numbers": ["111"]}],
                "specialists": [],
                "clients": [{"id": "test_c", "display_name": "Test C",
                             "match_strings": ["test"], "asset_tags": [],
                             "whatsapp_numbers": [], "owner_emails": [],
                             "api_key": ""}],
            }
            clients_mod.save_registry(data)

            assert path.exists()
            loaded = json.loads(path.read_text(encoding="utf-8"))
            assert loaded["admins"][0]["email"] == "a@x.com"
            assert loaded["clients"][0]["id"] == "test_c"
            # _meta agregado/actualizado
            assert "last_updated" in loaded["_meta"]
        finally:
            _restore_path(original)


def test_save_registry_rejects_invalid_input():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "clients.json"
        original = _swap_registry_path(path)
        try:
            with pytest.raises(ValueError):
                clients_mod.save_registry("not a dict")  # type: ignore
            with pytest.raises(ValueError):
                clients_mod.save_registry({"admins": "should be a list"})  # type: ignore
        finally:
            _restore_path(original)


def test_save_registry_invalidates_cache():
    """Después de save_registry, list_clients() refleja el nuevo estado."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "clients.json"
        original = _swap_registry_path(path)
        try:
            clients_mod.save_registry({
                "admins": [],
                "specialists": [],
                "clients": [{"id": "alpha", "display_name": "Alpha",
                             "match_strings": ["alpha"], "asset_tags": [],
                             "whatsapp_numbers": [], "owner_emails": [],
                             "api_key": ""}],
            })
            assert any(c.id == "alpha" for c in clients_mod.list_clients())

            clients_mod.save_registry({
                "admins": [],
                "specialists": [],
                "clients": [{"id": "beta", "display_name": "Beta",
                             "match_strings": ["beta"], "asset_tags": [],
                             "whatsapp_numbers": [], "owner_emails": [],
                             "api_key": ""}],
            })
            assert any(c.id == "beta" for c in clients_mod.list_clients())
            assert not any(c.id == "alpha" for c in clients_mod.list_clients())
        finally:
            _restore_path(original)
