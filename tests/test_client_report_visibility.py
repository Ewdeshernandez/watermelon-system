from types import SimpleNamespace

import core.clients as clients
from core.auth import is_page_visible_in_nav_for_role
from core.reports_archive import _can_view

ACME = SimpleNamespace(id="acme", match_strings=("acme",), asset_tags=("T100A", "T100B"))


def _sc(client="", tag="", shared=False, owner="analyst@sigasas.com"):
    return {"owner_email": owner, "shared_with_client": shared,
            "report_meta": {"client": client, "instance_tag": tag}}


def _as(monkeypatch, client):
    monkeypatch.setattr(clients, "get_client_for_email", lambda e: client)


def test_client_never_sees_other_clients_report_even_if_shared(monkeypatch):
    _as(monkeypatch, ACME)
    assert not _can_view("user@acme.com", "client", _sc("Otra Empresa", "LM6000", shared=True))


def test_client_sees_own_reports_by_name_or_tag(monkeypatch):
    _as(monkeypatch, ACME)
    assert _can_view("user@acme.com", "client", _sc("Acme Energy", "X1"))
    assert _can_view("user@acme.com", "client", _sc("", "T100B", shared=True))


def test_client_without_assignment_sees_nothing(monkeypatch):
    _as(monkeypatch, None)
    assert not _can_view("nobody@x.com", "client", _sc("Acme", "T100A", shared=True))


def test_internal_roles_unchanged():
    assert _can_view("boss@sigasas.com", "admin", _sc("Otra", "Z"))
    assert _can_view("a@sigasas.com", "specialist", _sc("Otra", "Z", owner="b@sigasas.com"))


def test_rotodinamica_hidden_from_client_nav_only():
    assert not is_page_visible_in_nav_for_role("pages/02_Analysis.py", "client")
    assert is_page_visible_in_nav_for_role("pages/02_Analysis.py", "specialist")
    assert is_page_visible_in_nav_for_role("pages/02_Live_Monitoring.py", "client")
