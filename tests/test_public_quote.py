import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import api.app as A  # noqa: E402
import core.billing.store as store  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    mem = {}
    monkeypatch.setattr(store, "get_setting", lambda k, d=None: mem.get(k, d))
    monkeypatch.setattr(store, "set_setting", lambda k, v: mem.__setitem__(k, v) or True)
    A._QUOTE_HITS.clear()
    return TestClient(A.create_app()), mem


def test_public_plans(client):
    c, _ = client
    r = c.get("/v1/public/plans")
    assert r.status_code == 200
    d = r.json()
    assert [p["key"] for p in d["plans"]] == ["esencial", "profesional", "integral"]
    assert {"from": 4, "discount_pct": 10} in d["volume_tiers"]
    assert d["annual_months_charged"] == 11


def test_public_quote_pdf_and_lead(client):
    c, mem = client
    r = c.post("/v1/public/quote", json={"company": "Planta Norte", "email": "ing@planta.com",
                                          "machines": 4, "plan": "integral"})
    assert r.status_code == 200 and r.content[:4] == b"%PDF"
    lead = mem["sales_requests"][0]
    assert lead["kind"] == "cotizacion" and not lead["notified"]
    assert lead["lead"]["machines"] == 4 and "−10" not in lead["detail"]


def test_public_quote_rejects_bad_input_and_bots(client):
    c, _ = client
    assert c.post("/v1/public/quote", json={"email": "no-es-correo"}).status_code == 422
    assert c.post("/v1/public/quote", json={"email": "a@b.com", "plan": "oro"}).status_code == 422
    assert c.post("/v1/public/quote", json={"email": "a@b.com", "website": "x"}).status_code == 400


def test_public_quote_rate_limit(client):
    c, _ = client
    codes = [c.post("/v1/public/quote", json={"email": "a@b.com"}).status_code for _ in range(7)]
    assert codes[:6] == [200] * 6 and codes[6] == 429


def test_notify_new_leads(client, monkeypatch):
    c, mem = client
    c.post("/v1/public/quote", json={"email": "ing@planta.com", "machines": 2})
    mem["billing_contact"] = "ventas@example.com"
    sent = []
    import core.email_sender as es
    monkeypatch.setattr(es, "send_email", lambda **kw: sent.append(kw) or {"ok": True})
    from core.billing.upsell import notify_new_leads
    assert notify_new_leads() == {"sent": 1}
    assert {s["to"] for s in sent} == {"ing@planta.com", "ventas@example.com"}
    assert sent[0]["attachments"][0][1][:4] == b"%PDF"
    assert notify_new_leads() == {"sent": 0}
