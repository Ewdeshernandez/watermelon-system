import core.billing.store as store
from core.billing import upsell

SUB = {"client_id": "cliente_x", "plan": "profesional", "machines": ["T1", "T2"],
       "billing_entity": "siga_co", "cycle": "mensual"}
EV = {"recs": [{"tag": "T2", "text": "Realizar balanceo del rotor del generador",
                "since": "24 ene 2026"}], "alarm_entries": {"1YD": 6, "4YV": 2}}


def test_integral_offer_numbers_and_breakeven():
    o = upsell.integral_offer(SUB, 50_000_000, EV)
    assert o["diff_month"] == (18_045_692 - 15_397_259) * 2
    assert o["diff_year"] == o["diff_month"] * 12
    assert o["breakeven"] == 2                      # 63,6 M / 50 M → 2 balanceos
    assert o["data_driven"] and any("1YD" in r for r in o["reasons"])
    one = upsell.integral_offer(dict(SUB, machines=["T1"]), 50_000_000, EV)
    assert one["breakeven"] == 1                    # 31,8 M < 50 M


def test_integral_offer_skipped_when_already_integral_or_usd():
    assert upsell.integral_offer(dict(SUB, plan="integral"), 50e6, EV) is None
    assert upsell.integral_offer(dict(SUB, billing_entity="wm_us"), 50e6, EV) is None


def test_add_machine_and_annual():
    m = upsell.add_machine_offer(SUB)
    assert m["extra_month"] == 15_397_259 and m["next_tier"] == (4, .10)
    m3 = upsell.add_machine_offer(dict(SUB, machines=["a", "b", "c"]))
    assert m3["extra_month"] == round(15_397_259 * 4 * .9) - 15_397_259 * 3
    a = upsell.annual_offer(SUB)
    assert a["saving_year"] == 15_397_259 * 2
    assert upsell.annual_offer(dict(SUB, cycle="anual")) is None


def test_record_and_close_request(monkeypatch):
    mem = {}
    monkeypatch.setattr(store, "get_setting", lambda k, d=None: mem.get(k, d))
    monkeypatch.setattr(store, "set_setting", lambda k, v: mem.__setitem__(k, v) or True)
    sent = []
    import core.email_sender as es
    monkeypatch.setattr(es, "send_email", lambda **kw: sent.append(kw) or {"ok": True})
    mem["billing_contact"] = "ventas@example.com"
    assert upsell.record_request(SUB, "integral", "user@cliente.com", "detalle")
    reqs = upsell.open_requests()
    assert len(reqs) == 1 and reqs[0]["kind"] == "integral"
    assert sent and sent[0]["to"] == "ventas@example.com"
    upsell.close_request(reqs[0]["at"])
    assert upsell.open_requests() == []
