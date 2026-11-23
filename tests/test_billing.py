from datetime import date

import pytest

from core.billing import periods, pricing
from core.billing.cycle import build_invoice_row


def test_volume_tiers():
    assert [pricing.volume_discount(n) for n in (1, 3, 4, 5, 6, 9, 10, 25)] == \
        [0, 0, .10, .10, .15, .15, .20, .20]
    assert pricing.next_tier(3) == (4, .10)
    assert pricing.next_tier(5) == (6, .15)
    assert pricing.next_tier(12) is None


def test_quote_cop_monthly_list_price():
    q = pricing.quote("profesional", 2)
    assert q["currency"] == "COP" and q["unit_price"] == 15_397_259
    assert q["amount"] == 2 * 15_397_259 and q["savings"] == 0


def test_quote_volume_and_annual_11_of_12():
    q = pricing.quote("integral", 4, cycle="anual")
    assert q["months"] == 12 and q["months_charged"] == 11
    assert q["subtotal"] == 18_045_692 * 4 * 12
    assert q["amount"] == round(18_045_692 * 4 * 0.9 * 11)


def test_quote_usd_discount_and_locked_unit():
    q = pricing.quote("profesional", 2, "wm_us", price_override=None, usd_unit=4000, trm=3500)
    assert q["currency"] == "USD" and q["unit_price"] == 4000
    assert q["amount"] == round(4000 * 2 * 0.95, 2)
    q2 = pricing.quote("esencial", 1, "wm_us", trm=4000)
    assert q2["unit_price"] == 11_500_000 / 4000


def test_quote_override_and_errors():
    assert pricing.quote("integral", 1, price_override=17_000_000)["amount"] == 17_000_000
    with pytest.raises(ValueError):
        pricing.quote("oro", 1)
    with pytest.raises(ValueError):
        pricing.quote("esencial", 1, "wm_us")      # sin TRM ni USD fijo


def test_fmt_money():
    assert pricing.fmt_money(15_397_259, "COP") == "COP 15.397.259"
    assert pricing.fmt_money(4011.5, "USD") == "USD 4.011,50"


def test_periods():
    assert periods.add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert periods.period_bounds(date(2026, 12, 1)) == (date(2026, 12, 1), date(2026, 12, 31))
    assert periods.period_bounds(date(2026, 12, 1), "anual")[1] == date(2027, 11, 30)
    assert periods.invoice_number("cliente_norte", date(2026, 12, 1)) == "WM-2026-12-CLIENTE-NORTE"
    assert periods.invoice_number("x", date(2026, 12, 1), "anual") == "WM-2026A-X"
    assert periods.effective_status("pendiente", date(2026, 12, 10), date(2026, 12, 11)) == "vencida"
    assert periods.effective_status("pagada", date(2026, 12, 10), date(2026, 12, 30)) == "pagada"


def test_build_invoice_row():
    sub = {"id": 7, "client_id": "cliente_norte", "plan": "integral",
           "machines": ["A1", "A2", "A3", "A4"], "billing_entity": "siga_co",
           "cycle": "mensual", "due_days": 10}
    row = build_invoice_row(sub, date(2026, 12, 1), date(2026, 12, 1))
    assert row["number"] == "WM-2026-12-CLIENTE-NORTE"
    assert row["period_end"] == "2026-12-31" and row["due_date"] == "2026-12-11"
    assert row["volume_discount_pct"] == 10 and row["machines"] == 4
    assert row["amount"] == round(18_045_692 * 4 * 0.9)
    assert row["status"] == "pendiente" and row["currency"] == "COP"
