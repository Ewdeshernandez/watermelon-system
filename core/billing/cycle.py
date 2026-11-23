"""Facturación: crear facturas por periodo, marcar pagadas, anular y el ciclo
automático del cron (run_due).

Reglas:
  · Una factura por (suscripción, inicio de periodo) — idempotente.
  · USD (Watermelon System, EE. UU.): el precio por máquina se CONGELA 12
    meses con la TRM del día en que se fija; pasado el año se vuelve a fijar.
  · El cron corre cada hora: cuando next_renewal <= hoy, factura el periodo y
    avanza next_renewal. Avisa por correo solo si la suscripción lo tiene
    activado (notify_email), a billing_emails (o a los autorizados).
  · Nada se corta solo: vencida = aviso, la decisión es de SIGA.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from core.billing import periods, pricing, store

log = logging.getLogger(__name__)

TRM_URL = ("https://www.datos.gov.co/resource/32sa-8pi3.json"
           "?$order=vigenciadesde%20DESC&$limit=1")


def _today() -> date:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Bogota")).date()
    except Exception:  # noqa: BLE001
        return date.today()


def _d(v: Any) -> Optional[date]:
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v)[:10])
    except Exception:  # noqa: BLE001
        return None


def get_trm() -> Optional[float]:
    """TRM: valor manual de Administración si existe; si no, la oficial
    (Superfinanciera, datos.gov.co)."""
    manual = store.get_setting("trm_manual")
    try:
        if manual and float(manual) > 0:
            return float(manual)
    except (TypeError, ValueError):
        pass
    try:
        import requests
        r = requests.get(TRM_URL, timeout=10)
        r.raise_for_status()
        rows = r.json() or []
        return float(rows[0]["valor"]) if rows else None
    except Exception as e:  # noqa: BLE001
        log.warning("TRM no disponible: %s", e)
        return None


def _ensure_usd_lock(sub: Dict[str, Any], today: date) -> Optional[Dict[str, Any]]:
    """Fija (o renueva tras 12 meses) el precio USD por máquina."""
    locked = _d(sub.get("usd_locked_at"))
    if sub.get("usd_price_per_machine") and locked and periods.add_months(locked, 12) > today:
        return sub
    trm = get_trm()
    if not trm:
        return None
    unit_cop = pricing.unit_price_cop(sub["plan"], sub.get("price_per_machine"))
    sub = dict(sub, usd_price_per_machine=pricing.usd_unit_from_cop(unit_cop, trm),
               usd_locked_at=today.isoformat())
    store.save_subscription(sub)
    return sub


def build_invoice_row(sub: Dict[str, Any], period_start: date, issued: date,
                      trm: Optional[float] = None, by: str = "cron") -> Dict[str, Any]:
    """Fila de factura (puro, salvo lo que ya trae `sub`)."""
    cycle = sub.get("cycle") or "mensual"
    p0, p1 = periods.period_bounds(period_start, cycle)
    q = pricing.quote(sub["plan"], len(sub.get("machines") or []),
                      sub.get("billing_entity") or "siga_co", cycle,
                      price_override=sub.get("price_per_machine"),
                      usd_unit=sub.get("usd_price_per_machine"), trm=trm)
    return {
        "number": periods.invoice_number(sub["client_id"], p0, cycle),
        "client_id": sub["client_id"], "subscription_id": sub["id"],
        "period_start": p0.isoformat(), "period_end": p1.isoformat(),
        "due_date": periods.due_date(issued, int(sub.get("due_days") or 10)).isoformat(),
        "billing_entity": sub.get("billing_entity") or "siga_co",
        "currency": q["currency"], "plan": q["plan"], "machines": q["machines"],
        "unit_price": q["unit_price"], "volume_discount_pct": q["volume_discount_pct"],
        "entity_discount_pct": q["entity_discount_pct"], "months": q["months"],
        "months_charged": q["months_charged"], "subtotal": q["subtotal"],
        "amount": q["amount"], "trm": q["trm"], "status": "pendiente",
        "provider": "manual", "created_by": by,
    }


def create_invoice(sub: Dict[str, Any], period_start: Optional[date] = None,
                   by: str = "cron", notify: Optional[bool] = None) -> Dict[str, Any]:
    """Crea la factura del periodo (o devuelve la existente)."""
    today = _today()
    period_start = period_start or _d(sub.get("next_renewal")) or today
    if not sub.get("machines"):
        return {"ok": False, "error": "La suscripción no tiene máquinas."}
    existing = store.find_invoice(sub["id"], period_start.isoformat())
    if existing:
        return {"ok": True, "invoice": existing, "created": False}
    trm = None
    if (sub.get("billing_entity") or "siga_co") == "wm_us":
        sub = _ensure_usd_lock(sub, today) or {}
        if not sub.get("usd_price_per_machine"):
            return {"ok": False, "error": "Sin TRM para fijar el precio en USD."}
        trm = get_trm()
    row = build_invoice_row(sub, period_start, today, trm=trm, by=by)
    inv = store.insert_invoice(row)
    if not inv:
        return {"ok": False, "error": "No se pudo guardar la factura."}
    out: Dict[str, Any] = {"ok": True, "invoice": inv, "created": True}
    if notify if notify is not None else bool(sub.get("notify_email")):
        out["email"] = send_invoice_email(sub, inv)
    return out


def run_due(today: Optional[date] = None) -> Dict[str, Any]:
    """Cron: factura las suscripciones activas cuyo periodo ya empezó."""
    today = today or _today()
    created: List[str] = []
    errors: List[str] = []
    for sub in store.list_subscriptions():
        if sub.get("status") != "activa":
            continue
        nr = _d(sub.get("next_renewal"))
        guard = 0
        while nr and nr <= today and guard < 24:
            guard += 1
            res = create_invoice(sub, nr, by="cron")
            if not res.get("ok"):
                errors.append(f"{sub['client_id']}: {res.get('error')}")
                break
            if res.get("created"):
                created.append(res["invoice"]["number"])
            nr = periods.next_renewal_after(nr, sub.get("cycle") or "mensual")
            # releer: create_invoice pudo fijar el precio USD en la base
            sub = store.get_subscription(sub["client_id"]) or sub
            sub = store.save_subscription(dict(sub, next_renewal=nr.isoformat())) or sub
    return {"created": created, "errors": errors}


def mark_paid(invoice_id: int, method: str, ref: str = "", by: str = "") -> bool:
    return store.update_invoice(
        invoice_id, status="pagada", paid_at=datetime.now(timezone.utc).isoformat(),
        paid_method=(method or "").strip(), paid_ref=(ref or "").strip(),
        notes=f"Marcada pagada por {by}".strip() if by else None)


def void_invoice(invoice_id: int, reason: str = "", by: str = "") -> bool:
    return store.update_invoice(
        invoice_id, status="anulada",
        notes=(f"Anulada por {by}: {reason}" if by else f"Anulada: {reason}").strip())


# ---------------------------------------------------------------- correo

def transfer_details(entity: str) -> Dict[str, str]:
    """Datos de pago por transferencia configurados en Administración."""
    v = store.get_setting(f"transfer_{entity}", {}) or {}
    return {k: str(v.get(k, "") or "") for k in
            ("beneficiary", "bank", "account_type", "account", "routing",
             "swift", "tax_id", "notes")}


def send_invoice_email(sub: Dict[str, Any], inv: Dict[str, Any]) -> Dict[str, Any]:
    to = sub.get("billing_emails") or sub.get("authorized_emails") or []
    if not to:
        return {"ok": False, "error": "Sin correos de cobro."}
    from core.email_sender import send_email
    p0, p1 = _d(inv["period_start"]), _d(inv["period_end"])
    per = periods.period_label(p0, p1, sub.get("cycle") or "mensual")
    amount = pricing.fmt_money(float(inv["amount"]), inv["currency"])
    ent = pricing.ENTITIES.get(inv.get("billing_entity") or "siga_co", {})
    subject = f"Factura {inv['number']} · {per} · {amount}"
    text = (f"Hola,\n\nYa está disponible la factura {inv['number']} de su suscripción "
            f"Watermelon System ({pricing.PLANS[inv['plan']].name}, {inv['machines']} "
            f"máquina(s)), periodo {per}.\n\nValor: {amount}\nVence: {inv['due_date']}\n"
            f"Factura: {ent.get('name', '')}\n\nPuede pagarla desde la pestaña Suscripción "
            "en https://watermelonsystem.app\n\nGracias,\nWatermelon System")
    html = f"""<div style="font-family:Arial,sans-serif;max-width:560px;margin:auto;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden">
<div style="background:#2f6b3a;color:#fff;padding:16px 20px;font-size:18px">&#127817; Watermelon System</div>
<div style="height:6px;background:linear-gradient(90deg,#e5484d,#f2a3a5)"></div>
<div style="padding:20px;color:#0b1f3a;font-size:14px;line-height:1.6">
<p>Ya está disponible la factura <b>{inv['number']}</b> de su suscripción
<b>{pricing.PLANS[inv['plan']].name}</b> ({inv['machines']} máquina(s)), periodo <b>{per}</b>.</p>
<table style="width:100%;border-collapse:collapse;font-size:14px">
<tr><td style="padding:6px 0;color:#64748b">Valor</td><td style="text-align:right;font-weight:bold">{amount}</td></tr>
<tr><td style="padding:6px 0;color:#64748b">Vence</td><td style="text-align:right">{inv['due_date']}</td></tr>
<tr><td style="padding:6px 0;color:#64748b">Factura</td><td style="text-align:right">{ent.get('name', '')}</td></tr>
</table>
<p style="margin-top:18px"><a href="https://watermelonsystem.app" style="background:#12305e;color:#fff;padding:10px 16px;border-radius:6px;text-decoration:none">Ver y pagar en Suscripción</a></p>
</div></div>"""
    sent, failed = [], []
    for e in to:
        try:
            r = send_email(to=e, subject=subject, body_text=text, body_html=html)
            (sent if (r or {}).get("ok") else failed).append(e)
        except Exception as ex:  # noqa: BLE001
            log.warning("factura email a %s falló: %s", e, ex)
            failed.append(e)
    return {"ok": bool(sent), "sent": sent, "failed": failed}
