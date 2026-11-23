"""Resumen de Administración: cifras de la cuenta y avisos que piden acción.

Todo best-effort: si una fuente falla, su cifra sale "—" y el resto sigue.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


def _safe(fn, default):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def collect() -> Dict[str, Any]:
    from core.clients import list_clients
    out: Dict[str, Any] = {}
    clients = _safe(list_clients, []) or []
    out["clients"] = len(clients)
    out["machines"] = sum(len(c.asset_tags or ()) for c in clients)

    from core.supabase_auth import list_all_users
    users = _safe(list_all_users, None)
    out["users"] = len(users) if users is not None else None

    def _lic():
        from core.admin.licenses import _lic_state, _load
        lic, act, _ = _load()
        return (sum(1 for x in lic if _lic_state(x)[0] == "ok"),
                sum(1 for a in act if not a.get("revoked")), lic)
    out["lic_active"], out["lic_machines"], out["_licenses"] = _safe(_lic, (None, None, []))

    def _subs():
        from core.billing import pricing, store
        if not store.available():
            return None, []
        subs = [s for s in store.list_subscriptions() if s.get("status") == "activa"]
        nxt = None
        for s in subs:
            try:
                q = pricing.quote(s["plan"], len(s.get("machines") or []), s["billing_entity"],
                                  s["cycle"], price_override=s.get("price_per_machine"),
                                  usd_unit=s.get("usd_price_per_machine"),
                                  trm=None if s.get("usd_price_per_machine") else 1)
                if s["billing_entity"] == "wm_us" and not s.get("usd_price_per_machine"):
                    continue
            except Exception:  # noqa: BLE001
                continue
            if nxt is None or str(s["next_renewal"]) < str(nxt[0]):
                nxt = (s["next_renewal"], q["amount"], q["currency"])
        return nxt, subs
    out["next_invoice"], out["_subs"] = _safe(_subs, (None, []))
    def _docs():
        from core.knowledge_base import list_documents
        return len(list_documents() or [])
    out["docs"] = _safe(_docs, None)
    out["_clients"] = clients
    return out


def short_money(v: float, currency: str) -> str:
    if currency == "COP" and v >= 1_000_000:
        return f"COP {v / 1_000_000:,.1f} M".replace(".", ",")
    from core.billing.pricing import fmt_money
    return fmt_money(v, currency)


def alerts(data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """[(severidad, texto, sección destino)] — solo lo que pide acción."""
    out: List[Tuple[str, str, str]] = []
    try:
        from core.billing import periods, store
        from core.billing.cycle import _today, transfer_details
        ents = {s.get("billing_entity") for s in data.get("_subs") or []}
        names = {"siga_co": "SIGA GROUP SAS", "wm_us": "Watermelon System (Mercury)"}
        for e in sorted(x for x in ents if x):
            if not any(transfer_details(e).values()):
                nxt = data.get("next_invoice")
                when = f" antes de la factura del {periods.fmt_date(nxt[0])}" if nxt else ""
                out.append(("warn", f"Faltan los datos de transferencia de {names.get(e, e)}{when}.",
                            "suscripciones"))
        today = _today()
        for s in data.get("_subs") or []:
            for inv in store.list_invoices(s["client_id"], limit=12):
                if periods.effective_status(inv["status"], periods.date.fromisoformat(inv["due_date"]),
                                            today) == "vencida":
                    out.append(("dang", f"Factura {inv['number']} vencida.", "suscripciones"))
    except Exception:  # noqa: BLE001
        pass
    try:
        from core.billing.upsell import _KIND_TXT, open_requests
        names = {c.id: c.display_name for c in (data.get("_clients") or [])}
        for r in open_requests()[:3]:
            out.insert(0, ("info", f"{names.get(r.get('client_id'), r.get('client_id'))} pidió "
                                   f"{_KIND_TXT.get(r.get('kind'), r.get('kind'))} "
                                   f"({r.get('by')}).", "suscripciones"))
    except Exception:  # noqa: BLE001
        pass
    for lic in data.get("_licenses") or []:
        try:
            exp = datetime.fromisoformat(str(lic.get("expires_at")).replace("Z", "+00:00"))
            days = (exp - datetime.now(timezone.utc)).days
            who = lic.get("customer") or "sin cliente"
            if str(lic.get("status")).lower() == "active" and 0 <= days <= 30:
                out.append(("warn", f"Licencia de {who} vence en {days} días.", "licencias"))
        except Exception:  # noqa: BLE001
            continue
    for c in data.get("_clients") or []:
        if not (c.owner_emails or ()):
            out.append(("info", f"{c.display_name} no tiene personas con acceso.", "clientes"))
    return out
