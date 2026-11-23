"""Acceso a las tablas de facturación en Supabase (service role).

Ver supabase/sql/billing.sql. Todo devuelve dicts planos; los errores se
registran y devuelven vacío / False (la UI muestra el aviso).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

SUBS = "wm_subscriptions"
INV = "wm_invoices"
SETTINGS = "wm_billing_settings"


def _client():
    from core.live_readings import _get_supabase_client
    return _get_supabase_client()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emails(v: Any) -> List[str]:
    if isinstance(v, str):
        v = v.replace(";", ",").replace("\n", ",").split(",")
    out = []
    for e in v or []:
        e = str(e).strip().lower()
        if "@" in e and e not in out:
            out.append(e)
    return out


def available() -> bool:
    """True si las tablas existen y responden."""
    c = _client()
    if c is None:
        return False
    try:
        c.table(SUBS).select("id").limit(1).execute()
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("billing no disponible (¿falta billing.sql?): %s", e)
        return False


# ---------------------------------------------------------------- suscripciones

def get_subscription(client_id: str) -> Optional[Dict[str, Any]]:
    c = _client()
    if c is None or not client_id:
        return None
    try:
        r = c.table(SUBS).select("*").eq("client_id", client_id).limit(1).execute()
        rows = getattr(r, "data", None) or []
        return rows[0] if rows else None
    except Exception as e:  # noqa: BLE001
        log.warning("get_subscription(%s): %s", client_id, e)
        return None


def list_subscriptions() -> List[Dict[str, Any]]:
    c = _client()
    if c is None:
        return []
    try:
        r = c.table(SUBS).select("*").order("client_id").execute()
        return list(getattr(r, "data", None) or [])
    except Exception as e:  # noqa: BLE001
        log.warning("list_subscriptions: %s", e)
        return []


def save_subscription(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Crea o actualiza por client_id. Normaliza los correos a minúscula."""
    c = _client()
    if c is None or not data.get("client_id"):
        return None
    row = dict(data)
    row.pop("id", None)
    row.pop("created_at", None)
    for k in ("authorized_emails", "billing_emails"):
        if k in row:
            row[k] = _emails(row[k])
    for k in ("start_date", "next_renewal", "usd_locked_at"):
        if hasattr(row.get(k), "isoformat"):
            row[k] = row[k].isoformat()
    row["updated_at"] = _now()
    try:
        r = c.table(SUBS).upsert(row, on_conflict="client_id").execute()
        rows = getattr(r, "data", None) or []
        return rows[0] if rows else get_subscription(row["client_id"])
    except Exception as e:  # noqa: BLE001
        log.error("save_subscription(%s): %s", row.get("client_id"), e)
        return None


def subscription_for_viewer(email: str) -> Optional[Dict[str, Any]]:
    """Suscripción que `email` está AUTORIZADO a ver (pestaña Suscripción)."""
    email = (email or "").strip().lower()
    c = _client()
    if c is None or "@" not in email:
        return None
    try:
        r = (c.table(SUBS).select("*").contains("authorized_emails", [email])
             .limit(1).execute())
        rows = getattr(r, "data", None) or []
        return rows[0] if rows else None
    except Exception as e:  # noqa: BLE001
        log.warning("subscription_for_viewer: %s", e)
        return None


# ---------------------------------------------------------------- facturas

def list_invoices(client_id: Optional[str] = None, limit: int = 60) -> List[Dict[str, Any]]:
    c = _client()
    if c is None:
        return []
    try:
        q = c.table(INV).select("*")
        if client_id:
            q = q.eq("client_id", client_id)
        r = q.order("period_start", desc=True).limit(limit).execute()
        return list(getattr(r, "data", None) or [])
    except Exception as e:  # noqa: BLE001
        log.warning("list_invoices: %s", e)
        return []


def get_invoice(invoice_id: int) -> Optional[Dict[str, Any]]:
    c = _client()
    if c is None:
        return None
    try:
        r = c.table(INV).select("*").eq("id", invoice_id).limit(1).execute()
        rows = getattr(r, "data", None) or []
        return rows[0] if rows else None
    except Exception as e:  # noqa: BLE001
        log.warning("get_invoice(%s): %s", invoice_id, e)
        return None


def find_invoice(subscription_id: int, period_start: str) -> Optional[Dict[str, Any]]:
    c = _client()
    if c is None:
        return None
    try:
        r = (c.table(INV).select("*").eq("subscription_id", subscription_id)
             .eq("period_start", period_start).limit(1).execute())
        rows = getattr(r, "data", None) or []
        return rows[0] if rows else None
    except Exception as e:  # noqa: BLE001
        log.warning("find_invoice: %s", e)
        return None


def insert_invoice(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    c = _client()
    if c is None:
        return None
    try:
        r = c.table(INV).insert(row).execute()
        rows = getattr(r, "data", None) or []
        return rows[0] if rows else None
    except Exception as e:  # noqa: BLE001
        log.error("insert_invoice(%s): %s", row.get("number"), e)
        return None


def update_invoice(invoice_id: int, **fields: Any) -> bool:
    c = _client()
    if c is None:
        return False
    try:
        c.table(INV).update(fields).eq("id", invoice_id).execute()
        return True
    except Exception as e:  # noqa: BLE001
        log.error("update_invoice(%s): %s", invoice_id, e)
        return False


# ---------------------------------------------------------------- ajustes

def get_setting(key: str, default: Any = None) -> Any:
    c = _client()
    if c is None:
        return default
    try:
        r = c.table(SETTINGS).select("value").eq("key", key).limit(1).execute()
        rows = getattr(r, "data", None) or []
        return rows[0]["value"] if rows else default
    except Exception as e:  # noqa: BLE001
        log.warning("get_setting(%s): %s", key, e)
        return default


def set_setting(key: str, value: Any) -> bool:
    c = _client()
    if c is None:
        return False
    try:
        c.table(SETTINGS).upsert({"key": key, "value": value, "updated_at": _now()},
                                 on_conflict="key").execute()
        return True
    except Exception as e:  # noqa: BLE001
        log.error("set_setting(%s): %s", key, e)
        return False
