"""Registro de actividad de clientes en el portal (tabla `portal_activity`).

Qué hace el cliente — entra a Live, baja un reporte, abre el archivo — para que
SIGA pueda hacer seguimiento antes de que el cliente llame. Solo registra
usuarios con rol `client` (nunca al personal SIGA, tampoco en "View as client").
Todo falla en silencio: el portal nunca se cae por no poder registrar.

DDL (Supabase SQL editor):
    create table if not exists public.portal_activity (
      id bigserial primary key,
      at timestamptz not null default now(),
      email text not null,
      event text not null,
      instance_id text,
      detail text
    );
    create index if not exists portal_activity_at_idx on public.portal_activity (at desc);
    alter table public.portal_activity enable row level security;
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)
TABLE = "portal_activity"

EVENT_LABELS = {
    "view_live": ("Abrió Live Monitoring", "Opened Live Monitoring"),
    "view_reports": ("Abrió el archivo de reportes", "Opened the report archive"),
    "download_live_report": ("Bajó el reporte del día", "Downloaded today's report"),
    "download_archived_report": ("Bajó un reporte del archivo", "Downloaded an archived report"),
}


def _client():
    try:
        from core.live_readings import _get_supabase_client
        return _get_supabase_client()
    except Exception:  # noqa: BLE001
        return None


def _session_user() -> Dict[str, str]:
    try:
        import streamlit as st
        ss = st.session_state
        return {
            "email": str(ss.get("auth_email", "") or "").strip().lower(),
            "role": str(ss.get("auth_role", "") or "").strip().lower(),
            # Vista previa de SIGA o lanzador local (_wm_no_activity_log) → no registra.
            "preview": "1" if (ss.get("live_view_as_client") or ss.get("_wm_no_activity_log")) else "",
        }
    except Exception:  # noqa: BLE001
        return {"email": "", "role": "", "preview": ""}


def log_activity(event: str, instance_id: str = "", detail: str = "",
                 *, once_per_session: bool = False) -> bool:
    """Registra un evento del cliente logueado. Devuelve True si se insertó."""
    u = _session_user()
    if u["role"] != "client" or not u["email"] or u["preview"]:
        return False
    if once_per_session:
        try:
            import streamlit as st
            k = f"_wm_act_{event}_{instance_id}"
            if st.session_state.get(k):
                return False
            st.session_state[k] = True
        except Exception:  # noqa: BLE001
            pass
    c = _client()
    if c is None:
        return False
    try:
        c.table(TABLE).insert({
            "email": u["email"], "event": event,
            "instance_id": instance_id or None, "detail": (detail or "")[:300] or None,
        }).execute()
        return True
    except Exception as e:  # noqa: BLE001
        log.info("portal_activity insert skipped: %s", e)
        return False


def recent_activity(days: int = 7, limit: int = 500) -> Optional[List[Dict[str, Any]]]:
    """Eventos de los últimos `days` días, más recientes primero.
    None si la tabla no existe / no hay conexión (la UI lo distingue de 'vacío')."""
    c = _client()
    if c is None:
        return None
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        r = (c.table(TABLE).select("at,email,event,instance_id,detail")
             .gte("at", since).order("at", desc=True).limit(limit).execute())
        return list(r.data or [])
    except Exception as e:  # noqa: BLE001
        log.info("portal_activity read failed: %s", e)
        return None


def summarize_by_user(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Agrupa por correo: última actividad, visitas a Live y descargas."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows or []:
        em = str(r.get("email", "") or "").lower()
        if not em:
            continue
        d = out.setdefault(em, {"email": em, "last_at": r.get("at"), "last_event": r.get("event"),
                                "visits": 0, "downloads": 0})
        ev = str(r.get("event", ""))
        if ev.startswith("view_"):
            d["visits"] += 1
        if ev.startswith("download_"):
            d["downloads"] += 1
        if str(r.get("at", "")) > str(d["last_at"] or ""):
            d["last_at"], d["last_event"] = r.get("at"), ev
    return sorted(out.values(), key=lambda d: str(d["last_at"] or ""), reverse=True)
