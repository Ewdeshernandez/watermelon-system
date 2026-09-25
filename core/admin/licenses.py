"""
core.admin.licenses — Administración · Licencias (campo).

Consola de licencias de las apps de campo (Watermelon Modal / Torsional /
Balanceo / Field). Fuente de verdad: tablas Supabase `licenses` + `activations`
(server/licensing/schema.sql + 2026_09_licenses_console.sql). El emisor firma
tokens Ed25519 en la Edge Function `activate`; aquí se ADMINISTRA el ciclo de
vida comercial:

    · Crear licencia (genera clave WM-XXXX-XXXX-XXXX).
    · Ver dónde vive cada licencia: PC, IP, ubicación, última conexión, VM.
    · Renovar (extiende vigencia).
    · Suspender por falta de pago  → el cliente ve "Licencia no renovada por
      falta de pago" al próximo arranque online (edge `activate` → payment_due).
    · Reactivar / Revocar licencia completa.
    · Revocar / reactivar / liberar UNA máquina (cupo).

Escribe con el cliente de service-role (core.live_readings), que salta RLS.
El hub (pages/20_Administracion.py) ya autenticó y validó role=admin.

NOTA: el antiguo módulo "Licencias Planta" (tabla `revoked_licenses` + edge
`license-check`) administraba el producto legacy `planta/` (Watermelon Planta
Edition, JWT RS256). Ese producto quedó fuera de uso; esta consola lo reemplaza
y apunta al sistema de campo real.
"""
from __future__ import annotations

import html as _html
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import streamlit as st

from core.live_readings import _get_supabase_client as get_supabase_client
from core.ui_industrial import dot, html_table

# Paquetes comerciales → features embebidas en el token (informativo; el modelo
# es de PAQUETE: una activación cubre todos los módulos de campo del PC).
PLANS: Dict[str, List[str]] = {
    "Paquete Campo (todo)": ["oma", "ema", "report", "torsional", "balance"],
    "Modal (OMA/EMA)": ["oma", "ema", "report"],
    "Torsional": ["torsional", "report"],
    "Balanceo": ["balance", "report"],
}


# =====================================================================
# Helpers
# =====================================================================
def _gen_key() -> str:
    """Clave única formato WM-XXXX-XXXX-XXXX (hex mayúsculas)."""
    return "WM-" + "-".join(secrets.token_hex(2).upper() for _ in range(3))


def _parse_dt(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        txt = str(s).replace("Z", "+00:00")
        dt = datetime.fromisoformat(txt)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def _rel(s: Any) -> str:
    """'hace 3 min' / 'hace 2 d' a partir de un timestamp ISO."""
    dt = _parse_dt(s)
    if not dt:
        return "—"
    delta = datetime.now(timezone.utc) - dt
    sec = int(delta.total_seconds())
    if sec < 0:
        return "ahora"
    if sec < 90:
        return "hace segundos"
    if sec < 3600:
        return f"hace {sec // 60} min"
    if sec < 86400:
        return f"hace {sec // 3600} h"
    if sec < 30 * 86400:
        return f"hace {sec // 86400} d"
    return dt.strftime("%Y-%m-%d")


def _days_left(expires_at: Any) -> int | None:
    dt = _parse_dt(expires_at)
    if not dt:
        return None
    return (dt - datetime.now(timezone.utc)).days


def _lic_state(lic: Dict[str, Any]) -> tuple[str, str, str]:
    """(severidad_dot, etiqueta, color_hex) del estado de una licencia."""
    status = str(lic.get("status", "")).lower()
    dl = _days_left(lic.get("expires_at"))
    if status == "suspended":
        return "dang", "Suspendida · falta de pago", "#dc3545"
    if status == "revoked":
        return "off", "Revocada", "#8090a6"
    if dl is not None and dl < 0:
        return "warn", "Vencida", "#e8890c"
    if dl is not None and dl < 30:
        return "warn", f"Activa · vence en {dl} d", "#e8890c"
    return "ok", "Activa", "#1f9d55"


@st.cache_data(ttl=15)
def _load() -> tuple[list, list, list]:
    """(licenses, activations, events) desde Supabase."""
    sb = get_supabase_client()
    if sb is None:
        return [], [], []
    try:
        lic = sb.table("licenses").select("*").order("created_at", desc=True).execute()
        act = sb.table("activations").select("*").execute()
        evt = []
        try:
            evt = list((sb.table("license_events").select("*")
                        .order("created_at", desc=True).limit(1000).execute()).data or [])
        except Exception:  # noqa: BLE001 — tabla puede no existir aún
            evt = []
        return list(lic.data or []), list(act.data or []), evt
    except Exception as e:  # noqa: BLE001
        st.warning(f"No se pudo leer licencias/activaciones: {e}")
        return [], [], []


def _fmt_dt(s: Any) -> str:
    dt = _parse_dt(s)
    return dt.strftime("%Y-%m-%d %H:%M") if dt else "—"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =====================================================================
# Render
# =====================================================================
def render() -> None:
    _user_email = st.session_state.get("auth_email", "")
    if not _user_email.endswith("@sigasas.com"):
        st.error("Acceso denegado. Sección solo para administradores de SIGA GROUP.")
        return

    sb = get_supabase_client()
    if sb is None:
        st.error("Supabase no configurado (falta SUPABASE_URL / SUPABASE_SERVICE_KEY).")
        return

    licenses, activations, events = _load()

    # --- Índice de activaciones por licencia ---
    by_lic: Dict[str, List[Dict[str, Any]]] = {}
    for a in activations:
        by_lic.setdefault(a.get("license_id"), []).append(a)

    # --- Índice de eventos (historial de conexiones) por licencia ---
    ev_lic: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        ev_lic.setdefault(e.get("license_id"), []).append(e)

    # --- KPIs ---
    n_total = len(licenses)
    n_active = sum(1 for l in licenses if _lic_state(l)[0] == "ok")
    n_suspended = sum(1 for l in licenses if str(l.get("status")).lower() == "suspended")
    n_expsoon = sum(1 for l in licenses if _lic_state(l)[1].startswith("Activa · vence"))
    n_machines = sum(1 for a in activations if not a.get("revoked"))
    _kpi_row = "".join(
        f'<div class="wi-kpi"><div class="n">{dot(sev)} {val}</div>'
        f'<div class="l">{lbl}</div></div>'
        for sev, val, lbl in [
            ("info", n_total, "Licencias"),
            ("ok", n_active, "Activas"),
            ("dang", n_suspended, "Suspendidas"),
            ("warn", n_expsoon, "Vencen pronto"),
            ("ok", n_machines, "Máquinas activas"),
        ])
    st.markdown(f'<div class="wi-kpis">{_kpi_row}</div>', unsafe_allow_html=True)
    st.markdown('<div class="wi-label">Licencias de campo · Modal · Torsional · '
                'Balanceo · Field</div>', unsafe_allow_html=True)

    # --- Crear licencia ---
    with st.expander("＋  Crear licencia nueva", expanded=(n_total == 0)):
        with st.form("create_license", clear_on_submit=True):
            _c1, _c2 = st.columns(2)
            with _c1:
                _customer = st.text_input("Cliente / empresa",
                                          placeholder="ej: Termoeléctrica Norte SAS")
                _account = st.text_input("Cuenta (email del cliente)",
                                         placeholder="ej: ingenieria@termonorte.com")
                _plan = st.selectbox("Paquete", list(PLANS.keys()))
            with _c2:
                _seats = st.number_input("Máquinas (seats)", min_value=1, max_value=50,
                                         value=1, step=1)
                _months = st.number_input("Vigencia (meses)", min_value=1, max_value=120,
                                          value=12, step=1)
                _notes = st.text_input("Notas internas (opcional)",
                                       placeholder="ej: OC-2026-118, contacto Juan")
            _submit = st.form_submit_button("CREAR LICENCIA", type="primary",
                                            use_container_width=True)
            if _submit:
                if not _account.strip():
                    st.error("La cuenta (email del cliente) es obligatoria.")
                else:
                    key = _gen_key()
                    exp = (datetime.now(timezone.utc)
                           + timedelta(days=int(_months) * 30)).replace(
                        hour=23, minute=59, second=59, microsecond=0).isoformat()
                    try:
                        sb.table("licenses").insert({
                            "key": key,
                            "account": _account.strip(),
                            "customer": _customer.strip() or None,
                            "seats": int(_seats),
                            "features": PLANS[_plan],
                            "plan": _plan,
                            "notes": _notes.strip() or None,
                            "expires_at": exp,
                            "status": "active",
                            "updated_at": _now_iso(),
                        }).execute()
                        st.cache_data.clear()
                        st.success(f"Licencia creada. Clave del cliente: **{key}**  "
                                   f"— entrégala para activar su equipo.")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Error al crear: {e}")

    st.divider()

    if not licenses:
        st.info("No hay licencias todavía. Crea la primera arriba.")
        return

    # --- Actividad reciente (todas las licencias) ---
    if events:
        with st.expander(f"Actividad reciente · últimas conexiones ({len(events)})"):
            _cust = {l.get("id"): (l.get("customer") or l.get("account") or "—")
                     for l in licenses}
            rows = [[
                _fmt_dt(e.get("created_at")), _cust.get(e.get("license_id"), "—"),
                e.get("app") or "—", e.get("hostname") or "—",
                e.get("ip") or "—", e.get("ip_geo") or "—",
            ] for e in events[:60]]
            html_table(["Fecha/hora", "Cliente", "Módulo", "PC", "IP", "Ubicación"], rows)

    # --- Lista de licencias ---
    for lic in licenses:
        _render_license_card(sb, lic, by_lic.get(lic.get("id"), []),
                             ev_lic.get(lic.get("id"), []))


def _render_license_card(sb, lic: Dict[str, Any], acts: List[Dict[str, Any]],
                         evts: List[Dict[str, Any]] | None = None) -> None:
    lid = lic.get("id")
    key = lic.get("key") or "—"
    customer = lic.get("customer") or lic.get("account") or "—"
    account = lic.get("account") or "—"
    plan = lic.get("plan") or ", ".join(lic.get("features") or []) or "—"
    seats = int(lic.get("seats") or 1)
    used = sum(1 for a in acts if not a.get("revoked"))
    dl = _days_left(lic.get("expires_at"))
    exp_dt = _parse_dt(lic.get("expires_at"))
    exp_txt = exp_dt.strftime("%Y-%m-%d") if exp_dt else "—"
    sev, label, color = _lic_state(lic)

    _dl_txt = ("vencida" if (dl is not None and dl < 0)
               else (f"{dl} días" if dl is not None else "—"))

    st.markdown(
        f"""
        <div style="background:#fff;border:1px solid #e2e8f2;border-left:4px solid {color};
                    border-radius:14px;padding:16px 18px 6px;margin-bottom:6px;
                    box-shadow:0 1px 2px rgba(11,31,58,.05),0 8px 22px rgba(11,31,58,.06);
                    font-family:'IBM Plex Sans',sans-serif;">
          <div style="display:flex;align-items:center;gap:11px;flex-wrap:wrap;">
            <span style="font:800 16px 'IBM Plex Sans';color:#0b1f3a;">{_html.escape(customer)}</span>
            <span style="background:{color}1a;color:{color};border:1px solid {color}55;
                         padding:3px 10px;border-radius:999px;font:800 10px 'IBM Plex Sans';
                         letter-spacing:.06em;text-transform:uppercase;">{dot(sev)} {label}</span>
            <span style="flex:1;"></span>
            <code style="font:700 14px 'IBM Plex Mono';color:#274b7d;letter-spacing:1px;">{_html.escape(key)}</code>
          </div>
          <div style="font-size:12px;color:#3a4c66;line-height:1.9;margin-top:6px;">
            {_html.escape(account)} &nbsp;·&nbsp; {_html.escape(str(plan))} &nbsp;·&nbsp;
            <b>{used}/{seats}</b> máquinas &nbsp;·&nbsp; vence {exp_txt} ({_dl_txt})
          </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Máquinas (dónde vive la licencia) ---
    if acts:
        rows = []
        for a in sorted(acts, key=lambda x: str(x.get("last_seen") or ""), reverse=True):
            estado = ("● revocada" if a.get("revoked") else "● activa")
            rows.append([
                a.get("hostname") or "—",
                (a.get("machine_fp") or "")[:12] + "…" if a.get("machine_fp") else "—",
                a.get("ip") or "—",
                a.get("ip_geo") or "—",
                a.get("app") or "—",
                "sí" if a.get("is_vm") else "no",
                _rel(a.get("last_seen")),
                estado,
            ])
        html_table(
            ["PC", "Máquina", "IP", "Ubicación", "Módulo", "VM", "Última conexión", "Estado"],
            rows)
    else:
        st.caption("Sin máquinas activadas todavía con esta clave.")

    # --- Historial de conexiones (Nivel B) ---
    if evts:
        with st.expander(f"Historial de conexiones ({len(evts)})"):
            hrows = [[
                _fmt_dt(e.get("created_at")), e.get("app") or "—",
                e.get("hostname") or "—", e.get("ip") or "—", e.get("ip_geo") or "—",
            ] for e in evts[:40]]
            html_table(["Fecha/hora", "Módulo", "PC", "IP", "Ubicación"], hrows)

    # --- Acciones ---
    _a1, _a2, _a3, _a4, _a5 = st.columns([1.1, 1.4, 1.1, 1.1, 1.4])

    # Renovar
    with _a1:
        with st.popover("Renovar", use_container_width=True):
            _m = st.number_input("Extender (meses)", min_value=1, max_value=120, value=12,
                                 step=1, key=f"renew_m_{lid}")
            if st.button("Aplicar renovación", key=f"renew_btn_{lid}",
                         type="primary", use_container_width=True):
                base = _parse_dt(lic.get("expires_at")) or datetime.now(timezone.utc)
                base = max(base, datetime.now(timezone.utc))  # no renovar hacia el pasado
                new_exp = (base + timedelta(days=int(_m) * 30)).isoformat()
                try:
                    sb.table("licenses").update({
                        "expires_at": new_exp, "status": "active",
                        "suspended_reason": None, "updated_at": _now_iso(),
                    }).eq("id", lid).execute()
                    st.cache_data.clear()
                    st.success("Renovada y reactivada.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")

    # Suspender por falta de pago
    with _a2:
        with st.popover("Suspender (no pago)", use_container_width=True):
            _reason = st.text_input(
                "Motivo (lo ve el cliente)",
                value="Licencia no renovada por falta de pago",
                key=f"susp_r_{lid}")
            st.caption("Al próximo arranque online, el cliente será bloqueado con este motivo.")
            if st.button("SUSPENDER", key=f"susp_btn_{lid}", type="primary",
                         use_container_width=True):
                try:
                    sb.table("licenses").update({
                        "status": "suspended",
                        "suspended_reason": _reason.strip() or "Licencia no renovada por falta de pago",
                        "updated_at": _now_iso(),
                    }).eq("id", lid).execute()
                    st.cache_data.clear()
                    st.success("Licencia suspendida.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")

    # Reactivar (solo si suspendida/revocada)
    with _a3:
        _blocked = str(lic.get("status")).lower() in ("suspended", "revoked")
        if st.button("Reactivar", key=f"react_{lid}", use_container_width=True,
                     disabled=not _blocked):
            try:
                sb.table("licenses").update({
                    "status": "active", "suspended_reason": None, "updated_at": _now_iso(),
                }).eq("id", lid).execute()
                st.cache_data.clear()
                st.success("Reactivada.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Error: {e}")

    # Revocar licencia completa
    with _a4:
        with st.popover("Revocar", use_container_width=True):
            st.caption("Bloqueo definitivo de TODA la licencia (todas las máquinas).")
            _ok = st.checkbox("Confirmo revocar esta licencia", key=f"revk_ok_{lid}")
            if st.button("REVOCAR", key=f"revk_btn_{lid}", type="primary",
                         disabled=not _ok, use_container_width=True):
                try:
                    sb.table("licenses").update({
                        "status": "revoked", "updated_at": _now_iso(),
                    }).eq("id", lid).execute()
                    st.cache_data.clear()
                    st.success("Licencia revocada.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")

    # Gestionar equipos (por máquina)
    with _a5:
        if acts:
            with st.popover("Gestionar equipos", use_container_width=True):
                _opts = {
                    f"{(a.get('hostname') or '—')} · {(a.get('machine_fp') or '')[:10]}"
                    f"{' · revocada' if a.get('revoked') else ''}": a
                    for a in acts}
                _pick = st.selectbox("Equipo", list(_opts.keys()), key=f"mach_pick_{lid}")
                a = _opts.get(_pick, {})
                _aid = a.get("id")
                _b1, _b2, _b3 = st.columns(3)
                with _b1:
                    if st.button("Revocar", key=f"mrev_{lid}_{_aid}",
                                 use_container_width=True, disabled=bool(a.get("revoked"))):
                        _mach_update(sb, _aid, {"revoked": True})
                with _b2:
                    if st.button("Reactivar", key=f"mact_{lid}_{_aid}",
                                 use_container_width=True, disabled=not a.get("revoked")):
                        _mach_update(sb, _aid, {"revoked": False})
                with _b3:
                    if st.button("Liberar cupo", key=f"mdel_{lid}_{_aid}",
                                 use_container_width=True):
                        try:
                            sb.table("activations").delete().eq("id", _aid).execute()
                            st.cache_data.clear()
                            st.success("Cupo liberado.")
                            st.rerun()
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Error: {e}")

    st.divider()


def _mach_update(sb, aid, patch: Dict[str, Any]) -> None:
    try:
        patch = {**patch, "updated_at": _now_iso()}
        sb.table("activations").update(patch).eq("id", aid).execute()
        st.cache_data.clear()
        st.success("Máquina actualizada.")
        st.rerun()
    except Exception as e:  # noqa: BLE001
        st.error(f"Error: {e}")


__all__ = ["render"]
