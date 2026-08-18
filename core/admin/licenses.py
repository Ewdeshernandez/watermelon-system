"""
core.admin.licenses — Sección Administración · Licencias Planta.

Revocación/reactivación de licencias de Watermelon Planta (fuente de verdad:
tabla `revoked_licenses` en Supabase). Extraído de pages/20_License_Admin.py
como render() sin efectos de import. El hub ya autenticó y validó role=admin.
"""
from __future__ import annotations

import re

import streamlit as st

from core.live_readings import _get_supabase_client as get_supabase_client

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@st.cache_data(ttl=15)
def _load_revoked_licenses() -> list:
    """Lee la tabla revoked_licenses de Supabase."""
    sb = get_supabase_client()
    if sb is None:
        return []
    try:
        result = sb.table("revoked_licenses") \
            .select("license_id, revoked_at, revoked_by, reason, "
                    "customer, customer_email") \
            .order("revoked_at", desc=True) \
            .execute()
        return list(result.data or [])
    except Exception as e:  # noqa: BLE001
        st.warning(f"No se pudo leer la tabla revoked_licenses: {e}")
        return []


def render() -> None:
    _user_email = st.session_state.get("auth_email", "")
    if not _user_email.endswith("@sigasas.com"):
        st.error("🔒 Acceso denegado. Esta sección es solo para administradores "
                 "de SIGA GROUP.")
        return

    _revoked = _load_revoked_licenses()

    _k1, _k2 = st.columns(2)
    _k1.metric("🔒 Licencias revocadas", len(_revoked))
    _k2.metric("Endpoint heartbeat", "✓ Activo",
               help="https://yxeqwkhybueelmkrdkgq.supabase.co/functions/v1/license-check")
    st.divider()

    # --- Revocar nueva licencia ---
    st.markdown("### 🔒 Revocar una licencia")
    st.caption(
        "Pegá el `license_id` (UUID) de la licencia que querés revocar. "
        "Lo encontrás en `tools/licenses_issued/<cliente>/license.json` en "
        "el equipo de SIGA donde se emitió, o en el archivo `license.token` "
        "del cliente (segundo campo del JWT decodificado).")

    with st.form("revoke_form", clear_on_submit=True):
        _r1, _r2 = st.columns(2)
        with _r1:
            _new_lid = st.text_input(
                "License ID (UUID)",
                placeholder="ej: 6b4a78cf-0f18-4b4f-b906-e0abf33d18ca",
                key="new_revoke_lid")
            _new_customer = st.text_input(
                "Nombre del cliente", placeholder="ej: Termoeléctrica Norte SAS",
                key="new_revoke_customer")
        with _r2:
            _new_email = st.text_input(
                "Email del cliente", placeholder="ej: ingenieria@termonorte.com",
                key="new_revoke_email")
            _new_reason = st.text_input(
                "Motivo (visible al cliente al ser bloqueado)",
                placeholder="ej: Incumplimiento contractual — pago vencido 60 días",
                key="new_revoke_reason")

        _confirm = st.checkbox(
            "✓ Confirmo que esta acción bloqueará la app del cliente al "
            "próximo arranque con internet", key="new_revoke_confirm")
        _submitted = st.form_submit_button("🔒 REVOCAR LICENCIA", type="primary",
                                           use_container_width=True)

        if _submitted:
            _lid_clean = _new_lid.strip().lower()
            if not _lid_clean:
                st.error("El License ID es obligatorio.")
            elif not UUID_RE.match(_lid_clean):
                st.error("El License ID no es un UUID válido. Debe tener formato:\n"
                         "  `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` (8-4-4-4-12 hex)")
            elif not _new_reason.strip():
                st.error("El motivo es obligatorio.")
            elif not _confirm:
                st.error("Tenés que marcar la confirmación.")
            elif any(r.get("license_id") == _lid_clean for r in _revoked):
                st.warning("Esta licencia ya está revocada. Usá la sección "
                           "de abajo para reactivarla si querés.")
            else:
                try:
                    sb = get_supabase_client()
                    sb.table("revoked_licenses").insert({
                        "license_id": _lid_clean,
                        "revoked_by": _user_email,
                        "reason": _new_reason.strip(),
                        "customer": _new_customer.strip() or None,
                        "customer_email": _new_email.strip() or None,
                    }).execute()
                    st.cache_data.clear()
                    st.success(
                        f"✓ Licencia `{_lid_clean[:8]}...` REVOCADA. "
                        f"La próxima vez que el cliente abra Watermelon Planta "
                        f"con internet, será bloqueado con tu motivo.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error al revocar: {e}")

    st.divider()

    # --- Licencias actualmente revocadas ---
    st.markdown("### 📋 Licencias actualmente revocadas")
    if not _revoked:
        st.info("No hay ninguna licencia revocada en este momento. "
                "Todas las licencias emitidas están activas (mientras no estén vencidas).")
    else:
        st.caption(f"Total: {len(_revoked)} licencia(s) en blacklist")
        for r in _revoked:
            _lid = r.get("license_id", "")
            _customer = r.get("customer") or "—"
            _email = r.get("customer_email") or "—"
            _reason = r.get("reason", "")
            _revoked_at = r.get("revoked_at", "")[:10]
            _revoked_by = r.get("revoked_by", "")
            st.markdown(
                f"""
                <div style="background:#ffffff;border-radius:10px;padding:14px;
                            border:1px solid rgba(239,68,68,0.25);
                            border-left:4px solid #ef4444;margin-bottom:12px;
                            box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                        <span style="background:rgba(239,68,68,0.10);color:#ef4444;
                                     padding:2px 8px;border-radius:5px;font-size:10px;
                                     font-weight:800;letter-spacing:1px;
                                     text-transform:uppercase;">🔒 Revocada</span>
                        <span style="font-size:15px;font-weight:700;color:#0f172a;">{_customer}</span>
                    </div>
                    <div style="font-size:12px;color:#475569;line-height:1.6;">
                        📧 {_email}<br>
                        🔑 <code style="font-size:10px;">{_lid}</code><br>
                        📅 Revocada el {_revoked_at} por {_revoked_by}<br>
                        📝 <i>{_reason}</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            _b1, _b2, _b3 = st.columns([1, 1, 4])
            with _b1:
                if st.button("♻ Reactivar", key=f"reactivate_{_lid}",
                             use_container_width=True):
                    try:
                        sb = get_supabase_client()
                        sb.table("revoked_licenses").delete().eq("license_id", _lid).execute()
                        st.cache_data.clear()
                        st.success(f"✓ Licencia `{_lid[:8]}...` reactivada. "
                                   f"Próximo arranque online del cliente → desbloqueado.")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Error al reactivar: {e}")

    st.divider()
    with st.expander("ℹ Cómo funciona el sistema de revocación"):
        st.markdown(
            """
            **Flujo técnico:**

            1. Cuando hacés click en "Revocar", el `license_id` se inserta en la
               tabla `revoked_licenses` de Supabase con el motivo y tu email.
            2. La Edge Function `license-check` lee de esa tabla cada vez que
               Watermelon Planta del cliente la consulta.
            3. Planta chequea ese endpoint **al arrancar la app**, con timeout
               de 5 segundos y máximo 1 vez cada 24h (cached localmente).
            4. Si el endpoint responde `revoked` → la app del cliente se bloquea
               inmediatamente con tu motivo y el email de contacto SIGA.
            5. Si el cliente está offline → sigue funcionando hasta que se
               conecte. Si pasan **> 30 días sin poder validar** → bloqueo
               automático por seguridad.

            **Reactivación:** click en ♻ borra el `license_id` de la blacklist.

            **Cómo obtener el `license_id`:** output de `tools/license_issue.py`
            o el campo `jti` del `license.token` decodificado en jwt.io.
            """)


__all__ = ["render"]
