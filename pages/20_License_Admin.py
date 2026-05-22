"""
pages/20_License_Admin.py — Admin de licencias Watermelon Planta
=================================================================

Página INTERNA de SIGA. Permite ver y revocar licencias activas de
Watermelon Planta Edition. La revocación se hace insertando el
license_id en la tabla `revoked_licenses` de Supabase, y la próxima vez
que Planta arranque con internet, recibe la señal de revocación y se
bloquea.

Solo accesible para usuarios con `is_admin=True` y email @sigasas.com.

Flujo típico:
  1. SIGA detecta cliente moroso o malicioso
  2. Abre esta página
  3. Busca al cliente en la tabla "Licencias emitidas"
  4. Click "Revocar" → escribe motivo → confirma
  5. La próxima vez que el cliente abra Planta con internet → bloqueado
  6. Si arrepiente, click "Reactivar" → borra de la blacklist
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

from core.auth import require_role
from core.db import get_supabase_client

st.set_page_config(
    page_title="Watermelon · Admin Licencias",
    page_icon="🔐",
    layout="wide",
)

# ============================================================
# 1. Auth — solo admin SIGA
# ============================================================
require_role(allowed_roles=("admin",))

_user_email = st.session_state.get("auth_email", "")
if not _user_email.endswith("@sigasas.com"):
    st.error(
        "🔒 Acceso denegado. Esta página es solo para administradores "
        "de SIGA GROUP."
    )
    st.stop()

# ============================================================
# 2. Header
# ============================================================
st.markdown(
    """
    <div style="background:linear-gradient(135deg,#1e3a8a 0%,#0f766e 100%);
                padding:24px 28px;border-radius:12px;color:white;
                margin-bottom:24px;
                box-shadow:0 4px 16px rgba(15,118,110,0.20);">
        <div style="font-size:11px;color:#a7f3d0;letter-spacing:2.5px;
                    text-transform:uppercase;font-weight:700;margin-bottom:4px;">
            SIGA INTERNAL · Operations
        </div>
        <div style="font-size:24px;font-weight:800;line-height:1.1;">
            🔐 Admin de Licencias · Watermelon Planta
        </div>
        <div style="font-size:13px;opacity:0.85;margin-top:6px;">
            Emisión, revocación y monitoreo de licencias activas
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. Cargar licencias emitidas (desde tools/licenses_issued/)
# ============================================================
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LICENSES_DIR = _REPO_ROOT / "tools" / "licenses_issued"


@st.cache_data(ttl=30)
def _load_issued_licenses() -> List[Dict[str, Any]]:
    """Lee todos los license.json del directorio interno de SIGA."""
    if not _LICENSES_DIR.exists():
        return []
    licenses = []
    for client_dir in sorted(_LICENSES_DIR.iterdir()):
        if not client_dir.is_dir():
            continue
        json_path = client_dir / "license.json"
        if not json_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            licenses.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return licenses


@st.cache_data(ttl=15)
def _load_revoked_licenses() -> Dict[str, dict]:
    """Lee la tabla revoked_licenses de Supabase. Returns dict[license_id → row]."""
    try:
        sb = get_supabase_client()
        result = sb.table("revoked_licenses") \
            .select("license_id, revoked_at, revoked_by, reason") \
            .execute()
        return {row["license_id"]: row for row in (result.data or [])}
    except Exception as e:  # noqa: BLE001
        st.warning(f"No se pudo leer la tabla revoked_licenses: {e}")
        return {}


_issued = _load_issued_licenses()
_revoked_map = _load_revoked_licenses()

# ============================================================
# 4. KPI Bar
# ============================================================
_n_total = len(_issued)
_n_revoked = sum(1 for lic in _issued if lic.get("license_id") in _revoked_map)
_n_expired = sum(
    1 for lic in _issued
    if lic.get("expires_at_utc")
    and datetime.fromisoformat(lic["expires_at_utc"]) < datetime.now(timezone.utc)
)
_n_active = _n_total - _n_revoked - _n_expired

_k1, _k2, _k3, _k4 = st.columns(4)
_k1.metric("Total emitidas", _n_total)
_k2.metric("✓ Activas", _n_active)
_k3.metric("⌛ Vencidas", _n_expired)
_k4.metric("🔒 Revocadas", _n_revoked)

st.divider()

# ============================================================
# 5. Tabla de licencias + acciones
# ============================================================
st.markdown("### Licencias emitidas")

if not _issued:
    st.info(
        f"No hay licencias emitidas todavía. Genera una con:\n\n"
        f"```bash\npython tools/license_issue.py --customer \"<Cliente>\" "
        f"--email \"<email>\" --plan pro\n```"
    )
    st.stop()

# Filtros
_f1, _f2, _f3 = st.columns([2, 1, 1])
with _f1:
    _search = st.text_input(
        "🔍 Buscar por cliente / email / license_id",
        key="lic_search",
    ).lower().strip()
with _f2:
    _filter_status = st.selectbox(
        "Estado",
        ["Todos", "Activas", "Vencidas", "Revocadas"],
        key="lic_filter_status",
    )
with _f3:
    _filter_plan = st.selectbox(
        "Plan",
        ["Todos"] + sorted({lic.get("plan", "") for lic in _issued if lic.get("plan")}),
        key="lic_filter_plan",
    )


def _classify(lic: dict) -> str:
    if lic.get("license_id") in _revoked_map:
        return "Revocadas"
    try:
        exp = datetime.fromisoformat(lic["expires_at_utc"])
        if exp < datetime.now(timezone.utc):
            return "Vencidas"
    except (KeyError, ValueError):
        pass
    return "Activas"


# Aplicar filtros
_filtered = []
for lic in _issued:
    if _search:
        haystack = " ".join([
            str(lic.get("customer", "")),
            str(lic.get("email", "")),
            str(lic.get("license_id", "")),
        ]).lower()
        if _search not in haystack:
            continue
    status = _classify(lic)
    if _filter_status != "Todos" and status != _filter_status:
        continue
    if _filter_plan != "Todos" and lic.get("plan") != _filter_plan:
        continue
    _filtered.append((lic, status))

st.caption(f"Mostrando {len(_filtered)} de {len(_issued)} licencias")

# ============================================================
# 6. Render cada licencia como card con acciones
# ============================================================

_STATUS_COLORS = {
    "Activas":   ("#10b981", "rgba(16,185,129,0.08)"),
    "Vencidas":  ("#6b7280", "rgba(107,114,128,0.08)"),
    "Revocadas": ("#ef4444", "rgba(239,68,68,0.08)"),
}

for lic, status in _filtered:
    _lid = lic.get("license_id", "")
    _is_revoked = _lid in _revoked_map
    _rev = _revoked_map.get(_lid, {})
    _col, _bg = _STATUS_COLORS[status]

    with st.container():
        st.markdown(
            f"""
            <div style="background:#ffffff;border-radius:10px;padding:16px;
                        border:1px solid {_col}33;border-left:4px solid {_col};
                        margin-bottom:14px;
                        box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;
                            gap:16px;">
                    <div style="flex:1;min-width:0;">
                        <div style="display:flex;align-items:center;gap:10px;
                                    margin-bottom:6px;">
                            <span style="font-size:16px;font-weight:700;color:#0f172a;">
                                {lic.get('customer', '—')}
                            </span>
                            <span style="background:{_bg};color:{_col};
                                         padding:2px 8px;border-radius:5px;
                                         font-size:10px;font-weight:700;
                                         letter-spacing:1px;text-transform:uppercase;">
                                {status}
                            </span>
                            <span style="background:rgba(15,118,110,0.08);color:#0f766e;
                                         padding:2px 8px;border-radius:5px;
                                         font-size:10px;font-weight:700;
                                         letter-spacing:1px;text-transform:uppercase;">
                                {lic.get('plan', '—')}
                            </span>
                        </div>
                        <div style="font-size:12px;color:#475569;line-height:1.5;">
                            📧 {lic.get('email', '—')}<br>
                            📅 Emitida {lic.get('issued_at_utc', '—')[:10]} ·
                            Vence {lic.get('expires_at_utc', '—')[:10]}<br>
                            🔑 <code style="font-size:10px;">{_lid}</code><br>
                            📦 {', '.join(lic.get('modules', [])).upper()} ·
                            hasta {lic.get('max_channels', '—')} canales
                            {f"<br>📝 <i>{lic.get('internal_notes')}</i>" if lic.get('internal_notes') else ""}
                            {f"<br>🔒 <b style='color:#ef4444;'>REVOCADA</b> el {_rev.get('revoked_at','')[:10]} por {_rev.get('revoked_by','')} — <i>{_rev.get('reason','')}</i>" if _is_revoked else ""}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Botones de acción
        _b1, _b2, _b3 = st.columns([1, 1, 4])
        if _is_revoked:
            with _b1:
                if st.button("♻ Reactivar", key=f"reactivate_{_lid}",
                              type="secondary", use_container_width=True):
                    try:
                        sb = get_supabase_client()
                        sb.table("revoked_licenses") \
                            .delete().eq("license_id", _lid).execute()
                        st.cache_data.clear()
                        st.success(f"✓ Licencia {_lid[:8]}... reactivada")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Error al reactivar: {e}")
        else:
            with _b1:
                _show_revoke_form = st.checkbox(
                    "🔒 Revocar",
                    key=f"show_rev_{_lid}",
                )
            if _show_revoke_form:
                with st.form(f"revoke_form_{_lid}", clear_on_submit=False):
                    _reason = st.text_input(
                        "Motivo de la revocación (visible al cliente)",
                        placeholder="Ej: Incumplimiento contractual — pago vencido 60 días",
                        key=f"reason_{_lid}",
                    )
                    _confirm = st.text_input(
                        f"Escribe el nombre del cliente para confirmar: «{lic.get('customer', '')}»",
                        key=f"confirm_{_lid}",
                    )
                    _submitted = st.form_submit_button("🔒 CONFIRMAR REVOCACIÓN",
                                                         type="primary")
                    if _submitted:
                        if not _reason.strip():
                            st.error("El motivo es obligatorio.")
                        elif _confirm.strip() != lic.get("customer", "").strip():
                            st.error("El nombre del cliente no coincide. Cancelado.")
                        else:
                            try:
                                sb = get_supabase_client()
                                sb.table("revoked_licenses").insert({
                                    "license_id": _lid,
                                    "revoked_by": _user_email,
                                    "reason": _reason.strip(),
                                    "customer": lic.get("customer"),
                                    "customer_email": lic.get("email"),
                                }).execute()
                                st.cache_data.clear()
                                st.success(
                                    f"✓ Licencia REVOCADA. La próxima vez que el "
                                    f"cliente abra Planta con internet, será bloqueado."
                                )
                                st.rerun()
                            except Exception as e:  # noqa: BLE001
                                st.error(f"Error al revocar: {e}")

# ============================================================
# 7. Footer info
# ============================================================
st.divider()
with st.expander("ℹ Cómo funciona la revocación"):
    st.markdown(
        """
        **Flujo técnico de la revocación:**

        1. Cuando hacés click en "Revocar", el `license_id` se inserta en la tabla
           `revoked_licenses` de Supabase con el motivo y tu email.
        2. La Edge Function `license-check` lee de esa tabla.
        3. Watermelon Planta del cliente chequea ese endpoint **al arrancar**,
           con timeout de 5 segundos y máximo 1 vez cada 24h (cached).
        4. Si el endpoint responde `revoked` → la app del cliente se bloquea
           inmediatamente con tu motivo y email de contacto SIGA.
        5. Si el cliente está offline → sigue funcionando hasta que se conecte.
           Si pasan > 30 días sin poder validar → bloqueo automático por seguridad.

        **Reactivación:** click en ♻ borra el `license_id` de la blacklist.
        Próximo arranque online del cliente → desbloqueado.

        **No funciona si:**
        - El cliente NUNCA conecta a internet (queda en grace period 30 días)
        - El endpoint Supabase está caído (cliente queda en último estado conocido)
        """
    )
