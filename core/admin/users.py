"""
core.admin.users — Sección Administración · Usuarios.

Gestión de usuarios de Supabase Auth (crear, role, bloquear, reset password,
eliminar). Extraído de pages/_admin_users.py como render() sin efectos de
import. El hub ya autenticó; acá se conserva el guard is_admin_email (defensa
en profundidad) y el chequeo de Supabase Auth.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import streamlit as st

from core.auth import get_current_user
from core.supabase_auth import (
    ADMIN_EMAIL, ROLES, block_user, create_user, delete_user,
    generate_temp_password, infer_role_from_email, is_admin_email,
    is_supabase_auth_enabled, list_all_users, reset_user_password,
    unblock_user, update_user_full_name, update_user_role,
)

_STYLES = """
<style>
.wmu-hero { border-radius:18px; padding:22px 28px; margin-bottom:22px;
  background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#0f172a 100%);
  border:1px solid rgba(148,163,184,0.16); color:#f8fafc; }
.wmu-pill { display:inline-block; padding:5px 12px; border-radius:999px;
  background:rgba(248,113,113,0.15); border:1px solid rgba(248,113,113,0.3);
  color:#fca5a5; font-size:10px; font-weight:800; letter-spacing:0.18em;
  text-transform:uppercase; margin-bottom:10px; }
.wmu-title { font-size:28px; font-weight:800; margin:0 0 6px 0; color:#f8fafc; }
.wmu-subtitle { color:rgba(226,232,240,0.78); font-size:14px; }
.wmu-card { background:white; border:1px solid #e6ebf2; border-radius:14px;
  padding:14px 18px; margin-bottom:10px; }
.wmu-row-head { display:flex; align-items:center; justify-content:space-between; gap:14px; }
.wmu-email { font-weight:800; color:#0f172a; font-size:15px; }
.wmu-name { color:#475569; font-size:13px; margin-top:2px; }
.wmu-meta { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px;
  color:#94a3b8; margin-top:6px; }
.wmu-role { font-size:11px; font-weight:800; padding:4px 10px; border-radius:999px;
  letter-spacing:0.06em; }
.wmu-role-admin { background:#fee2e2; color:#b91c1c; }
.wmu-role-specialist { background:#dbeafe; color:#1d4ed8; }
.wmu-role-client { background:#d1fae5; color:#047857; }
.wmu-role-viewer { background:#f1f5f9; color:#475569; }
.wmu-blocked-pill { display:inline-block; background:#fef3c7; color:#b45309;
  font-size:10px; font-weight:800; padding:3px 9px; border-radius:999px; margin-left:8px; }
.wmu-admin-pill { background:linear-gradient(135deg,#ef4444,#dc2626); color:white;
  font-size:10px; font-weight:800; padding:3px 9px; border-radius:999px; margin-left:8px; }
.wmu-section { font-size:11px; font-weight:800; letter-spacing:0.18em; text-transform:uppercase;
  color:#475569; margin:22px 0 10px 0; display:flex; align-items:center; gap:10px; }
.wmu-section .bar { flex:1; height:1px; background:linear-gradient(90deg,#cbd5e1 0%,transparent 100%); }
</style>
"""


def _load_users(force_refresh: bool = False) -> List[Dict[str, Any]]:
    if force_refresh:
        st.session_state.pop("_admin_users_cache", None)
    cached = st.session_state.get("_admin_users_cache")
    if cached is not None:
        return cached
    users = list_all_users()
    st.session_state["_admin_users_cache"] = users
    return users


def _humanize_iso(ts: str) -> str:
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")[:25])
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts[:16]


def render() -> None:
    _user = get_current_user() or {}
    if not _user or not is_admin_email(_user.get("email", "")):
        st.error("🚫 **Acceso denegado.** Esta sección es exclusiva del administrador "
                 f"del sistema (`{ADMIN_EMAIL}`).")
        return
    if not is_supabase_auth_enabled():
        st.error("⚠️ **Supabase Auth no está configurado.** Verificá que "
                 "`st.secrets['supabase']['url']` y `service_key` estén definidos.")
        return

    # Force refresh del cliente Supabase admin (evita key vieja cacheada).
    st.session_state.pop("_supabase_admin_client", None)

    st.markdown(_STYLES, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="wmu-hero">
            <span class="wmu-pill">🔐 ZONA ADMIN · Solo {ADMIN_EMAIL}</span>
            <div class="wmu-title">👥 Gestión de Usuarios</div>
            <div class="wmu-subtitle">
                Crear, modificar, bloquear y eliminar usuarios del sistema.
                Los roles se asignan automáticamente según el dominio del email
                (admin · sigasas.com → specialist · otros → client) pero podés
                sobreescribirlos manualmente.
            </div>
        </div>
        """, unsafe_allow_html=True)

    _all_users = _load_users()
    _n_total = len(_all_users)
    _n_admin = sum(1 for u in _all_users if is_admin_email(u.get("email", "")))
    _n_specialist = sum(1 for u in _all_users if u.get("role") == "specialist")
    _n_client = sum(1 for u in _all_users if u.get("role") == "client")
    _n_blocked = sum(1 for u in _all_users if u.get("is_blocked"))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total usuarios", _n_total)
    k2.metric("Administradores", _n_admin)
    k3.metric("Especialistas", _n_specialist)
    k4.metric("Clientes", _n_client)
    k5.metric("Bloqueados", _n_blocked)

    # --- Crear nuevo usuario ---
    st.markdown('<div class="wmu-section">➕ Crear nuevo usuario <div class="bar"></div></div>',
                unsafe_allow_html=True)
    with st.expander("Abrir formulario de creación", expanded=False):
        with st.form("new_user_form", clear_on_submit=True):
            nf1, nf2 = st.columns([0.5, 0.5])
            with nf1:
                new_email = st.text_input("Correo del nuevo usuario",
                                          placeholder="nombre.apellido@sigasas.com",
                                          key="new_user_email").strip().lower()
                new_full_name = st.text_input("Nombre completo", placeholder="Nombre Apellido",
                                              key="new_user_name").strip()
            with nf2:
                _suggested_role = infer_role_from_email(new_email) if new_email else "client"
                _role_codes = list(ROLES.keys())
                _role_labels = [f"{code} — {label}" for code, label in ROLES.items()]
                _idx_default = _role_codes.index(_suggested_role) if _suggested_role in _role_codes else 2
                new_role_idx = st.selectbox("Role", options=range(len(_role_codes)),
                                            format_func=lambda i: _role_labels[i],
                                            index=_idx_default, key="new_user_role",
                                            help=f"Sugerido automáticamente según dominio: "
                                                 f"'{_suggested_role}'. Podés cambiarlo.")
                new_role = _role_codes[new_role_idx]
                new_pwd = generate_temp_password()
                st.caption("🔑 Acceso **sin contraseña**: el usuario ingresará con un "
                           "código de un solo uso enviado a su correo.")

            _client_target_id: str = ""
            if new_role == "client":
                try:
                    from core.clients import list_clients
                    _all_clients = list_clients()
                    _client_options = [(c.id, c.display_name) for c in _all_clients]
                    if _client_options:
                        _opt_labels = ["(no asignar todavía)"] + [
                            f"{disp}  ·  {cid}" for cid, disp in _client_options]
                        _opt_ids = [""] + [cid for cid, _ in _client_options]
                        _sel = st.selectbox("Asignar al cliente", options=range(len(_opt_labels)),
                                            format_func=lambda i: _opt_labels[i], index=0,
                                            key="new_user_client_target",
                                            help="El email se agrega al owner_emails de ese cliente.")
                        _client_target_id = _opt_ids[_sel]
                    else:
                        st.caption("_(no hay clientes en el registry — agregá uno en data/clients.json)_")
                except Exception as _e:  # noqa: BLE001
                    st.caption(f"_(error cargando clientes: {_e})_")

            submitted = st.form_submit_button("Crear usuario", type="primary",
                                              use_container_width=True)
        if submitted:
            if not new_email or "@" not in new_email:
                st.error("Email inválido.")
            elif not new_full_name:
                st.error("Falta el nombre completo.")
            else:
                result = create_user(email=new_email, password=new_pwd,
                                     full_name=new_full_name, role=new_role)
                if result.get("ok"):
                    _assigned_msg = ""
                    if new_role == "client" and _client_target_id:
                        try:
                            from core.clients import assign_client_to_email
                            if assign_client_to_email(_client_target_id, new_email):
                                _assigned_msg = (f"\n\n✓ Asignado al cliente "
                                                 f"**{_client_target_id}** — verá solo los activos "
                                                 "de ese cliente en Live Monitoring.")
                            else:
                                _assigned_msg = (f"\n\n⚠ No se pudo asignar al cliente "
                                                 f"`{_client_target_id}` — editá data/clients.json "
                                                 f"y agregá `{new_email}` a su owner_emails.")
                        except Exception as _ae:  # noqa: BLE001
                            _assigned_msg = f"\n\n⚠ Error asignando cliente: {_ae}"
                    st.success(f"✓ Usuario **{new_email}** creado como **{new_role}**.\n\n"
                               f"Ingresa **sin contraseña**: que entre a la app, escriba "
                               f"**{new_email}** y reciba el código en su correo." + _assigned_msg)
                    st.session_state.pop("_admin_users_cache", None)
                    st.session_state.pop("_new_user_temp_pwd", None)
                else:
                    st.error(f"✗ No se pudo crear: {result.get('error', 'error desconocido')}")

    # --- Búsqueda + refresh ---
    st.markdown('<div class="wmu-section">📋 Usuarios registrados <div class="bar"></div></div>',
                unsafe_allow_html=True)
    ctop1, ctop2 = st.columns([0.7, 0.3])
    with ctop1:
        _q = st.text_input("Buscar por email o nombre", placeholder="Buscar...",
                           key="admin_users_search", label_visibility="collapsed").strip().lower()
    with ctop2:
        if st.button("🔄  Refrescar lista", use_container_width=True, key="refresh_users"):
            _all_users = _load_users(force_refresh=True)
            st.rerun()

    if _q:
        visible = [u for u in _all_users
                   if _q in (u.get("email") or "").lower()
                   or _q in (u.get("full_name") or "").lower()]
    else:
        visible = _all_users
    ROLE_ORDER = {"admin": 0, "specialist": 1, "client": 2, "viewer": 3}
    visible.sort(key=lambda u: (ROLE_ORDER.get(u.get("role", "client"), 9),
                                (u.get("email") or "").lower()))

    if not visible:
        st.info("No hay usuarios que coincidan con tu búsqueda.")
    else:
        for u in visible:
            uid = u.get("id", "")
            email = u.get("email", "")
            full_name = u.get("full_name", "")
            role = u.get("role", "client")
            is_blocked = u.get("is_blocked", False)
            is_protected_admin = is_admin_email(email)

            blocked_pill = '<span class="wmu-blocked-pill">🚫 BLOQUEADO</span>' if is_blocked else ""
            admin_pill = '<span class="wmu-admin-pill">🔐 ADMIN ÚNICO</span>' if is_protected_admin else ""
            role_class = f"wmu-role-{role}" if role in ROLES else "wmu-role-viewer"
            role_label = ROLES.get(role, role)
            st.markdown(
                f"""
                <div class="wmu-card">
                    <div class="wmu-row-head">
                        <div>
                            <div class="wmu-email">{email}{admin_pill}{blocked_pill}</div>
                            <div class="wmu-name">{full_name or '(sin nombre)'}</div>
                            <div class="wmu-meta">
                                ID: {uid[:8]}…  ·  Creado: {_humanize_iso(u.get('created_at', ''))}  ·
                                Último login: {_humanize_iso(u.get('last_sign_in_at', ''))}
                            </div>
                        </div>
                        <div><span class="wmu-role {role_class}">{role_label.upper()}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if is_protected_admin:
                st.caption("🔒 El administrador único del sistema no puede ser modificado "
                           "desde esta UI (protección anti-lockout).")
            else:
                ac1, ac2, ac3, ac4, ac5 = st.columns(5)
                with ac1:
                    with st.popover("🎭  Cambiar role", use_container_width=True):
                        _opts = list(ROLES.keys())
                        _idx = _opts.index(role) if role in _opts else 2
                        new_role_pick = st.selectbox(f"Nuevo role para {email}", options=_opts,
                                                     format_func=lambda c: f"{c} — {ROLES[c]}",
                                                     index=_idx, key=f"role_pick_{uid}")
                        if st.button("Aplicar cambio", key=f"role_apply_{uid}",
                                     use_container_width=True):
                            res = update_user_role(uid, new_role_pick)
                            if res.get("ok"):
                                st.success("Role actualizado.")
                                st.session_state.pop("_admin_users_cache", None)
                                st.rerun()
                            else:
                                st.error(res.get("error", "Falló."))
                with ac2:
                    with st.popover("🔑  Reset password", use_container_width=True):
                        _temp = st.text_input("Nueva password (auto-generada)",
                                              value=generate_temp_password(), key=f"pwd_reset_{uid}")
                        if st.button("Aplicar reset", key=f"pwd_apply_{uid}",
                                     use_container_width=True):
                            res = reset_user_password(uid, _temp)
                            if res.get("ok"):
                                st.success(f"✓ Password reseteada. Entregale al usuario: `{_temp}`")
                                st.session_state.pop("_admin_users_cache", None)
                            else:
                                st.error(res.get("error", "Falló."))
                with ac3:
                    if is_blocked:
                        if st.button("✅  Desbloquear", key=f"unblock_{uid}",
                                     use_container_width=True):
                            res = unblock_user(uid)
                            if res.get("ok"):
                                st.success("Usuario desbloqueado.")
                                st.session_state.pop("_admin_users_cache", None)
                                st.rerun()
                            else:
                                st.error(res.get("error", "Falló."))
                    else:
                        if st.button("🚫  Bloquear", key=f"block_{uid}",
                                     use_container_width=True):
                            res = block_user(uid)
                            if res.get("ok"):
                                st.success("Usuario bloqueado.")
                                st.session_state.pop("_admin_users_cache", None)
                                st.rerun()
                            else:
                                st.error(res.get("error", "Falló."))
                with ac4:
                    with st.popover("✏️  Editar nombre", use_container_width=True):
                        _new_name = st.text_input("Nuevo nombre completo", value=full_name,
                                                  key=f"name_edit_{uid}")
                        if st.button("Guardar nombre", key=f"name_save_{uid}",
                                     use_container_width=True):
                            res = update_user_full_name(uid, _new_name)
                            if res.get("ok"):
                                st.success("Nombre actualizado.")
                                st.session_state.pop("_admin_users_cache", None)
                                st.rerun()
                            else:
                                st.error(res.get("error", "Falló."))
                with ac5:
                    with st.popover("🗑️  Eliminar", use_container_width=True):
                        st.warning(f"⚠️ Vas a eliminar **{email}** permanentemente. "
                                   "Esta acción no se puede deshacer.")
                        _confirm = st.text_input(f"Para confirmar, escribí: **{email}**",
                                                 key=f"del_confirm_{uid}").strip().lower()
                        if st.button("Eliminar definitivamente", key=f"del_apply_{uid}",
                                     type="primary", use_container_width=True):
                            if _confirm != email.lower():
                                st.error("El email no coincide. Operación cancelada.")
                            else:
                                res = delete_user(uid)
                                if res.get("ok"):
                                    st.success(f"Usuario {email} eliminado.")
                                    st.session_state.pop("_admin_users_cache", None)
                                    st.rerun()
                                else:
                                    st.error(res.get("error", "Falló."))

    st.divider()
    st.caption(f"📌 Total: {_n_total} usuarios · Roles: admin={_n_admin}, "
               f"specialist={_n_specialist}, client={_n_client}, bloqueados={_n_blocked}. "
               f"Conectado a Supabase como `{_user.get('email', '')}`.")


__all__ = ["render"]
