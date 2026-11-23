"""Administración → Usuarios (vista simple).

Una sola lista de personas (Supabase Auth = fuente única): nombre, empresa,
rol. Invitar usuario; al elegir una persona: cambiar rol,
nombre, bloquear/desbloquear, eliminar (con confirmación). El panel completo
anterior queda en "Avanzado".
"""
from __future__ import annotations

import html
import streamlit as st

from core.admin import ui

_ROLE_PILL = {"admin": ("Admin", "ac"), "specialist": ("SIGA", "ne"),
              "client": ("Cliente", "ok"), "viewer": ("Visor", "ne")}


def render() -> None:
    from core.auth import get_current_user
    from core.supabase_auth import (ROLES, block_user, create_user, delete_user,
                                    generate_temp_password, is_admin_email,
                                    is_supabase_auth_enabled, list_all_users,
                                    unblock_user, update_user_full_name,
                                    update_user_role)
    me = get_current_user() or {}
    if not is_admin_email(me.get("email", "")):
        st.error("Sección exclusiva del administrador.")
        return
    if not is_supabase_auth_enabled():
        st.error("El inicio de sesión (Supabase Auth) no está configurado.")
        return

    users = sorted(list_all_users() or [],
                   key=lambda u: ({"admin": 0, "specialist": 1, "client": 2}.get(u.get("role"), 3),
                                  str(u.get("full_name") or u.get("email")).lower()))
    company = {}
    try:
        from core.clients import list_clients
        for c in list_clients() or []:
            for e in c.owner_emails or ():
                company[e.lower()] = c.display_name
    except Exception:  # noqa: BLE001
        pass

    ui.title("Usuarios", "Quién entra y qué puede hacer")
    rows = []
    for u in users:
        lbl, kind = _ROLE_PILL.get(u.get("role"), (u.get("role", "—"), "ne"))
        if u.get("is_blocked"):
            lbl, kind = "Bloqueado", "da"
        comp = company.get(str(u.get("email", "")).lower(), "")
        left = (f"<b>{html.escape(u.get('full_name') or u.get('email'))}</b> "
                f"<span class='adm-mut'>{html.escape(comp or u.get('email', ''))}</span>")
        rows.append((left, ui.pill(lbl, kind)))
    ui.group("", rows or [("Sin usuarios", "")])

    c1, c2 = st.columns(2)
    with c1.popover("Invitar usuario", use_container_width=True):
        with st.form("adm_invite", clear_on_submit=True):
            em = st.text_input("Correo").strip().lower()
            nm = st.text_input("Nombre completo").strip()
            role = st.selectbox("Rol", ["specialist", "client", "admin"],
                                format_func=lambda r: {"specialist": "SIGA (especialista)",
                                                       "client": "Cliente", "admin": "Admin"}[r])
            ok = st.form_submit_button("Invitar", type="primary")
        if ok:
            if "@" not in em:
                st.error("Escribe un correo válido.")
            else:
                r = create_user(email=em, password=generate_temp_password(),
                                full_name=nm or em, role=role)
                if r.get("ok"):
                    st.success(f"{em} invitado. Entra con un código que le llega al correo."
                               + (" Asígnalo a su empresa en Clientes." if role == "client" else ""))
                    st.rerun()
                else:
                    st.error(f"No se pudo crear: {r.get('error')}")

    with c2.popover("Editar persona", use_container_width=True):
        if not users:
            st.caption("Sin usuarios.")
        else:
            i = st.selectbox("Persona", range(len(users)),
                             format_func=lambda k: users[k].get("full_name") or users[k].get("email"),
                             key="adm_edit_pick")
            u = users[i]
            is_self = str(u.get("email", "")).lower() == str(me.get("email", "")).lower()
            with st.form(f"adm_edit_{u['id']}"):
                nm = st.text_input("Nombre", u.get("full_name") or "")
                rl = st.selectbox("Rol", list(ROLES), index=list(ROLES).index(u.get("role"))
                                  if u.get("role") in ROLES else 2,
                                  format_func=lambda r: ROLES[r], disabled=is_self)
                save = st.form_submit_button("Guardar", type="primary")
            if save:
                if nm and nm != u.get("full_name"):
                    update_user_full_name(u["id"], nm)
                if not is_self and rl != u.get("role"):
                    update_user_role(u["id"], rl)
                st.success("Guardado.")
                st.rerun()
            if not is_self:
                b1, b2 = st.columns(2)
                if u.get("is_blocked"):
                    if b1.button("Desbloquear", key=f"adm_unb_{u['id']}"):
                        unblock_user(u["id"])
                        st.rerun()
                elif b1.button("Bloquear acceso", key=f"adm_blk_{u['id']}"):
                    block_user(u["id"])
                    st.rerun()
                conf = b2.checkbox("Confirmo eliminar", key=f"adm_delc_{u['id']}")
                if b2.button("Eliminar usuario", key=f"adm_del_{u['id']}", disabled=not conf):
                    delete_user(u["id"])
                    st.rerun()

    with st.expander("Avanzado · panel completo de usuarios"):
        from core.admin.users import render as legacy
        legacy()
