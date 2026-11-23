"""Administración → Clientes (vista simple).

Por cliente: sus máquinas con estado en línea, sus personas (y quién ve la
Suscripción), su plan. Acciones: agregar persona, agregar/quitar máquina,
nuevo cliente. Lo técnico (palabras de coincidencia, WhatsApp, API key,
registro crudo) queda en "Avanzado".
"""
from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st

from core.admin import ui


def _machine_status(tag: str, inst_by_tag: Dict[str, Dict[str, Any]]) -> tuple:
    """(pill, texto, detalle) del estado en línea de la máquina."""
    inst = inst_by_tag.get(tag.upper())
    if not inst:
        return "ne", "Sin configurar", "No existe en Machines"
    iid = inst.get("instance_id")
    kind = ""
    try:
        from core.machines_meta import machine_header
        kind = (machine_header(iid) or {}).get("type") or ""
    except Exception:  # noqa: BLE001
        pass
    try:
        from core.live_readings import latest_for_instance
        rows = latest_for_instance(iid) or []
        ts = [datetime.fromisoformat(str(r["captured_at"]).replace("Z", "+00:00"))
              for r in rows if r.get("captured_at")]
        if not ts:
            return "ne", "Sin datos", kind
        age_h = (datetime.now(timezone.utc) - max(ts)).total_seconds() / 3600
        if age_h <= 1:
            return "ok", "En línea", kind
        since = max(ts).astimezone().strftime("%d/%m")
        return "ne", "Fuera de línea", (f"{kind} · " if kind else "") + f"desde {since}"
    except Exception:  # noqa: BLE001
        return "ne", "—", kind


def render() -> None:
    from core.clients import (assign_client_to_email, list_clients, reload_registry,
                              remove_email_from_client, update_client)
    reload_registry()
    clients = list_clients() or []

    users = []
    try:
        from core.supabase_auth import list_all_users
        users = list_all_users() or []
    except Exception:  # noqa: BLE001
        pass
    name_by_email = {str(u.get("email", "")).lower(): u.get("full_name") or u.get("email")
                     for u in users}

    if not clients:
        st.info("No hay clientes todavía.")
    else:
        if len(clients) > 1:
            k = st.selectbox("Cliente", range(len(clients)),
                             format_func=lambda i: clients[i].display_name,
                             key="adm_cli_pick", label_visibility="collapsed")
        else:
            k = 0
        c = clients[k]

        sub = None
        try:
            from core.billing.store import get_subscription
            sub = get_subscription(c.id)
        except Exception:  # noqa: BLE001
            pass
        plan_txt = "Sin suscripción"
        if sub:
            from core.billing.pricing import PLANS
            plan_txt = f"Plan {PLANS[sub['plan']].name}" + ("" if sub.get("status") == "activa"
                                                            else f" · {sub.get('status')}")
        ui.title(c.display_name, f"Cliente activo · {plan_txt}")

        # --- Máquinas ---
        try:
            from core.instance_state import list_instances
            inst_by_tag = {str(i.get("tag", "")).upper(): i for i in (list_instances() or [])
                           if isinstance(i, dict)}
        except Exception:  # noqa: BLE001
            inst_by_tag = {}
        rows = []
        for t in c.asset_tags or ():
            pill, txt, det = _machine_status(t, inst_by_tag)
            rows.append((f'{ui.pill(txt, pill)} <b>{html.escape(t)}</b>', html.escape(det or "")))
        ui.group("Máquinas", rows or [("Sin máquinas", "")])

        # --- Personas ---
        authorized = {e.lower() for e in ((sub or {}).get("authorized_emails") or [])}
        prow = []
        for e in c.owner_emails or ():
            el = e.lower()
            nm = name_by_email.get(el)
            left = (f"<b>{html.escape(nm)}</b> <span class='adm-mut'>{html.escape(e)}</span>"
                    if nm and nm != e else f"<b>{html.escape(e)}</b>"
                    + ("" if nm else " <span class='adm-mut'>· sin usuario creado</span>"))
            prow.append((left, ui.pill("Ve Suscripción", "ac") if el in authorized else ""))
        ui.group("Personas", prow or [("Nadie tiene acceso todavía", "")])

        # --- Acciones ---
        a1, a2, a3 = st.columns(3)
        with a1.popover("Agregar persona", use_container_width=True):
            with st.form(f"adm_add_person_{c.id}", clear_on_submit=True):
                em = st.text_input("Correo").strip().lower()
                nm = st.text_input("Nombre completo").strip()
                can_pay = st.checkbox("Puede ver Suscripción (pagos)")
                ok = st.form_submit_button("Agregar", type="primary")
            if ok:
                if "@" not in em:
                    st.error("Escribe un correo válido.")
                else:
                    msg = _add_person(c.id, em, nm, can_pay, name_by_email)
                    st.success(msg)
                    st.rerun()
        with a2.popover("Máquinas", use_container_width=True):
            opts = sorted(set(list(inst_by_tag) + [t.upper() for t in (c.asset_tags or ())]))
            with st.form(f"adm_machines_{c.id}"):
                sel = st.multiselect("Máquinas del cliente", opts,
                                     default=[t.upper() for t in (c.asset_tags or ())])
                if st.form_submit_button("Guardar", type="primary"):
                    update_client(c.id, asset_tags=sel)
                    st.success("Máquinas actualizadas.")
                    st.rerun()
        with a3.popover("Quitar persona", use_container_width=True):
            if c.owner_emails:
                with st.form(f"adm_rm_person_{c.id}"):
                    who = st.selectbox("Persona", list(c.owner_emails))
                    if st.form_submit_button("Quitar acceso"):
                        remove_email_from_client(c.id, who)
                        st.success(f"{who} ya no ve los activos de {c.display_name}.")
                        st.rerun()
            else:
                st.caption("No hay personas.")

        with st.expander("Avanzado · palabras de coincidencia, WhatsApp"):
            with st.form(f"adm_adv_{c.id}"):
                dn = st.text_input("Nombre", c.display_name)
                ms = st.text_input("Palabras de coincidencia (separadas por coma)",
                                   ", ".join(c.match_strings or ()),
                                   help="Un reporte o activo pertenece al cliente si su nombre "
                                        "contiene alguna de estas palabras.")
                wa = st.text_input("WhatsApp (separados por coma)",
                                   ", ".join(c.whatsapp_numbers or ()))
                if st.form_submit_button("Guardar"):
                    update_client(c.id, display_name=dn.strip() or c.display_name,
                                  match_strings=[x.strip().lower() for x in ms.split(",") if x.strip()],
                                  whatsapp_numbers=[x.strip() for x in wa.split(",") if x.strip()])
                    st.success("Guardado.")
                    st.rerun()

    st.write("")
    with st.popover("Nuevo cliente"):
        with st.form("adm_new_client", clear_on_submit=True):
            dn = st.text_input("Nombre de la empresa")
            ms = st.text_input("Palabra para reconocer sus activos y reportes",
                               placeholder="ej. nombre corto de la empresa")
            if st.form_submit_button("Crear", type="primary"):
                cid = re.sub(r"[^a-z0-9]+", "_", dn.strip().lower()).strip("_")
                if not cid:
                    st.error("Escribe el nombre.")
                elif cid in {x.id for x in clients}:
                    st.error("Ya existe un cliente con ese nombre.")
                else:
                    update_client(cid, display_name=dn.strip(),
                                  match_strings=[ms.strip().lower()] if ms.strip() else [cid])
                    st.success("Cliente creado.")
                    st.rerun()

    with st.expander("Editor completo del registro (técnico)"):
        from core.admin.clients import render as legacy
        legacy(embedded=True)


def _add_person(client_id: str, email: str, full_name: str, can_pay: bool,
                name_by_email: Dict[str, Any]) -> str:
    from core.clients import assign_client_to_email
    parts: List[str] = []
    if email not in name_by_email:
        try:
            from core.supabase_auth import create_user, generate_temp_password
            r = create_user(email=email, password=generate_temp_password(),
                            full_name=full_name or email, role="client")
            parts.append("usuario creado (entra con código a su correo)" if r.get("ok")
                         else f"no se pudo crear el usuario: {r.get('error')}")
        except Exception as e:  # noqa: BLE001
            parts.append(f"no se pudo crear el usuario: {e}")
    assign_client_to_email(client_id, email)
    parts.append("ya ve los activos del cliente")
    if can_pay:
        try:
            from core.billing import store
            s = store.get_subscription(client_id)
            if s:
                store.save_subscription(dict(s, authorized_emails=list(s.get("authorized_emails") or [])
                                             + [email]))
                parts.append("puede ver Suscripción")
            else:
                parts.append("el cliente aún no tiene suscripción")
        except Exception:  # noqa: BLE001
            pass
    return f"{email}: " + ", ".join(parts) + "."
