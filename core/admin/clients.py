"""
core.admin.clients — Sección Administración · Clientes, Specialists & Admins.

CRUD del registry multi-tenant `data/clients.json`. Extraído de
pages/_admin_clients.py como render() sin efectos de import. El hub ya autenticó
y validó role=admin. Los helpers quedan a nivel de módulo (resolubles siempre —
esto además corrige un uso-antes-de-definir latente del original).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from core.clients import (
    list_admins, list_clients, list_specialists, reload_registry, save_registry,
    _load_registry,
)


def _normalize_phone(p: str) -> str:
    return re.sub(r"[^\d]", "", p or "")


def _phones_to_list(text: str) -> List[str]:
    raw = re.split(r"[,\n;]+", text or "")
    return [n for n in (_normalize_phone(p) for p in raw) if n]


def _list_to_text(items: List[str]) -> str:
    return ", ".join(items or [])


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower())
    return s.strip("_") or "client"


def _build_full_data(admins, specialists, clients):
    return {"admins": admins, "specialists": specialists, "clients": clients}


def _enrich_with_apikey(c: Dict[str, Any]) -> Dict[str, Any]:
    """Re-asocia api_key del raw json (la cache de list_clients no la expone)."""
    raw = _load_registry()
    api_key = ""
    for c_raw in raw.get("clients", []):
        if c_raw.get("id") == c.get("id"):
            api_key = c_raw.get("api_key", "")
            break
    out = dict(c)
    out["api_key"] = api_key
    return out


def render() -> None:
    st.title("Administración · Clientes, Specialists & Admins")
    st.caption(
        "Gestión del registry multi-tenant `data/clients.json`. "
        "Cada cambio se escribe atómicamente y toma efecto inmediato. "
        "Recordá hacer `git commit` periódicamente para versionarlo.")

    reload_registry()
    admins_data: List[Dict[str, Any]] = list_admins()
    specialists_data: List[Dict[str, Any]] = list_specialists()
    clients_data: List[Any] = [c.as_dict() for c in list_clients()]

    st.markdown(f"**Estado actual:** {len(admins_data)} admins · "
                f"{len(specialists_data)} specialists · {len(clients_data)} clientes")

    tab_clients, tab_specialists, tab_admins, tab_raw = st.tabs(
        ["🏭 Clientes", "🛠️ Specialists", "🔑 Admins", "📄 Ver JSON crudo"])

    # ---------------- CLIENTES ----------------
    with tab_clients:
        st.subheader("Clientes externos")
        st.caption("Cada cliente solo verá los activos cuyos `report_meta.client / "
                   "instance_tag / asset_class / train_description` contengan alguno "
                   "de sus *match_strings* (case-insensitive).")
        if clients_data:
            st.dataframe(pd.DataFrame([{
                "ID": c["id"], "Cliente": c["display_name"],
                "Match": ", ".join(c.get("match_strings", [])),
                "Assets": ", ".join(c.get("asset_tags", [])),
                "WhatsApp": ", ".join(c.get("whatsapp_numbers", [])),
                "Owners": ", ".join(c.get("owner_emails", [])),
            } for c in clients_data]), use_container_width=True, hide_index=True)
        else:
            st.info("Aún no hay clientes registrados.")

        st.markdown("---")
        st.markdown("### Editar cliente existente")
        if clients_data:
            cli_ids = [c["id"] for c in clients_data]
            sel_id = st.selectbox("Seleccionar cliente", options=["— nuevo —"] + cli_ids,
                                  key="client_edit_picker")
            if sel_id == "— nuevo —":
                current = {"id": "", "display_name": "", "match_strings": [],
                           "asset_tags": [], "whatsapp_numbers": [], "owner_emails": []}
            else:
                current = next(c for c in clients_data if c["id"] == sel_id)

            with st.form("client_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    new_id = st.text_input("ID (slug, no editable si ya existe)",
                                           value=current["id"],
                                           disabled=(sel_id != "— nuevo —"),
                                           placeholder="ej: termocaribe")
                    new_display = st.text_input("Nombre del cliente",
                                                value=current["display_name"],
                                                placeholder="Ej: Termocaribe S.A.")
                    new_match = st.text_input(
                        "Match strings (separadas por coma)",
                        value=", ".join(current["match_strings"]),
                        help="Substrings que deben aparecer en el meta del reporte. Case-insensitive.")
                with col_b:
                    new_assets = st.text_input("Asset tags informativos (separados por coma)",
                                               value=", ".join(current["asset_tags"]),
                                               placeholder="TES1, TES3, C-200-C")
                    new_phones = st.text_area("WhatsApp numbers (uno por línea o coma)",
                                              value=_list_to_text(current["whatsapp_numbers"]),
                                              height=70, placeholder="573185551234, 573185559876")
                    new_owners = st.text_area("Owner emails (separados por coma)",
                                              value=", ".join(current["owner_emails"]),
                                              height=70, placeholder="mantenimiento@cliente.com")

                col_save, col_del = st.columns([3, 1])
                with col_save:
                    save_clicked = st.form_submit_button("💾 Guardar cliente", type="primary",
                                                         use_container_width=True)
                with col_del:
                    del_clicked = st.form_submit_button("🗑️ Eliminar", use_container_width=True,
                                                        disabled=(sel_id == "— nuevo —"))

                if save_clicked:
                    target_id = (current["id"] or _slug(new_id) or _slug(new_display))
                    if not target_id:
                        st.error("ID y nombre no pueden estar ambos vacíos.")
                    elif not new_display.strip():
                        st.error("El nombre del cliente es obligatorio.")
                    else:
                        new_entry = {
                            "id": target_id, "display_name": new_display.strip(),
                            "match_strings": [s.strip().lower() for s in new_match.split(",") if s.strip()],
                            "asset_tags": [s.strip() for s in new_assets.split(",") if s.strip()],
                            "whatsapp_numbers": _phones_to_list(new_phones),
                            "owner_emails": [e.strip().lower() for e in new_owners.split(",") if e.strip()],
                            "api_key": "",
                        }
                        if sel_id != "— nuevo —":
                            raw = _load_registry()
                            for c_raw in raw.get("clients", []):
                                if c_raw.get("id") == sel_id:
                                    new_entry["api_key"] = c_raw.get("api_key", "")
                                    break
                            new_clients = [new_entry if c["id"] == sel_id else _enrich_with_apikey(c)
                                           for c in clients_data]
                        else:
                            if any(c["id"] == target_id for c in clients_data):
                                st.error(f"Ya existe un cliente con ID '{target_id}'.")
                                st.stop()
                            new_clients = [_enrich_with_apikey(c) for c in clients_data] + [new_entry]
                        try:
                            save_registry(_build_full_data(admins_data, specialists_data, new_clients))
                            st.success(f"✅ Cliente '{target_id}' guardado.")
                            st.rerun()
                        except Exception as e:  # noqa: BLE001
                            st.error(f"No se pudo guardar: {e}")

                if del_clicked and sel_id != "— nuevo —":
                    new_clients = [_enrich_with_apikey(c) for c in clients_data if c["id"] != sel_id]
                    try:
                        save_registry(_build_full_data(admins_data, specialists_data, new_clients))
                        st.success(f"🗑️ Cliente '{sel_id}' eliminado.")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"No se pudo eliminar: {e}")

    # ---------------- SPECIALISTS ----------------
    with tab_specialists:
        st.subheader("Specialists (equipo SIGA Cat IV)")
        st.caption("Tienen el mismo nivel de visibilidad que admin (ven TODOS los "
                   "activos del archivo) pero NO gestionan clientes.")
        if specialists_data:
            st.dataframe(pd.DataFrame([{
                "Nombre": s.get("name", ""), "Email": s.get("email", ""),
                "WhatsApp": ", ".join(s.get("whatsapp_numbers", [])),
            } for s in specialists_data]), use_container_width=True, hide_index=True)
        else:
            st.info("Aún no hay specialists registrados.")

        st.markdown("---")
        st.markdown("### Agregar / editar specialist")
        spec_emails = [s["email"] for s in specialists_data]
        sel_spec = st.selectbox("Seleccionar specialist", options=["— nuevo —"] + spec_emails,
                                key="spec_edit_picker")
        if sel_spec == "— nuevo —":
            cur_spec = {"name": "", "email": "", "whatsapp_numbers": []}
        else:
            cur_spec = next(s for s in specialists_data if s["email"] == sel_spec)

        with st.form("spec_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                new_name = st.text_input("Nombre", value=cur_spec.get("name", ""))
                new_email = st.text_input("Email corporativo", value=cur_spec.get("email", ""),
                                          disabled=(sel_spec != "— nuevo —"),
                                          placeholder="ej: nombre@sigasas.com")
            with col_b:
                new_phones = st.text_area("WhatsApp numbers (uno por línea o coma)",
                                          value=_list_to_text(cur_spec.get("whatsapp_numbers", [])),
                                          height=100)
            col_save, col_del = st.columns([3, 1])
            with col_save:
                save_spec = st.form_submit_button("💾 Guardar specialist", type="primary",
                                                  use_container_width=True)
            with col_del:
                del_spec = st.form_submit_button("🗑️ Eliminar", use_container_width=True,
                                                 disabled=(sel_spec == "— nuevo —"))
            if save_spec:
                if not new_name.strip() or not new_email.strip():
                    st.error("Nombre y email son obligatorios.")
                else:
                    new_entry = {"name": new_name.strip(), "email": new_email.strip().lower(),
                                 "whatsapp_numbers": _phones_to_list(new_phones)}
                    if sel_spec == "— nuevo —":
                        if any(s["email"] == new_entry["email"] for s in specialists_data):
                            st.error(f"Ya existe specialist con email '{new_entry['email']}'.")
                            st.stop()
                        new_specs = specialists_data + [new_entry]
                    else:
                        new_specs = [new_entry if s["email"] == sel_spec else s
                                     for s in specialists_data]
                    try:
                        save_registry(_build_full_data(
                            admins_data, new_specs,
                            [_enrich_with_apikey(c) for c in clients_data]))
                        st.success(f"✅ Specialist '{new_entry['name']}' guardado.")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"No se pudo guardar: {e}")
            if del_spec and sel_spec != "— nuevo —":
                new_specs = [s for s in specialists_data if s["email"] != sel_spec]
                try:
                    save_registry(_build_full_data(
                        admins_data, new_specs,
                        [_enrich_with_apikey(c) for c in clients_data]))
                    st.success(f"🗑️ Specialist '{sel_spec}' eliminado.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"No se pudo eliminar: {e}")

    # ---------------- ADMINS ----------------
    with tab_admins:
        st.subheader("Admins")
        st.warning("⚠️ Cuidado: los admins gestionan TODO el sistema. "
                   "**No te elimines a vos mismo** o perdés acceso a esta página.")
        if admins_data:
            st.dataframe(pd.DataFrame([{
                "Nombre": a.get("name", ""), "Email": a.get("email", ""),
                "WhatsApp": ", ".join(a.get("whatsapp_numbers", [])),
            } for a in admins_data]), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Agregar admin nuevo")
        with st.form("admin_form", clear_on_submit=True):
            col_a, col_b = st.columns(2)
            with col_a:
                new_name = st.text_input("Nombre admin")
                new_email = st.text_input("Email", placeholder="ej: nombre@sigasas.com")
            with col_b:
                new_phones = st.text_area("WhatsApp numbers (separados por coma)",
                                          placeholder="573185551234", height=100)
            save_adm = st.form_submit_button("💾 Agregar admin", type="primary")
            if save_adm:
                if not new_name.strip() or not new_email.strip():
                    st.error("Nombre y email son obligatorios.")
                elif any(a["email"].lower() == new_email.strip().lower() for a in admins_data):
                    st.error("Ya existe admin con ese email.")
                else:
                    new_admins = admins_data + [{
                        "name": new_name.strip(), "email": new_email.strip().lower(),
                        "whatsapp_numbers": _phones_to_list(new_phones)}]
                    try:
                        save_registry(_build_full_data(
                            new_admins, specialists_data,
                            [_enrich_with_apikey(c) for c in clients_data]))
                        st.success(f"✅ Admin '{new_name}' agregado.")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"No se pudo guardar: {e}")

    # ---------------- JSON CRUDO ----------------
    with tab_raw:
        st.subheader("Vista cruda del registry")
        st.caption("Útil para auditar y debug. Los api_keys se enmascaran.")
        raw = dict(_load_registry())
        masked_clients = []
        for c in raw.get("clients", []):
            c_copy = dict(c)
            if c_copy.get("api_key"):
                c_copy["api_key"] = c_copy["api_key"][:6] + "***"
            masked_clients.append(c_copy)
        raw["clients"] = masked_clients
        st.json(raw)


__all__ = ["render"]
