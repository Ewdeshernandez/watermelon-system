"""Administración → Suscripciones.

Por cliente: plan, precio pactado por máquina, máquinas, entidad que factura
(SIGA GROUP SAS en COP / Watermelon System EE. UU. en USD), ciclo, contactos
AUTORIZADOS a ver la pestaña Suscripción, correos de cobro y avisos. Emite la
factura del periodo, la marca pagada o la anula. Ajustes globales: datos de
transferencia por entidad y TRM manual.
"""
from __future__ import annotations

from datetime import date

import streamlit as st

from core.billing import cycle as bc
from core.billing import periods, pricing, store
from core.ui_industrial import dot, html_table

_PLAN_KEYS = list(pricing.PLANS)
_ENT_KEYS = list(pricing.ENTITIES)
_ENT_LBL = {"siga_co": "SIGA GROUP SAS · Colombia · COP",
            "wm_us": "Watermelon System · EE. UU. · USD (Mercury)"}


def _d(v, default: date) -> date:
    try:
        return date.fromisoformat(str(v)[:10])
    except Exception:  # noqa: BLE001
        return default


def _who() -> str:
    return str(st.session_state.get("auth_email", "") or "admin")


def render() -> None:
    if not store.available():
        st.warning("Faltan las tablas de facturación en Supabase. Aplica "
                   "`supabase/sql/billing.sql` en el editor SQL y recarga.")
        return

    from core.clients import list_clients, reload_registry
    reload_registry()
    clients = list_clients() or []
    if not clients:
        st.info("No hay clientes en el registro.")
        return
    subs = {s["client_id"]: s for s in store.list_subscriptions()}

    # ---- Resumen ----
    rows = []
    for c in clients:
        s = subs.get(c.id)
        if not s:
            rows.append([c.display_name, "—", "—", "—", "—", f"{dot('off')} Sin suscripción"])
            continue
        try:
            q = pricing.quote(s["plan"], len(s.get("machines") or []), s["billing_entity"],
                              s["cycle"], price_override=s.get("price_per_machine"),
                              usd_unit=s.get("usd_price_per_machine"),
                              trm=None if s.get("usd_price_per_machine") else 1)
            val = (pricing.fmt_money(q["amount"], q["currency"])
                   if (s["billing_entity"] == "siga_co" or s.get("usd_price_per_machine")) else "USD por fijar")
        except Exception:  # noqa: BLE001
            val = "—"
        sd = {"activa": "ok", "pausada": "warn", "cancelada": "off"}.get(s.get("status"), "off")
        rows.append([c.display_name, pricing.PLANS[s["plan"]].name,
                     len(s.get("machines") or []), val, periods.fmt_date(s.get("next_renewal")),
                     f"{dot(sd)} {s.get('status')}"])
    from core.admin import ui
    ui.title("Suscripciones", "Planes, facturas y datos de pago")
    _pk = {"activa": "ok", "pausada": "wa", "cancelada": "ne"}
    ui.group("", [(f"<b>{r[0]}</b> <span class='adm-mut'>{r[1]} · {r[2]} máquinas · "
                   f"renueva {r[4]}</span>",
                   f"<span class='adm-mut'>{r[3]}</span> "
                   + ui.pill(str(r[5]).split('</span>')[-1].strip() or '—',
                             _pk.get(str(r[5]).split('</span>')[-1].strip(), "ne")))
                  for r in rows])

    # ---- Editar ----
    labels = [f"{c.display_name} · {c.id}" for c in clients]
    i = st.selectbox("Cliente", range(len(clients)), format_func=lambda k: labels[k],
                     key="adm_sub_client")
    cl = clients[i]
    s = subs.get(cl.id) or {}
    today = bc._today()
    tag_opts = sorted(set(list(cl.asset_tags or ()) + list(s.get("machines") or [])))

    _edit = st.expander("Editar suscripción" if s else "Crear suscripción", expanded=not s)
    with _edit, st.form(f"adm_sub_form_{cl.id}"):
        c1, c2, c3 = st.columns(3)
        plan = c1.selectbox("Plan", _PLAN_KEYS,
                            index=_PLAN_KEYS.index(s.get("plan", "profesional")),
                            format_func=lambda k: f"{pricing.PLANS[k].name} · "
                                                  f"{pricing.fmt_money(pricing.PLANS[k].price_cop, 'COP')}")
        price = c2.number_input("Precio pactado por máquina (COP, 0 = lista)", min_value=0,
                                value=int(s.get("price_per_machine") or 0), step=1000)
        status = c3.selectbox("Estado", ["activa", "pausada", "cancelada"],
                              index=["activa", "pausada", "cancelada"].index(s.get("status", "activa")))
        machines = st.multiselect("Máquinas facturadas", tag_opts,
                                  default=list(s.get("machines") or tag_opts))
        c4, c5 = st.columns(2)
        ent = c4.radio("Factura", _ENT_KEYS, format_func=lambda k: _ENT_LBL[k],
                       index=_ENT_KEYS.index(s.get("billing_entity", "siga_co")))
        cyc = c5.radio("Ciclo", ["mensual", "anual"],
                       format_func=lambda k: "Mensual" if k == "mensual" else "Anual (12 meses por 11)",
                       index=["mensual", "anual"].index(s.get("cycle", "mensual")))
        c6, c7, c8 = st.columns(3)
        start = c6.date_input("Inicio", value=_d(s.get("start_date"), today))
        _first_next = periods.add_months(today.replace(day=1), 1)
        renew = c7.date_input("Próxima facturación", value=_d(s.get("next_renewal"), _first_next))
        due_days = c8.number_input("Días para pagar", min_value=0, max_value=90,
                                   value=int(s.get("due_days") or 10))
        auth = st.text_area("Contactos AUTORIZADOS a ver la pestaña Suscripción (uno por línea)",
                            value="\n".join(s.get("authorized_emails") or []), height=80)
        bill = st.text_area("Correos de cobro (si se deja vacío, a los autorizados)",
                            value="\n".join(s.get("billing_emails") or []), height=68)
        notify = st.checkbox("Enviar la factura por correo al emitirla (también la del cron)",
                             value=bool(s.get("notify_email")))
        usd_txt = ""
        if s.get("usd_price_per_machine"):
            usd_txt = (f"Precio USD fijado: {pricing.fmt_money(float(s['usd_price_per_machine']), 'USD')}"
                       f" por máquina desde {s.get('usd_locked_at')} (12 meses).")
        if usd_txt:
            st.caption(usd_txt)
        notes = st.text_input("Notas internas", value=s.get("notes") or "")
        saved = st.form_submit_button("Guardar suscripción", type="primary")
    if saved:
        row = {"client_id": cl.id, "plan": plan, "price_per_machine": price or None,
               "machines": machines, "billing_entity": ent, "cycle": cyc, "status": status,
               "start_date": start, "next_renewal": renew, "due_days": int(due_days),
               "authorized_emails": auth, "billing_emails": bill, "notify_email": notify,
               "notes": notes or None}
        if s.get("usd_price_per_machine") and ent == "wm_us" and plan == s.get("plan") \
                and (price or None) == s.get("price_per_machine"):
            row["usd_price_per_machine"] = s["usd_price_per_machine"]
            row["usd_locked_at"] = s.get("usd_locked_at")
        else:
            row["usd_price_per_machine"] = None       # cambió el plan/precio → se vuelve a fijar
            row["usd_locked_at"] = None
        if store.save_subscription(row):
            for k in [k for k in st.session_state.keys() if str(k).startswith("_wm_sub_visible_")]:
                st.session_state.pop(k, None)
            st.success("Suscripción guardada.")
            st.rerun()
        else:
            st.error("No se pudo guardar.")

    if not s:
        return

    # ---- Vista previa del cobro ----
    try:
        trm = None if (s.get("billing_entity") == "siga_co" or s.get("usd_price_per_machine")) else bc.get_trm()
        q = pricing.quote(s["plan"], len(s.get("machines") or []), s["billing_entity"], s["cycle"],
                          price_override=s.get("price_per_machine"),
                          usd_unit=s.get("usd_price_per_machine"), trm=trm)
        st.caption(f"Cobro por periodo: **{pricing.fmt_money(q['amount'], q['currency'])}** "
                   f"(lista {pricing.fmt_money(q['subtotal'], q['currency'])}, ahorro "
                   f"{pricing.fmt_money(q['savings'], q['currency'])})"
                   + (f" · TRM {trm:,.2f}" if trm else ""))
    except Exception as e:  # noqa: BLE001
        st.caption(f"No se pudo calcular el cobro: {e}")

    # ---- Facturas ----
    st.markdown("##### Facturas")
    cA, cB = st.columns([1, 3])
    if cA.button("Emitir factura del periodo", key=f"adm_bill_{cl.id}"):
        nr = _d(s.get("next_renewal"), today)
        res = bc.create_invoice(s, nr, by=_who())
        if res.get("ok") and res.get("created"):
            nxt = periods.next_renewal_after(nr, s.get("cycle") or "mensual")
            fresh = store.get_subscription(cl.id) or s
            store.save_subscription(dict(fresh, next_renewal=nxt.isoformat()))
            st.success(f"Factura {res['invoice']['number']} emitida."
                       + (f" Correo: {res['email']}" if res.get("email") else ""))
            st.rerun()
        elif res.get("ok"):
            st.info(f"Ya existía la factura {res['invoice']['number']} para ese periodo.")
        else:
            st.error(res.get("error"))
    cB.caption(f"Emite la factura que empieza el {s.get('next_renewal')} y avanza la "
               "próxima facturación. El cron lo hace solo cada día de renovación.")

    invs = store.list_invoices(cl.id)
    if invs:
        _IS = {"pagada": "ok", "pendiente": "warn", "vencida": "dang", "anulada": "off"}
        rows = []
        for v in invs:
            stt = periods.effective_status(v["status"], _d(v["due_date"], today), today)
            rows.append([v["number"], f"{v['period_start']} → {v['period_end']}",
                         pricing.fmt_money(float(v["amount"]), v["currency"]), v["due_date"],
                         f"{dot(_IS.get(stt, 'off'))} {stt}",
                         (v.get("paid_method") or "") + (f" · {v['paid_ref']}" if v.get("paid_ref") else "")])
        html_table(["Factura", "Periodo", "Valor", "Vence", "Estado", "Pago"], rows, raw_cols=[4])

        open_ = [v for v in invs if v["status"] == "pendiente"]
        if open_:
            with st.form(f"adm_pay_{cl.id}"):
                k = st.selectbox("Factura", range(len(open_)),
                                 format_func=lambda j: f"{open_[j]['number']} · "
                                 f"{pricing.fmt_money(float(open_[j]['amount']), open_[j]['currency'])}")
                m = st.selectbox("Medio de pago", list(pricing.ENTITIES[open_[0]["billing_entity"]]["methods"])
                                 + ["Otro"])
                ref = st.text_input("Referencia / comprobante")
                b1, b2 = st.columns(2)
                paid = b1.form_submit_button("Marcar pagada", type="primary")
                void = b2.form_submit_button("Anular factura")
            if paid and bc.mark_paid(open_[k]["id"], m, ref, by=_who()):
                st.success("Marcada pagada.")
                st.rerun()
            if void and bc.void_invoice(open_[k]["id"], ref or "sin motivo", by=_who()):
                st.success("Factura anulada.")
                st.rerun()
    else:
        st.caption("Sin facturas todavía.")

    # ---- Ajustes globales ----
    with st.expander("Datos de pago por transferencia y TRM (para todos los clientes)"):
        for ek in _ENT_KEYS:
            cur = bc.transfer_details(ek)
            st.markdown(f"**{_ENT_LBL[ek]}**")
            with st.form(f"adm_tr_{ek}"):
                a1, a2, a3 = st.columns(3)
                v = {
                    "beneficiary": a1.text_input("Beneficiario", cur["beneficiary"]),
                    "bank": a2.text_input("Banco", cur["bank"]),
                    "account_type": a3.text_input("Tipo de cuenta", cur["account_type"]),
                }
                b1, b2, b3 = st.columns(3)
                v["account"] = b1.text_input("Número de cuenta", cur["account"])
                v["routing"] = b2.text_input("Routing / ABA", cur["routing"])
                v["swift"] = b3.text_input("SWIFT", cur["swift"])
                d1, d2 = st.columns(2)
                v["tax_id"] = d1.text_input("NIT / Tax ID", cur["tax_id"])
                v["notes"] = d2.text_input("Nota para el cliente", cur["notes"])
                if st.form_submit_button("Guardar datos"):
                    st.success("Guardado.") if store.set_setting(f"transfer_{ek}", v) else st.error("No se pudo guardar.")
        with st.form("adm_trm"):
            t = st.number_input("TRM manual (0 = TRM oficial del día)", min_value=0.0,
                                value=float(store.get_setting("trm_manual", 0) or 0), step=1.0)
            if st.form_submit_button("Guardar TRM"):
                store.set_setting("trm_manual", t or 0)
                st.success("Guardado.")
        st.caption(f"TRM vigente para fijar precios USD: {bc.get_trm() or '—'}")
