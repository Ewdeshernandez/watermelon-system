"""
pages/24_Suscripcion.py — Suscripción del cliente
=================================================

Plan contratado, factura del periodo, cómo pagar e historial. Solo la ven:
  · admin (con selector de cliente, "vista del cliente")
  · los correos AUTORIZADOS de la suscripción (wm_subscriptions.authorized_emails)
Cualquier otro usuario (incluido el resto del cliente) recibe acceso denegado.

Pago en línea (Wompi para SIGA GROUP SAS / Mercury para Watermelon System EE. UU.)
se enciende cuando existan las cuentas; mientras tanto: datos de transferencia.
"""
from __future__ import annotations

import html

import streamlit as st

from core.auth import get_current_user, render_user_menu, require_login

st.set_page_config(page_title="Watermelon System | Suscripción", layout="wide",
                   initial_sidebar_state="auto")
require_login()
render_user_menu()

from core.ui_theme import apply_watermelon_page_style, page_header  # noqa: E402
from core.ui_industrial import dot, html_table, inject_industrial_css  # noqa: E402

apply_watermelon_page_style()
inject_industrial_css()

from core.billing import periods, pricing, store  # noqa: E402
from core.billing.cycle import _today, transfer_details  # noqa: E402

user = get_current_user() or {}
role = (user.get("role") or "").strip().lower()
email = (user.get("email") or "").strip().lower()
is_admin = bool(user.get("is_admin")) or role == "admin"

page_header("Suscripción", subtitle="Su plan, sus facturas y cómo pagarlas.")

if not store.available():
    st.info("La suscripción todavía no está activa en este sistema.")
    st.stop()

sub = None
if is_admin:
    subs = store.list_subscriptions()
    if not subs:
        st.info("No hay suscripciones creadas. Créalas en Administración → Suscripciones.")
        st.stop()
    ids = [s["client_id"] for s in subs]
    pick = st.selectbox("Vista del cliente (solo admin)", ids, key="sub_view_client")
    sub = next(s for s in subs if s["client_id"] == pick)
else:
    sub = store.subscription_for_viewer(email)
    if not sub:
        st.error("No tiene acceso a esta sección. Si debe gestionar los pagos de su "
                 "empresa, pida a SIGA que lo agregue como contacto autorizado.")
        st.stop()

plan = pricing.PLANS.get(sub.get("plan") or "profesional", pricing.PLANS["profesional"])
entity_key = sub.get("billing_entity") or "siga_co"
entity = pricing.ENTITIES[entity_key]
cycle = sub.get("cycle") or "mensual"
machines = list(sub.get("machines") or [])
today = _today()

try:
    from core.clients import get_client_by_id
    _c = get_client_by_id(sub["client_id"])
    client_name = _c.display_name if _c else sub["client_id"]
except Exception:  # noqa: BLE001
    client_name = sub["client_id"]

quote = None
try:
    quote = pricing.quote(plan.key, len(machines), entity_key, cycle,
                          price_override=sub.get("price_per_machine"),
                          usd_unit=sub.get("usd_price_per_machine"),
                          trm=None if sub.get("usd_price_per_machine") else 1)
    if entity_key == "wm_us" and not sub.get("usd_price_per_machine"):
        quote = None                    # sin precio USD fijado todavía
except Exception:  # noqa: BLE001
    quote = None

invoices = store.list_invoices(sub["client_id"])
for inv in invoices:
    inv["_status"] = periods.effective_status(inv.get("status", ""),
                                              periods.date.fromisoformat(inv["due_date"]), today)
open_inv = next((i for i in invoices if i["_status"] in ("pendiente", "vencida")), None)

_ST = {"activa": ("ok", "Activa"), "pausada": ("warn", "Pausada"), "cancelada": ("off", "Cancelada")}
_IS = {"pagada": ("ok", "Pagada"), "pendiente": ("warn", "Pendiente"),
       "vencida": ("dang", "Vencida"), "anulada": ("off", "Anulada")}
st_dot, st_txt = _ST.get(sub.get("status"), ("off", sub.get("status", "—")))
al_dia = not any(i["_status"] == "vencida" for i in invoices)

st.markdown("""
<style>
.sb-grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:14px;margin:4px 0 14px}
@media (max-width:820px){.sb-grid{grid-template-columns:1fr}}
.sb-card{background:#fff;border:1px solid #e2e8f2;border-radius:12px;padding:16px 18px}
.sb-k{font:600 11px/1.3 'IBM Plex Mono',monospace;letter-spacing:.06em;color:#64748b;text-transform:uppercase}
.sb-t{font:600 18px/1.3 'IBM Plex Sans',sans-serif;color:#0b1f3a;margin-top:6px}
.sb-s{font:400 13px/1.55 'IBM Plex Sans',sans-serif;color:#475569}
.sb-big{font:600 26px/1.2 'IBM Plex Sans',sans-serif;color:#0b1f3a;margin-top:6px}
.sb-li{font:400 13px/1.7 'IBM Plex Sans',sans-serif;color:#0b1f3a}
.sb-li s{color:#94a3b8}
.sb-pill{display:inline-block;font:500 12px/1 'IBM Plex Sans',sans-serif;padding:4px 9px;border-radius:999px}
.sb-due{background:#fff;border:2px solid #e8890c;border-radius:12px;padding:16px 18px;margin-bottom:14px}
.sb-due.vencida{border-color:#dc3545}
.sb-chip{display:inline-block;font:400 12px/1 'IBM Plex Sans',sans-serif;background:#eef2f7;color:#0b1f3a;padding:5px 9px;border-radius:6px;margin:3px 4px 0 0}
</style>""", unsafe_allow_html=True)

_inc = "<br>".join(f"{dot('ok')} {html.escape(x)}" for x in plan.includes)
_exc = "<br>".join(f"{dot('off')} <s>{html.escape(x)}</s>" for x in plan.excludes)
_price = (pricing.fmt_money(quote["amount"], quote["currency"]) + (" / año" if cycle == "anual" else " / mes")
          if quote else "Por confirmar")
_unit = (f"{len(machines)} × {pricing.fmt_money(quote['unit_price'], quote['currency'])} por máquina"
         + (f" · −{quote['volume_discount_pct']:.0f} % por volumen" if quote and quote["volume_discount_pct"] else "")
         + (" · −5 % pago en USD" if quote and quote["entity_discount_pct"] else "")
         + (" · 12 meses por el precio de 11" if cycle == "anual" else "")
         if quote else "")
_nt = pricing.next_tier(len(machines))
_upsell = (f"Con {_nt[0]} máquinas el descuento por volumen sube a {_nt[1] * 100:.0f} %." if _nt else "")
_pill = ('<span class="sb-pill" style="background:#e6f6ec;color:#136c37">Activa · al día</span>'
         if (sub.get("status") == "activa" and al_dia) else
         f'<span class="sb-pill" style="background:#fdecea;color:#a12a2a">{dot(st_dot)} {st_txt}'
         + ("" if al_dia else " · con saldo vencido") + "</span>")

st.markdown(f"""
<div class="sb-grid">
 <div class="sb-card">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
   <span class="sb-k">Su plan · {html.escape(client_name)}</span>{_pill}</div>
  <div class="sb-t">{plan.name} · {len(machines)} máquina{'s' if len(machines) != 1 else ''}</div>
  <div class="sb-s">{html.escape(' · '.join(machines)) or '—'}</div>
  <div class="sb-li" style="margin-top:10px">{_inc}{('<br>' + _exc) if _exc else ''}</div>
 </div>
 <div class="sb-card">
  <div class="sb-k">Próxima renovación</div>
  <div class="sb-big">{periods.fmt_date(sub.get('next_renewal'))}</div>
  <div class="sb-s">{'Anual' if cycle == 'anual' else 'Mensual'} · {_price}</div>
  <div class="sb-s">{html.escape(_unit)}</div>
  <div class="sb-s" style="margin-top:6px">Factura: {entity['name']} ({entity['country']}, {entity['currency']})</div>
  <div class="sb-s" style="margin-top:6px;color:#1f6f43">{html.escape(_upsell)}</div>
 </div>
</div>""", unsafe_allow_html=True)

# ---- Factura abierta + cómo pagar ----
if open_inv:
    p0 = periods.date.fromisoformat(open_inv["period_start"])
    p1 = periods.date.fromisoformat(open_inv["period_end"])
    _per = periods.period_label(p0, p1, cycle)
    _cls = "sb-due vencida" if open_inv["_status"] == "vencida" else "sb-due"
    _sd, _stx = _IS[open_inv["_status"]]
    st.markdown(f"""
<div class="{_cls}">
 <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap">
  <div><div class="sb-k">Factura del periodo</div>
   <div class="sb-t">{html.escape(open_inv['number'])} · {_per}</div>
   <div class="sb-s">Vence el {periods.fmt_date(open_inv['due_date'])}</div></div>
  <div style="text-align:right"><div class="sb-big">{pricing.fmt_money(float(open_inv['amount']), open_inv['currency'])}</div>
   <div class="sb-s">{dot(_sd)} {_stx}</div></div>
 </div>
</div>""", unsafe_allow_html=True)

    online = store.get_setting(f"online_{entity_key}", {}) or {}
    pay_url = (open_inv.get("provider_ref") or "").strip() if str(open_inv.get("provider_ref", "")).startswith("http") else ""
    c1, c2 = st.columns([1, 2])
    with c1:
        if pay_url and online.get("enabled"):
            st.link_button("Pagar en Mercury ↗" if entity_key == "wm_us" else "Pagar con PSE, Bre-B o tarjeta ↗",
                           pay_url, type="primary", use_container_width=True)
        else:
            _has_td = any(transfer_details(entity_key).values())
            st.markdown('<div class="sb-s"><b>Pago en línea:</b> se activa muy pronto. '
                        + ('Mientras tanto puede pagar por transferencia con estos datos.'
                           if _has_td else
                           'Mientras tanto SIGA le envía los datos de transferencia con la factura.')
                        + '</div>', unsafe_allow_html=True)
    with c2:
        _m = "".join(f'<span class="sb-chip">{html.escape(m)}</span>' for m in entity["methods"])
        st.markdown(f'<div class="sb-s">Medios de pago {entity["name"]}:</div>{_m}',
                    unsafe_allow_html=True)
    td = transfer_details(entity_key)
    if any(td.values()):
        rows = [[k, v] for k, v in (
            ("Beneficiario", td["beneficiary"]), ("Banco", td["bank"]),
            ("Tipo de cuenta", td["account_type"]), ("Cuenta", td["account"]),
            ("Routing / ABA", td["routing"]), ("SWIFT", td["swift"]),
            ("NIT / Tax ID", td["tax_id"]), ("Referencia", open_inv["number"]),
            ("Nota", td["notes"])) if v]
        html_table(["Transferencia", ""], rows)
elif not invoices:
    # Cliente nuevo: todavía no hay facturas → bienvenida con qué va a pasar.
    _first_amt = (pricing.fmt_money(quote["amount"], quote["currency"]) if quote else "por confirmar")
    _contact = str(store.get_setting("billing_contact", "") or "").strip()
    _m = [m[0].lower() + m[1:] if m.split()[0] in ("Tarjeta", "Transferencia") else m
          for m in entity["methods"]]
    _ways = ", ".join(_m[:-1]) + " o " + _m[-1]
    _notice = ("le llega aviso por correo" if sub.get("notify_email")
               else "la verá aquí en Suscripción")
    st.markdown(f"""
<div class="sb-card" style="margin-bottom:14px;border:2px solid #1f9d55">
 <div class="sb-k">Bienvenido a Watermelon System</div>
 <div class="sb-t">Su primera factura: {periods.fmt_date(sub.get('next_renewal'))} · {_first_amt}</div>
 <div class="sb-li" style="margin-top:8px">
  {dot('ok')} Ese día la emitimos y {_notice}; tendrá {int(sub.get('due_days') or 10)} días para pagarla.<br>
  {dot('ok')} Podrá pagar aquí mismo con {html.escape(_ways)}.<br>
  {dot('ok')} Mientras tanto su monitoreo ya está activo en Live Monitoring.<br>
  {dot('info')} ¿Dudas sobre su plan o la factura? Escríbanos{(' a <b>' + html.escape(_contact) + '</b>') if _contact else ' a su contacto en SIGA'}.
 </div>
</div>""", unsafe_allow_html=True)
else:
    st.markdown(f'<div class="sb-card" style="margin-bottom:14px"><span class="sb-s">'
                f'{dot("ok")} No tiene facturas pendientes.</span></div>', unsafe_allow_html=True)

# ---- Para sus máquinas: ofertas con base en sus datos ----
from core.billing import upsell  # noqa: E402

_ev = upsell.evidence_for(machines)
_oi = upsell.integral_offer(sub, upsell.balance_price(), _ev)
_om = upsell.add_machine_offer(sub)
_oa = upsell.annual_offer(sub)
if _oi or _om or _oa:
    st.markdown('<div class="sb-k" style="margin:6px 0 8px">Para sus máquinas</div>',
                unsafe_allow_html=True)
    _who = email or "admin"


    def _ask(kind: str, detail: str) -> None:
        if is_admin:
            st.info("Vista de admin: la solicitud no se registra.")
            return
        if upsell.record_request(sub, kind, _who, detail):
            st.success("Listo. SIGA recibió su solicitud y le contactará para confirmarla. "
                       "No se cobra nada hasta que usted lo apruebe.")
        else:
            st.error("No se pudo enviar la solicitud. Intente de nuevo.")

    if _oi:
        _fm = lambda v: pricing.fmt_money(v, _oi["currency"])  # noqa: E731
        _why = "".join(f"{dot('warn')} {html.escape(r)}<br>" for r in _oi["reasons"])
        _be = _oi["breakeven"]
        _pay = (f"Un balanceo de turbomaquinaria contratado aparte cuesta {_fm(_oi['balance_price'])}. "
                + ("Con un solo balanceo al año, Integral ya le sale más económico."
                   if _be == 1 else
                   f"Con {_be} balanceos al año Integral se paga solo, y además incluye el análisis "
                   "modal y la confiabilidad API 670."))
        st.markdown(f"""
<div class="sb-card" style="border:2px solid #e8890c;margin-bottom:10px">
 <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
  <span class="sb-t" style="margin:0">{html.escape(_oi['title'])}</span>
  {'<span class="sb-pill" style="background:#fff1dc;color:#8a4b00">Según sus datos</span>' if _oi['data_driven'] else ''}
 </div>
 <div class="sb-li" style="margin-top:8px">{_why}</div>
 <div class="sb-s" style="margin-top:6px">{html.escape(_pay)}</div>
 <div class="sb-s" style="margin-top:6px"><b>+ {_fm(_oi['diff_month'])} al mes</b> por sus
 {len(machines)} máquina{'s' if len(machines) != 1 else ''} ({_fm(_oi['diff_year'])} al año).</div>
</div>""", unsafe_allow_html=True)
        if st.button("Quiero pasar a Integral", key="up_integral"):
            _ask("integral", f"Diferencia {_fm(_oi['diff_month'])}/mes · motivos: "
                             + " | ".join(_oi["reasons"]))
    _c1, _c2 = st.columns(2)
    if _om:
        with _c1:
            _fm = lambda v: pricing.fmt_money(v, _om["currency"])  # noqa: E731
            _nt = _om["next_tier"]
            _tier = (f" Con {_nt[0]} máquinas, −{_nt[1] * 100:.0f} % en todas." if _nt else "")
            st.markdown(f'<div class="sb-card"><div class="sb-t" style="margin:0">Agregue una máquina</div>'
                        f'<div class="sb-s" style="margin-top:4px">Su máquina n.º {_om["n"] + 1}: '
                        f'+ {_fm(_om["extra_month"])} al mes.{_tier} Turbinas, generadores, compresores, '
                        f'bombas multietapa o ventiladores: toda su turbomaquinaria.</div></div>',
                        unsafe_allow_html=True)
            if st.button("Agregar máquina", key="up_machine"):
                _ask("maquina", f"Máquina n.º {_om['n'] + 1}: + {_fm(_om['extra_month'])}/mes")
    if _oa:
        with _c2:
            _fm = lambda v: pricing.fmt_money(v, _oa["currency"])  # noqa: E731
            st.markdown(f'<div class="sb-card"><div class="sb-t" style="margin:0">Pague el año completo</div>'
                        f'<div class="sb-s" style="margin-top:4px">12 meses por el precio de 11: '
                        f'ahorra {_fm(_oa["saving_year"])} al año.</div></div>',
                        unsafe_allow_html=True)
            if st.button("Cambiar a anual", key="up_annual"):
                _ask("anual", f"Anual {_fm(_oa['annual_amount'])} · ahorro {_fm(_oa['saving_year'])}")

# ---- Historial ----
st.markdown('<div class="sb-k" style="margin:6px 0 8px">Historial de facturas</div>',
            unsafe_allow_html=True)
if invoices:
    rows = []
    for i in invoices:
        _sd, _stx = _IS.get(i["_status"], ("off", i["_status"]))
        p0 = periods.date.fromisoformat(i["period_start"])
        p1 = periods.date.fromisoformat(i["period_end"])
        paid = (f"{_stx} {periods.fmt_date(i.get('paid_at'))}"
                + (f" · {i['paid_method']}" if i.get("paid_method") else "")) if i["_status"] == "pagada" else _stx
        rows.append([i["number"], periods.period_label(p0, p1, cycle),
                     pricing.fmt_money(float(i["amount"]), i["currency"]),
                     periods.fmt_date(i["due_date"]), f"{dot(_sd)} {html.escape(paid)}"])
    html_table(["Factura", "Periodo", "Valor", "Vence", "Estado"], rows, raw_cols=[4])
else:
    st.caption("Todavía no hay facturas.")
