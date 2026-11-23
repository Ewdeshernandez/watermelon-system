"""Ofertas para el cliente actual (pestaña Suscripción) y solicitudes a SIGA.

Las ofertas se apoyan en los datos del propio cliente:
  · Pasar a Integral: recomendaciones abiertas de balanceo / análisis modal /
    sistema de protección (API 670) en sus máquinas y cruces de alarma de los
    últimos 30 días. Compara la diferencia anual contra el precio de un
    balanceo de turbomaquinaria contratado aparte (ajuste en Administración).
  · Agregar una máquina: costo de la siguiente y el escalón de descuento.
  · Pago anual: 12 meses por 11.

Nada se cobra solo: cada botón registra una SOLICITUD y avisa a SIGA.
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from core.billing import pricing, store

log = logging.getLogger(__name__)

BALANCE_PRICE_KEY = "balance_service_price_cop"
DEFAULT_BALANCE_PRICE = 50_000_000
REQUESTS_KEY = "sales_requests"

_KW = {
    "balanceo": re.compile(r"balance", re.I),
    "modal": re.compile(r"modal|resonan|frecuencia natural|velocidad cr[ií]tica", re.I),
    "api670": re.compile(r"api\s*670|sistema de protecci", re.I),
}


# ------------------------------------------------------------------ puro

def integral_offer(sub: Dict[str, Any], balance_price: float,
                   evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if sub.get("plan") == "integral":
        return None
    n = len(sub.get("machines") or [])
    if not n:
        return None
    if (sub.get("billing_entity") or "siga_co") != "siga_co":
        return None                       # USD: la oferta se arma con la TRM fijada (fase 2)
    cur = pricing.quote(sub["plan"], n, "siga_co", "mensual",
                        price_override=sub.get("price_per_machine"))
    up = pricing.quote("integral", n, "siga_co", "mensual")
    diff_m = max(0.0, up["amount"] - cur["amount"])
    diff_y = diff_m * 12
    breakeven = max(1, math.ceil(diff_y / balance_price)) if balance_price > 0 else None
    reasons: List[str] = []
    for r in sorted(evidence.get("recs", []), key=lambda r: not r.get("balance")):
        reasons.append(f"Su reporte recomienda: “{r['text']}” (abierta desde {r['since']}).")
    if evidence.get("alarm_entries"):
        worst = max(evidence["alarm_entries"].items(), key=lambda kv: kv[1])
        reasons.append(f"{worst[0]} entró en alerta {worst[1]} veces en los últimos 30 días.")
    return {
        "kind": "integral",
        "title": "Pase a Integral: balanceo, análisis modal y API 670 incluidos",
        "reasons": reasons[:3],
        "diff_month": diff_m, "diff_year": diff_y,
        "balance_price": balance_price, "breakeven": breakeven,
        "currency": cur["currency"], "data_driven": bool(reasons),
    }


def add_machine_offer(sub: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    n = len(sub.get("machines") or [])
    if sub.get("billing_entity", "siga_co") != "siga_co" or not n:
        return None
    a = pricing.quote(sub["plan"], n, "siga_co", "mensual", price_override=sub.get("price_per_machine"))
    b = pricing.quote(sub["plan"], n + 1, "siga_co", "mensual", price_override=sub.get("price_per_machine"))
    nt = pricing.next_tier(n)
    return {"kind": "maquina", "extra_month": b["amount"] - a["amount"],
            "next_tier": nt, "currency": a["currency"], "n": n}


def annual_offer(sub: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if (sub.get("cycle") or "mensual") == "anual" or sub.get("billing_entity", "siga_co") != "siga_co":
        return None
    n = len(sub.get("machines") or [])
    if not n:
        return None
    m = pricing.quote(sub["plan"], n, "siga_co", "mensual", price_override=sub.get("price_per_machine"))
    y = pricing.quote(sub["plan"], n, "siga_co", "anual", price_override=sub.get("price_per_machine"))
    return {"kind": "anual", "saving_year": m["amount"] * 12 - y["amount"],
            "currency": m["currency"], "annual_amount": y["amount"]}


# ------------------------------------------------------------------ datos

def balance_price() -> float:
    try:
        v = float(store.get_setting(BALANCE_PRICE_KEY, DEFAULT_BALANCE_PRICE) or 0)
        return v if v > 0 else DEFAULT_BALANCE_PRICE
    except (TypeError, ValueError):
        return DEFAULT_BALANCE_PRICE


def evidence_for(machines: List[str]) -> Dict[str, Any]:
    """Recomendaciones abiertas que Integral cubre + entradas a alerta (30 d)."""
    out: Dict[str, Any] = {"recs": [], "alarm_entries": {}}
    try:
        from core.instance_state import list_instances
        by_tag = {str(i.get("tag", "")).upper(): i.get("instance_id")
                  for i in (list_instances() or []) if isinstance(i, dict)}
    except Exception:  # noqa: BLE001
        by_tag = {}
    from core.billing.periods import fmt_date
    for tag in machines or []:
        iid = by_tag.get(str(tag).upper())
        if not iid:
            continue
        try:
            from core.briefing_recommendations import list_recommendations
            for r in list_recommendations(iid) or []:
                txt = str(r.get("text") or "")
                if any(p.search(txt) for p in _KW.values()):
                    short = re.split(r"\s+según|\s*\(", txt)[0].strip().rstrip(".")
                    if len(short) > 110:
                        short = short[:110].rsplit(" ", 1)[0].rstrip(",;:") + "…"
                    out["recs"].append({"tag": tag, "text": short,
                                        "since": fmt_date(r.get("started_at")),
                                        "balance": bool(_KW["balanceo"].search(txt))})
        except Exception:  # noqa: BLE001
            pass
        aux = set()
        try:                                    # auxiliares (bomba, arrancador): Integral no los cubre
            from core.report_extras import aux_groups_for
            aux = {k.upper() for k in (aux_groups_for(iid)[0] or {})}
        except Exception:  # noqa: BLE001
            pass
        try:
            from core.live_readings import _get_supabase_client
            cl = _get_supabase_client()
            since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            res = (cl.table("severity_events").select("sensor_label,from_status,to_status")
                   .eq("instance_id", iid).gte("crossed_at", since).limit(1000).execute())
            rank = {"Alarma": 1, "Danger": 2}
            for e in getattr(res, "data", None) or []:
                if rank.get(e.get("to_status"), 0) > rank.get(e.get("from_status"), 0):
                    k = str(e.get("sensor_label") or "").replace("_", "")
                    if k.upper() in aux:
                        continue
                    out["alarm_entries"][k] = out["alarm_entries"].get(k, 0) + 1
        except Exception:  # noqa: BLE001
            pass
    return out


# ------------------------------------------------------------------ solicitudes

_KIND_TXT = {"integral": "pasar a plan Integral", "maquina": "agregar una máquina",
             "anual": "cambiar a pago anual", "cotizacion": "una cotización desde la web"}


def record_request(sub: Dict[str, Any], kind: str, by: str, detail: str = "") -> bool:
    """Guarda la solicitud (Administración la muestra) y avisa a SIGA por correo."""
    reqs = list(store.get_setting(REQUESTS_KEY, []) or [])
    item = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "client_id": sub.get("client_id"), "kind": kind, "by": by,
            "detail": detail, "status": "nueva"}
    reqs.insert(0, item)
    ok = store.set_setting(REQUESTS_KEY, reqs[:200])
    to = str(store.get_setting("billing_contact", "") or "").strip()
    if not to:
        try:
            from core.clients import list_admins
            to = next((a.get("email") for a in list_admins() if a.get("email")), "")
        except Exception:  # noqa: BLE001
            to = ""
    if to:
        try:
            from core.email_sender import send_email
            what = _KIND_TXT.get(kind, kind)
            send_email(to=to, subject=f"Solicitud de cliente: {what} · {sub.get('client_id')}",
                       body_text=(f"{by} ({sub.get('client_id')}) pidió {what} desde la pestaña "
                                  f"Suscripción.\n\n{detail}\n\nGestiónalo en Administración → "
                                  "Suscripciones."))
        except Exception as e:  # noqa: BLE001
            log.warning("aviso de solicitud no enviado: %s", e)
    return ok


def open_requests() -> List[Dict[str, Any]]:
    return [r for r in (store.get_setting(REQUESTS_KEY, []) or []) if r.get("status") == "nueva"]


def close_request(at: str) -> bool:
    reqs = list(store.get_setting(REQUESTS_KEY, []) or [])
    for r in reqs:
        if r.get("at") == at:
            r["status"] = "atendida"
    return store.set_setting(REQUESTS_KEY, reqs)


# ------------------------------------------------------------------ prospectos (web)

def notify_new_leads() -> dict:
    """Cron: por cada cotización nueva de la web → correo al prospecto con su PDF
    y aviso a SIGA. Marca notified=True (no reenvía)."""
    from datetime import date as _date
    reqs = list(store.get_setting(REQUESTS_KEY, []) or [])
    pend = [r for r in reqs if r.get("kind") == "cotizacion" and not r.get("notified")
            and r.get("lead")]
    if not pend:
        return {"sent": 0}
    from core.billing.quote_pdf import build_quote_pdf
    from core.email_sender import send_email
    to_siga = str(store.get_setting("billing_contact", "") or "").strip()
    if not to_siga:
        try:
            from core.clients import list_admins
            to_siga = next((a.get("email") for a in list_admins() if a.get("email")), "")
        except Exception:  # noqa: BLE001
            to_siga = ""
    sent = 0
    for r in pend:
        lead = r["lead"]
        try:
            day = _date.fromisoformat(str(r.get("at", ""))[:10])
        except ValueError:
            day = _date.today()
        pdf = build_quote_pdf(lead, r.get("number", "COT"), today=day)
        att = [(f"{r.get('number', 'cotizacion')}.pdf", pdf, "application/pdf")]
        plan = pricing.PLANS.get(lead.get("plan"), pricing.PLANS["profesional"]).name
        try:
            en = (lead.get("lang") or "es") == "en"
            who = (lead.get("name") or "").split(" ")[0]
            pilot = bool(lead.get("pilot"))
            if en:
                subj = f"Your Watermelon System quote · {r.get('number')}"
                hi = f"Hi {who}," if who else "Hello,"
                body = (f"Thank you for your interest in Watermelon System. Attached is your quote for the "
                        f"{plan} plan for {lead.get('machines')} machine(s)"
                        + (", starting with the free 30-day pilot" if pilot else "") + ".")
                nxt = ("A specialist will contact you shortly to review your machines and sensors. "
                       "You can simply reply to this email with any question.")
                sign = "Ewdes Hernández · Machinery Diagnostics Champion"
            else:
                subj = f"Su cotización Watermelon System · {r.get('number')}"
                hi = f"Hola {who}," if who else "Hola,"
                body = (f"Gracias por su interés en Watermelon System. Le adjuntamos la cotización del plan "
                        f"{plan} para {lead.get('machines')} máquina(s)"
                        + (", empezando con el piloto de 30 días sin costo" if pilot else "") + ".")
                nxt = ("Un especialista le contactará en breve para revisar sus máquinas y sensores. "
                       "Puede responder directamente a este correo con cualquier pregunta.")
                sign = "Ewdes Hernández · Machinery Diagnostics Champion"
            text = f"{hi}\n\n{body}\n\n{nxt}\n\n{sign}\nWatermelon System · watermelonsystem.com"
            html = f"""<div style="font-family:Arial,Helvetica,sans-serif;max-width:600px;margin:auto;color:#0b1f3a">
<div style="background:#0f2748;color:#fff;padding:18px 22px;border-radius:10px 10px 0 0">
<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#2f8f4e;border:2px solid #9fd4a9;vertical-align:-1px"></span>
<b style="font-size:16px;margin-left:6px">Watermelon System</b></div>
<div style="border:1px solid #e3e9f1;border-top:0;border-radius:0 0 10px 10px;padding:22px;font-size:15px;line-height:1.6">
<p>{hi}</p><p>{body}</p><p>{nxt}</p>
<p style="margin-top:22px"><a href="https://watermelonsystem.com/{'en/' if en else ''}demo" style="background:#2f8f4e;color:#fff;padding:10px 16px;border-radius:8px;text-decoration:none">{'See the live demo' if en else 'Ver la demo en vivo'}</a></p>
<p style="margin-top:26px">{sign}<br><span style="color:#5b6b80">Watermelon System · SIGA GROUP SAS</span></p>
</div></div>"""
            send_email(to=lead["email"], subject=subj, body_text=text, body_html=html,
                       attachments=att, reply_to=to_siga,
                       from_name="Ewdes Hernández · Watermelon System")
            if to_siga:
                send_email(to=to_siga,
                           subject=f"Nuevo prospecto web: {lead.get('company') or lead.get('email')}",
                           body_text=(f"{r.get('detail')}\nCorreo: {lead.get('email')}\n"
                                      f"Cotización {r.get('number')} adjunta. Responde este correo "
                                      "y le llega directo al prospecto."),
                           attachments=att, reply_to=lead["email"])
            r["notified"] = True
            sent += 1
        except Exception as e:  # noqa: BLE001
            log.warning("prospecto %s no notificado: %s", r.get("number"), e)
    store.set_setting(REQUESTS_KEY, reqs)
    return {"sent": sent}
