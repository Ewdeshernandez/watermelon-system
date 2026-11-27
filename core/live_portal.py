"""Live Monitoring — capa de portal.

Dos audiencias sobre los MISMOS datos en vivo:

* Vista CLIENTE (rol client, o "View as client" del personal SIGA): veredicto
  en lenguaje claro, "¿Qué necesita hoy?", resumen de la semana, puntos de
  medición agrupados por componente y contacto con su especialista. ES / EN.
* Centro SIGA (admin / especialista): flota de todos los clientes, cola
  "Needs attention", nota semanal para el cliente y actividad de clientes.

Las funciones puras (sin Streamlit) están arriba y tienen tests en
tests/test_live_portal.py. El render usa el sistema de diseño de la web
(azul acero, IBM Plex, puntos de color — nunca emojis).
"""
from __future__ import annotations

import html as _html
import re
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SIGA_WHATSAPP = "573008888883"
LIVE_APP_URL = "https://watermelon-live.vercel.app"
CLIENT_NOTE_KEY = "client_note"
CHATTER_MIN = 5            # entradas a alarma en la ventana → "entra y sale"
NEAR_LIMIT_PCT = 85        # % de la alarma a partir del cual se marca "cerca del límite"

_RANK = {"Normal": 0, "Sin Norma": 0, "No Data": 0, "Alarma": 1, "Danger": 2}

COMP_NAMES: Dict[str, Tuple[str, str]] = {
    "turbine": ("Turbina", "Turbine"),
    "gearbox": ("Caja", "Gearbox"),
    "generator": ("Generador", "Generator"),
    "motor": ("Motor", "Motor"),
    "compressor": ("Compresor", "Compressor"),
    "pump": ("Bomba", "Pump"),
    "fan": ("Ventilador", "Fan"),
    "aux": ("Auxiliares", "Auxiliaries"),
    "other": ("Otros puntos", "Other points"),
}
_COMP_ORDER = ["turbine", "motor", "gearbox", "compressor", "pump", "fan", "generator", "other", "aux"]
_AUX_WORDS = re.compile(r"\b(lube|lubric\w*|bomba|starter|arrancador|aux\w*)\b", re.I)
_COMP_WORDS = [
    ("turbine", re.compile(r"turbin", re.I)),
    ("gearbox", re.compile(r"gear|caja|reductor", re.I)),
    ("generator", re.compile(r"generad|generator", re.I)),
    ("motor", re.compile(r"\bmotor\b", re.I)),
    ("compressor", re.compile(r"compres", re.I)),
    ("fan", re.compile(r"\bfan\b|ventilad", re.I)),
    ("pump", re.compile(r"\bpump\b|\bbomba\b", re.I)),
]
_BEARING_COMP = {1: "turbine", 2: "turbine", 3: "gearbox", 4: "gearbox", 5: "generator", 6: "generator"}


def _L(pair: Tuple[str, str], lang: str) -> str:
    return pair[1] if lang == "en" else pair[0]


def _esc(s: Any) -> str:
    return _html.escape(str(s if s is not None else ""))


def _num(v: Any, dec: int, lang: str) -> str:
    try:
        txt = f"{float(v):,.{dec}f}"
    except (TypeError, ValueError):
        return "—"
    return txt if lang == "en" else txt.replace(",", " ").replace(".", ",")


# ---------------------------------------------------------------------------
# Puntos de medición
# ---------------------------------------------------------------------------

def tag_of(sensor_label: Any) -> str:
    """'1Y_D' → '1YD' (display sin guion bajo)."""
    return re.sub(r"[\s_]+", "", str(sensor_label or "")).upper()


def bearing_no(tag: str) -> int:
    m = re.match(r"(\d+)", tag or "")
    return int(m.group(1)) if m else 99


def axis_rank(tag: str) -> int:
    """Y antes que X (convención Bently: Y = CH1)."""
    m = re.match(r"\d*([A-Z])", tag or "")
    return {"Y": 0, "X": 1}.get(m.group(1) if m else "", 2)


def aux_kind(plane_label: str) -> Optional[Tuple[str, str]]:
    pl = str(plane_label or "")
    if re.search(r"starter|arrancador", pl, re.I):
        return ("Arrancador", "Starter")
    if re.search(r"lube|lubric|bomba", pl, re.I):
        return ("Bomba de lubricación", "Lube oil pump")
    return None


def component_of(tag: str, plane_label: str = "") -> str:
    pl = str(plane_label or "")
    if _AUX_WORDS.search(pl):
        return "aux"
    for comp, rx in _COMP_WORDS:
        if rx.search(pl):
            return comp
    return _BEARING_COMP.get(bearing_no(tag), "other")


def pct_of_alarm(value: Any, alarm: Any) -> Optional[int]:
    try:
        v, a = float(value), float(alarm)
    except (TypeError, ValueError):
        return None
    if a <= 0:
        return None
    return int(round(abs(v) / a * 100))


def point_groups(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Agrupa los puntos (rendered_rows) por componente, Y antes que X."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows or []:
        tag = tag_of(r.get("sensor_label"))
        if not tag:
            continue
        comp = component_of(tag, r.get("plane_label", ""))
        pct = pct_of_alarm(r.get("value"), r.get("alarm_used"))
        groups.setdefault(comp, []).append({
            "tag": tag, "value": r.get("value"), "unit": str(r.get("unit") or ""),
            "alarm": r.get("alarm_used"), "danger": r.get("danger_used"),
            "pct": pct, "status": r.get("status", "Normal"),
            "plane": str(r.get("plane_label") or ""),
        })
    out = []
    for comp in sorted(groups, key=lambda c: _COMP_ORDER.index(c) if c in _COMP_ORDER else 99):
        pts = sorted(groups[comp], key=lambda p: (bearing_no(p["tag"]), axis_rank(p["tag"]), p["tag"]))
        out.append({"key": comp, "points": pts})
    return out


def friendly_point(tag: str, plane_label: str, lang: str) -> str:
    """Nombre que entiende el cliente: 'Cojinete 1 de turbina', 'Bomba de lubricación'."""
    ak = aux_kind(plane_label)
    if ak:
        return _L(ak, lang)
    comp = component_of(tag, plane_label)
    n = bearing_no(tag)
    cname = _L(COMP_NAMES.get(comp, COMP_NAMES["other"]), lang)
    if n == 99:
        return cname
    if lang == "en":
        return f"{cname} bearing {n}"
    return f"Cojinete {n} de {cname.lower()}"


def hottest_point(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Punto más cerca de su alarma (para el veredicto y la flota)."""
    best = None
    for r in rows or []:
        p = pct_of_alarm(r.get("value"), r.get("alarm_used"))
        if p is None:
            continue
        if best is None or p > best["pct"]:
            best = {"tag": tag_of(r.get("sensor_label")), "pct": p, "value": r.get("value"),
                    "unit": r.get("unit") or "", "plane": r.get("plane_label", "")}
    return best


# ---------------------------------------------------------------------------
# Eventos → "Esta semana"
# ---------------------------------------------------------------------------

def _parse_ts(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        s = str(v).replace("Z", "+00:00")
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def summarize_events(events: Sequence[Dict[str, Any]], now: Optional[datetime] = None,
                     days: int = 7) -> List[Dict[str, Any]]:
    """Agrupa los cruces de umbral por punto en la ventana: veces que ENTRÓ a
    alarma/peligro, máximo, estado actual y si 'entra y sale' (chattering)."""
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    out: Dict[str, Dict[str, Any]] = {}
    evs = sorted(events or [], key=lambda e: str(e.get("crossed_at", "")), reverse=True)
    for e in evs:
        tag = tag_of(e.get("sensor_label"))
        if not tag:
            continue
        ts = _parse_ts(e.get("crossed_at"))
        g = out.get(tag)
        if g is None:
            g = out[tag] = {"tag": tag, "n_up": 0, "max_value": None, "unit": e.get("unit") or "",
                            "alarm": e.get("alarm"), "danger": e.get("danger"), "worst": "Normal",
                            "current": e.get("to_status") or "Normal", "last_at": ts}
        if ts is None or ts < since:
            continue
        frm, to = e.get("from_status") or "Normal", e.get("to_status") or "Normal"
        if _RANK.get(to, 0) > _RANK.get(frm, 0):
            g["n_up"] += 1
            if _RANK.get(to, 0) > _RANK.get(g["worst"], 0):
                g["worst"] = to
            try:
                v = float(e.get("value"))
                if g["max_value"] is None or v > g["max_value"]:
                    g["max_value"] = v
            except (TypeError, ValueError):
                pass
    res = [g for g in out.values() if g["n_up"] > 0 or _RANK.get(g["current"], 0) > 0]
    for g in res:
        g["active"] = _RANK.get(g["current"], 0) > 0
        g["chattering"] = g["n_up"] >= CHATTER_MIN
    res.sort(key=lambda g: (-int(g["active"]), -_RANK.get(g["worst"], 0), -g["n_up"]))
    return res


def week_lines(ev_groups: Sequence[Dict[str, Any]], groups: Sequence[Dict[str, Any]],
               note: Optional[Dict[str, Any]], lang: str, max_lines: int = 4,
               days: int = 7) -> List[Dict[str, str]]:
    """Frases en lenguaje claro para la tarjeta 'Esta semana'.
    kind ∈ note | dang | warn | info | ok. `html` ya viene escapado."""
    lines: List[Dict[str, str]] = []
    if note and str(note.get("text", "")).strip():
        lead = "Su especialista:" if lang != "en" else "Your specialist:"
        lines.append({"kind": "note", "html": f"<b>{lead}</b> {_esc(note['text']).strip()}"})
    planes = {p["tag"]: p["plane"] for g in groups for p in g["points"]}
    with_events = set()
    for g in list(ev_groups)[:max_lines]:
        tag = g["tag"]
        name = friendly_point(tag, planes.get(tag, ""), lang)
        lvl_es = "peligro" if g["worst"] == "Danger" else "alarma"
        lvl_en = "danger" if g["worst"] == "Danger" else "alarm"
        n = g["n_up"]
        mx = _num(g["max_value"], 2, lang) if g["max_value"] is not None else "—"
        lim_v = g["danger"] if g["worst"] == "Danger" else g["alarm"]
        lim = _num(lim_v, 1, lang) if lim_v not in (None, "") else "—"
        unit = _esc(g["unit"])
        if lang == "en":
            times = "once" if n == 1 else f"{n} times"
            txt = (f"<b>{_esc(name)} ({tag}):</b> went into {lvl_en} {times} in {days} days"
                   + (", briefly each time" if g["chattering"] else "")
                   + f". Max {mx} {unit}, limit {lim}.")
            txt += f" In {lvl_en} now." if g["active"] else " Normal today."
        else:
            times = "una vez" if n == 1 else f"{n} veces"
            txt = (f"<b>{_esc(name)} ({tag}):</b> entró a {lvl_es} {times} en {days} días"
                   + (", por momentos" if g["chattering"] else "")
                   + f". Máximo {mx} {unit}, límite {lim}.")
            txt += f" Ahora está en {lvl_es}." if g["active"] else " Hoy está normal."
        if g["active"] and g["worst"] == "Danger":
            kind = "dang"
        elif g["active"] or g["chattering"]:
            kind = "warn"
        else:
            kind = "info"
        lines.append({"kind": kind, "html": txt})
        with_events.add(component_of(tag, planes.get(tag, "")))
    stable = [g["key"] for g in groups if g["key"] not in ("aux", "other") and g["key"] not in with_events]
    if not ev_groups:
        lines.append({"kind": "ok", "html": (
            f"<b>No alarm crossings in the last {days} days.</b> Stable operation."
            if lang == "en" else
            f"<b>Sin cruces de alarma en los últimos {days} días.</b> Operación estable.")})
    elif stable:
        names = [_L(COMP_NAMES[k], lang) for k in stable]
        if len(names) > 1:
            joined = ", ".join(names[:-1]) + (" and " if lang == "en" else " y ") + names[-1]
        else:
            joined = names[0]
        lines.append({"kind": "ok", "html": (
            f"<b>{_esc(joined)} stable.</b> No alarm crossings in {days} days."
            if lang == "en" else
            f"<b>{_esc(joined)} estable{'s' if len(names) > 1 else ''}.</b> Sin cruces de alarma en {days} días.")})
    return lines


# ---------------------------------------------------------------------------
# Veredicto, salud, próximo reporte
# ---------------------------------------------------------------------------

def verdict(tag: str, summary: Optional[Dict[str, int]], *, no_data: bool, stale: bool,
            stopped: bool, n_points: int, lang: str) -> Dict[str, str]:
    s = summary or {}
    n_d, n_a = int(s.get("Danger", 0) or 0), int(s.get("Alarma", 0) or 0)
    t = _esc(tag)
    if no_data or stale:
        return {"tone": "off",
                "title": f"No recent data from {t}" if lang == "en" else f"Sin datos recientes de {t}",
                "sub": ("We are checking the link with the plant. The last values received are shown below."
                        if lang == "en" else
                        "Estamos revisando la conexión con la planta. Abajo ve los últimos valores recibidos.")}
    if stopped:
        return {"tone": "off",
                "title": f"{t} is stopped" if lang == "en" else f"{t} está detenida",
                "sub": ("Sensors are online; the machine is not turning."
                        if lang == "en" else "Los sensores están en línea; la máquina no está girando.")}
    if n_d:
        pts = (f"{n_d} point{'s' if n_d > 1 else ''}" if lang == "en"
               else f"{n_d} punto{'s' if n_d > 1 else ''}")
        return {"tone": "dang",
                "title": f"{t}: {pts} in danger" if lang == "en" else f"{t}: {pts} en peligro",
                "sub": ("SIGA receives this alarm automatically and will contact you about the next step."
                        if lang == "en" else
                        "SIGA recibe esta alarma automáticamente y le contactará con el siguiente paso.")}
    if n_a:
        pts = (f"{n_a} point{'s' if n_a > 1 else ''}" if lang == "en"
               else f"{n_a} punto{'s' if n_a > 1 else ''}")
        return {"tone": "warn",
                "title": f"{t}: {pts} in alarm" if lang == "en" else f"{t}: {pts} en alarma",
                "sub": ("The machine keeps running. SIGA receives these alarms automatically and "
                        "will contact you if action is needed."
                        if lang == "en" else
                        "La máquina sigue operando. SIGA recibe estas alarmas automáticamente y le "
                        "contactará si se requiere una acción.")}
    return {"tone": "ok",
            "title": f"{t} is running normally" if lang == "en" else f"{t} opera normal",
            "sub": (f"All {n_points} measurement points are within their limits (ISO 20816 / API 670). "
                    "No active alarms."
                    if lang == "en" else
                    f"Los {n_points} puntos de medición están dentro de sus límites "
                    "(ISO 20816 / API 670). No hay alarmas activas.")}


def health_word(score: Optional[int], lang: str) -> str:
    if score is None:
        return "No data" if lang == "en" else "Sin datos"
    if score >= 90:
        return "Excellent" if lang == "en" else "Excelente"
    if score >= 75:
        return "Good" if lang == "en" else "Buena"
    if score >= 50:
        return "Attention" if lang == "en" else "Atención"
    return "Critical" if lang == "en" else "Crítica"


def next_report_at(schedules: Sequence[Dict[str, Any]], now: datetime,
                   period: str = "Semanal") -> Optional[datetime]:
    """Próximo envío programado (slots [día 0=lunes, hora]) del periodo dado."""
    best = None
    for e in schedules or []:
        if not e.get("enabled") or not str(e.get("period", "")).startswith(period):
            continue
        for d, h in e.get("slots") or []:
            delta = (int(d) - now.weekday()) % 7
            cand = (now + timedelta(days=delta)).replace(hour=int(h), minute=0, second=0, microsecond=0)
            if cand <= now:
                cand += timedelta(days=7)
            if best is None or cand < best:
                best = cand
    return best


_MONTHS = {"es": ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"],
           "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}
_DAYS = {"es": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
         "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}


def fmt_day(d: Optional[datetime], lang: str, weekday: bool = False) -> str:
    if d is None:
        return "—"
    m = _MONTHS["en" if lang == "en" else "es"][d.month - 1]
    base = f"{m} {d.day}" if lang == "en" else f"{d.day} {m}"
    if weekday:
        wd = _DAYS["en" if lang == "en" else "es"][d.weekday()]
        return f"{wd} {base}"
    return base


def whatsapp_url(text: str) -> str:
    return f"https://wa.me/{SIGA_WHATSAPP}?text={urllib.parse.quote(text)}"


# ---------------------------------------------------------------------------
# Centro SIGA — cola "Needs attention"
# ---------------------------------------------------------------------------

def attention_items(fleet: Sequence[Dict[str, Any]],
                    events_by_iid: Dict[str, List[Dict[str, Any]]],
                    pending: Sequence[Tuple[str, str, Dict[str, Any]]] = ()) -> List[Dict[str, Any]]:
    """Lo que SIGA tiene que mirar, ordenado: rojo → ámbar → azul."""
    items: List[Dict[str, Any]] = []
    for a in fleet:
        tag, iid = a.get("tag") or a["iid"], a["iid"]
        state = a.get("state")
        if state in ("nodata", "offline"):
            items.append({"sev": 0, "color": "dang", "iid": iid,
                          "title": f"{tag} · no data {a.get('age_txt') or ''}".strip(),
                          "body": "No live readings from the collector. Check the plant link / gateway."})
            continue
        if a.get("n_danger"):
            items.append({"sev": 0, "color": "dang", "iid": iid,
                          "title": f"{tag} · {a['n_danger']} point(s) in DANGER",
                          "body": "Confirm with the trend and call the client."})
        if a.get("n_alarm"):
            items.append({"sev": 1, "color": "warn", "iid": iid,
                          "title": f"{tag} · {a['n_alarm']} point(s) in alarm",
                          "body": "Review the trend and leave a note for the client."})
        for g in events_by_iid.get(iid, []):
            if g.get("chattering"):
                mx = f"{g['max_value']:.2f}" if g.get("max_value") is not None else "—"
                items.append({"sev": 1, "color": "warn", "iid": iid,
                              "title": f"{tag} · {g['tag']} chattering",
                              "body": (f"{g['n_up']} alarm entries in 7 d, max {mx} {g.get('unit', '')} "
                                       f"vs {g.get('alarm')}. Consider an alarm delay or check the source.")})
        hp = a.get("hottest")
        if hp and hp.get("pct") is not None and NEAR_LIMIT_PCT <= hp["pct"] < 100 and not a.get("n_alarm"):
            items.append({"sev": 2, "color": "warn", "iid": iid,
                          "title": f"{tag} · {hp['tag']} at {hp['pct']} % of alarm",
                          "body": "Close to its limit — watch the trend."})
    for iid, tag, d in pending or []:
        items.append({"sev": 3, "color": "info", "iid": iid,
                      "title": f"{tag} · {d.get('period', 'Report')} report waiting for approval",
                      "body": "Review and sign it in Report Center → Approval."})
    items.sort(key=lambda i: i["sev"])
    return items


# ---------------------------------------------------------------------------
# Nota semanal para el cliente (parámetro del activo)
# ---------------------------------------------------------------------------

def get_client_note(instance_id: str) -> Optional[Dict[str, Any]]:
    try:
        from core.instance_state import get_instance_parameters
        v = (get_instance_parameters(instance_id) or {}).get(CLIENT_NOTE_KEY)
        return v if isinstance(v, dict) and str(v.get("text", "")).strip() else None
    except Exception:  # noqa: BLE001
        return None


def save_client_note(instance_id: str, text: str, by: str) -> bool:
    try:
        from core.instance_state import update_instance_parameter
        val = ({"text": text.strip()[:600], "by": by,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
               if (text or "").strip() else {})
        return bool(update_instance_parameter(instance_id, CLIENT_NOTE_KEY, val))
    except Exception:  # noqa: BLE001
        return False


# ===========================================================================
# RENDER (Streamlit)
# ===========================================================================

_TONE = {"ok": "#3fd17f", "warn": "#f0b429", "dang": "#ff6b61", "off": "#9aa8ba"}
_DOT = {"ok": "#1f9d55", "warn": "#d99a00", "dang": "#d64541", "info": "#1d5fa8",
        "note": "#00a9e0", "off": "#8a97a8"}

_CSS = """
<style>
.lp, .lp * { box-sizing: border-box; }
.lp { font-family: 'IBM Plex Sans', system-ui, sans-serif; color: #0f172a; }
.lp-mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.lp-hero { background: #0b1a33; color: #fff; border-radius: 20px; padding: 30px 34px 26px;
  display: flex; flex-wrap: wrap; gap: 26px 34px; align-items: center; margin: 4px 0 18px;
  background-image: radial-gradient(900px 300px at 100% 0%, rgba(0,169,224,.12), transparent 60%); }
.lp-hero-main { flex: 999 1 460px; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.lp-eyebrow { display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
  font: 600 12px 'IBM Plex Mono', monospace; letter-spacing: .14em; color: #8fb3dd; text-transform: uppercase; }
.lp-live { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 4px 12px; }
.lp-live i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.lp-live.on { background: rgba(31,157,85,.2); color: #6fe3a1; }
.lp-live.on i { background: #3fd17f; box-shadow: 0 0 0 4px rgba(63,209,127,.18); animation: lp-pulse 1.8s infinite; }
.lp-live.off { background: rgba(154,168,186,.18); color: #c3cdda; }
.lp-live.off i { background: #9aa8ba; }
@keyframes lp-pulse { 0%,100% { box-shadow: 0 0 0 3px rgba(63,209,127,.25); } 50% { box-shadow: 0 0 0 7px rgba(63,209,127,0); } }
@media (prefers-reduced-motion: reduce) { .lp-live.on i { animation: none; } }
.lp-h1 { margin: 0; font-size: clamp(30px, 3.6vw, 46px); line-height: 1.06; font-weight: 700; letter-spacing: -.02em; color: #fff; text-wrap: balance; }
.lp-sub { margin: 0; font-size: 17px; line-height: 1.5; color: #c9d6e8; max-width: 640px; }
.lp-health { flex: 1 1 250px; display: flex; align-items: center; gap: 18px; }
.lp-health b { display: block; font-size: 21px; color: #fff; }
.lp-health span { font-size: 14px; color: #c9d6e8; line-height: 1.45; }
.lp-kpis { flex-basis: 100%; display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 1px; background: #1c3258; border-radius: 14px; overflow: hidden; }
.lp-kpi { background: #10223f; padding: 16px 20px; }
.lp-kpi .v { font: 600 26px 'IBM Plex Mono', monospace; color: #fff; font-variant-numeric: tabular-nums; }
.lp-kpi .v small { font-size: 13px; color: #8fb3dd; font-weight: 500; }
.lp-kpi .l { font-size: 13px; color: #8fb3dd; margin-top: 3px; }
.lp-h2 { margin: 6px 0 10px; font-size: 20px; font-weight: 700; color: #0f172a; }
.lp-card-ic { width: 40px; height: 40px; border-radius: 11px; background: #e8f1fb; display: flex;
  align-items: center; justify-content: center; margin-bottom: 8px; }
.lp-card-t { font-size: 16px; font-weight: 600; color: #0f172a; line-height: 1.3; min-height: 42px; }
.lp-card-d { font-size: 13px; color: #56667e; line-height: 1.45; min-height: 58px; margin: 2px 0 6px; }
.lp-panel { background: #fff; border: 1px solid #dbe3ee; border-radius: 18px; padding: 20px 22px; }
.lp-panel-h { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.lp-ph { margin: 0; font: 600 17px 'IBM Plex Sans', sans-serif; color: #0f172a; letter-spacing: -.005em; }
.lp-panel-h span { font: 500 12px 'IBM Plex Mono', monospace; color: #56667e; }
.lp-line { display: flex; gap: 12px; padding: 7px 0; font-size: 14.5px; line-height: 1.5; color: #24344d; }
.lp-line i { flex: none; width: 10px; height: 10px; border-radius: 50%; margin-top: 6px; }
.lp-label { font: 600 11px 'IBM Plex Mono', monospace; letter-spacing: .12em; color: #56667e; text-transform: uppercase; }
.lp-pt { display: grid; grid-template-columns: 50px 1fr 108px; gap: 12px; align-items: center; padding: 4px 0; }
.lp-pt .t { font: 600 13.5px 'IBM Plex Mono', monospace; color: #0f172a; }
.lp-pt .bar { height: 9px; background: #edf1f6; border-radius: 999px; overflow: hidden; position: relative; }
.lp-pt .bar b { display: block; height: 100%; border-radius: 999px; }
.lp-pt .v { font: 500 13px 'IBM Plex Mono', monospace; text-align: right; color: #24344d; font-variant-numeric: tabular-nums; }
.lp-grp { display: flex; flex-direction: column; gap: 4px; }
.lp-grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px 34px; }
.lp-spec { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; background: #e8f1fb;
  border-radius: 18px; padding: 18px 22px; margin-top: 4px; }
.lp-av { width: 50px; height: 50px; border-radius: 50%; background: #1d5fa8; color: #fff; display: flex;
  align-items: center; justify-content: center; font-weight: 700; font-size: 18px; flex: none; }
.lp-ops-card { display: flex; flex-direction: column; gap: 6px; }
.lp-ops-top { display: flex; justify-content: space-between; gap: 8px; font: 600 11.5px 'IBM Plex Mono', monospace; letter-spacing: .1em; }
.lp-ops-tag { font-size: 21px; font-weight: 700; color: #0f172a; }
.lp-ops-meta { font: 500 13px 'IBM Plex Mono', monospace; color: #56667e; }
.lp-att { display: grid; grid-template-columns: 10px 1fr; gap: 12px; padding: 10px 0; border-bottom: 1px solid #edf1f6; }
.lp-att:last-child { border-bottom: 0; }
.lp-att i { width: 10px; height: 10px; border-radius: 50%; margin-top: 6px; }
.lp-att b { display: block; font-size: 14.5px; color: #0f172a; }
.lp-att span { font-size: 13px; color: #56667e; line-height: 1.45; }
.lp-act-row { display: flex; flex-direction: column; gap: 2px; padding: 8px 0; border-bottom: 1px solid #edf1f6; font-size: 13px; color: #56667e; }
.lp-act-row:last-child { border-bottom: 0; }
.lp-act-row b { color: #0f172a; font-size: 13.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lp-preview { background: #fff7e0; border: 1px solid #f3d27a; border-radius: 12px; padding: 10px 16px;
  font-size: 14px; color: #6b4b00; margin-bottom: 10px; }
div[class*="st-key-lp_card_"] { background: #fff; border-radius: 16px !important; border-color: #dbe3ee !important;
  transition: box-shadow .16s ease, transform .16s ease; height: 100%; }
div[class*="st-key-lp_card_"]:hover { box-shadow: 0 12px 26px rgba(15,42,74,.10); transform: translateY(-1px); }
div[class*="st-key-lp_fleet_"] { background: #fff; border-radius: 16px !important; border-color: #dbe3ee !important; }
div[class*="st-key-lp_fleet_"][class*="_sel"] { border: 2px solid #00a9e0 !important; box-shadow: 0 10px 24px rgba(0,169,224,.14); }
div[class*="st-key-lp_note_box"], div[class*="st-key-lp_activity_box"], div[class*="st-key-lp_lastrep"] {
  background: #fff; border-radius: 16px !important; border-color: #dbe3ee !important; }
div[class*="st-key-lp_card_"] [data-testid="stPageLink"] a { justify-content: center; border: 1px solid #dbe4f0;
  border-radius: 11px; background: #fff; padding: 6px 12px; font-weight: 700; color: #0f2a4a; }
div[class*="st-key-lp_card_"] [data-testid="stPageLink"] a:hover { background: #12305e; }
div[class*="st-key-lp_card_"] [data-testid="stPageLink"] a:hover p { color: #fff; }
.lp-rep-meta { font-size: 13.5px; color: #56667e; padding-bottom: 6px; }
</style>
"""

_ICONS = {
    "trend": '<path d="M3 17l5-5 4 4 8-9"/><path d="M14 7h6v6"/>',
    "report": '<path d="M12 3v12"/><path d="M7 10l5 5 5-5"/><path d="M4 20h16"/>',
    "archive": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18"/><path d="M8 14h8"/>',
    "chat": '<path d="M21 12a8.5 8.5 0 0 1-12.6 7.4L3 21l1.6-5.2A8.5 8.5 0 1 1 21 12z"/>',
    "phone": '<rect x="7" y="2" width="10" height="20" rx="2"/><path d="M11 18h2"/>',
}


def _icon(name: str, color: str = "#1d5fa8") -> str:
    return (f'<div class="lp-card-ic"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{_ICONS[name]}</svg></div>')


def inject_css() -> None:
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)


def render_lang_toggle(key: str = "wm_lang_toggle") -> None:
    """Selector ES | EN (vista cliente). Persiste en session_state['wm_lang']."""
    import streamlit as st
    from core.portal_i18n import LANG_KEY, lang

    def _sync():
        v = st.session_state.get(key)
        if v in ("ES", "EN"):
            st.session_state[LANG_KEY] = v.lower()

    if key not in st.session_state:
        st.session_state[key] = lang().upper()
    st.segmented_control("Idioma / Language", ["ES", "EN"], key=key, on_change=_sync,
                         label_visibility="collapsed")


def _health_ring(score: Optional[int], color: str, label: str) -> str:
    s = 0 if score is None else max(0, min(100, int(score)))
    circ = 2 * 3.14159 * 52
    dash = circ * s / 100
    txt = "—" if score is None else str(s)
    return (f'<svg width="124" height="124" viewBox="0 0 124 124" role="img" aria-label="{_esc(label)} {txt}">'
            f'<circle cx="62" cy="62" r="52" fill="none" stroke="#1c3258" stroke-width="11"/>'
            f'<circle cx="62" cy="62" r="52" fill="none" stroke="{color}" stroke-width="11" '
            f'stroke-linecap="round" stroke-dasharray="{dash:.1f} {circ:.1f}" transform="rotate(-90 62 62)"/>'
            f'<text x="62" y="68" text-anchor="middle" fill="#fff" '
            f'style="font:700 34px IBM Plex Sans,sans-serif">{txt}</text>'
            f'<text x="62" y="88" text-anchor="middle" fill="#8fb3dd" '
            f'style="font:600 10px IBM Plex Mono,monospace;letter-spacing:.1em">{_esc(label.upper())}</text></svg>')


def render_client_hero(k: Dict[str, Any], lang: str) -> None:
    """Veredicto grande + salud + KPIs. `k` lo arma la página (live_hero_kpis)."""
    import streamlit as st
    v = k["verdict"]
    live = k.get("live", False)
    pill = (f'<span class="lp-live on"><i></i>{"LIVE" if lang == "en" else "EN VIVO"}</span>' if live else
            f'<span class="lp-live off"><i></i>{"NO SIGNAL" if lang == "en" else "SIN SEÑAL"}</span>')
    reading = ("LAST READING " if lang == "en" else "ÚLTIMA LECTURA ") + _esc(k.get("age_txt") or "—")
    score, hcolor = k.get("health"), k.get("health_color") or "#3fd17f"
    if v["tone"] in ("off",):
        hcolor = "#9aa8ba"
    hot = k.get("hottest")
    if hot:
        hot_txt = (f"Most loaded point: {hot['tag']} at {hot['pct']} % of its alarm" if lang == "en"
                   else f"Punto más cargado: {hot['tag']} al {hot['pct']} % de su alarma")
    else:
        hot_txt = ""
    tiles = []
    for lbl, val, unit in k.get("tiles", []):
        tiles.append(f'<div class="lp-kpi"><div class="v">{val}<small> {_esc(unit)}</small></div>'
                     f'<div class="l">{_esc(lbl)}</div></div>')
    st.markdown(
        f'<div class="lp lp-hero">'
        f'<div class="lp-hero-main"><div class="lp-eyebrow">{pill}<span>{_esc(k.get("asset_line", ""))}'
        f' · {reading}</span></div>'
        f'<h1 class="lp-h1">{v["title"]}</h1><p class="lp-sub">{v["sub"]}</p></div>'
        f'<div class="lp-health">{_health_ring(score, hcolor, "Health" if lang == "en" else "Salud")}'
        f'<div><b>{_esc(health_word(score, lang))}</b><span>{_esc(hot_txt)}</span></div></div>'
        f'<div class="lp-kpis">{"".join(tiles)}</div></div>',
        unsafe_allow_html=True)


def render_client_actions(*, instance_id: str, tag: str, client_name: str, pdf_bytes: Optional[bytes],
                          n_reports: Optional[int], lang: str, tab_key: str) -> None:
    """'¿Qué necesita hoy?' — 5 tarjetas con su acción real."""
    import streamlit as st
    from core.portal_activity import log_activity

    st.markdown(f'<div class="lp"><h2 class="lp-h2">{"What do you need today?" if lang == "en" else "¿Qué necesita hoy?"}</h2></div>',
                unsafe_allow_html=True)
    cols = st.columns(5, gap="small")

    def _card(col, key, icon, title, desc, icolor="#1d5fa8"):
        with col:
            box = st.container(border=True, key=f"lp_card_{key}")
            box.markdown(f'<div class="lp">{_icon(icon, icolor)}<div class="lp-card-t">{_esc(title)}</div>'
                         f'<div class="lp-card-d">{_esc(desc)}</div></div>', unsafe_allow_html=True)
            return box

    def _go_trend():
        st.session_state[tab_key] = "trend"

    b = _card(cols[0], "trend", "trend", "See trends" if lang == "en" else "Ver tendencias",
              "How each point evolved: 24 h, 7 days or 30 days." if lang == "en"
              else "Cómo evolucionó cada punto: 24 h, 7 días o 30 días.")
    b.button("Open trends" if lang == "en" else "Abrir tendencias", key=f"lp_btn_trend_{instance_id}",
             on_click=_go_trend, use_container_width=True)

    b = _card(cols[1], "report", "report", "Today's report" if lang == "en" else "Reporte de hoy",
              "PDF with the current status, ready to share." if lang == "en"
              else "PDF con el estado actual, listo para compartir.")
    if pdf_bytes:
        b.download_button("Download PDF" if lang == "en" else "Descargar PDF", data=pdf_bytes,
                          file_name=f"Reporte_{tag}_{datetime.now():%Y%m%d_%H%M}.pdf",
                          mime="application/pdf", use_container_width=True,
                          key=f"lp_dl_{instance_id}",
                          on_click=lambda: log_activity("download_live_report", instance_id))
    else:
        b.caption("No live readings to build it." if lang == "en" else "Sin lecturas para armarlo.")

    nr = f"{n_reports} " if n_reports else ""
    b = _card(cols[2], "archive", "archive", "Report archive" if lang == "en" else "Archivo de reportes",
              f"{nr}weekly, monthly and technical reports." if lang == "en"
              else f"{nr}reportes semanales, mensuales y técnicos.")
    try:
        b.page_link("pages/16_Reports.py", label="Open archive" if lang == "en" else "Abrir archivo",
                    use_container_width=True)
    except Exception:  # noqa: BLE001
        pass

    msg = (f"Hello, this is the {client_name} team. I'd like to ask about {tag} (Watermelon Live Monitoring)."
           if lang == "en" else
           f"Hola, le escribo del equipo de {client_name}. Quisiera consultar sobre la {tag} (Watermelon Live Monitoring).")
    b = _card(cols[3], "chat", "chat", "Your specialist" if lang == "en" else "Su especialista",
              "WhatsApp with the machine already in the message." if lang == "en"
              else "WhatsApp con la máquina ya en el mensaje.", "#1f9d55")
    b.link_button("Open WhatsApp" if lang == "en" else "Abrir WhatsApp", whatsapp_url(msg),
                  use_container_width=True)

    b = _card(cols[4], "app", "phone", "On your phone" if lang == "en" else "En su celular",
              "Install Watermelon Live and get alarms instantly." if lang == "en"
              else "Instale Watermelon Live y reciba las alarmas al instante.")
    b.link_button("Open the app" if lang == "en" else "Abrir la app", LIVE_APP_URL, use_container_width=True)


CLIENT_TABS = ["summary", "trend", "events", "tech"]
_TAB_LABELS = {"summary": ("Resumen", "Summary"), "trend": ("Tendencias", "Trends"),
               "events": ("Eventos", "Events"), "tech": ("Vista técnica", "Technical view")}


def client_tab_selector(key: str, lang: str) -> str:
    import streamlit as st
    if st.session_state.get(key) not in CLIENT_TABS:
        st.session_state[key] = "summary"
    v = st.segmented_control("Sección", CLIENT_TABS, key=key, label_visibility="collapsed",
                             format_func=lambda o: _L(_TAB_LABELS[o], lang))
    return v or "summary"


def render_this_week(lines: List[Dict[str, str]], *, last_report: Optional[Dict[str, Any]],
                     next_at: Optional[datetime], lang: str, viewer_email: str, viewer_role: str) -> None:
    import streamlit as st
    now = datetime.now()
    rng = f"{fmt_day(now - timedelta(days=7), lang)} – {fmt_day(now, lang)}"
    body = "".join(f'<div class="lp-line"><i style="background:{_DOT.get(l["kind"], "#1d5fa8")}"></i>'
                   f'<div>{l["html"]}</div></div>' for l in lines)
    st.markdown(f'<div class="lp lp-panel"><div class="lp-panel-h"><div class="lp-ph">{"This week" if lang == "en" else "Esta semana"}</div>'
                f'<span>{_esc(rng)}</span></div>{body}</div>', unsafe_allow_html=True)
    box = st.container(border=True, key="lp_lastrep")
    if last_report:
        from core.client_report_archive import (_pdf_loader, download_name, fmt_date, report_kind)
        from core.portal_activity import log_activity as _log
        rm = last_report.get("report_meta") or {}
        consec = str(rm.get("consecutive") or "").strip() or ("Report" if lang == "en" else "Reporte")
        kind = report_kind(last_report)
        kind_txt = {"Semanal": ("Semanal", "Weekly"), "Mensual": ("Mensual", "Monthly")}.get(kind, (kind, kind))
        sev = str(rm.get("executive_severity") or "").strip()
        meta = f"{_L(kind_txt, lang)} · {fmt_date(str(rm.get('report_date') or last_report.get('archived_at', ''))[:10])}"
        if sev:
            meta += f" · {sev}"
        box.markdown(f'<div class="lp"><div class="lp-label">{"Last report" if lang == "en" else "Último reporte"}</div>'
                     f'<div class="lp-mono" style="font-weight:600;font-size:14px;margin-top:6px">{_esc(consec)}</div>'
                     f'<div class="lp-rep-meta">{_esc(meta)}</div></div>',
                     unsafe_allow_html=True)
        aid = last_report.get("archive_id", "")
        try:
            box.download_button(
                "Download PDF" if lang == "en" else "Descargar PDF",
                data=_pdf_loader(last_report, viewer_email, viewer_role),
                file_name=download_name(last_report), mime="application/pdf",
                key=f"lp_lastrep_dl_{aid}", use_container_width=True,
                on_click=lambda: _log("download_archived_report", detail=aid))
        except Exception:  # noqa: BLE001
            pass
    else:
        box.markdown(f'<div class="lp"><div class="lp-label">{"Last report" if lang == "en" else "Último reporte"}</div>'
                     f'<div style="font-size:13.5px;color:#56667e;margin-top:6px">'
                     f'{"No reports archived yet." if lang == "en" else "Aún no hay reportes archivados."}</div></div>',
                     unsafe_allow_html=True)
    if next_at:
        box.caption((f"Next weekly report: {fmt_day(next_at, lang, weekday=True)}" if lang == "en"
                     else f"Próximo reporte semanal: {fmt_day(next_at, lang, weekday=True)}"))


def render_points(groups: List[Dict[str, Any]], lang: str) -> None:
    import streamlit as st
    blocks = []
    for g in groups:
        rows = []
        for p in g["points"]:
            pct = p["pct"]
            if p["status"] == "Danger":
                col = "#d64541"
            elif p["status"] == "Alarma":
                col = "#e08a00"
            elif pct is not None and pct >= NEAR_LIMIT_PCT:
                col = "#d99a00"
            else:
                col = "#1f9d55"
            w = 0 if pct is None else min(100, pct)
            val = f"{_num(p['value'], 2 if abs(float(p['value'] or 0)) < 10 else 1, lang)} {_esc(p['unit'])}"
            title = (f"{p['tag']}: {pct} % of alarm" if lang == "en" else f"{p['tag']}: {pct} % de la alarma") \
                if pct is not None else p["tag"]
            rows.append(f'<div class="lp-pt" title="{_esc(title)}"><span class="t">{p["tag"]}</span>'
                        f'<div class="bar"><b style="width:{w}%;background:{col}"></b></div>'
                        f'<span class="v">{val}</span></div>')
        blocks.append(f'<div class="lp-grp"><div class="lp-label">{_esc(_L(COMP_NAMES.get(g["key"], COMP_NAMES["other"]), lang))}</div>'
                      f'{"".join(rows)}</div>')
    hint = ("The bar shows how close each point is to its alarm limit" if lang == "en"
            else "La barra muestra qué tan cerca está cada punto de su límite de alarma")
    st.markdown(f'<div class="lp lp-panel"><div class="lp-panel-h"><div class="lp-ph">{"Measurement points" if lang == "en" else "Puntos de medición"}</div>'
                f'<span>{_esc(hint)}</span></div><div class="lp-grid2">{"".join(blocks)}</div></div>',
                unsafe_allow_html=True)


def render_specialist(signers: Dict[str, Any], *, tag: str, client_name: str, lang: str) -> None:
    import streamlit as st
    name = str(signers.get("prepared_by") or "SIGA").strip()
    role = str(signers.get("prepared_role") or "").strip()
    initials = "".join(w[0] for w in name.split()[:2]).upper() or "S"
    c1, c2 = st.columns([4, 1.3], vertical_alignment="center")
    with c1:
        st.markdown(
            f'<div class="lp lp-spec"><div class="lp-av">{_esc(initials)}</div><div>'
            f'<div style="font-size:16.5px;font-weight:600">{"Your specialist" if lang == "en" else "Su especialista"}: {_esc(name)}</div>'
            f'<div style="font-size:13.5px;color:#3d4f69">SIGA S.A.S.{" · " + _esc(role) if role else ""}</div></div></div>',
            unsafe_allow_html=True)
    with c2:
        msg = (f"Hello {name.split()[0]}, this is the {client_name} team, about {tag}."
               if lang == "en" else f"Hola {name.split()[0]}, le escribo del equipo de {client_name} sobre la {tag}.")
        st.link_button("Write on WhatsApp" if lang == "en" else "Escribir por WhatsApp", whatsapp_url(msg),
                       use_container_width=True, type="primary")


def render_event_summary(ev_groups: List[Dict[str, Any]], groups: List[Dict[str, Any]], lang: str) -> None:
    import streamlit as st
    planes = {p["tag"]: p["plane"] for g in groups for p in g["points"]}
    if not ev_groups:
        st.markdown(f'<div class="lp lp-panel"><div class="lp-line"><i style="background:#1f9d55"></i><div>'
                    f'{"No alarm crossings in the last 7 days." if lang == "en" else "Sin cruces de alarma en los últimos 7 días."}'
                    f'</div></div></div>', unsafe_allow_html=True)
        return
    rows = []
    for g in ev_groups:
        name = friendly_point(g["tag"], planes.get(g["tag"], ""), lang)
        mx = _num(g["max_value"], 2, lang) if g["max_value"] is not None else "—"
        state = (("In alarm now" if lang == "en" else "En alarma ahora") if g["active"]
                 else ("Normal now" if lang == "en" else "Normal ahora"))
        col = _DOT["dang" if g["worst"] == "Danger" and g["active"] else ("warn" if g["active"] else "ok")]
        n_txt = f"{g['n_up']} {'entries' if lang == 'en' else 'entradas'}"
        rows.append(f'<div class="lp-att"><i style="background:{col}"></i><div><b>{_esc(name)} · {g["tag"]}</b>'
                    f'<span>{n_txt} · {"max" if lang == "en" else "máx"} {mx} {_esc(g["unit"])} · {state}</span></div></div>')
    st.markdown(f'<div class="lp lp-panel"><div class="lp-panel-h"><div class="lp-ph">{"Last 7 days, by point" if lang == "en" else "Últimos 7 días, por punto"}</div>'
                f'<span>{len(ev_groups)} {"points" if lang == "en" else "puntos"}</span></div>{"".join(rows)}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Centro SIGA
# ---------------------------------------------------------------------------

def render_fleet(fleet: List[Dict[str, Any]], selected: str, select_key: str) -> None:
    import streamlit as st
    st.markdown('<div class="lp"><div class="lp-panel-h" style="margin:2px 0 8px"><div class="lp-ph" style="font-size:22px">'
                'Fleet · all clients</div><span>'
                f'{sum(1 for a in fleet if a["state"] == "online")} online · {len(fleet)} in service · refresh 30 s'
                '</span></div></div>', unsafe_allow_html=True)
    per_row = 4
    for i in range(0, len(fleet), per_row):
        cols = st.columns(per_row, gap="small")
        for col, a in zip(cols, fleet[i:i + per_row]):
            with col:
                st_col = {"online": ("#1f9d55", "ONLINE"), "stopped": ("#8a97a8", "STOPPED"),
                          "nodata": ("#d64541", "NO DATA"), "offline": ("#d64541", "NO DATA")}[a["state"]]
                if a["state"] == "online" and a.get("n_danger"):
                    st_col = ("#d64541", "DANGER")
                elif a["state"] == "online" and a.get("n_alarm"):
                    st_col = ("#d99a00", "ALARM")
                meta = a.get("meta") or ""
                box = st.container(border=True, key=f"lp_fleet_{a['iid']}" + ("_sel" if a["iid"] == selected else ""))
                box.markdown(
                    f'<div class="lp lp-ops-card"><div class="lp-ops-top"><span style="color:#56667e">'
                    f'{_esc((a.get("client") or "—").upper())}</span><span style="color:{st_col[0]};display:flex;'
                    f'align-items:center;gap:6px"><i style="width:8px;height:8px;border-radius:50%;'
                    f'background:{st_col[0]};display:inline-block"></i>{st_col[1]}</span></div>'
                    f'<div class="lp-ops-tag">{_esc(a["tag"])}</div><div class="lp-ops-meta">{_esc(meta)}</div></div>',
                    unsafe_allow_html=True)

                def _sel(iid=a["iid"]):
                    st.session_state[select_key] = iid
                    st.session_state["_live_sel"] = iid
                box.button("Selected" if a["iid"] == selected else "Open", key=f"lp_fleet_btn_{a['iid']}",
                           on_click=_sel, use_container_width=True, disabled=a["iid"] == selected)


def render_attention(items: List[Dict[str, Any]]) -> None:
    import streamlit as st
    body = "".join(f'<div class="lp-att"><i style="background:{_DOT[it["color"]]}"></i><div>'
                   f'<b>{_esc(it["title"])}</b><span>{_esc(it["body"])}</span></div></div>' for it in items[:8])
    if not items:
        body = ('<div class="lp-line"><i style="background:#1f9d55"></i><div>Nothing needs attention. '
                'All assets online and within limits.</div></div>')
    st.markdown(f'<div class="lp lp-panel" style="height:100%"><div class="lp-panel-h"><div class="lp-ph">Needs attention</div>'
                f'<span style="color:{"#b45309" if items else "#1f9d55"}">{len(items)} open</span></div>{body}</div>',
                unsafe_allow_html=True)


def render_client_note_editor(instance_id: str, tag: str, me_email: str) -> None:
    import streamlit as st
    note = get_client_note(instance_id) or {}
    box = st.container(border=True, key="lp_note_box")
    box.markdown(f'<div class="lp"><div class="lp-panel-h" style="margin:0"><div class="lp-ph">Note for the client</div>'
                 f'<span>{_esc(tag)}</span></div></div>', unsafe_allow_html=True)
    k = f"lp_note_{instance_id}"
    txt = box.text_area("Shown first in the client's “Esta semana” card (write it in Spanish).",
                        value=note.get("text", ""), key=k, height=110, max_chars=600)
    c1, c2 = box.columns(2)
    if c1.button("Publish", key=f"{k}_pub", use_container_width=True, type="primary"):
        if save_client_note(instance_id, txt, me_email):
            st.toast("Published — the client sees it now.")
        else:
            st.error("Could not save the note.")
    if c2.button("Clear", key=f"{k}_clr", use_container_width=True, disabled=not note):
        if save_client_note(instance_id, "", me_email):
            st.session_state.pop(k, None)
            st.toast("Note removed.")
            st.rerun()
    if note.get("at"):
        box.caption(f"Last published {str(note['at'])[:16].replace('T', ' ')} UTC by {note.get('by', '—')}")


def render_client_activity() -> None:
    import streamlit as st
    from core.portal_activity import recent_activity, summarize_by_user, EVENT_LABELS
    rows = recent_activity(days=7)
    box = st.container(border=True, key="lp_activity_box")
    box.markdown('<div class="lp"><div class="lp-panel-h" style="margin:0"><div class="lp-ph">Client activity</div>'
                 '<span>last 7 d</span></div></div>', unsafe_allow_html=True)
    if rows is None:
        box.caption("Activity log not available (table portal_activity missing or offline).")
        return
    users = summarize_by_user(rows)
    if not users:
        box.caption("No client visits in the last 7 days.")
        return
    out = []
    for u in users[:8]:
        when = _parse_ts(u["last_at"])
        when_txt = when.astimezone(timezone(timedelta(hours=-5))).strftime("%d %b %H:%M") if when else "—"
        last = EVENT_LABELS.get(u["last_event"], (u["last_event"], u["last_event"]))[1]
        out.append(f'<div class="lp-act-row"><b title="{_esc(u["email"])}">{_esc(u["email"])}</b>'
                   f'<span>{_esc(last)} · <span class="lp-mono">{when_txt}</span></span>'
                   f'<span>{u["visits"]} visits · {u["downloads"]} downloads</span></div>')
    box.markdown(f'<div class="lp">{"".join(out)}</div>', unsafe_allow_html=True)
