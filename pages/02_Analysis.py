"""
pages/02_Analysis.py — Workspace ÚNICO de análisis (Fase 2).
================================================================

Clase ADRE/System1: INGESTA + ANÁLISIS en un solo lugar. Subes uno o varios CSV
(System1 exporta 1 canal por archivo → se emparejan X/Y por cojinete), eliges
canal/cojinete y navegas las VISTAS con un segmented control, sin recargar el
dato (vive en session_state).

Vistas de UNA captura (CSV manual): Waveform · Espectro · Órbita · Shaft Centerline.
Vistas de BARRIDO/HISTÓRICO (run-up / varias capturas): Polar · Bode · Trends →
se avisan (requieren fuente de sweep/historial, no una sola captura).
"""
from __future__ import annotations

import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.auth import require_login, render_user_menu

st.set_page_config(page_title="Watermelon System | Rotodinámica", layout="wide",
                   initial_sidebar_state="collapsed")
require_login()
render_user_menu()

from core.analysis.model import AnalysisRecord, build_record
from core.ui_industrial import inject_industrial_css
from core.ui_theme import apply_watermelon_page_style, page_header

apply_watermelon_page_style()
inject_industrial_css()
page_header("Rotodinámica", "Análisis de vibración · todas las vistas en un solo lugar")

SS = st.session_state
_NAVY = "#12305e"
_AMBER = "#e8890c"
_STEEL = "#274b7d"


@st.cache_data(show_spinner=False)
def _record_from(payload) -> AnalysisRecord:
    # payload = [(name, bytes), ...] para que el cache sea estable
    class _F:
        def __init__(self, n, b): self.name = n; self._b = b
        def getvalue(self): return self._b
    return build_record([_F(n, b) for n, b in payload])


@st.cache_data(ttl=120, show_spinner=False)
def _list_assets():
    from core.instance_state import get_instance, list_reportable_instances
    out = []
    for r in (list_reportable_instances() or []):
        iid = r.get("instance_id") if isinstance(r, dict) else getattr(r, "instance_id", "")
        if not iid:
            continue
        tag = (r.get("tag") if isinstance(r, dict) else getattr(r, "tag", "")) or ""
        if not tag:
            try:
                tag = getattr(get_instance(iid), "tag", "") or iid
            except Exception:  # noqa: BLE001
                tag = iid
        out.append((iid, tag))
    return out


@st.cache_data(ttl=90, show_spinner=False)
def _list_caps(iid):
    from core.dynamic_raw import list_captures
    return list_captures(iid) or []


@st.cache_data(ttl=600, show_spinner=False)
def _rec_history(label, keys):
    from core.analysis.model import record_from_history
    return record_from_history(label, list(keys))


@st.cache_data(ttl=600, show_spinner=False)
def _trend(caps_tuple, point, metric, axis="X", limit=800):
    from core.analysis.model import trend_series
    caps = [{"point": p, "day": d, "time": t, "key": k}
            for p, d, t, k in caps_tuple]
    return trend_series(caps, point, metric, axis=axis, limit=limit)


@st.cache_data(ttl=600, show_spinner=False)
def _spectra(caps_tuple, limit=24):
    from core.analysis.model import spectra_series
    caps = [{"point": p, "day": d, "time": t, "key": k}
            for p, d, t, k in caps_tuple]
    return spectra_series(caps, limit=limit)


@st.cache_data(ttl=300, show_spinner=False)
def _inst_cfg(iid):
    """Config de setpoints de la máquina: (sensors_map, alert_level, danger_level,
    setpoint_unit). Para dar PRIORIDAD a lo configurado sobre la norma ISO."""
    if not iid:
        return [], 0.0, 0.0, ""
    try:
        from core.instance_state import get_instance
        inst = get_instance(iid)
        return (list(getattr(inst, "sensors", []) or []),
                float(getattr(inst, "alert_level", 0.0) or 0.0),
                float(getattr(inst, "danger_level", 0.0) or 0.0),
                (getattr(inst, "setpoint_unit", "") or "").strip())
    except Exception:  # noqa: BLE001
        return [], 0.0, 0.0, ""


# Reutiliza los MOTORES de norma ya validados (core.diagnostics / core.rotordynamics):
# ISO 20816-2 zona A/B/C/D, críticas con Q, margen API 684.
_ISO = {"A": "#1f9d55", "B": "#84cc16", "C": "#e8890c", "D": "#dc3545", "—": "#8090a6"}


@st.cache_data(ttl=600, show_spinner=False)
def _sweep_diag(rpm_t, amp_t, phase_t, op_rpm, group, mtype, unit, label):
    from core.diagnostics import build_polar_diagnostics_rotordyn
    return build_polar_diagnostics_rotordyn(
        rpm=np.array(rpm_t, float), amp=np.array(amp_t, float),
        phase=np.array(phase_t, float), operating_rpm=float(op_rpm),
        machine_group=group, amp_unit=unit, measurement_type=mtype,
        analysis_label=label)


def _diag_panel(diag, zone, zc):
    s = diag.get("structured", {})
    crits = s.get("criticals", [])
    crit_html = ""
    if crits:
        rows = "".join(
            f"<tr><td>{c['rpm']:.0f}</td><td>{c.get('q_factor',0):.1f}</td>"
            f"<td>{c.get('api684_actual_margin_pct',0):.0f}%</td>"
            f"<td style='color:{'#1f9d55' if c.get('api684_compliant') else '#dc3545'}'>"
            f"{c.get('api684_zone','—')}</td></tr>" for c in crits)
        crit_html = ("<table style='width:100%;font:12px IBM Plex Mono;margin-top:6px;border-collapse:collapse'>"
                     "<tr style='color:#8090a6'><td>Crítica rpm</td><td>Q</td>"
                     "<td>Margen API684</td><td>Zona</td></tr>" + rows + "</table>")
    st.markdown(
        f"<div style='border-left:4px solid {zc};background:#f8fafc;border:1px solid #e2e8f2;"
        f"border-radius:8px;padding:11px 15px;margin-top:10px'>"
        f"<span style='background:{zc};color:#fff;border-radius:6px;padding:2px 9px;"
        f"font:700 11px IBM Plex Mono;'>ISO 20816-2 · Zona {zone}</span>  "
        f"<b style='font-size:13px'>{diag.get('headline','')}</b>"
        f"<div style='color:#3a4c66;font-size:12.5px;margin-top:5px'>{diag.get('detail','')}</div>"
        f"<div style='color:#0b1f3a;font-size:12.5px;margin-top:4px'><b>Acción:</b> {diag.get('action','')}</div>"
        f"{crit_html}</div>", unsafe_allow_html=True)


# =========================================================
# 1) FUENTE — histórico automático (Supabase) o CSV manual
# =========================================================
src = st.radio("Fuente", ["Histórico del activo (automático)", "CSV manual (System1 / ADRE)"],
               horizontal=True, key="an_src")
rec: AnalysisRecord | None = None
hist_caps = None          # capturas del histórico (para Trends, solo en modo histórico)
cur_iid = None            # instance_id activo (modo histórico) → setpoints configurados
cur_asset = ""            # nombre del activo (para el nombre del archivo descargado)

if src.startswith("Histórico"):
    assets = _list_assets()
    if not assets:
        st.info("No hay activos disponibles.")
        st.stop()
    tags = [t for _i, t in assets]
    c0, c1 = st.columns([1, 2])
    with c0:
        tsel = st.selectbox("Activo", tags, key="an_asset")
        iid = assets[tags.index(tsel)][0]
        cur_iid = iid
        cur_asset = tsel
    caps = _list_caps(iid)
    if not caps:
        st.info(f"**{tsel}** aún no tiene onda cruda en el histórico. El robot System1 "
                "la sube automáticamente; en cuanto llegue, aparece aquí.")
        st.stop()
    hist_caps = caps                       # para la vista Trends
    from core.analysis.model import session_stamps
    sessions = session_stamps(caps)        # [(label, [keys]), …] reciente primero
    labels = [s[0] for s in sessions]
    with c1:
        lsel = st.selectbox("Captura (sesión de export)", labels, key="an_stamp")
    _, keys = sessions[labels.index(lsel)]
    with st.spinner("Cargando captura del histórico…"):
        rec = _rec_history(lsel, tuple(keys))
else:
    ups = st.file_uploader(
        "Registro CSV importado (uno o varios canales)", type=["csv"],
        accept_multiple_files=True,
        help="Export de System1 (1yd.csv, 1xd.csv…), Bode/Polar de System1, o ADRE. "
             "Sube el par X/Y de un cojinete para ver la órbita.")
    if ups:
        SS["an_payload"] = [(f.name, f.getvalue()) for f in ups]
    payload = SS.get("an_payload")
    if not payload:
        st.info("Sube uno o varios CSV para empezar. Con el **par X/Y** de un cojinete "
                "se habilita la órbita y el shaft centerline.")
        st.stop()
    rec = _record_from(payload)

if rec is None or not rec.channels:
    st.warning("No se pudo cargar ningún canal. ¿Formato / captura correctos?")
    st.stop()

_rpm = rec.rpm
cur_asset = cur_asset or rec.name or "CSV"
import datetime as _dt0
_fecha = (rec.name.split()[0] if rec.name and rec.name[:4].isdigit()
          else _dt0.date.today().isoformat())


def _vfile(view_name):
    """Nombre de archivo para descargar una vista: '<Vista> <máquina> <fecha>.png'."""
    return f"{view_name} {cur_asset} {_fecha}.png"


# ---- Setpoints de la máquina (compartido Tabular + Tendencias) ----
_sensors_map, _m_alert, _m_danger, _m_unit = _inst_cfg(cur_iid)


def _sp_tok(s):
    m = re.search(r"(\d+)\s*([XYZ])\s*([A-Z]?)",
                  (s or "").upper().replace(" ", "").replace("_", ""))
    return (m.group(1), m.group(2), m.group(3)) if m else None


_SP_LUT = {}
for _s in (_sensors_map or []):
    _t = _sp_tok(_s.get("csv_match_pattern", ""))
    _a = float(_s.get("alarm", 0.0) or 0.0)
    _d = float(_s.get("danger", 0.0) or 0.0)
    if _t and _a > 0 and _d > 0:
        _SP_LUT[_t] = (_a, _d, str(_s.get("unit_native", "") or ""))


def _ch_setpoint(ch):
    """(alarma, danger, unidad) configurado de un Channel, por token coj/eje/letra."""
    if ch.axis in ("X", "Y", "Z"):
        for _l in ("D", "A", "V", ""):
            if (ch.bearing, ch.axis, _l) in _SP_LUT:
                return _SP_LUT[(ch.bearing, ch.axis, _l)]
        return None
    k = _sp_tok(ch.source or ch.label or "")
    if not k:
        return None
    if k in _SP_LUT:
        return _SP_LUT[k]
    for _l in ("D", "A", "V", ""):
        if (k[0], k[1], _l) in _SP_LUT:
            return _SP_LUT[(k[0], k[1], _l)]
    return None


st.caption(f"Registro · {rec.name}  ·  {len(rec.channels)} canal(es)  ·  "
           f"{len(rec.bearings)} cojinete(s) con par X/Y"
           + (f"  ·  {_rpm:.0f} rpm" if _rpm else ""))


# =========================================================
# 2) VISTAS (segmented control) — dato compartido
# =========================================================
VIEWS = ["Tabular", "Tendencias", "Forma de Onda", "Espectro", "Full Spectrum",
         "Órbita", "Shaft Centerline", "Bode", "Polar", "Waterfall", "Cascada"]
view = st.segmented_control("Vista", VIEWS, default="Tabular",
                            selection_mode="single") or "Tabular"


def _layout(fig, title, h=440):
    fig.update_layout(template="plotly_white", height=h,
                      title=dict(text=title, font=dict(size=15)),
                      margin=dict(l=62, r=24, t=48, b=48),
                      font=dict(family="IBM Plex Sans"), showlegend=False)
    return fig


def _pick_channel(key):
    labels = [c.label for c in rec.channels]
    sel = st.selectbox("Canal", labels, key=key)
    return rec.channels[labels.index(sel)]


def _pick_bearing(key):
    brgs = [b for b in rec.bearings if b.has_xy]
    if not brgs:
        return None
    names = [b.name for b in brgs]
    sel = st.selectbox("Cojinete (par X/Y)", names, key=key)
    return brgs[names.index(sel)]


def _pick_sweep(key):
    if not rec.sweeps:
        return None
    names = [s.label for s in rec.sweeps]
    sel = st.selectbox("Barrido (Bode/Polar de System1)", names, key=key)
    return rec.sweeps[names.index(sel)]


def _sweep_controls(sw, key):
    cA, cB, cC = st.columns(3)
    op = cA.number_input("RPM operación", value=float(round(float(np.nanmax(sw.rpm)))),
                         step=100.0, key=f"{key}_op")
    grp = cB.selectbox("Grupo ISO 20816-2", ["group1", "group2"], index=1, key=f"{key}_g")
    mt = cC.selectbox("Medición", ["Desplazamiento eje (prox)", "Velocidad carcasa (acel/vel)"],
                      key=f"{key}_m")
    mtype = "casing_velocity" if mt.startswith("Velocidad") else "shaft_displacement"
    unit = "mm/s rms" if mtype == "casing_velocity" else "µm pp"
    diag = _sweep_diag(tuple(sw.rpm.tolist()), tuple(sw.amp.tolist()),
                       tuple(sw.phase.tolist()), op, grp, mtype, unit,
                       "Bode" if key == "bode" else "polar")
    s = diag.get("structured", {})
    return op, unit, diag, s, s.get("iso_zone", "—")


_TREND_PAL = ["#12305e", "#e8890c", "#1f9d55", "#8b5cf6", "#0ea5e9", "#dc3545",
              "#84cc16", "#d946ef", "#14b8a6", "#f59e0b"]


def _trend_block():
    if not hist_caps:
        st.info("**Tendencias** — overall vs tiempo. Requiere varias capturas en el "
                "tiempo (modo **Histórico del activo**).")
        return
    # Canales reales del registro (nombres System1), con su captura/eje para trend.
    chans = [c for c in rec.channels]
    labels = [c.label for c in chans]
    cmap = {c.label: c for c in chans}
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        sel = st.multiselect("Canales (uno o varios)", labels,
                             default=labels[:2] if len(labels) >= 2 else labels,
                             key="tr_chs")
    with c2:
        met = st.selectbox("Métrica", ["Overall pp", "RMS", "1X amp"], key="tr_met")
    with c3:
        win = st.selectbox("Ventana", ["Todo", "30 días", "7 días", "24 h"],
                           key="tr_win")
    mk = {"Overall pp": "pp", "RMS": "rms", "1X amp": "1x"}[met]
    wlim = {"24 h": 30, "7 días": 180, "30 días": 800, "Todo": 5000}[win]
    if not sel:
        st.info("Elegí al menos un canal.")
        return
    _ct = tuple((c["point"], c["day"], c["time"], c["key"]) for c in hist_caps
                if c.get("point") and c.get("day") and c.get("time") and c.get("key"))

    fig = go.Figure()
    units_seen, n_pts, lim_lines = set(), 0, {}
    with st.spinner("Calculando tendencias…"):
        for i, lab in enumerate(sel):
            ch = cmap[lab]
            ts, vals, unit = _trend(_ct, ch.cap_point, mk, axis=ch.cap_axis, limit=wlim)
            if not ts:
                continue
            n_pts = max(n_pts, len(ts))
            units_seen.add(unit or ch.unit or "")
            col = _TREND_PAL[i % len(_TREND_PAL)]
            fig.add_trace(go.Scatter(
                x=ts, y=vals, mode="lines", name=lab,
                line=dict(color=col, width=2.4, shape="spline", smoothing=0.6),
                hovertemplate=f"<b>{lab}</b><br>%{{x|%d %b %H:%M}}<br>"
                              f"%{{y:.1f}} {unit or ''}<extra></extra>"))
            # alarma/danger configurados del canal (solo si la métrica es overall/pp)
            if mk == "pp":
                sp = _ch_setpoint(ch)
                if sp:
                    a, d, _su = sp
                    lim_lines.setdefault(round(a, 2), ("#e8890c", f"Alarma {lab}"))
                    lim_lines.setdefault(round(d, 2), ("#dc3545", f"Danger {lab}"))
    if not fig.data:
        st.warning("Sin datos para los canales elegidos.")
        return
    # líneas de alarma/danger (dedup por valor; etiqueta a la IZQUIERDA, dentro del
    # plot, sobre fondo claro → no se corta en el borde derecho).
    for val, (col, _txt) in lim_lines.items():
        _lbl = f"{'Danger' if col == '#dc3545' else 'Alarma'} {val:g}"
        fig.add_hline(y=val, line=dict(color=col, width=1.4, dash="dash"),
                      annotation_text=_lbl, annotation_position="top left",
                      annotation_font=dict(size=10.5, color=col, family="IBM Plex Mono"),
                      annotation_bgcolor="rgba(255,255,255,0.75)")
    _unit = next(iter(units_seen)) if len(units_seen) == 1 else "mixto"
    fig = _layout(fig, f"Tendencia · {met}", h=500)
    fig.update_xaxes(title="", showgrid=True, gridcolor="rgba(120,140,170,.12)",
                     showline=True, linecolor="#d9e1ec",
                     rangeslider=dict(visible=False))
    fig.update_yaxes(title=f"{met} ({_unit or 'a.u.'})", showgrid=True,
                     gridcolor="rgba(120,140,170,.12)", zeroline=False)
    # leyenda ABAJO (no choca con el título) + márgenes amplios + hover unificado
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=64, r=28, t=52, b=72),
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    _download_fig_button(fig, _vfile(f"Tendencia {met}"), key="trend")
    _nota = (f" · unidades {_unit}" if _unit != "mixto"
             else " · unidades mixtas (ojo: distinto tipo de medición)")
    st.caption(f"Ventana: {win} · {n_pts} puntos por canal · líneas punteadas = alarma "
               f"(ámbar) / danger (rojo) de la máquina{_nota}. Overall pp muestra límites "
               "(los setpoints son de overall).")


def _tabular_png(cols, rows, sevs, title="", subtitle=""):
    """Renderiza la tabla como PNG (header navy + zebra + columna de estado de
    color), clase System1. Devuelve bytes PNG para st.download_button."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SEV = {"ok": "#1f9d55", "warn": "#e8890c", "dang": "#dc3545", "info": "#8090a6"}
    headers = ["Estado"] + list(cols)          # col 0 = punto de color
    body = [["●"] + list(r) for r in rows]  # ● coloreado por severidad
    nrow, ncol = len(body), len(headers)
    fig_w = min(1.15 * ncol + 1.2, 16)
    fig_h = 0.5 * nrow + 1.1                    # ajustado al contenido (sin hueco)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])              # el axes llena TODO el frame
    # título compacto arriba (identifica máquina+fecha para el reporte); la tabla
    # llena el resto. Franja del título = fija en pulgadas → sin hueco muerto.
    _top = 1.0
    if title:
        _top = 1.0 - 0.62 / fig_h              # franja del título
        fig.text(0.010, 0.985, title, fontsize=13.5, fontweight="bold",
                 color="#12305e", va="top", ha="left")
        if subtitle:
            fig.text(0.010, 0.985 - 0.30 / fig_h, subtitle, fontsize=9.5,
                     color="#5a6b85", va="top", ha="left")
    # la tabla LLENA el frame (bbox), título ocupa la franja superior
    tbl = ax.table(cellText=body, colLabels=headers, cellLoc="center",
                   bbox=[0.006, 0.0, 0.988, _top])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    for j in range(ncol):                      # header navy
        h = tbl[0, j]
        h.set_facecolor("#12305e")
        h.get_text().set_color("white")
        h.get_text().set_fontweight("bold")
    for i in range(nrow):                       # filas + estado
        sv = SEV.get(sevs[i] if i < len(sevs) else "info", "#8090a6")
        for j in range(ncol):
            cc = tbl[i + 1, j]
            cc.set_facecolor("#f4f7fb" if i % 2 else "#ffffff")
            cc.get_text().set_color("#0b1f3a")
        cc0 = tbl[i + 1, 0]                      # punto de estado coloreado
        cc0.get_text().set_color(sv)
        cc0.get_text().set_fontsize(15)
        tbl[i + 1, 1].get_text().set_fontweight("bold")   # Sensor en negrita
    # anchos de columna PROPORCIONALES al contenido (con bbox las columnas son
    # iguales por defecto → 'Tipo' se apretaría). Estado fijo angosto.
    colw = []
    for j, h in enumerate(headers):
        if j == 0:
            colw.append(2.8)                   # 'Estado' + punto, que no se corte
            continue
        colw.append(max([len(str(h))] + [len(str(body[i][j])) for i in range(nrow)]))
    _tot = float(sum(colw)) or 1.0
    for j in range(ncol):
        w = colw[j] / _tot
        for i in range(nrow + 1):
            tbl[i, j].set_width(w)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _download_png_button(png_bytes, filename, key="wmdl"):
    """Botón de descarga clase mundial: data-URI (baja DIRECTO a Downloads con el
    nombre correcto, sin pasar por el endpoint /media de Streamlit que a veces da
    'Site wasn't available') + lluvia de sandías 🍉 al descargar."""
    import base64
    import html as _h
    import re as _re2

    import streamlit.components.v1 as _components
    b64 = base64.b64encode(png_bytes).decode()
    fn = _h.escape(filename, quote=True)
    eid = "wmdl_" + _re2.sub(r"[^a-zA-Z0-9]", "", str(key))[:24]  # id único por vista
    html = """
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@600&display=swap" rel="stylesheet">
<a id="__EID__" href="data:image/png;base64,__B64__" download="__FN__" onclick="wmMelons()"
   style="display:inline-flex;align-items:center;gap:9px;text-decoration:none;
          font-family:'IBM Plex Sans',system-ui,sans-serif;font-weight:600;font-size:13.5px;
          background:linear-gradient(135deg,#12305e,#274b7d);color:#fff;padding:10px 18px;
          border-radius:10px;border:1px solid #0e2547;box-shadow:0 2px 6px rgba(11,31,58,.18);
          transition:transform .08s ease,box-shadow .15s ease;cursor:pointer">
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e8890c"
       stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 3v12"/><path d="m7 11 5 5 5-5"/><path d="M5 21h14"/></svg>
  Descargar imagen
</a>
<script>
const a=document.getElementById('__EID__');
a.onmouseenter=()=>{a.style.transform='translateY(-1px)';a.style.boxShadow='0 6px 16px rgba(11,31,58,.28)';};
a.onmouseleave=()=>{a.style.transform='none';a.style.boxShadow='0 2px 6px rgba(11,31,58,.18)';};
function wmMelons(){
  for(let i=0;i<30;i++){
    const m=document.createElement('div'); m.textContent='🍉';
    m.style.cssText='position:fixed;left:'+(Math.random()*100)+'vw;top:-24px;'+
      'font-size:'+(15+Math.random()*20)+'px;z-index:99999;pointer-events:none;'+
      'transition:transform '+(1.3+Math.random()*1.1)+'s ease-in,opacity .5s ease-out';
    document.body.appendChild(m);
    requestAnimationFrame(()=>{m.style.transform='translateY(130vh) rotate('+(Math.random()*720-360)+'deg)';});
    setTimeout(()=>{m.style.opacity='0';},1400);
    setTimeout(()=>{m.remove();},2600);
  }
}
</script>
"""
    html = html.replace("__EID__", eid).replace("__B64__", b64).replace("__FN__", fn)
    _components.html(html, height=120)


@st.cache_data(show_spinner=False)
def _fig_png_cached(fig_json):
    import plotly.io as _pio
    return _pio.from_json(fig_json).to_image(format="png", scale=2)


def _download_fig_button(fig, filename, key):
    """Descarga una figura Plotly como PNG (kaleido) con el mismo botón+sandías.
    Cacheado por contenido → no re-renderiza en cada interacción (kaleido es lento)."""
    try:
        import plotly.io as _pio
        png = _fig_png_cached(_pio.to_json(fig))
    except Exception:  # noqa: BLE001
        st.caption("Descarga PNG no disponible (kaleido).")
        return
    _download_png_button(png, filename, key=key)


# ====================== DISPATCH DE VISTAS (una sola cadena) ======================
if view == "Tabular":
    from core.ui_industrial import dot, html_table
    grp = st.selectbox("Grupo ISO 20816-2", ["group2", "group1"], index=0, key="tab_grp",
                       help="group2 = eje montado (turbogeneradores, caso típico). "
                            "Define los límites de zona A/B/C/D.")

    def _meas(ch):
        """(tipo_es, measurement_type_iso|None, unidad, convención_amp)."""
        u = (ch.unit or "").lower()
        if "mm/s" in u or "mm / s" in u:
            return "Velocidad", "casing_velocity", ch.unit or "mm/s rms", "rms"
        if u.strip() in ("g", "g pk", "g rms") or "m/s2" in u or "m/s²" in u:
            return "Aceleración", None, ch.unit or "g", "rms"
        return "Desplazamiento", "shaft_displacement", ch.unit or "µm pp", "pp"

    def _conv(vv, mode):
        return float(np.ptp(vv)) if mode == "pp" else float(np.std(vv))

    def _ukind(u):
        """Familia+unidad base para comparar/convertir setpoints con seguridad."""
        u = (u or "").lower()
        if "mil" in u:
            return "disp", "mil"
        if "µm" in u or "um" in u or "μm" in u:
            return "disp", "um"
        if "mm/s" in u or "mm / s" in u:
            return "vel", "mm/s"
        if "in/s" in u or "ips" in u:
            return "vel", "in/s"
        if u.strip() in ("g", "g pk", "g rms") or "m/s2" in u or "m/s²" in u:
            return "accel", "g"
        return "", ""

    _DISP = {"um": 1.0, "mil": 25.4}       # a µm
    _VEL = {"mm/s": 1.0, "in/s": 25.4}     # a mm/s

    def _to(value, from_u, to_u):
        """Convierte value de from_u a to_u si misma familia; None si incompatible."""
        fk, fb = _ukind(from_u)
        tk, tb = _ukind(to_u)
        if not fk or not tk or fk != tk:
            return None
        if fk == "disp":
            return value * _DISP.get(fb, 1.0) / _DISP.get(tb, 1.0)
        if fk == "vel":
            return value * _VEL.get(fb, 1.0) / _VEL.get(tb, 1.0)
        return value if fb == tb else None

    # Setpoints configurados en la máquina (PRIORIDAD sobre la norma ISO).
    sensors_map, m_alert, m_danger, m_unit = _inst_cfg(cur_iid)
    import re as _re
    _cfg_used = False

    def _tok(s):
        """Token cojinete+eje(+letra) de un texto: '1XD DE turbina'→('1','X','D'),
        'GB_4XA'→('4','X','A'), 'BRG1 · X'→None (no tiene eje pegado al nº)."""
        m = _re.search(r"(\d+)\s*([XYZ])\s*([A-Z]?)",
                       (s or "").upper().replace(" ", "").replace("_", ""))
        return (m.group(1), m.group(2), m.group(3)) if m else None

    # LUT de setpoints por-sensor, por token (cojinete, eje, letra) → (alarma, danger, unidad).
    _sp_lut = {}
    for _s in (sensors_map or []):
        _t = _tok(_s.get("csv_match_pattern", ""))
        _a = float(_s.get("alarm", 0.0) or 0.0)
        _d = float(_s.get("danger", 0.0) or 0.0)
        if _t and _a > 0 and _d > 0:
            _sp_lut[_t] = (_a, _d, str(_s.get("unit_native", "") or ""))

    def _sensor_sp(c):
        """Setpoint configurado del canal: match por (cojinete, eje[, letra])."""
        if c.axis in ("X", "Y", "Z"):                 # cojinete (despl): letra = D
            for _l in ("D", "A", "V", ""):
                if (c.bearing, c.axis, _l) in _sp_lut:
                    return _sp_lut[(c.bearing, c.axis, _l)]
            return None
        k = _tok(c.source or c.label or "")           # gearbox canal único
        if k is None:
            return None
        if k in _sp_lut:
            return _sp_lut[k]
        for _l in ("D", "A", "V", ""):                # sin letra exacta, cualquiera
            if (k[0], k[1], _l) in _sp_lut:
                return _sp_lut[(k[0], k[1], _l)]
        return None

    rows = []
    img_rows = []            # filas planas (sin HTML) para la imagen PNG
    sev_list = []            # severidad por fila (para colorear la imagen)
    n_alarm = n_dang = 0
    for c in rec.channels:
        vv = c.v[np.isfinite(c.v)]
        if not vv.size:
            continue
        tipo, mtype, unit, mode = _meas(c)
        overall = _conv(vv, mode)
        # 1X / 2X por FFT (misma convención que la directa: pp = 2×pico, rms = pico/√2)
        a1 = a2 = ph1 = ph2 = None
        if c.fs_hz and c.rpm:
            w = vv - np.mean(vv)
            win = np.hanning(w.size)
            spc = np.fft.rfft(w * win)
            am = np.abs(spc) * 2.0 / np.sum(win)   # ganancia coherente del Hanning
            frq = np.fft.rfftfreq(w.size, 1.0 / c.fs_hz)
            f1 = c.rpm / 60.0
            # am = cero-a-pico → pp = ×2 (desplazamiento) · rms = ÷√2 (vel/acel)
            kfac = 2.0 if mode == "pp" else (1.0 / np.sqrt(2.0))

            def _pk(fh):
                k = int(np.argmin(np.abs(frq - fh)))
                lo, hi = max(0, k - 2), min(am.size, k + 3)
                kk = lo + int(np.argmax(am[lo:hi])) if hi > lo else k
                return float(am[kk]) * kfac, float((np.degrees(np.angle(spc[kk])) + 360) % 360)
            a1, ph1 = _pk(f1)
            a2, ph2 = _pk(2 * f1)
        # ---- Alarma / Danger: PRIORIDAD a lo configurado en la máquina ----
        # Todo se compara en la unidad del canal (`unit`); los setpoints/ISO se
        # convierten a esa unidad. Si son incompatibles (ej. vel vs despl) se omiten.
        alarm = danger = None
        src = ""                         # "sensor" | "máquina" | "ISO"
        # 1) Setpoint por-sensor (sensor map de la instancia, match por token)
        sp = _sensor_sp(c)
        if sp is not None:
            sa, sd, su = sp
            ac, dc = _to(sa, su, unit), _to(sd, su, unit)
            if ac is not None and dc is not None:
                alarm, danger, src = ac, dc, "sensor"
        # 2) Setpoint a nivel máquina
        if alarm is None and m_alert > 0 and m_danger > 0:
            ac, dc = _to(m_alert, m_unit, unit), _to(m_danger, m_unit, unit)
            if ac is not None and dc is not None:
                alarm, danger, src = ac, dc, "máquina"
        # 3) Norma ISO 20816-2 (fallback) — tabla en µm pp (despl) / mm/s (casing)
        if alarm is None and mtype and c.rpm:
            try:
                from core.rotordynamics import iso_20816_2_zone
                z = iso_20816_2_zone(overall, measurement_type=mtype, machine_group=grp,
                                     operating_speed_rpm=c.rpm)
                _bu = "µm" if mtype == "shaft_displacement" else "mm/s"
                ac, dc = _to(z.boundary_BC, _bu, unit), _to(z.boundary_CD, _bu, unit)
                if ac is None:                       # unidad del canal no estándar → usa cruda
                    ac, dc = z.boundary_BC, z.boundary_CD
                alarm, danger, src = ac, dc, "ISO"
            except Exception:  # noqa: BLE001
                pass
        # Estado por umbrales (misma lógica sea config o norma)
        sev = "info"
        if alarm is not None and danger is not None:
            _cfg_used = _cfg_used or src in ("sensor", "máquina")
            sev = "dang" if overall >= danger else ("warn" if overall >= alarm else "ok")
            n_alarm += sev == "warn"
            n_dang += sev == "dang"
        _f = lambda x: "—" if x is None else f"{x:.1f}"  # noqa: E731
        _p = lambda x: "—" if x is None else f"{x:.0f}°"  # noqa: E731
        rows.append([dot(sev), c.label, tipo, unit, _f(alarm), _f(danger),
                     _f(overall), _f(a1), _p(ph1), _f(a2), _p(ph2)])
        img_rows.append([c.label, tipo, unit, _f(alarm), _f(danger),
                         _f(overall), _f(a1), _p(ph1), _f(a2), _p(ph2)])
        sev_list.append(sev)
    _cols = ["", "Sensor", "Tipo", "Unidades", "Alarma", "Danger", "Directa",
             "1X", "1X ∠", "2X", "2X ∠"]
    if rows:
        html_table(_cols, rows, raw_cols=[0])
        # ---- Descargar la tabla como IMAGEN (un botón) ----
        _png = _tabular_png(_cols[1:], img_rows, sev_list,
                            title=f"Tabular List · {cur_asset}",
                            subtitle=f"{rec.name}  ·  {rec.rpm or 0:.0f} rpm  ·  "
                                     f"{'setpoints de la máquina' if _cfg_used else 'norma ISO 20816-2'}")
        _download_png_button(_png, f"Tabular list {cur_asset} {_fecha}.png", key="tabular")
        _origen = ("setpoints configurados en la máquina" if _cfg_used
                   else f"norma ISO 20816-2 {grp} (sin setpoints configurados)")
        st.caption(f"{len(rows)} sensor(es) · {rec.rpm or 0:.0f} rpm · límites de **{_origen}** "
                   f"(prioridad: sensor → máquina → norma). Punto verde = normal, "
                   f"ámbar = alarma, rojo = danger. "
                   + (f"{n_alarm} en alarma · {n_dang} en danger."
                      if (n_alarm or n_dang) else "Todos normales."))
    else:
        st.info("Sin canales para tabular.")

elif view == "Tendencias":
    _trend_block()

elif view == "Forma de Onda":
    ch = _pick_channel("wf_ch")
    fig = go.Figure(go.Scatter(x=ch.t, y=ch.v, mode="lines",
                               line=dict(color=_NAVY, width=1.3)))
    fig.update_xaxes(title="Tiempo (s)")
    fig.update_yaxes(title=f"{ch.label} ({ch.unit or 'a.u.'})")
    st.plotly_chart(_layout(fig, f"Forma de onda · {ch.label}"), use_container_width=True)
    _download_fig_button(fig, _vfile(f"Forma de onda {ch.label}"), key="wf")
    # Métricas + diagnóstico (reutiliza core.waveform_metrics / waveform_diagnostics)
    from core.waveform_metrics import compute_waveform_metrics
    mx = compute_waveform_metrics(ch.v)
    if mx:
        u = ch.unit or ""
        cells = [("RMS", f"{mx.get('rms',0):.2f} {u}"), ("Pico", f"{mx.get('peak',0):.2f} {u}"),
                 ("Pico-pico", f"{mx.get('peak_to_peak',0):.2f} {u}"),
                 ("Crest", f"{mx.get('crest_factor',0):.2f}"),
                 ("Kurtosis", f"{mx.get('kurtosis',0):.2f}")]
        cols = st.columns(len(cells))
        for col, (lbl, val) in zip(cols, cells):
            col.markdown(f"<div style='background:#f7f9fc;border:1px solid #e2e8f2;border-radius:8px;"
                         f"padding:8px 10px'><div style='font:600 10px IBM Plex Mono;color:#8090a6;"
                         f"text-transform:uppercase;letter-spacing:.08em'>{lbl}</div>"
                         f"<div style='font:700 17px IBM Plex Sans;color:#0b1f3a'>{val}</div></div>",
                         unsafe_allow_html=True)
        try:
            from core.waveform_diagnostics import build_waveform_diagnostics_rotordyn
            dW = build_waveform_diagnostics_rotordyn(
                time_s=ch.t, amplitude=ch.v, metrics=mx, machine_label=rec.name,
                point_label=ch.label, amplitude_unit=u)
            _sev = str(dW.get("severity_global", "")).upper()
            _c = {"DANGER": "#dc3545", "CRITICAL": "#dc3545", "WARNING": "#e8890c",
                  "CAUTION": "#84cc16"}.get(_sev, "#1f9d55")
            st.markdown(
                f"<div style='border-left:4px solid {_c};background:#f8fafc;border:1px solid #e2e8f2;"
                f"border-radius:8px;padding:11px 15px;margin-top:10px'>"
                f"<b style='font-size:13px'>{dW.get('headline','')}</b>"
                f"<div style='color:#3a4c66;font-size:12.5px;margin-top:5px'>{dW.get('detail','')}</div>"
                f"<div style='color:#0b1f3a;font-size:12.5px;margin-top:4px'><b>Acción:</b> "
                f"{dW.get('action','')}</div></div>", unsafe_allow_html=True)
        except Exception:  # noqa: BLE001
            pass

elif view == "Espectro":
    ch = _pick_channel("sp_ch")
    v = ch.v[np.isfinite(ch.v)]
    fs = ch.fs_hz
    if v.size < 8 or not fs:
        st.warning("Espectro: faltan muestras o no se pudo estimar la frecuencia de muestreo.")
    else:
        w = v - np.mean(v)
        amp = np.abs(np.fft.rfft(w * np.hanning(w.size))) * (2.0 / w.size)
        fr = np.fft.rfftfreq(w.size, d=1.0 / fs)
        fig = go.Figure(go.Scatter(x=fr, y=amp, mode="lines",
                                   line=dict(color=_NAVY, width=1.1)))
        if ch.rpm:
            f1 = ch.rpm / 60.0
            for k, lab in [(1, "1X"), (2, "2X"), (3, "3X")]:
                if k * f1 < fr[-1]:
                    fig.add_vline(x=k * f1, line=dict(color=_AMBER, width=1, dash="dot"),
                                  annotation_text=lab, annotation_position="top")
        fig.update_xaxes(title="Frecuencia (Hz)")
        fig.update_yaxes(title=f"Amplitud ({ch.unit or 'a.u.'})")
        st.plotly_chart(_layout(fig, f"Espectro (FFT) · {ch.label}"), use_container_width=True)
        _download_fig_button(fig, _vfile(f"Espectro {ch.label}"), key="sp")
        # Diagnóstico por norma (reutiliza core.spectrum_diagnostics)
        if ch.rpm:
            from core.spectrum_diagnostics import evaluate_spectrum_diagnostic
            f1 = ch.rpm / 60.0

            def _amp_at(fhz):
                k = int(np.argmin(np.abs(fr - fhz)))
                lo, hi = max(0, k - 2), min(amp.size, k + 3)
                return float(np.max(amp[lo:hi])) if hi > lo else 0.0
            harms = [{"order": k, "amp_peak": _amp_at(k * f1),
                      "freq_cpm": k * f1 * 60.0} for k in range(1, 7)]
            dk = int(np.argmax(amp))
            diagS = evaluate_spectrum_diagnostic(
                one_x_amp=harms[0]["amp_peak"], harmonics=harms,
                overall_spec_rms=float(np.sqrt(np.mean(w ** 2))),
                dominant_peak_freq_cpm=float(fr[dk] * 60.0),
                dominant_peak_amp=float(amp[dk]), rpm=ch.rpm)
            _col = diagS.get("color", "#8090a6")
            st.markdown(
                f"<div style='border-left:4px solid {_col};background:#f8fafc;border:1px solid "
                f"#e2e8f2;border-radius:8px;padding:11px 15px;margin-top:8px'>"
                f"<span style='background:{_col};color:#fff;border-radius:6px;padding:2px 9px;"
                f"font:700 11px IBM Plex Mono'>{diagS.get('status','')}</span>  "
                f"<b style='font-size:13px'>{diagS.get('headline','')}</b>"
                f"<div style='color:#3a4c66;font-size:12.5px;margin-top:5px'>{diagS.get('narrative','')}</div>"
                f"</div>", unsafe_allow_html=True)

elif view == "Full Spectrum":
    b = _pick_bearing("fs_b")
    if b is None:
        st.warning("Full Spectrum: necesita el **par X/Y** de un cojinete (espectro "
                   "direccional · precesión forward/backward · supera a System1).")
    else:
        n = min(b.x.v.size, b.y.v.size)
        xv = b.x.v[:n] - np.nanmean(b.x.v[:n])
        yv = b.y.v[:n] - np.nanmean(b.y.v[:n])
        fs = b.x.fs_hz or b.y.fs_hz
        if n < 16 or not fs:
            st.warning("Full Spectrum: faltan muestras o frecuencia de muestreo.")
        else:
            # z = X + iY → FFT complejo → +f = precesión forward, −f = backward
            win = np.hanning(n)
            z = (xv * win) + 1j * (yv * win)
            sp = np.fft.fftshift(np.fft.fft(z)) * (2.0 / n)
            fr = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
            amp = np.abs(sp)
            fwd = fr >= 0
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fr[fwd], y=amp[fwd], mode="lines",
                          line=dict(color=_NAVY, width=1.2), name="Forward (+)"))
            fig.add_trace(go.Scatter(x=fr[~fwd], y=amp[~fwd], mode="lines",
                          line=dict(color=_AMBER, width=1.2), name="Backward (−)"))
            if b.x.rpm:
                f1 = b.x.rpm / 60.0
                for k, lab in [(1, "+1X"), (2, "+2X"), (-1, "−1X")]:
                    if abs(k) * f1 < fr[-1]:
                        fig.add_vline(x=k * f1, line=dict(color="#8090a6", width=1, dash="dot"),
                                      annotation_text=lab, annotation_position="top")
            fig.update_xaxes(title="Frecuencia direccional (Hz) · + forward / − backward")
            fig.update_yaxes(title=f"Amplitud ({b.x.unit or 'µm'})")
            st.plotly_chart(_layout(fig, f"Full Spectrum · {b.name}", h=500),
                            use_container_width=True)
            _download_fig_button(fig, _vfile(f"Full Spectrum {b.name}"), key="fs")
            # dominancia forward vs backward
            ef = float(np.sum(amp[fwd] ** 2))
            eb = float(np.sum(amp[~fwd] ** 2))
            tot = ef + eb or 1.0
            dom = "FORWARD" if ef >= eb else "BACKWARD"
            dc = "#1f9d55" if dom == "FORWARD" else "#e8890c"
            st.markdown(
                f"<div style='border-left:4px solid {dc};background:#f8fafc;border:1px solid "
                f"#e2e8f2;border-radius:8px;padding:11px 15px;margin-top:8px'>"
                f"<span style='background:{dc};color:#fff;border-radius:6px;padding:2px 9px;"
                f"font:700 11px IBM Plex Mono'>{dom}</span>  "
                f"<b>Precesión {ef/tot*100:.0f}% fwd / {eb/tot*100:.0f}% bwd.</b>"
                f"<div style='color:#3a4c66;font-size:12.5px;margin-top:5px'>Backward fuerte "
                f"(&gt;30%) sugiere roce (rub), anisotropía de soporte o inestabilidad. "
                f"El espectro direccional separa fuerzas co- y contra-rotatorias (no lo da "
                f"System1 estándar).</div></div>", unsafe_allow_html=True)

elif view == "Órbita":
    b = _pick_bearing("orb_b")
    if b is None:
        st.warning("Órbita: necesita el **par X/Y** de un cojinete. Sube ambos CSV "
                   "(ej. 1xd.csv + 1yd.csv).")
    else:
        fmode = st.selectbox("Filtro", ["Direct", "1X", "2X"], index=0, key="orb_f")
        xv, yv, prec, trav, sub = b.x.v, b.y.v, "", "", ""

        class _Sig:
            def __init__(self, t, x): self.time = t; self.x = x; self.metadata = {}
        try:
            from core.orbit import compute_orbit
            res = compute_orbit(_Sig(b.x.t, b.x.v), _Sig(b.y.t, b.y.v),
                                filter_mode=fmode, samples_per_rev=b.x.spr or None,
                                rpm_override=b.x.rpm)
            xv = np.asarray(res.plot_x, float); yv = np.asarray(res.plot_y, float)
            prec = str(getattr(res, "precession", "") or "")
            trav = str(getattr(res, "traversal", "") or "")
            sp = getattr(res, "start_point", None)
        except Exception:  # noqa: BLE001
            n = min(b.x.v.size, b.y.v.size); xv, yv = b.x.v[:n], b.y.v[:n]; sp = None
        fig = go.Figure(go.Scatter(x=xv, y=yv, mode="lines",
                                   line=dict(color=_NAVY, width=1.4)))
        _sx = sp[0] if (sp is not None and len(sp) == 2) else (xv[0] if xv.size else 0)
        _sy = sp[1] if (sp is not None and len(sp) == 2) else (yv[0] if yv.size else 0)
        fig.add_trace(go.Scatter(x=[_sx], y=[_sy], mode="markers",
                      marker=dict(color="#1f9d55", size=10),
                      name="keyphasor"))
        fig.update_xaxes(title=f"{b.x.label} ({b.x.unit or 'µm'})",
                         scaleanchor="y", scaleratio=1, zeroline=True)
        fig.update_yaxes(title=f"{b.y.label} ({b.y.unit or 'µm'})", zeroline=True)
        _ttl = f"Órbita · {b.name} · {fmode}"
        if prec:
            _ttl += f"  ·  precesión {prec}"
        st.plotly_chart(_layout(fig, _ttl, h=500), use_container_width=True)
        _download_fig_button(fig, _vfile(f"Orbita {b.name}"), key="orb")
        _bits = []
        if prec:
            _bits.append(f"Precesión: **{prec}**")
        if trav:
            _bits.append(f"sentido: {trav}")
        if b.x.rpm:
            _bits.append(f"{b.x.rpm:.0f} rpm")
        st.caption(" · ".join(_bits) + ("  ·  punto verde = marca keyphasor." if _bits
                   else "Punto verde = keyphasor."))

elif view == "Shaft Centerline":
    b = _pick_bearing("sc_b")
    if b is None:
        st.warning("Shaft Centerline: necesita el **par X/Y** (gap DC de los proximitores).")
    else:
        gx, gy = float(np.nanmean(b.x.v)), float(np.nanmean(b.y.v))
        gmag = float(np.hypot(gx, gy))
        clr = st.number_input("Claro radial del cojinete (µm)",
                              value=float(round(max(gmag * 2.0, 50.0))), step=10.0,
                              help="Del montaje / API 670. Define la zona de operación segura.")
        from core.diagnostics import (boundary_utilization_pct,  # noqa: PLC0415
                                      get_semaforo_status, remaining_margin_pct)
        util = boundary_utilization_pct(gx, gy, 0.0, 0.0, clr, clr)
        status, color = get_semaforo_status(util)
        fig = go.Figure()
        th = np.linspace(0, 2 * np.pi, 160)
        # zonas semáforo (60% verde / 85% ámbar / resto rojo) como anillos
        for frac, c in [(1.0, "#dc3545"), (0.85, "#e8890c"), (0.60, "#1f9d55")]:
            fig.add_trace(go.Scatter(x=clr * frac * np.cos(th), y=clr * frac * np.sin(th),
                          mode="lines", line=dict(color=c, width=1, dash="dot"),
                          hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=b.x.v, y=b.y.v, mode="lines",
                      line=dict(color="#cbd5e1", width=0.8), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[gx], y=[gy], mode="markers+text",
                      marker=dict(color=color, size=14, line=dict(color="#fff", width=1.5)),
                      text=["eje"], textposition="top center"))
        fig.update_xaxes(title=f"{b.x.label} (µm)", scaleanchor="y", scaleratio=1, zeroline=True,
                         range=[-clr * 1.1, clr * 1.1])
        fig.update_yaxes(title=f"{b.y.label} (µm)", zeroline=True, range=[-clr * 1.1, clr * 1.1])
        st.plotly_chart(_layout(fig, f"Shaft Centerline · {b.name}", h=500),
                        use_container_width=True)
        _download_fig_button(fig, _vfile(f"Shaft Centerline {b.name}"), key="sc")
        st.markdown(
            f"<div style='border-left:4px solid {color};background:#f8fafc;border:1px solid #e2e8f2;"
            f"border-radius:8px;padding:11px 15px;margin-top:8px'>"
            f"<span style='background:{color};color:#fff;border-radius:6px;padding:2px 9px;"
            f"font:700 11px IBM Plex Mono'>{status}</span>  "
            f"<b>Uso del claro: {util:.0f}%</b> · margen {remaining_margin_pct(util):.0f}% · "
            f"gap DC X={gx:.1f} Y={gy:.1f} µm (|gap|={gmag:.1f}). "
            f"Anillos: verde 60% · ámbar 85% · rojo = borde del claro (API 670).</div>",
            unsafe_allow_html=True)

elif view == "Bode":
    sw = _pick_sweep("bode_sw")
    if sw is None:
        st.warning("Bode: sube el **CSV de Bode/Polar de System1** (speed/rpm + "
                   "amplitude + phase).")
    else:
        op, unit, diag, s, zone = _sweep_controls(sw, "bode")
        zc = _ISO.get(zone, "#8090a6")
        ab, bc, cd = (s.get("iso_boundary_AB"), s.get("iso_boundary_BC"),
                      s.get("iso_boundary_CD"))
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                            row_heights=[0.56, 0.44])
        amax = max(float(np.nanmax(sw.amp)), cd or 0) * 1.12
        for lo, hi, z in [(0.0, ab, "A"), (ab, bc, "B"), (bc, cd, "C"), (cd, amax, "D")]:
            if lo is None or hi is None:
                continue
            fig.add_hrect(y0=lo, y1=hi, line_width=0, fillcolor=_ISO[z],
                          opacity=0.08, row=1, col=1)
        fig.add_trace(go.Scatter(x=sw.rpm, y=sw.amp, mode="lines+markers",
                                 line=dict(color=_NAVY, width=1.9),
                                 marker=dict(size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=sw.rpm, y=sw.phase, mode="lines+markers",
                                 line=dict(color=_AMBER, width=1.9),
                                 marker=dict(size=4)), row=2, col=1)
        for c in s.get("criticals", []):
            col = "#dc3545" if c.get("kind") == "confirmed" else "#8090a6"
            fig.add_vline(x=c["rpm"], line=dict(color=col, width=1.2, dash="dash"),
                          annotation_text=f"fc {c['rpm']:.0f} · Q={c.get('q_factor',0):.1f}",
                          annotation_position="top", annotation_font_size=10,
                          row=1, col=1)
            fig.add_vline(x=c["rpm"], line=dict(color=col, width=1, dash="dash"),
                          row=2, col=1)
        fig.add_vline(x=op, line=dict(color=_NAVY, width=1, dash="dot"),
                      annotation_text="op", annotation_position="bottom", row=1, col=1)
        fig.update_yaxes(title=f"Amplitud ({unit})", row=1, col=1)
        fig.update_yaxes(title="Fase (°)", row=2, col=1)
        fig.update_xaxes(title="Velocidad (rpm)", row=2, col=1)
        st.plotly_chart(_layout(fig, f"Bode · {sw.label}", h=560), use_container_width=True)
        _download_fig_button(fig, _vfile(f"Bode {sw.label}"), key="bode")
        _diag_panel(diag, zone, zc)

elif view == "Polar":
    sw = _pick_sweep("polar_sw")
    if sw is None:
        st.warning("Polar: sube el **CSV de Bode/Polar de System1** (speed/rpm + "
                   "amplitude + phase).")
    else:
        op, unit, diag, s, zone = _sweep_controls(sw, "polar")
        zc = _ISO.get(zone, "#8090a6")
        fig = go.Figure()
        th = list(range(0, 365, 5))
        for z, b in [("A", s.get("iso_boundary_AB")), ("B", s.get("iso_boundary_BC")),
                     ("C", s.get("iso_boundary_CD"))]:
            if b:
                fig.add_trace(go.Scatterpolar(r=[b] * len(th), theta=th, mode="lines",
                              line=dict(color=_ISO[z], width=1, dash="dot"),
                              hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatterpolar(
            r=sw.amp, theta=sw.phase, mode="lines+markers",
            line=dict(color=_NAVY, width=1.8),
            marker=dict(size=5, color=sw.rpm, colorscale="Blues", showscale=True,
                        colorbar=dict(title="rpm", thickness=12)),
            customdata=sw.rpm, hovertemplate="rpm %{customdata:.0f}<br>amp %{r:.1f}<br>fase %{theta:.0f}°"))
        for c in s.get("criticals", []):
            fig.add_trace(go.Scatterpolar(r=[c["amp_peak"]], theta=[c.get("phase_change_deg", 0)],
                          mode="markers", marker=dict(color="#dc3545", size=10, symbol="x"),
                          hoverinfo="skip", showlegend=False))
        fig.update_layout(template="plotly_white", height=560,
                          title=dict(text=f"Polar · {sw.label}", font=dict(size=15)),
                          font=dict(family="IBM Plex Sans"),
                          polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
                          margin=dict(l=40, r=40, t=56, b=40))
        st.plotly_chart(fig, use_container_width=True)
        _download_fig_button(fig, _vfile(f"Polar {sw.label}"), key="polar")
        st.caption(f"Radio = amplitud ({unit}) · ángulo = fase · color = rpm · "
                   "anillos punteados = límites ISO 20816-2 (A/B/C).")
        _diag_panel(diag, zone, zc)

elif view == "Waterfall":
    # Espectros apilados en el TIEMPO (una traza por sesión del histórico).
    if not hist_caps:
        st.info("**Waterfall** — espectros apilados en el tiempo. Requiere varias "
                "capturas (modo **Histórico del activo**). Separa picos síncronos de "
                "los que cambian con la condición.")
    else:
        points = sorted({c.get("point", "") for c in hist_caps if c.get("point")})
        pt = st.selectbox("Punto", points, key="wf_pt")
        _ct = tuple((c["point"], c["day"], c["time"], c["key"]) for c in hist_caps
                    if c.get("point") == pt and c.get("day") and c.get("time") and c.get("key"))
        with st.spinner("Calculando espectros por sesión…"):
            spectra = _spectra(_ct, limit=24)
        if not spectra:
            st.warning(f"Sin espectros para {pt}.")
        else:
            fig = go.Figure()
            step = (max(float(np.max(a)) for _, _, a in spectra) or 1.0) * 0.6
            for i, (stamp, fr, amp) in enumerate(spectra):
                fig.add_trace(go.Scatter(
                    x=fr, y=amp + i * step, mode="lines",
                    line=dict(color=_NAVY if i % 2 == 0 else _STEEL, width=0.9),
                    name=stamp, hovertemplate=f"{stamp}<br>%{{x:.1f}} Hz<br>%{{y:.1f}}"))
            fig.update_xaxes(title="Frecuencia (Hz)")
            fig.update_yaxes(title="Amplitud apilada (offset por sesión)", showticklabels=False)
            st.plotly_chart(_layout(fig, f"Waterfall · {pt} · {len(spectra)} sesiones", h=560),
                            use_container_width=True)
            _download_fig_button(fig, _vfile(f"Waterfall {pt}"), key="wf3d")
            st.caption("Cada traza = una sesión (offset vertical). Picos que crecen con el "
                       "tiempo = degradación; líneas rectas = síncronos estables.")

elif view == "Cascada":
    # Espectros vs RPM (run-up / coast-down). Fuente: NI/ADRE run-up o barrido System1.
    if rec.sweeps:
        st.info("Barrido de System1 cargado — la **Cascada** vive mejor en **Bode/Polar** "
                "(amplitud y fase 1X vs rpm). La cascada por espectros completos vs rpm "
                "llega con el run-up de NI 9229/9234.")
    else:
        st.info("**Cascada** — espectros vs **RPM** en arranque/parada (run-up/coast-down). "
                "Separa resonancias (fijas en Hz) de órdenes (siguen la rpm) y revela las "
                "velocidades críticas. Requiere captura transitoria continua — llega con la "
                "adquisición directa **NI 9229/9234** o un run-up de ADRE 408.")

st.caption("Rotodinámica · ingesta + 11 vistas sobre la misma captura o histórico. "
           "Fuentes: histórico Supabase automático · CSV System1/ADRE · (NI en camino).")
