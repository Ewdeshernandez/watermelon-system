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
_NOBAR = {"displayModeBar": False}   # oculta la modebar de Plotly (barra fea de Streamlit)


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


# Keyphasor por tren: turbina (coj 1,2 + gbx) ~14000 · generador (coj 5,6) ~1800.
_rpm_turb = next((c.rpm for c in rec.channels if c.bearing in ("1", "2") and c.rpm), None)
_rpm_gen = next((c.rpm for c in rec.channels if c.bearing in ("5", "6") and c.rpm), None)
_kph = []
if _rpm_turb:
    _kph.append(f"Keyphasor 1 (turbina) {_rpm_turb:,.0f} rpm")
if _rpm_gen and (not _rpm_turb or abs(_rpm_gen - _rpm_turb) > 50):
    _kph.append(f"Keyphasor 2 (generador) {_rpm_gen:,.0f} rpm")
st.caption(f"Registro · {rec.name}  ·  {len(rec.channels)} canal(es)  ·  "
           f"{len(rec.bearings)} cojinete(s) con par X/Y"
           + ("  ·  " + " · ".join(_kph) if _kph
              else (f"  ·  {_rpm:.0f} rpm" if _rpm else "")))


# =========================================================
# 2) VISTAS (segmented control) — dato compartido
# =========================================================
VIEWS = ["Tabular", "Tendencias", "Forma de Onda", "Espectro", "Full Spectrum",
         "Órbita", "Deflexión del eje", "Shaft Centerline", "Bode", "Polar",
         "Waterfall", "Cascada"]
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
    sel = st.segmented_control("Cojinete (par X/Y)", names, default=names[0], key=key)
    if sel not in names:
        sel = names[0]
    return brgs[names.index(sel)]


def _pick_sweep(key):
    if not rec.sweeps:
        return None
    names = [s.label for s in rec.sweeps]
    sel = st.selectbox("Barrido (Bode/Polar de System1)", names, key=key)
    return rec.sweeps[names.index(sel)]


def _runout_comp(rpm, amp, phase, sr_thr):
    """Compensación de giro lento (runout): resta el vector 1X de baja velocidad
    (runout eléctrico + mecánico/arqueo) de cada punto (Eisenmann p35/64, Bently 5-5).
    El vector de giro lento = promedio de los vectores con rpm ≤ sr_thr (o el de rpm
    mínima). Devuelve amplitud/fase compensadas + el vector restado."""
    rpm = np.asarray(rpm, float); amp = np.asarray(amp, float)
    ph = np.radians(np.asarray(phase, float))
    x = amp * np.cos(ph); y = amp * np.sin(ph)
    m = np.isfinite(rpm) & (rpm <= sr_thr)
    if m.sum() >= 1:
        sx, sy, n = float(np.nanmean(x[m])), float(np.nanmean(y[m])), int(m.sum())
    else:
        i = int(np.nanargmin(rpm)); sx, sy, n = float(x[i]), float(y[i]), 1
    xc, yc = x - sx, y - sy
    amp_c = np.hypot(xc, yc)
    phase_c = np.degrees(np.arctan2(yc, xc)) % 360.0
    srmag = float(np.hypot(sx, sy)); srang = float(np.degrees(np.arctan2(sy, sx)) % 360.0)
    return amp_c, phase_c, srmag, srang, n


def _sweep_controls(sw, key):
    cA, cB, cC = st.columns(3)
    op = cA.number_input("RPM operación", value=float(round(float(np.nanmax(sw.rpm)))),
                         step=100.0, key=f"{key}_op")
    grp = cB.selectbox("Grupo ISO 20816-2", ["group1", "group2"], index=1, key=f"{key}_g")
    mt = cC.selectbox("Medición", ["Desplazamiento eje (prox)", "Velocidad carcasa (acel/vel)"],
                      key=f"{key}_m")
    mtype = "casing_velocity" if mt.startswith("Velocidad") else "shaft_displacement"
    unit = "mm/s rms" if mtype == "casing_velocity" else "µm pp"
    rpm = np.asarray(sw.rpm, float)
    amp_c = np.asarray(sw.amp, float); phase_c = np.asarray(sw.phase, float)
    cD, cE = st.columns([1, 2], vertical_alignment="center")
    comp = cD.toggle("Compensar giro lento (runout)", value=False, key=f"{key}_rc",
                     help="Resta el vector 1X de giro lento (runout eléctrico + mecánico/"
                          "arqueo) medido a baja velocidad. Sin esto el Bode/Polar puede "
                          "mentir: pico de crítica corrido, amplitud y fase falsas (Eisenmann p64).")
    if comp:
        _smin, _smax = float(np.nanmin(rpm)), float(np.nanmax(rpm))
        sr_thr = cE.slider("Velocidad de giro lento (rpm)",
                           min_value=float(round(_smin)),
                           max_value=float(round(_smin + 0.20 * (_smax - _smin))),
                           value=float(round(_smin + 0.05 * (_smax - _smin))), step=10.0,
                           key=f"{key}_srt")
        amp_c, phase_c, _sm, _sa, _sn = _runout_comp(rpm, sw.amp, sw.phase, sr_thr)
        cE.caption(f"⬤ Vector de giro lento restado: {_sm:.2f} {unit} ∠{_sa:.0f}° "
                   f"({_sn} muestras ≤ {sr_thr:.0f} rpm).")
    diag = _sweep_diag(tuple(rpm.tolist()), tuple(amp_c.tolist()),
                       tuple(phase_c.tolist()), op, grp, mtype, unit,
                       "Bode" if key == "bode" else "polar")
    s = diag.get("structured", {})
    return op, unit, diag, s, s.get("iso_zone", "—"), amp_c, phase_c


_TREND_PAL = ["#12305e", "#e8890c", "#1f9d55", "#8b5cf6", "#0ea5e9", "#dc3545",
              "#84cc16", "#d946ef", "#14b8a6", "#f59e0b"]

# Claros DIAMETRALES de armado (µm) del gas generator SGT-300 — dossier OEM RW166
# (Section 05 Core Engine). Mismo modelo en SGT300A y SGT300B → por nº de cojinete.
# El cojinete de la TURBINA DE POTENCIA queda PENDIENTE (sin valor OEM) → auto.
_SGT300_CLEARANCE_UM = {
    "1": 152.0,   # Inlet bearing  (compresor/frío, muñón ~100 mm): Ø99.980 − Ø99.828
    "2": 317.0,   # CT bearing     (turbina/caliente, muñón ~150 mm): Ø150.006 − Ø149.689
}


def _ecc_status(eps: float):
    """Semáforo por relación de excentricidad ε = |pos| / claro radial (Bently 12-13).
    ε→1 contacto/precarga · ε medio-alto = cargado normal · ε→0 centrado = riesgo de
    inestabilidad (oil whirl/whip, Bently 14; Eisenmann p267)."""
    if eps >= 0.95:
        return "CONTACTO / PRECARGA SEVERA", "#dc3545"
    if eps >= 0.50:
        return "CARGADO NORMAL", "#1f9d55"
    if eps >= 0.30:
        return "POCO CARGADO", "#e8890c"
    return "MUY CENTRADO · RIESGO WHIRL", "#dc3545"


def _gap_to_vh(dgx: np.ndarray, dgy: np.ndarray):
    """Convierte cambios de gap de las sondas a desplazamiento en el marco de máquina
    (H derecha+, V arriba+). Geometría real SGT-300 (verificada al destapar, conv.
    Bently 0°=arriba): sondas montadas ABAJO, X en lado R a 135°, Y en lado L a 135°
    → û_X=(+a,−a), û_Y=(−a,−a), a=cos45°. Con Δgap = −(d·û_sonda):
    V = (Δgx+Δgy)/(2a) ; H = (Δgy−Δgx)/(2a)."""
    a = np.sqrt(0.5)
    V = (dgx + dgy) / (2.0 * a)
    H = (dgy - dgx) / (2.0 * a)
    return H, V


def _add_probe_glyphs(fig, R, ax_deg, ay_deg, lbl_x, lbl_y, with_cone=True, with_top=True,
                      row=None, col=None):
    """Dibuja las sondas de proximidad tipo dibujo técnico (cuerpo + rosca + cara) en su
    ángulo real, FUERA del radio R, + cono de visión hacia el eje + marca TOPE 0°.
    ax_deg = grados del TOP hacia el lado R (sonda X); ay_deg = grados del TOP hacia el
    lado L (sonda Y). Marco: 0°=arriba, +H=derecha, +V=arriba. row/col → celda subplot."""
    if R <= 0:
        return
    _rc = {} if row is None else dict(row=row, col=col)
    for _lbl, _deg, _s in [(lbl_x, float(ax_deg), 1.0), (lbl_y, float(ay_deg), -1.0)]:
        _th = np.radians(_deg)
        _ux, _uy = _s * np.sin(_th), np.cos(_th)
        if with_cone:
            _ang = np.arctan2(_uy, _ux)
            _aa = np.linspace(_ang - np.radians(13), _ang + np.radians(13), 18)
            fig.add_trace(go.Scatter(
                x=np.concatenate([[0], R * 0.95 * np.cos(_aa), [0]]),
                y=np.concatenate([[0], R * 0.95 * np.sin(_aa), [0]]),
                fill="toself", mode="lines", line=dict(color="rgba(18,48,94,0)"),
                fillcolor="rgba(18,48,94,0.05)", hoverinfo="skip", showlegend=False), **_rc)
        _ri, _rb, _ro = R * 1.12, R * 1.20, R * 1.36
        _perp = np.array([-_uy, _ux])
        fig.add_trace(go.Scatter(x=[_ux * _rb, _ux * _ro], y=[_uy * _rb, _uy * _ro],
                      mode="lines", line=dict(color="#12305e", width=10),
                      hoverinfo="skip", showlegend=False), **_rc)
        for _t in (0.55, 0.70, 0.85):   # rosca
            _cx, _cy = _ux * (_rb + (_ro - _rb) * _t), _uy * (_rb + (_ro - _rb) * _t)
            fig.add_trace(go.Scatter(
                x=[_cx - _perp[0] * R * 0.045, _cx + _perp[0] * R * 0.045],
                y=[_cy - _perp[1] * R * 0.045, _cy + _perp[1] * R * 0.045],
                mode="lines", line=dict(color="#8fa0bf", width=1.5),
                hoverinfo="skip", showlegend=False), **_rc)
        fig.add_trace(go.Scatter(x=[_ux * _ri], y=[_uy * _ri], mode="markers",
                      marker=dict(color="#e8890c", size=12, symbol="circle",
                                  line=dict(color="#12305e", width=2)),
                      hoverinfo="skip", showlegend=False), **_rc)
        _sd = "R" if _s > 0 else "L"
        fig.add_annotation(x=_ux * _ro, y=_uy * _ro, text=f"{_lbl}<br>{_deg:.0f}°{_sd}",
                           showarrow=False, yshift=-16,
                           font=dict(size=10.5, family="IBM Plex Mono", color="#12305e"), **_rc)
    if with_top:
        fig.add_trace(go.Scatter(x=[0], y=[R * 1.12], mode="markers",
                      marker=dict(color="#0f172a", size=7, symbol="triangle-down"),
                      hoverinfo="skip", showlegend=False), **_rc)


def _orbit_data(b, fmode, ang_x, ang_y):
    """Calcula la órbita de un cojinete (método Live Monitoring = idéntico a System1):
    rota a coords de máquina, filtra el orden, arma keyphasor, precesión, forma."""
    px, py = np.asarray(b.x.v, float), np.asarray(b.y.v, float)
    _n = min(px.size, py.size); px, py = px[:_n], py[:_n]
    fs = b.x.fs_hz or b.y.fs_hz; spr = b.x.spr or b.y.spr; rpm = b.x.rpm
    _ax = np.radians(float(ang_x)); _ay = np.radians(-float(ang_y))
    _ca, _cb = np.sin(_ax), np.cos(_ax); _cc, _cd = np.sin(_ay), np.cos(_ay)
    _det = _ca * _cd - _cb * _cc
    if abs(_det) > 1e-6:
        H = (_cd * px - _cb * py) / _det; V = (-_cc * px + _ca * py) / _det
    else:
        H, V = px, py
    H = H - np.nanmean(H); V = V - np.nanmean(V)

    def _ofilt(sig, order):
        if not order or not fs or not rpm or sig.size < 8:
            return sig
        spc = np.fft.rfft(sig); frq = np.fft.rfftfreq(sig.size, 1.0 / fs)
        k = int(np.argmin(np.abs(frq - order * rpm / 60.0)))
        out = np.zeros_like(spc); out[k] = spc[k]
        return np.fft.irfft(out, sig.size)
    _order = {"Direct": 0, "1X": 1, "2X": 2}.get(fmode, 0)
    Hf, Vf = _ofilt(H, _order), _ofilt(V, _order)
    sp = None; _kmarks = []
    if spr and spr > 4 and Hf.size >= 2 * spr:
        _nr = Hf.size // spr
        if _order == 0:
            xv, yv = Hf, Vf
            _kmarks = [(float(Hf[i * spr]), float(Vf[i * spr])) for i in range(_nr)]
            sp = _kmarks[0] if _kmarks else None
        else:
            Ha = Hf[:_nr * spr].reshape(_nr, spr).mean(0)
            Va = Vf[:_nr * spr].reshape(_nr, spr).mean(0)
            _g = max(1, spr // 40)
            xv = Ha[:max(1, spr - _g)]; yv = Va[:max(1, spr - _g)]
            sp = (float(Ha[0]), float(Va[0])); _kmarks = [sp]
    else:
        xv, yv = Hf, Vf
        sp = (float(Hf[0]), float(Vf[0])) if Hf.size else None
        _kmarks = [sp] if sp else []
    _area = float(np.sum(xv[:-1] * yv[1:] - xv[1:] * yv[:-1])) if xv.size > 2 else 0.0
    prec = "CCW" if _area > 0 else "CW"
    if fs and rpm:
        _z = _ofilt(H, 1) + 1j * _ofilt(V, 1)
        _Z = np.fft.fftshift(np.fft.fft(_z))
        _frz = np.fft.fftshift(np.fft.fftfreq(_z.size, 1.0 / fs)); _f1 = rpm / 60.0
        _kf = int(np.argmin(np.abs(_frz - _f1))); _kr = int(np.argmin(np.abs(_frz + _f1)))
        prec = "Forward" if abs(_Z[_kf]) >= abs(_Z[_kr]) else "Reverse"
    _precd = ("Hacia adelante (forward)" if prec == "Forward"
              else "Inversa (reverse)" if prec == "Reverse" else prec)
    _lbl_b = (b.x.label or b.name or ""); _bnum = next((int(c) for c in _lbl_b if c.isdigit()), 0)
    rot_api = "CCW" if _bnum >= 5 else "CW"; _maq = "generador" if _bnum >= 5 else "turbina"
    cx, cy = float(np.nanmean(xv)), float(np.nanmean(yv))
    _pc = np.column_stack([xv - cx, yv - cy]); major = minor = ellip = smax = 0.0
    if _pc.shape[0] > 2:
        try:
            _, _, _vt = np.linalg.svd(_pc, full_matrices=False)
            _pj = _pc @ _vt[0]; _pn = _pc @ _vt[1]
            major = float(_pj.max() - _pj.min()); minor = float(_pn.max() - _pn.min())
            ellip = (minor / major) if major else 0.0
            smax = float(np.max(np.hypot(_pc[:, 0], _pc[:, 1])))
        except Exception:  # noqa: BLE001
            pass
    if ellip >= 0.8:
        shp, shc = "Circular — desbalance / condición sana", "#1f9d55"
    elif ellip >= 0.4:
        shp, shc = "Elíptica — precarga o anisotropía de rigidez", "#e8890c"
    else:
        shp, shc = "Muy plana — precarga fuerte / desalineación", "#dc3545"
    _kpos = ""
    if sp is not None:
        _kang = float(np.degrees(np.arctan2(sp[0], sp[1])))
        _kpos = f"{abs(_kang):.0f}°{'R' if _kang >= 0 else 'L'}"
    Rp = max(float(smax) if smax else 0.0,
             float(np.nanmax(np.hypot(xv, yv))) if len(xv) else 0.0)
    return dict(xv=xv, yv=yv, kmarks=_kmarks, sp=sp, order=_order, smax=smax, Rp=Rp,
                prec=prec, precd=_precd, rot_api=rot_api, maq=_maq, shp=shp, shc=shc,
                major=major, minor=minor, ellip=ellip, kpos=_kpos, rpm=rpm)


def _orbit_fig(b, d, R, ang_x, ang_y, fmode, h=560, compact=False):
    """Arma la figura de UNA órbita con escala forzada R (misma para todas). El sentido
    de giro real (API 670) = solo texto CW/CCW en la esquina (sin arco)."""
    fig = go.Figure()
    _add_probe_glyphs(fig, R, ang_x, ang_y, b.x.label, b.y.label,
                      with_top=False, with_cone=not compact)
    fig.add_trace(go.Scatter(x=d["xv"], y=d["yv"], mode="lines",
                  line=dict(color=_NAVY, width=1.6), name=b.name, showlegend=False))
    km = d["kmarks"]
    if km:
        _kx = [p[0] for p in km]; _ky = [p[1] for p in km]
        if d["order"] == 0 and len(km) > 1:
            fig.add_trace(go.Scatter(x=_kx, y=_ky, mode="markers",
                          marker=dict(color="#1f9d55", size=6,
                                      line=dict(color="white", width=0.8)), showlegend=False))
        else:
            _ks = (d["smax"] if d["smax"] else 1.0) * 0.07
            for _th in range(0, 360, 60):
                _r = np.radians(_th)
                fig.add_trace(go.Scatter(x=[_kx[0], _kx[0] + _ks * np.sin(_r)],
                              y=[_ky[0], _ky[0] + _ks * np.cos(_r)], mode="lines",
                              line=dict(color="#1f9d55", width=1.2),
                              hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=_kx[:1], y=_ky[:1], mode="markers",
                          marker=dict(color="#1f9d55", size=12,
                                      line=dict(color="white", width=1.5)), showlegend=False))
    if d["sp"] is not None and not compact:
        fig.add_annotation(x=d["sp"][0], y=d["sp"][1], text=f"KΦ {d['kpos']}", showarrow=False,
                           yshift=-14, font=dict(color="#1f9d55", size=10, family="IBM Plex Mono"))
    # GIRO real (API 670) = SOLO texto CW/CCW en la esquina (sin arco, no roba espacio).
    fig.add_annotation(xref="paper", yref="paper", x=0.985, y=0.985, xanchor="right",
                       yanchor="top", text=f"<b>{d['rot_api']}</b>", showarrow=False,
                       font=dict(color="#64748b", size=13, family="IBM Plex Mono"),
                       bgcolor="rgba(255,255,255,0.65)")
    _rng = R * 1.5 if R else None
    fig.update_xaxes(title=(b.x.unit or "µm") if not compact else "", scaleanchor="y",
                     scaleratio=1, zeroline=True, zerolinecolor="rgba(120,140,170,.45)",
                     gridcolor="rgba(120,140,170,.12)", range=[-_rng, _rng] if _rng else None)
    fig.update_yaxes(title=(b.y.unit or "µm") if not compact else "", zeroline=True,
                     zerolinecolor="rgba(120,140,170,.45)", gridcolor="rgba(120,140,170,.12)",
                     range=[-_rng, _rng] if _rng else None)
    _ttl = (f"{b.name} · {fmode}" if compact
            else f"Órbita · {b.name} · {fmode}  ·  precesión {d['precd']}")
    fig = _layout(fig, _ttl, h=h)
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    return fig


def _orbit_add_cell(fig, bb, d, R, ang_x, ang_y, row, col):
    """Pinta UNA órbita (probetas + trazo + keyphasor + texto CW/CCW) en la celda
    (row, col) de un subplot, con escala R (misma para todas)."""
    _rc = dict(row=row, col=col)
    _add_probe_glyphs(fig, R, ang_x, ang_y, bb.x.label, bb.y.label,
                      with_top=False, with_cone=False, row=row, col=col)
    fig.add_trace(go.Scatter(x=d["xv"], y=d["yv"], mode="lines",
                  line=dict(color=_NAVY, width=1.6), showlegend=False), **_rc)
    km = d["kmarks"]
    if km:
        _kx = [p[0] for p in km]; _ky = [p[1] for p in km]
        if d["order"] == 0 and len(km) > 1:
            fig.add_trace(go.Scatter(x=_kx, y=_ky, mode="markers",
                          marker=dict(color="#1f9d55", size=5,
                                      line=dict(color="white", width=0.7)),
                          showlegend=False), **_rc)
        else:
            _ks = (d["smax"] if d["smax"] else 1.0) * 0.07
            for _th in range(0, 360, 60):
                _r = np.radians(_th)
                fig.add_trace(go.Scatter(x=[_kx[0], _kx[0] + _ks * np.sin(_r)],
                              y=[_ky[0], _ky[0] + _ks * np.cos(_r)], mode="lines",
                              line=dict(color="#1f9d55", width=1.1),
                              hoverinfo="skip", showlegend=False), **_rc)
            fig.add_trace(go.Scatter(x=_kx[:1], y=_ky[:1], mode="markers",
                          marker=dict(color="#1f9d55", size=11,
                                      line=dict(color="white", width=1.4)),
                          showlegend=False), **_rc)
    fig.add_annotation(x=R * 1.33, y=R * 1.33, text=f"<b>{d['rot_api']}</b>", showarrow=False,
                       font=dict(color="#64748b", size=12, family="IBM Plex Mono"),
                       bgcolor="rgba(255,255,255,0.65)", **_rc)


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
    rpm_on = st.checkbox("Superponer RPM (eje derecho) — comparar vibración vs velocidad",
                         key="tr_rpm")
    mk = {"Overall pp": "pp", "RMS": "rms", "1X amp": "1x"}[met]
    wlim = {"24 h": 30, "7 días": 180, "30 días": 800, "Todo": 5000}[win]
    if not sel:
        st.info("Elegí al menos un canal.")
        return
    _ct = tuple((c["point"], c["day"], c["time"], c["key"]) for c in hist_caps
                if c.get("point") and c.get("day") and c.get("time") and c.get("key"))

    fig = go.Figure()
    units_seen, n_pts, lim_lines = set(), 0, {}
    ends, t_min, t_max = [], None, None     # último punto de cada canal + rango X
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
                line=dict(color=col, width=1.7, shape="spline", smoothing=1.0),
                hovertemplate=f"<b>{lab}</b><br>%{{x|%d %b %H:%M}}<br>"
                              f"%{{y:.1f}} {unit or ''}<extra></extra>"))
            ends.append((ts[-1], vals[-1], col, lab))
            t_min = ts[0] if t_min is None else min(t_min, ts[0])
            t_max = ts[-1] if t_max is None else max(t_max, ts[-1])
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
    # VALOR ACTUAL al final de cada línea (marcador + etiqueta, color del canal)
    for xt, yv, col, lab in ends:
        fig.add_trace(go.Scatter(x=[xt], y=[yv], mode="markers",
                                 marker=dict(color=col, size=7),
                                 hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=xt, y=yv, text=f"<b>{yv:.1f}</b>", showarrow=False,
                           xanchor="left", xshift=9, font=dict(color=col, size=12.5,
                           family="IBM Plex Mono"),
                           bgcolor="rgba(255,255,255,0.72)")
    # SUPERPONER RPM en el eje DERECHO (y2): una línea por keyphasor distinto de los
    # canales elegidos (turbina ~14000 / generador ~1800). Escala aparte de la vibración.
    rpm_on_axis = False
    if rpm_on:
        # dedupe por KEYPHASOR (turbina/generador), no por punto: 1XD y 2XD comparten
        # la turbina → una sola línea RPM, no dos iguales.
        _seen_kph, _rcol = set(), "#8090a6"
        for lab in sel:
            ch = cmap[lab]
            _kname = "turbina" if (ch.rpm or 0) > 5000 else "generador"
            if _kname in _seen_kph:
                continue
            rts, rvals, _ = _trend(_ct, ch.cap_point, "rpm", axis=ch.cap_axis, limit=wlim)
            if not rts:
                continue
            _seen_kph.add(_kname)
            fig.add_trace(go.Scatter(
                x=rts, y=rvals, mode="lines", name=f"RPM {_kname}", yaxis="y2",
                line=dict(color=_rcol, width=1.3, dash="dot", shape="spline", smoothing=1.0),
                hovertemplate=f"<b>RPM {_kname}</b><br>%{{x|%d %b %H:%M}}<br>"
                              f"%{{y:,.0f}} rpm<extra></extra>"))
            fig.add_annotation(x=rts[-1], y=rvals[-1], yref="y2", text=f"{rvals[-1]:,.0f}",
                               showarrow=False, xanchor="left", xshift=9,
                               font=dict(color=_rcol, size=11, family="IBM Plex Mono"),
                               bgcolor="rgba(255,255,255,0.72)")
            rpm_on_axis = True
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
    # Rango X PEGADO al inicio (sin padding a la izquierda) + ~6% a la derecha
    # para que quepa la etiqueta del valor actual al final de cada línea.
    _xr = None
    if t_min is not None and t_max is not None and t_max > t_min:
        _pad = (t_max - t_min) * 0.06
        _xr = [t_min, t_max + _pad]
    fig.update_xaxes(title="", showgrid=True, gridcolor="rgba(120,140,170,.12)",
                     showline=True, linecolor="#d9e1ec", range=_xr,
                     rangeslider=dict(visible=False))
    fig.update_layout(yaxis=dict(title=f"{met} ({_unit or 'a.u.'})", showgrid=True,
                                 gridcolor="rgba(120,140,170,.12)", zeroline=False))
    # leyenda ABAJO (no choca con el título) + márgenes amplios + hover unificado
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=64, r=78 if rpm_on_axis else 52, t=52, b=72),
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white")
    if rpm_on_axis:
        fig.update_layout(yaxis2=dict(title="RPM", overlaying="y", side="right",
                                      showgrid=False, color="#5a6b85",
                                      rangemode="tozero"))
    st.plotly_chart(fig, use_container_width=True, config=_NOBAR)
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
    _components.html(html, height=58)


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


def _table_png(cols, rows, title=""):
    """Tabla simple como PNG (header navy + zebra, sin columna de estado) para
    adjuntar debajo de un gráfico en la descarga."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol, nrow = len(cols), len(rows)
    fig_w = min(1.2 * ncol + 1.0, 16)
    fig_h = 0.5 * nrow + (0.85 if title else 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    _top = 1.0
    if title:
        _top = 1.0 - 0.42 / fig_h
        fig.text(0.010, 0.97, title, fontsize=9.5, fontweight="normal",
                 color="#5a6b85", va="top", ha="left")      # chico, suave, discreto
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                   bbox=[0.006, 0.0, 0.988, _top])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    for j in range(ncol):
        h = tbl[0, j]
        h.set_facecolor("#12305e")
        h.get_text().set_color("white")
        h.get_text().set_fontweight("bold")
    for i in range(nrow):
        for j in range(ncol):
            cc = tbl[i + 1, j]
            cc.set_facecolor("#f4f7fb" if i % 2 else "#ffffff")
            cc.get_text().set_color("#0b1f3a")
        tbl[i + 1, 0].get_text().set_fontweight("bold")
    colw = [max([len(str(cols[j]))] + [len(str(rows[i][j])) for i in range(nrow)])
            for j in range(ncol)]
    _tot = float(sum(colw)) or 1.0
    for j in range(ncol):
        for i in range(nrow + 1):
            tbl[i, j].set_width(colw[j] / _tot)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return _round_png(buf.getvalue())      # esquinas redondeadas


def _round_png(png_bytes, radius=18, border="#e2e8f2", border_w=2):
    """Redondea las esquinas de un PNG (fondo blanco) + marco fino → tablas con
    borde redondeado consistente con las de pantalla."""
    import io

    from PIL import Image, ImageDraw
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = im.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    im = Image.composite(im, Image.new("RGB", (w, h), "white"), mask)
    ImageDraw.Draw(im).rounded_rectangle([border_w // 2, border_w // 2,
                                          w - 1 - border_w // 2, h - 1 - border_w // 2],
                                         radius=radius, outline=border, width=border_w)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _combine_vertical(png_top, png_bottom, gap=22):
    """Apila dos PNG verticalmente (centrados, fondo blanco) → un solo PNG."""
    import io

    from PIL import Image
    a = Image.open(io.BytesIO(png_top)).convert("RGB")
    b = Image.open(io.BytesIO(png_bottom)).convert("RGB")
    w = max(a.width, b.width)
    out = Image.new("RGB", (w, a.height + gap + b.height), "white")
    out.paste(a, ((w - a.width) // 2, 0))
    out.paste(b, ((w - b.width) // 2, a.height + gap))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def _download_fig_table_button(fig, cols, rows, filename, key, table_title=""):
    """Descarga el gráfico + una tabla de métricas apilados en una sola imagen."""
    try:
        import plotly.io as _pio
        top = _fig_png_cached(_pio.to_json(fig))
        combined = _combine_vertical(top, _table_png(cols, rows, table_title))
    except Exception:  # noqa: BLE001
        _download_fig_button(fig, filename, key)
        return
    _download_png_button(combined, filename, key=key)


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
        _rp = "—" if not c.rpm else f"{c.rpm:,.0f}"       # rpm del keyphasor del canal
        rows.append([dot(sev), c.label, tipo, unit, _rp, _f(alarm), _f(danger),
                     _f(overall), _f(a1), _p(ph1), _f(a2), _p(ph2)])
        img_rows.append([c.label, tipo, unit, _rp, _f(alarm), _f(danger),
                         _f(overall), _f(a1), _p(ph1), _f(a2), _p(ph2)])
        sev_list.append(sev)
    _cols = ["", "Sensor", "Tipo", "Unidades", "RPM", "Alarma", "Danger", "Directa",
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
    from core.waveform_metrics import compute_waveform_metrics
    _wcmap = {c.label: c for c in rec.channels}
    _wlabels = [c.label for c in rec.channels]
    wc1, wc2 = st.columns([3, 1])
    with wc1:
        wsel = st.multiselect("Canales (uno o varios)", _wlabels,
                              default=_wlabels[:2] if len(_wlabels) >= 2 else _wlabels,
                              key="wf_chs")
    with wc2:
        wkph = st.checkbox("Marcar keyphasor (1X)", value=True, key="wf_kph")
    if not wsel:
        st.info("Elegí al menos un canal.")
    else:
        fig = go.Figure()
        _wunits, _wrows, _wtmax, _wymax = set(), [], 0.0, 0.0
        for i, lab in enumerate(wsel):
            ch = _wcmap[lab]
            vv = ch.v
            t_ms = ch.t * 1000.0 if ch.t.size else np.arange(vv.size) * 1.0
            if t_ms.size:
                _wtmax = max(_wtmax, float(t_ms[-1]))
            _vf = vv[np.isfinite(vv)]
            if _vf.size:
                _wymax = max(_wymax, float(np.max(np.abs(_vf))))
            col = _TREND_PAL[i % len(_TREND_PAL)]
            fig.add_trace(go.Scatter(
                x=t_ms, y=vv, mode="lines", name=lab,
                line=dict(color=col, width=1.4, shape="spline", smoothing=0.4),
                hovertemplate=f"<b>{lab}</b><br>%{{x:.2f}} ms<br>"
                              f"%{{y:.2f}} {ch.unit or ''}<extra></extra>"))
            _wunits.add(ch.unit or "")
            # marcas de KEYPHASOR (una por vuelta, t=k·60/rpm) — como los puntos de System1
            if wkph and ch.rpm and t_ms.size:
                Tms = 60000.0 / ch.rpm
                if Tms < (t_ms[-1] - t_ms[0]):
                    kts = np.arange(float(t_ms[0]), float(t_ms[-1]), Tms)
                    kvs = np.interp(kts, t_ms, vv)
                    fig.add_trace(go.Scatter(
                        x=kts, y=kvs, mode="markers",
                        marker=dict(color=col, size=7, symbol="circle",
                                    line=dict(color="white", width=1)),
                        name=f"KPH {lab}", showlegend=False, hoverinfo="skip"))
            # métricas por canal (para la tabla System1-class)
            mx = compute_waveform_metrics(vv) or {}
            _u = ch.unit or ""
            _wrows.append([lab, f"{mx.get('peak_to_peak',0):.2f} {_u}",
                           f"{mx.get('rms',0):.2f} {_u}", f"{mx.get('peak',0):.2f} {_u}",
                           f"{mx.get('crest_factor',0):.2f}",
                           f"{ch.rpm:,.0f}" if ch.rpm else "—"])
        _wu = next(iter(_wunits)) if len(_wunits) == 1 else "mixto"
        fig = _layout(fig, "Forma de onda", h=500)
        _ym = _wymax * 1.08 if _wymax else None
        # X ARRANCA EN 0 (sin padding) hasta el fin de la captura; Y CENTRADO en 0
        # y simétrico (como System1) → el 0 queda al medio.
        fig.update_xaxes(title="Tiempo (ms)", showgrid=True,
                         gridcolor="rgba(120,140,170,.12)", showline=True,
                         linecolor="#d9e1ec", zeroline=False,
                         range=[0, _wtmax] if _wtmax else None,
                         ticks="outside", ticklen=4, tickcolor="#d9e1ec")
        fig.update_layout(
            yaxis=dict(title=f"Amplitud ({_wu or 'a.u.'})", showgrid=True,
                       gridcolor="rgba(120,140,170,.12)", zeroline=True,
                       zerolinecolor="rgba(120,140,170,.45)", zerolinewidth=1.2,
                       range=[-_ym, _ym] if _ym else None,
                       ticks="outside", ticklen=4, tickcolor="#d9e1ec"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center",
                        x=0.5, font=dict(size=12)),
            margin=dict(l=64, r=28, t=52, b=72), hovermode="x unified",
            plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config=_NOBAR)
        _download_fig_table_button(
            fig, ["Sensor", "Pico-pico", "RMS", "Pico", "Crest", "RPM"], _wrows,
            _vfile(f"Forma de onda {'-'.join(wsel[:3])}"), key="wf",
            table_title=f"Forma de onda · {cur_asset} · {_fecha}")
        # Panel de métricas clase System1 (pp · RMS · Pico · Crest · RPM por canal)
        from core.ui_industrial import html_table
        html_table(["Sensor", "Pico-pico", "RMS", "Pico", "Crest", "RPM"], _wrows)
        # Diagnóstico por norma solo con UN canal (evita saturar con varios)
        if len(wsel) == 1:
            ch = _wcmap[wsel[0]]
            mx = compute_waveform_metrics(ch.v) or {}
            try:
                from core.waveform_diagnostics import build_waveform_diagnostics_rotordyn
                dW = build_waveform_diagnostics_rotordyn(
                    time_s=ch.t, amplitude=ch.v, metrics=mx, machine_label=rec.name,
                    point_label=ch.label, amplitude_unit=ch.unit or "")
                _sev = str(dW.get("severity_global", "")).upper()
                _c = {"DANGER": "#dc3545", "CRITICAL": "#dc3545", "WARNING": "#e8890c",
                      "CAUTION": "#84cc16"}.get(_sev, "#1f9d55")
                st.markdown(
                    f"<div style='border-left:4px solid {_c};background:#f8fafc;border:1px "
                    f"solid #e2e8f2;border-radius:8px;padding:11px 15px;margin-top:10px'>"
                    f"<b style='font-size:13px'>{dW.get('headline','')}</b>"
                    f"<div style='color:#3a4c66;font-size:12.5px;margin-top:5px'>"
                    f"{dW.get('detail','')}</div><div style='color:#0b1f3a;font-size:12.5px;"
                    f"margin-top:4px'><b>Acción:</b> {dW.get('action','')}</div></div>",
                    unsafe_allow_html=True)
            except Exception:  # noqa: BLE001
                pass
        st.caption("Tiempo en ms · puntos = marca de keyphasor (1 por vuelta) · "
                   f"amplitud en {_wu}. Varios canales = compará X/Y o cojinetes en el "
                   "mismo eje de tiempo.")

elif view == "Espectro":
    _scmap = {c.label: c for c in rec.channels}
    _slabels = [c.label for c in rec.channels]

    def _spk(u):
        u = (u or "").lower()
        if "mm/s" in u:
            return "vel"
        if u.strip() in ("g", "g pk", "g rms") or "m/s2" in u or "m/s²" in u:
            return "accel"
        return "disp"
    _CPMAX = {"disp": 60000, "vel": 300000, "accel": 600000}

    sc1, sc2 = st.columns([3, 1])
    with sc1:
        ssel = st.multiselect("Canales (uno o varios)", _slabels,
                              default=_slabels[:1], key="sp_chs")
    with sc2:
        xsc = st.selectbox("Escala X", ["Auto (por tipo)", "60.000 cpm (despl)",
                           "300.000 cpm (vel)", "600.000 cpm (accel)", "Completo"],
                           key="sp_xsc")
    if not ssel:
        st.info("Elegí al menos un canal.")
    else:
        fig = go.Figure()
        _sunits, _hrows, _cpmax_auto, _ord_ref = set(), [], 0, None
        for i, lab in enumerate(ssel):
            ch = _scmap[lab]
            v = ch.v[np.isfinite(ch.v)]
            fs = ch.fs_hz
            if v.size < 8 or not fs:
                continue
            w = v - np.mean(v)
            win = np.hanning(w.size)
            am = np.abs(np.fft.rfft(w * win)) * 2.0 / np.sum(win)   # cero-a-pico corregido
            fr = np.fft.rfftfreq(w.size, d=1.0 / fs)
            cpm = fr * 60.0
            kind = _spk(ch.unit)
            mode = "pp" if kind == "disp" else "rms"
            ampd = am * (2.0 if mode == "pp" else 1.0 / np.sqrt(2.0))
            ud = ch.unit or ("µm pp" if kind == "disp" else
                             "mm/s rms" if kind == "vel" else "g")
            _sunits.add(ud)
            _cpmax_auto = max(_cpmax_auto, _CPMAX[kind])
            col = _TREND_PAL[i % len(_TREND_PAL)]
            fig.add_trace(go.Scatter(
                x=cpm, y=ampd, mode="lines", name=lab,
                line=dict(color=col, width=1.2),      # sin spline: los picos deben ser nítidos
                hovertemplate=f"<b>{lab}</b><br>%{{x:,.0f}} cpm<br>"
                              f"%{{y:.2f}} {ud}<extra></extra>"))
            if ch.rpm:
                if _ord_ref is None:
                    _ord_ref = ch.rpm                 # 1X en cpm = rpm del primer keyphasor

                def _aat(fh):
                    k = int(np.argmin(np.abs(fr - fh)))
                    lo, hi = max(0, k - 2), min(ampd.size, k + 3)
                    return float(np.max(ampd[lo:hi])) if hi > lo else 0.0
                f1 = ch.rpm / 60.0
                _ov = float(np.ptp(v)) if mode == "pp" else float(np.std(v))
                _hrows.append([lab, f"{ch.rpm:,.0f}", f"{_ov:.2f} {ch.unit or ''}",
                               f"{_aat(f1):.2f}", f"{_aat(2*f1):.2f}",
                               f"{_aat(3*f1):.2f}", f"{_aat(4*f1):.2f}"])
        if not fig.data:
            st.warning("Espectro: faltan muestras o no se pudo estimar la fs.")
        else:
            _xmax = {"Auto (por tipo)": _cpmax_auto, "60.000 cpm (despl)": 60000,
                     "300.000 cpm (vel)": 300000, "600.000 cpm (accel)": 600000,
                     "Completo": None}[xsc]
            _su = next(iter(_sunits)) if len(_sunits) == 1 else "mixto"
            # marcas de ORDEN 1X-4X (del primer keyphasor) en cpm
            if _ord_ref:
                for _k, _ol in [(1, "1X"), (2, "2X"), (3, "3X"), (4, "4X")]:
                    _xo = _k * _ord_ref
                    if _xmax is None or _xo <= _xmax:
                        fig.add_vline(x=_xo, line=dict(color="#8090a6", width=1, dash="dot"),
                                      annotation_text=_ol, annotation_position="top",
                                      annotation_font_size=10)
            fig = _layout(fig, "Espectro (FFT)", h=500)
            fig.update_xaxes(title="Frecuencia (cpm)", showgrid=True,
                             gridcolor="rgba(120,140,170,.12)", showline=True,
                             linecolor="#d9e1ec", zeroline=False,
                             range=[0, _xmax] if _xmax else None)
            fig.update_layout(
                yaxis=dict(title=f"Amplitud ({_su})", showgrid=True,
                           gridcolor="rgba(120,140,170,.12)", rangemode="tozero"),
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center",
                            x=0.5, font=dict(size=12)),
                margin=dict(l=64, r=28, t=52, b=72), hovermode="x unified",
                plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True, config=_NOBAR)
            if _hrows:
                _spcols = ["Sensor", "RPM", "Overall", "1X", "2X", "3X", "4X"]
                _download_fig_table_button(
                    fig, _spcols, _hrows,
                    _vfile(f"Espectro {'-'.join(ssel[:3])}"), key="sp",
                    table_title=f"Espectro · {cur_asset} · {_fecha}")
                from core.ui_industrial import html_table
                html_table(_spcols, _hrows)
            else:
                _download_fig_button(fig, _vfile(f"Espectro {'-'.join(ssel[:3])}"), key="sp")
            # Diagnóstico por norma con UN canal (reutiliza core.spectrum_diagnostics)
            if len(ssel) == 1 and _scmap[ssel[0]].rpm:
                ch = _scmap[ssel[0]]
                v = ch.v[np.isfinite(ch.v)]
                w = v - np.mean(v)
                win = np.hanning(w.size)
                amp = np.abs(np.fft.rfft(w * win)) * 2.0 / np.sum(win)
                fr = np.fft.rfftfreq(w.size, d=1.0 / (ch.fs_hz or 1.0))
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
                    f"<div style='border-left:4px solid {_col};background:#f8fafc;border:1px "
                    f"solid #e2e8f2;border-radius:8px;padding:11px 15px;margin-top:8px'>"
                    f"<span style='background:{_col};color:#fff;border-radius:6px;padding:2px "
                    f"9px;font:700 11px IBM Plex Mono'>{diagS.get('status','')}</span>  "
                    f"<b style='font-size:13px'>{diagS.get('headline','')}</b>"
                    f"<div style='color:#3a4c66;font-size:12.5px;margin-top:5px'>"
                    f"{diagS.get('narrative','')}</div></div>", unsafe_allow_html=True)
            st.caption("X en cpm (ciclos/min) · default por tipo: desplazamiento 60.000 · "
                       "velocidad 300.000 · aceleración 600.000 · órdenes 1X-4X marcados. "
                       "Amplitud: pp (desplazamiento) / rms (vel·accel). Picos sin suavizar.")

elif view == "Full Spectrum":
    _fbs = [bb for bb in rec.bearings if bb.has_xy]
    if not _fbs:
        st.warning("Full Spectrum: necesita el **par X/Y** de un cojinete (espectro "
                   "direccional · precesión forward/backward · supera a System1).")
    else:
        _fnames = [bb.name for bb in _fbs]
        _fbmap = {bb.name: bb for bb in _fbs}

        def _spk2(u):
            u = (u or "").lower()
            if "mm/s" in u:
                return "vel", 300000
            if u.strip() in ("g", "g pk", "g rms") or "m/s2" in u or "m/s²" in u:
                return "accel", 600000
            return "disp", 60000

        fc1, fc2 = st.columns([3, 1])
        with fc1:
            fsel = st.multiselect("Cojinetes (par X/Y)", _fnames,
                                 default=_fnames[:1], key="fs_brs")
        with fc2:
            fxsc = st.selectbox("Escala X", ["Auto (por tipo)", "60.000 cpm (despl)",
                               "300.000 cpm (vel)", "600.000 cpm (accel)", "Completo"],
                               key="fs_xsc")
        if not fsel:
            st.info("Elegí al menos un cojinete.")
        else:
            fig = go.Figure()
            _funits, _frows, _cpauto, _ordref = set(), [], 0, None
            _multi = len(fsel) > 1
            for i, nm in enumerate(fsel):
                b = _fbmap[nm]
                n = min(b.x.v.size, b.y.v.size)
                if n < 16:
                    continue
                xv = b.x.v[:n] - np.nanmean(b.x.v[:n])
                yv = b.y.v[:n] - np.nanmean(b.y.v[:n])
                fs = b.x.fs_hz or b.y.fs_hz
                if not fs:
                    continue
                win = np.hanning(n)
                z = (xv * win) + 1j * (yv * win)
                sp = np.fft.fftshift(np.fft.fft(z)) * (2.0 / np.sum(win))
                fr = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
                cpm = fr * 60.0
                kind, cpmx = _spk2(b.x.unit)
                amp = np.abs(sp) * (2.0 if kind == "disp" else 1.0 / np.sqrt(2.0))
                ud = b.x.unit or ("µm pp" if kind == "disp" else
                                  "mm/s rms" if kind == "vel" else "g")
                _funits.add(ud)
                _cpauto = max(_cpauto, cpmx)
                fwd = cpm >= 0
                # color por cojinete; forward sólido, backward punteado
                if _multi:
                    col = _TREND_PAL[i % len(_TREND_PAL)]
                    cf, cb = col, col
                else:
                    cf, cb = _NAVY, _AMBER      # clásico: fwd navy / bwd ámbar
                fig.add_trace(go.Scatter(x=cpm[fwd], y=amp[fwd], mode="lines",
                              line=dict(color=cf, width=1.2),
                              name=f"{nm} →" if _multi else "Forward · hacia adelante (+)",
                              hovertemplate=f"<b>{nm} fwd</b><br>%{{x:,.0f}} cpm<br>"
                                            f"%{{y:.2f}} {ud}<extra></extra>"))
                fig.add_trace(go.Scatter(x=cpm[~fwd], y=amp[~fwd], mode="lines",
                              line=dict(color=cb, width=1.2,
                                        dash="dot" if _multi else "solid"),
                              name=f"{nm} ←" if _multi else "Reverse · inversa (−)",
                              hovertemplate=f"<b>{nm} rev</b><br>%{{x:,.0f}} cpm<br>"
                                            f"%{{y:.2f}} {ud}<extra></extra>"))
                ef = float(np.sum(amp[fwd] ** 2))
                eb = float(np.sum(amp[~fwd] ** 2))
                tot = ef + eb or 1.0
                dom = "Forward" if ef >= eb else "Reverse"
                if b.x.rpm and _ordref is None:
                    _ordref = b.x.rpm
                _frows.append([nm, f"{b.x.rpm:,.0f}" if b.x.rpm else "—",
                               f"{ef/tot*100:.0f}%", f"{eb/tot*100:.0f}%", dom])
            if not fig.data:
                st.warning("Full Spectrum: faltan muestras o frecuencia de muestreo.")
            else:
                _fxmax = {"Auto (por tipo)": _cpauto, "60.000 cpm (despl)": 60000,
                          "300.000 cpm (vel)": 300000, "600.000 cpm (accel)": 600000,
                          "Completo": None}[fxsc]
                _fu = next(iter(_funits)) if len(_funits) == 1 else "mixto"
                if _ordref:
                    for _k, _ol in [(1, "+1X"), (2, "+2X"), (3, "+3X"),
                                    (-1, "−1X"), (-2, "−2X")]:
                        _xo = _k * _ordref
                        if _fxmax is None or abs(_xo) <= _fxmax:
                            fig.add_vline(x=_xo, line=dict(color="#8090a6", width=1,
                                          dash="dot"), annotation_text=_ol,
                                          annotation_position="top", annotation_font_size=10)
                fig = _layout(fig, "Full Spectrum", h=500)
                fig.update_xaxes(
                    title="Frecuencia direccional (cpm) · + forward (hacia adelante) / "
                          "− reverse (inversa)",
                    showgrid=True, gridcolor="rgba(120,140,170,.12)", showline=True,
                    linecolor="#d9e1ec", zeroline=True, zerolinecolor="rgba(120,140,170,.45)",
                    range=[-_fxmax, _fxmax] if _fxmax else None)
                fig.update_layout(
                    yaxis=dict(title=f"Amplitud ({_fu})", showgrid=True,
                               gridcolor="rgba(120,140,170,.12)", rangemode="tozero"),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center",
                                x=0.5, font=dict(size=12)),
                    margin=dict(l=64, r=28, t=52, b=72), hovermode="x unified",
                    plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True, config=_NOBAR)
                _fscols = ["Cojinete", "RPM", "Forward %", "Reverse %", "Precesión"]
                _download_fig_table_button(
                    fig, _fscols, _frows,
                    _vfile(f"Full Spectrum {'-'.join(fsel[:3])}"), key="fs",
                    table_title=f"Full Spectrum · {cur_asset} · {_fecha}")
                from core.ui_industrial import html_table
                html_table(_fscols, _frows)
                # Interpretación (del primer cojinete) — nomenclatura Bently Nevada
                if _frows:
                    _b0 = _frows[0]
                    _rev = float(_b0[3].rstrip("%"))
                    _dc = "#1f9d55" if _b0[4] == "Forward" else "#e8890c"
                    _warn = (" Reverse (inversa) >30% → sospecha de roce (rub), anisotropía "
                             "de soporte, desalineación o inestabilidad por fluido." if _rev > 30
                             else " Precesión mayormente hacia adelante (forward) = sano "
                             "(sincrónico normal, p.ej. desbalance puro = 1X forward).")
                    st.markdown(
                        f"<div style='border-left:4px solid {_dc};background:#f8fafc;border:1px "
                        f"solid #e2e8f2;border-radius:8px;padding:11px 15px;margin-top:8px'>"
                        f"<span style='background:{_dc};color:#fff;border-radius:6px;padding:2px "
                        f"9px;font:700 11px IBM Plex Mono'>{_b0[4].upper()}</span>  "
                        f"<b>{_b0[0]}: {_b0[2]} forward / {_b0[3]} reverse.</b>"
                        f"<div style='color:#3a4c66;font-size:12.5px;margin-top:5px'>{_warn} "
                        f"Nomenclatura Bently Nevada (hacia adelante / inversa). Suma fwd+rev "
                        f"≈ eje mayor de la órbita; diferencia ≈ eje menor; la componente mayor "
                        f"marca la dirección de precesión.</div></div>", unsafe_allow_html=True)
                st.caption("X direccional en cpm: +forward (hacia adelante) / −reverse (inversa) "
                           "· default por tipo (despl 60k · vel 300k · accel 600k) · órdenes "
                           "±1X,±2X,±3X. Varios cojinetes = color por cojinete (fwd sólido / "
                           "rev punteado). Base: curso Bently Nevada de diagnósticos.")

elif view == "Órbita":
    _obrgs = [bb for bb in rec.bearings if bb.has_xy]
    if not _obrgs:
        st.warning("Órbita: necesita el **par X/Y** de un cojinete. Sube ambos CSV "
                   "(ej. 1xd.csv + 1yd.csv).")
    else:
        oc1, oc2, oc3 = st.columns([2, 1, 1])
        with oc1:
            fmode = st.segmented_control("Tipo de órbita", ["Direct", "1X", "2X"],
                                         default="Direct", key="orb_f") or "Direct"
        with oc2:
            ang_x = st.number_input("X · °R del TOP", value=135, step=5,
                                    key="orb_ax", help="Montaje de la sonda X, lado R, grados "
                                    "desde el tope. SGT-300 real = 135° (abajo-derecha).")
        with oc3:
            ang_y = st.number_input("Y · °L del TOP", value=135, step=5,
                                    key="orb_ay", help="Montaje de la sonda Y, lado L, grados "
                                    "desde el tope. SGT-300 real = 135° (abajo-izquierda).")
        _onames = [bb.name for bb in _obrgs]
        _osel = st.multiselect("Cojinetes (órbitas)", _onames, default=[_onames[0]],
                               key="orb_sel",
                               help="Elegí uno o varios. Varias órbitas = MISMA escala para "
                                    "comparar tamaños a ojo (como System1).")
        _bsel = [_obrgs[_onames.index(n)] for n in _osel] or [_obrgs[0]]
        from core.ui_industrial import html_table  # noqa: PLC0415
        # Pass 1: data de cada órbita → escala GLOBAL (misma para todas)
        _datas = [(bb, _orbit_data(bb, fmode, ang_x, ang_y)) for bb in _bsel]
        _gR = max((d["Rp"] for _, d in _datas), default=1.0) or 1.0
        _ocols = ["Cojinete", "RPM", "Precesión", "Giro (API 670)",
                  "Eje mayor", "Eje menor", "Elipticidad", "Smax"]

        def _orow(bb, d):
            _u = bb.x.unit or "µm"
            return [bb.name, f"{d['rpm']:,.0f}" if d["rpm"] else "—", d["precd"],
                    f"{d['rot_api']} ({d['maq']})", f"{d['major']:.1f} {_u}",
                    f"{d['minor']:.1f} {_u}", f"{d['ellip']:.2f}", f"{d['smax']:.1f} {_u}"]

        if len(_datas) == 1:
            bb, d = _datas[0]
            fig = _orbit_fig(bb, d, _gR, ang_x, ang_y, fmode, h=580, compact=False)
            st.plotly_chart(fig, use_container_width=True, config=_NOBAR)
            _u = bb.x.unit or "µm"
            st.caption(
                f"Giro del eje por **API 670** (visto desde el cojinete #1, conductor→conducido): "
                f"**{d['maq']} = {d['rot_api']}** (turbina CW; tras el gearbox el generador "
                f"invierte a CCW). Punto verde = keyphasor (disparo, muestra 0)"
                + (f", en **{d['kpos']}**" if d["kpos"] else "") +
                f". Diagnóstico = **precesión {d['prec']}**: forward = eje y órbita giran igual "
                f"(desbalance); reverse = opuesto (roce/inestabilidad).")
            _orows = [_orow(bb, d)]
            _download_fig_table_button(fig, _ocols, _orows,
                                       _vfile(f"Orbita {bb.name} {fmode}"), key="orb",
                                       table_title=f"Órbita · {cur_asset} · {bb.name} · {fmode}")
            html_table(_ocols, _orows)
            st.markdown(
                f"<div style='border-left:4px solid {d['shc']};background:#f8fafc;border:1px solid "
                f"#e2e8f2;border-radius:8px;padding:11px 15px;margin-top:8px'>"
                f"<span style='background:{d['shc']};color:#fff;border-radius:6px;padding:2px 9px;"
                f"font:700 11px IBM Plex Mono'>FORMA</span>  <b>{d['shp']}.</b>"
                f"<div style='color:#3a4c66;font-size:12.5px;margin-top:5px'>Elipticidad "
                f"{d['ellip']:.2f} (1 = círculo · 0 = línea) · eje mayor {d['major']:.1f} {_u} · "
                f"menor {d['minor']:.1f} {_u} · Smax {d['smax']:.1f} {_u}. Círculo = solo "
                f"desbalance con rigidez igual; elíptica/plana = precarga o rigidez desigual "
                f"(desalineación). Punto verde = keyphasor · CW/CCW (esquina) = giro real."
                f"</div></div>", unsafe_allow_html=True)
        else:
            from plotly.subplots import make_subplots  # noqa: PLC0415
            import math as _math  # noqa: PLC0415
            _ncol = 2
            _nrow = _math.ceil(len(_datas) / _ncol)
            _titles = [f"{bb.name} · {d['prec']} · giro {d['rot_api']}" for bb, d in _datas]
            figM = make_subplots(rows=_nrow, cols=_ncol, subplot_titles=_titles,
                                 horizontal_spacing=0.08,
                                 vertical_spacing=0.14 if _nrow > 1 else 0.1)
            _L = _gR * 1.5
            for _i, (bb, d) in enumerate(_datas):
                _r, _c = _i // _ncol + 1, _i % _ncol + 1
                _orbit_add_cell(figM, bb, d, _gR, ang_x, ang_y, _r, _c)
                _idx = (_r - 1) * _ncol + _c
                _yid = "y" if _idx == 1 else f"y{_idx}"
                figM.update_xaxes(range=[-_L, _L], scaleanchor=_yid, scaleratio=1,
                                  row=_r, col=_c, zeroline=True,
                                  zerolinecolor="rgba(120,140,170,.4)",
                                  gridcolor="rgba(120,140,170,.1)")
                figM.update_yaxes(range=[-_L, _L], row=_r, col=_c, zeroline=True,
                                  zerolinecolor="rgba(120,140,170,.4)",
                                  gridcolor="rgba(120,140,170,.1)")
            figM.update_layout(
                template="plotly_white", height=360 * _nrow + 40, showlegend=False,
                title=dict(text=f"Órbitas · {fmode} · misma escala (±{_L:,.0f} µm)",
                           font=dict(color="#12305e", size=16)),
                font=dict(family="IBM Plex Sans"), plot_bgcolor="white",
                paper_bgcolor="white", margin=dict(l=55, r=35, t=66, b=45))
            st.plotly_chart(figM, use_container_width=True, config=_NOBAR)
            _orows_all = [_orow(bb, d) for bb, d in _datas]
            _download_fig_table_button(
                figM, _ocols, _orows_all, _vfile(f"Orbitas {fmode} {len(_datas)}coj"),
                key="orb_multi_dl", table_title=f"Órbitas · {cur_asset} · {fmode}")
            html_table(_ocols, _orows_all)
        st.caption("Coordenadas de MÁQUINA (Vertical↑ / Horizontal→, sondas rotadas desde su "
                   "ángulo de montaje) · aspecto 1:1 real · promedio síncrono · punto verde = "
                   "keyphasor (muestra 0), idéntico a System1. Base: Bently Nevada + Eisenmann.")

elif view == "Deflexión del eje":
    # FORMA DE DEFLEXIÓN DEL EJE (Bently manual 3-8 / 9-4): se colocan las órbitas
    # 1X de cada cojinete a lo largo de la posición axial y se UNEN los puntos
    # Keyphasor → se ve la deflexión aparente del eje (forma modal del rotor:
    # cilíndrico/traslacional, pivotal/cónico, etc.) al disparo del keyphasor.
    import re as _re2
    _brgs = [b for b in rec.bearings if b.has_xy]
    if len(_brgs) < 2:
        st.warning("Deflexión del eje: necesita **≥2 cojinetes** con par X/Y "
                   "(sube los CSV X/Y de cada cojinete del tren).")
    else:
        def _bnum(b):
            m = _re2.match(r"\s*(\d+)", (b.x.label or b.name or ""))
            return int(m.group(1)) if m else 999
        _brgs = sorted(_brgs, key=_bnum)
        _ord = st.segmented_control("Tipo de órbita", ["1X", "2X"],
                                    default="1X", key="defl_f") or "1X"
        _order_d = {"1X": 1, "2X": 2}[_ord]

        def _f_ord(sig, fs, rpm, order):
            if not fs or not rpm or sig.size < 8:
                return sig
            spc = np.fft.rfft(sig)
            frq = np.fft.rfftfreq(sig.size, 1.0 / fs)
            k = int(np.argmin(np.abs(frq - order * rpm / 60.0)))
            out = np.zeros_like(spc); out[k] = spc[k]
            return np.fft.irfft(out, sig.size)

        _stations = []
        _maxr = 1e-9
        for b in _brgs:
            px = np.asarray(b.x.v, float); py = np.asarray(b.y.v, float)
            _n = min(px.size, py.size); px, py = px[:_n], py[:_n]
            fs = b.x.fs_hz or b.y.fs_hz; spr = b.x.spr or b.y.spr; rpm = b.x.rpm
            # Coords de máquina (sondas a 45°): H=(X−Y)/√2, V=(X+Y)/√2
            H = (px - py) / np.sqrt(2.0); V = (px + py) / np.sqrt(2.0)
            H = H - np.nanmean(H); V = V - np.nanmean(V)
            Hf = _f_ord(H, fs, rpm, _order_d); Vf = _f_ord(V, fs, rpm, _order_d)
            if spr and spr > 4 and Hf.size >= 2 * spr:
                nr = Hf.size // spr
                Ha = Hf[:nr * spr].reshape(nr, spr).mean(0)
                Va = Vf[:nr * spr].reshape(nr, spr).mean(0)
            else:
                Ha, Va = Hf, Vf
            kp = (float(Ha[0]), float(Va[0])) if Ha.size else (0.0, 0.0)
            _r = float(np.max(np.hypot(Ha, Va))) if Ha.size else 0.0
            _maxr = max(_maxr, _r)
            _stations.append({"name": b.name, "label": b.x.label,
                              "H": Ha, "V": Va, "kp": kp, "rpm": rpm})

        _dx = 1.0
        _S = (0.40 * _dx / _maxr) if _maxr > 0 else 1.0
        _xs = [i * _dx for i in range(len(_stations))]
        _u = _brgs[0].x.unit or "µm"

        fig = go.Figure()
        # línea de centros (eje sin deflexión) = referencia
        fig.add_trace(go.Scatter(x=[_xs[0] - 0.5 * _dx, _xs[-1] + 0.5 * _dx], y=[0, 0],
                      mode="lines", line=dict(color="#c0c8d4", width=1, dash="dot"),
                      hoverinfo="skip", showlegend=False))
        _kp = []
        for i, s in enumerate(_stations):
            x0 = _xs[i]
            fig.add_trace(go.Scatter(x=x0 + _S * s["H"], y=_S * s["V"], mode="lines",
                          line=dict(color=_NAVY, width=1.5), name=s["label"],
                          hovertemplate=f"{s['label']}<extra></extra>", showlegend=False))
            kx, ky = x0 + _S * s["kp"][0], _S * s["kp"][1]
            _kp.append((kx, ky))
            fig.add_trace(go.Scatter(x=[kx], y=[ky], mode="markers",
                          marker=dict(color="#1f9d55", size=11,
                                      line=dict(color="white", width=1.3)),
                          hoverinfo="skip", showlegend=False))
            fig.add_annotation(x=x0, y=-0.5 * _dx, text=s["label"], showarrow=False,
                               font=dict(size=10, family="IBM Plex Mono", color="#5a6b85"))
        # CURVA de deflexión = une los keyphasor (posición del eje en el disparo)
        fig.add_trace(go.Scatter(x=[p[0] for p in _kp], y=[p[1] for p in _kp],
                      mode="lines", line=dict(color="#e8890c", width=2.4),
                      name="deflexión (keyphasor)"))
        fig.update_xaxes(showticklabels=False, zeroline=False,
                         title="posición axial del tren →",
                         gridcolor="rgba(120,140,170,.10)")
        fig.update_yaxes(title=f"{_u}", zeroline=False,
                         gridcolor="rgba(120,140,170,.12)")
        fig = _layout(fig, f"Forma de deflexión del eje · {cur_asset} · {_ord}", h=470)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config=_NOBAR)
        _download_fig_button(fig, _vfile(f"Deflexion eje {cur_asset} {_ord}"), key="defl")
        st.caption("Cada órbita " + _ord + " en su posición axial; la **curva ámbar** une "
                   "los puntos **Keyphasor** (posición del eje en el disparo, muestra 0) → "
                   "**forma de deflexión aparente del eje** = forma modal del rotor "
                   "(cilíndrico/traslacional, pivotal/cónico…). Base: Bently manual 3-8 / 9-4. "
                   "Cojinetes ordenados por número (posición axial).")

elif view == "Shaft Centerline":
    _brgs = [bb for bb in rec.bearings if bb.has_xy]
    if not _brgs:
        st.warning("Shaft Centerline: necesita el **par X/Y** (gap DC de los proximitores). "
                   "Sube el export **Gap** de System1 (par 1XD/1YD) o usa el histórico.")
    else:
        st.markdown(
            "<style>"
            "div[data-testid='stPopover'] button{border-radius:9px;border:1px solid #cbd5e1;"
            "background:#fff;padding:5px 14px;font:600 12.5px 'IBM Plex Sans';color:#1e293b}"
            "div[data-testid='stPopover'] button:hover{border-color:#12305e;color:#12305e}"
            ".sc-bar{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin:0 0 10px}"
            ".sc-pill{background:#fff;border:1px solid #e2e8f2;border-radius:999px;"
            "padding:4px 13px;font:600 12px 'IBM Plex Sans';color:#475569;white-space:nowrap}"
            ".sc-pill b{color:#0f172a;font-weight:700}"
            ".sc-badge{color:#fff;border-radius:999px;padding:4px 13px;font:700 11px 'IBM Plex Mono';"
            "letter-spacing:.3px;white-space:nowrap}"
            ".sc-hint{color:#94a3b8;font:400 11.5px 'IBM Plex Sans';margin:6px 0 0}"
            "</style>", unsafe_allow_html=True)
        _tb1, _tb2 = st.columns([5, 1], vertical_alignment="center")
        with _tb1:
            _names = [bb.name for bb in _brgs]
            _seln = st.segmented_control("Cojinete", _names, default=_names[0],
                                         key="sc_b", label_visibility="collapsed")
            b = _brgs[_names.index(_seln)] if _seln in _names else _brgs[0]
        with _tb2:
            _pop = st.popover("⚙ Ajustes", use_container_width=True)

        _is_gap = getattr(b.x, "kind", "wave") == "gap"
        _rawx = np.asarray(b.x.v, float)
        _rawy = np.asarray(b.y.v, float)
        _spraw = np.asarray(getattr(b.x, "speed", np.array([])), float)
        _tsraw = list(getattr(b.x, "times", []) or [])
        _has_spd = _is_gap and _spraw.size == _rawx.size and np.isfinite(_spraw).any()
        _bn = str(getattr(b.x, "bearing", "") or getattr(b.y, "bearing", ""))
        _oem_clr = _SGT300_CLEARANCE_UM.get(_bn)
        _ntot = int(_rawx.size)

        def _fmt_ts(ts):  # "7/4/2026 10:09:04 PM" → "7/4/2026 10:09 PM"
            try:
                d, tm, ap = ts.split(" ")
                return f"{d} {':'.join(tm.split(':')[:2])} {ap}"
            except Exception:  # noqa: BLE001
                return ts or "—"

        # Controles en el popover flotante → la gráfica manda. Defaults ya buenos.
        with _pop:
            # --- recorte: ventana de muestras (orden de captura = línea de tiempo) ---
            if _ntot >= 3:
                _rng = st.slider("Rango de muestras (recorte · orden de captura)",
                                 0, _ntot - 1, (0, _ntot - 1),
                                 help="Data en orden de captura (arranque→parada). Recorta a un "
                                      "tramo: sólo arranque, quitar la parada, un rango de rpm…")
                _i0, _i1 = int(_rng[0]), int(_rng[1])
            else:
                _i0, _i1 = 0, max(_ntot - 1, 0)
            if _i1 - _i0 >= 2:
                _rawx = _rawx[_i0:_i1 + 1]
                _rawy = _rawy[_i0:_i1 + 1]
                if _spraw.size == _ntot:
                    _spraw = _spraw[_i0:_i1 + 1]
                if len(_tsraw) == _ntot:
                    _tsraw = _tsraw[_i0:_i1 + 1]
                _has_spd = _is_gap and _spraw.size == _rawx.size and np.isfinite(_spraw).any()
            _nwin = int(_rawx.size)
            if _nwin != _ntot:
                _wr = (f" · {np.nanmin(_spraw):.0f}–{np.nanmax(_spraw):.0f} rpm"
                       if _has_spd else "")
                st.caption(f"⬤ Ventana: {_i0}–{_i1} de {_ntot} muestras{_wr}.")

            # --- giro lento (slow roll): stepper estilo Bently (fecha·hora·rpm) ---
            if _is_gap and _nwin >= 2:
                if _has_spd:  # default = última muestra ≤ 8% del rango de rpm
                    _smin, _smax = float(np.nanmin(_spraw)), float(np.nanmax(_spraw))
                    _cand = np.where(np.isfinite(_spraw) & (_spraw <= _smin + 0.08 * (_smax - _smin)))[0]
                    _sr_def = int(_cand[-1]) if _cand.size else 0
                else:
                    _sr_def = 0
                _sel = int(st.number_input(
                    "Muestra de giro lento (reposo) ▲▼", min_value=0, max_value=_nwin - 1,
                    value=min(_sr_def, _nwin - 1), step=1,
                    help="Estilo Bently: avanza muestra a muestra (fecha·hora·rpm) y fija el "
                         "reposo. Reposo = promedio desde el inicio hasta esta muestra = muñón "
                         "en el FONDO del claro."))
                _ox = float(np.nanmean(_rawx[:_sel + 1]))
                _oy = float(np.nanmean(_rawy[:_sel + 1]))
                _ts_sel = _fmt_ts(_tsraw[_sel]) if _sel < len(_tsraw) else "—"
                _rpm_sel = _spraw[_sel] if _has_spd and _sel < _spraw.size else float("nan")
                _rpm_txt = f"{_rpm_sel:.0f} rpm" if np.isfinite(_rpm_sel) else "s/rpm"
                st.caption(f"⬤ Reposo = promedio muestras 0–{_sel} · fijado en "
                           f"{_ts_sel} · {_rpm_txt}.")
            else:
                _ox = float(_rawx[0]) if _is_gap and _rawx.size else 0.0
                _oy = float(_rawy[0]) if _is_gap and _rawy.size else 0.0

            # --- claro diametral ---
            _rise0 = float(np.nanmax(np.hypot(_rawx - _ox, _rawy - _oy))) if _rawx.size else 100.0
            _auto_clr = float(round(max(_rise0 * 2.2, 100.0) / 10) * 10)
            clr = st.number_input("Claro diametral del cojinete (µm)",
                                  value=_oem_clr if _oem_clr is not None else _auto_clr, step=10.0,
                                  help="Del montaje / API 670. Radio del claro = la mitad. "
                                       "Default OEM SGT-300: cojinete 1 (Inlet)=152, 2 (CT)=317 µm.")
            if _oem_clr is not None:
                st.caption(f"⬤ Claro OEM (dossier SGT-300) cojinete {_bn}: {_oem_clr:.0f} µm "
                           f"diametral · radio {_oem_clr / 2:.0f} µm.")

        # cambios de gap referidos al giro lento
        _dgx = _rawx - _ox
        _dgy = _rawy - _oy
        _m = np.isfinite(_dgx) & np.isfinite(_dgy)
        _dgx, _dgy = _dgx[_m], _dgy[_m]
        _sp = _spraw[_m] if _has_spd else np.array([])
        rad = clr / 2.0

        if _is_gap:
            # --- orientación física: sondas ±45° → marco máquina (H der+, V arriba+) ---
            _H0, _V0 = _gap_to_vh(_dgx, _dgy)
            # signo global: a velocidad el muñón SUBE (Bently 12-10) → V_operación > 0
            _iop = int(np.nanargmax(_sp)) if _sp.size else (_H0.size - 1)
            _sgn = -1.0 if _V0[_iop] < 0 else 1.0
            _H, _V = _sgn * _H0, _sgn * _V0
            # posición del muñón EN EL COJINETE: reposo en el fondo (0,−rad)
            _px = _H
            _py = _V - rad
            gx_op, gy_op = float(_px[_iop]), float(_py[_iop])
            eps = float(np.hypot(gx_op, gy_op) / rad) if rad > 0 else 0.0
            eps = min(eps, 1.2)
            _rise = float(_V[_iop])  # ascenso neto desde el fondo
            status, color = _ecc_status(eps)
            # ángulo de actitud: desde la carga (vertical abajo) a la línea centro→muñón
            _emag = float(np.hypot(gx_op, gy_op))
            _att = float(np.degrees(np.arccos(max(-1.0, min(1.0, -gy_op / _emag))))) if _emag > 1e-6 else 0.0
            _side = "derecha" if gx_op >= 0 else "izquierda"
        else:
            # fallback sin gap DC (canales de onda): no es un centerline real
            _px, _py = _dgx, _dgy
            gx_op, gy_op = (float(_px[-1]), float(_py[-1])) if _px.size else (0.0, 0.0)
            eps = float(np.hypot(gx_op, gy_op) / rad) if rad > 0 else 0.0
            _rise = _rise0
            status, color = _ecc_status(min(eps, 1.2))
            _att, _side = 0.0, ""

        fig = go.Figure()
        th = np.linspace(0, 2 * np.pi, 180)
        _a2 = np.sqrt(0.5)
        # cono de visión de cada sonda (sector tenue hacia el eje), detrás de todo
        if _is_gap:
            for _ux, _uy in [(_a2, -_a2), (-_a2, -_a2)]:
                _ang = np.arctan2(_uy, _ux)
                _aa = np.linspace(_ang - np.radians(14), _ang + np.radians(14), 20)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([[0], rad * 0.98 * np.cos(_aa), [0]]),
                    y=np.concatenate([[0], rad * 0.98 * np.sin(_aa), [0]]),
                    fill="toself", mode="lines", line=dict(color="rgba(18,48,94,0)"),
                    fillcolor="rgba(18,48,94,0.05)", hoverinfo="skip", showlegend=False))
        # claro (borde físico) + anillos de excentricidad de referencia
        fig.add_trace(go.Scatter(x=rad * np.cos(th), y=rad * np.sin(th), mode="lines",
                      line=dict(color="#334155", width=1.6), hoverinfo="skip",
                      name="claro", showlegend=False))
        for frac, c in [(0.95, "#dc3545"), (0.30, "#0ea5e9")]:
            fig.add_trace(go.Scatter(x=rad * frac * np.cos(th), y=rad * frac * np.sin(th),
                          mode="lines", line=dict(color=c, width=1, dash="dot"),
                          hoverinfo="skip", showlegend=False))
        # recorrido reposo→velocidad coloreado por rpm
        if _sp.size and np.isfinite(_sp).any():
            fig.add_trace(go.Scatter(x=_px, y=_py, mode="markers",
                          marker=dict(color=_sp, colorscale="Turbo", size=4,
                                      showscale=True, colorbar=dict(title="rpm", thickness=12)),
                          hovertemplate="H %{x:.1f} · V %{y:.1f} µm · %{marker.color:.0f} rpm<extra></extra>",
                          showlegend=False))
        fig.add_trace(go.Scatter(x=_px, y=_py, mode="lines",
                      line=dict(color="#94a3b8", width=0.9), hoverinfo="skip", showlegend=False))
        # centro del cojinete + flecha de carga (hacia abajo) + reposo (fondo)
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text",
                      marker=dict(color="#64748b", size=8, symbol="cross-thin",
                                  line=dict(color="#334155", width=2)),
                      text=["centro"], textposition="top right",
                      hoverinfo="skip", showlegend=False))
        if _is_gap:
            fig.add_annotation(x=0, y=-rad * 0.78, ax=0, ay=0, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=2,
                               arrowwidth=1.4, arrowcolor="#94a3b8", text="")
            fig.add_trace(go.Scatter(x=[0], y=[-rad], mode="markers+text",
                          marker=dict(color="#64748b", size=10, symbol="circle-open",
                                      line=dict(color="#334155", width=2)),
                          text=["reposo (fondo)"], textposition="bottom center",
                          hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[gx_op], y=[gy_op], mode="markers+text",
                      marker=dict(color=color, size=15, line=dict(color="#fff", width=1.5)),
                      text=[f"{b.x.rpm:.0f} rpm" if b.x.rpm else "operación"],
                      textposition="top center", hoverinfo="skip", showlegend=False))
        # marca TOPE 0° (referencia Bently) + sondas tipo dibujo técnico (cuerpo+rosca+cara)
        if _is_gap:
            fig.add_trace(go.Scatter(x=[0], y=[rad], mode="markers+text",
                          marker=dict(color="#0f172a", size=7, symbol="triangle-down"),
                          text=["TOPE 0°"], textposition="top center",
                          hoverinfo="skip", showlegend=False))
            for _lbl, _ux, _uy in [(b.x.label, _a2, -_a2), (b.y.label, -_a2, -_a2)]:
                _ri, _rb, _ro = rad * 1.03, rad * 1.22, rad * 1.42
                _perp = np.array([-_uy, _ux])
                fig.add_trace(go.Scatter(x=[_ux * _rb, _ux * _ro], y=[_uy * _rb, _uy * _ro],
                              mode="lines", line=dict(color="#12305e", width=10),
                              hoverinfo="skip", showlegend=False))
                for _t in (0.55, 0.70, 0.85):   # rosca
                    _cx, _cy = _ux * (_rb + (_ro - _rb) * _t), _uy * (_rb + (_ro - _rb) * _t)
                    fig.add_trace(go.Scatter(
                        x=[_cx - _perp[0] * rad * 0.05, _cx + _perp[0] * rad * 0.05],
                        y=[_cy - _perp[1] * rad * 0.05, _cy + _perp[1] * rad * 0.05],
                        mode="lines", line=dict(color="#8fa0bf", width=1.5),
                        hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=[_ux * _ri], y=[_uy * _ri], mode="markers",
                              marker=dict(color="#e8890c", size=12, symbol="circle",
                                          line=dict(color="#12305e", width=2)),
                              hoverinfo="skip", showlegend=False))
                _sd = "R" if _ux > 0 else "L"
                fig.add_annotation(x=_ux * _ro, y=_uy * _ro, text=f"{_lbl}<br>135°{_sd}",
                                   showarrow=False, yshift=-16,
                                   font=dict(size=11, family="IBM Plex Mono", color="#12305e"))
        if _is_gap:
            _xl, _yl = "Horizontal H (µm)", "Vertical V (µm) · arriba +"
        else:
            _xl = f"{b.x.label} (µm)"
            _yl = f"{b.y.label} (µm)"
        _axr = rad * (1.55 if _is_gap else 1.15)
        fig.update_xaxes(title=_xl, scaleanchor="y", scaleratio=1, zeroline=True,
                         range=[-_axr, _axr])
        fig.update_yaxes(title=_yl, zeroline=True, range=[-_axr, _axr])
        st.plotly_chart(_layout(fig, f"Shaft Centerline · {b.name}", h=540),
                        use_container_width=True, config=_NOBAR)
        _download_fig_button(fig, _vfile(f"Shaft Centerline {b.name}"), key="sc")
        if _is_gap:
            _attf = "#1f9d55" if 20 <= _att <= 50 else "#e8890c"
            _chips = (
                f"<div class='sc-bar'>"
                f"<span class='sc-badge' style='background:{color}'>{status}</span>"
                f"<span class='sc-pill'>ε <b>{eps:.2f}</b></span>"
                f"<span class='sc-pill' style='border-color:{_attf}44'>actitud "
                f"<b>{_att:.0f}°</b> {_side}</span>"
                f"<span class='sc-pill'>ascenso <b>{_rise:.0f} µm</b></span>"
                f"<span class='sc-pill'>claro <b>{clr:.0f} µm</b></span>"
                f"<span class='sc-pill'>op <b>{(b.x.rpm or 0):.0f} rpm</b></span></div>"
                f"<div class='sc-hint'>Círculo = claro · anillo rojo ε 0.95 (contacto) · "
                f"anillo azul ε 0.30 (riesgo whirl) · flecha = carga · sondas abajo "
                f"X 135°R / Y 135°L (Bently) → H/V real. ε 0=centro, 1=contacto.</div>")
        else:
            _chips = (
                f"<div class='sc-bar'><span class='sc-badge' style='background:{color}'>{status}</span>"
                f"<span class='sc-pill'>X <b>{gx_op:.1f}</b> · Y <b>{gy_op:.1f} µm</b></span>"
                f"<span class='sc-pill'>ε≈<b>{eps:.2f}</b></span></div>"
                f"<div class='sc-hint'>Sin gap DC: no es un centerline físico. "
                f"Sube el export <b>Gap</b> de System1.</div>")
        st.markdown(_chips, unsafe_allow_html=True)

elif view == "Bode":
    sw = _pick_sweep("bode_sw")
    if sw is None:
        st.warning("Bode: sube el **CSV de Bode/Polar de System1** (speed/rpm + "
                   "amplitude + phase).")
    else:
        op, unit, diag, s, zone, _ampc, _phc = _sweep_controls(sw, "bode")
        zc = _ISO.get(zone, "#8090a6")
        ab, bc, cd = (s.get("iso_boundary_AB"), s.get("iso_boundary_BC"),
                      s.get("iso_boundary_CD"))
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                            row_heights=[0.56, 0.44])
        amax = max(float(np.nanmax(_ampc)), cd or 0) * 1.12
        for lo, hi, z in [(0.0, ab, "A"), (ab, bc, "B"), (bc, cd, "C"), (cd, amax, "D")]:
            if lo is None or hi is None:
                continue
            fig.add_hrect(y0=lo, y1=hi, line_width=0, fillcolor=_ISO[z],
                          opacity=0.08, row=1, col=1)
        fig.add_trace(go.Scatter(x=sw.rpm, y=_ampc, mode="lines+markers",
                                 line=dict(color=_NAVY, width=1.9),
                                 marker=dict(size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=sw.rpm, y=_phc, mode="lines+markers",
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
        st.plotly_chart(_layout(fig, f"Bode · {sw.label}", h=560), use_container_width=True, config=_NOBAR)
        _download_fig_button(fig, _vfile(f"Bode {sw.label}"), key="bode")
        _diag_panel(diag, zone, zc)

elif view == "Polar":
    sw = _pick_sweep("polar_sw")
    if sw is None:
        st.warning("Polar: sube el **CSV de Bode/Polar de System1** (speed/rpm + "
                   "amplitude + phase).")
    else:
        op, unit, diag, s, zone, _ampc, _phc = _sweep_controls(sw, "polar")
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
            r=_ampc, theta=_phc, mode="lines+markers",
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
        st.plotly_chart(fig, use_container_width=True, config=_NOBAR)
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
                            use_container_width=True, config=_NOBAR)
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
