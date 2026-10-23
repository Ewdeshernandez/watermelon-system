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

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.auth import require_login, render_user_menu

st.set_page_config(page_title="Watermelon System | Análisis", layout="wide",
                   initial_sidebar_state="collapsed")
require_login()
render_user_menu()

from core.analysis.model import AnalysisRecord, build_record
from core.ui_industrial import inject_industrial_css
from core.ui_theme import apply_watermelon_page_style, page_header

apply_watermelon_page_style()
inject_industrial_css()
page_header("Análisis", "Ingesta manual de CSV · todas las vistas en un solo lugar")

SS = st.session_state
_NAVY = "#12305e"
_AMBER = "#e8890c"


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
def _trend(caps_tuple, point, metric):
    from core.analysis.model import trend_series
    caps = [{"point": p, "day": d, "time": t, "key": k}
            for p, d, t, k in caps_tuple]
    return trend_series(caps, point, metric)


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
st.caption(f"Registro · {rec.name}  ·  {len(rec.channels)} canal(es)  ·  "
           f"{len(rec.bearings)} cojinete(s) con par X/Y"
           + (f"  ·  {_rpm:.0f} rpm" if _rpm else ""))


# =========================================================
# 2) VISTAS (segmented control) — dato compartido
# =========================================================
VIEWS = ["Waveform", "Espectro", "Órbita", "Shaft Centerline", "Polar", "Bode", "Trends"]
view = st.segmented_control("Vista", VIEWS, default="Waveform",
                            selection_mode="single") or "Waveform"


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


# ---- Vistas de UNA captura ----
if view == "Waveform":
    ch = _pick_channel("wf_ch")
    fig = go.Figure(go.Scatter(x=ch.t, y=ch.v, mode="lines",
                               line=dict(color=_NAVY, width=1.3)))
    fig.update_xaxes(title="Tiempo (s)")
    fig.update_yaxes(title=f"{ch.label} ({ch.unit or 'a.u.'})")
    st.plotly_chart(_layout(fig, f"Forma de onda · {ch.label}"), use_container_width=True)

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

elif view == "Órbita":
    b = _pick_bearing("orb_b")
    if b is None:
        st.warning("Órbita: necesita el **par X/Y** de un cojinete. Sube ambos CSV "
                   "(ej. 1xd.csv + 1yd.csv).")
    else:
        n = min(b.x.v.size, b.y.v.size)
        xv, yv = b.x.v[:n], b.y.v[:n]
        fig = go.Figure(go.Scatter(x=xv, y=yv, mode="lines",
                                   line=dict(color=_NAVY, width=1.2)))
        fig.add_trace(go.Scatter(x=[xv[0]], y=[yv[0]], mode="markers",
                                 marker=dict(color=_AMBER, size=9)))
        fig.update_xaxes(title=f"{b.x.label} ({b.x.unit or 'µm'})",
                         scaleanchor="y", scaleratio=1, zeroline=True)
        fig.update_yaxes(title=f"{b.y.label} ({b.y.unit or 'µm'})", zeroline=True)
        st.plotly_chart(_layout(fig, f"Órbita · {b.name}", h=480), use_container_width=True)
        st.caption("Órbita cruda (sin filtrar). El punto ámbar = inicio. Filtrado 1X con "
                   "keyphasor: en el roadmap.")

elif view == "Shaft Centerline":
    b = _pick_bearing("sc_b")
    if b is None:
        st.warning("Shaft Centerline: necesita el **par X/Y** (gap DC de los proximitores).")
    else:
        gx, gy = float(np.nanmean(b.x.v)), float(np.nanmean(b.y.v))
        fig = go.Figure()
        th = np.linspace(0, 2 * np.pi, 120)
        r = max(abs(gx), abs(gy), 1.0) * 1.6
        fig.add_trace(go.Scatter(x=r * np.cos(th), y=r * np.sin(th), mode="lines",
                                 line=dict(color="#cbd5e1", width=1)))
        fig.add_trace(go.Scatter(x=[gx], y=[gy], mode="markers+text",
                                 marker=dict(color=_NAVY, size=13),
                                 text=["eje"], textposition="top center"))
        fig.update_xaxes(title=f"{b.x.label}", scaleanchor="y", scaleratio=1, zeroline=True)
        fig.update_yaxes(title=f"{b.y.label}", zeroline=True)
        st.plotly_chart(_layout(fig, f"Shaft Centerline · {b.name}", h=460),
                        use_container_width=True)
        st.caption(f"Posición DC del eje: X={gx:.1f}, Y={gy:.1f}. El claro real "
                   "depende del montaje — referencia relativa.")

# ---- Vistas de BARRIDO (Bode/Polar) desde CSV de System1 ----
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


if view == "Bode":
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
        st.caption(f"Radio = amplitud ({unit}) · ángulo = fase · color = rpm · "
                   "anillos punteados = límites ISO 20816-2 (A/B/C).")
        _diag_panel(diag, zone, zc)

else:  # Trends
    if not hist_caps:
        st.info("**Trends** — overall vs tiempo. Requiere varias capturas en el tiempo "
                "(modo **Histórico del activo**). En CSV manual de una captura no aplica.")
    else:
        points = sorted({c.get("point", "") for c in hist_caps if c.get("point")})
        cc1, cc2 = st.columns([2, 1])
        with cc1:
            pt = st.selectbox("Punto", points, key="tr_pt")
        with cc2:
            met = st.selectbox("Métrica", ["Overall pp", "RMS", "1X amp"], key="tr_met")
        mk = {"Overall pp": "pp", "RMS": "rms", "1X amp": "1x"}[met]
        _ct = tuple((c["point"], c["day"], c["time"], c["key"]) for c in hist_caps
                    if c.get("point") and c.get("day") and c.get("time") and c.get("key"))
        with st.spinner("Calculando tendencia del histórico…"):
            ts, vals, unit = _trend(_ct, pt, mk)
        if not ts:
            st.warning(f"Sin datos para la tendencia de {pt}.")
        else:
            fig = go.Figure(go.Scatter(x=ts, y=vals, mode="lines+markers",
                                       line=dict(color=_NAVY, width=1.6),
                                       marker=dict(size=5, color=_NAVY)))
            fig.update_xaxes(title="Tiempo")
            fig.update_yaxes(title=f"{met} ({unit or 'a.u.'})")
            st.plotly_chart(_layout(fig, f"Tendencia · {pt} · {met}", h=440),
                            use_container_width=True)
            st.caption(f"{len(ts)} capturas · {pt} · métrica sobre el canal X.")

st.caption("Workspace unificado (Fase 2) · ingesta + vistas de una captura. "
           "Barrido/histórico (Polar·Bode·Trends) en integración.")
