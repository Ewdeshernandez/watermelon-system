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


# =========================================================
# 1) FUENTE — histórico automático (Supabase) o CSV manual
# =========================================================
src = st.radio("Fuente", ["Histórico del activo (automático)", "CSV manual (System1 / ADRE)"],
               horizontal=True, key="an_src")
rec: AnalysisRecord | None = None

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


if view == "Bode":
    sw = _pick_sweep("bode_sw")
    if sw is None:
        st.warning("Bode: sube el **CSV de Bode/Polar de System1** (tabla con "
                   "columnas speed/rpm + amplitude + phase).")
    else:
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            row_heights=[0.55, 0.45])
        fig.add_trace(go.Scatter(x=sw.rpm, y=sw.amp, mode="lines+markers",
                                 line=dict(color=_NAVY, width=1.6),
                                 marker=dict(size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=sw.rpm, y=sw.phase, mode="lines+markers",
                                 line=dict(color=_AMBER, width=1.6),
                                 marker=dict(size=4)), row=2, col=1)
        fig.update_yaxes(title="Amplitud", row=1, col=1)
        fig.update_yaxes(title="Fase (°)", row=2, col=1)
        fig.update_xaxes(title="Velocidad (rpm)", row=2, col=1)
        st.plotly_chart(_layout(fig, f"Bode · {sw.label}", h=520), use_container_width=True)

elif view == "Polar":
    sw = _pick_sweep("polar_sw")
    if sw is None:
        st.warning("Polar: sube el **CSV de Bode/Polar de System1** (speed/rpm + "
                   "amplitude + phase).")
    else:
        fig = go.Figure(go.Scatterpolar(
            r=sw.amp, theta=sw.phase, mode="lines+markers",
            line=dict(color=_NAVY, width=1.6), marker=dict(size=5,
            color=sw.rpm, colorscale="Blues", showscale=True,
            colorbar=dict(title="rpm", thickness=12))))
        fig.update_layout(template="plotly_white", height=520,
                          title=dict(text=f"Polar · {sw.label}", font=dict(size=15)),
                          font=dict(family="IBM Plex Sans"),
                          polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
                          margin=dict(l=40, r=40, t=56, b=40))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Radio = amplitud · ángulo = fase · color = rpm.")

else:  # Trends
    st.info("**Trends** — overall de vibración vs **tiempo** (histórico del activo). "
            "Fuente: varias capturas en el tiempo (Live Monitoring / histórico), "
            "no una sola captura manual. Integración al workspace: en el roadmap.")

st.caption("Workspace unificado (Fase 2) · ingesta + vistas de una captura. "
           "Barrido/histórico (Polar·Bode·Trends) en integración.")
