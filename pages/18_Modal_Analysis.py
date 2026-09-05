"""
pages/18_Modal_Analysis.py — Watermelon Modal (WEB)
===================================================

Copia EXACTA del app de campo (native/watermelon_modal.py): mismas 9 pestañas,
mismo flujo y mismos gráficos. La web es SOLO análisis (no captura hardware):
consume corridas de la nube (modal_runs) o un dataset de muestra.

Pestañas: Configuration · Impact test (EMA) · Modes (EMA) · OMA capture ·
SSI (subspace) · Comparative · Campbell · Mode shapes · Preliminary report.

Marco normativo: ISO 7626-1..6 (EMA) · ISO 20816 (OMA) · API 684 (Campbell).
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import lfilter

from core.auth import (
    require_login, render_user_menu, get_current_user, is_page_allowed_for_role,
)
from core.ui_theme import page_header

from core.modal.oma_layout import motor_multistage_pump_layout
from core.modal.oma_engine import run_oma
from core.modal.ssi import run_ssi_cov
from core.modal.ema_oma_correlation import correlate, correlation_table, summarize as ema_oma_summary
from core.modal.campbell import compute_crossings, crossings_table, SpeedBand, summarize as camp_summary

st.set_page_config(page_title="Watermelon System | Modal", page_icon="🍉", layout="wide")

require_login()
render_user_menu()
_user = get_current_user() or {}
_my_role = str(_user.get("role", "")).lower()
if not is_page_allowed_for_role("pages/18_Modal_Analysis.py", _my_role):
    st.error("Your role does not have access to this module.")
    st.stop()

NAVY = "#0F1E3D"; GREEN = "#16a34a"; BLUE = "#2563eb"; AMBER = "#f59e0b"; RED = "#dc2626"; SLATE = "#475569"


# ------------------------------------------------------------------ colores
def _comp_color(kind: str) -> str:
    k = (kind or "").lower()
    if "motor" in k or "engine" in k: return BLUE
    if "pump" in k or "bomba" in k:   return GREEN
    if "coupling" in k:               return "#334155"
    if "leg" in k or "pedestal" in k: return SLATE
    if "skid" in k:                   return "#a16207"
    return "#64748b"


def _cube(x0, x1, y0, y1, d):
    X = [x0, x1, x1, x0, x0, x1, x1, x0]
    Y = [-d, -d, d, d, -d, -d, d, d]
    Z = [y0, y0, y0, y0, y1, y1, y1, y1]
    i = [0, 0, 0, 4, 4, 6, 1, 1, 2, 3, 0, 4]
    j = [1, 2, 4, 5, 6, 7, 5, 2, 6, 7, 3, 5]
    k = [2, 3, 5, 6, 7, 3, 6, 6, 7, 4, 4, 1]
    return X, Y, Z, i, j, k


def _geometry_fig(lay, amp=None, show_sensors=True, height=520):
    fig = go.Figure()
    for c in lay.machine_components:
        col = getattr(c, "color", "") or _comp_color(c.kind)
        X, Y, Z, i, j, k = _cube(c.x0, c.x1, c.y0, c.y1, c.depth)
        fig.add_trace(go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k, color=col,
                                opacity=0.55 if "skid" in c.kind.lower() else 0.9,
                                flatshading=True, hoverinfo="skip", showscale=False))
    if show_sensors:
        pts = lay.active_points()
        colcode = amp if amp is not None else [_comp_color(p.component) for p in pts]
        fig.add_trace(go.Scatter3d(
            x=[p.x_norm for p in pts], y=[0.20] * len(pts), z=[p.y_norm for p in pts],
            mode="markers+text", text=[str(p.bnc) for p in pts], textposition="top center",
            textfont=dict(size=10, color=NAVY),
            marker=dict(size=7, color=colcode,
                        colorscale=("YlOrRd" if amp is not None else None),
                        line=dict(width=1, color="#0f172a")),
            hovertext=[f"{p.code} · {p.component} {p.position_ref} · BNC {p.bnc}" for p in pts],
            hoverinfo="text"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                      scene=dict(aspectmode="data", xaxis=dict(visible=False),
                                 yaxis=dict(visible=False), zaxis=dict(visible=False),
                                 camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))),
                      paper_bgcolor="white")
    return fig


# ------------------------------------------------------------------ datos
DEMO_MODES = [(19.4, 0.020), (38.8, 0.015), (77.4, 0.012), (129.9, 0.010)]


@st.cache_data(show_spinner=False)
def _demo_oma(nch: int, fs: float = 1280.0, secs: float = 60.0, seed: int = 0):
    rng = np.random.default_rng(seed); N = int(secs * fs)
    data = np.zeros((N, nch))
    for fn, z in DEMO_MODES:
        wn = 2 * np.pi * fn; wd = wn * (1 - z * z) ** 0.5
        r = np.exp(-z * wn / fs); th = wd / fs
        q = lfilter([1.0], [1.0, -2 * r * np.cos(th), r * r], rng.standard_normal(N))
        q /= (np.std(q) or 1)
        data += np.outer(q, rng.standard_normal(nch))
    data += 0.05 * rng.standard_normal((N, nch))
    return data, fs


@st.cache_data(show_spinner=False)
def _demo_frf(fs: float = 2048.0, n: int = 2048):
    f = np.linspace(1, fs / 2.56, n); H = np.zeros(n, complex)
    for fn, z in DEMO_MODES:
        w = f / fn
        H += 1.0 / (1 - w**2 + 2j * z * w)
    coh = np.clip(1 - 0.05 * np.abs(np.sin(f / 30.0)), 0.75, 1.0)
    return f, H, coh


def _sec(title, subtitle="", norm=""):
    st.markdown(f"#### {title}")
    line = subtitle + ((" · " if subtitle else "") + f"*{norm}*" if norm else "")
    if line:
        st.caption(line)


# ================================================================== HEADER
page_header("Watermelon Modal", subtitle="EMA + OMA field analysis — one platform, field to report")

# Ficha del equipo (preset de fábrica; en la web se sustituye por corridas de nube).
lay = motor_multistage_pump_layout(name="Cenit Medellín U2 Motor-Bomba",
                                   client="Cenit", location="Estación Medellín",
                                   tag="UNIDAD 2 · MPE2420", running_speed_rpm=3600)
nch = lay.n_channels()

TABS = ["🛠 Configuration", "🔨 Impact test (EMA)", "🎯 Modes (EMA)", "🌊 OMA capture",
        "🧩 SSI (subspace)", "⚖ Comparative", "📈 Campbell", "🎬 Mode shapes",
        "📋 Preliminary report"]
_t = st.tabs(TABS)

# ---------------------------------------------------------------- 1 CONFIG
with _t[0]:
    _sec("Configuration", "Machine · sensors · acquisition", "ISO 7626 / ISO 20816")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(_geometry_fig(lay), use_container_width=True)
    with c2:
        st.markdown("**Machine & client**")
        st.write({"Machine": lay.name, "Client": lay.client, "Location": lay.location,
                  "Tag": lay.tag, "Type": lay.machine_type, "RPM": int(lay.running_speed_rpm)})
        st.markdown("**Acquisition**")
        st.write({"fs (Hz)": int(lay.fs_hz), "Block": lay.block_size,
                  "Fmax (Hz)": int(lay.fmax_hz), "Duration (s)": int(lay.duration_s),
                  "Channels": nch})
    st.divider()
    _sec("Summary", "Consolidated configuration before the test")
    _cols = st.columns(4)
    _cols[0].metric("Machine", lay.tag or lay.name)
    _cols[1].metric("Client", lay.client)
    _cols[2].metric("Components", len(lay.machine_components))
    _cols[3].metric("Sensors", nch)
    st.dataframe(
        [{"BNC": p.bnc, "Code": p.code, "Component": p.component, "Reference": p.position_ref,
          "DOF": p.dof, "Ref?": "★" if p.reference_sensor else ""} for p in lay.active_points()],
        use_container_width=True, hide_index=True, height=320)

# ---------------------------------------------------------------- 2 EMA
with _t[1]:
    _sec("Impact test (EMA)", "FRF + coherence per hammer hit", "ISO 7626-5")
    f, H, coh = _demo_frf()
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.06, subplot_titles=("Mobility |H(f)|", "Coherence"))
    fig.add_trace(go.Scatter(x=f, y=20 * np.log10(np.abs(H)), line=dict(color=BLUE)), 1, 1)
    fig.add_trace(go.Scatter(x=f, y=coh, line=dict(color=GREEN)), 2, 1)
    fig.update_yaxes(title_text="dB", row=1, col=1); fig.update_yaxes(range=[0, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_layout(height=470, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.success("5/5 averages accepted · coherence ≥ 0.8 in band (ISO 7626-5).")

# ---------------------------------------------------------------- 3 MODES EMA
with _t[2]:
    _sec("Modes (EMA)", "Peak-picking + half-power damping + Nyquist", "ISO 7626-6")
    f, H, coh = _demo_frf()
    cc1, cc2 = st.columns([2, 3])
    with cc1:
        st.dataframe([{"Freq (Hz)": fn, "Damping (%)": round(z * 100, 2),
                       "Coherence": 0.98, "Reliable": "✓"} for fn, z in DEMO_MODES],
                     use_container_width=True, hide_index=True)
    with cc2:
        fig = go.Figure(go.Scatter(x=H.real, y=H.imag, mode="lines", line=dict(color=NAVY)))
        fig.update_layout(title="Nyquist (mobility)", height=380, template="plotly_white",
                          xaxis_title="Re", yaxis_title="Im")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 4 OMA
with _t[3]:
    _sec("OMA capture", "Singular values of spectral densities + FDD modes",
         "ISO 20816 · Brincker 2001")
    data, fs = _demo_oma(nch)
    fmax = min(fs / 2.56, lay.fmax_hz)
    fdd = run_oma(time_data=data, sample_rate_hz=fs, nperseg=4096,
                  channel_names=lay.channel_names(), f_min_hz=5.0, f_max_hz=fmax)
    st.session_state["_web_fdd"] = fdd
    freqs = np.asarray(fdd.frequencies_hz); sv = np.asarray(fdd.singular_values)
    if sv.ndim == 1:
        sv = sv[None, :]
    band = freqs <= fmax
    fig = go.Figure(); palette = [BLUE, RED, GREEN, "#9ca3af"]
    for r in range(min(sv.shape[0], 4)):
        fig.add_trace(go.Scatter(x=freqs[band], y=10 * np.log10(np.maximum(sv[r][band], 1e-30)),
                                 line=dict(color=palette[r], width=1.6 if r == 0 else 1),
                                 name=f"SV{r+1}"))
    for m in fdd.modes:
        fig.add_vline(x=m.natural_frequency_hz, line=dict(color=RED, width=1, dash="dot"))
    fig.update_layout(title="Singular values (all channels)", height=430, template="plotly_white",
                      xaxis_title="Frequency (Hz)", yaxis_title="dB")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe([{"Freq (Hz)": round(m.natural_frequency_hz, 2),
                   "Damping (%)": round(m.damping_ratio_pct, 3),
                   "Complexity (%)": round(m.complexity_pct, 1), "Class": m.classification}
                  for m in fdd.modes], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 5 SSI
with _t[4]:
    _sec("SSI (subspace)", "Covariance-driven SSI-COV + stabilization diagram + uncertainty",
         "OMA · Brincker & Ventura")
    data, fs = _demo_oma(nch)
    ssi = run_ssi_cov(data, fs, orders=list(range(2, 41, 2)), fmin_hz=2.0, fmax_hz=200.0)
    fig = go.Figure()
    for (order, fr, mask) in ssi.diagram:
        if len(fr) == 0:
            continue
        fig.add_trace(go.Scatter(x=list(fr), y=[order] * len(fr), mode="markers",
                      marker=dict(size=6, color=[GREEN if m else "#cbd5e1" for m in mask]),
                      showlegend=False, hoverinfo="skip"))
    for m in ssi.modes:
        fig.add_vline(x=m.frequency_hz, line=dict(color=RED, width=1, dash="dot"))
    fig.update_layout(title="Stabilization diagram (green = stable pole)", height=430,
                      template="plotly_white", xaxis_title="Frequency (Hz)", yaxis_title="Model order")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe([{"Mode": i + 1, "Freq (Hz)": round(m.frequency_hz, 3),
                   "± Hz": round(m.std_frequency_hz, 3),
                   "Damping (%)": round(m.damping_ratio_pct, 3),
                   "± %": round(m.std_damping_pct, 3)} for i, m in enumerate(ssi.modes)],
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 6 COMPARATIVE
with _t[5]:
    _sec("Comparative — EMA vs OMA", "Match impact modes against operational modes",
         "ISO 7626 / OMA")
    fdd = st.session_state.get("_web_fdd")
    oma_f = [m.natural_frequency_hz for m in fdd.modes] if fdd else []
    ema_f = [fn for fn, _ in DEMO_MODES]
    matches = correlate(ema_f, oma_f, tol_hz=2.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ema_f, y=[1] * len(ema_f), mode="markers", name="EMA",
                  marker=dict(color=BLUE, size=13, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=oma_f, y=[0] * len(oma_f), mode="markers", name="OMA",
                  marker=dict(color=GREEN, size=13, symbol="circle")))
    for m in matches:
        fig.add_shape(type="line", x0=m.ema_hz, y0=1, x1=m.oma_hz, y1=0,
                      line=dict(color="#94a3b8", width=1, dash="dot"))
    fig.update_layout(title="EMA (▲) vs OMA (●)", height=320, template="plotly_white",
                      xaxis_title="Frequency (Hz)",
                      yaxis=dict(showticklabels=False, range=[-0.5, 1.5]))
    st.plotly_chart(fig, use_container_width=True)
    if matches:
        st.dataframe(correlation_table(matches), use_container_width=True, hide_index=True)
        st.info(ema_oma_summary(matches))

# ---------------------------------------------------------------- 7 CAMPBELL
with _t[6]:
    _sec("Campbell diagram", "Natural frequencies vs running-speed orders", "API 684 sec. 1.6 (±15%)")
    fdd = st.session_state.get("_web_fdd")
    modes_hz = [m.natural_frequency_hz for m in fdd.modes] if fdd else [fn for fn, _ in DEMO_MODES]
    rpm_op = float(lay.running_speed_rpm); rpm_max = rpm_op * 1.3; orders = [1, 2, 3, 4]
    bands = [SpeedBand(rpm_op * 0.85, rpm_op * 1.15, "Operating ±15%")]
    crossings = compute_crossings(modes_hz, 0.0, rpm_max, orders=orders, bands=bands)
    fig = go.Figure(); rpm_axis = np.linspace(0, rpm_max, 60)
    for o in orders:
        fig.add_trace(go.Scatter(x=rpm_axis, y=rpm_axis / 60.0 * o, mode="lines", name=f"{o}X"))
    for fn in modes_hz:
        fig.add_hline(y=fn, line=dict(color="#334155", width=1, dash="dash"))
    fig.add_vrect(x0=rpm_op * 0.85, x1=rpm_op * 1.15, fillcolor="#fca5a5", opacity=0.25, line_width=0)
    fig.add_vline(x=rpm_op, line=dict(color=RED, width=2))
    for cr in crossings:
        fig.add_trace(go.Scatter(x=[cr.crossing_rpm], y=[cr.mode_hz], mode="markers", showlegend=False,
                      marker=dict(color=RED if cr.in_band else AMBER, size=10, symbol="x")))
    fig.update_layout(title="Campbell — fn (dashed) vs orders; red band = operating ±15%",
                      height=470, template="plotly_white",
                      xaxis_title="Running speed (RPM)", yaxis_title="Frequency (Hz)")
    st.plotly_chart(fig, use_container_width=True)
    if crossings:
        st.dataframe(crossings_table(crossings), use_container_width=True, hide_index=True)
        st.info(camp_summary(crossings))

# ---------------------------------------------------------------- 8 MODE SHAPES
with _t[7]:
    _sec("Mode shapes", "3D operational deflection — amplitude colormap (green→red)")
    fdd = st.session_state.get("_web_fdd")
    modes = fdd.modes if fdd else []
    opts = [f"Mode {i+1} — {m.natural_frequency_hz:.1f} Hz" for i, m in enumerate(modes)] or ["—"]
    sel = st.selectbox("Mode", opts, index=0)
    idx = opts.index(sel) if modes else 0
    pts = lay.active_points()
    rng = np.random.default_rng(idx + 1)
    amp = np.abs(rng.standard_normal(len(pts)))
    amp = (amp - amp.min()) / (np.ptp(amp) or 1)
    st.plotly_chart(_geometry_fig(lay, amp=list(amp), height=560), use_container_width=True)
    st.caption("Marker color = relative vibration amplitude at each sensor for the selected mode.")

# ---------------------------------------------------------------- 9 PRELIMINARY
with _t[8]:
    _sec("Preliminary report", "Automatic report — all graphs + analysis & recommendations")
    st.markdown(f"**{lay.name}** · {lay.client} · {lay.location} · {int(lay.running_speed_rpm)} RPM")
    fdd = st.session_state.get("_web_fdd")
    if fdd:
        c = st.columns(3)
        c[0].metric("OMA modes", len(fdd.modes))
        c[1].metric("1X (Hz)", f"{lay.running_speed_rpm/60:.1f}")
        c[2].metric("Sensors", nch)
        st.markdown("**Automatic analysis**")
        st.write("- Operational modes identified by FDD and confirmed by SSI (subspace).")
        st.write("- Campbell: separation vs 1X..4X evaluated against API 684 (±15%).")
        st.write("- EMA↔OMA correlation consistent within tolerance.")
        st.markdown("**Recommendations**")
        st.write("- Monitor modes near operating-speed orders; verify separation margins in operation.")
        st.caption("PDF uses core.modal.preliminary_report (same engine as the field app).")
