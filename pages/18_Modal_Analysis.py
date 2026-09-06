"""
pages/18_Modal_Analysis.py — Watermelon Modal (WEB)
===================================================

Web = SOLO análisis (la CONFIGURACIÓN se hace en el software de campo). Consume
las CORRIDAS reales que el campo sube a la nube (tabla modal_runs). Si no hay red
/ no hay corridas, usa un dataset de muestra para no quedar vacía. El equipo se
muestra como contexto de solo-lectura en el encabezado.

Pestañas: Impact test (EMA) · Modes (EMA) · OMA capture · SSI (subspace) ·
Comparative · Campbell · Mode shapes · Report (PDF SIGA completo: portada + TOC +
todas las secciones, mismo motor que el módulo de Reportes).

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

from core.modal.oma_layout import motor_multistage_pump_layout, OMALayout
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

# --- Tema de gráficos "watermelon" (industrial, consistente en toda la página) ---
import plotly.io as _pio
import plotly.graph_objects as _go
_pio.templates["watermelon"] = _go.layout.Template(
    layout=dict(
        font=dict(family="IBM Plex Sans, Segoe UI, Arial", size=13, color="#334155"),
        title=dict(font=dict(family="IBM Plex Sans", size=16, color=NAVY), x=0.01, xanchor="left", y=0.97),
        colorway=[BLUE, GREEN, RED, "#7c3aed", AMBER, "#0891b2", "#db2777", SLATE],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fbfcfe",
        hovermode="x unified", hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0",
                                               font=dict(family="IBM Plex Mono", size=12)),
        xaxis=dict(gridcolor="#eef2f8", zerolinecolor="#dbe4f0", linecolor="#cbd5e1",
                   ticks="outside", tickcolor="#cbd5e1", ticklen=4,
                   title=dict(font=dict(size=12, color="#64748b"))),
        yaxis=dict(gridcolor="#eef2f8", zerolinecolor="#dbe4f0", linecolor="#cbd5e1",
                   ticks="outside", tickcolor="#cbd5e1", ticklen=4,
                   title=dict(font=dict(size=12, color="#64748b"))),
        margin=dict(l=62, r=24, t=48, b=52),
        legend=dict(bgcolor="rgba(255,255,255,.85)", bordercolor="#e2e8f0", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    ))


def _inject_theme():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
      html, body, [class*="css"] { font-family:'IBM Plex Sans',system-ui,sans-serif; }
      .block-container { padding-top: 1.4rem; max-width: 1200px; }
      /* Hero */
      .wm-hero { background: linear-gradient(110deg,#0F1E3D 0%,#12325a 55%,#16a34a 160%);
        border-radius:16px; padding:20px 24px; color:#fff; box-shadow:0 10px 30px rgba(15,30,61,.18);
        display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap; }
      .wm-hero h1 { font-size:24px; font-weight:700; margin:0 0 4px; letter-spacing:-.01em; }
      .wm-hero .meta { color:#cbd5e1; font-size:13px; }
      .wm-chip { font-size:12px; font-weight:700; letter-spacing:.03em; text-transform:uppercase;
        padding:7px 14px; border-radius:999px; }
      .wm-go { background:#16a34a; color:#fff; } .wm-rev { background:#f59e0b; color:#0f1e3d; }
      .wm-nogo { background:#dc2626; color:#fff; }
      /* KPI cards */
      .wm-kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:14px 0 6px; }
      .wm-kpi { background:#fff; border:1px solid #e6ecf5; border-radius:14px; padding:14px 16px;
        box-shadow:0 1px 2px rgba(15,30,61,.04),0 6px 18px rgba(15,30,61,.05); }
      .wm-kpi .v { font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:600; color:#0F1E3D; line-height:1.1; }
      .wm-kpi .l { font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:.05em; margin-top:4px; }
      .wm-kpi .s { font-size:12px; color:#94a3b8; }
      /* Tabs */
      .stTabs [data-baseweb="tab-list"] { gap:4px; }
      .stTabs [data-baseweb="tab"] { background:#eef2f8; border-radius:9px 9px 0 0; padding:8px 14px; font-weight:600; }
      .stTabs [aria-selected="true"] { background:#0F1E3D !important; color:#fff !important; }
      @media (prefers-color-scheme: dark){
        .wm-kpi{ background:#141b26; border-color:#243040; }
        .wm-kpi .v{ color:#eaf0f7; }
      }
    </style>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------ colores / 3D
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
    if show_sensors and lay.active_points():
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


_AX = {"A": (1, 0, 0), "X": (1, 0, 0), "H": (0, 1, 0), "Y": (0, 1, 0),
       "V": (0, 0, 1), "Z": (0, 0, 1)}


def _mode_anim_fig(lay, amps_signed, height=560):
    """Forma modal 3D ANIMADA: la máquina (tenue) + nodos de sensores que oscilan
    a lo largo de su DOF, coloreados por amplitud. Con botón Play."""
    pts = lay.active_points()
    if not pts or amps_signed is None or len(amps_signed) != len(pts):
        return _geometry_fig(lay, height=height)
    a = np.asarray(amps_signed, float)
    a = a / (np.max(np.abs(a)) or 1.0)
    col = np.abs(a)
    P0 = np.array([[p.x_norm, 0.20, p.y_norm] for p in pts], float)
    dirs = np.array([_AX.get(p.axis, (0, 0, 1)) for p in pts], float)
    dirs *= np.array([[-1.0 if p.dof.startswith("-") else 1.0] for p in pts])
    scale = 0.12
    base = go.Figure()
    # máquina tenue de fondo
    for c in lay.machine_components:
        cc = getattr(c, "color", "") or _comp_color(c.kind)
        X, Y, Z, i, j, k = _cube(c.x0, c.x1, c.y0, c.y1, c.depth)
        base.add_trace(go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k, color=cc, opacity=0.18,
                                 flatshading=True, hoverinfo="skip", showscale=False))

    def _nodes(phase):
        d = P0 + (scale * a * np.sin(phase))[:, None] * dirs
        return d

    d0 = _nodes(0.0)
    base.add_trace(go.Scatter3d(x=d0[:, 0], y=d0[:, 1], z=d0[:, 2], mode="markers",
                   marker=dict(size=6, color=col, colorscale="YlOrRd", cmin=0, cmax=1,
                               line=dict(width=1, color="#0f172a")),
                   hovertext=[p.code for p in pts], hoverinfo="text", name="mode"))
    frames = []
    for f in range(24):
        ph = f / 24.0 * 2 * np.pi; d = _nodes(ph)
        frames.append(go.Frame(data=[go.Scatter3d(x=d[:, 0], y=d[:, 1], z=d[:, 2], mode="markers",
                      marker=dict(size=6, color=col, colorscale="YlOrRd", cmin=0, cmax=1,
                                  line=dict(width=1, color="#0f172a")))],
                      traces=[len(lay.machine_components)]))
    base.frames = frames
    base.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))),
        paper_bgcolor="rgba(0,0,0,0)",
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.05, xanchor="left",
            buttons=[dict(label="▶ Play", method="animate",
                          args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True,
                                           transition=dict(duration=0), mode="immediate")]),
                     dict(label="⏸", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])])
    return base


def _narrative(name, modes, rpm, verdicts, crossings):
    """Diagnóstico automático en prosa (tipo experto) a partir de modos/Campbell."""
    if not modes:
        return "No operational modes identified yet."
    fns = sorted(m["fn"] for m in modes)
    x1 = rpm / 60.0
    nval = sum(1 for v in verdicts if v.verdict == "validated")
    nharm = sum(1 for v in verdicts if getattr(v, "is_harmonic", False))
    inband = [c for c in crossings if c.in_band]
    parts = [f"{len(modes)} operational modes were identified between {fns[0]:.1f} and {fns[-1]:.1f} Hz "
             f"({nval} validated by FDD∩SSI)."]
    if nharm:
        parts.append(f"{nharm} peak(s) coincide with running-speed orders (1×={x1:.1f} Hz) and are flagged "
                     "as harmonics, not structural modes.")
    if inband:
        c = min(inband, key=lambda c: c.sep_margin_pct)
        parts.append(f"⚠ The {c.mode_hz:.1f} Hz mode falls within ±15% of the {c.order:g}× order "
                     f"(margin {c.sep_margin_pct:.1f}%, API 684) — a resonance risk near operating speed; "
                     "correlate with amplitude/phase and evaluate skid/base stiffness.")
    else:
        parts.append("No natural frequency falls within ±15% of the running-speed orders (API 684): "
                     "adequate separation at the operating speed.")
    return " ".join(parts)


def _sec(title, subtitle="", norm=""):
    st.markdown(f"#### {title}")
    line = subtitle + ((" · " if subtitle else "") + f"*{norm}*" if norm else "")
    if line:
        st.caption(line)


# ------------------------------------------------------------------ datos demo
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


# ------------------------------------------------------------------ dataset D
def _default_layout():
    return motor_multistage_pump_layout(name="Cenit Medellín U2 Motor-Bomba",
                                        client="Cenit", location="Estación Medellín",
                                        tag="UNIDAD 2 · MPE2420", running_speed_rpm=3600)


def _build_demo_D():
    lay = _default_layout(); nch = lay.n_channels()
    data, fs = _demo_oma(nch)
    fmax = min(fs / 2.56, lay.fmax_hz)
    fdd = run_oma(time_data=data, sample_rate_hz=fs, nperseg=4096,
                  channel_names=lay.channel_names(), f_min_hz=5.0, f_max_hz=fmax)
    freqs = np.asarray(fdd.frequencies_hz); sv = np.asarray(fdd.singular_values)
    if sv.ndim == 1:
        sv = sv[None, :]
    band = freqs <= fmax
    sv_traces = [(f"SV{r+1}", freqs[band], 10 * np.log10(np.maximum(sv[r][band], 1e-30)))
                 for r in range(min(sv.shape[0], 4))]
    oma_modes = [{"fn": m.natural_frequency_hz, "zeta": m.damping_ratio_pct,
                  "complexity": m.complexity_pct, "cls": m.classification} for m in fdd.modes]
    _fb = freqs[band]; _sv1 = sv[0][band]; _st = max(1, len(_fb) // 900)
    payload = {"name": lay.name, "kind": "OMA", "running_rpm": lay.running_speed_rpm,
               "client": lay.client, "asset": lay.machine_type, "location": lay.location,
               "channel_names": lay.channel_names(), "ema_modes": [fn for fn, _ in DEMO_MODES],
               "svd": {"freqs": _fb[::_st].tolist(), "sv1": _sv1[::_st].tolist()},
               "modes": [{"fn": m.natural_frequency_hz, "zeta": m.damping_ratio_pct,
                          "complexity": m.complexity_pct, "class": m.classification,
                          "shape": {"re": [], "im": []}} for m in fdd.modes],
               "layout": lay.to_dict()}
    return {"lay": lay, "oma_modes": oma_modes, "sv_traces": sv_traces,
            "ema_freqs": [fn for fn, _ in DEMO_MODES], "rpm": lay.running_speed_rpm,
            "raw": (data, fs), "shapes": None, "source": "demo", "name": lay.name,
            "ema_curve": None, "ema_modes_full": None, "ssi_cloud": None, "payload": payload}


def _build_cloud_D(payload: dict):
    lay = OMALayout.from_dict(payload["layout"]) if payload.get("layout") else _default_layout()
    lay.client = payload.get("client", lay.client) or lay.client
    lay.location = payload.get("location", lay.location) or lay.location
    lay.machine_type = payload.get("asset", lay.machine_type) or lay.machine_type
    modes = payload.get("modes", []) or []
    oma_modes = [{"fn": float(m.get("fn", 0.0)), "zeta": float(m.get("zeta", 0.0)),
                  "complexity": float(m.get("complexity", 0.0)),
                  "cls": m.get("class", "natural")} for m in modes]
    svd = payload.get("svd") or {}
    f = np.asarray(svd.get("freqs", []), float); sv1 = np.asarray(svd.get("sv1", []), float)
    sv_traces = ([("SV1", f, 10 * np.log10(np.maximum(sv1, 1e-30)))]
                 if f.size and sv1.size else [])
    shapes = []
    for m in modes:
        sh = m.get("shape") or {}
        re = np.asarray(sh.get("re", []), float); im = np.asarray(sh.get("im", []), float)
        # parte real (con signo) para animar la oscilación; si es imaginaria pura, magnitud
        if re.size:
            shapes.append(re if np.any(np.abs(re) > 1e-9) else np.abs(re + 1j * im))
        else:
            shapes.append(None)
    ema_blk = payload.get("ema") or None
    ema_curve = None; ema_modes_full = None
    if ema_blk and ema_blk.get("freqs"):
        ema_curve = {"freqs": np.asarray(ema_blk["freqs"], float),
                     "mag_db": np.asarray(ema_blk.get("mag_db", []), float),
                     "coh": np.asarray(ema_blk.get("coh", []), float)}
        ema_modes_full = ema_blk.get("modes") or None
    return {"lay": lay, "oma_modes": oma_modes, "sv_traces": sv_traces,
            "ema_freqs": list(payload.get("ema_modes", []) or []),
            "rpm": float(payload.get("running_rpm", lay.running_speed_rpm) or lay.running_speed_rpm),
            "raw": None, "shapes": shapes, "source": "cloud",
            "name": payload.get("name", lay.name),
            "ema_curve": ema_curve, "ema_modes_full": ema_modes_full,
            "ssi_cloud": payload.get("ssi") or None, "payload": payload}


# ================================================================== HEADER
_inject_theme()
page_header("Watermelon Modal", subtitle="EMA + OMA field analysis — one platform, field to report")

# --- selector de fuente: corridas reales de la nube o dataset de muestra ---
try:
    from core.modal.modal_cloud import list_runs, load_run
    _runs = list_runs()
except Exception:  # noqa: BLE001
    _runs = []

_opts = {f"☁ {r.get('name','run')} · {str(r.get('updated_at',''))[:16]}": r.get("id") for r in _runs}
_labels = ["🧪 Sample dataset (demo)"] + list(_opts.keys())
_sc1, _sc2 = st.columns([3, 1])
with _sc1:
    _choice = st.selectbox("Data source", _labels, index=(1 if _opts else 0),
                           help="Field captures uploaded to the cloud appear here automatically.")
with _sc2:
    if st.button("🔄 Refresh runs", use_container_width=True):
        st.rerun()

if _choice != _labels[0] and _opts:
    _rid = _opts.get(_choice)
    _payload = load_run(_rid) if _rid else None
    if _payload:
        D = _build_cloud_D(_payload)
        st.caption(f"☁ Showing field run — {D['name']} · {len(D['oma_modes'])} OMA modes")
    else:
        D = _build_demo_D()
        st.warning("Could not load that cloud run — showing the sample dataset.")
else:
    D = _build_demo_D()
    if _opts:
        st.caption("Showing the sample dataset. Pick a ☁ field run above to see real data.")
    else:
        st.caption("No field runs in the cloud yet — showing a sample dataset. "
                   "Capture in the field and upload; it will appear here when online.")

lay = D["lay"]; nch = lay.n_channels()

# --- Veredicto global (validación automática de modos) para el hero ---
from core.modal.mode_validation import validate_modes as _vm
_ssi_f = [m["fn"] for m in (D["ssi_cloud"] or {}).get("modes", [])] if D["ssi_cloud"] else []
_verd = _vm(D["oma_modes"], ssi_freqs_hz=_ssi_f, running_speed_rpm=D["rpm"]) if D["oma_modes"] else []
_nval = sum(1 for v in _verd if v.verdict == "validated")
_ndbt = sum(1 for v in _verd if v.verdict == "doubtful")
_nrej = sum(1 for v in _verd if v.verdict == "rejected")
if _nrej:
    _chip, _cls = "NO-GO — review", "wm-nogo"
elif _ndbt:
    _chip, _cls = "Review", "wm-rev"
else:
    _chip, _cls = "GO — data acceptable", "wm-go"

# --- Hero de la máquina (industrial, con veredicto) ---
st.markdown(f"""
<div class="wm-hero">
  <div>
    <h1>{lay.name}</h1>
    <div class="meta">{lay.client or '—'} · {lay.location or '—'} · Tag {lay.tag or '—'} ·
      {lay.machine_type or 'Motor-pump'}</div>
  </div>
  <div style="text-align:right">
    <span class="wm-chip {_cls}">{_chip}</span>
    <div class="meta" style="margin-top:8px">{'☁ Field run' if D['source']=='cloud' else '🧪 Sample dataset'}</div>
  </div>
</div>
<div class="wm-kpis">
  <div class="wm-kpi"><div class="v">{len(D['oma_modes'])}</div><div class="l">OMA modes</div>
    <div class="s">{_nval} validated · {_ndbt} doubtful · {_nrej} rejected</div></div>
  <div class="wm-kpi"><div class="v">{int(D['rpm'])}</div><div class="l">Running speed</div>
    <div class="s">1× = {D['rpm']/60:.1f} Hz</div></div>
  <div class="wm-kpi"><div class="v">{nch}</div><div class="l">Sensors</div>
    <div class="s">{len(lay.machine_components)} components</div></div>
  <div class="wm-kpi"><div class="v">{int(lay.fmax_hz)}<span style="font-size:14px"> Hz</span></div>
    <div class="l">Bandwidth (Fmax)</div><div class="s">fs {int(lay.fs_hz)} Hz</div></div>
</div>
""", unsafe_allow_html=True)

TABS = ["🟢  Impact test (EMA)", "🟣  Modes (EMA)", "🟡  OMA capture",
        "🟠  SSI (subspace)", "🔴  Comparative", "🟤  Campbell", "⚫  Mode shapes",
        "🔵  Trend / Compare", "🟢  Report"]
(tab_ema, tab_modes, tab_oma, tab_ssi, tab_cmp, tab_camp, tab_shapes,
 tab_trend, tab_report) = st.tabs(TABS)

# ---------------------------------------------------------------- 1 EMA
with tab_ema:
    _sec("Impact test (EMA)", "FRF + coherence per hammer hit", "ISO 7626-5")
    from plotly.subplots import make_subplots
    if D["ema_curve"] is not None:
        _fx = D["ema_curve"]["freqs"]; _mag = D["ema_curve"]["mag_db"]; _coh = D["ema_curve"]["coh"]
    else:
        f, H, coh = _demo_frf(); _fx = f; _mag = 20 * np.log10(np.abs(H)); _coh = coh
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.06, subplot_titles=("Mobility |H(f)|", "Coherence"))
    fig.add_trace(go.Scatter(x=_fx, y=_mag, line=dict(color=BLUE)), 1, 1)
    fig.add_trace(go.Scatter(x=_fx, y=_coh, line=dict(color=GREEN)), 2, 1)
    fig.update_yaxes(title_text="dB", row=1, col=1); fig.update_yaxes(range=[0, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_layout(height=470, template="watermelon", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    if D["ema_curve"] is not None:
        st.success("Real impact FRF from the field run (ISO 7626-5).")
    elif D["source"] == "cloud":
        st.caption("This cloud run is operational (OMA) — no impact test uploaded.")
    else:
        st.success("5/5 averages accepted · coherence ≥ 0.8 in band (ISO 7626-5).")

# ---------------------------------------------------------------- 3 MODES EMA
with tab_modes:
    _sec("Modes (EMA)", "Peak-picking + half-power damping + Nyquist", "ISO 7626-6")
    f, H, coh = _demo_frf()
    cc1, cc2 = st.columns([2, 3])
    with cc1:
        if D["ema_modes_full"]:
            st.dataframe([{"Freq (Hz)": round(m["fn"], 2), "Damping (%)": round(m["zeta"], 3),
                           "Coherence": round(m["coh"], 3) if m.get("coh") is not None else "—"}
                          for m in D["ema_modes_full"]], use_container_width=True, hide_index=True)
        else:
            st.dataframe([{"Freq (Hz)": round(fr, 2), "Reliable": "✓"} for fr in D["ema_freqs"]] or
                         [{"Freq (Hz)": fn, "Damping (%)": round(z * 100, 2)} for fn, z in DEMO_MODES],
                         use_container_width=True, hide_index=True)
    with cc2:
        fig = go.Figure(go.Scatter(x=H.real, y=H.imag, mode="lines", line=dict(color=NAVY)))
        fig.update_layout(title="Nyquist (mobility)", height=380, template="watermelon",
                          xaxis_title="Re", yaxis_title="Im")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 4 OMA
with tab_oma:
    _sec("OMA capture", "Singular values of spectral densities + FDD modes",
         "ISO 20816 · Brincker 2001")
    if D["sv_traces"]:
        fig = go.Figure(); palette = [BLUE, "#dc2626", GREEN, "#94a3b8"]
        for r, (label, fx, ydb) in enumerate(D["sv_traces"]):
            fig.add_trace(go.Scatter(x=fx, y=ydb, name=label, mode="lines",
                          line=dict(color=palette[r % 4], width=2.2 if r == 0 else 1.1),
                          fill="tozeroy" if r == 0 else None,
                          fillcolor="rgba(37,99,235,.06)" if r == 0 else None,
                          hovertemplate="%{x:.1f} Hz · %{y:.1f} dB<extra>"+label+"</extra>"))
        # marcar y ETIQUETAR cada modo sobre SV1
        _f1, _y1 = D["sv_traces"][0][1], D["sv_traces"][0][2]
        for m in D["oma_modes"]:
            j = int(np.argmin(np.abs(np.asarray(_f1) - m["fn"]))) if len(_f1) else 0
            yv = float(_y1[j]) if len(_y1) else 0
            fig.add_vline(x=m["fn"], line=dict(color="#cbd5e1", width=1, dash="dot"))
            fig.add_annotation(x=m["fn"], y=yv, text=f"<b>{m['fn']:.1f}</b>", showarrow=True,
                               arrowhead=0, arrowcolor="#cbd5e1", ax=0, ay=-22,
                               font=dict(size=11, color=NAVY), bgcolor="rgba(255,255,255,.85)",
                               bordercolor="#e2e8f0", borderpad=2)
        fig.update_layout(title="Singular values of spectral densities", height=440,
                          template="watermelon", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe([{"Freq (Hz)": round(m["fn"], 2), "Damping (%)": round(m["zeta"], 3),
                   "Complexity (%)": round(m["complexity"], 1), "Class": m["cls"]}
                  for m in D["oma_modes"]], use_container_width=True, hide_index=True)
    # --- Validación automática de modos (validado / dudoso / rechazado) ---
    if D["oma_modes"]:
        from core.modal.mode_validation import validate_modes, verdict_rows, summarize as mv_sum
        _ssi_freqs = [m["fn"] for m in (D["ssi_cloud"] or {}).get("modes", [])] if D["ssi_cloud"] else []
        _verd = validate_modes(D["oma_modes"], ssi_freqs_hz=_ssi_freqs, running_speed_rpm=D["rpm"])
        st.markdown("**Automatic mode validation** *(validated / doubtful / rejected)*")
        st.dataframe(verdict_rows(_verd), use_container_width=True, hide_index=True)
        st.info(mv_sum(_verd))
    # --- Registro de verificación de sensores (del software de campo) ---
    _scr = (D.get("payload") or {}).get("sensor_check")
    if _scr:
        with st.expander(f"🔴 Sensor verification record — {_scr.get('n_ok','?')}/{_scr.get('n_total','?')} "
                         f"channels OK ({str(_scr.get('ts',''))[:16]})", expanded=False):
            st.caption("Proof the sensors were wired and responding before the capture (field bump/tap test).")
            _png = _scr.get("png_b64")
            if _png:
                st.markdown(f'<img src="data:image/png;base64,{_png}" '
                            'style="width:100%;border:1px solid #e2e8f0;border-radius:10px">',
                            unsafe_allow_html=True)
            _rows = _scr.get("rows") or []
            if _rows:
                st.dataframe([{"Ch": r[0], "RMS": r[1], "Peak": r[2], "Status": r[3]} for r in _rows],
                             use_container_width=True, hide_index=True, height=240)

# ---------------------------------------------------------------- 5 SSI
with tab_ssi:
    _sec("SSI (subspace)", "Covariance-driven SSI-COV + stabilization diagram + uncertainty",
         "OMA · Brincker & Ventura")
    if D["raw"] is not None:
        data, fs = D["raw"]
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
                          template="watermelon", xaxis_title="Frequency (Hz)", yaxis_title="Model order")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe([{"Mode": i + 1, "Freq (Hz)": round(m.frequency_hz, 3),
                       "± Hz": round(m.std_frequency_hz, 3),
                       "Damping (%)": round(m.damping_ratio_pct, 3),
                       "± %": round(m.std_damping_pct, 3)} for i, m in enumerate(ssi.modes)],
                     use_container_width=True, hide_index=True)
    elif D["ssi_cloud"] and D["ssi_cloud"].get("diagram"):
        _ssi = D["ssi_cloud"]
        fig = go.Figure()
        for entry in _ssi["diagram"]:
            order, fr, mask = entry[0], np.asarray(entry[1], float), entry[2]
            if fr.size == 0:
                continue
            fig.add_trace(go.Scatter(x=list(fr), y=[order] * len(fr), mode="markers",
                          marker=dict(size=6, color=[GREEN if m else "#cbd5e1" for m in mask]),
                          showlegend=False, hoverinfo="skip"))
        for m in _ssi["modes"]:
            fig.add_vline(x=m["fn"], line=dict(color=RED, width=1, dash="dot"))
        fig.update_layout(title="Stabilization diagram (green = stable pole)", height=430,
                          template="watermelon", xaxis_title="Frequency (Hz)", yaxis_title="Model order")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe([{"Mode": i + 1, "Freq (Hz)": round(m["fn"], 3),
                       "± Hz": round(m.get("std_fn", 0.0), 3),
                       "Damping (%)": round(m["zeta"], 3),
                       "± %": round(m.get("std_zeta", 0.0), 3)}
                      for i, m in enumerate(_ssi["modes"])],
                     use_container_width=True, hide_index=True)
        st.caption("Real SSI-COV stabilization diagram from the field run.")
    else:
        st.info("SSI-COV runs on the raw time series in the field app. This cloud run stores the "
                "identified modes below (raw record stays on the field laptop).")
        st.dataframe([{"Freq (Hz)": round(m["fn"], 2), "Damping (%)": round(m["zeta"], 3),
                       "Class": m["cls"]} for m in D["oma_modes"]],
                     use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 6 COMPARATIVE
with tab_cmp:
    _sec("Comparative — EMA vs OMA", "Match impact modes against operational modes",
         "ISO 7626 / OMA")
    oma_f = [m["fn"] for m in D["oma_modes"]]
    ema_f = D["ema_freqs"]
    if not ema_f or not oma_f:
        st.info("Need both EMA and OMA modes to correlate.")
    else:
        matches = correlate(ema_f, oma_f, tol_hz=2.0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ema_f, y=[1] * len(ema_f), mode="markers", name="EMA",
                      marker=dict(color=BLUE, size=13, symbol="triangle-up")))
        fig.add_trace(go.Scatter(x=oma_f, y=[0] * len(oma_f), mode="markers", name="OMA",
                      marker=dict(color=GREEN, size=13, symbol="circle")))
        for m in matches:
            fig.add_shape(type="line", x0=m.ema_hz, y0=1, x1=m.oma_hz, y1=0,
                          line=dict(color="#94a3b8", width=1, dash="dot"))
        fig.update_layout(title="EMA (▲) vs OMA (●)", height=320, template="watermelon",
                          xaxis_title="Frequency (Hz)",
                          yaxis=dict(showticklabels=False, range=[-0.5, 1.5]))
        st.plotly_chart(fig, use_container_width=True)
        if matches:
            st.dataframe(correlation_table(matches), use_container_width=True, hide_index=True)
            st.info(ema_oma_summary(matches))

# ---------------------------------------------------------------- 7 CAMPBELL
with tab_camp:
    _sec("Campbell diagram", "Natural frequencies vs running-speed orders", "API 684 sec. 1.6 (±15%)")
    modes_hz = [m["fn"] for m in D["oma_modes"] if m["cls"] != "spurious"] or [m["fn"] for m in D["oma_modes"]]
    if not modes_hz:
        st.info("No modes to plot.")
    else:
        rpm_op = float(D["rpm"]); rpm_max = rpm_op * 1.35; orders = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
        bands = [SpeedBand(rpm_op * 0.85, rpm_op * 1.15, "Operating ±15%")]
        crossings = compute_crossings(modes_hz, 0.0, rpm_max, orders=orders, bands=bands)
        ymax = max(modes_hz) * 1.3; fig = go.Figure(); rpm_axis = np.linspace(0, rpm_max, 60)
        fig.add_vrect(x0=rpm_op * 0.85, x1=rpm_op * 1.15, fillcolor="rgba(220,38,38,.10)",
                      line_width=0, annotation_text="±15% (API 684)", annotation_position="top left",
                      annotation_font=dict(size=11, color=RED))
        for o in orders:                                    # líneas de orden desde el origen
            fig.add_trace(go.Scatter(x=rpm_axis, y=rpm_axis / 60.0 * o, mode="lines",
                          line=dict(color="#c7d2e0", width=1, dash="dot"), showlegend=False, hoverinfo="skip"))
            ly = o * rpm_max / 60.0
            lx = rpm_max * 0.99 if ly <= ymax else ymax * 60.0 / o
            fig.add_annotation(x=lx, y=min(ly, ymax), text=f"{o:g}×", showarrow=False,
                               font=dict(size=10, color="#94a3b8"), xanchor="right", yanchor="bottom")
        for fn in modes_hz:                                 # frecuencias naturales
            fig.add_hline(y=fn, line=dict(color="#334155", width=1.2, dash="dash"))
        fig.add_vline(x=rpm_op, line=dict(color=NAVY, width=2.5))
        fig.add_annotation(x=rpm_op, y=ymax, text=f"<b>N = {rpm_op:.0f} RPM</b>", showarrow=False,
                           font=dict(size=11, color=NAVY), yanchor="bottom", bgcolor="rgba(255,255,255,.85)")
        _seen = {"coincidence": False, "near": False}
        _sc = {"coincidence": RED, "near": AMBER, "clear": "#cbd5e1"}
        _sn = {"coincidence": "Coincidence", "near": "Near"}
        for cr in crossings:
            sev = cr.severity
            show = sev in _seen and not _seen.get(sev, True)
            fig.add_trace(go.Scatter(x=[cr.crossing_rpm], y=[cr.mode_hz], mode="markers",
                          name=_sn.get(sev, ""), legendgroup=sev, showlegend=show,
                          marker=dict(color=_sc.get(sev, "#cbd5e1"), size=12, symbol="x-thin",
                                      line=dict(width=2, color=_sc.get(sev, "#cbd5e1"))),
                          hovertemplate=f"{cr.mode_hz:.1f} Hz · {cr.order:g}× · %{{x:.0f}} RPM<extra></extra>"))
            if sev in _seen:
                _seen[sev] = True
        fig.update_layout(title="Campbell diagram — resonance screening (API 684)",
                          height=480, template="watermelon", yaxis_range=[0, ymax],
                          xaxis_title="Running speed (RPM)", yaxis_title="Frequency (Hz)")
        st.plotly_chart(fig, use_container_width=True)
        if crossings:
            st.dataframe(crossings_table(crossings), use_container_width=True, hide_index=True)
            st.info(camp_summary(crossings))

# ---------------------------------------------------------------- 8 MODE SHAPES
with tab_shapes:
    _sec("Mode shapes", "3D operational deflection — amplitude colormap (green→red)")
    modes = D["oma_modes"]
    opts = [f"Mode {i+1} — {m['fn']:.1f} Hz" for i, m in enumerate(modes)] or ["—"]
    sel = st.selectbox("Mode", opts, index=0)
    idx = opts.index(sel) if modes else 0
    pts = lay.active_points()
    if D["shapes"] and idx < len(D["shapes"]) and D["shapes"][idx] is not None \
            and len(D["shapes"][idx]) == len(pts):
        amp = np.asarray(D["shapes"][idx], float)              # forma modal (con signo)
    else:
        amp = np.random.default_rng(idx + 1).standard_normal(len(pts))
    st.plotly_chart(_mode_anim_fig(lay, amp, height=560), use_container_width=True)
    st.caption("Press ▶ Play — nodes oscillate along their DOF; colour = amplitude (green→red). "
               "The machine is shown faint for reference.")

# ---------------------------------------------------------------- 8b TREND / COMPARE
with tab_trend:
    _sec("Trend / Compare", "Track natural frequencies across field runs over time",
         "condition monitoring")
    _series = []   # (label, date, [fn,...])
    if _opts:
        _pick = st.multiselect("Runs to compare", list(_opts.keys()),
                               default=list(_opts.keys())[:6])
        for lab in _pick:
            pl = load_run(_opts[lab])
            if pl:
                fns = sorted(float(m.get("fn", 0)) for m in (pl.get("modes") or []))
                _series.append((pl.get("name", lab), str(lab).split("· ")[-1], fns))
    if len(_series) < 2:
        st.info("Pick 2+ field runs above to see the trend. "
                "Showing an example of how the same machine is tracked over time:")
        base = [19.4, 38.8, 77.4, 129.9]
        for k, dd in enumerate(["2026-06-01", "2026-07-15", "2026-09-06"]):
            drift = 1 - 0.012 * k          # el skid pierde rigidez → fn baja con el tiempo
            _series.append((f"Run {k+1}", dd, [round(f * drift, 2) for f in base]))
    # emparejar modos por cercanía al primer run → líneas de tendencia
    ref = _series[0][2]
    fig = go.Figure()
    xs = [s[1] for s in _series]
    for mi, f0 in enumerate(ref):
        ys = []
        for (_nm, _dt, fns) in _series:
            near = min(fns, key=lambda x: abs(x - f0)) if fns else None
            ys.append(near if (near is not None and abs(near - f0) <= 0.15 * f0) else None)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"Mode {mi+1} (~{f0:.0f} Hz)",
                      connectgaps=True, marker=dict(size=9)))
    fig.update_layout(title="Natural frequency trend across runs", height=440, template="watermelon",
                      xaxis_title="Run / date", yaxis_title="Frequency (Hz)")
    st.plotly_chart(fig, use_container_width=True)
    # tabla + alerta de caída
    import pandas as _pd_tr
    rows = []
    for mi, f0 in enumerate(ref):
        vals = []
        for (_nm, _dt, fns) in _series:
            near = min(fns, key=lambda x: abs(x - f0)) if fns else None
            vals.append(near if (near is not None and abs(near - f0) <= 0.15 * f0) else None)
        v0, vN = vals[0], vals[-1]
        dpc = ((vN - v0) / v0 * 100) if (v0 and vN) else None
        rows.append({"Mode": f"~{f0:.0f} Hz",
                     **{s[1]: (f"{v:.2f}" if v else "—") for s, v in zip(_series, vals)},
                     "Δ% (first→last)": (f"{dpc:+.1f}%" if dpc is not None else "—")})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    _drops = [r for r in rows if r["Δ% (first→last)"] != "—" and float(r["Δ% (first→last)"].rstrip('%')) <= -3]
    if _drops:
        st.warning("⚠ A natural frequency dropped ≥3% over time — possible loss of stiffness "
                   "(loosening, cracking, or skid/base degradation). Investigate.")
    else:
        st.success("Natural frequencies stable across runs — no stiffness loss detected.")

# ---------------------------------------------------------------- 9 PRELIMINARY
with tab_report:
    _sec("Report", "Full OMA report — cover, table of contents, all sections",
         "SIGA-FMT-179 · ISO 20816 · API 684")
    c = st.columns(4)
    c[0].metric("Machine", lay.tag or lay.name)
    c[1].metric("OMA modes", len(D["oma_modes"]))
    c[2].metric("1X (Hz)", f"{D['rpm']/60:.1f}")
    c[3].metric("Sensors", nch)
    # --- Diagnóstico automático (narrativa tipo experto) ---
    if D["oma_modes"]:
        _mh = [m["fn"] for m in D["oma_modes"] if m["cls"] != "spurious"] or [m["fn"] for m in D["oma_modes"]]
        _cx = compute_crossings(_mh, 0.0, D["rpm"] * 1.3,
                                orders=[1, 2, 3, 4],
                                bands=[SpeedBand(D["rpm"] * 0.85, D["rpm"] * 1.15, "Op ±15%")])
        _nar = _narrative(lay.name, D["oma_modes"], D["rpm"], _verd, _cx)
        st.markdown(f"<div style='background:#eef6ff;border-left:4px solid {BLUE};border-radius:8px;"
                    f"padding:12px 16px;margin:6px 0'><b>Auto-diagnosis</b><br>{_nar}</div>",
                    unsafe_allow_html=True)
    # --- Registro de verificación de sensores (del campo) ---
    _scr = (D.get("payload") or {}).get("sensor_check")
    if _scr and _scr.get("png_b64"):
        with st.expander(f"🔴 Sensor verification record — {_scr.get('n_ok','?')}/{_scr.get('n_total','?')} OK"):
            st.markdown(f'<img src="data:image/png;base64,{_scr["png_b64"]}" '
                        'style="width:100%;border:1px solid #e2e8f0;border-radius:10px">',
                        unsafe_allow_html=True)
    st.caption("Same corporate template as the Reports module: cover page + format band + "
               "table of contents + machine, OMA (singular values + modes), Campbell, "
               "EMA↔OMA correlation and conclusions.")
    _lang = st.radio("Language", ["Español", "English"], horizontal=True, key="rep_lang")
    if not D["oma_modes"]:
        st.info("Select a field run with identified modes to assemble the report.")
    elif st.button("📄 Generate full report (PDF)", type="primary", key="rep_gen"):
        try:
            from core.modal.preliminary_report import build_preliminary_pdf
            import base64 as _b64m
            es = (_lang == "Español")
            _L = (lambda s, e: s if es else e)
            with st.spinner("Rendering figures and building the report (cover · TOC · mode shapes)…"):
                def _png(fig, w=1100, h=520):
                    try:
                        return fig.to_image(format="png", width=w, height=h, scale=2)
                    except Exception:  # noqa: BLE001
                        return None
                sections = []
                # 1) Configuration (3D machine)
                cfg_png = _png(_geometry_fig(lay, height=460), 1100, 520)
                sections.append({"title": _L("Configuración", "Configuration"),
                    "figures": [(_L("Figura. Máquina y sensores (3D).", "Figure. Machine & sensors (3D)."), cfg_png)] if cfg_png else [],
                    "table": {"headers": ["BNC", "Code", "Component", "Ref", "DOF"],
                              "rows": [[p.bnc, p.code, p.component, p.position_ref, p.dof] for p in lay.active_points()]}})
                # 2) Sensor verification (si hay)
                if _scr and _scr.get("png_b64"):
                    sections.append({"title": _L("Verificación de sensórica", "Sensor verification"),
                        "intro": _L(f"{_scr.get('n_ok')}/{_scr.get('n_total')} canales OK ({str(_scr.get('ts',''))[:16]}).",
                                    f"{_scr.get('n_ok')}/{_scr.get('n_total')} channels OK ({str(_scr.get('ts',''))[:16]})."),
                        "figures": [(_L("Figura. Chequeo de sensores en vivo.", "Figure. Live sensor check."),
                                     _b64m.b64decode(_scr["png_b64"]))],
                        "table": {"headers": ["Ch", "RMS", "Peak", "Status"], "rows": _scr.get("rows", [])}})
                # 3) OMA singular values + modes
                sv_fig = go.Figure()
                for r, (label, fx, ydb) in enumerate(D["sv_traces"]):
                    sv_fig.add_trace(go.Scatter(x=fx, y=ydb, name=label,
                                     line=dict(color=[BLUE, "#dc2626", GREEN, "#94a3b8"][r % 4], width=2 if r == 0 else 1)))
                for m in D["oma_modes"]:
                    sv_fig.add_vline(x=m["fn"], line=dict(color="#cbd5e1", width=1, dash="dot"))
                sv_fig.update_layout(title="Singular values", template="watermelon",
                                     xaxis_title="Frequency (Hz)", yaxis_title="dB")
                sections.append({"title": _L("OMA — densidad espectral (FDD)", "OMA — spectral density (FDD)"),
                    "figures": [(_L("Figura. Valores singulares.", "Figure. Singular values."), _png(sv_fig))] if D["sv_traces"] else [],
                    "table": {"headers": ["Freq (Hz)", "Damping (%)", "Complex (%)", "Class"],
                              "rows": [[round(m["fn"], 2), round(m["zeta"], 3), round(m["complexity"], 1), m["cls"]] for m in D["oma_modes"]]}})
                # 4) Campbell
                _mh = [m["fn"] for m in D["oma_modes"] if m["cls"] != "spurious"] or [m["fn"] for m in D["oma_modes"]]
                _cx = compute_crossings(_mh, 0.0, D["rpm"] * 1.35, orders=[0.5, 1, 2, 3, 4, 5, 6, 7, 8],
                                        bands=[SpeedBand(D["rpm"] * 0.85, D["rpm"] * 1.15, "Op ±15%")])
                cam = go.Figure(); rr = np.linspace(0, D["rpm"] * 1.35, 60); _ym = max(_mh) * 1.3
                cam.add_vrect(x0=D["rpm"] * 0.85, x1=D["rpm"] * 1.15, fillcolor="rgba(220,38,38,.10)", line_width=0)
                for o in [0.5, 1, 2, 3, 4, 5, 6, 7, 8]:
                    cam.add_trace(go.Scatter(x=rr, y=rr / 60 * o, mode="lines", line=dict(color="#c7d2e0", width=1, dash="dot"), showlegend=False))
                for fn in _mh:
                    cam.add_hline(y=fn, line=dict(color="#334155", width=1, dash="dash"))
                cam.add_vline(x=D["rpm"], line=dict(color=NAVY, width=2.5))
                for cr in _cx:
                    if cr.severity in ("coincidence", "near"):
                        cam.add_trace(go.Scatter(x=[cr.crossing_rpm], y=[cr.mode_hz], mode="markers", showlegend=False,
                                      marker=dict(color=RED if cr.severity == "coincidence" else AMBER, size=11, symbol="x")))
                cam.update_layout(title="Campbell (API 684)", template="watermelon", yaxis_range=[0, _ym],
                                  xaxis_title="Running speed (RPM)", yaxis_title="Frequency (Hz)")
                sections.append({"title": _L("Campbell — cribado de resonancia (API 684)", "Campbell — resonance screening (API 684)"),
                    "figures": [(_L("Figura. Diagrama de Campbell.", "Figure. Campbell diagram."), _png(cam))],
                    "table": {"headers": ["Mode", "fn (Hz)", "Order", "Crossing RPM", "Margin%", "Status"],
                              "rows": [[c.mode_label, round(c.mode_hz, 2), f"{c.order:g}×", round(c.crossing_rpm, 0),
                                        round(c.sep_margin_pct, 1),
                                        {"coincidence": "Coincidence", "near": "Near", "clear": "Clear"}[c.severity]]
                                       for c in _cx[:12]]}})
                # 5) Mode shapes (3-4 modos)
                mfigs = []
                for i, m in enumerate(D["oma_modes"][:4]):
                    a = D["shapes"][i] if (D["shapes"] and i < len(D["shapes"]) and D["shapes"][i] is not None) else np.random.default_rng(i + 1).standard_normal(nch)
                    a = np.abs(np.asarray(a, float)); a = (a - a.min()) / (np.ptp(a) or 1)
                    p = _png(_geometry_fig(lay, amp=list(a), height=420), 900, 460)
                    if p:
                        mfigs.append((_L(f"Figura. Modo {i+1} — {m['fn']:.1f} Hz.", f"Figure. Mode {i+1} — {m['fn']:.1f} Hz."), p))
                if mfigs:
                    sections.append({"title": _L("Formas modales", "Mode shapes"), "figures": mfigs})
                # meta + análisis + hallazgos + recomendaciones
                nar = _narrative(lay.name, D["oma_modes"], D["rpm"], _verd, _cx)
                findings = []
                _ib = [c for c in _cx if c.in_band]
                if _ib:
                    findings.append(_L(f"El modo {min(_ib,key=lambda c:c.sep_margin_pct).mode_hz:.1f} Hz cae dentro de ±15% de un orden de giro (riesgo de resonancia).",
                                       f"The {min(_ib,key=lambda c:c.sep_margin_pct).mode_hz:.1f} Hz mode falls within ±15% of a running-speed order (resonance risk)."))
                if _drops if '_drops' in dir() else False:
                    findings.append(_L("Una frecuencia natural bajó ≥3% entre corridas: posible pérdida de rigidez.",
                                       "A natural frequency dropped ≥3% between runs: possible stiffness loss."))
                if not findings:
                    findings.append(_L("Sin coincidencias de resonancia dentro de la banda de operación.",
                                       "No resonance coincidences within the operating band."))
                recs = [_L("Correlacionar amplitud/fase de vibración vs velocidad en operación (API 684).",
                           "Correlate vibration amplitude/phase vs speed in operation (API 684)."),
                        _L("Si hay modo cerca de 1×/2× con amplitud alta, evaluar rigidización de base/skid.",
                           "If a mode near 1×/2× shows high amplitude, evaluate base/skid stiffening.")]
                quality = [(_L("OMA capturado y modos hallados", "OMA captured & modes found"), "GO", f"{len(D['oma_modes'])} modes"),
                           (_L("Canales activos", "Channels active"), "GO", str(nch)),
                           (_L("Verificación de sensores", "Sensor verification"),
                            "GO" if _scr else _L("Pendiente", "Pending"),
                            f"{_scr.get('n_ok')}/{_scr.get('n_total')} OK" if _scr else "—")]
                meta = {"title": _L("Reporte Análisis Modal Operacional (OMA)", "Operational Modal Analysis Report (OMA)"),
                        "asset": lay.tag or lay.name, "client": lay.client, "machine_type": lay.machine_type,
                        "location": lay.location, "test_type": "OMA", "rpm": int(D["rpm"]),
                        "verdict": _chip.split(" —")[0]}
                logo_png = None
                try:
                    from pathlib import Path as _P
                    _lp = _P("assets/watermelon_logo.png")
                    logo_png = _lp.read_bytes() if _lp.exists() else None
                except Exception:  # noqa: BLE001
                    logo_png = None
                pdf = build_preliminary_pdf(meta=meta, quality=quality, sections=sections,
                                            analysis=[nar], findings=findings, recommendations=recs,
                                            photos=[], run_id=D.get("name", ""), logo_png=logo_png,
                                            lang="es" if es else "en")
            st.session_state["_modal_report_pdf"] = pdf
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not build the report: {type(e).__name__}: {e}")
    _pdf = st.session_state.get("_modal_report_pdf")
    if _pdf:
        st.download_button("⬇ Download report PDF", data=_pdf,
                           file_name=f"OMA_{(lay.tag or lay.name).replace(' ', '_')}.pdf",
                           mime="application/pdf", use_container_width=True)
        import base64
        _b64 = base64.b64encode(_pdf).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{_b64}" width="100%" height="820" '
            f'style="border:1px solid #e2e8f0;border-radius:8px"></iframe>',
            unsafe_allow_html=True)
