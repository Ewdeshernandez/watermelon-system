"""
pages/18_Modal_Analysis.py — Watermelon Modal (WEB)
===================================================

Web = SOLO análisis (la CONFIGURACIÓN se hace en el software de campo). Consume
las CORRIDAS reales que el campo sube a la nube (tabla modal_runs). Si no hay red
/ no hay corridas, usa un dataset de muestra para no quedar vacía. El equipo se
muestra como contexto de solo-lectura en el encabezado.

Navegación PERSISTENTE (segmented control, no st.tabs — así no se "salta" de
sección al generar el reporte): Spectral density (FDD) · Mode shapes · SSI ·
Campbell · Impact (EMA) · Modes (EMA) · Comparative · Trend · Report.

Densidad espectral (FDD): todas las curvas de valores singulares (SV1 en color,
SV2–SVn en gris) + peak-picking (clic en la curva para agregar/quitar modos).
Report: PDF SIGA completo (portada + TOC + configuración 3D + verificación de
sensores + densidad espectral + formas modales 3D + Campbell + hallazgos/
recomendaciones editables + consecutivo + realizado/aprobado) con archivado.

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
      .block-container { padding-top: 0.8rem; max-width: 1200px; }
      /* espaciado compacto entre widgets superiores (data source, captions) */
      div[data-testid="stSelectbox"] { margin-bottom: 0; }
      /* Hero (compacto) */
      .wm-hero { background: linear-gradient(110deg,#0F1E3D 0%,#12325a 55%,#16a34a 160%);
        border-radius:12px; padding:11px 18px; color:#fff; box-shadow:0 6px 18px rgba(15,30,61,.16);
        display:flex; justify-content:space-between; align-items:center; gap:14px; flex-wrap:wrap; }
      .wm-hero h1 { font-size:18px; font-weight:700; margin:0 0 2px; letter-spacing:-.01em; }
      .wm-hero .meta { color:#cbd5e1; font-size:12px; }
      .wm-chip { font-size:11px; font-weight:700; letter-spacing:.03em; text-transform:uppercase;
        padding:5px 12px; border-radius:999px; }
      .wm-go { background:#16a34a; color:#fff; } .wm-rev { background:#f59e0b; color:#0f1e3d; }
      .wm-nogo { background:#dc2626; color:#fff; }
      /* KPI cards (compactas) */
      .wm-kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:8px 0 2px; }
      .wm-kpi { background:#fff; border:1px solid #e6ecf5; border-radius:11px; padding:8px 13px;
        box-shadow:0 1px 2px rgba(15,30,61,.04),0 4px 12px rgba(15,30,61,.04); }
      .wm-kpi .v { font-family:'IBM Plex Mono',monospace; font-size:20px; font-weight:600; color:#0F1E3D; line-height:1.1; }
      .wm-kpi .l { font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.05em; margin-top:2px; }
      .wm-kpi .s { font-size:11px; color:#94a3b8; }
      /* Tabs */
      .stTabs [data-baseweb="tab-list"] { gap:4px; }
      .stTabs [data-baseweb="tab"] { background:#eef2f8; border-radius:9px 9px 0 0; padding:8px 14px; font-weight:600; }
      .stTabs [aria-selected="true"] { background:#0F1E3D !important; color:#fff !important; }
      /* Tablas bonitas (mismo estilo en TODO el módulo) */
      table.wm-modes { width:100%; border-collapse:separate; border-spacing:0;
        font-family:'IBM Plex Sans',sans-serif; border:1px solid #e6ecf5; border-radius:14px;
        overflow:hidden; box-shadow:0 6px 18px rgba(15,30,61,.06); margin:2px 0 10px; }
      table.wm-modes th { background:#0F1E3D; color:#fff; font-size:11px; font-weight:600;
        letter-spacing:.04em; text-transform:uppercase; text-align:center; padding:11px 10px; }
      table.wm-modes td { padding:10px 10px; text-align:center; border-top:1px solid #eef2f8;
        font-size:14px; color:#0f1e3d; }
      table.wm-modes tr:nth-child(even) td { background:#f7fafd; }
      table.wm-modes tr:hover td { background:#eef6ff; }
      table.wm-modes td.idx { color:#94a3b8; font-family:'IBM Plex Mono',monospace; width:38px; }
      table.wm-modes td.fn { font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:15px; }
      table.wm-modes td.num { font-family:'IBM Plex Mono',monospace; }
      table.wm-modes .u { color:#94a3b8; font-size:11px; font-weight:400; }
      table.wm-modes .cls { font-size:12px; color:#475569; }
      table.wm-modes .badge, table.wm-modes .pill { padding:3px 10px; border-radius:999px;
        font-size:11px; font-weight:700; white-space:nowrap; }
      @media (prefers-color-scheme: dark){
        .wm-kpi{ background:#141b26; border-color:#243040; }
        .wm-kpi .v{ color:#eaf0f7; }
      }
    </style>
    """, unsafe_allow_html=True)


def _pill(text, color, bg):
    return f"<span class='pill' style='color:{color};background:{bg}'>{text}</span>"


def _status_pill(s):
    s0 = str(s).strip().lower()
    if s0.startswith("ok") or "pass" in s0:
        return _pill("OK", "#16a34a", "#eaf7ef")
    if "respond" in s0 or "warn" in s0:
        return _pill(str(s).replace("●", "").strip() or "responding", "#b45309", "#fef3e2")
    if "fail" in s0 or "no" in s0:
        return _pill(str(s), "#dc2626", "#fdeaea")
    return _pill(str(s), "#64748b", "#eef2f8")


def _pretty_table(headers, rows):
    """Tabla HTML con el estilo del módulo (marcos, chips, monoespaciado). `rows`
    es una lista de listas; cada celda puede llevar HTML (chips, <span class=...>)."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="wm-modes"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _show_table(headers, rows):
    import streamlit as _st
    _st.markdown(_pretty_table(headers, rows), unsafe_allow_html=True)


def _show_dicts(rows):
    """Renderiza una lista de dicts (p.ej. de correlation_table/crossings_table) con
    el estilo bonito. Colorea la columna de estado/severidad si existe."""
    import streamlit as _st
    if not rows:
        return
    heads = list(rows[0].keys())
    _stat_keys = {"Estado", "Status", "Severidad", "Severity"}
    body = []
    for r in rows:
        cells = []
        for h in heads:
            v = r.get(h, "")
            if h in _stat_keys:
                vs = str(v).lower()
                if any(k in vs for k in ("coincid", "fail", "no-go", "rechaz")):
                    cells.append(_pill(str(v), "#dc2626", "#fdeaea"))
                elif any(k in vs for k in ("cercan", "near", "dudos", "doubt", "warn")):
                    cells.append(_pill(str(v), "#b45309", "#fef3e2"))
                elif any(k in vs for k in ("libre", "clear", "ok", "valid")):
                    cells.append(_pill(str(v), "#16a34a", "#eaf7ef"))
                else:
                    cells.append(_pill(str(v), "#64748b", "#eef2f8"))
            else:
                cells.append(f"<span class='num'>{v}</span>" if isinstance(v, (int, float)) else str(v))
        body.append(cells)
    _st.markdown(_pretty_table(heads, body), unsafe_allow_html=True)


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
    _svlist = svd.get("sv")            # lista de curvas SV1..SVn (nuevo: multi-sensor)
    if f.size and _svlist:
        sv_traces = [(f"SV{r+1}", f, 10 * np.log10(np.maximum(np.asarray(c, float), 1e-30)))
                     for r, c in enumerate(_svlist) if len(c) == f.size]
    elif f.size and sv1.size:
        sv_traces = [("SV1", f, 10 * np.log10(np.maximum(sv1, 1e-30)))]
    else:
        sv_traces = []
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
            "raw": None, "raw_ref": payload.get("raw_ref"), "shapes": shapes, "source": "cloud",
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

# --- Data cruda en la nube: recalcular TODO en la web (SVD completo + SSI en vivo) ---
@st.cache_data(show_spinner=False)
def _load_raw_cached(path, bucket, fs):
    from core.modal.modal_cloud import download_raw
    return download_raw({"path": path, "bucket": bucket, "fs": fs})


_rref = D.get("raw_ref")
if _rref and D.get("source") == "cloud" and _rref.get("path"):
    _mb = (_rref.get("size_bytes", 0) or 0) / 1e6
    _use_raw = st.checkbox(
        f"⚡ Recompute from raw data ({_rref.get('n_ch','?')} ch · full SVD + live SSI · ~{_mb:.1f} MB)",
        value=False, key="use_raw",
        help="Downloads the full raw waveform this run uploaded and recomputes everything on the web.")
    if _use_raw:
        with st.spinner("Downloading raw waveform and recomputing FDD…"):
            _rr = _load_raw_cached(_rref.get("path"), _rref.get("bucket", "modal-raw"), _rref.get("fs"))
        if _rr is not None:
            _rdata, _rfs = _rr
            D["raw"] = (_rdata, _rfs)
            try:
                _fmax = min(_rfs / 2.56, lay.fmax_hz)
                _fdd = run_oma(time_data=_rdata, sample_rate_hz=_rfs, nperseg=4096,
                               channel_names=lay.channel_names(), f_min_hz=5.0, f_max_hz=_fmax)
                _fr = np.asarray(_fdd.frequencies_hz); _sv = np.asarray(_fdd.singular_values)
                if _sv.ndim == 1:
                    _sv = _sv[None, :]
                _bd = _fr <= _fmax
                D["sv_traces"] = [(f"SV{r+1}", _fr[_bd], 10 * np.log10(np.maximum(_sv[r][_bd], 1e-30)))
                                  for r in range(min(_sv.shape[0], 4))]
                st.caption(f"⚡ Recomputed from raw — {min(_sv.shape[0],4)} singular-value curves · live SSI enabled.")
            except Exception as _e:  # noqa: BLE001
                st.warning(f"Raw recompute failed: {type(_e).__name__}")
        else:
            st.warning("Could not download the raw data for this run.")

# --- Modos manuales (peak-picking en la web) — se fusionan con los automáticos.
#     Los automáticos (FDD) quedan PROTEGIDOS: sólo se pueden quitar los manuales. ---
_run_key = str(D.get("name", "run"))
_MANUAL_KEY = f"modal_manual::{_run_key}"
_manual_modes = st.session_state.setdefault(_MANUAL_KEY, [])
for _am in D["oma_modes"]:
    _am.setdefault("source", "auto")
for _mm in _manual_modes:
    _mm.setdefault("source", "manual")
D["auto_n"] = len(D["oma_modes"])
D["oma_modes"] = sorted(D["oma_modes"] + list(_manual_modes), key=lambda m: m["fn"])


def _pick_mode_from_curve(fx, ydb, f_click, win_frac=0.06):
    """Ajusta un modo a partir de un clic: busca el pico local cercano y estima
    el amortiguamiento por half-power (-3 dB) sobre la curva SV1 (en dB)."""
    fx = np.asarray(fx, float); ydb = np.asarray(ydb, float)
    if fx.size == 0:
        return None
    win = max(1.0, win_frac * max(f_click, 1.0))
    m = np.abs(fx - f_click) <= win
    if not np.any(m):
        m = np.abs(fx - f_click) <= (2 * win)
    if not np.any(m):
        return None
    idx = int(np.where(m)[0][np.argmax(ydb[m])])
    fn = float(fx[idx]); peak = float(ydb[idx]); half = peak - 3.0103
    lo = idx
    while lo > 0 and ydb[lo] > half:
        lo -= 1
    hi = idx
    while hi < len(ydb) - 1 and ydb[hi] > half:
        hi += 1
    bw = float(fx[hi] - fx[lo])
    zeta = max(0.05, min(20.0, (bw / (2.0 * fn) * 100.0))) if fn > 0 else 0.0
    return {"fn": round(fn, 3), "zeta": round(zeta, 3), "complexity": 0.0,
            "cls": "manual", "source": "manual"}


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

T_OMA = "🟡  Spectral density (FDD)"
T_SHAPES = "⚫  Mode shapes"
T_SSI = "🟠  SSI (subspace)"
T_CAMP = "🟤  Campbell"
T_EMA = "🟢  Impact test (EMA)"
T_MODES = "🟣  Modes (EMA)"
T_CMP = "🔴  Comparative"
T_TREND = "🔵  Trend / Compare"
T_REPORT = "📄  Report"
_NAVOPTS = [T_OMA, T_SSI, T_CAMP, T_SHAPES, T_CMP, T_EMA, T_MODES, T_TREND, T_REPORT]

# Navegación PERSISTENTE (segmented control con estado) — a diferencia de st.tabs,
# conserva la sección activa tras cada rerun (arregla el "salto" al generar reporte).
if "modal_nav" not in st.session_state:
    st.session_state["modal_nav"] = T_OMA
if hasattr(st, "segmented_control"):
    nav = st.segmented_control("Section", _NAVOPTS, key="modal_nav",
                               label_visibility="collapsed") or st.session_state["modal_nav"]
else:  # respaldo para versiones viejas de Streamlit
    nav = st.radio("Section", _NAVOPTS, key="modal_nav", horizontal=True,
                   label_visibility="collapsed")
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- 1 EMA
if nav == T_EMA:
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
if nav == T_MODES:
    _sec("Modes (EMA)", "Peak-picking + half-power damping + Nyquist", "ISO 7626-6")
    f, H, coh = _demo_frf()
    cc1, cc2 = st.columns([2, 3])
    with cc1:
        if D["ema_modes_full"]:
            _show_table(["#", "Frequency", "Damping ζ", "Coherence"],
                        [[f"<span class='idx'>{i}</span>", f"<span class='fn'>{m['fn']:.2f}<span class='u'> Hz</span></span>",
                          f"<span class='num'>{m['zeta']:.2f}<span class='u'> %</span></span>",
                          f"<span class='num'>{round(m['coh'],3) if m.get('coh') is not None else '—'}</span>"]
                         for i, m in enumerate(D["ema_modes_full"], 1)])
        else:
            _er = ([[f"<span class='fn'>{fr:.2f}<span class='u'> Hz</span></span>", _pill("reliable", "#16a34a", "#eaf7ef")]
                    for fr in D["ema_freqs"]] or
                   [[f"<span class='fn'>{fn:.2f}<span class='u'> Hz</span></span>",
                     f"<span class='num'>{round(z*100,2)}<span class='u'> %</span></span>"] for fn, z in DEMO_MODES])
            _show_table(["Frequency", "Damping ζ" if not D["ema_freqs"] else "Status"], _er)
    with cc2:
        fig = go.Figure(go.Scatter(x=H.real, y=H.imag, mode="lines", line=dict(color=NAVY)))
        fig.update_layout(title="Nyquist (mobility)", height=380, template="watermelon",
                          xaxis_title="Re", yaxis_title="Im")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 4 OMA (análisis)
if nav == T_OMA:
    _sec("Spectral density (FDD)", "", "ISO 20816")
    from core.modal.mode_validation import validate_modes, summarize as mv_sum
    _ssi_freqs = [m["fn"] for m in (D["ssi_cloud"] or {}).get("modes", [])] if D["ssi_cloud"] else []
    _verd = validate_modes(D["oma_modes"], ssi_freqs_hz=_ssi_freqs, running_speed_rpm=D["rpm"]) \
        if D["oma_modes"] else []
    _vmap = {round(v.frequency_hz, 2): v for v in _verd} if _verd else {}

    if D["sv_traces"]:
        _f1, _y1 = np.asarray(D["sv_traces"][0][1], float), np.asarray(D["sv_traces"][0][2], float)
        fig = go.Figure()
        # SV2..SVn de fondo (gris), reveal de modos cercanos
        for r, (label, fx, ydb) in enumerate(D["sv_traces"][1:], start=2):
            fig.add_trace(go.Scatter(x=fx, y=ydb, name=f"SV{r}", mode="lines",
                          line=dict(color="#cbd5e1", width=1.0), opacity=0.9,
                          hovertemplate="%{x:.1f} Hz · %{y:.1f} dB<extra>SV"+str(r)+"</extra>"))
        # SV1 dominante (color + relleno)
        fig.add_trace(go.Scatter(x=_f1, y=_y1, name="SV1", mode="lines",
                      line=dict(color=BLUE, width=2.4), fill="tozeroy",
                      fillcolor="rgba(37,99,235,.07)",
                      hovertemplate="%{x:.1f} Hz · %{y:.1f} dB<extra>SV1</extra>"))
        # capa de PUNTOS invisible sobre SV1 → hace clicable toda la curva (para agregar picos)
        fig.add_trace(go.Scatter(x=_f1, y=_y1, mode="markers", name="pick",
                      marker=dict(size=9, color="rgba(0,0,0,0)"), showlegend=False,
                      hovertemplate="%{x:.1f} Hz — click to add<extra></extra>"))
        # Órdenes de la velocidad de giro (1×..5×): si un modo cae sobre una línea,
        # la rotación lo excita → riesgo de resonancia (API 684).
        _x1 = D["rpm"] / 60.0
        _fmaxplot = float(_f1.max()) if _f1.size else 0
        for _o in (1, 2, 3, 4, 5):
            if _o * _x1 <= _fmaxplot:
                fig.add_vline(x=_o * _x1, line=dict(color=AMBER, width=1, dash="dot"))
                fig.add_annotation(x=_o * _x1, y=1.0, yref="paper", yanchor="bottom",
                                   text=f"{_o}×", showarrow=False, font=dict(size=10, color=AMBER))
        # marcadores: automáticos (verde, PROTEGIDOS) + manuales (púrpura, removibles)
        def _ynear(fn):
            j = int(np.argmin(np.abs(_f1 - fn))) if _f1.size else 0
            return float(_y1[j]) if _y1.size else 0.0
        for m in D["oma_modes"]:
            yv = _ynear(m["fn"])
            _man = m.get("source") == "manual"
            fig.add_trace(go.Scatter(x=[m["fn"]], y=[yv], mode="markers",
                          marker=dict(size=13, color=("#7c3aed" if _man else GREEN),
                                      symbol=("diamond" if _man else "circle"),
                                      line=dict(width=1.6, color="white")), showlegend=False,
                          hovertemplate=f"<b>{m['fn']:.2f} Hz</b> · ζ {m['zeta']:.2f}%%"
                                        + ("  —  click to remove" if _man else " · FDD (protected)")
                                        + f"<extra>{'manual' if _man else 'FDD'}</extra>"))
            fig.add_annotation(x=m["fn"], y=yv, text=f"<b>{m['fn']:.1f}</b>", showarrow=True,
                               arrowhead=0, arrowcolor="#cbd5e1", ax=0, ay=-24,
                               font=dict(size=10, color=NAVY), bgcolor="rgba(255,255,255,.9)",
                               bordercolor="#e2e8f0", borderpad=2)
        _xmax = float(_f1.max()) if _f1.size else 1.0
        fig.update_layout(title_text="", height=540, template="watermelon", dragmode="zoom",
                          clickmode="event+select", margin=dict(l=62, r=16, t=16, b=52),
                          xaxis=dict(range=[0, _xmax], constrain="domain"),
                          xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)")
        _ev = st.plotly_chart(fig, use_container_width=True, key="oma_sv_pick",
                              on_select="rerun", selection_mode=["points", "box"],
                              config={"displayModeBar": False})
        # --- clic: cerca de un MANUAL lo borra; en un pico libre agrega manual;
        #     los AUTOMÁTICOS (FDD) están protegidos (clic sobre ellos no hace nada) ---
        try:
            _pts = (_ev.get("selection", {}) or {}).get("points", []) if _ev else []
        except Exception:  # noqa: BLE001
            _pts = []
        if _pts:
            _sig = tuple(sorted(round(float(p.get("x", 0)), 2) for p in _pts))
            if st.session_state.get("_oma_last_sig") != _sig:
                st.session_state["_oma_last_sig"] = _sig
                _dirty = False
                _autos = [e for e in D["oma_modes"] if e.get("source") != "manual"]
                for p in _pts:
                    x = float(p.get("x", 0))
                    tol = max(0.8, 0.012 * max(x, 1.0))
                    # 1) manual cercano → borrar
                    _mh = min(_manual_modes, key=lambda e: abs(e["fn"] - x)) if _manual_modes else None
                    if _mh and abs(_mh["fn"] - x) <= tol:
                        _manual_modes.remove(_mh); _dirty = True; continue
                    # 2) automático cercano → protegido, no hacer nada
                    if _autos and min(abs(e["fn"] - x) for e in _autos) <= tol:
                        continue
                    # 3) pico libre → agregar manual
                    cand = _pick_mode_from_curve(_f1, _y1, x)
                    if cand and all(abs(cand["fn"] - e["fn"]) > 0.5 for e in D["oma_modes"]):
                        _manual_modes.append(cand); _dirty = True
                if _dirty:
                    st.session_state[_MANUAL_KEY] = _manual_modes
                    st.rerun()
        st.caption("Click a peak to add a mode · click a purple marker to remove it.")

    # Un solo control: limpiar los modos manuales (los automáticos no se tocan)
    if _manual_modes:
        if st.button(f"↺ Clear {len(_manual_modes)} manual mode(s)", key="oma_reset"):
            st.session_state[_MANUAL_KEY] = []
            st.session_state.pop("_oma_last_sig", None); st.rerun()

    # --- Tabla consolidada BONITA (marcos, colores, chips) ---
    def _v_of(fn):
        return _vmap.get(round(fn, 2))
    _VCOL = {"validated": ("#16a34a", "#eaf7ef"), "doubtful": ("#b45309", "#fef3e2"),
             "rejected": ("#dc2626", "#fdeaea")}
    _VTXT = {"validated": "Validated", "doubtful": "Doubtful", "rejected": "Rejected"}
    _rows_html = []
    for i, m in enumerate(D["oma_modes"], 1):
        v = _v_of(m["fn"]); vk = getattr(v, "verdict", "") if v else ""
        vc, vb = _VCOL.get(vk, ("#64748b", "#eef2f8"))
        harm = "  ⚠" if (v and getattr(v, "is_harmonic", False)) else ""
        _man = m.get("source") == "manual"
        src_c, src_b, src_t = (("#7c3aed", "#f2ecfd", "Manual") if _man else ("#2563eb", "#e8f0ff", "FDD"))
        vlabel = (_VTXT.get(vk, "—") + harm) if vk else "—"
        _rows_html.append(
            f"<tr>"
            f"<td class='idx'>{i}</td>"
            f"<td class='fn'>{m['fn']:.2f}<span class='u'> Hz</span></td>"
            f"<td class='num'>{m['zeta']:.2f}<span class='u'> %</span></td>"
            f"<td class='num'>{m['complexity']:.0f}<span class='u'> %</span></td>"
            f"<td><span class='cls'>{m['cls']}</span></td>"
            f"<td><span class='badge' style='color:{src_c};background:{src_b}'>{src_t}</span></td>"
            f"<td><span class='pill' style='color:{vc};background:{vb}'>{vlabel}</span></td>"
            f"</tr>")
    _table_html = ('<table class="wm-modes"><thead><tr>'
                   "<th>#</th><th>Frequency</th><th>Damping ζ</th><th>Complexity</th>"
                   "<th>Class</th><th>Source</th><th>Validation</th></tr></thead>"
                   f"<tbody>{''.join(_rows_html)}</tbody></table>")
    if D["oma_modes"]:
        st.markdown(_table_html, unsafe_allow_html=True)
    if _verd:
        st.info(mv_sum(_verd))

    # --- Registro de verificación de sensores (del software de campo) ---
    _scr = (D.get("payload") or {}).get("sensor_check")
    if _scr:
        with st.expander(f"🔴 Sensor verification record — {_scr.get('n_ok','?')}/{_scr.get('n_total','?')} "
                         f"channels OK ({str(_scr.get('ts',''))[:16]})", expanded=False):
            _png = _scr.get("png_b64")
            if _png:
                st.markdown(f'<img src="data:image/png;base64,{_png}" '
                            'style="width:100%;border:1px solid #e2e8f0;border-radius:10px">',
                            unsafe_allow_html=True)
            _rows = _scr.get("rows") or []
            if _rows:
                _show_table(["Channel", "RMS", "Peak", "Status"],
                            [[f"<b>{r[0]}</b>", f"<span class='num'>{r[1]}</span>",
                              f"<span class='num'>{r[2]}</span>", _status_pill(r[3])] for r in _rows])

# ---------------------------------------------------------------- 5 SSI
if nav == T_SSI:
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
        _show_table(["#", "Frequency", "± Hz", "Damping ζ", "± %"],
                    [[f"<span class='idx'>{i+1}</span>", f"<span class='fn'>{m.frequency_hz:.3f}<span class='u'> Hz</span></span>",
                      f"<span class='num'>{m.std_frequency_hz:.3f}</span>",
                      f"<span class='num'>{m.damping_ratio_pct:.3f}<span class='u'> %</span></span>",
                      f"<span class='num'>{m.std_damping_pct:.3f}</span>"] for i, m in enumerate(ssi.modes)])
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
        _show_table(["#", "Frequency", "± Hz", "Damping ζ", "± %"],
                    [[f"<span class='idx'>{i+1}</span>", f"<span class='fn'>{m['fn']:.3f}<span class='u'> Hz</span></span>",
                      f"<span class='num'>{m.get('std_fn',0.0):.3f}</span>",
                      f"<span class='num'>{m['zeta']:.3f}<span class='u'> %</span></span>",
                      f"<span class='num'>{m.get('std_zeta',0.0):.3f}</span>"] for i, m in enumerate(_ssi["modes"])])
        st.caption("Real SSI-COV stabilization diagram from the field run.")
    else:
        st.info("SSI-COV runs on the raw time series in the field app. This cloud run stores the "
                "identified modes below (raw record stays on the field laptop).")
        _show_table(["Frequency", "Damping ζ", "Class"],
                    [[f"<span class='fn'>{m['fn']:.2f}<span class='u'> Hz</span></span>",
                      f"<span class='num'>{m['zeta']:.3f}<span class='u'> %</span></span>",
                      f"<span class='cls'>{m['cls']}</span>"] for m in D["oma_modes"]])

# ---------------------------------------------------------------- 6 COMPARATIVE
if nav == T_CMP:
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
            _show_dicts(correlation_table(matches))
            st.info(ema_oma_summary(matches))

# ---------------------------------------------------------------- 7 CAMPBELL
if nav == T_CAMP:
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
            _show_dicts(crossings_table(crossings))
            st.info(camp_summary(crossings))

# ---------------------------------------------------------------- 8 MODE SHAPES
if nav == T_SHAPES:
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
if nav == T_TREND:
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
    _show_dicts(rows)
    _drops = [r for r in rows if r["Δ% (first→last)"] != "—" and float(r["Δ% (first→last)"].rstrip('%')) <= -3]
    if _drops:
        st.warning("⚠ A natural frequency dropped ≥3% over time — possible loss of stiffness "
                   "(loosening, cracking, or skid/base degradation). Investigate.")
    else:
        st.success("Natural frequencies stable across runs — no stiffness loss detected.")

# ---------------------------------------------------------------- 9 PRELIMINARY
if nav == T_REPORT:
    _sec("Report", "Full OMA report — cover · table of contents · configuration · "
         "sensor check · spectral density · mode shapes · Campbell · findings",
         "SIGA-FMT-179 · ISO 20816 · API 684")
    from datetime import date as _date
    _scr = (D.get("payload") or {}).get("sensor_check")

    # --- Auto-diagnóstico + hallazgos/recomendaciones automáticos (editables) ---
    _mh = [m["fn"] for m in D["oma_modes"] if m["cls"] != "spurious"] or [m["fn"] for m in D["oma_modes"]]
    _cx = compute_crossings(_mh, 0.0, D["rpm"] * 1.35, orders=[0.5, 1, 2, 3, 4, 5, 6, 7, 8],
                            bands=[SpeedBand(D["rpm"] * 0.85, D["rpm"] * 1.15, "Operación ±15%")]) if _mh else []
    if D["oma_modes"]:
        _nar = _narrative(lay.name, D["oma_modes"], D["rpm"], _verd, _cx)
        st.markdown(f"<div style='background:#eef6ff;border-left:4px solid {BLUE};border-radius:8px;"
                    f"padding:12px 16px;margin:6px 0'><b>Auto-diagnosis</b><br>{_nar}</div>",
                    unsafe_allow_html=True)

    def _auto_findings(es):
        out = []
        _ib = [c for c in _cx if c.in_band]
        if _ib:
            _w = min(_ib, key=lambda c: c.sep_margin_pct)
            out.append(f"El modo de {_w.mode_hz:.1f} Hz coincide con el orden {_w.order:g}× dentro de la banda "
                       f"de ±15% de la velocidad de operación (riesgo de resonancia)." if es else
                       f"The {_w.mode_hz:.1f} Hz mode coincides with the {_w.order:g}× order within the ±15% "
                       f"operating-speed band (resonance risk).")
        else:
            out.append("No se detectaron coincidencias de resonancia dentro de la banda de operación." if es else
                       "No resonance coincidences within the operating band.")
        return out

    def _auto_recs(es):
        return ["Correlacionar amplitud y fase de vibración contra la velocidad en operación (API 684)." if es else
                "Correlate vibration amplitude and phase against running speed (API 684).",
                "Si un modo cercano a 1×/2× presenta amplitud elevada, evaluar la rigidización de la base/skid "
                "y verificar torque de anclajes." if es else
                "If a mode near 1×/2× shows high amplitude, evaluate base/skid stiffening and check anchor-bolt torque."]

    if not D["oma_modes"]:
        st.info("Select a field run with identified modes to assemble the report.")
    else:
        _lang = st.radio("Language", ["Español", "English"], horizontal=True, key="rep_lang")
        _es = (_lang == "Español")
        # ---------------- Metadata del reporte (consecutivo, firmas) -------------
        st.markdown("**Report identification**")
        _uname = (_user.get("name") or _user.get("full_name") or _user.get("email", "")).split("@")[0]
        m1 = st.columns([1.2, 1, 1.4])
        with m1[0]:
            _consec = st.text_input("Consecutive", key="rep_consec",
                                    value=st.session_state.get("rep_consec", f"OMA-{_date.today().year}-001"))
        with m1[1]:
            _rdate = st.text_input("Date", key="rep_date", value=st.session_state.get("rep_date", str(_date.today())))
        with m1[2]:
            _asset = st.text_input("Asset / Tag", key="rep_asset", value=st.session_state.get("rep_asset", lay.tag or lay.name))
        m2 = st.columns(2)
        with m2[0]:
            _client = st.text_input("Client", key="rep_client", value=st.session_state.get("rep_client", lay.client or ""))
        with m2[1]:
            _location = st.text_input("Location", key="rep_loc", value=st.session_state.get("rep_loc", lay.location or ""))
        m3 = st.columns(2)
        with m3[0]:
            _prep_by = st.text_input("Prepared by (Realizado por)", key="rep_prep", value=st.session_state.get("rep_prep", _uname))
            _prep_role = st.text_input("Role", key="rep_prep_role", value=st.session_state.get("rep_prep_role", "Especialista"))
        with m3[1]:
            _rev_by = st.text_input("Approved by (Aprobado por)", key="rep_rev", value=st.session_state.get("rep_rev", ""))
            _rev_role = st.text_input("Role ", key="rep_rev_role", value=st.session_state.get("rep_rev_role", "Aprobado por"))
        _city = st.text_input("City", key="rep_city", value=st.session_state.get("rep_city", "Bogotá D.C."))

        # ---------------- Hallazgos y Recomendaciones EDITABLES ------------------
        st.markdown("**Findings & recommendations** *(one per line — edit freely)*")
        fr = st.columns(2)
        with fr[0]:
            _find_txt = st.text_area("Findings (Hallazgos)", key="rep_find",
                                     value=st.session_state.get("rep_find", "\n".join(_auto_findings(_es))), height=150)
        with fr[1]:
            _rec_txt = st.text_area("Recommendations (Recomendaciones)", key="rep_rec",
                                    value=st.session_state.get("rep_rec", "\n".join(_auto_recs(_es))), height=150)
        st.caption("Embedded in the PDF: machine 3D configuration · sensor-check status · spectral density "
                   "(singular values) · mode shapes (3D) · Campbell (API 684) · EMA↔OMA correlation.")

        if st.button("📄 Generate full report (PDF)", type="primary", key="rep_gen"):
            try:
                from core.modal.run_report import build_report_from_run
                import base64 as _b64m
                with st.spinner("Rendering figures (config · sensors · mode shapes) and building the SIGA report…"):
                    pts = lay.active_points()
                    # formas modales 3D
                    shape_pngs = []
                    for i, m in enumerate(D["oma_modes"][:3]):
                        if D["shapes"] and i < len(D["shapes"]) and D["shapes"][i] is not None \
                                and len(D["shapes"][i]) == len(pts):
                            a = np.asarray(D["shapes"][i], float)
                        else:
                            a = np.random.default_rng(i + 1).standard_normal(len(pts))
                        try:
                            shape_pngs.append(_geometry_fig(lay, amp=a, height=460).to_image(
                                format="png", width=1100, height=620, scale=2))
                        except Exception:  # noqa: BLE001
                            shape_pngs.append(None)
                    # configuración 3D
                    try:
                        config_png = _geometry_fig(lay, height=520).to_image(format="png", width=1200, height=680, scale=2)
                    except Exception:  # noqa: BLE001
                        config_png = None
                    # verificación de sensores (imagen del campo)
                    sensor_png = None; sensor_rows = None
                    if _scr:
                        if _scr.get("png_b64"):
                            try:
                                sensor_png = _b64m.b64decode(_scr["png_b64"])
                            except Exception:  # noqa: BLE001
                                sensor_png = None
                        sensor_rows = _scr.get("rows") or None
                    _findings = [x.strip() for x in _find_txt.splitlines() if x.strip()]
                    _recs = [x.strip() for x in _rec_txt.splitlines() if x.strip()]
                    _meta_extra = {"consecutive": _consec, "report_date": _rdate, "asset": _asset,
                                   "client": _client, "location": _location,
                                   "prepared_by": _prep_by, "prepared_role": _prep_role, "prepared_city": _city,
                                   "reviewed_by": _rev_by, "reviewed_role": _rev_role, "reviewed_city": _city}
                    pdf = build_report_from_run(
                        D.get("payload") or {}, bilingual_es=_es, shape_pngs=shape_pngs,
                        findings=_findings, recommendations=_recs, meta_extra=_meta_extra,
                        config_png=config_png, sensor_png=sensor_png, sensor_rows=sensor_rows)
                st.session_state["_modal_report_pdf"] = pdf
                st.session_state["_modal_report_meta"] = {
                    "consecutive": _consec, "client": _client, "asset": _asset,
                    "location": _location, "report_date": _rdate, "report_title": "Reporte OMA",
                    "prepared_by": _prep_by, "reviewed_by": _rev_by, "format_code": "SIGA-FMT-179"}
                st.success("Report generated.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not build the report: {type(e).__name__}: {e}")

        _pdf = st.session_state.get("_modal_report_pdf")
        if _pdf:
            dl = st.columns([1, 1])
            with dl[0]:
                st.download_button("⬇ Download PDF", data=_pdf,
                                   file_name=f"{(_consec or 'OMA').replace(' ', '_')}.pdf",
                                   mime="application/pdf", use_container_width=True)
            with dl[1]:
                _share = st.checkbox("Share with client", key="rep_share")
                if st.button("✅ Approve & store", key="rep_archive", use_container_width=True):
                    try:
                        from core.reports_archive import archive_report_pdf
                        r = archive_report_pdf(pdf_bytes=_pdf,
                                               meta=st.session_state.get("_modal_report_meta", {}),
                                               owner_email=_user.get("email", ""),
                                               shared_with_client=bool(_share),
                                               extra_notes="Watermelon Modal (web) OMA report")
                        if r.get("ok"):
                            st.success(f"Stored · {r.get('archive_id','')}")
                        else:
                            st.error(f"Could not store: {r.get('error','?')}")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Could not store: {type(e).__name__}: {e}")
            import base64
            _b64 = base64.b64encode(_pdf).decode()
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{_b64}" width="100%" height="820" '
                f'style="border:1px solid #e2e8f0;border-radius:8px"></iframe>',
                unsafe_allow_html=True)

    # ---------------- Reportes almacenados (archivo) -------------------------
    with st.expander("📁 Stored OMA reports", expanded=False):
        try:
            from core.reports_archive import list_archived_reports, get_archived_pdf_bytes
            _arch = list_archived_reports(viewer_email=_user.get("email", ""), viewer_role=_my_role,
                                          text_search="OMA", limit=100)
        except Exception:  # noqa: BLE001
            _arch = []
        if not _arch:
            st.caption("No stored OMA reports yet. Generate one above and click **Approve & store**.")
        else:
            for _a in _arch[:25]:
                _am = _a.get("meta", {}) or {}
                _cid = _a.get("archive_id", "")
                cols = st.columns([3, 2, 2, 1.4])
                cols[0].markdown(f"**{_am.get('consecutive','—')}** · {_am.get('client','')}")
                cols[1].write(_am.get("asset", ""))
                cols[2].write(str(_a.get("archived_at", ""))[:16])
                if cols[3].button("⬇", key=f"dl_{_cid}"):
                    try:
                        _bytes = get_archived_pdf_bytes(_cid, _user.get("email", ""), _my_role)
                        if _bytes:
                            st.download_button("Download", data=_bytes, file_name=f"{_am.get('consecutive','OMA')}.pdf",
                                               mime="application/pdf", key=f"dlb_{_cid}")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"{type(e).__name__}: {e}")
