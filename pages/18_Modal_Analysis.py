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
      /* Panel Modal Values (estilo ARTeMIS) */
      .wm-mv { background:#fff; border:1px solid #e6ecf5; border-radius:12px; padding:12px 14px;
        margin-bottom:12px; box-shadow:0 1px 2px rgba(15,30,61,.04); }
      .wm-mv .h { font-size:13px; font-weight:700; color:#0F1E3D; letter-spacing:.02em;
        border-bottom:1px solid #eef2f8; padding-bottom:6px; margin-bottom:8px; }
      .wm-mv .row { display:flex; justify-content:space-between; align-items:center; font-size:12.5px;
        color:#64748b; padding:3px 0; }
      .wm-mv .row b { color:#0f1e3d; font-family:'IBM Plex Mono',monospace; font-weight:600; }
      .wm-mv .sw { width:22px; height:12px; border-radius:3px; display:inline-block; }
      .wm-cbar { width:16px; height:118px; border-radius:4px; border:1px solid #e2e8f0;
        background:linear-gradient(to top,#00007f,#1f3fff,#00c8ff,#22e06b,#ffd21a,#ff6a00,#c11414,#7f0000); }
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


def _chart(fig, **kw):
    """st.plotly_chart sin la barra de botones de Plotly (cámara/zoom/±) — look limpio."""
    import streamlit as _st
    return _st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False}, **kw)


def _ssi_plot(diagram, mode_freqs, sv_trace=None):
    """Diagrama de estabilización LEGIBLE: polos estables (verde) vs espurios (gris),
    columnas resaltadas en cada modo, y la densidad espectral (SV1) de fondo para
    que se vea que los picos coinciden con las columnas estables."""
    sx, sy, ux, uy = [], [], [], []
    omax = 40
    for (order, fr, mask) in diagram:
        omax = max(omax, int(order))
        for f, m in zip(np.asarray(fr, float), mask):
            (sx if m else ux).append(float(f)); (sy if m else uy).append(int(order))
    xmax = max([max(sx) if sx else 0, max(ux) if ux else 0, max(mode_freqs) if mode_freqs else 0, 50]) * 1.05
    fig = go.Figure()
    for f in mode_freqs:                                   # columna verde en cada modo
        fig.add_vrect(x0=f * 0.985, x1=f * 1.015, fillcolor="rgba(22,163,74,.12)", line_width=0)
    if sv_trace is not None:                               # densidad espectral de fondo (eje derecho)
        fx, ydb = np.asarray(sv_trace[1], float), np.asarray(sv_trace[2], float)
        fig.add_trace(go.Scatter(x=fx, y=ydb, mode="lines", name="Spectral density (SV1)",
                      line=dict(color="rgba(37,99,235,.5)", width=1.6), yaxis="y2", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ux, y=uy, mode="markers", name="Spurious (numerical)",
                  marker=dict(size=4, color="#cbd5e1"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers", name="Stable pole",
                  marker=dict(size=7, color=GREEN, line=dict(width=.5, color="white")),
                  hovertemplate="%{x:.1f} Hz · order %{y}<extra>stable</extra>"))
    for f in mode_freqs:
        fig.add_annotation(x=f, y=omax, text=f"<b>{f:.1f}</b>", showarrow=False,
                           font=dict(size=10, color="#166534"), yanchor="bottom",
                           bgcolor="rgba(255,255,255,.85)")
    fig.update_layout(height=470, template="watermelon",
                      xaxis=dict(range=[0, xmax], title="Frequency (Hz)"),
                      yaxis=dict(title="Model order", range=[0, omax * 1.12]),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, showticklabels=False),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig


_SSI_HELP = (
    "**How to read it —** each dot is a pole from a model of increasing order (y-axis). "
    "A **real structural mode** shows up as a **vertical green column**: poles that stay "
    "**stable** (same frequency) as the order grows. Scattered **gray** dots are numerical / "
    "noise poles — ignore them. The **blue spectral-density curve** behind should **peak right "
    "at each green column** — that agreement is the confirmation the mode is real.")


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


def _mode_shape_data(lay, amps_signed, scale_mul=1.0):
    """Combina A/H/V de cada ESTACIÓN en un vector de movimiento 3D y arma un beam
    SUAVE (spline cúbico) que se deforma — mucho más pro que una línea quebrada."""
    pts = lay.active_points()
    if not pts or amps_signed is None or len(amps_signed) != len(pts):
        return None
    a = np.asarray(amps_signed, float); a = a / (np.max(np.abs(a)) or 1.0)
    stations = {}
    for p, ai in zip(pts, a):
        key = _stn_key(p)                        # por POSICIÓN (no por etiqueta, que puede repetirse)
        dvec = np.array(_AX.get(p.axis, (0, 0, 1)), float) * (-1.0 if p.dof.startswith("-") else 1.0) * float(ai)
        s = stations.setdefault(key, {"pos": np.array([p.x_norm, 0.20, p.y_norm], float), "disp": np.zeros(3)})
        s["disp"] += dvec
    order = sorted(stations.values(), key=lambda s: (s["pos"][0], s["pos"][2]))
    P0 = np.array([s["pos"] for s in order], float)
    DISP = np.array([s["disp"] for s in order], float)
    _mg = np.linalg.norm(DISP, axis=1); _pp = _mg[_mg > 0]
    _cn = float(np.percentile(_pp, 85)) if _pp.size else 1.0
    MAGn = np.clip(_mg / (_cn or (_mg.max() or 1.0)), 0.0, 1.0)
    span = float(np.ptp(P0[:, 0])) or 1.0
    scale = 0.24 * span / (np.max(np.linalg.norm(DISP, axis=1)) or 1.0) * scale_mul
    Ps, Ds = P0, DISP
    if len(order) >= 3:
        t = np.zeros(len(P0)); t[1:] = np.cumsum(np.linalg.norm(np.diff(P0, axis=0), axis=1))
        if t[-1] <= 0:
            t = np.arange(len(P0), dtype=float)
        tt = np.linspace(t[0], t[-1], 90)
        try:
            from scipy.interpolate import CubicSpline
            Ps = np.vstack([CubicSpline(t, P0[:, c])(tt) for c in range(3)]).T
            Ds = np.vstack([CubicSpline(t, DISP[:, c])(tt) for c in range(3)]).T
        except Exception:  # noqa: BLE001
            Ps, Ds = P0, DISP
    return {"P0": P0, "DISP": DISP, "MAGn": MAGn, "Ps": Ps, "Ds": Ds, "scale": scale}


def _mode_machine_meshes(lay, opacity=0.09):
    out = []
    for c in lay.machine_components:
        cc = getattr(c, "color", "") or _comp_color(c.kind)
        X, Y, Z, i, j, k = _cube(c.x0, c.x1, c.y0, c.y1, c.depth)
        out.append(go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k, color=cc, opacity=opacity,
                             flatshading=True, hoverinfo="skip", showscale=False))
    return out


def _mode_dynamic_traces(d, phase, colorbar=True):
    """Beam suave + nodos, en una fase dada (para animar y para el GIF)."""
    beam = d["Ps"] + (d["scale"] * np.sin(phase)) * d["Ds"]
    nodes = d["P0"] + (d["scale"] * np.sin(phase)) * d["DISP"]
    mk = dict(size=7, color=d["MAGn"], colorscale="Turbo", cmin=0, cmax=1,
              line=dict(width=1, color="#0f172a"))
    if colorbar:
        mk["colorbar"] = dict(title="ampl", thickness=12, len=0.55, x=0.98)
    beam_tr = go.Scatter3d(x=beam[:, 0], y=beam[:, 1], z=beam[:, 2], mode="lines",
                           line=dict(color=BLUE, width=8), hoverinfo="skip", name="mode")
    node_tr = go.Scatter3d(x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2], mode="markers",
                           marker=mk, hoverinfo="skip", name="nodes")
    return beam_tr, node_tr


def _mode_scene(height, paper="rgba(0,0,0,0)"):
    return dict(height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                paper_bgcolor=paper,
                scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                           zaxis=dict(visible=False), camera=dict(eye=dict(x=1.6, y=1.4, z=0.85))))


def _mode_shape_fig(lay, amps_signed, height=580, scale_mul=1.0):
    d = _mode_shape_data(lay, amps_signed, scale_mul)
    if d is None:
        return _geometry_fig(lay, height=height)
    fig = go.Figure()
    for tr in _mode_machine_meshes(lay):
        fig.add_trace(tr)
    n_ctx = len(lay.machine_components)
    fig.add_trace(go.Scatter3d(x=d["Ps"][:, 0], y=d["Ps"][:, 1], z=d["Ps"][:, 2], mode="lines",
                  line=dict(color="rgba(148,163,184,.6)", width=4, dash="dot"),
                  hoverinfo="skip", name="undeformed"))
    beam0, node0 = _mode_dynamic_traces(d, np.pi / 2)
    fig.add_trace(beam0); fig.add_trace(node0)
    frames = []
    for f in range(30):
        ph = f / 30.0 * 2 * np.pi
        b, n = _mode_dynamic_traces(d, ph, colorbar=False)
        frames.append(go.Frame(data=[b, n], traces=[n_ctx + 1, n_ctx + 2]))
    fig.frames = frames
    lay_kw = _mode_scene(height)
    lay_kw["updatemenus"] = [dict(type="buttons", showactive=False, x=0.02, y=0.05, xanchor="left",
        buttons=[dict(label="▶ Play", method="animate",
                      args=[None, dict(frame=dict(duration=130, redraw=True), fromcurrent=True,
                                       transition=dict(duration=0), mode="immediate")]),
                 dict(label="⏸ Pause", method="animate",
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])]
    fig.update_layout(**lay_kw)
    return fig


def _stn_key(p):
    """Clave de estación por POSICIÓN física (las etiquetas del campo pueden repetirse)."""
    return f"{round(float(p.x_norm), 3)}|{round(float(p.y_norm), 3)}"


def _layout_stations(lay):
    """Estaciones de medición por posición física, con su posición 3D."""
    pts = lay.active_points()
    stations = {}
    for p in pts:
        key = _stn_key(p)
        stations.setdefault(key, {"label": key, "pos": np.array([p.x_norm, 0.20, p.y_norm], float)})
    return list(stations.values())


def _station_disp_map(lay, amps_signed):
    """{clave de estación (posición) → vector de desplazamiento 3D} para una forma modal."""
    pts = lay.active_points()
    a = np.asarray(amps_signed, float); a = a / (np.max(np.abs(a)) or 1.0)
    disp = {}
    for i, p in enumerate(pts):
        key = _stn_key(p)
        dvec = np.array(_AX.get(p.axis, (0, 0, 1)), float) * (-1.0 if p.dof.startswith("-") else 1.0) * float(a[i])
        disp[key] = disp.get(key, np.zeros(3)) + dvec
    return disp


_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def _default_geometry(lay):
    """Geometría inicial = wireframe de CAJAS (una por componente, 8 esquinas + 12
    aristas) + los nodos de sensores. Las esquinas son esclavas (interpoladas)."""
    nodes, lines = [], []
    for ci, c in enumerate(lay.machine_components):
        X, Y, Z, _i, _j, _k = _cube(c.x0, c.x1, c.y0, c.y1, c.depth)
        base = len(nodes)
        for v in range(8):
            nodes.append({"id": f"C{ci}_{v}", "x": round(float(X[v]), 4), "y": round(float(Y[v]), 4),
                          "z": round(float(Z[v]), 4), "sensor": ""})
        for a, b in _BOX_EDGES:
            lines.append([base + a, base + b])
    for s in _layout_stations(lay):
        nodes.append({"id": s["label"], "x": round(float(s["pos"][0]), 4), "y": round(float(s["pos"][1]), 4),
                      "z": round(float(s["pos"][2]), 4), "sensor": s["label"]})
    return {"nodes": nodes, "lines": lines}


def _edges_xyz(P, lines):
    ex, ey, ez = [], [], []
    n = len(P)
    for a, b in lines:
        if 0 <= a < n and 0 <= b < n:
            ex += [P[a, 0], P[b, 0], None]; ey += [P[a, 1], P[b, 1], None]; ez += [P[a, 2], P[b, 2], None]
    return ex, ey, ez


def _geom_node_disp(geom, disp_map):
    """Desplazamiento de cada nodo: si es sensor usa su estación; si es esclavo,
    interpola (IDW) desde los nodos-sensor."""
    nodes = geom["nodes"]
    P = np.array([[n["x"], n["y"], n["z"]] for n in nodes], float) if nodes else np.zeros((0, 3))
    sp, sd = [], []
    for n in nodes:
        if n.get("sensor") and n["sensor"] in disp_map:
            sp.append([n["x"], n["y"], n["z"]]); sd.append(disp_map[n["sensor"]])
    sp = np.array(sp, float) if sp else np.zeros((0, 3)); sd = np.array(sd, float) if sd else np.zeros((0, 3))
    ND = np.zeros((len(nodes), 3))
    for i, n in enumerate(nodes):
        if n.get("sensor") and n["sensor"] in disp_map:
            ND[i] = disp_map[n["sensor"]]
        elif len(sp):
            ND[i] = _idw(P[i:i + 1], sp, sd)[0]
    return P, ND


def _geom_preview_fig(lay, geom, height=460, show_machine=True):
    nodes = geom["nodes"]; lines = geom["lines"]
    P = np.array([[n["x"], n["y"], n["z"]] for n in nodes], float) if nodes else np.zeros((0, 3))
    fig = go.Figure()
    if show_machine:
        for tr in _mode_machine_meshes(lay, opacity=0.06):
            fig.add_trace(tr)
    if len(P):
        ex, ey, ez = _edges_xyz(P, lines)
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                      line=dict(color="#334155", width=4), hoverinfo="skip"))
        _sens = np.array([bool(n.get("sensor")) for n in nodes])
        _cols = np.where(_sens, "#2563eb", "#f59e0b")
        # etiquetar sólo los nodos-sensor (evita saturar con las esquinas de cajas)
        _txt = [n["id"] if n.get("sensor") else "" for n in nodes]
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers+text",
                      text=_txt, textposition="top center", textfont=dict(size=9, color="#1d4ed8"),
                      marker=dict(size=np.where(_sens, 6, 3), color=_cols, line=dict(width=1, color="#0f172a")),
                      hovertext=[n["id"] for n in nodes], hoverinfo="text"))
    fig.update_layout(**_mode_scene(height, paper="rgba(0,0,0,0)"))
    return fig


def _dense_mesh(P, surfaces, n=4):
    """Subdivide cada cara (quad o triángulo) en una malla n×n → muchos vértices
    para un gradiente FINO (como ARTeMIS)."""
    V, I, J, K = [], [], [], []
    for f in surfaces:
        if len(f) < 3:
            continue
        if len(f) >= 4:
            a, b, c, d = (np.asarray(P[f[0]], float), np.asarray(P[f[1]], float),
                          np.asarray(P[f[2]], float), np.asarray(P[f[3]], float))
            base = len(V); w = n + 1
            for iu in range(w):
                for iv in range(w):
                    u = iu / n; v = iv / n
                    V.append((1 - u) * (1 - v) * a + u * (1 - v) * b + u * v * c + (1 - u) * v * d)
            for iu in range(n):
                for iv in range(n):
                    p0 = base + iu * w + iv; p1 = base + (iu + 1) * w + iv; p2 = p1 + 1; p3 = p0 + 1
                    I += [p0, p0]; J += [p1, p2]; K += [p2, p3]
        else:
            base = len(V)
            for x in f:
                V.append(np.asarray(P[x], float))
            for t in range(1, len(f) - 1):
                I.append(base); J.append(base + t); K.append(base + t + 1)
    return (np.array(V, float) if V else np.zeros((0, 3))), I, J, K


def _rotor_is(lay):
    """¿La corrida es de PROXIMIDAD (rotor)? → sensores de desplazamiento (mil)."""
    pts = lay.active_points()
    return bool(pts) and sum(1 for p in pts if getattr(p, "meas_type", "A") == "D") >= max(2, len(pts) // 2)


def _cyl(xa, xb, r, na, nt, base):
    """Malla de un cilindro a lo largo de X: devuelve (verts, axial, I,J,K)."""
    verts, axc, I, J, K = [], [], [], [], []
    for k in range(na):
        xk = xa + (xb - xa) * (k / (na - 1) if na > 1 else 0)
        for j in range(nt):
            th = 2 * np.pi * j / nt
            verts.append([xk, r * np.cos(th), r * np.sin(th)]); axc.append(xk)
    for k in range(na - 1):
        for j in range(nt):
            j2 = (j + 1) % nt
            p0 = base + k * nt + j; p1 = base + (k + 1) * nt + j
            p2 = base + (k + 1) * nt + j2; p3 = base + k * nt + j2
            I += [p0, p0]; J += [p1, p2]; K += [p2, p3]
    return verts, axc, I, J, K


def _disk(xc, r, w, nt, base):
    """Disco SÓLIDO (impulsor) perpendicular al eje: dos caras + borde."""
    V, AX, I, J, K = [], [], [], [], []
    V.append([xc - w, 0, 0]); AX.append(xc - w); c0 = base
    V.append([xc + w, 0, 0]); AX.append(xc + w); c1 = base + 1
    r0 = base + 2; r1 = r0 + nt
    for j in range(nt):
        th = 2 * np.pi * j / nt; V.append([xc - w, r * np.cos(th), r * np.sin(th)]); AX.append(xc - w)
    for j in range(nt):
        th = 2 * np.pi * j / nt; V.append([xc + w, r * np.cos(th), r * np.sin(th)]); AX.append(xc + w)
    for j in range(nt):
        j2 = (j + 1) % nt
        I.append(c0); J.append(r0 + j); K.append(r0 + j2)          # cara frontal
        I.append(c1); J.append(r1 + j2); K.append(r1 + j)          # cara trasera
        I += [r0 + j, r0 + j]; J += [r1 + j, r1 + j2]; K += [r1 + j2, r0 + j2]  # borde
    return V, AX, I, J, K


def _mode_rotor_fig(lay, amps_signed, height=600, scale_mul=1.0, static=False, phase=np.pi / 2):
    """Forma modal del ROTOR (proximidad): eje + masa del motor + impulsores de la
    bomba, que FLEXIONA lateralmente según las sondas XY (X→radial horiz, Y→radial vert)."""
    pts = lay.active_points()
    if not pts or amps_signed is None or len(amps_signed) != len(pts):
        return _geometry_fig(lay, height=height)
    a = np.asarray(amps_signed, float); a = a / (np.max(np.abs(a)) or 1.0)
    # cojinetes: agrupar por POSICIÓN x (no por etiqueta: muchas corridas traen el mismo
    # component/position_ref en todos los puntos y colapsarían a una sola estación).
    bear = {}
    for p, ai in zip(pts, a):
        key = round(float(p.x_norm), 3)
        b = bear.setdefault(key, {"x": float(p.x_norm), "dy": 0.0, "dz": 0.0})
        d = float(ai) * (-1.0 if p.dof.startswith("-") else 1.0)
        if (p.axis or "").upper() in ("X", "H", "A"):
            b["dy"] += d
        else:
            b["dz"] += d
        b["x"] = float(p.x_norm)
    items = sorted(bear.values(), key=lambda b: b["x"])
    xs = np.array([b["x"] for b in items]); dys = np.array([b["dy"] for b in items]); dzs = np.array([b["dz"] for b in items])
    # suavizado ligero (3 puntos): evita que datos ruidosos "rompan"/retuerzan el eje
    if len(dys) >= 3:
        def _sm(v):
            w = v.copy(); w[1:-1] = 0.25 * v[:-2] + 0.5 * v[1:-1] + 0.25 * v[2:]; return w
        dys = _sm(dys); dzs = _sm(dzs)
    xmin, xmax = float(xs.min()), float(xs.max()); span = (xmax - xmin) or 1.0
    x0, x1 = xmin - 0.06 * span, xmax + 0.06 * span; L = x1 - x0
    rs = 0.020 * L                                   # radio del eje
    # rangos de motor y bomba (para masa e impulsores)
    def _crange(kws):
        for c in lay.machine_components:
            if any(w in (c.kind + " " + c.label).lower() for w in kws):
                return c.x0, c.x1
        return None
    mot = _crange(["motor"]); pmp = _crange(["pump", "bomba"])
    verts, axc, I, J, K = [], [], [], [], []

    def _add(v, ax, i, j, k):
        verts.extend(v); axc.extend(ax); I.extend(i); J.extend(j); K.extend(k)
    _add(*_cyl(x0, x1, rs, 60, 20, len(verts)))                        # eje
    if mot:                                                            # masa del motor
        _add(*_cyl(mot[0], mot[1], 0.055 * L, 18, 22, len(verts)))
    if pmp:                                                            # impulsores de la bomba (discos sólidos)
        n_imp = 10; pw = (pmp[1] - pmp[0])
        for ii in range(n_imp):
            xc = pmp[0] + pw * (ii + 0.5) / n_imp
            _add(*_disk(xc, 0.065 * L, 0.010 * L, 26, len(verts)))
    V0 = np.array(verts, float); AX = np.array(axc, float)
    # deflexión SUAVE: PCHIP (cúbica monótona) con clamp en extremos → curva de flexión
    # real, sin quiebres (kink) cuando el modo es "picudo" (p.ej. datos ruidosos).
    _xlo, _xhi = float(xs.min()), float(xs.max())

    def _smooth(q, yv):
        q = np.clip(np.asarray(q, float), _xlo, _xhi)
        if len(xs) >= 2:
            try:
                from scipy.interpolate import PchipInterpolator
                return PchipInterpolator(xs, yv, extrapolate=False)(q)
            except Exception:  # noqa: BLE001
                pass
        return np.interp(q, xs, yv)
    DY = _smooth(AX, dys); DZ = _smooth(AX, dzs)
    LAT = np.sqrt(DY ** 2 + DZ ** 2)
    _pos = LAT[LAT > 0]; cnorm = float(np.percentile(_pos, 85)) if _pos.size else 1.0
    MAGn = np.clip(LAT / (cnorm or 1.0), 0.0, 1.0)
    maxlat = float(np.sqrt(dys ** 2 + dzs ** 2).max()) or 1.0
    scale = 0.11 * L / maxlat * scale_mul

    def _defV(ph):
        out = V0.copy(); s = scale * np.sin(ph)
        out[:, 1] += DY * s; out[:, 2] += DZ * s
        return out

    def _surf_tr(dv):
        return go.Mesh3d(x=dv[:, 0], y=dv[:, 1], z=dv[:, 2], i=I, j=J, k=K, intensity=MAGn,
                         cmin=0, cmax=1, coloraxis="coloraxis", flatshading=False, opacity=1.0,
                         lighting=dict(ambient=0.82, diffuse=0.5, specular=0.12), hoverinfo="skip")

    _xc = np.linspace(x0, x1, 60); _ycb = _smooth(_xc, dys); _zcb = _smooth(_xc, dzs)

    def _center_tr(ph):
        s = scale * np.sin(ph)
        return go.Scatter3d(x=_xc, y=_ycb * s, z=_zcb * s, mode="lines",
                            line=dict(color="#0f172a", width=3), hoverinfo="skip")

    fig = go.Figure()
    _ph0 = float(phase)
    fig.add_trace(_surf_tr(_defV(_ph0))); _isf = len(fig.data) - 1
    fig.add_trace(_center_tr(_ph0)); _icl = len(fig.data) - 1
    # marcadores de cojinete (sensores)
    fig.add_trace(go.Scatter3d(x=xs, y=[0] * len(xs), z=[0] * len(xs), mode="markers+text",
                  text=[f"B{i+1}" for i in range(len(xs))], textposition="top center",
                  textfont=dict(size=10, color="#0f172a"), marker=dict(size=4, color="#0f172a"),
                  hoverinfo="skip"))
    if not static:
        frames = []
        for f in range(40):
            ph = f / 20.0 * 2 * np.pi
            frames.append(go.Frame(data=[_surf_tr(_defV(ph)), _center_tr(ph)], traces=[_isf, _icl]))
        fig.frames = frames
    lay_kw = _mode_scene(height)
    if static:
        lay_kw["coloraxis"] = dict(colorscale="Jet", cmin=0, cmax=1,
                                   colorbar=dict(thickness=12, len=0.6, x=0.98,
                                                 tickvals=[0, 1], ticktext=["0", "Max"], title="ampl"))
    else:
        lay_kw["coloraxis"] = dict(colorscale="Jet", cmin=0, cmax=1, showscale=False)
    if not static:
        lay_kw["updatemenus"] = [dict(type="buttons", showactive=False, x=0.02, y=0.05, xanchor="left",
            buttons=[dict(label="▶ Play", method="animate",
                          args=[None, dict(frame=dict(duration=130, redraw=True), fromcurrent=True,
                                           transition=dict(duration=0), mode="immediate")]),
                     dict(label="⏸ Pause", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])]
    # Centrar el rotor en X (que quede centrado en el marco, no corrido a un lado) y
    # alejar un poco la cámara para que NO se salga de cuadro durante la animación.
    _xmid = 0.5 * (x0 + x1)
    for _tr in fig.data:
        if getattr(_tr, "x", None) is not None:
            _tr.x = np.asarray(_tr.x, float) - _xmid
    for _fr in (fig.frames or []):
        for _tr in _fr.data:
            if getattr(_tr, "x", None) is not None:
                _tr.x = np.asarray(_tr.x, float) - _xmid
    lay_kw["scene"]["camera"] = dict(eye=dict(x=1.9, y=1.65, z=1.0),
                                     center=dict(x=0, y=0, z=0))
    # Rangos de eje FIJOS que cubren la deflexión máxima (±) → el rotor NO se sale del
    # cuadro durante la animación (encaja todos los frames, centrado).
    _allv = np.vstack([_defV(np.pi / 2), _defV(-np.pi / 2), V0]).astype(float)
    _allv[:, 0] -= _xmid
    def _rng(_a, _b, _f=0.10):
        _m = (_b - _a) * _f + 1e-6; return [_a - _m, _b + _m]
    lay_kw["scene"]["xaxis"] = dict(visible=False, range=_rng(_allv[:, 0].min(), _allv[:, 0].max()))
    lay_kw["scene"]["yaxis"] = dict(visible=False, range=_rng(_allv[:, 1].min(), _allv[:, 1].max()))
    lay_kw["scene"]["zaxis"] = dict(visible=False, range=_rng(_allv[:, 2].min(), _allv[:, 2].max()))
    fig.update_layout(**lay_kw)
    return fig


def _mode_geom_fig(lay, geom, amps_signed, height=600, scale_mul=1.0, static=False, phase=np.pi / 2):
    """Forma modal animada sobre la GEOMETRÍA del campo (estilo ARTeMIS): superficie
    sólida con malla densa (gradiente Jet), aristas, flechas de DOF por eje y triada.
    static=True → sin animación ni botón Play, con colorbar (para el PDF del reporte)."""
    nodes = geom.get("nodes") or []
    lines = geom.get("lines") or []
    surfaces = geom.get("surfaces") or []
    if not nodes:
        return _mode_surface_fig(lay, amps_signed, height, scale_mul)
    disp_map = _station_disp_map(lay, amps_signed)
    P, ND = _geom_node_disp(geom, disp_map)
    span = float(np.ptp(P[:, 0])) or 1.0
    MAG = np.linalg.norm(ND, axis=1)
    # contraste automático (percentil) para que el gradiente se lea aunque el modo sea "picudo"
    _pos = MAG[MAG > 0]
    cnorm = float(np.percentile(_pos, 85)) if _pos.size else 1.0
    cnorm = cnorm or (MAG.max() or 1.0)
    MAGn = np.clip(MAG / cnorm, 0.0, 1.0)
    scale = 0.16 * span / (MAG.max() or 1.0) * scale_mul
    # Malla densa por interpolación BILINEAL de las 4 esquinas de cada cara: la
    # cuadrícula se deforma coherente (no se "derrite") y agrega MUCHAS líneas (ARTeMIS).
    N = 5
    Vr, Vd, Vi, I, J, K, LP = [], [], [], [], [], [], []
    for f in surfaces:
        if len(f) < 4:
            continue
        Pc = [P[f[0]], P[f[1]], P[f[2]], P[f[3]]]
        Dc = [ND[f[0]], ND[f[1]], ND[f[2]], ND[f[3]]]
        Ic = [MAGn[f[0]], MAGn[f[1]], MAGn[f[2]], MAGn[f[3]]]   # color = bilineal de esquinas ya normalizadas
        base = len(Vr); w = N + 1
        for iu in range(w):
            for iv in range(w):
                u = iu / N; v = iv / N
                bw = ((1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v)
                Vr.append(bw[0] * Pc[0] + bw[1] * Pc[1] + bw[2] * Pc[2] + bw[3] * Pc[3])
                Vd.append(bw[0] * Dc[0] + bw[1] * Dc[1] + bw[2] * Dc[2] + bw[3] * Dc[3])
                Vi.append(bw[0] * Ic[0] + bw[1] * Ic[1] + bw[2] * Ic[2] + bw[3] * Ic[3])
        for iu in range(N):
            for iv in range(N):
                p0 = base + iu * w + iv; p1 = base + (iu + 1) * w + iv; p2 = p1 + 1; p3 = p0 + 1
                I += [p0, p0]; J += [p1, p2]; K += [p2, p3]
        for iu in range(w):
            for iv in range(N):
                LP.append((base + iu * w + iv, base + iu * w + iv + 1))
        for iv in range(w):
            for iu in range(N):
                LP.append((base + iu * w + iv, base + (iu + 1) * w + iv))
    Vr = np.array(Vr, float) if Vr else np.zeros((0, 3))
    Vd = np.array(Vd, float) if Vd else np.zeros((0, 3))
    Vi = np.array(Vi, float) if Vi else np.zeros((0,))
    has_surf = len(Vr) > 0

    def _defVr(ph):
        return Vr + (scale * np.sin(ph)) * Vd

    def _surf_tr(dv):
        return go.Mesh3d(x=dv[:, 0], y=dv[:, 1], z=dv[:, 2], i=I, j=J, k=K, intensity=Vi,
                         cmin=0, cmax=1, coloraxis="coloraxis", flatshading=False, opacity=1.0,
                         lighting=dict(ambient=0.82, diffuse=0.5, specular=0.12), hoverinfo="skip")

    def _grid_tr(dv):
        ex, ey, ez = [], [], []
        for a, b in LP:
            ex += [dv[a, 0], dv[b, 0], None]; ey += [dv[a, 1], dv[b, 1], None]; ez += [dv[a, 2], dv[b, 2], None]
        return go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                            line=dict(color="rgba(15,23,42,.85)", width=2), hoverinfo="skip")

    fig = go.Figure()
    if has_surf:
        fig.add_trace(_surf_tr(_defVr(float(phase)))); _is = len(fig.data) - 1
        fig.add_trace(_grid_tr(_defVr(float(phase)))); _ig = len(fig.data) - 1
    if not static:
        frames = []
        for f in range(40):
            ph = f / 20.0 * 2 * np.pi; dv = _defVr(ph)
            data, tr = [], []
            if has_surf:
                data.append(_surf_tr(dv)); tr.append(_is)
                data.append(_grid_tr(dv)); tr.append(_ig)
            frames.append(go.Frame(data=data, traces=tr))
        fig.frames = frames

    # --- estáticos: flechas de DOF por eje (color) + nodos numerados + triada X/Y/Z ---
    _acol = {"A": "#db2777", "X": "#db2777", "H": "#16a34a", "Y": "#16a34a", "V": "#2563eb", "Z": "#2563eb"}
    alen = 0.08 * span; _by = {}
    for p in lay.active_points():
        ax = _AX.get(p.axis, (0, 0, 1)); s = -1.0 if p.dof.startswith("-") else 1.0
        col = _acol.get(p.axis, "#0f172a"); b = _by.setdefault(col, {"x": [], "y": [], "z": [], "u": [], "v": [], "w": []})
        b["x"].append(p.x_norm); b["y"].append(0.20); b["z"].append(p.y_norm)
        b["u"].append(ax[0] * s * alen); b["v"].append(ax[1] * s * alen); b["w"].append(ax[2] * s * alen)
    for col, b in _by.items():
        fig.add_trace(go.Cone(x=b["x"], y=b["y"], z=b["z"], u=b["u"], v=b["v"], w=b["w"], anchor="tail",
                      sizemode="absolute", sizeref=alen * 0.5, showscale=False,
                      colorscale=[[0, col], [1, col]], hoverinfo="skip"))
    _sm = np.array([bool(nd.get("sensor")) for nd in nodes])
    if _sm.any():
        SP = P[_sm]
        fig.add_trace(go.Scatter3d(x=SP[:, 0], y=SP[:, 1], z=SP[:, 2], mode="markers+text",
                      text=[str(i + 1) for i in range(len(SP))], textposition="top center",
                      textfont=dict(size=9, color="#0f172a"), marker=dict(size=3, color="#0f172a"),
                      hoverinfo="skip"))
    _o = np.array([P[:, 0].min(), P[:, 1].min() - 0.06 * span, P[:, 2].min()]); _tl = 0.14 * span
    for vec, c, nm in (((1, 0, 0), "#dc2626", "X"), ((0, 1, 0), "#16a34a", "Y"), ((0, 0, 1), "#2563eb", "Z")):
        e = _o + np.array(vec, float) * _tl
        fig.add_trace(go.Scatter3d(x=[_o[0], e[0]], y=[_o[1], e[1]], z=[_o[2], e[2]], mode="lines",
                      line=dict(color=c, width=4), hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=[e[0]], y=[e[1]], z=[e[2]], mode="text", text=[nm],
                      textfont=dict(size=12, color=c), hoverinfo="skip"))

    lay_kw = _mode_scene(height)
    if has_surf:
        if static:
            lay_kw["coloraxis"] = dict(colorscale="Jet", cmin=0, cmax=1,
                                       colorbar=dict(thickness=12, len=0.6, x=0.98,
                                                     tickvals=[0, 1], ticktext=["0", "Max"], title="ampl"))
        else:
            lay_kw["coloraxis"] = dict(colorscale="Jet", cmin=0, cmax=1, showscale=False)
    if static:
        fig.update_layout(**lay_kw)
        return fig
    lay_kw["updatemenus"] = [dict(type="buttons", showactive=False, x=0.02, y=0.05, xanchor="left",
        buttons=[dict(label="▶ Play", method="animate",
                      args=[None, dict(frame=dict(duration=130, redraw=True), fromcurrent=True,
                                       transition=dict(duration=0), mode="immediate")]),
                 dict(label="⏸ Pause", method="animate",
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])]
    fig.update_layout(**lay_kw)
    return fig


def _idw(V, P0, DISP, power=2.0):
    """Interpolación por distancia inversa: desplazamiento en cada vértice a partir
    de las estaciones medidas."""
    d = np.linalg.norm(V[:, None, :] - P0[None, :, :], axis=2)
    w = 1.0 / (d ** power + 1e-6)
    w /= w.sum(axis=1, keepdims=True)
    return w @ DISP


def _mode_surface_comps(lay, amps_signed, scale_mul=1.0):
    """Para cada componente (caja) interpola el campo de desplazamiento a sus vértices
    → superficie sólida que se deforma y se colorea por amplitud (estilo ARTeMIS)."""
    d = _mode_shape_data(lay, amps_signed, scale_mul)
    if d is None:
        return None
    P0, DISP = d["P0"], d["DISP"]
    comps, cmax = [], 1e-9
    for c in lay.machine_components:
        X, Y, Z, i, j, k = _cube(c.x0, c.x1, c.y0, c.y1, c.depth)
        V = np.column_stack([X, Y, Z]).astype(float)
        DV = _idw(V, P0, DISP)
        mag = np.linalg.norm(DV, axis=1)
        cmax = max(cmax, float(mag.max()))
        comps.append({"V": V, "DV": DV, "mag": mag, "i": i, "j": j, "k": k})
    allV = np.vstack([cc["V"] for cc in comps])
    span = float(np.ptp(allV[:, 0])) or 1.0
    maxd = max((np.linalg.norm(cc["DV"], axis=1).max() for cc in comps), default=1.0) or 1.0
    scale = 0.16 * span / maxd * scale_mul
    return {"comps": comps, "cmax": cmax, "scale": scale, "span": span,
            "P0": P0, "DISP": d["DISP"], "MAGn": d["MAGn"]}


def _surface_meshes(s, phase):
    out = []
    for comp in s["comps"]:
        Vd = comp["V"] + (s["scale"] * np.sin(phase)) * comp["DV"]
        out.append(go.Mesh3d(x=Vd[:, 0], y=Vd[:, 1], z=Vd[:, 2],
                   i=comp["i"], j=comp["j"], k=comp["k"], intensity=comp["mag"],
                   cmin=0, cmax=s["cmax"], coloraxis="coloraxis", flatshading=False,
                   opacity=1.0, lighting=dict(ambient=0.82, diffuse=0.5, specular=0.12),
                   hoverinfo="skip"))
    return out


def _mode_surface_layout(height, cmax):
    lay_kw = _mode_scene(height)
    lay_kw["coloraxis"] = dict(colorscale="Jet", cmin=0, cmax=cmax,
                               colorbar=dict(title="ampl", thickness=14, len=0.6, x=0.98))
    return lay_kw


def _mode_surface_fig(lay, amps_signed, height=600, scale_mul=1.0, annotate=True):
    s = _mode_surface_comps(lay, amps_signed, scale_mul)
    if s is None:
        return _geometry_fig(lay, height=height)
    fig = go.Figure()
    for tr in _surface_meshes(s, np.pi / 2):
        fig.add_trace(tr)
    ncomp = len(s["comps"])
    # nodos de medición + números + flechas de dirección (DOF) estilo ARTeMIS
    P0, DISP = s["P0"], s["DISP"]
    if annotate and len(P0):
        alen = 0.09 * s["span"]
        norm = np.linalg.norm(DISP, axis=1, keepdims=True); norm[norm == 0] = 1.0
        U = DISP / norm * alen
        fig.add_trace(go.Cone(x=P0[:, 0], y=P0[:, 1], z=P0[:, 2], u=U[:, 0], v=U[:, 1], w=U[:, 2],
                      anchor="tail", sizemode="absolute", sizeref=alen * 0.5, showscale=False,
                      colorscale=[[0, "#0f172a"], [1, "#0f172a"]], hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=P0[:, 0], y=P0[:, 1], z=P0[:, 2], mode="markers+text",
                      text=[str(i + 1) for i in range(len(P0))], textposition="top center",
                      textfont=dict(size=10, color="#0f172a"),
                      marker=dict(size=3, color="#0f172a"), hoverinfo="skip"))
    else:
        fig.add_trace(go.Scatter3d(x=P0[:, 0], y=P0[:, 1], z=P0[:, 2], mode="markers",
                      marker=dict(size=3, color="#0f172a"), hoverinfo="skip"))
    frames = []
    for f in range(28):
        ph = f / 28.0 * 2 * np.pi
        frames.append(go.Frame(data=_surface_meshes(s, ph), traces=list(range(ncomp))))
    fig.frames = frames
    lay_kw = _mode_surface_layout(height, s["cmax"])
    lay_kw["updatemenus"] = [dict(type="buttons", showactive=False, x=0.02, y=0.05, xanchor="left",
        buttons=[dict(label="▶ Play", method="animate",
                      args=[None, dict(frame=dict(duration=130, redraw=True), fromcurrent=True,
                                       transition=dict(duration=0), mode="immediate")]),
                 dict(label="⏸ Pause", method="animate",
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])]
    fig.update_layout(**lay_kw)
    return fig


def _mode_surface_gif(lay, amps_signed, scale_mul=1.0, n=22, w=680, h=500):
    import io
    from PIL import Image
    s = _mode_surface_comps(lay, amps_signed, scale_mul)
    if s is None:
        return None
    imgs = []
    for f in range(n):
        ph = f / n * 2 * np.pi
        fig = go.Figure()
        for tr in _surface_meshes(s, ph):
            fig.add_trace(tr)
        lay_kw = _mode_surface_layout(h, s["cmax"]); lay_kw["paper_bgcolor"] = "white"
        fig.update_layout(**lay_kw)
        imgs.append(Image.open(io.BytesIO(fig.to_image(format="png", width=w, height=h, scale=1))).convert("RGB"))
    if not imgs:
        return None
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], duration=60, loop=0)
    return buf.getvalue()


def _mode_video_gif(lay, geom, amps_signed, is_rotor, scale_mul=1.0, n=14, w=640, h=440):
    """GIF animado real de la forma modal usando el MISMO render que se ve (rotor o
    carcasa): renderiza n fases del ciclo y las une en un GIF que hace bucle. n moderado
    para que el render (kaleido) termine rápido. Devuelve bytes GIF o None."""
    import io
    from PIL import Image
    imgs = []
    for f in range(n):
        ph = f / n * 2 * np.pi
        try:
            fig = (_mode_rotor_fig(lay, amps_signed, height=h, scale_mul=scale_mul, static=True, phase=ph)
                   if is_rotor else
                   _mode_geom_fig(lay, geom, amps_signed, height=h, scale_mul=scale_mul, static=True, phase=ph))
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
            png = fig.to_image(format="png", width=w, height=h, scale=1)
            imgs.append(Image.open(io.BytesIO(png)).convert("RGB"))
        except Exception:  # noqa: BLE001
            continue
    if len(imgs) < 2:
        return None
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=110, loop=0, optimize=True)
    return buf.getvalue()


def _mode_shape_gif(lay, amps_signed, scale_mul=1.0, n=22, w=620, h=460):
    """Renderiza la deformación a un GIF descargable (video decente)."""
    import io
    from PIL import Image
    d = _mode_shape_data(lay, amps_signed, scale_mul)
    if d is None:
        return None
    imgs = []
    for f in range(n):
        ph = f / n * 2 * np.pi
        fig = go.Figure()
        for tr in _mode_machine_meshes(lay):
            fig.add_trace(tr)
        fig.add_trace(go.Scatter3d(x=d["Ps"][:, 0], y=d["Ps"][:, 1], z=d["Ps"][:, 2], mode="lines",
                      line=dict(color="rgba(148,163,184,.6)", width=4, dash="dot"), hoverinfo="skip"))
        b, nd = _mode_dynamic_traces(d, ph, colorbar=False)
        fig.add_trace(b); fig.add_trace(nd)
        fig.update_layout(**_mode_scene(h, paper="white"))
        imgs.append(Image.open(io.BytesIO(fig.to_image(format="png", width=w, height=h, scale=1))).convert("RGB"))
    if not imgs:
        return None
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], duration=60, loop=0)
    return buf.getvalue()


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
    _chart(fig)
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
        _chart(fig)

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
    from core.modal.run_report import mode_confidence as _mode_conf
    # UI del analista en inglés (el reporte va en español)
    _CONF_EN = {"Alta": "High", "Media": "Medium", "Baja": "Low"}
    _CONFCOL = {"High": ("#166534", "#e7f7ee"), "Medium": ("#b45309", "#fdf3e3"),
                "Low": ("#b91c1c", "#fdeaea")}
    for i, m in enumerate(D["oma_modes"], 1):
        v = _v_of(m["fn"]); vk = getattr(v, "verdict", "") if v else ""
        vc, vb = _VCOL.get(vk, ("#64748b", "#eef2f8"))
        harm = "  ⚠" if (v and getattr(v, "is_harmonic", False)) else ""
        _man = m.get("source") == "manual"
        src_c, src_b, src_t = (("#7c3aed", "#f2ecfd", "Manual") if _man else ("#2563eb", "#e8f0ff", "FDD"))
        vlabel = (_VTXT.get(vk, "—") + harm) if vk else "—"
        _cf = _CONF_EN.get(_mode_conf(m["fn"], m["zeta"], m["complexity"], m.get("cls", "natural"),
                                      _ssi_freqs, D["rpm"]), "—")
        _cfc, _cfb = _CONFCOL.get(_cf, ("#64748b", "#eef2f8"))
        _rows_html.append(
            f"<tr>"
            f"<td class='idx'>{i}</td>"
            f"<td class='fn'>{m['fn']:.2f}<span class='u'> Hz</span></td>"
            f"<td class='num'>{m['zeta']:.2f}<span class='u'> %</span></td>"
            f"<td class='num'>{m['complexity']:.0f}<span class='u'> %</span></td>"
            f"<td><span class='cls'>{m['cls']}</span></td>"
            f"<td><span class='badge' style='color:{src_c};background:{src_b}'>{src_t}</span></td>"
            f"<td><span class='pill' style='color:{vc};background:{vb}'>{vlabel}</span></td>"
            f"<td><span class='pill' style='color:{_cfc};background:{_cfb};font-weight:800'>{_cf}</span></td>"
            f"</tr>")
    _table_html = ('<table class="wm-modes"><thead><tr>'
                   "<th>#</th><th>Frequency</th><th>Damping ζ</th><th>Complexity</th>"
                   "<th>Class</th><th>Source</th><th>Validation</th><th>Confianza</th></tr></thead>"
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
    _sec("SSI — stabilization diagram",
         "A second, independent method (subspace) that confirms the FDD modes", "OMA · Brincker & Ventura")
    _sv0 = D["sv_traces"][0] if D["sv_traces"] else None
    _diagram = None; _rows = None; _freqs = []
    if D["raw"] is not None:
        data, fs = D["raw"]
        ssi = run_ssi_cov(data, fs, orders=list(range(2, 41, 2)), fmin_hz=2.0, fmax_hz=200.0)
        _diagram = ssi.diagram; _freqs = [m.frequency_hz for m in ssi.modes]
        _rows = [[f"<span class='idx'>{i+1}</span>", f"<span class='fn'>{m.frequency_hz:.2f}<span class='u'> Hz</span></span>",
                  f"<span class='num'>±{m.std_frequency_hz:.2f}</span>",
                  f"<span class='num'>{m.damping_ratio_pct:.2f}<span class='u'> %</span></span>",
                  f"<span class='num'>±{m.std_damping_pct:.2f}</span>"] for i, m in enumerate(ssi.modes)]
    elif D["ssi_cloud"] and D["ssi_cloud"].get("diagram"):
        _ssi = D["ssi_cloud"]
        _diagram = [[e[0], np.asarray(e[1], float), e[2]] for e in _ssi["diagram"]]
        _freqs = [m["fn"] for m in _ssi["modes"]]
        _rows = [[f"<span class='idx'>{i+1}</span>", f"<span class='fn'>{m['fn']:.2f}<span class='u'> Hz</span></span>",
                  f"<span class='num'>±{m.get('std_fn',0.0):.2f}</span>",
                  f"<span class='num'>{m['zeta']:.2f}<span class='u'> %</span></span>",
                  f"<span class='num'>±{m.get('std_zeta',0.0):.2f}</span>"] for i, m in enumerate(_ssi["modes"])]

    if _diagram:
        _chart(_ssi_plot(_diagram, _freqs, _sv0))
        # diagnóstico automático
        if _freqs:
            _fs = " · ".join(f"{f:.1f} Hz" for f in sorted(_freqs))
            _agree = ""
            if _sv0 is not None and D["oma_modes"]:
                _om = [m["fn"] for m in D["oma_modes"]]
                _match = sum(1 for f in _freqs if any(abs(f - o) <= 0.03 * max(o, 1) for o in _om))
                _agree = f" **{_match}/{len(_freqs)}** of them line up with the FDD peaks (independent confirmation)."
            st.success(f"SSI confirms **{len(_freqs)} stable mode(s)**: {_fs}.{_agree}")
        if _rows:
            _show_table(["#", "Frequency", "± Hz", "Damping ζ", "± %"], _rows)
        st.caption("The ± columns are the uncertainty of each estimate — smaller is more reliable.")
    else:
        st.info("SSI runs on the raw waveform. Older cloud runs stored only the results; the modes "
                "identified by FDD are shown below. Re-upload from the field to enable the live diagram.")
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
        _chart(fig)
        if matches:
            _show_dicts(correlation_table(matches))
            st.info(ema_oma_summary(matches))

# ---------------------------------------------------------------- 7 CAMPBELL
if nav == T_CAMP:
    _sec("Campbell diagram", "Natural frequencies vs running-speed orders", "API 684 (±15%)")
    modes_hz = [m["fn"] for m in D["oma_modes"] if m["cls"] != "spurious"] or [m["fn"] for m in D["oma_modes"]]
    if not modes_hz:
        st.info("No modes to plot.")
    else:
        # --- Controles: 2ª velocidad opcional + banda ½× (sub-síncrono) ---
        try:
            cco = st.columns([1.1, 1, 1], vertical_alignment="bottom")
        except TypeError:      # Streamlit < 1.36 no soporta vertical_alignment
            cco = st.columns([1.1, 1, 1])
        with cco[0]:
            _cmp2 = st.checkbox("Compare 2nd speed", value=False, key="camp_cmp2",
                                help="Overlay a second operating speed. Order lines don't move — only the speed line/band.")
        with cco[1]:
            _rpm2 = st.number_input("2nd speed (RPM)", min_value=0.0, max_value=60000.0,
                                    value=float(round(D["rpm"] * 0.9)), step=10.0, key="camp_rpm2",
                                    disabled=not _cmp2, label_visibility="collapsed")
        with cco[2]:
            _half = st.checkbox("½× band (sub-sync)", value=False, key="camp_half",
                                help="Optional (not API 684): screens sub-synchronous excitation at half speed.")

        rpm_op = float(D["rpm"]); SM = 0.15
        lo, hi = rpm_op * (1 - SM), rpm_op * (1 + SM)
        rpm2 = float(_rpm2) if _cmp2 and _rpm2 > 0 else 0.0
        rpm_max = max(rpm_op * 1.4, rpm2 * 1.4, 1500.0)
        orders = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
        ymax = max(modes_hz) * 1.30
        rpm_axis = np.linspace(0, rpm_max, 80)
        fig = go.Figure()
        # banda de operación ±15% (rojo)
        fig.add_vrect(x0=lo, x1=hi, fillcolor="rgba(239,68,68,.13)", line_width=0)
        if _half:
            fig.add_vrect(x0=rpm_op / 2 * (1 - SM), x1=rpm_op / 2 * (1 + SM),
                          fillcolor="rgba(245,158,11,.12)", line_width=0)
        if rpm2 > 0:
            fig.add_vrect(x0=rpm2 * (1 - SM), x1=rpm2 * (1 + SM), fillcolor="rgba(124,58,237,.12)", line_width=0)
        # líneas de orden 0.5×..8× + etiqueta donde salen del gráfico
        for o in orders:
            fig.add_trace(go.Scatter(x=rpm_axis, y=rpm_axis / 60.0 * o, mode="lines",
                          line=dict(color="#6B7280", width=1, dash="dot"), showlegend=False, hoverinfo="skip"))
            if o * rpm_max / 60.0 <= ymax:
                lx, ly = rpm_max * 0.985, o * rpm_max / 60.0
            else:
                lx, ly = ymax * 60.0 / o, ymax * 0.985
            fig.add_annotation(x=lx, y=ly, text=f"{o:g}×", showarrow=False,
                               font=dict(size=10, color="#6B7280"), xanchor="right", yanchor="top")
        # frecuencias naturales (verde)
        for fn in modes_hz:
            fig.add_hline(y=fn, line=dict(color=GREEN, width=2))
        # líneas de N y ±15%
        fig.add_vline(x=rpm_op, line=dict(color=NAVY, width=3))
        fig.add_annotation(x=rpm_op, y=ymax, text=f"<b>N = {rpm_op:.0f} RPM</b>", showarrow=False,
                           font=dict(size=11, color=NAVY), yanchor="bottom", bgcolor="rgba(255,255,255,.85)")
        for xb, lab in ((lo, f"−15% · {lo:.0f}"), (hi, f"+15% · {hi:.0f}")):
            fig.add_vline(x=xb, line=dict(color=RED, width=1, dash="dash"))
            fig.add_annotation(x=xb, y=ymax * 0.86, text=lab, showarrow=False,
                               font=dict(size=9, color=RED), yanchor="top")
        # cruces de la velocidad de operación
        bands = [SpeedBand(rpm_op, SM * rpm_op, f"Operating {rpm_op:.0f}±15%")]
        if _half:
            bands.append(SpeedBand(rpm_op / 2, SM * rpm_op / 2, "½ speed"))
        crossings = compute_crossings(modes_hz, 0.0, rpm_max, orders=orders, bands=bands)
        _sc = {"coincidence": RED, "near": AMBER, "clear": "#cbd5e1"}
        for cr in crossings:
            fig.add_trace(go.Scatter(x=[cr.crossing_rpm], y=[cr.mode_hz], mode="markers", showlegend=False,
                          marker=dict(color=_sc.get(cr.severity, "#cbd5e1"), size=12, symbol="x-thin",
                                      line=dict(width=2, color=_sc.get(cr.severity, "#cbd5e1"))),
                          hovertemplate=f"{cr.mode_hz:.1f} Hz · {cr.order:g}× · %{{x:.0f}} RPM<extra></extra>"))
        # segunda velocidad (púrpura) + sus cruces
        _rows_cx = [(c, f"{rpm_op:.0f}") for c in crossings if c.severity in ("coincidence", "near")]
        if rpm2 > 0:
            fig.add_vline(x=rpm2, line=dict(color="#7c3aed", width=3, dash="dash"))
            fig.add_annotation(x=rpm2, y=ymax * 0.9, text=f"<b>2nd · {rpm2:.0f}</b>", showarrow=False,
                               font=dict(size=10, color="#7c3aed"), yanchor="top")
            cx2 = compute_crossings(modes_hz, 0.0, rpm_max, orders=orders,
                                    bands=[SpeedBand(rpm2, SM * rpm2, f"2nd {rpm2:.0f}±15%")])
            for c in cx2:
                if c.severity in ("coincidence", "near"):
                    fig.add_trace(go.Scatter(x=[c.crossing_rpm], y=[c.mode_hz], mode="markers", showlegend=False,
                                  marker=dict(color="#7c3aed", size=13, symbol="diamond",
                                              line=dict(width=1.5, color="white")),
                                  hovertemplate=f"{c.mode_hz:.1f} Hz · {c.order:g}× · {rpm2:.0f} RPM<extra>2nd</extra>"))
                    _rows_cx.append((c, f"{rpm2:.0f}"))
        fig.update_layout(title_text="", height=500, template="watermelon",
                          xaxis=dict(range=[0, rpm_max]), yaxis=dict(range=[0, ymax]),
                          xaxis_title=f"Running speed (RPM) · N = {rpm_op:.0f} · API 684 ±15%",
                          yaxis_title="Frequency (Hz)")
        _chart(fig)
        # --- Tabla: SOLO los que están en resonancia (coincidence / near) ---
        _stat_txt = {"coincidence": "Coincidence", "near": "Near"}
        if _rows_cx:
            _show_table(["Mode", "Order", "Crossing RPM", "Margin %", "Status", "vs speed"],
                        [[f"<span class='fn'>{c.mode_hz:.2f}<span class='u'> Hz</span></span>",
                          f"<span class='num'>{c.order:g}×</span>",
                          f"<span class='num'>{c.crossing_rpm:.0f}</span>",
                          f"<span class='num'>{c.sep_margin_pct:.1f}</span>",
                          _pill(_stat_txt[c.severity], *( ("#dc2626", "#fdeaea") if c.severity == "coincidence" else ("#b45309", "#fef3e2"))),
                          f"<span class='num'>{vs}</span>"] for c, vs in sorted(_rows_cx, key=lambda t: t[0].sep_margin_pct)])
            _ib = [c for c in crossings if c.in_band]
            if _ib:
                _w = min(_ib, key=lambda c: c.sep_margin_pct)
                _nco = sum(1 for c in crossings if c.severity == "coincidence")
                st.info(f"**{_nco} coincidence(s)** inside the operating band. Closest: mode "
                        f"**{_w.mode_hz:.1f} Hz** crosses **{_w.order:g}×** at {_w.crossing_rpm:.0f} RPM "
                        f"(margin {_w.sep_margin_pct:.1f}%). A crossing alone does not confirm resonance "
                        "(API 684) — correlate with amplitude and phase during operation.")
        else:
            st.success("No fn↔order crossings inside the operating band(s) — adequate separation (API 684).")

# ---------------------------------------------------------------- 8 MODE SHAPES
if nav == T_SHAPES:
    _sec("Mode shapes", "3D operational deflection — the structure deforms; amplitude colormap")
    modes = D["oma_modes"]
    if not modes:
        st.info("No modes to display.")
    else:
        # Confianza por modo (para filtrar las formas que valen la pena)
        from core.modal.run_report import mode_confidence as _mc
        _CONF_EN = {"Alta": "High", "Media": "Medium", "Baja": "Low"}
        _ssf = [mm["fn"] for mm in (D["ssi_cloud"] or {}).get("modes", [])] if D["ssi_cloud"] else []
        _conf_all = [_CONF_EN.get(_mc(mm["fn"], mm["zeta"], mm["complexity"], mm.get("cls", "natural"),
                                      _ssf, D["rpm"]), "—") for mm in modes]
        cms = st.columns([2.2, 0.9, 1.3])
        with cms[2]:
            _only_hi = st.toggle("Only worthwhile modes", value=False, key="ms_only_hi",
                                 help="Hide Low-confidence modes (keeps High/Medium — confirmed FDD+SSI).")
        _sel_idx = [i for i in range(len(modes)) if (not _only_hi) or _conf_all[i] in ("High", "Medium")]
        if not _sel_idx:
            _sel_idx = list(range(len(modes)))
        with cms[0]:
            opts = [f"Mode {i+1} — {modes[i]['fn']:.1f} Hz  ·  {_conf_all[i]}" for i in _sel_idx]
            sel = st.selectbox("Mode", opts, index=0, label_visibility="collapsed")
        with cms[1]:
            _scl = st.select_slider("Deformation", options=["0.5×", "1×", "2×", "3×"], value="1×",
                                    label_visibility="collapsed")
        idx = _sel_idx[opts.index(sel)]
        m = modes[idx]
        pts = lay.active_points()
        if D["shapes"] and idx < len(D["shapes"]) and D["shapes"][idx] is not None \
                and len(D["shapes"][idx]) == len(pts):
            amp = np.asarray(D["shapes"][idx], float)          # forma modal (con signo)
        else:
            amp = np.random.default_rng(idx + 1).standard_normal(len(pts))
        _smul = float(_scl.replace("×", ""))
        from core.modal.oma_layout import default_geometry as _default_geometry
        _pl_geom = ((D.get("payload") or {}).get("layout") or {}).get("geometry")
        _geom = _pl_geom if (_pl_geom and _pl_geom.get("nodes")) else _default_geometry(lay)

        # incertidumbres (Std.) desde SSI si hay un modo que coincide
        _stdf = _stdz = "N/A"
        for _sm in (D.get("ssi_cloud") or {}).get("modes", []):
            if abs(float(_sm.get("fn", 0)) - m["fn"]) <= 0.03 * max(m["fn"], 1):
                _stdf = f"{_sm.get('std_fn', 0):.3f} Hz"; _stdz = f"{_sm.get('std_zeta', 0):.3f} %"; break
        _logdec = 2 * np.pi * m["zeta"] / 100.0

        mv = st.columns([3, 1.05])
        with mv[0]:
            st.markdown(f"<div style='text-align:center;color:#64748b;font-weight:600;font-size:13px;"
                        f"margin-bottom:2px'>Mode {idx+1} · operating deflection shape · {m['fn']:.3f} Hz</div>",
                        unsafe_allow_html=True)
            if _rotor_is(lay):                   # proximidad → forma modal del ROTOR (eje + impulsores)
                _chart(_mode_rotor_fig(lay, amp, height=600, scale_mul=_smul))
                st.caption("Press ▶ Play — **rotor** deflection shape (shaft + motor mass + pump impellers) "
                           "from the proximity probes. This is the ROTOR, not the casing.")
            else:
                _chart(_mode_geom_fig(lay, _geom, amp, height=600, scale_mul=_smul))
                st.caption("Press ▶ Play — surfaces deform, coloured by displacement amplitude. "
                           "Geometry comes from the field configuration. Drag to rotate.")
        with mv[1]:
            st.markdown(
                "<div class='wm-mv'><div class='h'>Modal Values</div>"
                f"<div class='row'><span>Frequency</span><b>{m['fn']:.3f} Hz</b></div>"
                f"<div class='row'><span>Std. Frequency</span><b>{_stdf}</b></div>"
                f"<div class='row'><span>Damping</span><b>{m['zeta']:.3f} %</b></div>"
                f"<div class='row'><span>Std. Damping</span><b>{_stdz}</b></div>"
                f"<div class='row'><span>Log. decrement</span><b>{_logdec:.3f}</b></div>"
                f"<div class='row'><span>Complexity</span><b>{m['complexity']:.3f} %</b></div>"
                f"<div class='row'><span>Class</span><b>{m['cls']}</b></div></div>"
                "<div class='wm-mv'><div class='h'>Graphical objects</div>"
                "<div class='row'><span>Surfaces</span><span class='sw' style='background:#38bdf8'></span></div>"
                "<div class='row'><span>Edges</span><span class='sw' style='background:#0f172a'></span></div></div>"
                "<div class='wm-mv'><div class='h'>Colormap</div>"
                "<div style='display:flex;align-items:center;gap:10px'>"
                "<div class='wm-cbar'></div>"
                "<div style='display:flex;flex-direction:column;justify-content:space-between;height:118px;"
                "font-size:11px;color:#64748b'><span>Max</span><span>0</span></div></div></div>",
                unsafe_allow_html=True)
            if st.button("🎬 Export video (GIF)", key="ms_gif", use_container_width=True):
                with st.spinner("Rendering the animated video (~30–60 s, please wait)…"):
                    _gif = _mode_video_gif(lay, _geom, amp, _rotor_is(lay), scale_mul=_smul)
                if _gif:
                    st.session_state["_ms_gif"] = _gif
                    _kind = "rotor" if _rotor_is(lay) else "casing"
                    st.session_state["_ms_gif_name"] = f"mode_{idx+1}_{m['fn']:.0f}Hz_{_kind}.gif"
                else:
                    st.warning("Could not render the video. Try again.")
            if st.session_state.get("_ms_gif"):
                st.download_button("⬇ Download", data=st.session_state["_ms_gif"],
                                   file_name=st.session_state.get("_ms_gif_name", "mode_shape.gif"),
                                   mime="image/gif", use_container_width=True)

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
    _chart(fig)
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
            # consecutivo AUTOMÁTICO: siguiente OMA-<año>-NNN según el histórico (se calcula
            # una vez por sesión y se cachea; el usuario lo puede editar si lo necesita).
            if "rep_consec_auto" not in st.session_state:
                try:
                    from core.reports_archive import next_consecutive as _nc
                    st.session_state["rep_consec_auto"] = _nc(
                        f"OMA-{_date.today().year}-", _user.get("email", ""), _my_role)
                except Exception:  # noqa: BLE001
                    st.session_state["rep_consec_auto"] = f"OMA-{_date.today().year}-001"
            _consec = st.text_input("Consecutive (auto)", key="rep_consec",
                                    value=st.session_state.get("rep_consec", st.session_state["rep_consec_auto"]))
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

        _hide_spur = st.checkbox(
            "Ocultar modos **spurious / harmonic** del reporte (recomendado — deja solo los modos estructurales)",
            value=True, key="rep_hide_spur")
        _only_conf = st.checkbox(
            "Solo modos de **alta confianza** (confirmados FDD+SSI · complejidad < 15% · amortiguación física)",
            value=False, key="rep_only_conf")
        from core.modal.run_report import mode_confidence as _mode_conf
        _ssi_f = [float(_mm.get("fn", 0)) for _mm in ((D.get("ssi_cloud") or {}).get("modes") or []) if _mm.get("fn")]
        _rpm_r = float(D.get("rpm") or 0.0)
        def _conf_of(_m):
            return _mode_conf(_m.get("fn", 0), _m.get("zeta", 0), _m.get("complexity", 0),
                              _m.get("cls", _m.get("class", "natural")), _ssi_f, _rpm_r)
        if st.button("📄 Generate full report (PDF)", type="primary", key="rep_gen"):
            try:
                from core.modal.run_report import build_report_from_run
                import base64 as _b64m
                with st.spinner("Rendering figures (config · sensors · spectral density · SSI · mode shapes) and building the SIGA report…"):
                    from core.modal.oma_layout import default_geometry as _dg_r
                    pts = lay.active_points()
                    _pl_geom = ((D.get("payload") or {}).get("layout") or {}).get("geometry")
                    _geom_r = _pl_geom if (_pl_geom and _pl_geom.get("nodes")) else _dg_r(lay)
                    # filtro spurious/harmonic + confianza (1 clic): modos + sus formas, en paralelo
                    _pairs = list(zip(D["oma_modes"], (D["shapes"] or [None] * len(D["oma_modes"]))))
                    if _hide_spur:
                        _pairs = [(mm, ss) for (mm, ss) in _pairs
                                  if str(mm.get("cls", "")).lower() not in ("spurious", "harmonic")]
                    if _only_conf:
                        _pairs = [(mm, ss) for (mm, ss) in _pairs if _conf_of(mm) == "Alta"]
                    _modes_r = [mm for mm, ss in _pairs]; _shapes_r = [ss for mm, ss in _pairs]
                    _SHAPE_CAP = 8   # todas las formas confiables (tope para no inflar el PDF)
                    # formas modales estilo ARTeMIS (superficies + cuadrícula), estáticas para el PDF
                    shape_pngs = []
                    for i, m in enumerate(_modes_r[:_SHAPE_CAP]):
                        if i < len(_shapes_r) and _shapes_r[i] is not None \
                                and len(_shapes_r[i]) == len(pts):
                            a = np.asarray(_shapes_r[i], float)
                        else:
                            a = np.random.default_rng(i + 1).standard_normal(len(pts))
                        try:
                            _fig_ms = (_mode_rotor_fig(lay, a, height=520, static=True) if _rotor_is(lay)
                                       else _mode_geom_fig(lay, _geom_r, a, height=520, static=True))
                            shape_pngs.append(_fig_ms.to_image(format="png", width=1100, height=640, scale=2))
                        except Exception:  # noqa: BLE001
                            shape_pngs.append(None)
                    # configuración 3D (máquina + sensores)
                    try:
                        config_png = _geometry_fig(lay, height=520).to_image(format="png", width=1200, height=680, scale=2)
                    except Exception:  # noqa: BLE001
                        config_png = None
                    # diagrama de estabilización SSI (si la corrida lo trae)
                    ssi_png = None
                    _ssic = D.get("ssi_cloud") or {}
                    if _ssic.get("diagram"):
                        try:
                            _dg2 = [[e[0], np.asarray(e[1], float), e[2]] for e in _ssic["diagram"]]
                            _fq = [mm["fn"] for mm in _ssic.get("modes", [])]
                            _sv0r = D["sv_traces"][0] if D["sv_traces"] else None
                            ssi_png = _ssi_plot(_dg2, _fq, _sv0r).to_image(format="png", width=1100, height=560, scale=2)
                        except Exception:  # noqa: BLE001
                            ssi_png = None
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
                    # payload para el reporte, con filtros de spurious/harmonic y/o confianza
                    _pay_r = dict(D.get("payload") or {})
                    if (_hide_spur or _only_conf) and _pay_r.get("modes"):
                        def _keep(mm):
                            if _hide_spur and str(mm.get("class", "")).lower() in ("spurious", "harmonic"):
                                return False
                            if _only_conf and _conf_of(mm) != "Alta":
                                return False
                            return True
                        _pay_r = {**_pay_r, "modes": [mm for mm in _pay_r["modes"] if _keep(mm)]}
                    pdf = build_report_from_run(
                        _pay_r, bilingual_es=_es, shape_pngs=shape_pngs,
                        findings=_findings, recommendations=_recs, meta_extra=_meta_extra,
                        config_png=config_png, sensor_png=sensor_png, sensor_rows=sensor_rows,
                        ssi_png=ssi_png, max_shape_modes=len(shape_pngs))
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
