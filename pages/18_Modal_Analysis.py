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

# ---- Config / constantes del análisis (UNA sola fuente; antes estaban regadas) ----
CFG_NPERSEG = 4096              # ventana de Welch para EFDD
CFG_SSI_BAND = (2.0, 200.0)    # banda de frecuencia para el SSI (Hz)
CFG_SSI_ORDERS = range(2, 41, 2)  # órdenes del modelo SSI
CFG_MAC_REDUNDANT = 0.7        # MAC off-diagonal > este valor → modos redundantes
CFG_CAMPBELL_MARGIN = 0.15     # ±15% banda de operación (API 684)
CFG_REPORT_SHAPE_CAP = 8       # máx. formas modales embebidas en el PDF
CFG_CONF_EN = {"Alta": "High", "Media": "Medium", "Baja": "Low"}  # mapa de confianza ES→EN

# ---- Modelo de datos de una corrida (el dict `D`). Antes no había esquema; esto lo tipa
#      y documenta sin cambiar el acceso por clave (D["..."]). Fuente única de la forma. ----
from typing import TypedDict, Any, Optional, List, Tuple  # noqa: E402


class RunModel(TypedDict, total=False):
    lay: Any                       # OMALayout (geometría + sensores + adquisición)
    name: str                      # nombre de la corrida
    source: str                    # "demo" (dataset de muestra) | "cloud" (corrida real)
    oma_modes: List[dict]          # modos OMA: {fn, zeta, complexity, cls, source}
    shapes: List[Any]              # vector de forma modal por modo (o None si no resuelto)
    sv_traces: List[Tuple[str, Any, Any]]   # curvas de valores singulares (EFDD)
    fdd: Any                       # FDDResult (SVD completo) cuando se analiza desde cruda
    ema_freqs: List[float]         # frecuencias EMA (impacto)
    ema_modes_full: List[dict]     # modos EMA con amortiguamiento/coherencia
    ema_curve: Optional[dict]      # FRF de impacto real {freqs, mag_db, coh}
    ssi_cloud: Optional[dict]      # resultado SSI (modos + diagrama de estabilización)
    rpm: float                     # velocidad de operación
    raw: Any                       # (data[N,ch], fs) para SSI en vivo
    payload: dict                  # payload crudo de la corrida (incluye layout.geometry)
    raw_ref: Optional[dict]        # referencia a la data cruda en Storage
    auto_n: int                    # nº de modos automáticos


def _field_geometry(D, lay):
    """Geometría AUTORITATIVA del campo: la del payload si trae nodos; si no, la default
    derivada del layout. Fuente única (antes el patrón ((payload)...geometry) estaba en 3 sitios)."""
    from core.modal.oma_layout import default_geometry as _dg
    g = ((D.get("payload") or {}).get("layout") or {}).get("geometry")
    return g if (g and g.get("nodes")) else _dg(lay)


def _run_geometry(D, lay):
    """Geometría que se ANIMA: la mejorada por el analista en el editor (si aplicó una),
    o la del campo. Los sensores (nodos con 'sensor') vienen del campo y no se alteran."""
    import streamlit as _st
    _ov = _st.session_state.get(f"geom_edit::{D.get('name', 'run')}")
    if _ov and _ov.get("nodes"):
        return _ov
    return _field_geometry(D, lay)


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
      /* Panel Modal Values */
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
      /* Editor de geometría: headers de sección + tablas más limpias */
      .geo-h { font-weight:700; color:#0F1E3D; font-size:13px; margin:16px 0 5px;
        display:flex; align-items:baseline; gap:9px; letter-spacing:.01em; }
      .geo-h span { font-weight:400; color:#94a3b8; font-size:11px; letter-spacing:0; }
      div[data-testid="stDataFrame"], div[data-testid="stDataEditor"]{
        border-radius:10px; border:1px solid #e6ecf5; overflow:hidden;
        box-shadow:0 1px 2px rgba(15,30,61,.04); }
      @media (prefers-color-scheme: dark){
        .geo-h { color:#eaf0f7; }
        div[data-testid="stDataFrame"], div[data-testid="stDataEditor"]{ border-color:#243040; }
        .wm-kpi{ background:#141b26; border-color:#243040; }
        .wm-kpi .v{ color:#eaf0f7; }
        .wm-kpi .k{ color:#8ea0bd; }
        /* Tablas de modos en oscuro */
        table.wm-modes{ background:#0f1622; box-shadow:0 1px 3px rgba(0,0,0,.4); }
        table.wm-modes td{ color:#dbe4f0; border-top-color:#1e2836; }
        table.wm-modes tr:nth-child(even) td{ background:#141d2b; }
        table.wm-modes tr:hover td{ background:#1b2740; }
        table.wm-modes td.idx{ color:#5f7290; }
        table.wm-modes .u{ color:#6b7d99; }
        table.wm-modes .cls{ color:#93a4bd; }
        /* Panel Modal Values en oscuro */
        .wm-mv{ background:#141b26; border-color:#243040; box-shadow:0 1px 2px rgba(0,0,0,.4); }
        .wm-mv .h{ color:#eaf0f7; border-bottom-color:#243040; }
        .wm-mv .row{ color:#93a4bd; }
        .wm-mv .row b{ color:#eaf0f7; }
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


def _mac_from_shapes(shapes):
    """MAC (Modal Assurance Criterion) desde una lista de vectores de forma modal.
    Fuente ÚNICA de verdad: la usan la vista MAC y el reporte PDF (antes se calculaba
    con dos dobles-loops inline duplicados). Salta formas None/vacías.
    Devuelve (M, kept_idx) donde kept_idx mapea filas de M → índice del modo original."""
    vs, kept = [], []
    for i, s in enumerate(shapes or []):
        if s is not None and len(np.ravel(s)):
            vs.append(np.asarray(s, float).ravel()); kept.append(i)
    n = len(vs); M = np.eye(n)
    for i in range(n):
        for j in range(n):
            num = abs(np.vdot(vs[i], vs[j])) ** 2
            den = (np.vdot(vs[i], vs[i]).real * np.vdot(vs[j], vs[j]).real) or 1e-30
            M[i, j] = num / den
    return M, kept


def _cross_mac_from_shapes(shapes_a, shapes_b):
    """Cross-MAC entre dos sets de formas (EMA↔OMA). Devuelve (M[na,nb], idx_a, idx_b).
    Diagonal alta = mismos modos físicos por métodos distintos (validación cruzada, API 684)."""
    va, ia = [], []
    for i, s in enumerate(shapes_a or []):
        if s is not None and len(np.ravel(s)):
            va.append(np.asarray(s, float).ravel()); ia.append(i)
    vb, ib = [], []
    for j, s in enumerate(shapes_b or []):
        if s is not None and len(np.ravel(s)):
            vb.append(np.asarray(s, float).ravel()); ib.append(j)
    M = np.zeros((len(va), len(vb)))
    for i in range(len(va)):
        for j in range(len(vb)):
            if va[i].size != vb[j].size:
                continue
            num = abs(np.vdot(va[i], vb[j])) ** 2
            den = (np.vdot(va[i], va[i]).real * np.vdot(vb[j], vb[j]).real) or 1e-30
            M[i, j] = num / den
    return M, ia, ib


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
# Motor de gráficos 3D / geometría (extraído del monolito, ver Tier 2 de la auditoría)
from core.modal.modal_web_plots import *  # noqa: F401,F403
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


def _build_demo_D() -> RunModel:
    lay = _default_layout(); nch = lay.n_channels()
    data, fs = _demo_oma(nch)
    fmax = min(fs / 2.56, lay.fmax_hz)
    fdd = run_oma(time_data=data, sample_rate_hz=fs, nperseg=CFG_NPERSEG,
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
    _nsv = min(sv.shape[0], 4)   # SV1..SVn (no solo SV1) para que el reporte grafique todas
    payload = {"name": lay.name, "kind": "OMA", "running_rpm": lay.running_speed_rpm,
               "client": lay.client, "asset": lay.machine_type, "location": lay.location,
               "channel_names": lay.channel_names(), "ema_modes": [fn for fn, _ in DEMO_MODES],
               "svd": {"freqs": _fb[::_st].tolist(), "sv1": _sv1[::_st].tolist(),
                       "sv": [sv[r][band][::_st].tolist() for r in range(_nsv)]},
               "modes": [{"fn": m.natural_frequency_hz, "zeta": m.damping_ratio_pct,
                          "complexity": m.complexity_pct, "class": m.classification,
                          "shape": {"re": [], "im": []}} for m in fdd.modes],
               "layout": lay.to_dict()}
    _dshapes = [np.asarray(getattr(m, "mode_shape", []), complex).real for m in fdd.modes]
    return {"lay": lay, "oma_modes": oma_modes, "sv_traces": sv_traces,
            "ema_freqs": [fn for fn, _ in DEMO_MODES], "rpm": lay.running_speed_rpm,
            "raw": (data, fs), "shapes": _dshapes, "source": "demo", "name": lay.name, "fdd": fdd,
            "ema_curve": None, "ema_modes_full": None, "ssi_cloud": None, "payload": payload}


def _build_cloud_D(payload: dict) -> RunModel:
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
_runs = []; _cloud_err = None
try:
    from core.modal.modal_cloud import list_runs, load_run
    _runs = list_runs()
except Exception as _e:  # noqa: BLE001  — no ocultar: distinguir "sin runs" de "no se pudo conectar"
    _cloud_err = f"{type(_e).__name__}: {_e}"

_opts = {f"☁ {r.get('name','run')} · {str(r.get('updated_at',''))[:16]}": r.get("id") for r in _runs}
_labels = ["🧪 Sample dataset (demo)"] + list(_opts.keys())
_sc1, _sc2 = st.columns([3, 1])
with _sc1:
    _choice = st.selectbox("Data source", _labels, index=(1 if _opts else 0),
                           help="Field captures uploaded to the cloud appear here automatically.")
with _sc2:
    if st.button("🔄 Refresh runs", use_container_width=True):
        st.rerun()
if _cloud_err:
    st.warning(f"Could not reach the cloud to list field runs ({_cloud_err}). "
               "Check your connection — showing only the sample dataset below, not real data.")

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

# --- Data cruda en la nube: la web hace TODO el análisis con la cruda (EFDD + SVD
#     completo + armónicos + SSI + ODS + MAC). Cacheado para no recomputar en cada clic. ---
@st.cache_data(show_spinner=False)
def _load_raw_cached(path, bucket, fs):
    from core.modal.modal_cloud import download_raw
    return download_raw({"path": path, "bucket": bucket, "fs": fs})


@st.cache_data(show_spinner=False)
def _analyze_from_raw(path, bucket, fs, detrend, band, decf, harm, run_hz, fmax_lay, channels):
    """Descarga la cruda y hace el análisis FDD/EFDD completo (con preproceso y remoción
    de armónicos opcional). Devuelve dict con el FDDResult (incluye SVD completo U) o None.
    Cacheado por parámetros → no recomputa en cada interacción de Streamlit."""
    from core.modal.modal_cloud import download_raw
    from core.modal.oma_engine import preprocess_signals, run_oma
    rr = download_raw({"path": path, "bucket": bucket, "fs": fs})
    if rr is None:
        return None
    data, rfs = rr
    adata, afs = preprocess_signals(data, rfs, detrend=detrend, band=band, decimate_factor=int(decf))
    fmax = min(afs / 2.56, fmax_lay)
    fdd = run_oma(time_data=adata, sample_rate_hz=afs, nperseg=CFG_NPERSEG,
                  channel_names=list(channels), f_min_hz=5.0, f_max_hz=fmax, running_speed_hz=run_hz)
    nharm = 0
    if harm and fdd.modes:
        from core.modal.oma_engine import (kurtosis_harmonic_indicator, reduce_harmonics_sv,
                                           detect_oma_modes)
        import dataclasses as _dc
        cand = [m.natural_frequency_hz for m in fdd.modes]
        if run_hz:
            k = 1
            while run_hz * k <= fmax:
                cand.append(run_hz * k); k += 1
        kf = kurtosis_harmonic_indicator(adata, afs, sorted({round(c, 2) for c in cand}))
        hf = [d["freq"] for d in kf if d["is_harmonic"]]
        if hf:
            cl = reduce_harmonics_sv(fdd.frequencies_hz, fdd.singular_values, hf)
            fdd = _dc.replace(fdd, singular_values=np.asarray(cl))
            fdd.modes = detect_oma_modes(fdd, f_min_hz=5.0, f_max_hz=fmax, running_speed_hz=run_hz)
            nharm = len(hf)
    return {"fdd": fdd, "fs": float(afs), "fmax": float(fmax), "nharm": int(nharm)}


D.setdefault("fdd", None)
_rref = D.get("raw_ref")
_has_raw = bool(_rref and D.get("source") == "cloud" and _rref.get("path"))
if _has_raw:
    _mb = (_rref.get("size_bytes", 0) or 0) / 1e6
    with st.expander(f"⚙ Analysis from raw data — EFDD ({_rref.get('n_ch','?')} ch · ~{_mb:.1f} MB)", expanded=True):
        _pc = st.columns([1, 1, 1.2, 1.2, 1.4])
        _detr = _pc[0].checkbox("Detrend", value=True, help="Removes per-channel linear drift (recommended).")
        _bp = _pc[1].checkbox("Band-pass", value=False, help="Zero-phase band-pass filter.")
        _blo = _pc[2].number_input("lo (Hz)", 0.5, 5000.0, 5.0, step=1.0, disabled=not _bp)
        _bhi = _pc[3].number_input("hi (Hz)", 1.0, 25000.0, 500.0, step=10.0, disabled=not _bp)
        _dec = _pc[4].selectbox("Decimate", ["×1", "×2", "×4"], index=0,
                                help="Lower fs → more resolution in the low band.")
        _harm = st.checkbox("Reduce harmonics (kurtosis)", value=False,
                            help="Detects machine harmonics by kurtosis and removes them from the spectrum.")
    _band = (float(_blo), float(_bhi)) if _bp else None
    _decf = {"×1": 1, "×2": 2, "×4": 4}[_dec]
    _run_hz = (float(D.get("rpm") or 0.0) / 60.0) or None
    with st.spinner("Downloading raw data and analyzing (EFDD)…"):
        try:
            _res = _analyze_from_raw(_rref.get("path"), _rref.get("bucket", "modal-raw"),
                                     _rref.get("fs"), bool(_detr), _band, _decf, bool(_harm),
                                     _run_hz, float(lay.fmax_hz), tuple(lay.channel_names()))
        except Exception as _e:  # noqa: BLE001
            _res = None; st.warning(f"Raw-data analysis failed: {type(_e).__name__}: {_e}")
    if _res is not None:
        _fdd = _res["fdd"]; D["fdd"] = _fdd
        # reemplaza modos y formas con lo recomputado por EFDD desde la cruda
        D["oma_modes"] = [{"fn": m.natural_frequency_hz, "zeta": m.damping_ratio_pct,
                           "complexity": m.complexity_pct, "cls": m.classification, "source": "auto"}
                          for m in _fdd.modes]
        D["shapes"] = [np.asarray(getattr(m, "mode_shape", []), complex).real for m in _fdd.modes]
        _fr = np.asarray(_fdd.frequencies_hz); _sv = np.asarray(_fdd.singular_values)
        if _sv.ndim == 1:
            _sv = _sv[None, :]
        _bd = _fr <= _res["fmax"]
        D["sv_traces"] = [(f"SV{r+1} (EFDD)", _fr[_bd], 10 * np.log10(np.maximum(_sv[r][_bd], 1e-30)))
                          for r in range(min(_sv.shape[0], 4))]
        # data para SSI en vivo (descarga cacheada)
        _rr0 = _load_raw_cached(_rref.get("path"), _rref.get("bucket", "modal-raw"), _rref.get("fs"))
        if _rr0 is not None:
            D["raw"] = _rr0
        _hmsg = f" · {_res['nharm']} harmonic(s) removed" if _res["nharm"] else ""
        st.caption(f"⚡ EFDD analysis from raw — {len(_fdd.modes)} modes{_hmsg} · full SVD · live SSI · ODS and MAC available.")
    else:
        st.warning("Could not download/analyze the raw data for this run.")
elif D.get("source") == "cloud":
    st.info("This run has no raw data (uploaded with an older field version). "
            "Stored results are shown. Re-capture and upload with the new version "
            "for the full analysis (EFDD, harmonics, ODS, MAC) on the web.")

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
T_ODS = "🔵  ODS (operating)"
T_SSI = "🟠  SSI (subspace)"
T_MAC = "🔷  MAC / validation"
T_CAMP = "🟤  Campbell"
T_EMA = "🟢  Impact test (EMA)"
T_MODES = "🟣  Modes (EMA)"
T_CMP = "🔴  Comparative"
T_TREND = "🔵  Trend / Compare"
T_GEOM = "🟢  Geometry"
T_REPORT = "⚪  Report"
_NAVOPTS = [T_OMA, T_SSI, T_MAC, T_CAMP, T_SHAPES, T_GEOM, T_ODS, T_CMP, T_EMA, T_MODES, T_TREND, T_REPORT]

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
    _is_demo = D["source"] != "cloud"          # dataset de muestra (demo explícito)
    if D["ema_curve"] is not None:             # FRF de impacto REAL de la corrida
        _fx = D["ema_curve"]["freqs"]; _mag = D["ema_curve"]["mag_db"]; _coh = D["ema_curve"]["coh"]
        _show_frf = True
    elif _is_demo:                             # solo en el dataset de muestra se dibuja la curva demo
        f, H, coh = _demo_frf(); _fx = f; _mag = 20 * np.log10(np.abs(H)); _coh = coh
        _show_frf = True
    else:
        _show_frf = False                      # corrida real sin EMA → NO se dibuja una curva demo
    if _show_frf:
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
    elif _is_demo:
        st.caption("Sample dataset — illustrative impact FRF (not a real measurement).")
    else:
        st.info("This cloud run is operational (OMA) — no impact (EMA) test was uploaded, "
                "so there is no FRF to display.")

# ---------------------------------------------------------------- 3 MODES EMA
if nav == T_MODES:
    _sec("Modes (EMA)", "Peak-picking + half-power damping + Nyquist", "ISO 7626-6")
    _is_demo = D["source"] != "cloud"          # dataset de muestra (demo explícito)
    cc1, cc2 = st.columns([2, 3])
    with cc1:
        if D["ema_modes_full"]:
            _show_table(["#", "Frequency", "Damping ζ", "Coherence"],
                        [[f"<span class='idx'>{i}</span>", f"<span class='fn'>{m['fn']:.2f}<span class='u'> Hz</span></span>",
                          f"<span class='num'>{m['zeta']:.2f}<span class='u'> %</span></span>",
                          f"<span class='num'>{round(m['coh'],3) if m.get('coh') is not None else '—'}</span>"]
                         for i, m in enumerate(D["ema_modes_full"], 1)])
        elif D["ema_freqs"]:
            _show_table(["Frequency", "Status"],
                        [[f"<span class='fn'>{fr:.2f}<span class='u'> Hz</span></span>", _pill("reliable", "#16a34a", "#eaf7ef")]
                         for fr in D["ema_freqs"]])
        elif _is_demo:
            _show_table(["Frequency", "Damping ζ"],
                        [[f"<span class='fn'>{fn:.2f}<span class='u'> Hz</span></span>",
                          f"<span class='num'>{round(z*100,2)}<span class='u'> %</span></span>"] for fn, z in DEMO_MODES])
        else:
            st.info("No EMA (impact) modes in this run — it is operational (OMA) only.")
    with cc2:
        if _is_demo:                           # Nyquist ilustrativo solo en el dataset de muestra
            f, H, coh = _demo_frf()
            fig = go.Figure(go.Scatter(x=H.real, y=H.imag, mode="lines", line=dict(color=NAVY)))
            fig.update_layout(title="Nyquist (mobility)", height=380, template="watermelon",
                              xaxis_title="Re", yaxis_title="Im")
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            _chart(fig)
            st.caption("Sample dataset — illustrative Nyquist (not a real measurement).")
        else:
            st.info("Nyquist requires the complex impact FRF, which this run does not carry. "
                    "The modal table on the left is from the real run.")

# ---------------------------------------------------------------- 4 OMA (análisis)
if nav == T_OMA:
    _sec("Spectral density (FDD)", "", "ISO 20816")
    if D.get("fdd") is not None:
        st.caption("✓ **EFDD active** — modes are identified by Enhanced FDD: shape refined by the "
                   "SDOF bell (MAC≥0.80) and damping by **logarithmic decrement** (not half-power). "
                   "The curve shown is the spectral density (singular values).")
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
    _CONF_EN = CFG_CONF_EN
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
                   "<th>Class</th><th>Source</th><th>Validation</th><th>Confidence</th></tr></thead>"
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
        ssi = run_ssi_cov(data, fs, orders=list(CFG_SSI_ORDERS),
                          fmin_hz=CFG_SSI_BAND[0], fmax_hz=CFG_SSI_BAND[1])
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

        rpm_op = float(D["rpm"]); SM = CFG_CAMPBELL_MARGIN
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
        _CONF_EN = CFG_CONF_EN
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
        # Vector de forma modal REAL o nada. NUNCA se inventa (ni ruido aleatorio):
        # mostrar una forma falsa a un cliente destruiría la credibilidad del reporte.
        _has_shape = bool(D["shapes"] and idx < len(D["shapes"]) and D["shapes"][idx] is not None
                          and len(D["shapes"][idx]) == len(pts))
        amp = np.asarray(D["shapes"][idx], float) if _has_shape else None
        _smul = float(_scl.replace("×", ""))
        _geom = _run_geometry(D, lay)

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
            if not _has_shape:
                st.info("**Mode shape not available for this mode.** The captured data did not "
                        "resolve a shape vector for this frequency (needs enough measurement DOFs / "
                        "roving). Nothing is shown rather than fabricating a shape. The modal values "
                        "on the right (frequency, damping) are real.")
            elif _rotor_is(lay):                 # proximidad → forma modal del ROTOR (eje + impulsores)
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
            if _has_shape and st.button("🎬 Export video (GIF)", key="ms_gif", use_container_width=True):
                with st.spinner("Rendering the animated video (~30–60 s, please wait)…"):
                    _gif = _mode_video_gif(lay, _geom, amp, _rotor_is(lay), scale_mul=_smul)
                if _gif:
                    st.session_state["_ms_gif"] = _gif
                    _kind = "rotor" if _rotor_is(lay) else "casing"
                    st.session_state["_ms_gif_name"] = f"mode_{idx+1}_{m['fn']:.0f}Hz_{_kind}.gif"
                else:
                    st.warning("Could not render the video. Try again.")
            if _has_shape and st.session_state.get("_ms_gif"):
                st.download_button("⬇ Download", data=st.session_state["_ms_gif"],
                                   file_name=st.session_state.get("_ms_gif_name", "mode_shape.gif"),
                                   mime="image/gif", use_container_width=True)

# ---------------------------------------------------------------- MAC / validation
if nav == T_GEOM:
    _sec("Geometry", "Connect the sensors into a body so the mode shape looks like your machine — "
         "the sensors come from the field and stay locked", "")
    import pandas as _pd
    from core.modal.modal_web_plots import _AX as _DOF_AX, _stn_key as _stn
    _wk = f"geomw::{_run_key}"
    _field_g = _field_geometry(D, lay)
    _saved = st.session_state.get(f"geom_edit::{_run_key}")
    _srcg = _saved if (_saved and _saved.get("nodes")) else _field_g
    _sensor_nodes = [dict(n) for n in _srcg.get("nodes", []) if n.get("sensor")]
    # Estado de trabajo por IDS (robusto al reordenamiento). Se inicializa una vez.
    if _wk not in st.session_state:
        _n0 = _srcg.get("nodes", [])
        _snodes0 = [{"id": n["id"], "x": float(n.get("x", 0)), "y": float(n.get("y", 0)),
                     "z": float(n.get("z", 0))} for n in _n0 if not n.get("sensor")]
        _lines0 = []
        for a, b in _srcg.get("lines", []):
            _ia = _n0[a]["id"] if isinstance(a, int) and a < len(_n0) else a
            _ib = _n0[b]["id"] if isinstance(b, int) and b < len(_n0) else b
            _lines0.append([_ia, _ib])
        _surfs0 = [[_n0[i]["id"] if isinstance(i, int) and i < len(_n0) else i for i in s]
                   for s in _srcg.get("surfaces", [])]
        st.session_state[_wk] = {"snodes": _snodes0, "lines": _lines0, "surfs": _surfs0}
    _wg = st.session_state[_wk]

    # DOF por sensor (dirección física medida) para las flechas
    _dofmap = {}
    for _p in lay.active_points():
        _kk = _stn(_p); _vv = np.array(_DOF_AX.get(_p.axis, (0, 0, 1)), float)
        _sg = -1.0 if str(getattr(_p, "dof", "")).startswith("-") else 1.0
        _dofmap.setdefault(_kk, []).append(_vv * _sg)

    _s_ids = [n["id"] for n in _sensor_nodes]
    _st_ids = [n["id"] for n in _wg["snodes"]]
    _all_ids = _s_ids + _st_ids
    _lab = {nid: ("📍 " + nid) for nid in _s_ids}
    _lab.update({nid: ("○ " + nid) for nid in _st_ids})
    _flab = lambda x: _lab.get(x, str(x))   # noqa: E731

    st.caption(f"**{len(_sensor_nodes)} sensors** (📍 green, locked) · **{len(_wg['lines'])} connections**. "
               "Sensor positions are physical — you only draw the body around them.")

    _cE, _cP = st.columns([1.0, 1.4])
    with _cE:
        st.markdown("<div class='geo-h'>Connect two points<span>build the body</span></div>",
                    unsafe_allow_html=True)
        _cc = st.columns([1, 1, 0.8])
        _p1 = _cc[0].selectbox("From", _all_ids, format_func=_flab, key=f"cx1::{_run_key}",
                               label_visibility="collapsed")
        _p2 = _cc[1].selectbox("To", _all_ids, index=min(1, len(_all_ids) - 1), format_func=_flab,
                               key=f"cx2::{_run_key}", label_visibility="collapsed")
        if _cc[2].button("🔗 Connect", use_container_width=True):
            if _p1 != _p2 and [_p1, _p2] not in _wg["lines"] and [_p2, _p1] not in _wg["lines"]:
                _wg["lines"].append([_p1, _p2]); st.session_state[_wk] = _wg; st.rerun()
        _cc2 = st.columns(2)
        if _cc2[0].button("🔗 Auto-connect sensors", use_container_width=True,
                          help="Link the sensors in order along the machine (one click)"):
            _order = sorted(_sensor_nodes, key=lambda n: float(n.get("x", 0)))
            for _u, _v in zip(_order, _order[1:]):
                if [_u["id"], _v["id"]] not in _wg["lines"] and [_v["id"], _u["id"]] not in _wg["lines"]:
                    _wg["lines"].append([_u["id"], _v["id"]])
            st.session_state[_wk] = _wg; st.rerun()
        if _cc2[1].button("🧹 Clear all lines", use_container_width=True):
            _wg["lines"] = []; st.session_state[_wk] = _wg; st.rerun()

        st.markdown(f"<div class='geo-h'>Connections<span>{len(_wg['lines'])} · ✕ to remove</span></div>",
                    unsafe_allow_html=True)
        if not _wg["lines"]:
            st.caption("No connections yet — try **Auto-connect sensors**, or pick two points above and Connect.")
        else:
            for _i, _pair in enumerate(list(_wg["lines"])[:40]):
                _a, _b = _pair
                _rc = st.columns([6, 1])
                _rc[0].markdown(f"<div style='padding:5px 0;color:#334155;font-size:13px'>"
                                f"{_flab(_a)} &nbsp;↔&nbsp; {_flab(_b)}</div>", unsafe_allow_html=True)
                if _rc[1].button("✕", key=f"rmln::{_run_key}::{_i}", help="Remove"):
                    del _wg["lines"][_i]; st.session_state[_wk] = _wg; st.rerun()

        with st.expander("⚙ Advanced — exact coordinates & surfaces"):
            _ncol = st.column_config
            st.markdown("<div class='geo-h'>Structure points<span>extra shape points (optional)</span></div>",
                        unsafe_allow_html=True)
            _sdf = _pd.DataFrame([{"Node": n["id"], "X": round(n["x"], 4), "Y": round(n["y"], 4),
                                   "Z": round(n["z"], 4)} for n in _wg["snodes"]]
                                 or [{"Node": "", "X": 0.0, "Y": 0.0, "Z": 0.0}])
            _sed = st.data_editor(_sdf, num_rows="dynamic", use_container_width=True, hide_index=True,
                                  key=f"nodes_ed::{_run_key}", column_config={
                                      "Node": _ncol.TextColumn("Point", width="small"),
                                      "X": _ncol.NumberColumn("X", format="%.3f", step=0.05, width="small"),
                                      "Y": _ncol.NumberColumn("Y", format="%.3f", step=0.05, width="small"),
                                      "Z": _ncol.NumberColumn("Z", format="%.3f", step=0.05, width="small")})
            _wg["snodes"] = [{"id": str(r.get("Node", "")).strip(), "x": float(r.get("X", 0) or 0),
                              "y": float(r.get("Y", 0) or 0), "z": float(r.get("Z", 0) or 0)}
                             for _, r in _sed.iterrows() if str(r.get("Node", "")).strip()]
            st.markdown("<div class='geo-h'>Surfaces<span>fill faces (3–4 points)</span></div>",
                        unsafe_allow_html=True)
            _surdf = _pd.DataFrame([{"P1": (s + ["", "", "", ""])[0], "P2": (s + ["", "", "", ""])[1],
                                     "P3": (s + ["", "", "", ""])[2], "P4": (s + ["", "", "", ""])[3]}
                                    for s in _wg["surfs"]] or [{"P1": "", "P2": "", "P3": "", "P4": ""}])
            _sured = st.data_editor(_surdf, num_rows="dynamic", use_container_width=True, hide_index=True,
                                    key=f"surf_ed::{_run_key}", column_config={
                                        c: _ncol.SelectboxColumn(c, options=[""] + _all_ids, width="small")
                                        for c in ("P1", "P2", "P3", "P4")})
            _wg["surfs"] = [[v for v in (str(r.get(c, "")).strip() for c in ("P1", "P2", "P3", "P4")) if v]
                            for _, r in _sured.iterrows()
                            if len([v for v in (str(r.get(c, "")).strip() for c in ("P1", "P2", "P3", "P4")) if v]) >= 3]
            st.session_state[_wk] = _wg

        _bc = st.columns(3)
        _apply = _bc[0].button("✓ Apply to shapes", use_container_width=True, type="primary")
        _cloud = _bc[1].button("☁ Save to cloud", use_container_width=True)
        _reset = _bc[2].button("↺ Reset to field", use_container_width=True)

    # --- Geometría final (sensores del campo + estructura), ids → índices ---
    _new_nodes = [dict(n) for n in _sensor_nodes]
    for n in _wg["snodes"]:
        _new_nodes.append({"id": n["id"], "x": n["x"], "y": n["y"], "z": n["z"], "sensor": ""})
    _idmap = {n["id"]: k for k, n in enumerate(_new_nodes)}
    _new_lines = []
    for _a, _b in _wg["lines"]:
        if _a in _idmap and _b in _idmap and _a != _b:
            _pr = [_idmap[_a], _idmap[_b]]
            if _pr not in _new_lines and _pr[::-1] not in _new_lines:
                _new_lines.append(_pr)
    _new_surf = []
    for s in _wg["surfs"]:
        _ids = [v for v in s if v in _idmap]
        if len(_ids) >= 3:
            _face = [_idmap[v] for v in _ids[:4]]
            if _face not in _new_surf:
                _new_surf.append(_face)
    _gw2 = {"nodes": _new_nodes, "lines": _new_lines, "surfaces": _new_surf}

    with _cP:
        _fig = go.Figure()
        _coords = np.array([[n["x"], n["y"], n["z"]] for n in _new_nodes], float) if _new_nodes else np.zeros((1, 3))
        _span = float(np.ptp(_coords, axis=0).max()) or 1.0
        _L = 0.11 * _span
        if _new_surf:
            _xs = [n["x"] for n in _new_nodes]; _ys = [n["y"] for n in _new_nodes]; _zs = [n["z"] for n in _new_nodes]
            _si = []; _sj = []; _sk = []
            for s in _new_surf:
                if len(s) >= 3:
                    _si.append(s[0]); _sj.append(s[1]); _sk.append(s[2])
                if len(s) == 4:
                    _si.append(s[0]); _sj.append(s[2]); _sk.append(s[3])
            if _si:
                _fig.add_trace(go.Mesh3d(x=_xs, y=_ys, z=_zs, i=_si, j=_sj, k=_sk,
                                         color="#93c5fd", opacity=0.14, hoverinfo="skip"))
        for a, b in _new_lines:
            _fig.add_trace(go.Scatter3d(x=[_new_nodes[a]["x"], _new_nodes[b]["x"]],
                                        y=[_new_nodes[a]["y"], _new_nodes[b]["y"]],
                                        z=[_new_nodes[a]["z"], _new_nodes[b]["z"]],
                                        mode="lines", line=dict(color="#64748b", width=3),
                                        hoverinfo="skip", showlegend=False))
        # Flechas de DOF (dirección medida por cada sensor) — cono ámbar + astil
        _cx = []; _cy = []; _cz = []; _cu = []; _cv = []; _cw = []
        for n in _new_nodes:
            if not n["sensor"]:
                continue
            _seen = set()
            for d in _dofmap.get(n["sensor"], []):
                _t = (round(d[0], 2), round(d[1], 2), round(d[2], 2))
                if _t in _seen:
                    continue
                _seen.add(_t)
                _tx = n["x"] + _L * d[0]; _ty = n["y"] + _L * d[1]; _tz = n["z"] + _L * d[2]
                _fig.add_trace(go.Scatter3d(x=[n["x"], _tx], y=[n["y"], _ty], z=[n["z"], _tz],
                                            mode="lines", line=dict(color=AMBER, width=5),
                                            hoverinfo="skip", showlegend=False))
                _cx.append(_tx); _cy.append(_ty); _cz.append(_tz)
                _cu.append(d[0]); _cv.append(d[1]); _cw.append(d[2])
        if _cx:
            _fig.add_trace(go.Cone(x=_cx, y=_cy, z=_cz, u=_cu, v=_cv, w=_cw, anchor="tail",
                                   sizemode="absolute", sizeref=_L * 0.55, showscale=False,
                                   colorscale=[[0, AMBER], [1, AMBER]], hoverinfo="skip"))
        _stx = [n["x"] for n in _new_nodes if not n["sensor"]]
        if _stx:
            _fig.add_trace(go.Scatter3d(
                x=_stx, y=[n["y"] for n in _new_nodes if not n["sensor"]],
                z=[n["z"] for n in _new_nodes if not n["sensor"]],
                mode="markers", marker=dict(size=3.5, color=SLATE), hoverinfo="skip", showlegend=False))
        _snx = [n["x"] for n in _new_nodes if n["sensor"]]
        if _snx:
            _fig.add_trace(go.Scatter3d(
                x=_snx, y=[n["y"] for n in _new_nodes if n["sensor"]],
                z=[n["z"] for n in _new_nodes if n["sensor"]],
                mode="markers+text", text=[n["sensor"] for n in _new_nodes if n["sensor"]],
                textposition="bottom center", textfont=dict(size=10, color=GREEN),
                marker=dict(size=7, color=GREEN, symbol="diamond"), hoverinfo="text", showlegend=False))
        _fig.update_layout(height=580, template="watermelon", showlegend=False,
                           scene=dict(aspectmode="data", xaxis_title="", yaxis_title="", zaxis_title=""),
                           margin=dict(l=0, r=0, t=0, b=0))
        _chart(_fig)
        st.caption("📍 Sensors (green) with **amber DOF arrows** = the direction each one measures · "
                   "gray = structure points · lines = the body. Drag to rotate.")

    if _apply:
        st.session_state[f"geom_edit::{_run_key}"] = _gw2
        st.success("Applied — Mode shapes, ODS and the report now animate on this geometry.")
    if _reset:
        st.session_state.pop(_wk, None)
        st.session_state.pop(f"geom_edit::{_run_key}", None)
        st.rerun()
    if _cloud:
        try:
            lay.geometry = _gw2
            from core.modal.modal_cloud import save_layout_cloud
            _rc = save_layout_cloud(lay)
            if _rc.get("ok"):
                st.session_state[f"geom_edit::{_run_key}"] = _gw2
                st.success("Saved to the cloud — the field app will reuse this geometry.")
            else:
                st.warning(f"Could not save to cloud: {_rc.get('reason')}")
        except Exception as _eg:  # noqa: BLE001
            st.warning(f"Save failed: {type(_eg).__name__}: {_eg}")

if nav == T_MAC:
    _sec("MAC / validation", "Modal Assurance Criterion — mode-shape consistency (auto-MAC)", "ISO 7626-6")
    modes = D["oma_modes"]; shapes = D.get("shapes") or []
    M, _kept = _mac_from_shapes([shapes[i] if i < len(shapes) else None for i in range(len(modes))])
    if len(_kept) < 2:
        st.info("Need ≥2 modes with a resolved mode shape to compute the MAC.")
    else:
        fs_lbl = [float(modes[i]["fn"]) for i in _kept]
        import plotly.graph_objects as _goM
        lbl = [f"{f:.1f}" for f in fs_lbl]
        _figM = _goM.Figure(_goM.Heatmap(
            z=M, x=lbl, y=lbl, zmin=0, zmax=1,
            colorscale=[[0, "#eef4ff"], [0.5, "#78aaeb"], [0.8, "#f59e0b"], [1, "#c81e1e"]],
            text=[[f"{v:.2f}" for v in row] for row in M], texttemplate="%{text}",
            textfont={"size": 10}, colorbar=dict(title="MAC")))
        _figM.update_layout(height=520, yaxis_autorange="reversed",
                            xaxis_title="Mode (Hz)", yaxis_title="Mode (Hz)",
                            margin=dict(l=60, r=20, t=20, b=50))
        _chart(_figM)
        n = len(_kept)
        _dup = [(fs_lbl[i], fs_lbl[j], M[i, j]) for i in range(n) for j in range(i + 1, n) if M[i, j] > CFG_MAC_REDUNDANT]
        if _dup:
            st.warning("Redundant modes (MAC>0.7, same shape → review/remove one): "
                       + ", ".join(f"{a:.1f}↔{b:.1f} ({v:.2f})" for a, b, v in _dup))
        st.caption("Diagonal = 1 (a mode with itself). Off-diagonal: RED (>0.7) = redundant modes; "
                   "BLUE (~0) = independent (well separated).")
        # Cross-MAC EMA↔OMA: se activa cuando la corrida trae un impacto (EMA) con formas roving.
        _ema_shapes = D.get("ema_shapes")
        if _ema_shapes:
            CM, _ia, _ib = _cross_mac_from_shapes(_ema_shapes, shapes)
            if CM.size:
                st.markdown("**Cross-MAC — EMA ↔ OMA** (diagonal alta = mismo modo por ambos métodos, API 684)")
                _figCM = _goM.Figure(_goM.Heatmap(
                    z=CM, zmin=0, zmax=1,
                    colorscale=[[0, "#eef4ff"], [0.5, "#78aaeb"], [0.8, "#f59e0b"], [1, "#16a34a"]],
                    text=[[f"{v:.2f}" for v in row] for row in CM], texttemplate="%{text}",
                    textfont={"size": 10}, colorbar=dict(title="MAC")))
                _figCM.update_layout(height=380, yaxis_autorange="reversed",
                                     xaxis_title="OMA modes", yaxis_title="EMA modes",
                                     margin=dict(l=60, r=20, t=20, b=50))
                _chart(_figCM)
        else:
            st.caption("Cross-MAC (EMA↔OMA) activates when an impact test with roving mode shapes "
                       "is uploaded with this run.")

# ---------------------------------------------------------------- ODS (operating)
if nav == T_ODS:
    _sec("ODS — operating deflection", "How the machine moves at a given frequency (1×, blade-pass)")
    _fddO = D.get("fdd")
    if _fddO is None:
        st.info("ODS requires analysis from RAW DATA (full SVD). Pick a run that uploaded its raw "
                "data — the web analyzes it and enables ODS. Older runs (results only) cannot.")
    else:
        _frO = np.asarray(_fddO.frequencies_hz, float)
        _rpmO = float(D.get("rpm") or 0.0)
        _sug = []
        if _rpmO:
            _f1 = _rpmO / 60.0; _k = 1
            while _f1 * _k <= _frO.max() and _k <= 8:
                _sug.append((f"{_f1*_k:.1f} Hz ({_k}×)", _f1 * _k)); _k += 1
        for mm in D["oma_modes"]:
            _sug.append((f"{mm['fn']:.1f} Hz (pico)", float(mm["fn"])))
        _c1, _c2 = st.columns([2, 1])
        _f_sel = 0.0
        if _sug:
            _lblopts = [s[0] for s in _sug]
            _selO = _c1.selectbox("Suggested frequency", _lblopts)
            _f_sel = _sug[_lblopts.index(_selO)][1]
        _fpO = _c2.number_input("or exact Hz (0 = use selection)", 0.0, float(_frO.max()),
                                0.0, step=1.0)
        _f0 = _fpO if _fpO > 0 else _f_sel
        _UO = np.asarray(_fddO.mode_shapes_at_freq); _svO = np.asarray(_fddO.singular_values)
        if _UO.ndim == 3 and _f0 > 0:
            _j = int(np.argmin(np.abs(_frO - _f0))); _Uj = _UO[:, :, _j]
            _Sj = _Uj @ np.diag(_svO[:, _j].astype(complex)) @ _Uj.conj().T
            _dg2 = np.abs(np.diag(_Sj).real); _ref = int(np.argmax(_dg2)) if _dg2.size else 0
            _ods = _Sj[:, _ref] / np.sqrt(max(float(_Sj[_ref, _ref].real), 1e-30))
            _amp = np.asarray(_ods, complex).real
            _amp = _amp / (np.max(np.abs(_amp)) or 1.0)
            _ptsO = lay.active_points()
            if len(_amp) != len(_ptsO):
                st.warning("The ODS does not match the number of active sensors.")
            else:
                _geomO = _run_geometry(D, lay)
                _ordO = (_f0 / (_rpmO / 60.0)) if _rpmO else 0.0
                st.markdown(f"<div style='text-align:center;color:#64748b;font-weight:600;font-size:13px'>"
                            f"ODS @ {_f0:.2f} Hz" + (f" · {_ordO:.2f}×" if _rpmO else "") + "</div>",
                            unsafe_allow_html=True)
                if _rotor_is(lay):
                    _chart(_mode_rotor_fig(lay, _amp, height=560, scale_mul=1.5))
                else:
                    _chart(_mode_geom_fig(lay, _geomO, _amp, height=560, scale_mul=1.5))
                st.caption("OPERATIONAL deflection (not an identified mode): how the machine moves at "
                           "that frequency, with phase relative to the highest-response channel. Ideal "
                           "for seeing 1× (unbalance), blade-pass, etc.")
        else:
            st.info("Pick a frequency to reconstruct the ODS.")

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
            # consecutivo AUTOMÁTICO: el PREFIJO sale del tipo de ensayo (OMA / EMA / MODAL si
            # son ambos); la secuencia NNN se calcula por prefijo+año según el histórico. Cada
            # prefijo cuenta por separado. El usuario lo puede editar si lo necesita.
            _tm = [str(t).upper() for t in (getattr(lay, "test_modes", None) or ["OMA"])]
            _pfx = "MODAL" if ("EMA" in _tm and "OMA" in _tm) else ("EMA" if "EMA" in _tm else "OMA")
            _consec_pref = f"{_pfx}-{_date.today().year}-"
            if st.session_state.get("rep_consec_pref") != _consec_pref:
                # el tipo/año cambió → recalcular el consecutivo automático
                try:
                    from core.reports_archive import next_consecutive as _nc
                    st.session_state["rep_consec_auto"] = _nc(_consec_pref, _user.get("email", ""), _my_role)
                except Exception:  # noqa: BLE001
                    st.session_state["rep_consec_auto"] = f"{_consec_pref}001"
                st.session_state["rep_consec_pref"] = _consec_pref
                st.session_state.pop("rep_consec", None)   # refrescar el campo al nuevo prefijo
            _consec = st.text_input("Consecutive (auto)", key="rep_consec",
                                    value=st.session_state.get("rep_consec", st.session_state["rep_consec_auto"]),
                                    help="Prefijo automático por tipo: OMA / EMA / MODAL (ambos). Editable.")
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
                   "(singular values) · mode shapes (3D) · **MAC matrix** · **ODS (1×/2×)** · SSI "
                   "stabilization · Campbell (API 684) · EMA↔OMA correlation.")

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
                    pts = lay.active_points()
                    _geom_r = _run_geometry(D, lay)
                    # filtro spurious/harmonic + confianza (1 clic): modos + sus formas, en paralelo
                    _pairs = list(zip(D["oma_modes"], (D["shapes"] or [None] * len(D["oma_modes"]))))
                    if _hide_spur:
                        _pairs = [(mm, ss) for (mm, ss) in _pairs
                                  if str(mm.get("cls", "")).lower() not in ("spurious", "harmonic")]
                    if _only_conf:
                        _pairs = [(mm, ss) for (mm, ss) in _pairs if _conf_of(mm) == "Alta"]
                    _modes_r = [mm for mm, ss in _pairs]; _shapes_r = [ss for mm, ss in _pairs]
                    _SHAPE_CAP = CFG_REPORT_SHAPE_CAP   # tope de formas en el PDF
                    # formas modales  (superficies + cuadrícula), estáticas para el PDF
                    shape_pngs = []
                    for i, m in enumerate(_modes_r[:_SHAPE_CAP]):
                        # Solo se dibuja la forma modal si existe el vector REAL. Si falta,
                        # se omite (None) — JAMÁS se fabrica una forma para el reporte del cliente.
                        if not (i < len(_shapes_r) and _shapes_r[i] is not None
                                and len(_shapes_r[i]) == len(pts)):
                            shape_pngs.append(None)
                            continue
                        a = np.asarray(_shapes_r[i], float)
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
                    # --- MAC (validación de formas) para el reporte — mismo helper que la vista ---
                    mac_png = None
                    _MM, _kM = _mac_from_shapes(list(_shapes_r))
                    if len(_kM) >= 2:
                        try:
                            _macf = [_modes_r[i]["fn"] for i in _kM]
                            _lblM = [f"{f:.1f}" for f in _macf]
                            _figMac = go.Figure(go.Heatmap(
                                z=_MM, x=_lblM, y=_lblM, zmin=0, zmax=1,
                                colorscale=[[0, "#eef4ff"], [0.5, "#78aaeb"], [0.8, "#f59e0b"], [1, "#c81e1e"]],
                                text=[[f"{v:.2f}" for v in row] for row in _MM], texttemplate="%{text}",
                                textfont={"size": 11}, showscale=True))
                            _figMac.update_layout(height=560, width=560, yaxis_autorange="reversed",
                                                  xaxis_title="Modo (Hz)", yaxis_title="Modo (Hz)",
                                                  margin=dict(l=60, r=20, t=20, b=50),
                                                  paper_bgcolor="white", plot_bgcolor="white")
                            mac_png = _figMac.to_image(format="png", width=760, height=760, scale=2)
                        except Exception:  # noqa: BLE001
                            mac_png = None
                    # --- ODS a 1× y 2× para el reporte (si hay SVD completo desde la cruda) ---
                    ods_pngs = []
                    _fddR = D.get("fdd")
                    if _fddR is not None and _rpm_r:
                        try:
                            _frR = np.asarray(_fddR.frequencies_hz, float)
                            _UR = np.asarray(_fddR.mode_shapes_at_freq); _svR = np.asarray(_fddR.singular_values)
                            _f1r = _rpm_r / 60.0
                            for _ordk in (1, 2):
                                _f0r = _f1r * _ordk
                                if _UR.ndim != 3 or _f0r > _frR.max():
                                    continue
                                _jr = int(np.argmin(np.abs(_frR - _f0r))); _Ujr = _UR[:, :, _jr]
                                _Sjr = _Ujr @ np.diag(_svR[:, _jr].astype(complex)) @ _Ujr.conj().T
                                _dr = np.abs(np.diag(_Sjr).real); _refr = int(np.argmax(_dr)) if _dr.size else 0
                                _odsr = _Sjr[:, _refr] / np.sqrt(max(float(_Sjr[_refr, _refr].real), 1e-30))
                                _ar = np.asarray(_odsr, complex).real
                                _ar = _ar / (np.max(np.abs(_ar)) or 1.0)
                                if len(_ar) != len(pts):
                                    continue
                                _figO = (_mode_rotor_fig(lay, _ar, height=520, static=True) if _rotor_is(lay)
                                         else _mode_geom_fig(lay, _geom_r, _ar, height=520, static=True))
                                ods_pngs.append((f"{_f0r:.1f} Hz ({_ordk}×)",
                                                 _figO.to_image(format="png", width=1100, height=640, scale=2)))
                        except Exception:  # noqa: BLE001
                            pass
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
                        ssi_png=ssi_png, mac_png=mac_png, ods_pngs=(ods_pngs or None),
                        max_shape_modes=len(shape_pngs))
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
