from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

from core.auth import require_login, render_user_menu


st.set_page_config(page_title="Watermelon System | Bode Plot", layout="wide")
require_login()
render_user_menu()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"


# ============================================================
# PAGE STYLE
# ============================================================
def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .main > div { padding-top: 0.18rem; }
        .stApp { background-color: #f3f4f6; }

        section[data-testid="stSidebar"] {
            background: #e5e7eb;
            border-right: 1px solid #cbd5e1;
        }

        .wm-page-title {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.08rem;
            letter-spacing: -0.02em;
        }

        .wm-page-subtitle {
            color: #64748b;
            font-size: 0.96rem;
            margin-bottom: 1rem;
        }

        .wm-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.86));
            border: 1px solid #dbe3ee;
            border-radius: 18px;
            padding: 14px 16px 14px 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            margin-bottom: 12px;
        }

        .wm-card-title {
            font-size: 1.02rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.18rem;
        }

        .wm-card-subtitle {
            color: #64748b;
            font-size: 0.90rem;
            margin-bottom: 0.35rem;
        }

        .wm-meta {
            color: #334155;
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .wm-chip-row {
            display:flex;
            flex-wrap:wrap;
            gap:8px;
            margin-top:10px;
        }

        .wm-chip {
            border:1px solid #dbe3ee;
            background:#f8fafc;
            border-radius:999px;
            padding:5px 10px;
            color:#334155;
            font-size:0.82rem;
            font-weight:600;
        }

        section.main div[data-testid="stButton"] > button,
        section.main div[data-testid="stDownloadButton"] > button {
            min-height: 52px;
            border-radius: 16px;
            font-weight: 700;
            border: 1px solid #bfd8ff !important;
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%) !important;
            color: #2563eb !important;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.08);
            transition: all 0.18s ease;
        }

        section.main div[data-testid="stButton"] > button:hover,
        section.main div[data-testid="stDownloadButton"] > button:hover {
            border-color: #93c5fd !important;
            background: linear-gradient(180deg, #ffffff 0%, #f3f8ff 100%) !important;
            color: #1d4ed8 !important;
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.12);
        }

        .wm-export-actions {
            margin-top: 0.85rem;
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_page_style()


# ============================================================
# HELPERS
# ============================================================
def ensure_report_state() -> None:
    if "report_items" not in st.session_state:
        st.session_state["report_items"] = []


def format_number(value: Any, digits: int = 3, fallback: str = "—") -> str:
    if value is None:
        return fallback
    try:
        val = float(value)
        if not math.isfinite(val):
            return fallback
        return f"{val:.{digits}f}"
    except Exception:
        return fallback


def get_logo_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def rounded_rect_path(x0: float, y0: float, x1: float, y1: float, r: float) -> str:
    r = max(0.0, min(r, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
    return (
        f"M {x0+r},{y0} "
        f"L {x1-r},{y0} "
        f"Q {x1},{y0} {x1},{y0+r} "
        f"L {x1},{y1-r} "
        f"Q {x1},{y1} {x1-r},{y1} "
        f"L {x0+r},{y1} "
        f"Q {x0},{y1} {x0},{y1-r} "
        f"L {x0},{y0+r} "
        f"Q {x0},{y0} {x0+r},{y0} Z"
    )


# ============================================================
# CSV LOADER
# ============================================================
def read_bode_csv(file_obj) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame]:
    file_obj.seek(0)
    raw_bytes = file_obj.read()
    text = raw_bytes.decode("utf-8-sig", errors="replace") if isinstance(raw_bytes, bytes) else str(raw_bytes)

    lines = text.splitlines()
    if not lines:
        raise ValueError("Archivo vacío.")

    header_idx = None
    for i, line in enumerate(lines):
        if "X-Axis Value" in line and "Y-Axis Value" in line and "Phase" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("No se encontró el encabezado real del CSV de Bode.")

    meta_lines = lines[:header_idx]
    data_text = "\n".join(lines[header_idx:])

    meta: Dict[str, str] = {}
    for line in meta_lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) == 2:
            meta[parts[0]] = parts[1]

    df = pd.read_csv(io.StringIO(data_text), encoding="utf-8-sig")

    required = [
        "X-Axis Value",
        "Y-Axis Value",
        "Y-Axis Status",
        "Phase",
        "Phase Status",
        "Timestamp",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    df["rpm"] = pd.to_numeric(df["X-Axis Value"], errors="coerce")
    df["amp"] = pd.to_numeric(df["Y-Axis Value"], errors="coerce")
    df["phase"] = pd.to_numeric(df["Phase"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.dropna(subset=["rpm", "amp", "phase", "Timestamp"]).copy()
    df = df[
        df["Y-Axis Status"].astype(str).str.strip().str.lower().eq("valid")
        & df["Phase Status"].astype(str).str.strip().str.lower().eq("valid")
    ].copy()

    if df.empty:
        raise ValueError("No quedaron filas válidas después del filtrado.")

    raw_df = df.sort_values(["Timestamp", "rpm"]).reset_index(drop=True)

    grouped_df = (
        raw_df.groupby("rpm", as_index=False)
        .agg(
            amp=("amp", "median"),
            phase=("phase", lambda s: circular_mean_deg(s)),
            samples=("Timestamp", "size"),
            ts_min=("Timestamp", "min"),
            ts_max=("Timestamp", "max"),
        )
        .sort_values("rpm", kind="stable")
        .reset_index(drop=True)
    )

    return meta, raw_df, grouped_df


# ============================================================
# SIGNAL PROCESSING
# ============================================================
def smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series.copy()
    return series.rolling(window=window, center=True, min_periods=1).median()



def circular_mean_deg(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if vals.empty:
        return float("nan")
    rad = np.deg2rad(vals.to_numpy() % 360.0)
    c = np.mean(np.cos(rad))
    s = np.mean(np.sin(rad))
    ang = np.rad2deg(np.arctan2(s, c))
    return float((ang + 360.0) % 360.0)

def circular_smooth_deg(phase_deg: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return phase_deg.astype(float).copy()

    rad = np.deg2rad(phase_deg.astype(float).to_numpy())
    c = pd.Series(np.cos(rad)).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    s = pd.Series(np.sin(rad)).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    out = np.rad2deg(np.arctan2(s, c))
    out = (out + 360.0) % 360.0
    return pd.Series(out, index=phase_deg.index)


def unwrap_deg(phase_deg: pd.Series) -> pd.Series:
    rad = np.deg2rad(phase_deg.astype(float).to_numpy())
    return pd.Series(np.rad2deg(np.unwrap(rad)), index=phase_deg.index)


def choose_best_phase_rotation(unwrapped_deg: pd.Series) -> float:
    # Busca un offset que minimice saltos visuales en la fase envuelta
    values = unwrapped_deg.astype(float).to_numpy()
    best_offset = 0.0
    best_score = float("inf")

    for offset in range(0, 360, 2):
        wrapped = (values + offset) % 360.0
        diffs = np.abs(np.diff(wrapped))
        score = diffs.sum() + 25.0 * np.sum(diffs > 120.0)
        if score < best_score:
            best_score = score
            best_offset = float(offset)

    return best_offset



def make_pretty_wrapped_phase(phase_deg: pd.Series, smooth_window: int = 1) -> Tuple[pd.Series, float]:
    """
    Para Bode tipo System 1:
    - NO auto-rotation
    - prioridad a fase wrapped real
    - suavizado circular muy leve opcional
    """
    wrapped = (phase_deg.astype(float) % 360.0).copy()

    if smooth_window > 1:
        wrapped = circular_smooth_deg(wrapped, smooth_window)

    return wrapped.astype(float), 0.0


def nearest_row_for_rpm(df: pd.DataFrame, rpm_value: float) -> pd.Series:
    idx = int((df["rpm"] - rpm_value).abs().idxmin())
    return df.loc[idx]



def estimate_critical_speeds_api684_style(df: pd.DataFrame, max_count: int = 2) -> List[Dict[str, float]]:
    """
    Heurística práctica estilo API 684:
    - picos dominantes de amplitud
    - verifica cambio de fase local alrededor del pico
    - permite hasta 2 velocidades críticas
    """
    if df.empty or len(df) < 12:
        return []

    amp = df["amp"].astype(float).to_numpy()
    rpm = df["rpm"].astype(float).to_numpy()

    # usar fase wrapped suave para evaluar cambio local de fase, más parecido a System 1
    phase_ref = circular_smooth_deg(df["phase"].astype(float) % 360.0, 5).to_numpy()

    def shortest_angle(a, b):
        return ((b - a + 180.0) % 360.0) - 180.0

    candidates = []

    if find_peaks is not None:
        prominence = max(np.nanmax(amp) * 0.06, 0.12)
        distance = max(6, len(df) // 18)
        peaks, props = find_peaks(amp, prominence=prominence, distance=distance)

        for i, p in enumerate(peaks):
            left = max(0, p - 10)
            right = min(len(df) - 1, p + 10)

            phase_delta = shortest_angle(phase_ref[left], phase_ref[right])
            amp_peak = float(amp[p])
            prom = float(props["prominences"][i])

            # filtro más permisivo para capturar first y second critical
            if amp_peak < np.nanmax(amp) * 0.45:
                continue
            if abs(phase_delta) < 6.0:
                continue

            candidates.append(
                {
                    "rpm": float(rpm[p]),
                    "amp": amp_peak,
                    "phase_delta": float(phase_delta),
                    "idx": int(p),
                    "prominence": prom,
                }
            )
    else:
        p = int(np.nanargmax(amp))
        left = max(0, p - 10)
        right = min(len(df) - 1, p + 10)
        phase_delta = shortest_angle(phase_ref[left], phase_ref[right])
        candidates.append(
            {
                "rpm": float(rpm[p]),
                "amp": float(amp[p]),
                "phase_delta": float(phase_delta),
                "idx": int(p),
                "prominence": float(amp[p]),
            }
        )

    candidates = sorted(candidates, key=lambda x: (x["prominence"], x["amp"]), reverse=True)

    # eliminar duplicados cercanos en rpm
    filtered = []
    for cand in candidates:
        if all(abs(cand["rpm"] - kept["rpm"]) > 120 for kept in filtered):
            filtered.append(cand)
        if len(filtered) >= max_count:
            break

    return filtered


# ============================================================
# IN-CHART DECORATION
# ============================================================
def _draw_top_strip(
    fig: go.Figure,
    meta: Dict[str, str],
    row_a: pd.Series,
    row_b: pd.Series,
    logo_uri: Optional[str],
) -> None:
    x0, x1 = 0.006, 0.994
    y0, y1 = 1.014, 1.106
    radius = 0.015

    fig.add_shape(
        type="path",
        xref="paper",
        yref="paper",
        path=rounded_rect_path(x0, y0, x1, y1, radius),
        line=dict(color="#cfd8e3", width=1.15),
        fillcolor="rgba(255,255,255,0.97)",
        layer="below",
    )

    y_text = (y0 + y1) / 2.0

    machine = meta.get("Machine Name", "")
    point = meta.get("Point Name", "")
    variable = meta.get("Variable", "")
    angle = meta.get("Probe Angle", "")
    x_unit = meta.get("X-Axis Unit", "rpm") or "rpm"
    y_unit = meta.get("Y-Axis Unit", "") or ""

    if logo_uri:
        fig.add_layout_image(
            dict(
                source=logo_uri,
                xref="paper",
                yref="paper",
                x=0.014,
                y=y1 - 0.009,
                sizex=0.060,
                sizey=0.090,
                xanchor="left",
                yanchor="top",
                layer="above",
                sizing="contain",
                opacity=1.0,
            )
        )
        machine_x = 0.083
    else:
        machine_x = 0.020

    fig.add_annotation(
        xref="paper", yref="paper",
        x=machine_x, y=y_text,
        xanchor="left", yanchor="middle",
        text=f"<b>{machine}</b>",
        showarrow=False,
        font=dict(size=12.8, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.205, y=y_text,
        xanchor="left", yanchor="middle",
        text=point,
        showarrow=False,
        font=dict(size=12.1, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.370, y=y_text,
        xanchor="left", yanchor="middle",
        text=f"{variable} | {angle}",
        showarrow=False,
        font=dict(size=12.0, color="#111827"),
        align="left",
    )

    a_txt = (
        f"A: <b>{format_number(row_a['amp'],3)} {y_unit}</b> "
        f"∠{format_number(row_a['phase_header'],1)}° @ {int(round(row_a['rpm']))} {x_unit}"
    )
    b_txt = (
        f"B: <b>{format_number(row_b['amp'],3)} {y_unit}</b> "
        f"∠{format_number(row_b['phase_header'],1)}° @ {int(round(row_b['rpm']))} {x_unit}"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.605, y=y_text,
        xanchor="left", yanchor="middle",
        text=f"{a_txt} &nbsp;&nbsp;|&nbsp;&nbsp; {b_txt}",
        showarrow=False,
        font=dict(size=11.2, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.986, y=y_text,
        xanchor="right", yanchor="middle",
        text=f"{int(round(row_a['rpm']))} - {int(round(row_b['rpm']))} {x_unit}",
        showarrow=False,
        font=dict(size=11.2, color="#111827"),
        align="right",
    )


def _draw_right_info_box(fig: go.Figure, rows: List[Tuple[str, str]]) -> None:
    panel_x0 = 0.805
    panel_x1 = 0.965
    panel_y1 = 0.915
    header_h = 0.034
    row_h = 0.054
    panel_h = header_h + len(rows) * row_h + 0.018
    panel_y0 = panel_y1 - panel_h

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path=rounded_rect_path(panel_x0, panel_y0, panel_x1, panel_y1, 0.012),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(255,255,255,0.72)",
        layer="above",
    )

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path=rounded_rect_path(panel_x0, panel_y1 - header_h, panel_x1, panel_y1, 0.012),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(147,197,253,0.94)",
        layer="above",
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=(panel_x0 + panel_x1) / 2.0, y=panel_y1 - header_h / 2.0,
        text="<b>Bode Information</b>",
        showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=11.4, color="#111827"),
    )

    current_top = panel_y1 - header_h - 0.008
    for title, value in rows:
        title_y = current_top - 0.004
        value_y = current_top - 0.028

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.028, y=title_y,
            xanchor="left", yanchor="top",
            text=f"<b>{title}</b>",
            showarrow=False, font=dict(size=10.6, color="#111827"), align="left",
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.028, y=value_y,
            xanchor="left", yanchor="top",
            text=value,
            showarrow=False, font=dict(size=10.2, color="#111827"), align="left",
        )

        current_top -= row_h


def build_bode_info_rows(
    row_a: pd.Series,
    row_b: pd.Series,
    phase_mode: str,
    y_unit: str,
    x_unit: str,
    critical_speeds: List[Dict[str, float]],
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = [
        ("Cursor A", f"{format_number(row_a['amp'],3)} {y_unit} @ {int(round(row_a['rpm']))} {x_unit} | ∠{format_number(row_a['phase_header'],1)}°"),
        ("Cursor B", f"{format_number(row_b['amp'],3)} {y_unit} @ {int(round(row_b['rpm']))} {x_unit} | ∠{format_number(row_b['phase_header'],1)}°"),
        ("Phase Mode", phase_mode),
    ]

    for i, cs in enumerate(critical_speeds, start=1):
        rows.append((f"Critical Speed {i}", f"{int(round(cs['rpm']))} {x_unit} | {format_number(cs['amp'],3)} {y_unit}"))
        rows.append((f"Phase Delta {i}", f"{format_number(cs['phase_delta'],1)}°"))

    return rows


# ============================================================
# FIGURE
# ============================================================
def build_bode_figure(
    df: pd.DataFrame,
    meta: Dict[str, str],
    row_a: pd.Series,
    row_b: pd.Series,
    x_min: float,
    x_max: float,
    logo_uri: Optional[str],
    phase_mode: str,
    critical_speeds: List[Dict[str, float]],
    show_info_box: bool,
) -> go.Figure:
    x_unit = meta.get("X-Axis Unit", "rpm") or "rpm"
    y_unit = meta.get("Y-Axis Unit", "") or ""

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        row_heights=[0.48, 0.52],
    )

    phase_trace = go.Scattergl(
        x=df["rpm"],
        y=df["phase_plot"],
        mode="lines",
        line=dict(width=1.22, color="#5b9cf0"),
        name="Phase",
        hovertemplate=f"Speed: %{{x:.0f}} {x_unit}<br>Phase: %{{y:.1f}}°<extra></extra>",
        showlegend=False,
        connectgaps=False,
    )
    amp_trace = go.Scattergl(
        x=df["rpm"],
        y=df["amp"],
        mode="lines",
        line=dict(width=1.45, color="#5b9cf0"),
        name="Amplitude",
        hovertemplate=f"Speed: %{{x:.0f}} {x_unit}<br>Amplitude: %{{y:.3f}} {y_unit}<extra></extra>",
        showlegend=False,
        connectgaps=False,
    )

    fig.add_trace(phase_trace, row=1, col=1)
    fig.add_trace(amp_trace, row=2, col=1)

    # cursores
    for rpm_val, color in [
        (float(row_a["rpm"]), "#efb08c"),
        (float(row_b["rpm"]), "#7ac77b"),
    ]:
        fig.add_vline(x=rpm_val, line_width=1.5, line_dash="dot", line_color=color, row=1, col=1)
        fig.add_vline(x=rpm_val, line_width=1.5, line_dash="dot", line_color=color, row=2, col=1)

    # critical speeds
    cs_colors = ["#ef4444", "#f59e0b"]
    for idx, cs in enumerate(critical_speeds):
        color = cs_colors[idx % len(cs_colors)]
        cs_rpm = cs["rpm"]
        cs_amp = cs["amp"]
        cs_phase_row = nearest_row_for_rpm(df, cs_rpm)
        cs_phase = float(cs_phase_row["phase_plot"])

        fig.add_vline(x=cs_rpm, line_width=2.0, line_dash="dash", line_color=color, row=1, col=1)
        fig.add_vline(x=cs_rpm, line_width=2.0, line_dash="dash", line_color=color, row=2, col=1)

        fig.add_annotation(
            x=cs_rpm, y=cs_phase,
            xref="x", yref="y",
            text=f"Critical Speed {idx+1}<br>{int(round(cs_rpm))} rpm",
            showarrow=True, arrowhead=2, arrowcolor=color,
            ax=42, ay=-35,
            font=dict(size=10.5, color="#7f1d1d" if idx == 0 else "#92400e"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#fecaca" if idx == 0 else "#fde68a",
        )

        fig.add_annotation(
            x=cs_rpm, y=cs_amp,
            xref="x2", yref="y2",
            text=f"{format_number(cs_amp,3)} {y_unit}",
            showarrow=True, arrowhead=2, arrowcolor=color,
            ax=40, ay=-30,
            font=dict(size=10.5, color="#7f1d1d" if idx == 0 else "#92400e"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#fecaca" if idx == 0 else "#fde68a",
        )

    _draw_top_strip(fig, meta, row_a, row_b, logo_uri)

    if show_info_box:
        rows = build_bode_info_rows(
            row_a=row_a,
            row_b=row_b,
            phase_mode=phase_mode,
            y_unit=y_unit,
            x_unit=x_unit,
            critical_speeds=critical_speeds,
        )
        _draw_right_info_box(fig, rows)

    x_domain = [0.0, 0.77] if show_info_box else [0.0, 1.0]

    fig.update_layout(
        height=820,
        margin=dict(l=48, r=20, t=145, b=48),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f3f4f6",
        font=dict(color="#111827"),
        hovermode="closest",
        dragmode="pan",
    )

    fig.update_xaxes(
        title=f"Speed ({x_unit})",
        range=[x_min, x_max],
        domain=x_domain,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
        row=2, col=1,
    )

    fig.update_xaxes(
        range=[x_min, x_max],
        domain=x_domain,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
        row=1, col=1,
    )

    phase_title = "Phase (°)" if phase_mode != "Unwrapped" else "Phase Unwrapped (°)"
    fig.update_yaxes(
        title=phase_title,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
        row=1, col=1,
    )

    fig.update_yaxes(
        title=f"Amplitude ({y_unit})" if y_unit else "Amplitude",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
        row=2, col=1,
    )

    return fig


# ============================================================
# EXPORT / REPORT
# ============================================================
def _build_export_safe_figure(fig: go.Figure) -> go.Figure:
    export_fig = go.Figure()

    for trace in fig.data:
        if isinstance(trace, go.Scattergl):
            tj = trace.to_plotly_json()
            export_fig.add_trace(
                go.Scatter(
                    x=tj.get("x"),
                    y=tj.get("y"),
                    mode=tj.get("mode"),
                    line=tj.get("line"),
                    marker=tj.get("marker"),
                    hovertemplate=tj.get("hovertemplate"),
                    showlegend=tj.get("showlegend"),
                    name=tj.get("name"),
                    xaxis=tj.get("xaxis"),
                    yaxis=tj.get("yaxis"),
                    connectgaps=tj.get("connectgaps", False),
                )
            )
        else:
            export_fig.add_trace(trace)

    export_fig.update_layout(fig.layout)
    return export_fig


def _scale_export_figure(export_fig: go.Figure) -> go.Figure:
    fig = go.Figure(export_fig)

    scaled = []
    for trace in fig.data:
        tj = trace.to_plotly_json()
        if tj.get("type") == "scatter":
            mode = tj.get("mode", "")
            if "lines" in mode:
                line = dict(tj.get("line", {}) or {})
                line["width"] = max(3.8, float(line.get("width", 1.0)) * 2.4)
                tj["line"] = line
            if "markers" in mode:
                marker = dict(tj.get("marker", {}) or {})
                marker["size"] = max(12, float(marker.get("size", 6)) * 1.7)
                tj["marker"] = marker
        scaled.append(go.Scatter(**tj))

    fig = go.Figure(data=scaled, layout=fig.layout)

    fig.update_layout(
        width=4300,
        height=2200,
        margin=dict(l=110, r=60, t=330, b=110),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        font=dict(size=28, color="#111827"),
    )

    fig.update_xaxes(title_font=dict(size=38), tickfont=dict(size=24))
    fig.update_yaxes(title_font=dict(size=38), tickfont=dict(size=24))

    for shape in fig.layout.shapes or []:
        if shape.line is not None:
            width = getattr(shape.line, "width", 1) or 1
            shape.line.width = max(2.0, width * 2.0)

    for ann in fig.layout.annotations or []:
        if ann.font is not None:
            ann.font.size = max(21, int((ann.font.size or 12) * 1.9))

    for img in fig.layout.images or []:
        sx = getattr(img, "sizex", None)
        sy = getattr(img, "sizey", None)
        if sx is not None:
            img.sizex = sx * 1.15
        if sy is not None:
            img.sizey = sy * 1.15

    return fig


def build_export_png_bytes(fig: go.Figure) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        export_fig = _build_export_safe_figure(fig)
        export_fig = _scale_export_figure(export_fig)
        return export_fig.to_image(format="png", width=4300, height=2200, scale=2), None
    except Exception as e:
        return None, str(e)


def queue_bode_to_report(meta: Dict[str, str], fig: go.Figure, title: str) -> None:
    ensure_report_state()
    st.session_state.report_items.append(
        {
            "id": f"report-bode-{meta.get('Machine Name','')}-{meta.get('Point Name','')}-{title}",
            "type": "bode",
            "title": title,
            "notes": "",
            "signal_id": meta.get("Point Name", ""),
            "figure": go.Figure(fig),
            "machine": meta.get("Machine Name", ""),
            "point": meta.get("Point Name", ""),
            "variable": meta.get("Variable", ""),
            "timestamp": "",
        }
    )


# ============================================================
# MAIN
# ============================================================
if "wm_bode_export_store" not in st.session_state:
    st.session_state.wm_bode_export_store = {}
ensure_report_state()

st.markdown('<div class="wm-page-title">Bode Plot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wm-page-subtitle">Amplitude and phase versus speed from Bode CSV files.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Upload Bode CSV")
    uploaded_file = st.file_uploader("Upload Bode CSV", type=["csv"], accept_multiple_files=False)

if not uploaded_file:
    st.info("Carga un archivo CSV de Bode para visualizar amplitud y fase contra velocidad.")
    st.stop()

try:
    meta, raw_df, grouped_df = read_bode_csv(uploaded_file)
except Exception as e:
    st.error(f"No pude leer el CSV Bode: {e}")
    st.stop()

if grouped_df.empty:
    st.error("No hay datos válidos para construir el Bode.")
    st.stop()

with st.sidebar:
    st.markdown("### X Axis Control")
    auto_x = st.checkbox("Auto scale X", value=True)
    x_min_default = float(grouped_df["rpm"].min())
    x_max_default = float(grouped_df["rpm"].max())

    if auto_x:
        x_min = x_min_default
        x_max = x_max_default
    else:
        x_min = st.number_input("Min RPM", value=float(x_min_default), step=10.0)
        x_max = st.number_input("Max RPM", value=float(x_max_default), step=10.0)

    st.markdown("### Phase Mode")
    phase_mode = st.selectbox("Phase display", ["Wrapped Raw 0-360", "Wrapped Smoothed", "Unwrapped"], index=0)

    st.markdown("### Smoothing")
    smooth_window = st.slider("Median smoothing window", 1, 21, 3, step=2)

    st.markdown("### Critical Speed Detection")
    detect_cs = st.checkbox("Estimate critical speeds (API-684 heuristic)", value=True)
    max_critical_speeds = st.selectbox("Max critical speeds", [1, 2], index=1)

    st.markdown("### Information Box")
    show_info_box = st.checkbox("Show Bode Information", value=True)

    st.markdown("### Cursors")
    cursor_a_rpm = st.slider("Cursor A (RPM)", int(x_min_default), int(x_max_default), int(grouped_df["rpm"].iloc[0]))
    cursor_b_rpm = st.slider("Cursor B (RPM)", int(x_min_default), int(x_max_default), int(grouped_df["rpm"].iloc[-1]))

plot_df = grouped_df.copy()
plot_df["amp"] = smooth_series(plot_df["amp"], smooth_window)

# Fase: priorizar representación parecida a System 1
plot_df["phase_wrapped_raw"] = plot_df["phase"].astype(float) % 360.0
plot_df["phase_wrapped_smooth"] = circular_smooth_deg(plot_df["phase_wrapped_raw"], min(smooth_window, 5))
plot_df["phase_unwrapped"] = unwrap_deg(plot_df["phase_wrapped_smooth"])

if phase_mode == "Wrapped Raw 0-360":
    plot_df["phase_plot"] = plot_df["phase_wrapped_raw"]
    plot_df["phase_header"] = plot_df["phase_wrapped_raw"]
    phase_offset = 0.0
elif phase_mode == "Wrapped Smoothed":
    plot_df["phase_plot"] = plot_df["phase_wrapped_smooth"]
    plot_df["phase_header"] = plot_df["phase_wrapped_smooth"]
    phase_offset = 0.0
else:
    plot_df["phase_plot"] = plot_df["phase_unwrapped"]
    plot_df["phase_header"] = plot_df["phase_unwrapped"]
    phase_offset = 0.0

row_a = nearest_row_for_rpm(plot_df, cursor_a_rpm)
row_b = nearest_row_for_rpm(plot_df, cursor_b_rpm)

critical_speeds: List[Dict[str, float]] = []
if detect_cs:
    critical_speeds = estimate_critical_speeds_api684_style(plot_df, max_count=max_critical_speeds)

machine = meta.get("Machine Name", "-")
point = meta.get("Point Name", "-")
variable = meta.get("Variable", "-")
probe_angle = meta.get("Probe Angle", "-")
x_unit = meta.get("X-Axis Unit", "rpm")
y_unit = meta.get("Y-Axis Unit", "")

st.markdown(
    f"""
    <div class="wm-card">
        <div class="wm-card-title">{machine} · {point}</div>
        <div class="wm-card-subtitle">Bode run-up / coast-down view</div>
        <div class="wm-meta">
            Variable: <b>{variable}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
            Probe Angle: <b>{probe_angle}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
            Speed Range: <b>{int(plot_df['rpm'].min())} - {int(plot_df['rpm'].max())} {x_unit}</b>
        </div>
        <div class="wm-chip-row">
            <div class="wm-chip">Raw rows: {len(raw_df):,}</div>
            <div class="wm-chip">Grouped points: {len(plot_df):,}</div>
            <div class="wm-chip">Phase mode: {phase_mode}</div>
            <div class="wm-chip">Smoothing: {smooth_window}</div>
            <div class="wm-chip">Phase offset: {format_number(phase_offset,1)}°</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

logo_uri = get_logo_data_uri(LOGO_PATH)
fig = build_bode_figure(
    df=plot_df,
    meta=meta,
    row_a=row_a,
    row_b=row_b,
    x_min=x_min,
    x_max=x_max,
    logo_uri=logo_uri,
    phase_mode=phase_mode,
    critical_speeds=critical_speeds,
    show_info_box=show_info_box,
)

st.plotly_chart(
    fig,
    width="stretch",
    config={"displaylogo": False},
    key="wm_bode_plot",
)

title = f"Bode — {machine} — {point}"

export_state_key = (
    f"bode::{machine}::{point}::{variable}::{phase_mode}::{x_min}::{x_max}::"
    f"{smooth_window}::{detect_cs}::{max_critical_speeds}::{show_info_box}"
)

if export_state_key not in st.session_state.wm_bode_export_store:
    st.session_state.wm_bode_export_store[export_state_key] = {"png_bytes": None, "error": None}

st.markdown('<div class="wm-export-actions"></div>', unsafe_allow_html=True)
left_pad, col_export1, col_export2, col_report, right_pad = st.columns([2.0, 1.2, 1.2, 1.2, 2.0])

with col_export1:
    if st.button("Prepare PNG HD", key=f"prepare_bode_png_{export_state_key}", width="stretch"):
        with st.spinner("Generating HD export..."):
            png_bytes, export_error = build_export_png_bytes(fig)
            st.session_state.wm_bode_export_store[export_state_key]["png_bytes"] = png_bytes
            st.session_state.wm_bode_export_store[export_state_key]["error"] = export_error

with col_export2:
    png_bytes = st.session_state.wm_bode_export_store[export_state_key]["png_bytes"]
    if png_bytes is not None:
        st.download_button(
            "Download PNG HD",
            data=png_bytes,
            file_name=f"{Path(uploaded_file.name).stem}_bode_hd.png",
            mime="image/png",
            key=f"download_bode_png_{export_state_key}",
            width="stretch",
        )
    else:
        st.button("Download PNG HD", disabled=True, key=f"download_bode_disabled_{export_state_key}", width="stretch")

with col_report:
    if st.button("Enviar a Reporte", key=f"report_bode_{export_state_key}", width="stretch"):
        queue_bode_to_report(meta, fig, title)
        st.success("Bode enviado al reporte")

panel_error = st.session_state.wm_bode_export_store[export_state_key]["error"]
if panel_error:
    st.warning(f"PNG export error: {panel_error}")

