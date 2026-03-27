from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.auth import require_login, render_user_menu


st.set_page_config(page_title="Watermelon System | Polar Plot", layout="wide")
require_login()
render_user_menu()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"


# ============================================================
# STYLE
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
    rad = np.deg2rad(phase_deg.astype(float).to_numpy() % 360.0)
    c = pd.Series(np.cos(rad)).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    s = pd.Series(np.sin(rad)).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    out = np.rad2deg(np.arctan2(s, c))
    out = (out + 360.0) % 360.0
    return pd.Series(out, index=phase_deg.index)


def nearest_row_for_speed(df: pd.DataFrame, speed_value: float) -> pd.Series:
    idx = int((df["speed"] - speed_value).abs().idxmin())
    return df.loc[idx]


# ============================================================
# CSV LOADER
# ============================================================
def read_polar_csv(file_obj) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame]:
    file_obj.seek(0)
    raw_bytes = file_obj.read()
    text = raw_bytes.decode("utf-8-sig", errors="replace") if isinstance(raw_bytes, bytes) else str(raw_bytes)

    lines = text.splitlines()
    if not lines:
        raise ValueError("Archivo vacío.")

    header_idx = None
    for i, line in enumerate(lines):
        if "Amp" in line and "Phase" in line and "Speed" in line and "Timestamp" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("No se encontró el encabezado real del CSV Polar.")

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

    required = ["Amp", "Amp Status", "Phase", "Phase Status", "Speed", "Speed Status", "Timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    df["amp"] = pd.to_numeric(df["Amp"], errors="coerce")
    df["phase"] = pd.to_numeric(df["Phase"], errors="coerce")
    df["speed"] = pd.to_numeric(df["Speed"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.dropna(subset=["amp", "phase", "speed", "Timestamp"]).copy()
    df = df[
        df["Amp Status"].astype(str).str.strip().str.lower().eq("valid")
        & df["Phase Status"].astype(str).str.strip().str.lower().eq("valid")
        & df["Speed Status"].astype(str).str.strip().str.lower().eq("valid")
    ].copy()

    if df.empty:
        raise ValueError("No quedaron filas válidas después del filtrado.")

    raw_df = df.sort_values(["Timestamp", "speed"]).reset_index(drop=True)

    grouped_df = (
        raw_df.groupby("speed", as_index=False)
        .agg(
            amp=("amp", "median"),
            phase=("phase", lambda s: circular_mean_deg(s)),
            samples=("Timestamp", "size"),
            ts_min=("Timestamp", "min"),
            ts_max=("Timestamp", "max"),
        )
        .sort_values("speed", kind="stable")
        .reset_index(drop=True)
    )

    return meta, raw_df, grouped_df


# ============================================================
# IN-CHART DECORATION
# ============================================================
def _draw_top_strip(
    fig: go.Figure,
    meta: Dict[str, str],
    row_a: pd.Series,
    row_b: pd.Series,
    logo_uri: Optional[str],
    df: pd.DataFrame,
) -> None:
    x0, x1 = 0.006, 0.994
    y0, y1 = 1.014, 1.104
    radius = 0.014

    fig.add_shape(
        type="path",
        xref="paper",
        yref="paper",
        path=rounded_rect_path(x0, y0, x1, y1, radius),
        line=dict(color="#cfd8e3", width=1.1),
        fillcolor="rgba(255,255,255,0.97)",
        layer="below",
    )

    y_text = (y0 + y1) / 2.0

    machine = meta.get("Machine Name", "")
    point = meta.get("Point Name", "")
    variable = meta.get("Variable", "")
    angle = meta.get("Probe Angle", "")
    speed_unit = meta.get("Speed Unit", "rpm") or "rpm"
    amp_unit = meta.get("Amp Unit", "") or ""

    if logo_uri:
        fig.add_layout_image(
            dict(
                source=logo_uri,
                xref="paper",
                yref="paper",
                x=0.014,
                y=y1 - 0.008,
                sizex=0.055,
                sizey=0.085,
                xanchor="left",
                yanchor="top",
                layer="above",
                sizing="contain",
                opacity=1.0,
            )
        )
        machine_x = 0.082
    else:
        machine_x = 0.020

    fig.add_annotation(
        xref="paper", yref="paper",
        x=machine_x, y=y_text,
        xanchor="left", yanchor="middle",
        text=f"<b>{machine}</b>",
        showarrow=False,
        font=dict(size=12.3, color="#111827"),
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.205, y=y_text,
        xanchor="left", yanchor="middle",
        text=point,
        showarrow=False,
        font=dict(size=11.7, color="#111827"),
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.355, y=y_text,
        xanchor="left", yanchor="middle",
        text=f"{variable} | {angle}",
        showarrow=False,
        font=dict(size=11.5, color="#111827"),
    )

    dt_start = pd.to_datetime(df["ts_min"], errors="coerce").min()
    dt_end = pd.to_datetime(df["ts_max"], errors="coerce").max()
    dt_text = "—"
    if pd.notna(dt_start) and pd.notna(dt_end):
        dt_text = f"{dt_start.strftime('%Y-%m-%d %H:%M:%S')} → {dt_end.strftime('%Y-%m-%d %H:%M:%S')}"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.60, y=y_text,
        xanchor="left", yanchor="middle",
        text=dt_text,
        showarrow=False,
        font=dict(size=10.8, color="#111827"),
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.986, y=y_text,
        xanchor="right", yanchor="middle",
        text=f"{int(round(df['speed'].min()))} - {int(round(df['speed'].max()))} {speed_unit}",
        showarrow=False,
        font=dict(size=10.8, color="#111827"),
    )


def _draw_right_info_box(fig: go.Figure, rows: List[Tuple[str, str]]) -> None:
    panel_x0 = 0.805
    panel_x1 = 0.965
    panel_y1 = 0.915
    header_h = 0.033
    row_h = 0.050
    panel_h = header_h + len(rows) * row_h + 0.016
    panel_y0 = panel_y1 - panel_h

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path=rounded_rect_path(panel_x0, panel_y0, panel_x1, panel_y1, 0.012),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(255,255,255,0.74)",
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
        text="<b>Polar Information</b>",
        showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=11.1, color="#111827"),
    )

    current_top = panel_y1 - header_h - 0.008
    for title, value in rows:
        title_y = current_top - 0.003
        value_y = current_top - 0.026

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.026, y=title_y,
            xanchor="left", yanchor="top",
            text=f"<b>{title}</b>",
            showarrow=False, font=dict(size=10.2, color="#111827"), align="left",
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.026, y=value_y,
            xanchor="left", yanchor="top",
            text=value,
            showarrow=False, font=dict(size=9.9, color="#111827"), align="left",
        )

        current_top -= row_h


def build_info_rows(
    row_a: pd.Series,
    row_b: pd.Series,
    amp_unit: str,
    speed_unit: str,
    show_rpm_labels: bool,
    marker_stride: int,
) -> List[Tuple[str, str]]:
    return [
        ("Cursor A", f"{format_number(row_a['amp'],3)} {amp_unit} @ {int(round(row_a['speed']))} {speed_unit} | ∠{format_number(row_a['phase_plot'],1)}°"),
        ("Cursor B", f"{format_number(row_b['amp'],3)} {amp_unit} @ {int(round(row_b['speed']))} {speed_unit} | ∠{format_number(row_b['phase_plot'],1)}°"),
        ("RPM Labels", "Enabled" if show_rpm_labels else "Disabled"),
        ("Label Step", f"Every {marker_stride} points"),
    ]


# ============================================================
# FIGURE BUILD
# ============================================================
def build_polar_figure(
    df: pd.DataFrame,
    meta: Dict[str, str],
    row_a: pd.Series,
    row_b: pd.Series,
    logo_uri: Optional[str],
    show_info_box: bool,
    show_rpm_labels: bool,
    marker_stride: int,
) -> go.Figure:
    amp_unit = meta.get("Amp Unit", "") or ""
    speed_unit = meta.get("Speed Unit", "rpm") or "rpm"

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=df["amp"],
            theta=df["phase_plot"],
            mode="lines",
            line=dict(width=1.35, color="#5b9cf0"),
            hovertemplate=(
                f"Amplitude: %{{r:.3f}} {amp_unit}<br>"
                f"Phase: %{{theta:.1f}}°<br>"
                f"Speed: %{{customdata[0]:.0f}} {speed_unit}<extra></extra>"
            ),
            customdata=np.stack([df["speed"]], axis=1),
            showlegend=False,
            name="Polar Path",
        )
    )

    # Cursor A/B markers
    for row, color, name in [
        (row_a, "#efb08c", "Cursor A"),
        (row_b, "#7ac77b", "Cursor B"),
    ]:
        fig.add_trace(
            go.Scatterpolar(
                r=[row["amp"]],
                theta=[row["phase_plot"]],
                mode="markers",
                marker=dict(size=10, color=color, line=dict(width=1.2, color="#ffffff")),
                name=name,
                showlegend=False,
                hovertemplate=(
                    f"{name}<br>"
                    f"Amplitude: %{{r:.3f}} {amp_unit}<br>"
                    f"Phase: %{{theta:.1f}}°<br>"
                    f"Speed: {int(round(row['speed']))} {speed_unit}<extra></extra>"
                ),
            )
        )

    # Optional RPM labels
    if show_rpm_labels and len(df) > 0:
        idxs = list(range(0, len(df), max(1, marker_stride)))
        if idxs[-1] != len(df) - 1:
            idxs.append(len(df) - 1)

        fig.add_trace(
            go.Scatterpolar(
                r=df.iloc[idxs]["amp"],
                theta=df.iloc[idxs]["phase_plot"],
                mode="text",
                text=[str(int(round(v))) for v in df.iloc[idxs]["speed"]],
                textfont=dict(size=9, color="#6b7280"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    rows = build_info_rows(row_a, row_b, amp_unit, speed_unit, show_rpm_labels, marker_stride)

    # Increase right margin domain if info box visible
    domain_x = [0.0, 0.78] if show_info_box else [0.0, 1.0]

    fig.update_layout(
        polar=dict(
            domain=dict(x=domain_x, y=[0.05, 0.96]),
            bgcolor="#f8fafc",
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                tickfont=dict(size=12, color="#111827"),
                gridcolor="rgba(148, 163, 184, 0.18)",
                linecolor="#9ca3af",
                showline=True,
                ticks="outside",
            ),
            radialaxis=dict(
                tickfont=dict(size=11, color="#111827"),
                gridcolor="rgba(148, 163, 184, 0.18)",
                linecolor="#9ca3af",
                showline=True,
                ticks="outside",
                angle=225,
            ),
        ),
        height=820,
        margin=dict(l=48, r=20, t=145, b=48),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f3f4f6",
        font=dict(color="#111827"),
        showlegend=False,
    )

    _draw_top_strip(fig, meta, row_a, row_b, logo_uri, df)

    if show_info_box:
        _draw_right_info_box(fig, rows)

    return fig


# ============================================================
# EXPORT / REPORT
# ============================================================
def _build_export_safe_figure(fig: go.Figure) -> go.Figure:
    export_fig = go.Figure(fig.to_dict())
    return export_fig


def _scale_export_figure(export_fig: go.Figure) -> go.Figure:
    fig = go.Figure(export_fig)

    for trace in fig.data:
        tj = trace.to_plotly_json()
        if tj.get("mode") and "lines" in tj.get("mode", ""):
            line = dict(tj.get("line", {}) or {})
            line["width"] = max(3.6, float(line.get("width", 1.0)) * 2.25)
            trace.line = line
        if tj.get("mode") and "markers" in tj.get("mode", ""):
            marker = dict(tj.get("marker", {}) or {})
            marker["size"] = max(10, float(marker.get("size", 6)) * 1.6)
            trace.marker = marker

    fig.update_layout(
        width=4300,
        height=2200,
        margin=dict(l=110, r=60, t=320, b=110),
        paper_bgcolor="#f3f4f6",
        font=dict(size=27, color="#111827"),
    )

    for ann in fig.layout.annotations or []:
        if ann.font is not None:
            ann.font.size = max(20, int((ann.font.size or 12) * 1.8))

    for img in fig.layout.images or []:
        sx = getattr(img, "sizex", None)
        sy = getattr(img, "sizey", None)
        if sx is not None:
            img.sizex = sx * 1.12
        if sy is not None:
            img.sizey = sy * 1.12

    return fig


def build_export_png_bytes(fig: go.Figure) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        export_fig = _build_export_safe_figure(fig)
        export_fig = _scale_export_figure(export_fig)
        return export_fig.to_image(format="png", width=4300, height=2200, scale=2), None
    except Exception as e:
        return None, str(e)


def queue_polar_to_report(meta: Dict[str, str], fig: go.Figure, title: str) -> None:
    ensure_report_state()
    st.session_state.report_items.append(
        {
            "id": f"report-polar-{meta.get('Machine Name','')}-{meta.get('Point Name','')}-{title}",
            "type": "polar",
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
if "wm_polar_export_store" not in st.session_state:
    st.session_state.wm_polar_export_store = {}
ensure_report_state()

st.markdown('<div class="wm-page-title">Polar Plot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wm-page-subtitle">Dynamic polar trajectory from amplitude, phase and speed.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Upload Polar CSV")
    uploaded_file = st.file_uploader("Upload Polar CSV", type=["csv"], accept_multiple_files=False)

if not uploaded_file:
    st.info("Carga un archivo CSV Polar para visualizar trayectoria dinámica.")
    st.stop()

try:
    meta, raw_df, grouped_df = read_polar_csv(uploaded_file)
except Exception as e:
    st.error(f"No pude leer el CSV Polar: {e}")
    st.stop()

if grouped_df.empty:
    st.error("No hay datos válidos para construir el Polar.")
    st.stop()

with st.sidebar:
    st.markdown("### Polar Controls")
    smooth_window = st.slider("Circular phase smoothing", 1, 11, 3, step=2)
    amp_smooth_window = st.slider("Amplitude smoothing", 1, 11, 3, step=2)
    show_info_box = st.checkbox("Show Polar Information", value=True)
    show_rpm_labels = st.checkbox("Show RPM labels", value=True)
    marker_stride = st.slider("RPM label step", 10, 150, 45, step=5)

    st.markdown("### Cursors")
    speed_min = int(grouped_df["speed"].min())
    speed_max = int(grouped_df["speed"].max())
    cursor_a_speed = st.slider("Cursor A (RPM)", speed_min, speed_max, speed_min)
    cursor_b_speed = st.slider("Cursor B (RPM)", speed_min, speed_max, speed_max)

plot_df = grouped_df.copy()
plot_df["amp"] = smooth_series(plot_df["amp"], amp_smooth_window)
plot_df["phase_plot"] = circular_smooth_deg(plot_df["phase"], smooth_window) % 360.0

row_a = nearest_row_for_speed(plot_df, cursor_a_speed)
row_b = nearest_row_for_speed(plot_df, cursor_b_speed)

machine = meta.get("Machine Name", "-")
point = meta.get("Point Name", "-")
variable = meta.get("Variable", "-")
probe_angle = meta.get("Probe Angle", "-")
speed_unit = meta.get("Speed Unit", "rpm")
amp_unit = meta.get("Amp Unit", "")

st.markdown(
    f"""
    <div class="wm-card">
        <div class="wm-card-title">{machine} · {point}</div>
        <div class="wm-card-subtitle">Dynamic polar view</div>
        <div class="wm-meta">
            Variable: <b>{variable}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
            Probe Angle: <b>{probe_angle}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
            Speed Range: <b>{int(plot_df['speed'].min())} - {int(plot_df['speed'].max())} {speed_unit}</b>
        </div>
        <div class="wm-chip-row">
            <div class="wm-chip">Raw rows: {len(raw_df):,}</div>
            <div class="wm-chip">Grouped points: {len(plot_df):,}</div>
            <div class="wm-chip">Phase smoothing: {smooth_window}</div>
            <div class="wm-chip">Amplitude smoothing: {amp_smooth_window}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

logo_uri = get_logo_data_uri(LOGO_PATH)
fig = build_polar_figure(
    df=plot_df,
    meta=meta,
    row_a=row_a,
    row_b=row_b,
    logo_uri=logo_uri,
    show_info_box=show_info_box,
    show_rpm_labels=show_rpm_labels,
    marker_stride=marker_stride,
)

st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="wm_polar_plot")

title = f"Polar — {machine} — {point}"

export_state_key = (
    f"polar::{machine}::{point}::{variable}::{smooth_window}::{amp_smooth_window}::"
    f"{show_info_box}::{show_rpm_labels}::{marker_stride}"
)

if export_state_key not in st.session_state.wm_polar_export_store:
    st.session_state.wm_polar_export_store[export_state_key] = {"png_bytes": None, "error": None}

left_pad, col_export1, col_export2, col_report, right_pad = st.columns([2.0, 1.2, 1.2, 1.2, 2.0])

with col_export1:
    if st.button("Prepare PNG HD", key=f"prepare_polar_png_{export_state_key}", width="stretch"):
        with st.spinner("Generating HD export..."):
            png_bytes, export_error = build_export_png_bytes(fig)
            st.session_state.wm_polar_export_store[export_state_key]["png_bytes"] = png_bytes
            st.session_state.wm_polar_export_store[export_state_key]["error"] = export_error

with col_export2:
    png_bytes = st.session_state.wm_polar_export_store[export_state_key]["png_bytes"]
    if png_bytes is not None:
        st.download_button(
            "Download PNG HD",
            data=png_bytes,
            file_name=f"{Path(uploaded_file.name).stem}_polar_hd.png",
            mime="image/png",
            key=f"download_polar_png_{export_state_key}",
            width="stretch",
        )
    else:
        st.button("Download PNG HD", disabled=True, key=f"download_polar_disabled_{export_state_key}", width="stretch")

with col_report:
    if st.button("Enviar a Reporte", key=f"report_polar_{export_state_key}", width="stretch"):
        queue_polar_to_report(meta, fig, title)
        st.success("Polar enviado al reporte.")
