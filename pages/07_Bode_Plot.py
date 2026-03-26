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

from core.auth import require_login, render_user_menu


st.set_page_config(page_title="Watermelon System | Bode Plot", layout="wide")
require_login()
render_user_menu()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
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

        .wm-export-actions {
            margin-top: 0.85rem;
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_page_style()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Bode CSV loader
# ------------------------------------------------------------
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
            phase=("phase", "median"),
            samples=("Timestamp", "size"),
            ts_min=("Timestamp", "min"),
            ts_max=("Timestamp", "max"),
        )
        .sort_values("rpm", kind="stable")
        .reset_index(drop=True)
    )

    return meta, raw_df, grouped_df


# ------------------------------------------------------------
# Signal processing
# ------------------------------------------------------------
def smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series.copy()
    return series.rolling(window=window, center=True, min_periods=1).median()


def phase_unwrap_deg(phase_deg: pd.Series) -> pd.Series:
    rad = np.deg2rad(phase_deg.astype(float).to_numpy())
    unwrapped = np.unwrap(rad)
    return pd.Series(np.rad2deg(unwrapped), index=phase_deg.index)


def wrapped_phase_with_breaks(unwrapped_deg: pd.Series) -> pd.Series:
    wrapped = ((unwrapped_deg + 360.0) % 360.0).astype(float)
    out = wrapped.copy()
    if len(out) >= 2:
        jumps = np.abs(np.diff(out.to_numpy())) > 180.0
        for i, j in enumerate(jumps, start=1):
            if j:
                out.iloc[i] = np.nan
    return out


def nearest_row_for_rpm(df: pd.DataFrame, rpm_value: float) -> pd.Series:
    idx = int((df["rpm"] - rpm_value).abs().idxmin())
    return df.loc[idx]


def estimate_critical_speed_api684_style(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if df.empty or len(df) < 9:
        return None

    amp = df["amp"].astype(float)
    phase = df["phase_unwrapped"].astype(float)
    rpm = df["rpm"].astype(float)

    peak_idx = int(amp.idxmax())
    if peak_idx <= 2 or peak_idx >= len(df) - 3:
        return None

    left_idx = max(0, peak_idx - 3)
    right_idx = min(len(df) - 1, peak_idx + 3)

    phase_left = float(phase.iloc[left_idx])
    phase_right = float(phase.iloc[right_idx])
    phase_delta = phase_right - phase_left

    return {
        "rpm": float(rpm.iloc[peak_idx]),
        "amp": float(amp.iloc[peak_idx]),
        "phase": float(phase.iloc[peak_idx]),
        "phase_delta": float(phase_delta),
        "idx": peak_idx,
    }


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------
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
        showarrow=False, font=dict(size=12.8, color="#111827"), align="left",
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.205, y=y_text,
        xanchor="left", yanchor="middle",
        text=point,
        showarrow=False, font=dict(size=12.1, color="#111827"), align="left",
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.355, y=y_text,
        xanchor="left", yanchor="middle",
        text=f"{variable} | {angle}",
        showarrow=False, font=dict(size=12.0, color="#111827"), align="left",
    )

    a_txt = (
        f"A: <b>{format_number(row_a['amp'], 3)} {y_unit}</b> "
        f"∠{format_number(row_a['phase_header'], 1)}° @ {int(round(row_a['rpm']))} {x_unit}"
    )
    b_txt = (
        f"B: <b>{format_number(row_b['amp'], 3)} {y_unit}</b> "
        f"∠{format_number(row_b['phase_header'], 1)}° @ {int(round(row_b['rpm']))} {x_unit}"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.630, y=y_text,
        xanchor="left", yanchor="middle",
        text=f"{a_txt} &nbsp;&nbsp;|&nbsp;&nbsp; {b_txt}",
        showarrow=False, font=dict(size=11.3, color="#111827"), align="left",
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.986, y=y_text,
        xanchor="right", yanchor="middle",
        text=f"{int(row_a['rpm']) if pd.notna(row_a['rpm']) else '—'} - {int(row_b['rpm']) if pd.notna(row_b['rpm']) else '—'} {x_unit}",
        showarrow=False, font=dict(size=11.2, color="#111827"), align="right",
    )


def _draw_right_info_box(fig: go.Figure, rows: List[Tuple[str, str]]) -> None:
    panel_x0 = 0.836
    panel_x1 = 0.975
    panel_y1 = 0.915
    header_h = 0.034
    row_h = 0.058
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
        showarrow=False, xanchor="center", yanchor="middle",
        font=dict(size=11.4, color="#111827"),
    )

    current_top = panel_y1 - header_h - 0.008
    for title, value in rows:
        title_y = current_top - 0.004
        value_y = current_top - 0.030

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.030, y=title_y,
            xanchor="left", yanchor="top",
            text=f"<b>{title}</b>",
            showarrow=False, font=dict(size=10.7, color="#111827"), align="left",
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.030, y=value_y,
            xanchor="left", yanchor="top",
            text=value,
            showarrow=False, font=dict(size=10.4, color="#111827"), align="left",
        )

        current_top -= row_h


def build_bode_rows(
    row_a: pd.Series,
    row_b: pd.Series,
    critical_speed_result: Optional[Dict[str, float]],
    phase_mode: str,
    y_unit: str,
    x_unit: str,
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = [
        ("Cursor A", f"{format_number(row_a['amp'],3)} {y_unit} @ {int(round(row_a['rpm']))} {x_unit} | ∠{format_number(row_a['phase_header'],1)}°"),
        ("Cursor B", f"{format_number(row_b['amp'],3)} {y_unit} @ {int(round(row_b['rpm']))} {x_unit} | ∠{format_number(row_b['phase_header'],1)}°"),
        ("Phase Mode", phase_mode),
    ]

    if critical_speed_result is not None:
        rows.extend(
            [
                ("Critical Speed", f"{int(round(critical_speed_result['rpm']))} {x_unit}"),
                ("Peak Amplitude", f"{format_number(critical_speed_result['amp'],3)} {y_unit}"),
                ("Phase Delta", f"{format_number(critical_speed_result['phase_delta'],1)}°"),
            ]
        )

    return rows


def build_bode_figure(
    df: pd.DataFrame,
    meta: Dict[str, str],
    row_a: pd.Series,
    row_b: pd.Series,
    x_min: float,
    x_max: float,
    logo_uri: Optional[str],
    phase_mode: str,
    critical_speed_result: Optional[Dict[str, float]],
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

    phase_xaxis_name = "x"
    amp_xaxis_name = "x2"

    fig.add_trace(
        go.Scattergl(
            x=df["rpm"],
            y=df["phase_plot"],
            mode="lines",
            line=dict(width=1.25, color="#5b9cf0"),
            name="Phase",
            hovertemplate=f"Speed: %{{x:.0f}} {x_unit}<br>Phase: %{{y:.1f}}°<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df["rpm"],
            y=df["amp"],
            mode="lines",
            line=dict(width=1.45, color="#5b9cf0"),
            name="Amplitude",
            hovertemplate=f"Speed: %{{x:.0f}} {x_unit}<br>Amplitude: %{{y:.3f}} {y_unit}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    for rpm_val, phase_val, amp_val, color in [
        (float(row_a["rpm"]), float(row_a["phase_plot"]), float(row_a["amp"]), "#efb08c"),
        (float(row_b["rpm"]), float(row_b["phase_plot"]), float(row_b["amp"]), "#7ac77b"),
    ]:
        fig.add_vline(x=rpm_val, line_width=1.5, line_dash="dot", line_color=color, row=1, col=1)
        fig.add_vline(x=rpm_val, line_width=1.5, line_dash="dot", line_color=color, row=2, col=1)

    if critical_speed_result is not None:
        cs_rpm = critical_speed_result["rpm"]
        cs_amp = critical_speed_result["amp"]
        cs_phase = float(nearest_row_for_rpm(df, cs_rpm)["phase_plot"])

        fig.add_vline(x=cs_rpm, line_width=2.0, line_dash="dash", line_color="#ef4444", row=1, col=1)
        fig.add_vline(x=cs_rpm, line_width=2.0, line_dash="dash", line_color="#ef4444", row=2, col=1)

        fig.add_annotation(
            x=cs_rpm, y=cs_phase,
            xref=phase_xaxis_name, yref="y",
            text=f"Estimated Critical Speed<br>{int(round(cs_rpm))} rpm",
            showarrow=True, arrowhead=2, arrowcolor="#ef4444",
            ax=55, ay=-40,
            font=dict(size=11, color="#991b1b"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#fecaca",
        )

        fig.add_annotation(
            x=cs_rpm, y=cs_amp,
            xref=amp_xaxis_name, yref="y2",
            text=f"{format_number(cs_amp,3)} {y_unit}",
            showarrow=True, arrowhead=2, arrowcolor="#ef4444",
            ax=45, ay=-35,
            font=dict(size=11, color="#991b1b"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#fecaca",
        )

    _draw_top_strip(fig, meta, row_a, row_b, logo_uri)

    if show_info_box:
        rows = build_bode_rows(row_a, row_b, critical_speed_result, phase_mode, y_unit, x_unit)
        _draw_right_info_box(fig, rows)

    x_domain = [0.0, 0.81] if show_info_box else [0.0, 1.0]

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

    phase_title = "Phase (°)" if phase_mode == "Wrapped 0-360" else "Phase Unwrapped (°)"
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


# ------------------------------------------------------------
# Export
# ------------------------------------------------------------
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
                )
            )
        else:
            export_fig.add_trace(trace)

    export_fig.update_layout(fig.layout)
    return export_fig


def _scale_export_figure(export_fig: go.Figure) -> go.Figure:
    fig = go.Figure(export_fig)

    new_data = []
    for trace in fig.data:
        tj = trace.to_plotly_json()
        if tj.get("type") == "scatter":
            mode = tj.get("mode", "")
            if "lines" in mode:
                line = dict(tj.get("line", {}) or {})
                line["width"] = max(4.2, float(line.get("width", 1.0)) * 2.5)
                tj["line"] = line
            if "markers" in mode:
                marker = dict(tj.get("marker", {}) or {})
                marker["size"] = max(12, float(marker.get("size", 6)) * 1.8)
                tj["marker"] = marker
        new_data.append(go.Scatter(**tj))

    fig = go.Figure(data=new_data, layout=fig.layout)

    fig.update_layout(
        width=4200,
        height=2200,
        margin=dict(l=110, r=80, t=330, b=110),
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
            ann.font.size = max(21, int((ann.font.size or 12) * 1.95))

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
        png_bytes = export_fig.to_image(format="png", width=4200, height=2200, scale=2)
        return png_bytes, None
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


# ------------------------------------------------------------
# Session defaults
# ------------------------------------------------------------
if "wm_bode_export_store" not in st.session_state:
    st.session_state.wm_bode_export_store = {}
ensure_report_state()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
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
    phase_mode = st.selectbox("Phase display", ["Wrapped 0-360", "Unwrapped"], index=0)

    st.markdown("### Smoothing")
    smooth_window = st.slider("Median smoothing window", 1, 21, 3, step=2)

    st.markdown("### Critical Speed")
    detect_cs = st.checkbox("Estimate critical speed (API-684 heuristic)", value=True)

    st.markdown("### Information Box")
    show_info_box = st.checkbox("Show Bode Information", value=True)

    st.markdown("### Cursors")
    cursor_a_rpm = st.slider("Cursor A (RPM)", int(x_min_default), int(x_max_default), int(grouped_df["rpm"].iloc[0]))
    cursor_b_rpm = st.slider("Cursor B (RPM)", int(x_min_default), int(x_max_default), int(grouped_df["rpm"].iloc[-1]))

plot_df = grouped_df.copy()
plot_df["amp"] = smooth_series(plot_df["amp"], smooth_window)
plot_df["phase_unwrapped"] = smooth_series(phase_unwrap_deg(plot_df["phase"]), smooth_window)

if phase_mode == "Wrapped 0-360":
    plot_df["phase_plot"] = wrapped_phase_with_breaks(plot_df["phase_unwrapped"])
    plot_df["phase_header"] = plot_df["phase_unwrapped"] % 360.0
else:
    plot_df["phase_plot"] = plot_df["phase_unwrapped"]
    plot_df["phase_header"] = plot_df["phase_unwrapped"]

row_a = nearest_row_for_rpm(plot_df, cursor_a_rpm)
row_b = nearest_row_for_rpm(plot_df, cursor_b_rpm)

critical_speed_result = estimate_critical_speed_api684_style(plot_df) if detect_cs else None

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
    critical_speed_result=critical_speed_result,
    show_info_box=show_info_box,
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displaylogo": False},
    key="wm_bode_plot",
)

title = f"Bode — {machine} — {point}"

export_state_key = f"bode::{machine}::{point}::{variable}::{phase_mode}::{x_min}::{x_max}::{smooth_window}::{detect_cs}::{show_info_box}"

if export_state_key not in st.session_state.wm_bode_export_store:
    st.session_state.wm_bode_export_store[export_state_key] = {"png_bytes": None, "error": None}

st.markdown('<div class="wm-export-actions"></div>', unsafe_allow_html=True)
left_pad, col_export1, col_export2, col_report, right_pad = st.columns([2.0, 1.2, 1.2, 1.2, 2.0])

with col_export1:
    if st.button("Prepare PNG HD", key=f"prepare_bode_png_{export_state_key}", use_container_width=True):
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
            use_container_width=True,
        )
    else:
        st.button(
            "Download PNG HD",
            disabled=True,
            key=f"download_bode_disabled_{export_state_key}",
            use_container_width=True,
        )

with col_report:
    if st.button("Enviar a Reporte", key=f"report_bode_{export_state_key}", use_container_width=True):
        queue_bode_to_report(meta, fig, title)
        st.success("Bode enviado al reporte")

panel_error = st.session_state.wm_bode_export_store[export_state_key]["error"]
if panel_error:
    st.warning(f"PNG export error: {panel_error}")

with st.expander("Grouped Data", expanded=False):
    st.dataframe(plot_df, use_container_width=True, hide_index=True)
