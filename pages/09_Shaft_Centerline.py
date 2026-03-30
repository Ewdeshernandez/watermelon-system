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


st.set_page_config(page_title="Watermelon System | Shaft Centerline", layout="wide")
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


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window is None or window < 2:
        return series.astype(float).copy()
    return series.astype(float).rolling(window=window, center=True, min_periods=1).mean()


def nearest_row_for_speed(df: pd.DataFrame, speed_value: float) -> pd.Series:
    idx = int((df["speed"] - speed_value).abs().idxmin())
    return df.loc[idx]


def parse_probe_angle_text(text: str) -> Tuple[float, str]:
    text = str(text or "").strip()
    angle = 0.0
    side = ""
    if not text:
        return angle, side

    import re
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    if m:
        try:
            angle = float(m.group(1))
        except Exception:
            angle = 0.0

    low = text.lower()
    if "left" in low:
        side = "Left"
    elif "right" in low:
        side = "Right"

    return angle, side


# ============================================================
# CSV LOADER
# ============================================================
def read_scl_csv(file_obj) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame]:
    file_obj.seek(0)
    raw_bytes = file_obj.read()
    text = raw_bytes.decode("utf-8-sig", errors="replace") if isinstance(raw_bytes, bytes) else str(raw_bytes)

    lines = text.splitlines()
    if not lines:
        raise ValueError("Archivo vacío.")

    header_idx = None
    for i, line in enumerate(lines):
        if (
            "Point Value" in line
            and "Paired Point Value" in line
            and "Speed" in line
            and "Timestamp" in line
        ):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("No se encontró el encabezado real del CSV Shaft Centerline.")

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
        "Point Value",
        "Value Status",
        "Paired Point Value",
        "Paired Value Status",
        "Speed",
        "Speed Status",
        "Timestamp",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    df["Point Value"] = pd.to_numeric(df["Point Value"], errors="coerce")
    df["Paired Point Value"] = pd.to_numeric(df["Paired Point Value"], errors="coerce")
    df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.dropna(subset=["Point Value", "Paired Point Value", "Speed", "Timestamp"]).copy()
    df = df[
        df["Value Status"].astype(str).str.strip().str.lower().eq("valid")
        & df["Paired Value Status"].astype(str).str.strip().str.lower().eq("valid")
        & df["Speed Status"].astype(str).str.strip().str.lower().eq("valid")
    ].copy()

    if df.empty:
        raise ValueError("No quedaron filas válidas después del filtrado.")

    raw_df = df.sort_values(["Speed", "Timestamp"], kind="stable").reset_index(drop=True)

    grouped_df = (
        raw_df.groupby("Speed", as_index=False)
        .agg(
            y_gap=("Point Value", "median"),
            x_gap=("Paired Point Value", "median"),
            samples=("Timestamp", "size"),
            ts_min=("Timestamp", "min"),
            ts_max=("Timestamp", "max"),
        )
        .sort_values("Speed", kind="stable")
        .reset_index(drop=True)
        .rename(columns={"Speed": "speed"})
    )

    return meta, raw_df, grouped_df


# ============================================================
# MULTI-FILE LOADER
# ============================================================
def uploaded_file_label(file_obj) -> str:
    return Path(getattr(file_obj, "name", "Shaft_Centerline.csv")).name


def uploaded_file_stem(file_obj) -> str:
    return Path(getattr(file_obj, "name", "Shaft_Centerline.csv")).stem


def parse_uploaded_scl_files(files: List[Any]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    parsed_items: List[Dict[str, Any]] = []
    failed_items: List[Tuple[str, str]] = []

    for file_obj in files:
        try:
            meta, raw_df, grouped_df = read_scl_csv(file_obj)
            label = uploaded_file_label(file_obj)
            machine = meta.get("Machine Name", "-")
            point = meta.get("Point Name", label)
            paired_point = meta.get("Paired Point Name", "-")
            item_id = f"{label}::{machine}::{point}::{paired_point}"

            parsed_items.append(
                {
                    "id": item_id,
                    "label": label,
                    "file_name": label,
                    "file_stem": uploaded_file_stem(file_obj),
                    "meta": meta,
                    "raw_df": raw_df,
                    "grouped_df": grouped_df,
                    "machine": machine,
                    "point": point,
                    "paired_point": paired_point,
                    "variable": meta.get("Variable", "-"),
                }
            )
        except Exception as e:
            failed_items.append((uploaded_file_label(file_obj), str(e)))

    return parsed_items, failed_items


# ============================================================
# DECORATION
# ============================================================
def _draw_top_strip(
    fig: go.Figure,
    meta: Dict[str, str],
    df: pd.DataFrame,
    logo_uri: Optional[str],
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
    paired_point = meta.get("Paired Point Name", "")
    variable = meta.get("Variable", "")
    speed_unit = meta.get("Speed Unit", "rpm") or "rpm"

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
        text=f"{point} / {paired_point}",
        showarrow=False,
        font=dict(size=11.7, color="#111827"),
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.43, y=y_text,
        xanchor="left", yanchor="middle",
        text=variable,
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
        x=0.66, y=y_text,
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
    row_h = 0.054
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
        text="<b>Shaft Centerline</b>",
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


# ============================================================
# FIGURE
# ============================================================
def build_scl_figure(
    df: pd.DataFrame,
    meta: Dict[str, str],
    row_a: pd.Series,
    row_b: pd.Series,
    logo_uri: Optional[str],
    show_info_box: bool,
    show_rpm_labels: bool,
    marker_stride: int,
    show_reference_circle: bool,
    normalize_to_origin: bool,
) -> go.Figure:
    gap_unit = meta.get("Gap Unit", "").strip() or "mil"
    speed_unit = meta.get("Speed Unit", "rpm").strip() or "rpm"

    plot_df = df.copy()

    if normalize_to_origin:
        x0 = float(plot_df["x_gap"].iloc[0])
        y0 = float(plot_df["y_gap"].iloc[0])
        plot_df["x_plot"] = plot_df["x_gap"] - x0
        plot_df["y_plot"] = plot_df["y_gap"] - y0
        row_a_x = float(row_a["x_gap"] - x0)
        row_a_y = float(row_a["y_gap"] - y0)
        row_b_x = float(row_b["x_gap"] - x0)
        row_b_y = float(row_b["y_gap"] - y0)
    else:
        plot_df["x_plot"] = plot_df["x_gap"]
        plot_df["y_plot"] = plot_df["y_gap"]
        row_a_x = float(row_a["x_gap"])
        row_a_y = float(row_a["y_gap"])
        row_b_x = float(row_b["x_gap"])
        row_b_y = float(row_b["y_gap"])

    x = plot_df["x_plot"].to_numpy(dtype=float)
    y = plot_df["y_plot"].to_numpy(dtype=float)

    max_abs = max(
        float(np.nanmax(np.abs(x))) if len(x) else 0.0,
        float(np.nanmax(np.abs(y))) if len(y) else 0.0,
        0.1,
    )
    lim = max_abs * 1.20

    fig = go.Figure()

    if show_reference_circle:
        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        r = max_abs if max_abs > 0 else 1.0
        fig.add_trace(
            go.Scatter(
                x=r * np.cos(theta),
                y=r * np.sin(theta),
                mode="lines",
                line=dict(color="rgba(17,24,39,0.45)", width=2, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(width=2.0, color="#5b9cf0"),
            marker=dict(
                size=6,
                color=plot_df["speed"],
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title=speed_unit, thickness=14, len=0.75, y=0.5),
                line=dict(width=0.5, color="rgba(255,255,255,0.35)"),
            ),
            customdata=np.stack([plot_df["speed"]], axis=1),
            hovertemplate=(
                f"X: %{{x:.3f}} {gap_unit}<br>"
                f"Y: %{{y:.3f}} {gap_unit}<br>"
                f"Speed: %{{customdata[0]:.0f}} {speed_unit}<extra></extra>"
            ),
            showlegend=False,
            name="Centerline",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[x[0]],
            y=[y[0]],
            mode="markers+text",
            marker=dict(size=11, color="#22c55e", symbol="diamond"),
            text=["START"],
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[x[-1]],
            y=[y[-1]],
            mode="markers+text",
            marker=dict(size=11, color="#ef4444", symbol="diamond"),
            text=["END"],
            textposition="bottom center",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[row_a_x],
            y=[row_a_y],
            mode="markers",
            marker=dict(size=10, color="#efb08c", line=dict(width=1.2, color="#ffffff")),
            showlegend=False,
            hovertemplate=(
                f"Cursor A<br>X: {row_a_x:.3f} {gap_unit}<br>Y: {row_a_y:.3f} {gap_unit}<br>"
                f"Speed: {int(round(row_a['speed']))} {speed_unit}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[row_b_x],
            y=[row_b_y],
            mode="markers",
            marker=dict(size=10, color="#7ac77b", line=dict(width=1.2, color="#ffffff")),
            showlegend=False,
            hovertemplate=(
                f"Cursor B<br>X: {row_b_x:.3f} {gap_unit}<br>Y: {row_b_y:.3f} {gap_unit}<br>"
                f"Speed: {int(round(row_b['speed']))} {speed_unit}<extra></extra>"
            ),
        )
    )

    if show_rpm_labels and len(plot_df) > 0:
        idxs = list(range(0, len(plot_df), max(1, marker_stride)))
        if idxs[-1] != len(plot_df) - 1:
            idxs.append(len(plot_df) - 1)

        fig.add_trace(
            go.Scatter(
                x=plot_df.iloc[idxs]["x_plot"],
                y=plot_df.iloc[idxs]["y_plot"],
                mode="text",
                text=[str(int(round(v))) for v in plot_df.iloc[idxs]["speed"]],
                textfont=dict(size=10, color="#6b7280"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    rows = [
        ("Cursor A", f"X={format_number(row_a_x,3)} / Y={format_number(row_a_y,3)} {gap_unit} @ {int(round(row_a['speed']))} {speed_unit}"),
        ("Cursor B", f"X={format_number(row_b_x,3)} / Y={format_number(row_b_y,3)} {gap_unit} @ {int(round(row_b['speed']))} {speed_unit}"),
        ("Probe Pair", f"{meta.get('Paired Point Name', '-')} / {meta.get('Point Name', '-')}"),
        ("Probe Angles", f"{meta.get('Paired Probe Angle', '-')} / {meta.get('Probe Angle', '-')}"),
        ("Normalize", "Enabled" if normalize_to_origin else "Disabled"),
        ("RPM Labels", "Enabled" if show_rpm_labels else "Disabled"),
    ]

    fig.update_layout(
        height=820,
        margin=dict(l=60, r=20, t=145, b=60),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f3f4f6",
        font=dict(color="#111827"),
        showlegend=False,
    )

    fig.update_xaxes(
        title_text=f"{meta.get('Paired Point Name', 'X')} ({gap_unit})",
        range=[-lim, lim],
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=True,
        zerolinecolor="rgba(148,163,184,0.35)",
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        scaleanchor="y",
        scaleratio=1,
    )

    fig.update_yaxes(
        title_text=f"{meta.get('Point Name', 'Y')} ({gap_unit})",
        range=[-lim, lim],
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=True,
        zerolinecolor="rgba(148,163,184,0.35)",
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
    )

    _draw_top_strip(fig=fig, meta=meta, df=df, logo_uri=logo_uri)

    if show_info_box:
        _draw_right_info_box(fig, rows)

    return fig


# ============================================================
# EXPORT / REPORT
# ============================================================
def _build_export_safe_figure(fig: go.Figure) -> go.Figure:
    return go.Figure(fig.to_dict())


def _scale_export_figure(export_fig: go.Figure) -> go.Figure:
    fig = go.Figure(export_fig)

    for trace in fig.data:
        tj = trace.to_plotly_json()
        mode = tj.get("mode", "") or ""

        if "lines" in mode:
            line = dict(tj.get("line", {}) or {})
            line["width"] = max(3.2, float(line.get("width", 1.0)) * 2.0)
            trace.line = line

        if "markers" in mode:
            marker = dict(tj.get("marker", {}) or {})
            marker["size"] = max(10, float(marker.get("size", 6)) * 1.5)
            trace.marker = marker

        if "text" in mode:
            textfont = dict(tj.get("textfont", {}) or {})
            textfont["size"] = max(16, int(float(textfont.get("size", 10)) * 1.8))
            trace.textfont = textfont

    fig.update_layout(
        width=4300,
        height=2400,
        margin=dict(l=120, r=80, t=320, b=120),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        font=dict(size=26, color="#111827"),
    )

    for ann in fig.layout.annotations or []:
        if ann.font is not None:
            ann.font.size = max(20, int((ann.font.size or 12) * 1.75))

    for img in fig.layout.images or []:
        sx = getattr(img, "sizex", None)
        sy = getattr(img, "sizey", None)
        if sx is not None:
            img.sizex = sx * 1.10
        if sy is not None:
            img.sizey = sy * 1.10

    return fig


def build_export_png_bytes(fig: go.Figure) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        export_fig = _build_export_safe_figure(fig)
        export_fig = _scale_export_figure(export_fig)
        return export_fig.to_image(format="png", width=4300, height=2200, scale=2), None
    except Exception as e:
        return None, str(e)


def queue_scl_to_report(meta: Dict[str, str], fig: go.Figure, title: str) -> None:
    ensure_report_state()
    st.session_state.report_items.append(
        {
            "id": f"report-scl-{meta.get('Machine Name','')}-{meta.get('Point Name','')}-{title}",
            "type": "shaft_centerline",
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
# PANEL RENDER
# ============================================================
def render_scl_panel(
    item: Dict[str, Any],
    panel_index: int,
    *,
    logo_uri: Optional[str],
    smooth_window: int,
    show_info_box: bool,
    show_rpm_labels: bool,
    marker_stride: int,
    show_reference_circle: bool,
    normalize_to_origin: bool,
) -> None:
    meta = item["meta"]
    raw_df = item["raw_df"]
    grouped_df = item["grouped_df"]

    plot_df = grouped_df.copy()
    plot_df["x_gap"] = smooth_series(plot_df["x_gap"], smooth_window)
    plot_df["y_gap"] = smooth_series(plot_df["y_gap"], smooth_window)

    speed_min = int(plot_df["speed"].min())
    speed_max = int(plot_df["speed"].max())

    c1, c2 = st.columns(2)
    with c1:
        cursor_a_speed = st.slider(
            f"Cursor A (RPM) · Panel {panel_index + 1}",
            speed_min,
            speed_max,
            speed_min,
            key=f"scl_cursor_a_{panel_index}_{item['id']}",
        )
    with c2:
        cursor_b_speed = st.slider(
            f"Cursor B (RPM) · Panel {panel_index + 1}",
            speed_min,
            speed_max,
            speed_max,
            key=f"scl_cursor_b_{panel_index}_{item['id']}",
        )

    row_a = nearest_row_for_speed(plot_df, cursor_a_speed)
    row_b = nearest_row_for_speed(plot_df, cursor_b_speed)

    machine = meta.get("Machine Name", "-")
    point = meta.get("Point Name", "-")
    paired_point = meta.get("Paired Point Name", "-")
    variable = meta.get("Variable", "-")
    speed_unit = meta.get("Speed Unit", "rpm")
    gap_unit = meta.get("Gap Unit", "mil")

    probe_angle, probe_side = parse_probe_angle_text(meta.get("Probe Angle", ""))
    paired_angle, paired_side = parse_probe_angle_text(meta.get("Paired Probe Angle", ""))

    st.markdown(
        f"""
        <div class="wm-card">
            <div class="wm-card-title">Shaft Centerline {panel_index + 1} · {machine}</div>
            <div class="wm-card-subtitle">{point} / {paired_point}</div>
            <div class="wm-meta">
                Variable: <b>{variable}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Probe Angles: <b>{probe_angle:.0f}° {probe_side}</b> / <b>{paired_angle:.0f}° {paired_side}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Speed Range: <b>{int(plot_df['speed'].min())} - {int(plot_df['speed'].max())} {speed_unit}</b>
            </div>
            <div class="wm-chip-row">
                <div class="wm-chip">File: {item["file_name"]}</div>
                <div class="wm-chip">Raw rows: {len(raw_df):,}</div>
                <div class="wm-chip">Grouped points: {len(plot_df):,}</div>
                <div class="wm-chip">Gap Unit: {gap_unit}</div>
                <div class="wm-chip">Smoothing: {smooth_window}</div>
                <div class="wm-chip">Normalize: {"Yes" if normalize_to_origin else "No"}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = build_scl_figure(
        df=plot_df,
        meta=meta,
        row_a=row_a,
        row_b=row_b,
        logo_uri=logo_uri,
        show_info_box=show_info_box,
        show_rpm_labels=show_rpm_labels,
        marker_stride=marker_stride,
        show_reference_circle=show_reference_circle,
        normalize_to_origin=normalize_to_origin,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False},
        key=f"wm_scl_plot_{panel_index}_{item['id']}",
    )

    title = f"Shaft Centerline {panel_index + 1} — {machine} — {point} / {paired_point}"

    export_state_key = (
        f"scl::{item['id']}::{panel_index}::{smooth_window}::{show_info_box}::{show_rpm_labels}::"
        f"{marker_stride}::{show_reference_circle}::{normalize_to_origin}::{cursor_a_speed}::{cursor_b_speed}"
    )

    if "wm_scl_export_store" not in st.session_state:
        st.session_state["wm_scl_export_store"] = {}

    if export_state_key not in st.session_state["wm_scl_export_store"]:
        st.session_state["wm_scl_export_store"][export_state_key] = {"png_bytes": None, "error": None}

    left_pad, col_export1, col_export2, col_report, right_pad = st.columns([2.0, 1.2, 1.2, 1.2, 2.0])

    with col_export1:
        if st.button("Prepare PNG HD", key=f"prepare_scl_png_{export_state_key}", width="stretch"):
            with st.spinner("Generating HD export..."):
                png_bytes, export_error = build_export_png_bytes(fig)
                st.session_state["wm_scl_export_store"][export_state_key]["png_bytes"] = png_bytes
                st.session_state["wm_scl_export_store"][export_state_key]["error"] = export_error

    with col_export2:
        png_bytes = st.session_state["wm_scl_export_store"][export_state_key]["png_bytes"]
        if png_bytes is not None:
            st.download_button(
                "Download PNG HD",
                data=png_bytes,
                file_name=f"{item['file_stem']}_shaft_centerline_hd.png",
                mime="image/png",
                key=f"download_scl_png_{export_state_key}",
                width="stretch",
            )
        else:
            st.button(
                "Download PNG HD",
                disabled=True,
                key=f"download_scl_disabled_{export_state_key}",
                width="stretch",
            )

    with col_report:
        if st.button("Enviar a Reporte", key=f"report_scl_{export_state_key}", width="stretch"):
            queue_scl_to_report(meta, fig, title)
            st.success("Shaft Centerline enviado al reporte.")


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_report_state()

    st.markdown('<div class="wm-page-title">Shaft Centerline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="wm-page-subtitle">Centerline position from paired X/Y gap probes versus speed.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Shaft Centerline input")
        uploaded_files = st.file_uploader(
            "Upload one or more Shaft Centerline CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )

        st.markdown("### Controls")
        smooth_window = st.slider("Gap smoothing", 1, 11, 3, step=2)
        show_info_box = st.checkbox("Show information box", value=True)
        show_rpm_labels = st.checkbox("Show RPM labels", value=True)
        marker_stride = st.slider("RPM label step", 10, 150, 45, step=5)
        show_reference_circle = st.checkbox("Show reference circle", value=True)
        normalize_to_origin = st.checkbox("Normalize to first point", value=False)

    if not uploaded_files:
        st.info("Carga uno o varios archivos CSV de Shaft Centerline para visualizar la posición del eje.")
        return

    parsed_items, failed_items = parse_uploaded_scl_files(uploaded_files)

    if failed_items:
        for file_name, error_text in failed_items:
            st.warning(f"No pude leer {file_name}: {error_text}")

    if not parsed_items:
        st.error("No se pudo cargar ningún archivo válido de Shaft Centerline.")
        return

    logo_uri = get_logo_data_uri(LOGO_PATH)

    for panel_index, item in enumerate(parsed_items):
        render_scl_panel(
            item=item,
            panel_index=panel_index,
            logo_uri=logo_uri,
            smooth_window=smooth_window,
            show_info_box=show_info_box,
            show_rpm_labels=show_rpm_labels,
            marker_stride=marker_stride,
            show_reference_circle=show_reference_circle,
            normalize_to_origin=normalize_to_origin,
        )

        if panel_index < len(parsed_items) - 1:
            st.markdown("---")


if __name__ == "__main__":
    main()
