import base64
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.auth import render_user_menu, require_login


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Shaft Centerline", layout="wide")
LOGO_PATH = Path("assets/watermelon_logo.png")


# ============================================================
# STYLE
# ============================================================
def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.15rem;
            padding-bottom: 2.0rem;
            max-width: 100%;
        }

        .wm-page-title {
            font-size: 2.15rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.15rem;
            letter-spacing: -0.02em;
        }

        .wm-page-subtitle {
            color: #64748b;
            font-size: 0.98rem;
            margin-bottom: 1.15rem;
        }

        .wm-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.88));
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


def compute_xy_ranges(
    x: np.ndarray,
    y: np.ndarray,
    auto_scale_xy: bool,
    manual_x_min: float,
    manual_x_max: float,
    manual_y_min: float,
    manual_y_max: float,
) -> Tuple[List[float], List[float]]:
    if auto_scale_xy:
        x_span = max(float(np.nanmax(np.abs(x))) if len(x) else 0.0, 0.1) * 1.20
        y_span = max(float(np.nanmax(np.abs(y))) if len(y) else 0.0, 0.1) * 1.20
        return [-x_span, x_span], [-y_span, y_span]

    x_lo = min(float(manual_x_min), float(manual_x_max))
    x_hi = max(float(manual_x_min), float(manual_x_max))
    y_lo = min(float(manual_y_min), float(manual_y_max))
    y_hi = max(float(manual_y_min), float(manual_y_max))

    if math.isclose(x_lo, x_hi):
        x_hi = x_lo + 1.0
    if math.isclose(y_lo, y_hi):
        y_hi = y_lo + 1.0

    return [x_lo, x_hi], [y_lo, y_hi]


def resolve_clearance_boundary(
    x: np.ndarray,
    y: np.ndarray,
    mode: str,
    center_mode: str,
    manual_cx: float,
    manual_cy: float,
    manual_center_x: float,
    manual_center_y: float,
) -> Dict[str, float]:
    if center_mode == "Origin (0,0)":
        cx0 = 0.0
        cy0 = 0.0
    elif center_mode == "Data Mean":
        cx0 = float(np.nanmean(x)) if len(x) else 0.0
        cy0 = float(np.nanmean(y)) if len(y) else 0.0
    else:
        cx0 = float(manual_center_x)
        cy0 = float(manual_center_y)

    x_rel = x - cx0
    y_rel = y - cy0

    if mode == "Auto":
        cx = max(float(np.nanmax(np.abs(x_rel))) if len(x_rel) else 0.0, 0.1) * 1.08
        cy = max(float(np.nanmax(np.abs(y_rel))) if len(y_rel) else 0.0, 0.1) * 1.08
    else:
        cx = max(abs(float(manual_cx)), 0.001)
        cy = max(abs(float(manual_cy)), 0.001)

    return {
        "center_x": cx0,
        "center_y": cy0,
        "clearance_x": cx,
        "clearance_y": cy,
    }


def boundary_utilization_pct(
    px: float,
    py: float,
    center_x: float,
    center_y: float,
    clearance_x: float,
    clearance_y: float,
) -> float:
    nx = (px - center_x) / max(clearance_x, 1e-9)
    ny = (py - center_y) / max(clearance_y, 1e-9)
    return math.sqrt(nx * nx + ny * ny) * 100.0


def remaining_margin_pct(util_pct: float) -> float:
    return max(0.0, 100.0 - util_pct)


def build_boundary_curve(center_x: float, center_y: float, clearance_x: float, clearance_y: float) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    bx = center_x + clearance_x * np.cos(theta)
    by = center_y + clearance_y * np.sin(theta)
    return bx, by


def get_clearance_status(util_pct: float) -> Tuple[str, str]:
    if util_pct < 60.0:
        return "SAFE", "#16a34a"
    if util_pct < 85.0:
        return "WARNING", "#f59e0b"
    return "DANGER", "#dc2626"


def detect_early_rub(
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    center_x: float,
    center_y: float,
    clearance_x: float,
    clearance_y: float,
    warning_util_pct: float,
    danger_util_pct: float,
) -> Dict[str, Any]:
    if len(x) < 3:
        return {
            "triggered": False,
            "severity": "SAFE",
            "color": "#16a34a",
            "message": "Insufficient points",
            "max_util_pct": 0.0,
            "contact_points": 0,
            "warning_points": 0,
            "trend_score": 0.0,
            "first_warning_speed": None,
            "first_danger_speed": None,
        }

    utils = []
    for px, py in zip(x, y):
        util = boundary_utilization_pct(px, py, center_x, center_y, clearance_x, clearance_y)
        utils.append(util)
    utils = np.array(utils, dtype=float)

    warning_mask = utils >= warning_util_pct
    danger_mask = utils >= danger_util_pct

    warning_points = int(np.sum(warning_mask))
    contact_points = int(np.sum(danger_mask))

    # Tendencia creciente al límite
    if len(utils) >= 5:
        idx = np.arange(len(utils), dtype=float)
        slope = float(np.polyfit(idx, utils, 1)[0])
    else:
        slope = 0.0

    last_n = min(8, len(utils))
    tail_mean = float(np.mean(utils[-last_n:])) if last_n else 0.0
    max_util = float(np.max(utils)) if len(utils) else 0.0

    if contact_points > 0 or max_util >= danger_util_pct:
        severity = "DANGER"
        color = "#dc2626"
        triggered = True
        message = "Early rub risk high"
    elif warning_points >= 2 or tail_mean >= warning_util_pct or slope > 1.5:
        severity = "WARNING"
        color = "#f59e0b"
        triggered = True
        message = "Possible early rub tendency"
    else:
        severity = "SAFE"
        color = "#16a34a"
        triggered = False
        message = "No early rub tendency detected"

    first_warning_speed = None
    first_danger_speed = None

    if np.any(warning_mask):
        first_warning_speed = float(speed[np.argmax(warning_mask)])
    if np.any(danger_mask):
        first_danger_speed = float(speed[np.argmax(danger_mask)])

    return {
        "triggered": triggered,
        "severity": severity,
        "color": color,
        "message": message,
        "max_util_pct": max_util,
        "contact_points": contact_points,
        "warning_points": warning_points,
        "trend_score": slope,
        "first_warning_speed": first_warning_speed,
        "first_danger_speed": first_danger_speed,
    }


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
    panel_x0 = 0.855
    panel_x1 = 0.970
    panel_y1 = 0.950
    header_h = 0.022
    row_h = 0.033
    body_pad = 0.008
    radius = 0.008

    panel_h = header_h + len(rows) * row_h + body_pad
    panel_y0 = panel_y1 - panel_h

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path=rounded_rect_path(panel_x0, panel_y0, panel_x1, panel_y1, radius),
        line=dict(color="rgba(203,213,225,0.90)", width=1),
        fillcolor="rgba(255,255,255,0.78)",
        layer="above",
    )

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path=rounded_rect_path(panel_x0, panel_y1 - header_h, panel_x1, panel_y1, radius),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(147,197,253,0.88)",
        layer="above",
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=(panel_x0 + panel_x1) / 2.0,
        y=panel_y1 - header_h / 2.0,
        text="<b>Shaft Centerline</b>",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font=dict(size=8.8, color="#111827"),
    )

    current_top = panel_y1 - header_h - 0.004

    for title, value in rows:
        title_y = current_top
        value_y = current_top - 0.013

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.008,
            y=title_y,
            xanchor="left",
            yanchor="top",
            text=f"<b>{title}</b>",
            showarrow=False,
            font=dict(size=7.8, color="#111827"),
            align="left",
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.008,
            y=value_y,
            xanchor="left",
            yanchor="top",
            text=value,
            showarrow=False,
            font=dict(size=7.2, color="#111827"),
            align="left",
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
    normalize_to_origin: bool,
    x_range: List[float],
    y_range: List[float],
    clearance_mode: str,
    clearance_center_mode: str,
    clearance_center_x: float,
    clearance_center_y: float,
    clearance_x: float,
    clearance_y: float,
    display_speed_min: float,
    display_speed_max: float,
    semaforo_status: str,
    semaforo_color: str,
) -> Tuple[go.Figure, Dict[str, float]]:
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

    fig = go.Figure()

    bx, by = build_boundary_curve(
        center_x=clearance_center_x,
        center_y=clearance_center_y,
        clearance_x=clearance_x,
        clearance_y=clearance_y,
    )

    fig.add_trace(
        go.Scatter(
            x=bx,
            y=by,
            mode="lines",
            line=dict(color=semaforo_color, width=2.4, dash="dot"),
            hovertemplate=(
                f"Clearance Boundary<br>"
                f"Center X: {clearance_center_x:.3f} {gap_unit}<br>"
                f"Center Y: {clearance_center_y:.3f} {gap_unit}<br>"
                f"Cx: {clearance_x:.3f} {gap_unit}<br>"
                f"Cy: {clearance_y:.3f} {gap_unit}<br>"
                f"Status: {semaforo_status}<extra></extra>"
            ),
            showlegend=False,
            name="Clearance Boundary",
        )
    )

    if semaforo_status == "DANGER":
        fig.add_trace(
            go.Scatter(
                x=bx,
                y=by,
                mode="lines",
                line=dict(color=semaforo_color, width=7),
                opacity=0.12,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[clearance_center_x],
            y=[clearance_center_y],
            mode="markers",
            marker=dict(size=8, color="#111827", symbol="x"),
            hovertemplate=(
                f"Boundary Center<br>X: {clearance_center_x:.3f} {gap_unit}<br>"
                f"Y: {clearance_center_y:.3f} {gap_unit}<extra></extra>"
            ),
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

    util_a = boundary_utilization_pct(row_a_x, row_a_y, clearance_center_x, clearance_center_y, clearance_x, clearance_y)
    util_b = boundary_utilization_pct(row_b_x, row_b_y, clearance_center_x, clearance_center_y, clearance_x, clearance_y)
    util_end = boundary_utilization_pct(float(x[-1]), float(y[-1]), clearance_center_x, clearance_center_y, clearance_x, clearance_y)
    max_util = max(
        boundary_utilization_pct(float(px), float(py), clearance_center_x, clearance_center_y, clearance_x, clearance_y)
        for px, py in zip(x, y)
    ) if len(x) else 0.0

    diag = {
        "util_a": util_a,
        "util_b": util_b,
        "util_end": util_end,
        "util_max": max_util,
        "margin_a": remaining_margin_pct(util_a),
        "margin_b": remaining_margin_pct(util_b),
        "margin_end": remaining_margin_pct(util_end),
        "margin_min": remaining_margin_pct(max_util),
    }

    rows = [
        ("Cursor A", f"X={format_number(row_a_x,3)} / Y={format_number(row_a_y,3)} {gap_unit} @ {int(round(row_a['speed']))} {speed_unit}"),
        ("Cursor B", f"X={format_number(row_b_x,3)} / Y={format_number(row_b_y,3)} {gap_unit} @ {int(round(row_b['speed']))} {speed_unit}"),
        ("Boundary", f"{clearance_mode} · Cx={format_number(clearance_x,3)} / Cy={format_number(clearance_y,3)} {gap_unit}"),
        ("Boundary Center", f"{clearance_center_mode} · X={format_number(clearance_center_x,3)} / Y={format_number(clearance_center_y,3)}"),
        ("RPM Window", f"{int(round(display_speed_min))} to {int(round(display_speed_max))} {speed_unit}"),
        ("Status", f"<span style='color:{semaforo_color};'><b>{semaforo_status}</b></span>"),
        ("API 684 Helper", f"Max utilization {format_number(max_util,1)}% · Min margin {format_number(diag['margin_min'],1)}%"),
        ("Normalize", "Enabled" if normalize_to_origin else "Disabled"),
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
        range=x_range,
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
        range=y_range,
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

    return fig, diag


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
    show_rpm_labels_global: bool,
    marker_stride_global: int,
    normalize_to_origin: bool,
) -> None:
    meta = item["meta"]
    raw_df = item["raw_df"]
    grouped_df = item["grouped_df"]

    plot_df = grouped_df.copy()
    plot_df["x_gap"] = smooth_series(plot_df["x_gap"], smooth_window)
    plot_df["y_gap"] = smooth_series(plot_df["y_gap"], smooth_window)

    full_speed_min = int(plot_df["speed"].min())
    full_speed_max = int(plot_df["speed"].max())

    with st.expander(f"Panel {panel_index + 1} · Geometry / Scale / Clearance / RPM Window / Rub", expanded=False):
        st.markdown("#### RPM Display Range")
        rpm_range_mode = st.selectbox(
            "RPM display mode",
            ["Auto", "Manual"],
            index=0,
            key=f"scl_rpm_range_mode_{panel_index}_{item['id']}",
        )

        if rpm_range_mode == "Auto":
            display_speed_min = float(full_speed_min)
            display_speed_max = float(full_speed_max)
            st.caption("Showing the full RPM range available in this panel.")
        else:
            display_speed_min, display_speed_max = st.slider(
                "Visible RPM range",
                min_value=full_speed_min,
                max_value=full_speed_max,
                value=(full_speed_min, full_speed_max),
                step=1,
                key=f"scl_visible_rpm_range_{panel_index}_{item['id']}",
            )
            display_speed_min = float(display_speed_min)
            display_speed_max = float(display_speed_max)

        st.markdown("#### X / Y Scale")
        auto_scale_xy = st.checkbox(
            "Auto X/Y",
            value=True,
            key=f"scl_auto_xy_{panel_index}_{item['id']}",
        )

        if auto_scale_xy:
            manual_x_min = -10.0
            manual_x_max = 10.0
            manual_y_min = -10.0
            manual_y_max = 10.0
            st.caption("Using automatic independent X and Y ranges for this panel.")
        else:
            sx1, sx2 = st.columns(2)
            with sx1:
                manual_x_min = st.number_input(
                    "X min",
                    value=-10.0,
                    step=0.5,
                    format="%.3f",
                    key=f"scl_xmin_{panel_index}_{item['id']}",
                )
                manual_y_min = st.number_input(
                    "Y min",
                    value=-10.0,
                    step=0.5,
                    format="%.3f",
                    key=f"scl_ymin_{panel_index}_{item['id']}",
                )
            with sx2:
                manual_x_max = st.number_input(
                    "X max",
                    value=10.0,
                    step=0.5,
                    format="%.3f",
                    key=f"scl_xmax_{panel_index}_{item['id']}",
                )
                manual_y_max = st.number_input(
                    "Y max",
                    value=10.0,
                    step=0.5,
                    format="%.3f",
                    key=f"scl_ymax_{panel_index}_{item['id']}",
                )

        st.markdown("#### Clearance Boundary")
        cb1, cb2 = st.columns(2)
        clearance_mode = cb1.selectbox(
            "Boundary mode",
            ["Auto", "Manual"],
            index=0,
            key=f"scl_clear_mode_{panel_index}_{item['id']}",
        )
        clearance_center_mode = cb2.selectbox(
            "Boundary center",
            ["Origin (0,0)", "Data Mean", "Manual"],
            index=0,
            key=f"scl_clear_center_mode_{panel_index}_{item['id']}",
        )

        if clearance_center_mode == "Manual":
            cc1, cc2 = st.columns(2)
            manual_center_x = cc1.number_input(
                "Boundary center X",
                value=0.0,
                step=0.1,
                format="%.3f",
                key=f"scl_center_x_{panel_index}_{item['id']}",
            )
            manual_center_y = cc2.number_input(
                "Boundary center Y",
                value=0.0,
                step=0.1,
                format="%.3f",
                key=f"scl_center_y_{panel_index}_{item['id']}",
            )
        else:
            manual_center_x = 0.0
            manual_center_y = 0.0

        if clearance_mode == "Manual":
            cm1, cm2 = st.columns(2)
            manual_clearance_x = cm1.number_input(
                "Clearance X (Cx)",
                value=5.0,
                min_value=0.001,
                step=0.1,
                format="%.3f",
                key=f"scl_clear_x_{panel_index}_{item['id']}",
            )
            manual_clearance_y = cm2.number_input(
                "Clearance Y (Cy)",
                value=5.0,
                min_value=0.001,
                step=0.1,
                format="%.3f",
                key=f"scl_clear_y_{panel_index}_{item['id']}",
            )
        else:
            manual_clearance_x = 5.0
            manual_clearance_y = 5.0

        st.markdown("#### Early Rub Detection")
        er1, er2 = st.columns(2)
        early_rub_warning_pct = er1.slider(
            "Warning utilization %",
            min_value=50,
            max_value=98,
            value=80,
            step=1,
            key=f"scl_rub_warn_{panel_index}_{item['id']}",
        )
        early_rub_danger_pct = er2.slider(
            "Danger utilization %",
            min_value=60,
            max_value=100,
            value=95,
            step=1,
            key=f"scl_rub_danger_{panel_index}_{item['id']}",
        )

        if early_rub_danger_pct <= early_rub_warning_pct:
            early_rub_danger_pct = early_rub_warning_pct + 1

        st.caption(
            "API 684 helper: normalized position against clearance boundary. "
            "Early rub detection here is an analytical helper based on utilization proximity and trend."
        )

    display_df = plot_df[
        (plot_df["speed"] >= display_speed_min) & (plot_df["speed"] <= display_speed_max)
    ].copy()

    if display_df.empty:
        st.warning(f"Panel {panel_index + 1}: no hay puntos en el rango RPM seleccionado.")
        return

    speed_min = int(display_df["speed"].min())
    speed_max = int(display_df["speed"].max())

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

    row_a = nearest_row_for_speed(display_df, cursor_a_speed)
    row_b = nearest_row_for_speed(display_df, cursor_b_speed)

    if normalize_to_origin:
        base_x = float(display_df["x_gap"].iloc[0])
        base_y = float(display_df["y_gap"].iloc[0])
        x_plot = (display_df["x_gap"] - base_x).to_numpy(dtype=float)
        y_plot = (display_df["y_gap"] - base_y).to_numpy(dtype=float)
    else:
        x_plot = display_df["x_gap"].to_numpy(dtype=float)
        y_plot = display_df["y_gap"].to_numpy(dtype=float)

    x_range, y_range = compute_xy_ranges(
        x=x_plot,
        y=y_plot,
        auto_scale_xy=auto_scale_xy,
        manual_x_min=manual_x_min,
        manual_x_max=manual_x_max,
        manual_y_min=manual_y_min,
        manual_y_max=manual_y_max,
    )

    boundary = resolve_clearance_boundary(
        x=x_plot,
        y=y_plot,
        mode=clearance_mode,
        center_mode=clearance_center_mode,
        manual_cx=manual_clearance_x,
        manual_cy=manual_clearance_y,
        manual_center_x=manual_center_x,
        manual_center_y=manual_center_y,
    )

    early_rub = detect_early_rub(
        x=x_plot,
        y=y_plot,
        speed=display_df["speed"].to_numpy(dtype=float),
        center_x=boundary["center_x"],
        center_y=boundary["center_y"],
        clearance_x=boundary["clearance_x"],
        clearance_y=boundary["clearance_y"],
        warning_util_pct=float(early_rub_warning_pct),
        danger_util_pct=float(early_rub_danger_pct),
    )

    semaforo_status, semaforo_color = get_clearance_status(early_rub["max_util_pct"])

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
                Visible Speed Range: <b>{int(display_df['speed'].min())} - {int(display_df['speed'].max())} {speed_unit}</b>
            </div>
            <div class="wm-chip-row">
                <div class="wm-chip">File: {item["file_name"]}</div>
                <div class="wm-chip">Raw rows: {len(raw_df):,}</div>
                <div class="wm-chip">Grouped points: {len(display_df):,}</div>
                <div class="wm-chip">Gap Unit: {gap_unit}</div>
                <div class="wm-chip">Smoothing: {smooth_window}</div>
                <div class="wm-chip">Normalize: {"Yes" if normalize_to_origin else "No"}</div>
                <div class="wm-chip">RPM Window: {"Auto" if rpm_range_mode == "Auto" else "Manual"}</div>
                <div class="wm-chip">X/Y Scale: {"Auto" if auto_scale_xy else "Manual"}</div>
                <div class="wm-chip">Boundary: {clearance_mode}</div>
                <div class="wm-chip">Boundary Center: {clearance_center_mode}</div>
                <div class="wm-chip" style="color:{semaforo_color}; border-color:{semaforo_color};"><b>Status: {semaforo_status}</b></div>
                <div class="wm-chip" style="color:{early_rub['color']}; border-color:{early_rub['color']};"><b>Rub: {early_rub['severity']}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig, diag = build_scl_figure(
        df=display_df,
        meta=meta,
        row_a=row_a,
        row_b=row_b,
        logo_uri=logo_uri,
        show_info_box=show_info_box,
        show_rpm_labels=show_rpm_labels_global,
        marker_stride=marker_stride_global,
        normalize_to_origin=normalize_to_origin,
        x_range=x_range,
        y_range=y_range,
        clearance_mode=clearance_mode,
        clearance_center_mode=clearance_center_mode,
        clearance_center_x=boundary["center_x"],
        clearance_center_y=boundary["center_y"],
        clearance_x=boundary["clearance_x"],
        clearance_y=boundary["clearance_y"],
        display_speed_min=display_speed_min,
        display_speed_max=display_speed_max,
        semaforo_status=semaforo_status,
        semaforo_color=semaforo_color,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False},
        key=f"wm_scl_plot_{panel_index}_{item['id']}",
    )

    first_warning_text = "—"
    first_danger_text = "—"
    if early_rub["first_warning_speed"] is not None:
        first_warning_text = f"{early_rub['first_warning_speed']:.0f} {speed_unit}"
    if early_rub["first_danger_speed"] is not None:
        first_danger_text = f"{early_rub['first_danger_speed']:.0f} {speed_unit}"

    st.markdown(
        f"""
        <div class="wm-card">
            <div class="wm-card-title">API 684 Helper + Early Rub Detection · Panel {panel_index + 1}</div>
            <div class="wm-card-subtitle">Clearance utilization, margin, semáforo y tendencia temprana de roce.</div>
            <div class="wm-chip-row">
                <div class="wm-chip" style="color:{semaforo_color}; border-color:{semaforo_color};">Semáforo: {semaforo_status}</div>
                <div class="wm-chip">Max util: {diag["util_max"]:.1f}%</div>
                <div class="wm-chip">Minimum margin: {diag["margin_min"]:.1f}%</div>
                <div class="wm-chip">Cursor A util: {diag["util_a"]:.1f}%</div>
                <div class="wm-chip">Cursor B util: {diag["util_b"]:.1f}%</div>
                <div class="wm-chip" style="color:{early_rub['color']}; border-color:{early_rub['color']};">Early rub: {early_rub["severity"]}</div>
                <div class="wm-chip">Rub message: {early_rub["message"]}</div>
                <div class="wm-chip">Warning points: {early_rub["warning_points"]}</div>
                <div class="wm-chip">Danger points: {early_rub["contact_points"]}</div>
                <div class="wm-chip">Trend score: {early_rub["trend_score"]:.2f}</div>
                <div class="wm-chip">1st warning: {first_warning_text}</div>
                <div class="wm-chip">1st danger: {first_danger_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    title = f"Shaft Centerline {panel_index + 1} — {machine} — {point} / {paired_point}"

    export_state_key = (
        f"scl::{item['id']}::{panel_index}::{smooth_window}::{show_info_box}::{show_rpm_labels_global}::{marker_stride_global}::"
        f"{normalize_to_origin}::{rpm_range_mode}::{display_speed_min}::{display_speed_max}::"
        f"{auto_scale_xy}::{manual_x_min}::{manual_x_max}::{manual_y_min}::{manual_y_max}::"
        f"{clearance_mode}::{clearance_center_mode}::{boundary['center_x']}::{boundary['center_y']}::{boundary['clearance_x']}::{boundary['clearance_y']}::"
        f"{early_rub_warning_pct}::{early_rub_danger_pct}::{cursor_a_speed}::{cursor_b_speed}"
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
    require_login()
    ensure_report_state()

    st.markdown('<div class="wm-page-title">Shaft Centerline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="wm-page-subtitle">Centerline position from paired X/Y gap probes versus speed.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        render_user_menu()
        st.markdown("---")
        st.markdown("### Shaft Centerline input")
        uploaded_files = st.file_uploader(
            "Upload one or more Shaft Centerline CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )

        st.markdown("### Global Controls")
        smooth_window = st.slider("Gap smoothing", 1, 11, 3, step=2)
        show_info_box = st.checkbox("Show information box", value=True)
        show_rpm_labels = st.checkbox("Show RPM labels", value=True)
        marker_stride = st.slider("RPM label step", 10, 150, 45, step=5)
        normalize_to_origin = st.checkbox("Normalize to first point", value=False)

    if not uploaded_files:
        st.markdown(
            """
            <div class="wm-card" style="max-width: 980px;">
                <div class="wm-card-title">Carga archivos para comenzar</div>
                <div class="wm-card-subtitle">
                    Sube uno o varios archivos CSV de Shaft Centerline desde el panel izquierdo.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
            show_rpm_labels_global=show_rpm_labels,
            marker_stride_global=marker_stride,
            normalize_to_origin=normalize_to_origin,
        )

        if panel_index < len(parsed_items) - 1:
            st.markdown("---")


if __name__ == "__main__":
    main()
