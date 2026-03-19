from __future__ import annotations

from core.auth import require_login, render_user_menu

require_login()
render_user_menu()

import base64
import hashlib
import math
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# WATERMELON SYSTEM — TRENDS VIEWER
# Premium industrial trend module
# ============================================================

st.set_page_config(page_title="Watermelon System | Trends", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .main > div {
            padding-top: 0.18rem;
        }

        .stApp {
            background-color: #f3f4f6;
        }

        section[data-testid="stSidebar"] {
            background: #e5e7eb;
            border-right: 1px solid #cbd5e1;
        }

        .wm-export-note {
            font-size: 0.95rem;
            color: #374151;
            padding-top: 0.30rem;
        }

        .wm-top-note {
            font-size: 0.93rem;
            color: #475569;
            margin-bottom: 0.20rem;
        }

        .wm-cursor-shell {
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.78));
            border: 1px solid #dbe3ee;
            border-radius: 18px;
            padding: 14px 16px 12px 16px;
            margin-bottom: 12px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }

        .wm-cursor-title {
            font-size: 1.02rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 2px;
        }

        .wm-cursor-subtitle {
            font-size: 0.88rem;
            color: #64748b;
            margin-bottom: 10px;
        }

        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            font-family: monospace;
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
            min-height: 48px;
            border-radius: 12px;
            font-weight: 700;
        }

        div[data-testid="stFileUploader"] section {
            background: rgba(255,255,255,0.72);
            border: 1px solid #cbd5e1;
            border-radius: 14px;
            padding: 0.25rem;
        }

        div[data-testid="stSelectSlider"] label p {
            font-weight: 700;
            color: #0f172a;
        }

        div[data-testid="stExpander"] {
            border: 1px solid #dbe3ee;
            border-radius: 16px;
            background: rgba(255,255,255,0.65);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_page_style()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_logo_base64(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def get_logo_data_uri(path: Path) -> Optional[str]:
    b64 = get_logo_base64(path)
    if not b64:
        return None
    return f"data:image/png;base64,{b64}"


def make_export_state_key(parts: List[Any]) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


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


def safe_datetime(value: Any) -> Optional[pd.Timestamp]:
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts)
    except Exception:
        return None


def color_for_index(index: int) -> str:
    # Colores de señales sin rojo ni amarillo
    palette = [
        "#5b9cf0",
        "#10b981",
        "#8b5cf6",
        "#06b6d4",
        "#ec4899",
        "#14b8a6",
        "#6366f1",
        "#0f766e",
        "#7c3aed",
        "#2563eb",
    ]
    return palette[index % len(palette)]


def pretty_time(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return ts.strftime("%I:%M %p").lstrip("0")


def pretty_date(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return ts.strftime("%Y-%m-%d")


def trim_text(text: str, max_len: int) -> str:
    text = str(text or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def safe_percent_change(initial_value: Optional[float], final_value: Optional[float]) -> Optional[float]:
    if initial_value is None or final_value is None:
        return None
    try:
        init_val = float(initial_value)
        final_val = float(final_value)
        if not math.isfinite(init_val) or not math.isfinite(final_val):
            return None
        if abs(init_val) < 1e-12:
            return None
        return ((final_val - init_val) / abs(init_val)) * 100.0
    except Exception:
        return None


def ts_to_label(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def label_to_ts(text: str) -> Optional[pd.Timestamp]:
    return safe_datetime(text)


# ------------------------------------------------------------
# Data model
# ------------------------------------------------------------
@dataclass
class TrendRecord:
    trend_id: str
    file_name: str
    machine: str = "Unknown"
    point: str = "Point"
    variable: str = "Direct"
    y_axis_unit: str = ""
    speed_unit: str = "rpm"
    timestamp_min: Optional[pd.Timestamp] = None
    timestamp_max: Optional[pd.Timestamp] = None
    x_time: pd.Series = field(default_factory=lambda: pd.Series(dtype="datetime64[ns]"))
    y_value: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    phase: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    speed: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    y_status: pd.Series = field(default_factory=lambda: pd.Series(dtype=str))
    phase_status: pd.Series = field(default_factory=lambda: pd.Series(dtype=str))
    speed_status: pd.Series = field(default_factory=lambda: pd.Series(dtype=str))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return f"{self.point} | {self.variable}"

    @property
    def point_clean(self) -> str:
        return self.point if self.point else self.file_name

    @property
    def n_samples(self) -> int:
        return int(len(self.x_time))


# ------------------------------------------------------------
# Trend CSV parser
# ------------------------------------------------------------
def parse_trend_csv(uploaded_file) -> Optional[TrendRecord]:
    try:
        raw_bytes = uploaded_file.getvalue()
        text = raw_bytes.decode("utf-8-sig", errors="ignore")
    except Exception:
        return None

    lines = [line.rstrip("\r") for line in text.splitlines() if line.strip() != ""]
    if len(lines) < 2:
        return None

    header_map: Dict[str, str] = {}
    data_header_idx: Optional[int] = None

    for idx, line in enumerate(lines):
        if "X-Axis Value" in line and "Y-Axis Value" in line:
            data_header_idx = idx
            break
        parts = line.split(",", 1)
        key = parts[0].strip() if parts else ""
        value = parts[1].strip() if len(parts) > 1 else ""
        if key:
            header_map[key] = value

    if data_header_idx is None:
        return None

    csv_text = "\n".join(lines[data_header_idx:])
    try:
        df = pd.read_csv(BytesIO(csv_text.encode("utf-8")))
    except Exception:
        return None

    expected_cols = [
        "X-Axis Value",
        "Y-Axis Value",
        "Y-Axis Status",
        "Phase",
        "Phase Status",
        "Speed",
        "Speed Status",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["X-Axis Value"] = pd.to_datetime(df["X-Axis Value"], errors="coerce")
    df["Y-Axis Value"] = pd.to_numeric(df["Y-Axis Value"], errors="coerce")
    df["Phase"] = pd.to_numeric(df["Phase"], errors="coerce")
    df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")

    df = df.dropna(subset=["X-Axis Value"]).copy()
    df = df.sort_values("X-Axis Value").reset_index(drop=True)

    if df.empty:
        return None

    trend_id = f"trend::{uploaded_file.name}"
    point = str(header_map.get("Point Name", uploaded_file.name)).strip()
    variable = str(header_map.get("Variable", "Direct")).strip()
    machine = str(header_map.get("Machine Name", "Unknown")).strip()
    y_axis_unit = str(header_map.get("Y-Axis Unit", "")).strip()
    speed_unit = str(header_map.get("Speed Unit", "rpm")).strip()

    return TrendRecord(
        trend_id=trend_id,
        file_name=uploaded_file.name,
        machine=machine,
        point=point,
        variable=variable,
        y_axis_unit=y_axis_unit,
        speed_unit=speed_unit,
        timestamp_min=safe_datetime(df["X-Axis Value"].min()),
        timestamp_max=safe_datetime(df["X-Axis Value"].max()),
        x_time=df["X-Axis Value"],
        y_value=df["Y-Axis Value"],
        phase=df["Phase"],
        speed=df["Speed"],
        y_status=df["Y-Axis Status"].astype(str),
        phase_status=df["Phase Status"].astype(str),
        speed_status=df["Speed Status"].astype(str),
        metadata=header_map,
    )


def load_trend_records_from_uploader(files: List[Any]) -> List[TrendRecord]:
    records: List[TrendRecord] = []
    for file in files:
        rec = parse_trend_csv(file)
        if rec is not None:
            records.append(rec)
    return records


# ------------------------------------------------------------
# Figure chrome
# ------------------------------------------------------------
def _draw_top_strip(
    fig: go.Figure,
    machine_name: str,
    signal_names_text: str,
    metric_name: str,
    latest_text: str,
    logo_uri: Optional[str],
    time_range_text: str,
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
        xref="paper",
        yref="paper",
        x=machine_x,
        y=y_text,
        xanchor="left",
        yanchor="middle",
        text=f"<b>{trim_text(machine_name, 28)}</b>",
        showarrow=False,
        font=dict(size=12.8, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.325,
        y=y_text,
        xanchor="center",
        yanchor="middle",
        text=trim_text(signal_names_text, 34),
        showarrow=False,
        font=dict(size=11.4, color="#111827"),
        align="center",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.640,
        y=y_text,
        xanchor="center",
        yanchor="middle",
        text=f"Metric: <b>{metric_name}</b> | Latest: <b>{trim_text(latest_text, 32)}</b>",
        showarrow=False,
        font=dict(size=11.3, color="#111827"),
        align="center",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.986,
        y=y_text,
        xanchor="right",
        yanchor="middle",
        text=trim_text(time_range_text, 28),
        showarrow=False,
        font=dict(size=11.2, color="#111827"),
        align="right",
    )


def _draw_right_info_box(
    fig: go.Figure,
    rows: List[Tuple[str, str]],
) -> None:
    panel_x0 = 0.834
    panel_x1 = 0.976
    panel_y1 = 0.915
    header_h = 0.034
    row_h = 0.058
    panel_h = header_h + len(rows) * row_h + 0.018
    panel_y0 = panel_y1 - panel_h

    fig.add_shape(
        type="path",
        xref="paper",
        yref="paper",
        path=rounded_rect_path(panel_x0, panel_y0, panel_x1, panel_y1, 0.012),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(255,255,255,0.72)",
        layer="above",
    )

    fig.add_shape(
        type="path",
        xref="paper",
        yref="paper",
        path=rounded_rect_path(panel_x0, panel_y1 - header_h, panel_x1, panel_y1, 0.012),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(147,197,253,0.94)",
        layer="above",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=(panel_x0 + panel_x1) / 2.0,
        y=panel_y1 - header_h / 2.0,
        text="<b>Trend Information</b>",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font=dict(size=11.4, color="#111827"),
    )

    current_top = panel_y1 - header_h - 0.008

    for title, value in rows:
        title_y = current_top - 0.004
        value_y = current_top - 0.030

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=panel_x0 + 0.030,
            y=title_y,
            xanchor="left",
            yanchor="top",
            text=f"<b>{title}</b>",
            showarrow=False,
            font=dict(size=10.7, color="#111827"),
            align="left",
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=panel_x0 + 0.030,
            y=value_y,
            xanchor="left",
            yanchor="top",
            text=value,
            showarrow=False,
            font=dict(size=10.4, color="#111827"),
            align="left",
        )

        current_top -= row_h


# ------------------------------------------------------------
# Metric accessors
# ------------------------------------------------------------
def get_metric_series(record: TrendRecord, metric_key: str) -> Tuple[pd.Series, str]:
    if metric_key == "Amplitude":
        return record.y_value, record.y_axis_unit or ""
    if metric_key == "Phase":
        return record.phase, "deg"
    if metric_key == "Speed":
        return record.speed, record.speed_unit or "rpm"
    return record.y_value, record.y_axis_unit or ""


def get_clean_metric_df(record: TrendRecord, metric_key: str) -> pd.DataFrame:
    metric_series, _ = get_metric_series(record, metric_key)
    df = pd.DataFrame(
        {
            "x": pd.to_datetime(record.x_time, errors="coerce"),
            "y": pd.to_numeric(metric_series, errors="coerce"),
        }
    ).dropna(subset=["x", "y"])
    if df.empty:
        return df
    return df.sort_values("x").reset_index(drop=True)


def get_cursor_nearest_info(
    record: TrendRecord,
    metric_key: str,
    cursor_ts: Optional[pd.Timestamp],
) -> Optional[Tuple[float, pd.Timestamp, str]]:
    if cursor_ts is None:
        return None
    df = get_clean_metric_df(record, metric_key)
    if df.empty:
        return None
    idx = (df["x"] - cursor_ts).abs().idxmin()
    row = df.loc[idx]
    unit = get_metric_series(record, metric_key)[1]
    return float(row["y"]), pd.Timestamp(row["x"]), unit


def get_time_options_for_records(records: List[TrendRecord], metric_key: str) -> List[pd.Timestamp]:
    ts_values: List[pd.Timestamp] = []
    for record in records:
        df = get_clean_metric_df(record, metric_key)
        if not df.empty:
            ts_values.extend(list(pd.to_datetime(df["x"], errors="coerce").dropna()))
    if not ts_values:
        return []
    unique_sorted = sorted(pd.Series(ts_values).dropna().unique())
    return [pd.Timestamp(x) for x in unique_sorted]


# ------------------------------------------------------------
# Figure builder
# ------------------------------------------------------------
def build_trend_figure(
    records: List[TrendRecord],
    metric_key: str,
    show_markers: bool,
    fill_area: bool,
    y_axis_mode: str,
    y_axis_manual_min: Optional[float],
    y_axis_manual_max: Optional[float],
    x_axis_mode: str,
    x_axis_manual_start: Optional[pd.Timestamp],
    x_axis_manual_end: Optional[pd.Timestamp],
    warning_enabled: bool,
    warning_value: Optional[float],
    danger_enabled: bool,
    danger_value: Optional[float],
    show_right_info_box: bool,
    show_legend: bool,
    logo_uri: Optional[str],
    cursor_map: Dict[str, Optional[pd.Timestamp]],
) -> go.Figure:
    fig = go.Figure()
    visible_records: List[Tuple[TrendRecord, pd.DataFrame, str]] = []

    global_y_min = np.inf
    global_y_max = -np.inf
    global_x_min: Optional[pd.Timestamp] = None
    global_x_max: Optional[pd.Timestamp] = None

    for idx, record in enumerate(records):
        df = get_clean_metric_df(record, metric_key)
        metric_unit = get_metric_series(record, metric_key)[1]

        if df.empty:
            continue

        color = color_for_index(idx)
        mode = "lines+markers" if show_markers else "lines"

        fig.add_trace(
            go.Scattergl(
                x=df["x"],
                y=df["y"],
                mode=mode,
                line=dict(width=2.4, color=color),
                marker=dict(size=5, color=color),
                fill="tozeroy" if fill_area and len(records) == 1 else None,
                fillcolor="rgba(91, 156, 240, 0.10)" if fill_area and len(records) == 1 else None,
                name=record.point_clean,
                hovertemplate=(
                    "Point: %{fullData.name}<br>"
                    "Time: %{x}<br>"
                    f"{metric_key}: " + "%{y:.4f} " + f"{metric_unit}" + "<extra></extra>"
                ),
                showlegend=show_legend,
                connectgaps=False,
            )
        )

        y_min_local = float(df["y"].min())
        y_max_local = float(df["y"].max())
        global_y_min = min(global_y_min, y_min_local)
        global_y_max = max(global_y_max, y_max_local)

        x_min_local = pd.Timestamp(df["x"].min())
        x_max_local = pd.Timestamp(df["x"].max())
        global_x_min = x_min_local if global_x_min is None else min(global_x_min, x_min_local)
        global_x_max = x_max_local if global_x_max is None else max(global_x_max, x_max_local)

        visible_records.append((record, df, metric_unit))

    if not visible_records:
        fig.update_layout(
            height=640,
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#f3f4f6",
            margin=dict(l=46, r=18, t=84, b=40),
        )
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No valid trend data available",
            showarrow=False,
            font=dict(size=18, color="#6b7280"),
        )
        return fig

    if not math.isfinite(global_y_min):
        global_y_min = 0.0
    if not math.isfinite(global_y_max):
        global_y_max = 1.0

    if math.isclose(global_y_min, global_y_max, rel_tol=1e-12, abs_tol=1e-12):
        base_pad = max(abs(global_y_max) * 0.10, 0.25)
        global_y_min -= base_pad
        global_y_max += base_pad
    else:
        base_pad = max((global_y_max - global_y_min) * 0.12, 0.10)
        global_y_min -= base_pad
        global_y_max += base_pad

    alarm_values = []
    if warning_enabled and warning_value is not None and math.isfinite(float(warning_value)):
        alarm_values.append(float(warning_value))
    if danger_enabled and danger_value is not None and math.isfinite(float(danger_value)):
        alarm_values.append(float(danger_value))

    if alarm_values:
        global_y_max = max(global_y_max, max(alarm_values) * 1.08)

    if y_axis_mode == "Manual" and y_axis_manual_min is not None and y_axis_manual_max is not None:
        y_min_final = float(y_axis_manual_min)
        y_max_final = float(y_axis_manual_max)
        if y_min_final >= y_max_final:
            y_min_final, y_max_final = min(y_min_final, y_max_final), max(y_min_final, y_max_final) + 1.0
    else:
        y_min_final = float(global_y_min)
        y_max_final = float(global_y_max)

    if x_axis_mode == "Manual" and x_axis_manual_start is not None and x_axis_manual_end is not None:
        if x_axis_manual_start < x_axis_manual_end:
            x_min_final = x_axis_manual_start
            x_max_final = x_axis_manual_end
        else:
            x_min_final = global_x_min
            x_max_final = global_x_max
    else:
        x_min_final = global_x_min
        x_max_final = global_x_max

    if warning_enabled and warning_value is not None and math.isfinite(float(warning_value)):
        fig.add_hline(
            y=float(warning_value),
            line_width=1.8,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"Warning {format_number(warning_value, 3)}",
            annotation_position="top left",
            annotation_font_color="#92400e",
        )

    if danger_enabled and danger_value is not None and math.isfinite(float(danger_value)):
        fig.add_hline(
            y=float(danger_value),
            line_width=1.9,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Danger {format_number(danger_value, 3)}",
            annotation_position="top left",
            annotation_font_color="#991b1b",
        )

    cursor_line_specs = {
        "A Initial": "#334155",
        "A Current": "#64748b",
        "B Initial": "#111827",
        "B Current": "#475569",
    }
    for label, ts in cursor_map.items():
        if ts is not None:
            fig.add_vline(
                x=ts,
                line_width=1.8,
                line_dash="dot",
                line_color=cursor_line_specs.get(label, "#475569"),
            )

    machine_name = records[0].machine if records else "Unknown"
    signal_names_text = " | ".join([r.point_clean for r in records[:2]])
    if len(records) > 2:
        signal_names_text += f" +{len(records) - 2}"

    latest_values: List[str] = []
    for rec, df, unit in visible_records[:2]:
        if not df.empty:
            latest_values.append(f"{rec.point_clean}: {format_number(df['y'].iloc[-1], 3)} {unit}".strip())
    latest_text = " | ".join(latest_values) if latest_values else "—"

    unit_for_axis = visible_records[0][2] if visible_records else ""
    axis_title = f"{metric_key} ({unit_for_axis})" if unit_for_axis else metric_key

    time_range_text = "—"
    if global_x_min is not None and global_x_max is not None:
        time_range_text = f"{global_x_min.strftime('%Y-%m-%d %H:%M')} → {global_x_max.strftime('%Y-%m-%d %H:%M')}"

    _draw_top_strip(
        fig=fig,
        machine_name=machine_name,
        signal_names_text=signal_names_text,
        metric_name=metric_key,
        latest_text=latest_text,
        logo_uri=logo_uri,
        time_range_text=time_range_text,
    )

    if show_right_info_box:
        rows: List[Tuple[str, str]] = []

        first_rec = visible_records[0][0]
        second_rec = visible_records[1][0] if len(visible_records) >= 2 else first_rec

        a_initial_info = get_cursor_nearest_info(first_rec, metric_key, cursor_map.get("A Initial"))
        a_current_info = get_cursor_nearest_info(first_rec, metric_key, cursor_map.get("A Current"))
        b_initial_info = get_cursor_nearest_info(second_rec, metric_key, cursor_map.get("B Initial"))
        b_current_info = get_cursor_nearest_info(second_rec, metric_key, cursor_map.get("B Current"))

        rows.append(
            (
                f"A Initial {first_rec.point_clean}",
                f"{format_number(a_initial_info[0], 3)} {a_initial_info[2]} @ {pretty_time(a_initial_info[1])}".strip()
                if a_initial_info is not None else "—",
            )
        )
        rows.append(("A Initial Date", pretty_date(a_initial_info[1]) if a_initial_info is not None else "—"))

        rows.append(
            (
                f"A Current {first_rec.point_clean}",
                f"{format_number(a_current_info[0], 3)} {a_current_info[2]} @ {pretty_time(a_current_info[1])}".strip()
                if a_current_info is not None else "—",
            )
        )
        rows.append(("A Current Date", pretty_date(a_current_info[1]) if a_current_info is not None else "—"))

        rows.append(
            (
                f"B Initial {second_rec.point_clean}",
                f"{format_number(b_initial_info[0], 3)} {b_initial_info[2]} @ {pretty_time(b_initial_info[1])}".strip()
                if b_initial_info is not None else "—",
            )
        )
        rows.append(("B Initial Date", pretty_date(b_initial_info[1]) if b_initial_info is not None else "—"))

        rows.append(
            (
                f"B Current {second_rec.point_clean}",
                f"{format_number(b_current_info[0], 3)} {b_current_info[2]} @ {pretty_time(b_current_info[1])}".strip()
                if b_current_info is not None else "—",
            )
        )
        rows.append(("B Current Date", pretty_date(b_current_info[1]) if b_current_info is not None else "—"))

        a_change = safe_percent_change(
            a_initial_info[0] if a_initial_info else None,
            a_current_info[0] if a_current_info else None,
        )
        b_change = safe_percent_change(
            b_initial_info[0] if b_initial_info else None,
            b_current_info[0] if b_current_info else None,
        )

        rows.append(("A Change", f"{format_number(a_change, 2)}%" if a_change is not None else "—"))
        rows.append(("B Change", f"{format_number(b_change, 2)}%" if b_change is not None else "—"))

        _draw_right_info_box(fig, rows)

    fig.update_layout(
        height=640,
        margin=dict(l=46, r=18, t=84, b=40),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f3f4f6",
        font=dict(color="#111827"),
        hovermode="closest",
        dragmode="pan",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.005,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.70)",
            bordercolor="#d1d5db",
            borderwidth=1,
            font=dict(size=11.2),
        ),
        xaxis=dict(
            title="Time",
            range=[x_min_final, x_max_final] if x_min_final is not None and x_max_final is not None else None,
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.18)",
            zeroline=False,
            showline=True,
            linecolor="#9ca3af",
            ticks="outside",
            tickcolor="#6b7280",
            ticklen=4,
            showspikes=True,
            spikecolor="#6b7280",
            spikesnap="cursor",
            spikemode="across",
        ),
        yaxis=dict(
            title=axis_title,
            range=[y_min_final, y_max_final],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.18)",
            zeroline=False,
            showline=True,
            linecolor="#9ca3af",
            ticks="outside",
            tickcolor="#6b7280",
            ticklen=4,
        ),
    )

    return fig


# ------------------------------------------------------------
# Export helpers
# ------------------------------------------------------------
def _build_export_safe_figure(fig: go.Figure) -> go.Figure:
    export_fig = go.Figure()

    for trace in fig.data:
        if isinstance(trace, go.Scattergl):
            trace_json = trace.to_plotly_json()
            export_fig.add_trace(
                go.Scatter(
                    x=np.array(trace_json.get("x")) if trace_json.get("x") is not None else None,
                    y=np.array(trace_json.get("y")) if trace_json.get("y") is not None else None,
                    mode=trace_json.get("mode"),
                    line=trace_json.get("line"),
                    marker=trace_json.get("marker"),
                    fill=trace_json.get("fill"),
                    fillcolor=trace_json.get("fillcolor"),
                    hovertemplate=trace_json.get("hovertemplate"),
                    showlegend=trace_json.get("showlegend"),
                    connectgaps=trace_json.get("connectgaps", False),
                    name=trace_json.get("name"),
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
        trace_json = trace.to_plotly_json()

        if trace_json.get("type") == "scatter":
            mode = trace_json.get("mode", "")

            if "lines" in mode:
                line = dict(trace_json.get("line", {}) or {})
                line["width"] = max(4.8, float(line.get("width", 1.0)) * 2.8)
                trace_json["line"] = line

            if "markers" in mode:
                marker = dict(trace_json.get("marker", {}) or {})
                marker["size"] = max(14, float(marker.get("size", 6)) * 1.9)
                trace_json["marker"] = marker

        new_data.append(go.Scatter(**trace_json))

    fig = go.Figure(data=new_data, layout=fig.layout)

    fig.update_layout(
        width=4200,
        height=2200,
        margin=dict(l=120, r=90, t=360, b=120),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        font=dict(size=30, color="#111827"),
    )

    fig.update_xaxes(
        title_font=dict(size=40),
        tickfont=dict(size=26),
    )
    fig.update_yaxes(
        title_font=dict(size=40),
        tickfont=dict(size=26),
    )

    for shape in fig.layout.shapes:
        if shape.line is not None:
            width = getattr(shape.line, "width", 1) or 1
            shape.line.width = max(2.0, width * 2.2)

    for ann in fig.layout.annotations:
        if ann.font is not None:
            ann.font.size = max(22, int((ann.font.size or 12) * 2.05))

    for img in fig.layout.images:
        sx = getattr(img, "sizex", None)
        sy = getattr(img, "sizey", None)
        if sx is not None:
            img.sizex = sx * 1.22
        if sy is not None:
            img.sizey = sy * 1.22

    return fig


def build_export_png_bytes(fig: go.Figure) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        export_fig = _build_export_safe_figure(fig)
        export_fig = _scale_export_figure(export_fig)

        png_bytes = export_fig.to_image(
            format="png",
            width=4200,
            height=2200,
            scale=2,
        )
        return png_bytes, None
    except Exception as e:
        return None, str(e)


# ------------------------------------------------------------
# Session defaults
# ------------------------------------------------------------
if "trend_signals" not in st.session_state:
    st.session_state["trend_signals"] = {}

if "wm_tr_primary_signal_id" not in st.session_state:
    st.session_state.wm_tr_primary_signal_id = None

if "wm_tr_extra_signal_ids" not in st.session_state:
    st.session_state.wm_tr_extra_signal_ids = []

if "wm_tr_export_png_bytes" not in st.session_state:
    st.session_state.wm_tr_export_png_bytes = None

if "wm_tr_export_png_key" not in st.session_state:
    st.session_state.wm_tr_export_png_key = None

if "wm_tr_export_error" not in st.session_state:
    st.session_state.wm_tr_export_error = None

for key in [
    "wm_tr_cursor_a_initial",
    "wm_tr_cursor_a_current",
    "wm_tr_cursor_b_initial",
    "wm_tr_cursor_b_current",
    "wm_tr_x_manual_start",
    "wm_tr_x_manual_end",
]:
    if key not in st.session_state:
        st.session_state[key] = ""


# ------------------------------------------------------------
# Uploader
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### Trend CSV")
    uploaded_files = st.file_uploader(
        "Upload one or more trend CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="wm_trend_uploader",
    )

    if uploaded_files:
        parsed_records = load_trend_records_from_uploader(uploaded_files)
        trend_store = {rec.trend_id: rec for rec in parsed_records}
        st.session_state["trend_signals"] = trend_store

records_all: List[TrendRecord] = list(st.session_state.get("trend_signals", {}).values())
records_all = sorted(records_all, key=lambda r: (r.machine, r.point_clean, r.file_name))

if not records_all:
    st.warning("Cargue uno o más CSV de tendencia en este módulo. Estos archivos no deben mezclarse con el loader de waveform.")
    st.stop()


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### Signal Selection")

    signal_name_map = {r.display_name: r.trend_id for r in records_all}
    signal_names = list(signal_name_map.keys())

    if st.session_state.wm_tr_primary_signal_id not in [r.trend_id for r in records_all]:
        st.session_state.wm_tr_primary_signal_id = records_all[0].trend_id

    current_primary_name = next(
        (r.display_name for r in records_all if r.trend_id == st.session_state.wm_tr_primary_signal_id),
        signal_names[0],
    )

    selected_primary_name = st.selectbox(
        "Primary signal",
        options=signal_names,
        index=signal_names.index(current_primary_name),
    )
    st.session_state.wm_tr_primary_signal_id = signal_name_map[selected_primary_name]

    extra_options = [name for name in signal_names if name != selected_primary_name]
    default_extra_names = [
        r.display_name
        for r in records_all
        if r.trend_id in st.session_state.wm_tr_extra_signal_ids and r.display_name in extra_options
    ]

    selected_extra_names = st.multiselect(
        "Additional signals",
        options=extra_options,
        default=default_extra_names,
    )
    st.session_state.wm_tr_extra_signal_ids = [signal_name_map[name] for name in selected_extra_names]

    st.markdown("### Trend Processing")
    metric_key = st.selectbox(
        "Metric",
        options=["Amplitude", "Phase", "Speed"],
        index=0,
    )

    show_markers = st.checkbox("Show markers", value=False)
    fill_area = st.checkbox("Fill area (single trend)", value=True)

    st.markdown("### Axes")
    y_axis_mode = st.selectbox(
        "Y-axis scale",
        ["Auto", "Manual"],
        index=0,
    )

    y_axis_manual_min: Optional[float] = None
    y_axis_manual_max: Optional[float] = None
    if y_axis_mode == "Manual":
        c1, c2 = st.columns(2)
        with c1:
            y_axis_manual_min = float(
                st.number_input(
                    "Y min",
                    value=0.0,
                    step=0.1,
                    format="%.3f",
                )
            )
        with c2:
            y_axis_manual_max = float(
                st.number_input(
                    "Y max",
                    value=5.0,
                    step=0.1,
                    format="%.3f",
                )
            )

    x_axis_mode = st.selectbox(
        "X-axis scale",
        ["Auto", "Manual"],
        index=0,
    )

    show_right_info_box = st.checkbox("Show info box", value=True)
    show_legend = st.checkbox("Show legend", value=True)

    st.markdown("### Alarms")
    warning_enabled = st.checkbox("Enable warning line", value=True)
    warning_value: Optional[float] = None
    if warning_enabled:
        warning_value = float(
            st.number_input(
                "Warning value",
                value=3.500,
                step=0.1,
                format="%.3f",
            )
        )

    danger_enabled = st.checkbox("Enable danger line", value=True)
    danger_value: Optional[float] = None
    if danger_enabled:
        danger_value = float(
            st.number_input(
                "Danger value",
                value=5.000,
                step=0.1,
                format="%.3f",
            )
        )


# ------------------------------------------------------------
# Selected records
# ------------------------------------------------------------
selected_ids = [st.session_state.wm_tr_primary_signal_id] + st.session_state.wm_tr_extra_signal_ids
selected_ids = [sid for sid in selected_ids if sid is not None]

selected_records = [r for r in records_all if r.trend_id in selected_ids]

selected_records_sorted: List[TrendRecord] = []
for sid in selected_ids:
    rec = next((r for r in selected_records if r.trend_id == sid), None)
    if rec is not None:
        selected_records_sorted.append(rec)

if not selected_records_sorted:
    st.warning("No valid trend signals selected.")
    st.stop()

logo_uri = get_logo_data_uri(LOGO_PATH)

time_options = get_time_options_for_records(selected_records_sorted, metric_key)
time_labels = [ts_to_label(ts) for ts in time_options]

if not time_labels:
    st.warning("No hay datos válidos para los cursores en la métrica actual.")
    st.stop()


def get_valid_time_label(saved_value: str, fallback_label: str) -> str:
    saved_value = str(saved_value or "")
    if saved_value in time_labels:
        return saved_value
    return fallback_label


default_a_initial = time_labels[0]
default_a_current = time_labels[min(len(time_labels) - 1, max(0, len(time_labels) // 3))]
default_b_initial = time_labels[min(len(time_labels) - 1, max(0, (len(time_labels) * 2) // 3))]
default_b_current = time_labels[-1]
default_x_start = time_labels[0]
default_x_end = time_labels[-1]

st.session_state.wm_tr_cursor_a_initial = get_valid_time_label(st.session_state.wm_tr_cursor_a_initial, default_a_initial)
st.session_state.wm_tr_cursor_a_current = get_valid_time_label(st.session_state.wm_tr_cursor_a_current, default_a_current)
st.session_state.wm_tr_cursor_b_initial = get_valid_time_label(st.session_state.wm_tr_cursor_b_initial, default_b_initial)
st.session_state.wm_tr_cursor_b_current = get_valid_time_label(st.session_state.wm_tr_cursor_b_current, default_b_current)
st.session_state.wm_tr_x_manual_start = get_valid_time_label(st.session_state.wm_tr_x_manual_start, default_x_start)
st.session_state.wm_tr_x_manual_end = get_valid_time_label(st.session_state.wm_tr_x_manual_end, default_x_end)

x_axis_manual_start: Optional[pd.Timestamp] = None
x_axis_manual_end: Optional[pd.Timestamp] = None

st.markdown(
    f'<div class="wm-top-note">Premium trend viewer enabled: multi-variable overlay · metric = <b>{metric_key}</b> · alarm colors reserved exclusively for Warning/Danger · X-axis = <b>{x_axis_mode}</b> · Y-axis = <b>{y_axis_mode}</b>.</div>',
    unsafe_allow_html=True,
)

if x_axis_mode == "Manual":
    with st.expander("X-Axis Manual Window", expanded=True):
        st.markdown(
            """
            <div class="wm-cursor-shell">
                <div class="wm-cursor-title">X-Axis Window</div>
                <div class="wm-cursor-subtitle">Ajusta el inicio y fin del tiempo con sliders precisos, sin escribir fechas manualmente.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            st.select_slider(
                "X Start",
                options=time_labels,
                key="wm_tr_x_manual_start",
            )
        with col_x2:
            st.select_slider(
                "X End",
                options=time_labels,
                key="wm_tr_x_manual_end",
            )

    x_axis_manual_start = label_to_ts(st.session_state.wm_tr_x_manual_start)
    x_axis_manual_end = label_to_ts(st.session_state.wm_tr_x_manual_end)

with st.expander("Cursor Controls", expanded=True):
    st.markdown(
        """
        <div class="wm-cursor-shell">
            <div class="wm-cursor-title">Cursor Controls</div>
            <div class="wm-cursor-subtitle">Referencias temporales finas para comparar crecimiento real de vibración entre puntos y momentos.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.select_slider(
            "A Initial",
            options=time_labels,
            key="wm_tr_cursor_a_initial",
        )
        st.select_slider(
            "A Current",
            options=time_labels,
            key="wm_tr_cursor_a_current",
        )

    with c2:
        st.select_slider(
            "B Initial",
            options=time_labels,
            key="wm_tr_cursor_b_initial",
        )
        st.select_slider(
            "B Current",
            options=time_labels,
            key="wm_tr_cursor_b_current",
        )

cursor_map = {
    "A Initial": label_to_ts(st.session_state.wm_tr_cursor_a_initial),
    "A Current": label_to_ts(st.session_state.wm_tr_cursor_a_current),
    "B Initial": label_to_ts(st.session_state.wm_tr_cursor_b_initial),
    "B Current": label_to_ts(st.session_state.wm_tr_cursor_b_current),
}

fig = build_trend_figure(
    records=selected_records_sorted,
    metric_key=metric_key,
    show_markers=show_markers,
    fill_area=fill_area,
    y_axis_mode=y_axis_mode,
    y_axis_manual_min=y_axis_manual_min,
    y_axis_manual_max=y_axis_manual_max,
    x_axis_mode=x_axis_mode,
    x_axis_manual_start=x_axis_manual_start,
    x_axis_manual_end=x_axis_manual_end,
    warning_enabled=warning_enabled,
    warning_value=warning_value,
    danger_enabled=danger_enabled,
    danger_value=danger_value,
    show_right_info_box=show_right_info_box,
    show_legend=show_legend,
    logo_uri=logo_uri,
    cursor_map=cursor_map,
)


# ------------------------------------------------------------
# Export state
# ------------------------------------------------------------
export_state_key = make_export_state_key(
    [
        metric_key,
        y_axis_mode,
        y_axis_manual_min,
        y_axis_manual_max,
        x_axis_mode,
        st.session_state.wm_tr_x_manual_start,
        st.session_state.wm_tr_x_manual_end,
        warning_enabled,
        warning_value,
        danger_enabled,
        danger_value,
        st.session_state.wm_tr_cursor_a_initial,
        st.session_state.wm_tr_cursor_a_current,
        st.session_state.wm_tr_cursor_b_initial,
        st.session_state.wm_tr_cursor_b_current,
        show_markers,
        fill_area,
        show_right_info_box,
        show_legend,
        "|".join(selected_ids),
        "|".join([r.file_name for r in selected_records_sorted]),
        "|".join([r.point_clean for r in selected_records_sorted]),
    ]
)

if st.session_state.wm_tr_export_png_key != export_state_key:
    st.session_state.wm_tr_export_png_bytes = None
    st.session_state.wm_tr_export_png_key = export_state_key
    st.session_state.wm_tr_export_error = None


# ------------------------------------------------------------
# Export row
# ------------------------------------------------------------
col_export1, col_export2, col_export3 = st.columns([1.35, 1.35, 4.30])

with col_export1:
    if st.button("Prepare PNG HD", use_container_width=True):
        with st.spinner("Generating HD export..."):
            png_bytes, export_error = build_export_png_bytes(fig=fig)
            st.session_state.wm_tr_export_png_bytes = png_bytes
            st.session_state.wm_tr_export_error = export_error

with col_export2:
    if st.session_state.wm_tr_export_png_bytes is not None:
        st.download_button(
            "Download PNG HD",
            data=st.session_state.wm_tr_export_png_bytes,
            file_name="watermelon_trend_hd.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.button("Download PNG HD", disabled=True, use_container_width=True)

with col_export3:
    st.markdown(
        '<div class="wm-export-note">Premium trend view. Cursor A/B with Initial and Current references. Trend Information keeps only cursor diagnostics and percentage change A/B.</div>',
        unsafe_allow_html=True,
    )

if st.session_state.wm_tr_export_error:
    st.warning(f"PNG export error: {st.session_state.wm_tr_export_error}")


# ------------------------------------------------------------
# Main chart
# ------------------------------------------------------------
st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displaylogo": False},
    key="wm_trends_plot_main_view",
)