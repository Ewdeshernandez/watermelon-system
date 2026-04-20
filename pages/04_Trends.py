from __future__ import annotations

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
from plotly.subplots import make_subplots
import streamlit as st

from core.auth import require_login, render_user_menu
from core.trend_diagnostics import build_trend_report_narrative as build_trend_report_narrative_core

st.set_page_config(page_title="Watermelon System | Trends", layout="wide")

require_login()
render_user_menu()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"


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
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            font-family: monospace;
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
        .wm-control-shell {
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.78));
            border: 1px solid #dbe3ee;
            border-radius: 18px;
            padding: 14px 16px 12px 16px;
            margin-bottom: 12px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        .wm-control-title {
            font-size: 1.02rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 2px;
        }
        .wm-control-subtitle {
            font-size: 0.88rem;
            color: #64748b;
            margin-bottom: 10px;
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
        section.main div[data-testid="stButton"] > button *,
        section.main div[data-testid="stDownloadButton"] > button *,
        section.main div[data-testid="stButton"] > button p,
        section.main div[data-testid="stDownloadButton"] > button p,
        section.main div[data-testid="stButton"] > button span,
        section.main div[data-testid="stDownloadButton"] > button span,
        section.main div[data-testid="stButton"] > button div,
        section.main div[data-testid="stDownloadButton"] > button div {
            color: #2563eb !important;
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
    palette = [
        "#5b9cf0", "#10b981", "#8b5cf6", "#06b6d4", "#ec4899",
        "#14b8a6", "#6366f1", "#0f766e", "#7c3aed", "#2563eb",
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


@dataclass
class OperationalRecord:
    op_id: str
    file_name: str
    machine: str = "Operational Data"
    variable: str = ""
    unit: str = ""
    family: str = "generic"
    timestamp_min: Optional[pd.Timestamp] = None
    timestamp_max: Optional[pd.Timestamp] = None
    x_time: pd.Series = field(default_factory=lambda: pd.Series(dtype="datetime64[ns]"))
    y_value: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def display_name(self) -> str:
        unit_txt = f" ({self.unit})" if self.unit else ""
        return f"{self.variable}{unit_txt}"

    @property
    def n_samples(self) -> int:
        return int(len(self.x_time))


def infer_operational_unit(column_name: str, temperature_unit: str) -> str:
    name = str(column_name or "").lower()
    if "mw" in name or "power" in name or "load" in name:
        return "MW"
    if "temp" in name or "temperature" in name or "t48" in name or "t3" in name:
        return temperature_unit
    return ""


def infer_operational_family(column_name: str) -> str:
    name = str(column_name or "").lower()
    if "mw" in name or "power" in name or "load" in name:
        return "power"
    if "temp" in name or "temperature" in name or "t48" in name or "t3" in name:
        return "temperature"
    return "generic"


def parse_operational_csv(uploaded_file, temperature_unit: str = "°F") -> List[OperationalRecord]:
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return []

    if df.empty:
        return []

    timestamp_col = None
    for candidate in ["Timestamp", "Time", "DateTime", "Datetime", "timestamp", "time"]:
        if candidate in df.columns:
            timestamp_col = candidate
            break

    if timestamp_col is None:
        return []

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).copy()
    if df.empty:
        return []

    records: List[OperationalRecord] = []
    machine_name = Path(uploaded_file.name).stem

    for col in df.columns:
        if col == timestamp_col:
            continue

        y = pd.to_numeric(df[col], errors="coerce")
        tmp = pd.DataFrame({"x": df[timestamp_col], "y": y}).dropna(subset=["x", "y"]).copy()
        if tmp.empty:
            continue

        unit = infer_operational_unit(col, temperature_unit)
        family = infer_operational_family(col)

        records.append(
            OperationalRecord(
                op_id=f"operational::{uploaded_file.name}::{col}",
                file_name=uploaded_file.name,
                machine=machine_name,
                variable=str(col),
                unit=unit,
                family=family,
                timestamp_min=safe_datetime(tmp["x"].min()),
                timestamp_max=safe_datetime(tmp["x"].max()),
                x_time=tmp["x"].reset_index(drop=True),
                y_value=tmp["y"].reset_index(drop=True),
            )
        )

    return records


def load_operational_records_from_uploader(files: List[Any], temperature_unit: str = "°F") -> List[OperationalRecord]:
    records: List[OperationalRecord] = []
    for file in files:
        records.extend(parse_operational_csv(file, temperature_unit=temperature_unit))
    return records


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
        "X-Axis Value", "Y-Axis Value", "Y-Axis Status",
        "Phase", "Phase Status", "Speed", "Speed Status",
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
        xref="paper", yref="paper", x=machine_x, y=y_text,
        xanchor="left", yanchor="middle",
        text=f"<b>{trim_text(machine_name, 28)}</b>",
        showarrow=False, font=dict(size=12.8, color="#111827"), align="left",
    )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.325, y=y_text,
        xanchor="center", yanchor="middle",
        text=trim_text(signal_names_text, 34),
        showarrow=False, font=dict(size=11.4, color="#111827"), align="center",
    )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.640, y=y_text,
        xanchor="center", yanchor="middle",
        text=f"Metric: <b>{metric_name}</b> | Latest: <b>{trim_text(latest_text, 32)}</b>",
        showarrow=False, font=dict(size=11.3, color="#111827"), align="center",
    )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.986, y=y_text,
        xanchor="right", yanchor="middle",
        text=trim_text(time_range_text, 28),
        showarrow=False, font=dict(size=11.2, color="#111827"), align="right",
    )


def _draw_right_info_box(fig: go.Figure, rows: List[Tuple[str, str]]) -> None:
    panel_x0 = 0.834
    panel_x1 = 0.976
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
        text="<b>Trend Information</b>",
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
    df = pd.DataFrame({"x": pd.to_datetime(record.x_time, errors="coerce"), "y": pd.to_numeric(metric_series, errors="coerce")}).dropna(subset=["x", "y"])
    if df.empty:
        return df
    return df.sort_values("x").reset_index(drop=True)


def get_cursor_nearest_info(record: TrendRecord, metric_key: str, cursor_ts: Optional[pd.Timestamp]) -> Optional[Tuple[float, pd.Timestamp, str]]:
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


def get_operational_clean_df(record: OperationalRecord) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "x": pd.to_datetime(record.x_time, errors="coerce"),
            "y": pd.to_numeric(record.y_value, errors="coerce"),
        }
    ).dropna(subset=["x", "y"])
    if df.empty:
        return df
    return df.sort_values("x").reset_index(drop=True)


def get_operational_cursor_nearest_info(record: OperationalRecord, cursor_ts: Optional[pd.Timestamp]) -> Optional[Tuple[float, pd.Timestamp, str]]:
    if cursor_ts is None:
        return None
    df = get_operational_clean_df(record)
    if df.empty:
        return None
    idx = (df["x"] - cursor_ts).abs().idxmin()
    row = df.loc[idx]
    return float(row["y"]), pd.Timestamp(row["x"]), record.unit


def get_time_options_for_operational_records(records: List[OperationalRecord]) -> List[pd.Timestamp]:
    ts_values: List[pd.Timestamp] = []
    for record in records:
        df = get_operational_clean_df(record)
        if not df.empty:
            ts_values.extend(list(pd.to_datetime(df["x"], errors="coerce").dropna()))
    if not ts_values:
        return []
    unique_sorted = sorted(pd.Series(ts_values).dropna().unique())
    return [pd.Timestamp(x) for x in unique_sorted]



def align_trend_and_operational_for_correlation(
    trend_record: TrendRecord,
    operational_record: OperationalRecord,
    metric_key: str,
) -> pd.DataFrame:
    trend_df = get_clean_metric_df(trend_record, metric_key)
    op_df = get_operational_clean_df(operational_record)

    if trend_df.empty or op_df.empty:
        return pd.DataFrame(columns=["x", "trend", "operational"])

    trend_df = trend_df.rename(columns={"y": "trend"}).copy()
    op_df = op_df.rename(columns={"y": "operational"}).copy()

    trend_df["x"] = pd.to_datetime(trend_df["x"], errors="coerce")
    op_df["x"] = pd.to_datetime(op_df["x"], errors="coerce")

    trend_df = trend_df.dropna(subset=["x", "trend"]).sort_values("x").reset_index(drop=True)
    op_df = op_df.dropna(subset=["x", "operational"]).sort_values("x").reset_index(drop=True)

    if trend_df.empty or op_df.empty:
        return pd.DataFrame(columns=["x", "trend", "operational"])

    merged = pd.merge_asof(
        trend_df,
        op_df,
        on="x",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )

    merged = merged.dropna(subset=["trend", "operational"]).reset_index(drop=True)
    return merged


def classify_correlation_strength(corr_value: Optional[float]) -> Dict[str, str]:
    if corr_value is None or not math.isfinite(float(corr_value)):
        return {
            "strength": "Nula",
            "direction": "Indeterminada",
            "interpretation": "No fue posible calcular correlación válida entre vibración y variable operativa.",
            "color": "#64748b",
        }

    corr = float(corr_value)
    abs_corr = abs(corr)

    if corr >= 0.0:
        direction = "Positiva"
    else:
        direction = "Negativa"

    if abs_corr >= 0.75:
        strength = "Fuerte"
        color = "#16a34a"
    elif abs_corr >= 0.50:
        strength = "Moderada"
        color = "#f59e0b"
    elif abs_corr >= 0.25:
        strength = "Débil"
        color = "#f97316"
    else:
        strength = "Nula"
        color = "#64748b"

    if strength == "Fuerte" and direction == "Positiva":
        interpretation = "La vibración aumenta cuando aumenta la variable operativa, lo que sugiere influencia operativa importante."
    elif strength == "Fuerte" and direction == "Negativa":
        interpretation = "La vibración disminuye cuando aumenta la variable operativa, indicando relación inversa fuerte."
    elif strength == "Moderada" and direction == "Positiva":
        interpretation = "Existe relación operativa apreciable, aunque no completamente dominante."
    elif strength == "Moderada" and direction == "Negativa":
        interpretation = "Existe relación inversa moderada entre vibración y variable operativa."
    elif strength == "Débil":
        interpretation = "La dependencia operativa es débil; conviene complementar con diagnóstico mecánico."
    else:
        interpretation = "No se observa dependencia operativa clara; la condición podría estar dominada por factores mecánicos o por ruido operacional."

    return {
        "strength": strength,
        "direction": direction,
        "interpretation": interpretation,
        "color": color,
    }


def build_trend_operational_correlation(
    trend_record: Optional[TrendRecord],
    operational_record: Optional[OperationalRecord],
    metric_key: str,
) -> Dict[str, Any]:
    if trend_record is None or operational_record is None:
        return {
            "valid": False,
            "corr_value": None,
            "sample_count": 0,
            "strength": "Nula",
            "direction": "Indeterminada",
            "interpretation": "Seleccione una señal de vibración y una variable operativa para habilitar la correlación.",
            "color": "#64748b",
            "trend_name": trend_record.point_clean if trend_record else "—",
            "operational_name": operational_record.variable if operational_record else "—",
        }

    merged = align_trend_and_operational_for_correlation(
        trend_record=trend_record,
        operational_record=operational_record,
        metric_key=metric_key,
    )

    if len(merged) < 4:
        return {
            "valid": False,
            "corr_value": None,
            "sample_count": int(len(merged)),
            "strength": "Nula",
            "direction": "Indeterminada",
            "interpretation": "No hay suficientes puntos coincidentes en el tiempo para calcular correlación confiable.",
            "color": "#64748b",
            "trend_name": trend_record.point_clean,
            "operational_name": operational_record.variable,
        }

    corr_value = merged["trend"].corr(merged["operational"])
    meta = classify_correlation_strength(corr_value)

    return {
        "valid": True,
        "corr_value": float(corr_value) if corr_value is not None and math.isfinite(float(corr_value)) else None,
        "sample_count": int(len(merged)),
        "strength": meta["strength"],
        "direction": meta["direction"],
        "interpretation": meta["interpretation"],
        "color": meta["color"],
        "trend_name": trend_record.point_clean,
        "operational_name": operational_record.variable,
        "trend_unit": get_metric_series(trend_record, metric_key)[1],
        "operational_unit": operational_record.unit,
        "merged_df": merged,
    }


def build_correlation_scatter_figure(correlation_info: Dict[str, Any]) -> go.Figure:
    fig = go.Figure()

    merged = correlation_info.get("merged_df")
    if merged is None or not isinstance(merged, pd.DataFrame) or merged.empty:
        fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=40, r=40, t=40, b=40),
            title="Correlation Plot",
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=merged["operational"],
            y=merged["trend"],
            mode="markers",
            name="Samples",
            marker=dict(size=8),
            hovertemplate=(
                "Operational: %{x:.4f}<br>"
                "Trend: %{y:.4f}<extra></extra>"
            ),
        )
    )

    if len(merged) >= 2:
        x_vals = merged["operational"].astype(float).to_numpy()
        y_vals = merged["trend"].astype(float).to_numpy()
        if np.isfinite(x_vals).all() and np.isfinite(y_vals).all():
            try:
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(np.min(x_vals), np.max(x_vals), 100)
                y_line = slope * x_line + intercept
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name="Trend line",
                        line=dict(width=2.5),
                    )
                )
            except Exception:
                pass

    trend_name = correlation_info.get("trend_name") or "Trend"
    operational_name = correlation_info.get("operational_name") or "Operational"

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=50, b=50),
        title=f"Correlation: {trend_name} vs {operational_name}",
        xaxis_title=operational_name,
        yaxis_title=trend_name,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig



def _robust_scale(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").dropna().astype(float).to_numpy()
    if arr.size == 0:
        return 1.0
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad > 1e-12:
        return max(1.4826 * mad, 1e-9)
    std = float(np.std(arr))
    if std > 1e-12:
        return max(std, 1e-9)
    return 1.0


def detect_trend_anomalies(record: TrendRecord, metric_key: str) -> pd.DataFrame:
    df = get_clean_metric_df(record, metric_key).copy()
    if df.empty or len(df) < 8:
        return pd.DataFrame(columns=["x", "y", "anomaly_type", "severity", "point_score", "diff_score"])

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    if df.empty or len(df) < 8:
        return pd.DataFrame(columns=["x", "y", "anomaly_type", "severity", "point_score", "diff_score"])

    y_median = float(df["y"].median())
    y_scale = _robust_scale(df["y"])

    diffs = df["y"].diff()
    diff_median = float(diffs.dropna().median()) if diffs.dropna().size else 0.0
    diff_scale = _robust_scale(diffs.dropna()) if diffs.dropna().size else 1.0

    df["point_score"] = (df["y"] - y_median).abs() / max(y_scale, 1e-9)
    df["diff_score"] = (diffs - diff_median).abs() / max(diff_scale, 1e-9)
    df["diff_score"] = df["diff_score"].fillna(0.0)

    anomaly_mask = (df["point_score"] >= 4.5) | (df["diff_score"] >= 5.0)
    anomalies = df.loc[anomaly_mask, ["x", "y", "point_score", "diff_score"]].copy()

    if anomalies.empty:
        return pd.DataFrame(columns=["x", "y", "anomaly_type", "severity", "point_score", "diff_score"])

    def classify_row(row: pd.Series) -> str:
        y_val = float(row["y"])
        diff_score = float(row["diff_score"])
        if y_val > y_median and diff_score >= 5.0:
            return "Spike"
        if y_val < y_median and diff_score >= 5.0:
            return "Drop"
        return "Outlier"

    def classify_severity(row: pd.Series) -> str:
        max_score = max(float(row["point_score"]), float(row["diff_score"]))
        if max_score >= 8.0:
            return "High"
        if max_score >= 6.0:
            return "Medium"
        return "Low"

    anomalies["anomaly_type"] = anomalies.apply(classify_row, axis=1)
    anomalies["severity"] = anomalies.apply(classify_severity, axis=1)

    return anomalies.reset_index(drop=True)




def build_anomaly_table_for_report(records: List[TrendRecord], metric_key: str) -> pd.DataFrame:
    rows = []

    for rec in records:
        df = detect_trend_anomalies(rec, metric_key)
        if df.empty:
            continue

        for _, row in df.iterrows():
            rows.append({
                "Timestamp": row["x"],
                "Signal": rec.point_clean,
                "Value": float(row["y"]),
                "Type": row["anomaly_type"],
                "Severity": row["severity"],
            })

    if not rows:
        return pd.DataFrame(columns=["Timestamp", "Signal", "Value", "Type", "Severity"])

    table = pd.DataFrame(rows)
    table = table.sort_values("Timestamp").reset_index(drop=True)

    return table


def build_anomaly_narrative(records: List[TrendRecord], metric_key: str) -> str:
    all_anomalies = []

    for rec in records:
        df = detect_trend_anomalies(rec, metric_key)
        if not df.empty:
            df = df.copy()
            df["record"] = rec.point_clean
            all_anomalies.append(df)

    if not all_anomalies:
        return "No se identifican eventos anómalos relevantes en la señal dentro de la ventana analizada."

    df_all = pd.concat(all_anomalies, ignore_index=True)

    total = len(df_all)
    spikes = int((df_all["anomaly_type"] == "Spike").sum())
    drops = int((df_all["anomaly_type"] == "Drop").sum())
    outliers = int((df_all["anomaly_type"] == "Outlier").sum())

    high = int((df_all["severity"] == "High").sum())
    medium = int((df_all["severity"] == "Medium").sum())
    low = int((df_all["severity"] == "Low").sum())

    # ------------------------------------------------------------
    # Clasificación de comportamiento
    # ------------------------------------------------------------
    if total >= 15:
        pattern = "recurrente"
    elif total >= 6:
        pattern = "intermitente"
    else:
        pattern = "aislado"

    # ------------------------------------------------------------
    # Tipo dominante
    # ------------------------------------------------------------
    if spikes > drops and spikes > outliers:
        dominant = "spikes (incrementos abruptos)"
    elif drops > spikes and drops > outliers:
        dominant = "drops (caídas abruptas)"
    else:
        dominant = "outliers dispersos"

    # ------------------------------------------------------------
    # Severidad dominante
    # ------------------------------------------------------------
    if high > 0:
        severity_text = "con presencia de eventos de alta severidad"
    elif medium > 0:
        severity_text = "con eventos de severidad moderada"
    else:
        severity_text = "predominantemente de baja severidad"

    # ------------------------------------------------------------
    # Interpretación técnica
    # ------------------------------------------------------------
    if pattern == "recurrente":
        interpretation = (
            "La recurrencia de eventos anómalos sugiere un comportamiento no aleatorio, "
            "posiblemente asociado a condiciones operativas repetitivas o a una condición mecánica persistente."
        )
    elif pattern == "intermitente":
        interpretation = (
            "Los eventos anómalos aparecen de forma intermitente, lo que puede estar asociado "
            "a cambios operativos, transitorios o perturbaciones externas."
        )
    else:
        interpretation = (
            "Los eventos detectados son aislados, sin patrón repetitivo claro, "
            "posiblemente asociados a ruido o perturbaciones puntuales."
        )

    # ------------------------------------------------------------
    # Construcción final
    # ------------------------------------------------------------
    narrative = (
        f"Se detectaron {total} eventos anómalos en la señal, clasificados como comportamiento {pattern}, "
        f"con predominio de {dominant} y {severity_text}. "
        f"{interpretation}"
    )

    return narrative


def build_panel_anomaly_summary(records: List[TrendRecord], metric_key: str) -> Dict[str, Any]:
    total_count = 0
    affected_records = 0
    top_severity = "None"
    details: List[Dict[str, Any]] = []

    severity_rank = {"None": 0, "Low": 1, "Medium": 2, "High": 3}

    for rec in records:
        anomalies = detect_trend_anomalies(rec, metric_key)
        count = int(len(anomalies))
        if count > 0:
            affected_records += 1
            total_count += count
            local_top = "Low"
            if "High" in set(anomalies["severity"]):
                local_top = "High"
            elif "Medium" in set(anomalies["severity"]):
                local_top = "Medium"

            if severity_rank.get(local_top, 0) > severity_rank.get(top_severity, 0):
                top_severity = local_top

            details.append(
                {
                    "record_name": rec.point_clean,
                    "count": count,
                    "top_severity": local_top,
                }
            )

    if total_count == 0:
        interpretation = "No se detectaron anomalías puntuales relevantes en la señal dentro de la ventana mostrada."
        color = "#16a34a"
    elif top_severity == "High":
        interpretation = "Se detectaron anomalías de alta severidad. Conviene revisar eventos transitorios, instrumentación o condición mecánica local."
        color = "#dc2626"
    elif top_severity == "Medium":
        interpretation = "Se detectaron anomalías moderadas. Conviene revisar cambios operativos o perturbaciones puntuales."
        color = "#f59e0b"
    else:
        interpretation = "Se detectaron anomalías leves y aisladas. Mantener seguimiento y correlacionar con operación."
        color = "#f97316"

    return {
        "total_count": total_count,
        "affected_records": affected_records,
        "top_severity": top_severity,
        "interpretation": interpretation,
        "color": color,
        "details": details,
    }



def build_lagged_correlation_analysis(
    trend_record: Optional[TrendRecord],
    operational_record: Optional[OperationalRecord],
    metric_key: str,
    max_lag_minutes: int = 180,
    step_minutes: int = 10,
) -> Dict[str, Any]:
    if trend_record is None or operational_record is None:
        return {
            "valid": False,
            "best_corr": None,
            "best_lag_min": None,
            "direction": "Indeterminada",
            "strength": "Nula",
            "interpretation": "Seleccione una señal de vibración y una variable operativa para habilitar el análisis con desfase.",
            "lag_df": pd.DataFrame(columns=["lag_min", "corr"]),
            "color": "#64748b",
        }

    base_df = align_trend_and_operational_for_correlation(
        trend_record=trend_record,
        operational_record=operational_record,
        metric_key=metric_key,
    )

    if base_df.empty or len(base_df) < 6:
        return {
            "valid": False,
            "best_corr": None,
            "best_lag_min": None,
            "direction": "Indeterminada",
            "strength": "Nula",
            "interpretation": "No hay suficientes puntos coincidentes para analizar correlación con desfase.",
            "lag_df": pd.DataFrame(columns=["lag_min", "corr"]),
            "color": "#64748b",
        }

    trend_df = get_clean_metric_df(trend_record, metric_key).rename(columns={"y": "trend"}).copy()
    op_df = get_operational_clean_df(operational_record).rename(columns={"y": "operational"}).copy()

    trend_df["x"] = pd.to_datetime(trend_df["x"], errors="coerce")
    op_df["x"] = pd.to_datetime(op_df["x"], errors="coerce")

    trend_df = trend_df.dropna(subset=["x", "trend"]).sort_values("x").reset_index(drop=True)
    op_df = op_df.dropna(subset=["x", "operational"]).sort_values("x").reset_index(drop=True)

    lag_rows = []
    for lag_min in range(-max_lag_minutes, max_lag_minutes + 1, step_minutes):
        shifted = op_df.copy()
        shifted["x"] = shifted["x"] + pd.Timedelta(minutes=lag_min)

        merged = pd.merge_asof(
            trend_df,
            shifted,
            on="x",
            direction="nearest",
            tolerance=pd.Timedelta("30min"),
        ).dropna(subset=["trend", "operational"]).reset_index(drop=True)

        corr_val = None
        if len(merged) >= 6:
            try:
                c = merged["trend"].corr(merged["operational"])
                if c is not None and math.isfinite(float(c)):
                    corr_val = float(c)
            except Exception:
                corr_val = None

        lag_rows.append(
            {
                "lag_min": lag_min,
                "corr": corr_val,
                "samples": int(len(merged)),
            }
        )

    lag_df = pd.DataFrame(lag_rows)
    valid_df = lag_df.dropna(subset=["corr"]).copy()

    if valid_df.empty:
        return {
            "valid": False,
            "best_corr": None,
            "best_lag_min": None,
            "direction": "Indeterminada",
            "strength": "Nula",
            "interpretation": "No fue posible calcular correlaciones válidas en la ventana de desfases.",
            "lag_df": lag_df,
            "color": "#64748b",
        }

    best_idx = valid_df["corr"].abs().idxmax()
    best_row = valid_df.loc[best_idx]
    best_corr = float(best_row["corr"])
    best_lag = int(best_row["lag_min"])

    meta = classify_correlation_strength(best_corr)

    if abs(best_lag) <= step_minutes:
        lag_meaning = "La relación parece prácticamente simultánea entre vibración y variable operativa."
    elif best_lag > 0:
        lag_meaning = (
            f"La mejor correlación aparece con un desfase de +{best_lag} min, "
            "lo que sugiere que la variable operativa antecede la respuesta vibratoria."
        )
    else:
        lag_meaning = (
            f"La mejor correlación aparece con un desfase de {best_lag} min, "
            "lo que sugiere que la vibración antecede a la variable operativa o que existe inversión temporal en el comportamiento."
        )

    interpretation = f"{meta['interpretation']} {lag_meaning}"

    return {
        "valid": True,
        "best_corr": best_corr,
        "best_lag_min": best_lag,
        "direction": meta["direction"],
        "strength": meta["strength"],
        "interpretation": interpretation,
        "lag_df": lag_df,
        "color": meta["color"],
        "trend_name": trend_record.point_clean,
        "operational_name": operational_record.variable,
    }


def build_lag_correlation_figure(lag_info: Dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    lag_df = lag_info.get("lag_df")

    if lag_df is None or not isinstance(lag_df, pd.DataFrame) or lag_df.empty:
        fig.update_layout(
            template="plotly_white",
            height=360,
            margin=dict(l=40, r=40, t=40, b=40),
            title="Lag Correlation",
        )
        return fig

    valid_df = lag_df.dropna(subset=["corr"]).copy()

    fig.add_trace(
        go.Scatter(
            x=lag_df["lag_min"],
            y=lag_df["corr"],
            mode="lines+markers",
            name="Correlation vs lag",
            line=dict(width=2.5),
            marker=dict(size=7),
            hovertemplate="Lag: %{x} min<br>Correlation: %{y:.4f}<extra></extra>",
        )
    )

    if not valid_df.empty:
        best_idx = valid_df["corr"].abs().idxmax()
        best_row = valid_df.loc[best_idx]
        fig.add_trace(
            go.Scatter(
                x=[best_row["lag_min"]],
                y=[best_row["corr"]],
                mode="markers",
                name="Best lag",
                marker=dict(size=13, color="#ef4444", symbol="diamond"),
                hovertemplate="Best lag: %{x} min<br>Correlation: %{y:.4f}<extra></extra>",
            )
        )

    trend_name = lag_info.get("trend_name") or "Trend"
    operational_name = lag_info.get("operational_name") or "Operational"

    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=40, r=40, t=50, b=50),
        title=f"Lag Correlation: {trend_name} vs {operational_name}",
        xaxis_title="Lag (minutes)",
        yaxis_title="Correlation",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    fig.add_hline(y=0.0, line_dash="dot", line_color="#94a3b8", line_width=1.4)
    return fig


def build_trend_figure(
    records: List[TrendRecord],
    metric_key: str,
    show_markers: bool,
    show_anomaly_markers: bool,
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
    operational_records: Optional[List[OperationalRecord]] = None,
    mixed_mode: bool = False,
    operational_only_mode: bool = False,
    operational_y_axis_mode: str = "Auto",
    operational_y_manual_min: Optional[float] = None,
    operational_y_manual_max: Optional[float] = None,
) -> go.Figure:
    operational_records = operational_records or []
    use_secondary_axis = mixed_mode and len(records) > 0 and len(operational_records) > 0
    fig = make_subplots(specs=[[{"secondary_y": True}]]) if use_secondary_axis else go.Figure()

    visible_records: List[Tuple[TrendRecord, pd.DataFrame, str]] = []
    visible_operational_records: List[Tuple[OperationalRecord, pd.DataFrame]] = []

    global_y_min = np.inf
    global_y_max = -np.inf
    global_x_min: Optional[pd.Timestamp] = None
    global_x_max: Optional[pd.Timestamp] = None

    operational_y_min = np.inf
    operational_y_max = -np.inf

    if not operational_only_mode:
        for idx, record in enumerate(records):
            df = get_clean_metric_df(record, metric_key)
            metric_unit = get_metric_series(record, metric_key)[1]
            if df.empty:
                continue

            color = color_for_index(idx)
            mode = "lines+markers" if show_markers else "lines"

            trace = go.Scattergl(
                x=df["x"],
                y=df["y"],
                mode=mode,
                line=dict(width=2.4, color=color),
                marker=dict(size=5, color=color),
                fill="tozeroy" if fill_area and len(records) == 1 and not use_secondary_axis else None,
                fillcolor="rgba(91, 156, 240, 0.10)" if fill_area and len(records) == 1 and not use_secondary_axis else None,
                name=record.point_clean,
                hovertemplate=("Point: %{fullData.name}<br>" "Time: %{x}<br>" f"{metric_key}: " + "%{y:.4f} " + f"{metric_unit}" + "<extra></extra>"),
                showlegend=show_legend,
                connectgaps=False,
            )

            if use_secondary_axis:
                fig.add_trace(trace, secondary_y=False)
            else:
                fig.add_trace(trace)

            if show_anomaly_markers:
                anomaly_df = detect_trend_anomalies(record, metric_key)
                if not anomaly_df.empty:
                    anomaly_trace = go.Scatter(
                        x=anomaly_df["x"],
                        y=anomaly_df["y"],
                        mode="markers",
                        name=f"Anomalies — {record.point_clean}",
                        marker=dict(
                            size=11,
                            color="#ef4444",
                            symbol="x",
                            line=dict(width=1.0, color="#7f1d1d"),
                        ),
                        hovertemplate=(
                            "Point: %{fullData.name}<br>"
                            "Time: %{x}<br>"
                            "Value: %{y:.4f}<br>"
                            "Anomaly detected<extra></extra>"
                        ),
                        showlegend=show_legend,
                    )
                    if use_secondary_axis:
                        fig.add_trace(anomaly_trace, secondary_y=False)
                    else:
                        fig.add_trace(anomaly_trace)

            y_min_local = float(df["y"].min())
            y_max_local = float(df["y"].max())
            global_y_min = min(global_y_min, y_min_local)
            global_y_max = max(global_y_max, y_max_local)

            x_min_local = pd.Timestamp(df["x"].min())
            x_max_local = pd.Timestamp(df["x"].max())
            global_x_min = x_min_local if global_x_min is None else min(global_x_min, x_min_local)
            global_x_max = x_max_local if global_x_max is None else max(global_x_max, x_max_local)

            visible_records.append((record, df, metric_unit))

    operational_start_idx = len(records)
    for idx, record in enumerate(operational_records):
        df = get_operational_clean_df(record)
        if df.empty:
            continue

        color = color_for_index(operational_start_idx + idx)
        mode = "lines+markers" if show_markers else "lines"
        trace = go.Scattergl(
            x=df["x"],
            y=df["y"],
            mode=mode,
            line=dict(width=2.2, color=color, dash="dot" if mixed_mode else "solid"),
            marker=dict(size=5, color=color),
            name=record.variable,
            hovertemplate=("Signal: %{fullData.name}<br>" "Time: %{x}<br>" "Value: %{y:.4f} " + f"{record.unit}" + "<extra></extra>"),
            showlegend=show_legend,
            connectgaps=False,
        )

        if use_secondary_axis:
            fig.add_trace(trace, secondary_y=True)
        else:
            fig.add_trace(trace)

        y_min_local = float(df["y"].min())
        y_max_local = float(df["y"].max())
        if use_secondary_axis or operational_only_mode:
            operational_y_min = min(operational_y_min, y_min_local)
            operational_y_max = max(operational_y_max, y_max_local)
        else:
            global_y_min = min(global_y_min, y_min_local)
            global_y_max = max(global_y_max, y_max_local)

        x_min_local = pd.Timestamp(df["x"].min())
        x_max_local = pd.Timestamp(df["x"].max())
        global_x_min = x_min_local if global_x_min is None else min(global_x_min, x_min_local)
        global_x_max = x_max_local if global_x_max is None else max(global_x_max, x_max_local)

        visible_operational_records.append((record, df))

    if not visible_records and not visible_operational_records:
        fig.update_layout(height=640, plot_bgcolor="#f8fafc", paper_bgcolor="#f3f4f6", margin=dict(l=46, r=18, t=84, b=40))
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="No valid trend data available", showarrow=False, font=dict(size=18, color="#6b7280"))
        return fig

    def _pad_axis(ymin: float, ymax: float) -> Tuple[float, float]:
        if not math.isfinite(ymin):
            ymin = 0.0
        if not math.isfinite(ymax):
            ymax = 1.0
        if math.isclose(ymin, ymax, rel_tol=1e-12, abs_tol=1e-12):
            base_pad = max(abs(ymax) * 0.10, 0.25)
            ymin -= base_pad
            ymax += base_pad
        else:
            base_pad = max((ymax - ymin) * 0.12, 0.10)
            ymin -= base_pad
            ymax += base_pad
        return ymin, ymax

    if x_axis_mode == "Manual" and x_axis_manual_start is not None and x_axis_manual_end is not None and x_axis_manual_start < x_axis_manual_end:
        x_min_final = x_axis_manual_start
        x_max_final = x_axis_manual_end
    else:
        x_min_final = global_x_min
        x_max_final = global_x_max

    if use_secondary_axis:
        y1_min_final, y1_max_final = _pad_axis(global_y_min, global_y_max)
        y2_min_final, y2_max_final = _pad_axis(operational_y_min, operational_y_max)

        if y_axis_mode == "Manual" and y_axis_manual_min is not None and y_axis_manual_max is not None:
            y1_min_final = float(y_axis_manual_min)
            y1_max_final = float(y_axis_manual_max)
            if y1_min_final >= y1_max_final:
                y1_min_final, y1_max_final = min(y1_min_final, y1_max_final), max(y1_min_final, y1_max_final) + 1.0

        if operational_y_axis_mode == "Manual" and operational_y_manual_min is not None and operational_y_manual_max is not None:
            y2_min_final = float(operational_y_manual_min)
            y2_max_final = float(operational_y_manual_max)
            if y2_min_final >= y2_max_final:
                y2_min_final, y2_max_final = min(y2_min_final, y2_max_final), max(y2_min_final, y2_max_final) + 1.0
    else:
        base_ymin = operational_y_min if operational_only_mode else global_y_min
        base_ymax = operational_y_max if operational_only_mode else global_y_max
        y_min_final, y_max_final = _pad_axis(base_ymin, base_ymax)
        if warning_enabled and warning_value is not None and math.isfinite(float(warning_value)) and not operational_only_mode:
            y_max_final = max(y_max_final, float(warning_value) * 1.08)
        if danger_enabled and danger_value is not None and math.isfinite(float(danger_value)) and not operational_only_mode:
            y_max_final = max(y_max_final, float(danger_value) * 1.08)
        if y_axis_mode == "Manual" and y_axis_manual_min is not None and y_axis_manual_max is not None:
            y_min_final = float(y_axis_manual_min)
            y_max_final = float(y_axis_manual_max)
            if y_min_final >= y_max_final:
                y_min_final, y_max_final = min(y_min_final, y_max_final), max(y_min_final, y_max_final) + 1.0

    if warning_enabled and warning_value is not None and math.isfinite(float(warning_value)) and not operational_only_mode:
        fig.add_hline(
            y=float(warning_value), line_width=1.8, line_dash="dash", line_color="#f59e0b",
            annotation_text=f"Warning {format_number(warning_value, 3)}",
            annotation_position="top left", annotation_font_color="#92400e",
        )

    if danger_enabled and danger_value is not None and math.isfinite(float(danger_value)) and not operational_only_mode:
        fig.add_hline(
            y=float(danger_value), line_width=1.9, line_dash="dash", line_color="#ef4444",
            annotation_text=f"Danger {format_number(danger_value, 3)}",
            annotation_position="top left", annotation_font_color="#991b1b",
        )

    cursor_line_specs = {"A Initial": "#334155", "A Current": "#64748b", "B Initial": "#111827", "B Current": "#475569"}
    for label, ts in cursor_map.items():
        if ts is not None:
            fig.add_vline(x=ts, line_width=1.8, line_dash="dot", line_color=cursor_line_specs.get(label, "#475569"))

    if visible_records:
        machine_name = visible_records[0][0].machine
        signal_names_text = " | ".join([r.point_clean for r, _, _ in visible_records[:2]])
        if len(visible_records) > 2:
            signal_names_text += f" +{len(visible_records) - 2}"
    else:
        machine_name = visible_operational_records[0][0].machine
        signal_names_text = " | ".join([r.variable for r, _ in visible_operational_records[:2]])
        if len(visible_operational_records) > 2:
            signal_names_text += f" +{len(visible_operational_records) - 2}"

    if mixed_mode and visible_operational_records:
        signal_names_text = f"{signal_names_text} + Operational"

    latest_values: List[str] = []
    if visible_records:
        for rec, df, unit in visible_records[:2]:
            if not df.empty:
                latest_values.append(f"{rec.point_clean}: {format_number(df['y'].iloc[-1], 3)} {unit}".strip())
    elif visible_operational_records:
        for rec, df in visible_operational_records[:2]:
            if not df.empty:
                latest_values.append(f"{rec.variable}: {format_number(df['y'].iloc[-1], 3)} {rec.unit}".strip())
    latest_text = " | ".join(latest_values) if latest_values else "—"

    if operational_only_mode:
        unit_for_axis = visible_operational_records[0][0].unit if visible_operational_records else ""
        axis_title = f"Operational Data ({unit_for_axis})" if unit_for_axis else "Operational Data"
    else:
        unit_for_axis = visible_records[0][2] if visible_records else ""
        axis_title = f"{metric_key} ({unit_for_axis})" if unit_for_axis else metric_key

    operational_axis_title = ""
    if visible_operational_records:
        families = sorted(set(rec.family for rec, _ in visible_operational_records))
        units = sorted(set(rec.unit for rec, _ in visible_operational_records if rec.unit))
        if "power" in families and len(families) == 1:
            operational_axis_title = "Load / Power (MW)"
        elif "temperature" in families and len(families) == 1:
            operational_axis_title = f"Temperature ({units[0]})" if units else "Temperature"
        else:
            operational_axis_title = "Operational Data"

    time_range_text = "—"
    if global_x_min is not None and global_x_max is not None:
        time_range_text = f"{global_x_min.strftime('%Y-%m-%d %H:%M')} → {global_x_max.strftime('%Y-%m-%d %H:%M')}"

    metric_header_name = "Operational Data" if operational_only_mode else metric_key
    if mixed_mode and operational_axis_title:
        metric_header_name = f"{metric_key} + {operational_axis_title}"

    _draw_top_strip(fig, machine_name, signal_names_text, metric_header_name, latest_text, logo_uri, time_range_text)

    if show_right_info_box:
        rows: List[Tuple[str, str]] = []
        if visible_records:
            first_rec = visible_records[0][0]
            second_rec = visible_records[1][0] if len(visible_records) >= 2 else first_rec

            a_initial_info = get_cursor_nearest_info(first_rec, metric_key, cursor_map.get("A Initial"))
            a_current_info = get_cursor_nearest_info(first_rec, metric_key, cursor_map.get("A Current"))
            b_initial_info = get_cursor_nearest_info(second_rec, metric_key, cursor_map.get("B Initial"))
            b_current_info = get_cursor_nearest_info(second_rec, metric_key, cursor_map.get("B Current"))

            rows.append((f"A Initial {first_rec.point_clean}", f"{format_number(a_initial_info[0], 3)} {a_initial_info[2]} @ {pretty_time(a_initial_info[1])}".strip() if a_initial_info else "—"))
            rows.append(("A Initial Date", pretty_date(a_initial_info[1]) if a_initial_info else "—"))
            rows.append((f"A Current {first_rec.point_clean}", f"{format_number(a_current_info[0], 3)} {a_current_info[2]} @ {pretty_time(a_current_info[1])}".strip() if a_current_info else "—"))
            rows.append(("A Current Date", pretty_date(a_current_info[1]) if a_current_info else "—"))
            rows.append((f"B Initial {second_rec.point_clean}", f"{format_number(b_initial_info[0], 3)} {b_initial_info[2]} @ {pretty_time(b_initial_info[1])}".strip() if b_initial_info else "—"))
            rows.append(("B Initial Date", pretty_date(b_initial_info[1]) if b_initial_info else "—"))
            rows.append((f"B Current {second_rec.point_clean}", f"{format_number(b_current_info[0], 3)} {b_current_info[2]} @ {pretty_time(b_current_info[1])}".strip() if b_current_info else "—"))
            rows.append(("B Current Date", pretty_date(b_current_info[1]) if b_current_info else "—"))

            a_change = safe_percent_change(a_initial_info[0] if a_initial_info else None, a_current_info[0] if a_current_info else None)
            b_change = safe_percent_change(b_initial_info[0] if b_initial_info else None, b_current_info[0] if b_current_info else None)
            rows.append(("A Change", f"{format_number(a_change, 2)}%" if a_change is not None else "—"))
            rows.append(("B Change", f"{format_number(b_change, 2)}%" if b_change is not None else "—"))

        if mixed_mode and visible_operational_records:
            op_rec = visible_operational_records[0][0]
            op_a_initial = get_operational_cursor_nearest_info(op_rec, cursor_map.get("A Initial"))
            op_a_current = get_operational_cursor_nearest_info(op_rec, cursor_map.get("A Current"))
            op_change = safe_percent_change(op_a_initial[0] if op_a_initial else None, op_a_current[0] if op_a_current else None)
            rows.append((f"Op Initial {trim_text(op_rec.variable, 18)}", f"{format_number(op_a_initial[0], 3)} {op_rec.unit} @ {pretty_time(op_a_initial[1])}".strip() if op_a_initial else "—"))
            rows.append((f"Op Current {trim_text(op_rec.variable, 18)}", f"{format_number(op_a_current[0], 3)} {op_rec.unit} @ {pretty_time(op_a_current[1])}".strip() if op_a_current else "—"))
            rows.append(("Op Change", f"{format_number(op_change, 2)}%" if op_change is not None else "—"))

        if operational_only_mode and visible_operational_records:
            first_op = visible_operational_records[0][0]
            second_op = visible_operational_records[1][0] if len(visible_operational_records) >= 2 else first_op
            a_initial_info = get_operational_cursor_nearest_info(first_op, cursor_map.get("A Initial"))
            a_current_info = get_operational_cursor_nearest_info(first_op, cursor_map.get("A Current"))
            b_initial_info = get_operational_cursor_nearest_info(second_op, cursor_map.get("B Initial"))
            b_current_info = get_operational_cursor_nearest_info(second_op, cursor_map.get("B Current"))
            rows.extend([
                (f"A Initial {trim_text(first_op.variable, 18)}", f"{format_number(a_initial_info[0], 3)} {first_op.unit} @ {pretty_time(a_initial_info[1])}".strip() if a_initial_info else "—"),
                ("A Initial Date", pretty_date(a_initial_info[1]) if a_initial_info else "—"),
                (f"A Current {trim_text(first_op.variable, 18)}", f"{format_number(a_current_info[0], 3)} {first_op.unit} @ {pretty_time(a_current_info[1])}".strip() if a_current_info else "—"),
                ("A Current Date", pretty_date(a_current_info[1]) if a_current_info else "—"),
                (f"B Initial {trim_text(second_op.variable, 18)}", f"{format_number(b_initial_info[0], 3)} {second_op.unit} @ {pretty_time(b_initial_info[1])}".strip() if b_initial_info else "—"),
                ("B Initial Date", pretty_date(b_initial_info[1]) if b_initial_info else "—"),
                (f"B Current {trim_text(second_op.variable, 18)}", f"{format_number(b_current_info[0], 3)} {second_op.unit} @ {pretty_time(b_current_info[1])}".strip() if b_current_info else "—"),
                ("B Current Date", pretty_date(b_current_info[1]) if b_current_info else "—"),
            ])

        if rows:
            _draw_right_info_box(fig, rows)

    if use_secondary_axis:
        fig.update_layout(
            height=640,
            margin=dict(l=46, r=18, t=84, b=40),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#f3f4f6",
            font=dict(color="#111827"),
            hovermode="closest",
            dragmode="pan",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0.0,
                bgcolor="rgba(255,255,255,0.70)", bordercolor="#d1d5db", borderwidth=1,
                font=dict(size=11.2),
            ),
        )
        fig.update_xaxes(
            title="Time",
            range=[x_min_final, x_max_final] if x_min_final is not None and x_max_final is not None else None,
            showgrid=True, gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False,
            showline=True, linecolor="#9ca3af", ticks="outside", tickcolor="#6b7280", ticklen=4,
            showspikes=True, spikecolor="#6b7280", spikesnap="cursor", spikemode="across",
        )
        fig.update_yaxes(
            title_text=axis_title,
            range=[y1_min_final, y1_max_final],
            showgrid=True, gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False,
            showline=True, linecolor="#9ca3af", ticks="outside", tickcolor="#6b7280", ticklen=4,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text=operational_axis_title or "Operational Data",
            range=[y2_min_final, y2_max_final],
            showgrid=False, zeroline=False,
            showline=True, linecolor="#9ca3af", ticks="outside", tickcolor="#6b7280", ticklen=4,
            secondary_y=True,
        )
    else:
        fig.update_layout(
            height=640,
            margin=dict(l=46, r=18, t=84, b=40),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#f3f4f6",
            font=dict(color="#111827"),
            hovermode="closest",
            dragmode="pan",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0.0,
                bgcolor="rgba(255,255,255,0.70)", bordercolor="#d1d5db", borderwidth=1,
                font=dict(size=11.2),
            ),
            xaxis=dict(
                title="Time",
                range=[x_min_final, x_max_final] if x_min_final is not None and x_max_final is not None else None,
                showgrid=True, gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False,
                showline=True, linecolor="#9ca3af", ticks="outside", tickcolor="#6b7280", ticklen=4,
                showspikes=True, spikecolor="#6b7280", spikesnap="cursor", spikemode="across",
            ),
            yaxis=dict(
                title=axis_title,
                range=[y_min_final, y_max_final],
                showgrid=True, gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False,
                showline=True, linecolor="#9ca3af", ticks="outside", tickcolor="#6b7280", ticklen=4,
            ),
        )

    return fig



def _build_export_safe_figure(fig: go.Figure) -> go.Figure:
    export_fig = go.Figure()

    for trace in fig.data:
        trace_json = trace.to_plotly_json()

        # Convertir Scattergl -> Scatter preservando eje secundario y metadatos
        if trace_json.get("type") == "scattergl":
            trace_json["type"] = "scatter"

        export_fig.add_trace(go.Scatter(**trace_json))

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

    has_secondary_y = getattr(fig.layout, "yaxis2", None) is not None
    has_right_info_box = any(
        getattr(ann, "xref", None) == "paper" and float(getattr(ann, "x", 0) or 0) >= 0.83
        for ann in fig.layout.annotations
    )

    export_width = 4200
    export_height = 2200
    margin_left = 120
    margin_right = 90
    margin_top = 360
    margin_bottom = 120

    if has_secondary_y:
        export_width = 4700
        margin_right = 260

    if has_right_info_box:
        export_width = max(export_width, 4900)
        margin_right = max(margin_right, 320)

    fig.update_layout(
        width=export_width,
        height=export_height,
        margin=dict(l=margin_left, r=margin_right, t=margin_top, b=margin_bottom),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        font=dict(size=30, color="#111827"),
    )

    fig.update_xaxes(title_font=dict(size=40), tickfont=dict(size=26))
    fig.update_yaxes(title_font=dict(size=40), tickfont=dict(size=26))

    if has_secondary_y:
        yaxis2_cfg = dict(fig.layout.yaxis2.to_plotly_json()) if getattr(fig.layout, "yaxis2", None) is not None else {}
        yaxis2_cfg.update(
            dict(
                automargin=False,
                side="right",
                overlaying="y",
                anchor="free",
                position=0.80,
                ticks="outside",
                tickfont=dict(size=26, color="#111827"),
                title_font=dict(size=40, color="#111827"),
                showline=True,
                linecolor="#9ca3af",
                tickcolor="#6b7280",
                ticklen=6,
                zeroline=False,
                showgrid=False,
            )
        )
        fig.update_layout(yaxis2=yaxis2_cfg)

    if has_right_info_box:
        xaxis_cfg = dict(fig.layout.xaxis.to_plotly_json()) if getattr(fig.layout, "xaxis", None) is not None else {}
        xaxis_cfg["domain"] = [0.0, 0.72]
        fig.update_layout(xaxis=xaxis_cfg)

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
        png_bytes = export_fig.to_image(format="png", width=4200, height=2200, scale=2)
        return png_bytes, None
    except Exception as e:
        return None, str(e)


def _sanitize_series_for_analysis(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if arr.size == 0:
        return np.array([], dtype=float)
    return arr[np.isfinite(arr)]


def _classify_trend_behavior(values: pd.Series) -> Dict[str, Any]:
    arr = _sanitize_series_for_analysis(values)
    result: Dict[str, Any] = {
        "classification": "insufficient",
        "slope_ratio": None,
        "change_pct": None,
        "volatility_ratio": None,
        "jerk_ratio": None,
        "sample_count": int(arr.size),
    }
    if arr.size < 3:
        return result

    x = np.arange(arr.size, dtype=float)
    slope, intercept = np.polyfit(x, arr, 1)
    fitted = slope * x + intercept
    residual = arr - fitted

    mean_abs = float(np.mean(np.abs(arr)))
    value_span = float(np.max(arr) - np.min(arr))
    scale = max(mean_abs, value_span, 1e-9)

    slope_ratio = float(abs(slope) * max(arr.size - 1, 1) / scale)
    volatility_ratio = float(np.std(residual) / scale)
    diffs = np.diff(arr)
    jerk_ratio = float(np.std(diffs) / scale) if diffs.size else 0.0
    change_pct = safe_percent_change(float(arr[0]), float(arr[-1]))

    direction = "up" if slope > 0 else "down"
    classification = "stable"
    if jerk_ratio >= 0.28 or volatility_ratio >= 0.22:
        classification = "abrupt"
    elif slope_ratio >= 0.18 and direction == "up":
        classification = "progressive_increase"
    elif slope_ratio >= 0.18 and direction == "down":
        classification = "progressive_decrease"

    result.update(
        {
            "classification": classification,
            "direction": direction,
            "slope_ratio": slope_ratio,
            "change_pct": change_pct,
            "volatility_ratio": volatility_ratio,
            "jerk_ratio": jerk_ratio,
            "initial_value": float(arr[0]),
            "final_value": float(arr[-1]),
            "min_value": float(np.min(arr)),
            "max_value": float(np.max(arr)),
            "mean_value": float(np.mean(arr)),
        }
    )
    return result


def _trend_unit_for_metric(record: TrendRecord, metric_key: str) -> str:
    return get_metric_series(record, metric_key)[1]


def _build_single_trend_narrative(record: TrendRecord, metric_key: str) -> str:
    df = get_clean_metric_df(record, metric_key)
    unit = _trend_unit_for_metric(record, metric_key)
    if df.empty:
        return (
            f"{record.point_clean}: no se identificaron datos válidos para el análisis de {metric_key.lower()}, "
            "por lo que no fue posible emitir diagnóstico automático."
        )

    analysis = _classify_trend_behavior(df["y"])
    sample_count = analysis.get("sample_count", 0)
    start_ts = safe_datetime(df["x"].iloc[0])
    end_ts = safe_datetime(df["x"].iloc[-1])

    base = (
        f"{record.point_clean} — ventana analizada desde {pretty_date(start_ts)} {pretty_time(start_ts)} "
        f"hasta {pretty_date(end_ts)} {pretty_time(end_ts)}, con {sample_count} muestras válidas. "
        f"Valor inicial {format_number(analysis.get('initial_value'), 3)} {unit}, "
        f"valor final {format_number(analysis.get('final_value'), 3)} {unit}, "
        f"variación total {format_number(analysis.get('change_pct'), 2)}%."
    )

    classification = analysis.get("classification")
    if classification == "progressive_increase":
        return (
            f"{base} La tendencia presenta un incremento progresivo del {metric_key.lower()}, "
            "lo cual sugiere posible deterioro del estado mecánico o evolución de una condición incipiente. "
            "Se recomienda seguimiento estrecho y correlación con variables operativas y alarmas."
        )
    if classification == "progressive_decrease":
        return (
            f"{base} La señal muestra una disminución progresiva del {metric_key.lower()}, "
            "compatible con normalización de la condición o reducción de carga/excitación. "
            "Se recomienda verificar si el comportamiento coincide con cambios operativos esperados."
        )
    if classification == "abrupt":
        return (
            f"{base} Se observan variaciones bruscas y dispersión elevada en la señal, "
            "compatibles con condición transitoria, inestabilidad o cambios operativos repentinos. "
            "Se recomienda revisar eventos de proceso, transientes de arranque/parada y consistencia de la instrumentación."
        )
    if classification == "stable":
        return (
            f"{base} El comportamiento es estable y sin desviaciones significativas, "
            "lo que es consistente con una condición normal dentro de la ventana evaluada. "
            "Se recomienda continuar monitoreo rutinario."
        )
    return (
        f"{base} La cantidad de información disponible no es suficiente para clasificar con confianza la tendencia. "
        "Se recomienda ampliar la ventana temporal o validar la calidad de los datos."
    )


def _build_operational_only_narrative(records: List[OperationalRecord]) -> str:
    lines: List[str] = []
    for rec in records:
        df = get_operational_clean_df(rec)
        if df.empty:
            lines.append(
                f"{rec.variable}: no se identificaron datos válidos para emitir diagnóstico automático."
            )
            continue
        analysis = _classify_trend_behavior(df["y"])
        start_ts = safe_datetime(df["x"].iloc[0])
        end_ts = safe_datetime(df["x"].iloc[-1])
        unit = rec.unit or ""
        base = (
            f"{rec.variable} — ventana analizada desde {pretty_date(start_ts)} {pretty_time(start_ts)} "
            f"hasta {pretty_date(end_ts)} {pretty_time(end_ts)}. "
            f"Valor inicial {format_number(analysis.get('initial_value'), 3)} {unit}, "
            f"valor final {format_number(analysis.get('final_value'), 3)} {unit}, "
            f"variación total {format_number(analysis.get('change_pct'), 2)}%."
        )

        classification = analysis.get("classification")
        if classification == "progressive_increase":
            lines.append(f"{base} Tendencia operativa con incremento progresivo sostenido.")
        elif classification == "progressive_decrease":
            lines.append(f"{base} Tendencia operativa con descenso progresivo sostenido.")
        elif classification == "abrupt":
            lines.append(f"{base} Tendencia operativa con variaciones bruscas o comportamiento transitorio.")
        elif classification == "stable":
            lines.append(f"{base} Tendencia operativa estable durante la ventana evaluada.")
        else:
            lines.append(f"{base} Información insuficiente para clasificar la tendencia.")
    return "\n\n".join(lines)


def build_trend_report_narrative(
    records: List[TrendRecord],
    metric_key: str,
    operational_records: Optional[List[OperationalRecord]] = None,
    operational_only_mode: bool = False,
) -> str:
    operational_records = operational_records or []

    if operational_only_mode and operational_records:
        return _build_operational_only_narrative(operational_records)

    trend_lines = [_build_single_trend_narrative(rec, metric_key) for rec in records]
    if operational_records:
        op_summary = _build_operational_only_narrative(operational_records)
        trend_lines.append(
            "Correlación operativa disponible:\n\n"
            f"{op_summary}"
        )
    
context = st.session_state.get("asset_context", {})
ctx_text = f"\n\nContexto de máquina: {context.get('type','')} - {context.get('description','')}"




# session
if "trend_signals" not in st.session_state:
    st.session_state["trend_signals"] = {}
if "operational_signals" not in st.session_state:
    st.session_state["operational_signals"] = {}
if "wm_tr_operational_signal_ids" not in st.session_state:
    st.session_state.wm_tr_operational_signal_ids = []
if "wm_tr_operational_temp_unit" not in st.session_state:
    st.session_state.wm_tr_operational_temp_unit = "°F"
if "wm_tr_primary_signal_id" not in st.session_state:
    st.session_state.wm_tr_primary_signal_id = None
if "wm_tr_extra_signal_ids" not in st.session_state:
    st.session_state.wm_tr_extra_signal_ids = []
if "wm_tr_display_mode" not in st.session_state:
    st.session_state.wm_tr_display_mode = "Combined"
if "wm_tr_export_store" not in st.session_state:
    st.session_state.wm_tr_export_store = {}
if "report_items" not in st.session_state:
    st.session_state.report_items = []

if "wm_tr_asset_type" not in st.session_state:
    st.session_state.wm_tr_asset_type = ""
if "wm_tr_machine_configuration" not in st.session_state:
    st.session_state.wm_tr_machine_configuration = ""
if "wm_tr_primary_equipment" not in st.session_state:
    st.session_state.wm_tr_primary_equipment = ""
if "wm_tr_secondary_equipment" not in st.session_state:
    st.session_state.wm_tr_secondary_equipment = ""
if "wm_tr_machine_description" not in st.session_state:
    st.session_state.wm_tr_machine_description = ""

for key in [
    "wm_tr_cursor_a_initial", "wm_tr_cursor_a_current",
    "wm_tr_cursor_b_initial", "wm_tr_cursor_b_current",
    "wm_tr_x_manual_start", "wm_tr_x_manual_end",
]:
    if key not in st.session_state:
        st.session_state[key] = ""


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

    st.markdown("### Operational Data CSV")
    st.session_state.wm_tr_operational_temp_unit = st.selectbox(
        "Temperature unit in operational file",
        options=["°F", "°C"],
        index=0 if st.session_state.wm_tr_operational_temp_unit == "°F" else 1,
    )
    operational_uploaded_files = st.file_uploader(
        "Upload one or more operational CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="wm_operational_uploader",
    )
    if operational_uploaded_files:
        parsed_operational_records = load_operational_records_from_uploader(
            operational_uploaded_files,
            temperature_unit=st.session_state.wm_tr_operational_temp_unit,
        )
        operational_store = {rec.op_id: rec for rec in parsed_operational_records}
        st.session_state["operational_signals"] = operational_store

    st.markdown("### Machine Diagnostic Context")
    asset_type_options = [
        "",
        "Turbogenerador",
        "Turbina de gas",
        "Generador eléctrico",
        "Motor eléctrico",
        "Bomba",
        "Compresor",
        "Ventilador",
        "Gearbox",
        "Otro",
    ]
    st.session_state.wm_tr_asset_type = st.selectbox(
        "Asset type *",
        options=asset_type_options,
        index=asset_type_options.index(st.session_state.wm_tr_asset_type) if st.session_state.wm_tr_asset_type in asset_type_options else 0,
        key="wm_tr_asset_type_select",
    )

    config_options = ["", "Simple", "Compuesta / tren de máquinas"]
    st.session_state.wm_tr_machine_configuration = st.selectbox(
        "Machine configuration *",
        options=config_options,
        index=config_options.index(st.session_state.wm_tr_machine_configuration) if st.session_state.wm_tr_machine_configuration in config_options else 0,
        key="wm_tr_machine_configuration_select",
    )

    if st.session_state.wm_tr_machine_configuration == "Compuesta / tren de máquinas":
        st.session_state.wm_tr_primary_equipment = st.text_input(
            "Primary equipment *",
            value=st.session_state.wm_tr_primary_equipment,
            placeholder="Ejemplo: Turbina LM6000",
            key="wm_tr_primary_equipment_input",
        )
        st.session_state.wm_tr_secondary_equipment = st.text_input(
            "Secondary equipment *",
            value=st.session_state.wm_tr_secondary_equipment,
            placeholder="Ejemplo: Generador Brush",
            key="wm_tr_secondary_equipment_input",
        )
    else:
        st.session_state.wm_tr_primary_equipment = ""
        st.session_state.wm_tr_secondary_equipment = ""

    st.session_state.wm_tr_machine_description = st.text_area(
        "Machine technical description *",
        value=st.session_state.wm_tr_machine_description,
        height=120,
        placeholder="Ejemplo: Turbina LM6000 acoplada a generador Brush. No corresponde a sistema hidráulico.",
        key="wm_tr_machine_description_input",
    )

records_all: List[TrendRecord] = list(st.session_state.get("trend_signals", {}).values())
records_all = sorted(records_all, key=lambda r: (r.machine, r.point_clean, r.file_name))

operational_records_all: List[OperationalRecord] = list(st.session_state.get("operational_signals", {}).values())
operational_records_all = sorted(operational_records_all, key=lambda r: (r.machine, r.variable, r.file_name))


if not records_all and not operational_records_all:
    st.warning("Cargue al menos un CSV de tendencia o un CSV de data operativa en este módulo.")
    st.stop()

trend_context_errors: List[str] = []
if not st.session_state.wm_tr_asset_type:
    trend_context_errors.append("Asset type is required in Trends.")
if not st.session_state.wm_tr_machine_configuration:
    trend_context_errors.append("Machine configuration is required in Trends.")
if st.session_state.wm_tr_machine_configuration == "Compuesta / tren de máquinas":
    if not str(st.session_state.wm_tr_primary_equipment).strip():
        trend_context_errors.append("Primary equipment is required for composite machine trains.")
    if not str(st.session_state.wm_tr_secondary_equipment).strip():
        trend_context_errors.append("Secondary equipment is required for composite machine trains.")
if not str(st.session_state.wm_tr_machine_description).strip():
    trend_context_errors.append("Machine technical description is required in Trends.")

st.session_state["asset_context"] = {
    "type": st.session_state.wm_tr_asset_type,
    "description": st.session_state.wm_tr_machine_description.strip(),
    "asset_type": st.session_state.wm_tr_asset_type,
    "machine_configuration": st.session_state.wm_tr_machine_configuration,
    "primary_equipment": st.session_state.wm_tr_primary_equipment,
    "secondary_equipment": st.session_state.wm_tr_secondary_equipment,
    "machine_description": st.session_state.wm_tr_machine_description.strip(),
}

if trend_context_errors:
    for msg in trend_context_errors:
        st.warning(msg)

st.session_state["asset_context"] = {
    "type": st.session_state.wm_tr_asset_type,
    "description": st.session_state.wm_tr_machine_description.strip(),
    "asset_type": st.session_state.wm_tr_asset_type,
    "machine_configuration": st.session_state.wm_tr_machine_configuration,
    "primary_equipment": st.session_state.wm_tr_primary_equipment,
    "secondary_equipment": st.session_state.wm_tr_secondary_equipment,
    "machine_description": st.session_state.wm_tr_machine_description.strip(),
}


def push_linked_bode_context(records: List[TrendRecord], metric_key: str) -> None:
    if not records:
        return

    first = records[0]

    cursor_a_label = (
        st.session_state.get("wm_tr_cursor_a_current")
        or st.session_state.get("wm_tr_cursor_a_initial")
        or None
    )
    cursor_b_label = (
        st.session_state.get("wm_tr_cursor_b_current")
        or st.session_state.get("wm_tr_cursor_b_initial")
        or None
    )

    st.session_state["linked_bode_context"] = {
        "machine": first.machine,
        "point": first.point_clean,
        "variable": metric_key,
        "source_module": "04_Trends",
        "trend_cursor_a_label": cursor_a_label,
        "trend_cursor_b_label": cursor_b_label,
    }


def queue_trend_to_report(
    records: List[TrendRecord],
    fig: go.Figure,
    panel_title: str,
    metric_key: str,
    operational_records: Optional[List[OperationalRecord]] = None,
    operational_only_mode: bool = False,
) -> Tuple[bool, Optional[str]]:
    operational_records = operational_records or []

    trend_context_errors: List[str] = []
    if not st.session_state.get("wm_tr_asset_type"):
        trend_context_errors.append("Asset type is required in Trends before sending to report.")
    if not st.session_state.get("wm_tr_machine_configuration"):
        trend_context_errors.append("Machine configuration is required in Trends before sending to report.")
    if st.session_state.get("wm_tr_machine_configuration") == "Compuesta / tren de máquinas":
        if not str(st.session_state.get("wm_tr_primary_equipment", "")).strip():
            trend_context_errors.append("Primary equipment is required for composite machine trains before sending to report.")
        if not str(st.session_state.get("wm_tr_secondary_equipment", "")).strip():
            trend_context_errors.append("Secondary equipment is required for composite machine trains before sending to report.")
    if not str(st.session_state.get("wm_tr_machine_description", "")).strip():
        trend_context_errors.append("Machine technical description is required in Trends before sending to report.")

    if trend_context_errors:
        return False, " | ".join(trend_context_errors)
    if records:
        first = records[0]
        machine = first.machine
        point = " | ".join([r.point_clean for r in records[:2]])
        if len(records) > 2:
            point += f" +{len(records)-2}"
        signal_id = "|".join([r.trend_id for r in records])
        timestamp = str(records[0].timestamp_max or "")
        variable = f"Trend | {metric_key}"
    elif operational_records:
        first_op = operational_records[0]
        machine = first_op.machine
        point = " | ".join([r.variable for r in operational_records[:2]])
        if len(operational_records) > 2:
            point += f" +{len(operational_records)-2}"
        signal_id = "|".join([r.op_id for r in operational_records])
        timestamp = str(first_op.timestamp_max or "")
        variable = "Operational Data" if operational_only_mode else f"Trend + Operational | {metric_key}"
    else:
        return False, "No valid signals to send."

    narrative = build_trend_report_narrative_core(
        records=records,
        metric_key=metric_key,
        operational_records=operational_records,
        operational_only_mode=operational_only_mode,
        asset_context=st.session_state.get("asset_context", {}) or {},
    )

    image_bytes, image_error = build_export_png_bytes(fig=fig)

    item_payload = {
        "id": make_export_state_key(
            [
                "report-trend",
                metric_key,
                panel_title,
                machine,
                point,
                len(st.session_state.report_items),
            ]
        ),
        "type": "trends",
        "title": panel_title,
        "notes": narrative,
        "signal_id": signal_id,
        "figure": None,
        "image_bytes": image_bytes,
        "image_error": image_error,
        "source_module": "04_Trends",
        "report_payload_version": "v2",
        "machine": machine,
        "point": point,
        "variable": variable,
        "timestamp": timestamp,
    }
    st.session_state.report_items.append(item_payload)
    st.session_state["wm_tr_last_report_debug"] = {
        "notes_len": len(str(narrative or "")),
        "report_items_count": len(st.session_state.report_items),
        "last_title": panel_title,
        "has_image": image_bytes is not None,
    }
    return True, image_error


with st.sidebar:
    st.markdown("### Signal Selection")

    signal_name_map = {r.display_name: r.trend_id for r in records_all}
    signal_names = list(signal_name_map.keys())

    if records_all:
        if st.session_state.wm_tr_primary_signal_id not in [r.trend_id for r in records_all]:
            st.session_state.wm_tr_primary_signal_id = records_all[0].trend_id

        current_primary_name = next(
            (r.display_name for r in records_all if r.trend_id == st.session_state.wm_tr_primary_signal_id),
            signal_names[0],
        )

        selected_primary_name = st.selectbox(
            "Primary vibration signal",
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
            "Additional vibration signals",
            options=extra_options,
            default=default_extra_names,
        )
        st.session_state.wm_tr_extra_signal_ids = [signal_name_map[name] for name in selected_extra_names]
    else:
        st.info("No vibration trend CSV loaded.")
        st.session_state.wm_tr_primary_signal_id = None
        st.session_state.wm_tr_extra_signal_ids = []

    st.markdown("### Operational Selection")
    operational_name_map = {r.display_name: r.op_id for r in operational_records_all}
    operational_names = list(operational_name_map.keys())
    default_operational_names = [
        r.display_name
        for r in operational_records_all
        if r.op_id in st.session_state.wm_tr_operational_signal_ids and r.display_name in operational_names
    ]
    selected_operational_names = st.multiselect(
        "Operational signals (MW / Temp)",
        options=operational_names,
        default=default_operational_names,
    )
    st.session_state.wm_tr_operational_signal_ids = [operational_name_map[name] for name in selected_operational_names]

    st.markdown("### Display")
    display_options = ["Combined", "Independent", "Mixed"]
    if st.session_state.wm_tr_display_mode not in display_options:
        st.session_state.wm_tr_display_mode = "Combined"
    st.session_state.wm_tr_display_mode = st.selectbox(
        "Display mode",
        options=display_options,
        index=display_options.index(st.session_state.wm_tr_display_mode),
    )

    st.markdown("### Trend Processing")
    metric_key = st.selectbox("Metric", options=["Amplitude", "Phase", "Speed"], index=0)
    show_markers = st.checkbox("Show markers", value=False)
    show_anomaly_markers = st.checkbox("Show anomaly markers", value=True)
    fill_area = st.checkbox("Fill area (single trend)", value=True)

    st.markdown("### Axes")
    y_axis_mode = st.selectbox("Primary Y-axis scale", ["Auto", "Manual"], index=0)

    y_axis_manual_min: Optional[float] = None
    y_axis_manual_max: Optional[float] = None
    if y_axis_mode == "Manual":
        c1, c2 = st.columns(2)
        with c1:
            y_axis_manual_min = float(st.number_input("Y min", value=0.0, step=0.1, format="%.3f"))
        with c2:
            y_axis_manual_max = float(st.number_input("Y max", value=5.0, step=0.1, format="%.3f"))

    operational_y_axis_mode = st.selectbox("Operational Y-axis scale", ["Auto", "Manual"], index=0)
    operational_y_manual_min: Optional[float] = None
    operational_y_manual_max: Optional[float] = None
    if operational_y_axis_mode == "Manual":
        c3, c4 = st.columns(2)
        with c3:
            operational_y_manual_min = float(st.number_input("Operational Y min", value=0.0, step=1.0, format="%.3f"))
        with c4:
            operational_y_manual_max = float(st.number_input("Operational Y max", value=60.0, step=1.0, format="%.3f"))

    x_axis_mode = st.selectbox("X-axis scale", ["Auto", "Manual"], index=0)
    show_right_info_box = st.checkbox("Show info box", value=True)
    show_legend = st.checkbox("Show legend", value=True)

    st.markdown("### Alarms")
    warning_enabled = st.checkbox("Enable warning line", value=True)
    warning_value: Optional[float] = None
    if warning_enabled:
        warning_value = float(st.number_input("Warning value", value=3.500, step=0.1, format="%.3f"))

    danger_enabled = st.checkbox("Enable danger line", value=True)
    danger_value: Optional[float] = None
    if danger_enabled:
        danger_value = float(st.number_input("Danger value", value=5.000, step=0.1, format="%.3f"))

selected_ids = [st.session_state.wm_tr_primary_signal_id] + st.session_state.wm_tr_extra_signal_ids
selected_ids = [sid for sid in selected_ids if sid is not None]

selected_records = [r for r in records_all if r.trend_id in selected_ids]
selected_records_sorted: List[TrendRecord] = []
for sid in selected_ids:
    rec = next((r for r in selected_records if r.trend_id == sid), None)
    if rec is not None:
        selected_records_sorted.append(rec)

selected_operational_ids = [sid for sid in st.session_state.wm_tr_operational_signal_ids if sid is not None]
selected_operational_records = [r for r in operational_records_all if r.op_id in selected_operational_ids]
selected_operational_records_sorted: List[OperationalRecord] = []
for sid in selected_operational_ids:
    rec = next((r for r in selected_operational_records if r.op_id == sid), None)
    if rec is not None:
        selected_operational_records_sorted.append(rec)

if st.session_state.wm_tr_display_mode in ["Combined", "Independent"] and not selected_records_sorted and not selected_operational_records_sorted:
    st.warning("No valid signals selected.")
    st.stop()

if st.session_state.wm_tr_display_mode == "Mixed" and (not selected_records_sorted or not selected_operational_records_sorted):
    st.warning("Mixed mode requiere al menos una señal de vibración y una señal operativa.")
    st.stop()

mixed_operational_notice: Optional[str] = None
if st.session_state.wm_tr_display_mode == "Mixed" and len(selected_operational_records_sorted) > 1:
    families = [r.family for r in selected_operational_records_sorted]
    first_family = families[0]
    filtered = [r for r in selected_operational_records_sorted if r.family == first_family]
    if len(filtered) != len(selected_operational_records_sorted):
        mixed_operational_notice = (
            "Mixed mode solo mezcla una familia operativa por eje secundario. "
            f"Se usarán únicamente las señales de tipo '{first_family}'."
        )
    selected_operational_records_sorted = filtered

logo_uri = get_logo_data_uri(LOGO_PATH)

if selected_records_sorted:
    time_options = get_time_options_for_records(selected_records_sorted, metric_key)
else:
    time_options = []

if (not time_options) and selected_operational_records_sorted:
    time_options = get_time_options_for_operational_records(selected_operational_records_sorted)

time_labels = [ts_to_label(ts) for ts in time_options]

if not time_labels:
    st.warning("No hay datos válidos para los cursores en la selección actual.")
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
if x_axis_mode == "Manual":
    x_axis_manual_start = label_to_ts(st.session_state.wm_tr_x_manual_start)
    x_axis_manual_end = label_to_ts(st.session_state.wm_tr_x_manual_end)

cursor_map = {
    "A Initial": label_to_ts(st.session_state.wm_tr_cursor_a_initial),
    "A Current": label_to_ts(st.session_state.wm_tr_cursor_a_current),
    "B Initial": label_to_ts(st.session_state.wm_tr_cursor_b_initial),
    "B Current": label_to_ts(st.session_state.wm_tr_cursor_b_current),
}

if x_axis_mode == "Manual":
    with st.expander("X-Axis Manual Window", expanded=False):
        st.markdown(
            """
            <div class="wm-control-shell">
                <div class="wm-control-title">X-Axis Window</div>
                <div class="wm-control-subtitle">Ajusta el inicio y fin del tiempo con sliders precisos.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            st.select_slider("X Start", options=time_labels, key="wm_tr_x_manual_start")
        with col_x2:
            st.select_slider("X End", options=time_labels, key="wm_tr_x_manual_end")

with st.expander("Cursor Controls", expanded=False):
    st.markdown(
        """
        <div class="wm-control-shell">
            <div class="wm-control-title">Cursor Controls</div>
            <div class="wm-control-subtitle">Referencias temporales A/B para comparar comportamiento entre dos momentos.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.select_slider("A Initial", options=time_labels, key="wm_tr_cursor_a_initial")
        st.select_slider("A Current", options=time_labels, key="wm_tr_cursor_a_current")
    with c2:
        st.select_slider("B Initial", options=time_labels, key="wm_tr_cursor_b_initial")
        st.select_slider("B Current", options=time_labels, key="wm_tr_cursor_b_current")


def render_trend_panel(
    panel_records: List[TrendRecord],
    panel_index: int,
    panel_label: str,
    panel_operational_records: Optional[List[OperationalRecord]] = None,
    mixed_mode: bool = False,
    operational_only_mode: bool = False,
) -> None:
    panel_operational_records = panel_operational_records or []

    fig = build_trend_figure(
        records=panel_records,
        metric_key=metric_key,
        show_markers=show_markers,
        show_anomaly_markers=show_anomaly_markers,
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
        operational_records=panel_operational_records,
        mixed_mode=mixed_mode,
        operational_only_mode=operational_only_mode,
        operational_y_axis_mode=operational_y_axis_mode,
        operational_y_manual_min=operational_y_manual_min,
        operational_y_manual_max=operational_y_manual_max,
    )

    export_state_key = make_export_state_key(
        [
            st.session_state.wm_tr_display_mode,
            panel_label,
            metric_key,
            y_axis_mode, y_axis_manual_min, y_axis_manual_max,
            operational_y_axis_mode, operational_y_manual_min, operational_y_manual_max,
            x_axis_mode, st.session_state.wm_tr_x_manual_start, st.session_state.wm_tr_x_manual_end,
            warning_enabled, warning_value, danger_enabled, danger_value,
            st.session_state.wm_tr_cursor_a_initial, st.session_state.wm_tr_cursor_a_current,
            st.session_state.wm_tr_cursor_b_initial, st.session_state.wm_tr_cursor_b_current,
            show_markers, show_anomaly_markers, fill_area, show_right_info_box, show_legend,
            "|".join([r.trend_id for r in panel_records]),
            "|".join([r.file_name for r in panel_records]),
            "|".join([r.point_clean for r in panel_records]),
            "|".join([r.op_id for r in panel_operational_records]),
            "|".join([r.variable for r in panel_operational_records]),
            mixed_mode, operational_only_mode,
        ]
    )

    if export_state_key not in st.session_state.wm_tr_export_store:
        st.session_state.wm_tr_export_store[export_state_key] = {"png_bytes": None, "error": None}

    st.markdown(f"### {panel_label}")

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False},
        key=f"wm_trends_plot_{export_state_key}",
    )

    st.markdown('<div class="wm-export-actions"></div>', unsafe_allow_html=True)
    left_pad, col_export1, col_export2, col_report, col_bode, right_pad = st.columns([1.6, 1.2, 1.2, 1.2, 1.3, 1.5])

    with col_export1:
        if st.button("Prepare PNG HD", key=f"prepare_png_{export_state_key}", use_container_width=True):
            with st.spinner("Generating HD export..."):
                png_bytes, export_error = build_export_png_bytes(fig=fig)
                st.session_state.wm_tr_export_store[export_state_key]["png_bytes"] = png_bytes
                st.session_state.wm_tr_export_store[export_state_key]["error"] = export_error

    with col_export2:
        png_bytes = st.session_state.wm_tr_export_store[export_state_key]["png_bytes"]
        if png_bytes is not None:
            st.download_button(
                "Download PNG HD",
                data=png_bytes,
                file_name=f"watermelon_trend_{panel_index + 1}_hd.png",
                mime="image/png",
                use_container_width=True,
                key=f"download_png_{export_state_key}",
            )
        else:
            st.button("Download PNG HD", disabled=True, use_container_width=True, key=f"download_disabled_{export_state_key}")

    with col_report:
        if st.button("Enviar a Reporte", key=f"send_report_{export_state_key}", use_container_width=True):
            image_ok, image_error = queue_trend_to_report(
                panel_records,
                fig,
                panel_label,
                metric_key,
                operational_records=panel_operational_records,
                operational_only_mode=operational_only_mode,
            )
            if image_ok:
                st.success("Trend enviado al reporte")
            else:
                st.error(image_error or "No fue posible enviar el trend al reporte.")

    with col_bode:
        bode_disabled = operational_only_mode or len(panel_records) == 0
        if st.button("Open linked Bode", key=f"open_bode_{export_state_key}", use_container_width=True, disabled=bode_disabled):
            push_linked_bode_context(panel_records, metric_key)
            st.switch_page("pages/07_Bode_Plot.py")

    if st.session_state.get("wm_tr_last_report_debug"):
        dbg = st.session_state["wm_tr_last_report_debug"]
        st.caption(
            f"Report debug → notes_len={dbg.get('notes_len')} | report_items={dbg.get('report_items_count')} | "
            f"title={dbg.get('last_title')} | has_image={dbg.get('has_image')}"
        )

    panel_error = st.session_state.wm_tr_export_store[export_state_key]["error"]
    if panel_error:
        st.warning(f"PNG export error: {panel_error}")

    if panel_records:
        anomaly_summary = build_panel_anomaly_summary(panel_records, metric_key)
        st.markdown("#### Detección automática de anomalías")
        a1, a2, a3 = st.columns(3)
        with a1:
            st.metric("Anomalies", str(anomaly_summary.get("total_count", 0)))
        with a2:
            st.metric("Affected signals", str(anomaly_summary.get("affected_records", 0)))
        with a3:
            st.metric("Top severity", anomaly_summary.get("top_severity", "None"))
        st.info(anomaly_summary.get("interpretation", "Sin interpretación disponible."))

        anomaly_narrative = build_anomaly_narrative(panel_records, metric_key)
        st.markdown("**Interpretación técnica de anomalías:**")
        st.write(anomaly_narrative)

    # ------------------------------------------------------------
    # Automatic correlation: primary vibration vs first operational
    # ------------------------------------------------------------
    correlation_enabled = bool(panel_records) and bool(panel_operational_records)
    if correlation_enabled:
        primary_trend = panel_records[0]
        primary_operational = panel_operational_records[0]
        correlation_info = build_trend_operational_correlation(
            trend_record=primary_trend,
            operational_record=primary_operational,
            metric_key=metric_key,
        )

        st.markdown("#### Correlación automática vibración vs variable operativa")

        corr_value = correlation_info.get("corr_value")
        corr_text = format_number(corr_value, 3) if corr_value is not None else "—"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Correlation", corr_text)
        with c2:
            st.metric("Strength", correlation_info.get("strength") or "—")
        with c3:
            st.metric("Direction", correlation_info.get("direction") or "—")
        with c4:
            st.metric("Samples", str(correlation_info.get("sample_count") or 0))

        st.info(correlation_info.get("interpretation") or "Sin interpretación disponible.")

        scatter_fig = build_correlation_scatter_figure(correlation_info)
        st.plotly_chart(
            scatter_fig,
            use_container_width=True,
            config={"displaylogo": False},
            key=f"wm_trends_corr_{export_state_key}",
        )

        lag_info = build_lagged_correlation_analysis(
            trend_record=primary_trend,
            operational_record=primary_operational,
            metric_key=metric_key,
            max_lag_minutes=180,
            step_minutes=10,
        )

        st.markdown("#### Correlación con desfase temporal (lag)")
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            st.metric(
                "Best correlation",
                format_number(lag_info.get("best_corr"), 3),
            )
        with l2:
            best_lag_val = lag_info.get("best_lag_min")
            st.metric(
                "Best lag (min)",
                str(best_lag_val) if best_lag_val is not None else "—",
            )
        with l3:
            st.metric("Lag strength", lag_info.get("strength") or "—")
        with l4:
            st.metric("Lag direction", lag_info.get("direction") or "—")

        st.info(lag_info.get("interpretation") or "Sin interpretación disponible.")

        lag_fig = build_lag_correlation_figure(lag_info)
        st.plotly_chart(
            lag_fig,
            use_container_width=True,
            config={"displaylogo": False},
            key=f"wm_trends_lagcorr_{export_state_key}",
        )


if mixed_operational_notice:
    st.info(mixed_operational_notice)

if st.session_state.wm_tr_display_mode == "Combined":
    if selected_records_sorted:
        combined_label = f"Trend Combined — {selected_records_sorted[0].machine}"
        render_trend_panel(selected_records_sorted, 0, combined_label)
    elif selected_operational_records_sorted:
        combined_label = f"Operational Combined — {selected_operational_records_sorted[0].machine}"
        render_trend_panel([], 0, combined_label, panel_operational_records=selected_operational_records_sorted, operational_only_mode=True)
elif st.session_state.wm_tr_display_mode == "Mixed":
    combined_label = f"Trend + Operational — {selected_records_sorted[0].machine}"
    render_trend_panel(
        selected_records_sorted,
        0,
        combined_label,
        panel_operational_records=selected_operational_records_sorted,
        mixed_mode=True,
    )
else:
    panel_idx = 0
    if selected_records_sorted:
        for idx, rec in enumerate(selected_records_sorted):
            render_trend_panel([rec], panel_idx, f"Trend {idx + 1} — {rec.point_clean}")
            panel_idx += 1
            if idx < len(selected_records_sorted) - 1 or selected_operational_records_sorted:
                st.markdown("---")
    if selected_operational_records_sorted:
        for idx, rec in enumerate(selected_operational_records_sorted):
            render_trend_panel([], panel_idx, f"Operational {idx + 1} — {rec.variable}", panel_operational_records=[rec], operational_only_mode=True)
            panel_idx += 1
            if idx < len(selected_operational_records_sorted) - 1:
                st.markdown("---")
