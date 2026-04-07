from __future__ import annotations

import base64
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.auth import require_login, render_user_menu
from core.waveform_diagnostics import generate_waveform_diagnostic, build_waveform_report_notes

st.set_page_config(page_title="Watermelon System | Waveform", layout="wide")

require_login()
render_user_menu()

# ============================================================
# WATERMELON SYSTEM — TIME WAVEFORM VIEWER
# ============================================================

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

        div[data-testid="stNumberInput"] input {
            font-family: monospace;
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

        .wm-export-note-bottom {
            font-size: 0.90rem;
            color: #64748b;
            text-align: center;
            margin-top: 0.45rem;
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


def safe_slug(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "waveform"


# ------------------------------------------------------------
# Data model
# ------------------------------------------------------------
@dataclass
class SignalRecord:
    signal_id: str
    name: str
    machine: str = "Unknown"
    point: str = "Point 1"
    variable: str = "Waveform"
    amplitude_unit: str = ""
    time_s: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    amplitude: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sample_rate_hz: Optional[float] = None
    rpm: Optional[float] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_key: Optional[str] = None
    source_time_unit: str = "s"

    @property
    def duration_s(self) -> float:
        if self.time_s.size < 2:
            return 0.0
        return float(self.time_s[-1] - self.time_s[0])


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def parse_first_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        try:
            out = float(value)
            return out if math.isfinite(out) else None
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if not match:
        return None
    try:
        out = float(match.group(0))
        return out if math.isfinite(out) else None
    except Exception:
        return None


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, np.ndarray):
        try:
            return value.astype(float, copy=False).ravel()
        except Exception:
            return np.array([], dtype=float)
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").to_numpy(dtype=float)
    if isinstance(value, (list, tuple)):
        try:
            return pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(dtype=float)
        except Exception:
            return np.array([], dtype=float)
    return np.array([], dtype=float)


def infer_sample_rate_from_seconds(time_s: np.ndarray) -> Optional[float]:
    if time_s.size < 2:
        return None
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    mean_dt = float(np.mean(dt))
    if mean_dt <= 0:
        return None
    return 1.0 / mean_dt


def normalize_signal(y: np.ndarray, mode: str) -> np.ndarray:
    if y.size == 0:
        return y
    if mode == "None":
        return y
    if mode == "Remove mean":
        return y - np.mean(y)
    if mode == "Z-score":
        std = np.std(y)
        return (y - np.mean(y)) / std if std > 0 else y - np.mean(y)
    if mode == "Peak normalize":
        peak = np.max(np.abs(y))
        return y / peak if peak > 0 else y
    return y


def rms(y: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(y))))


def crest_factor(y: np.ndarray) -> float:
    y_rms = rms(y)
    if y.size == 0 or y_rms == 0 or np.isnan(y_rms):
        return float("nan")
    return float(np.max(np.abs(y)) / y_rms)


def nearest_index(x: np.ndarray, value: float) -> int:
    if x.size == 0:
        return 0
    return int(np.argmin(np.abs(x - value)))


def format_number(value: Any, digits: int = 4, fallback: str = "—") -> str:
    if value is None:
        return fallback
    try:
        val = float(value)
        if not math.isfinite(val):
            return fallback
        return f"{val:.{digits}f}"
    except Exception:
        return fallback


def find_meta(metadata: Dict[str, Any], candidates: List[str]) -> Any:
    lower_map = {str(k).lower(): v for k, v in metadata.items()}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for cand in candidates:
        c = cand.lower()
        for k, v in lower_map.items():
            if c in k:
                return v
    return None


def infer_amplitude_unit(metadata: Dict[str, Any]) -> str:
    y_axis_unit = find_meta(
        metadata,
        ["Y-Axis Unit", "Y Axis Unit", "y_axis_unit", "YAxisUnit", "Vertical Unit", "vertical_unit"],
    )
    if y_axis_unit:
        return str(y_axis_unit)

    generic = find_meta(metadata, ["Unit", "unit", "units"])
    if generic:
        return str(generic)

    variable = str(find_meta(metadata, ["Variable", "variable"]) or "").lower()
    if "mil" in variable:
        return "mil"
    if "um" in variable or "µm" in variable:
        return "µm"
    if "mm/s" in variable:
        return "mm/s"
    if "ips" in variable:
        return "ips"
    if "g" in variable and "pk" not in variable:
        return "g"
    return ""


def convert_input_time_to_seconds(raw_time: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    if raw_time.size == 0:
        return raw_time, "s"

    raw = raw_time.astype(float, copy=False)
    raw = raw - raw[0]
    duration = float(raw[-1] - raw[0]) if raw.size > 1 else 0.0

    if mode == "Seconds":
        return raw, "s"
    if mode == "Milliseconds":
        return raw / 1000.0, "ms"
    if duration > 5.0:
        return raw / 1000.0, "ms"
    return raw, "s"


# ------------------------------------------------------------
# Harmonic filter
# ------------------------------------------------------------
def harmonic_frequency_hz(rpm: Optional[float], multiple: int) -> Optional[float]:
    if rpm is None or rpm <= 0:
        return None
    return (float(rpm) / 60.0) * float(multiple)


def build_harmonic_component(
    time_s: np.ndarray,
    y: np.ndarray,
    freq_hz: Optional[float],
) -> np.ndarray:
    if freq_hz is None or freq_hz <= 0 or time_s.size < 3 or y.size < 3:
        return y.copy()

    omega = 2.0 * np.pi * freq_hz
    mean_y = float(np.mean(y))
    yc = y - mean_y

    c = np.cos(omega * time_s)
    s = np.sin(omega * time_s)

    design = np.column_stack([c, s])
    coeffs, *_ = np.linalg.lstsq(design, yc, rcond=None)
    a, b = coeffs

    filtered = mean_y + a * c + b * s
    return filtered.astype(float, copy=False)


def apply_waveform_view_mode(
    time_s: np.ndarray,
    y: np.ndarray,
    rpm: Optional[float],
    mode: str,
) -> Tuple[np.ndarray, str]:
    if mode == "Raw":
        return y, "Raw"

    if mode == "1X filtered":
        f1 = harmonic_frequency_hz(rpm, 1)
        return build_harmonic_component(time_s, y, f1), "1X"

    if mode == "2X filtered":
        f2 = harmonic_frequency_hz(rpm, 2)
        return build_harmonic_component(time_s, y, f2), "2X"

    return y, "Raw"


# ------------------------------------------------------------
# Cycle markers from synchronized start t=0
# ------------------------------------------------------------
def build_cycle_start_markers_from_sync(
    time_s: np.ndarray,
    y: np.ndarray,
    rpm: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if rpm is None or rpm <= 0 or time_s.size < 2 or y.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    period = 60.0 / float(rpm)
    if period <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    t0 = float(time_s[0])
    t_end = float(time_s[-1])
    duration = t_end - t0
    if duration <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    n_cycles = int(np.floor(duration / period)) + 1
    cycle_times = t0 + np.arange(n_cycles, dtype=float) * period
    cycle_times = cycle_times[(cycle_times >= t0) & (cycle_times <= t_end)]

    if cycle_times.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    idx = np.searchsorted(time_s, cycle_times, side="left")
    idx = np.clip(idx, 0, len(time_s) - 1)

    for i in range(len(idx)):
        j = idx[i]
        if j > 0:
            if abs(time_s[j] - cycle_times[i]) >= abs(time_s[j - 1] - cycle_times[i]):
                idx[i] = j - 1

    idx = np.unique(idx)
    return time_s[idx], y[idx]


# ------------------------------------------------------------
# Load signals
# ------------------------------------------------------------
def signal_object_to_record(
    signal_obj: Any,
    source_key: str,
    display_name: str,
    input_time_mode: str,
) -> Optional[SignalRecord]:
    if not hasattr(signal_obj, "time") or not hasattr(signal_obj, "x"):
        return None

    raw_time = to_numpy(getattr(signal_obj, "time", None))
    amplitude = to_numpy(getattr(signal_obj, "x", None))

    if raw_time.size == 0 or amplitude.size == 0:
        return None

    n = min(raw_time.size, amplitude.size)
    raw_time = raw_time[:n]
    amplitude = amplitude[:n]

    if n < 2:
        return None

    finite_mask = np.isfinite(raw_time) & np.isfinite(amplitude)
    raw_time = raw_time[finite_mask]
    amplitude = amplitude[finite_mask]

    if raw_time.size < 2 or amplitude.size < 2:
        return None

    time_s, detected_unit = convert_input_time_to_seconds(raw_time, input_time_mode)

    sort_idx = np.argsort(time_s)
    time_s = time_s[sort_idx]
    amplitude = amplitude[sort_idx]

    unique_mask = np.ones_like(time_s, dtype=bool)
    if time_s.size > 1:
        unique_mask[1:] = np.diff(time_s) > 0

    time_s = time_s[unique_mask]
    amplitude = amplitude[unique_mask]

    if time_s.size < 2 or amplitude.size < 2:
        return None

    sample_rate_hz = infer_sample_rate_from_seconds(time_s)

    metadata = getattr(signal_obj, "metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}

    machine = str(find_meta(metadata, ["Machine", "machine"]) or "Unknown")
    point = str(find_meta(metadata, ["Point", "point", "Point Name", "point name", "channel"]) or "Point 1")
    variable = str(find_meta(metadata, ["Variable", "variable"]) or "Waveform")
    timestamp = find_meta(metadata, ["Timestamp", "timestamp"])
    rpm = parse_first_float(find_meta(metadata, ["RPM", "rpm", "Sample Speed", "sample speed"]))
    amplitude_unit = infer_amplitude_unit(metadata)

    file_name = getattr(signal_obj, "file_name", display_name)
    signal_name = str(file_name or display_name)

    return SignalRecord(
        signal_id=source_key,
        name=signal_name,
        machine=machine,
        point=point,
        variable=variable,
        amplitude_unit=amplitude_unit,
        time_s=time_s,
        amplitude=amplitude,
        sample_rate_hz=sample_rate_hz,
        rpm=rpm,
        timestamp=str(timestamp) if timestamp is not None else None,
        metadata=metadata,
        source_key=source_key,
        source_time_unit=detected_unit,
    )


def load_signals_from_session(input_time_mode: str) -> List[SignalRecord]:
    records: List[SignalRecord] = []
    signals_dict = st.session_state.get("signals", {})

    if isinstance(signals_dict, dict):
        for key, value in signals_dict.items():
            rec = signal_object_to_record(
                value,
                source_key=f"signals.{key}",
                display_name=str(key),
                input_time_mode=input_time_mode,
            )
            if rec is not None:
                records.append(rec)

    return records


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------
def build_measurement_summary(record: SignalRecord) -> Dict[str, Any]:
    x = record.time_s
    y = record.amplitude
    if x.size == 0 or y.size == 0:
        return {
            "RMS": np.nan,
            "Peak-Peak": np.nan,
            "Crest Factor": np.nan,
        }

    return {
        "RMS": rms(y),
        "Peak-Peak": float(np.max(y) - np.min(y)),
        "Crest Factor": crest_factor(y),
    }


def cursor_measurements(record: SignalRecord, cursor_a: float, cursor_b: float) -> Dict[str, Any]:
    x = record.time_s
    y = record.amplitude
    ia = nearest_index(x, cursor_a)
    ib = nearest_index(x, cursor_b)
    xa = float(x[ia])
    xb = float(x[ib])
    ya = float(y[ia])
    yb = float(y[ib])
    return {
        "Cursor A t": xa,
        "Cursor A y": ya,
        "Cursor B t": xb,
        "Cursor B y": yb,
    }


def _nice_tick_step_ms(x_max_ms: float) -> float:
    if x_max_ms <= 20:
        return 5.0
    if x_max_ms <= 60:
        return 10.0
    if x_max_ms <= 120:
        return 20.0
    if x_max_ms <= 300:
        return 50.0
    if x_max_ms <= 600:
        return 100.0
    return 200.0


def _draw_top_strip(
    fig: go.Figure,
    record: SignalRecord,
    amp_pp_text: str,
    logo_uri: Optional[str],
    waveform_mode_label: str,
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
                sizex=0.046,
                sizey=0.072,
                xanchor="left",
                yanchor="top",
                layer="above",
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
        text=f"<b>{record.machine}</b>",
        showarrow=False,
        font=dict(size=12.8, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.205,
        y=y_text,
        xanchor="left",
        yanchor="middle",
        text=record.point,
        showarrow=False,
        font=dict(size=12.1, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.355,
        y=y_text,
        xanchor="left",
        yanchor="middle",
        text=f"{record.variable} | {waveform_mode_label}",
        showarrow=False,
        font=dict(size=12.0, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.640,
        y=y_text,
        xanchor="left",
        yanchor="middle",
        text=f"Wf Amp: <b>{amp_pp_text}</b>",
        showarrow=False,
        font=dict(size=12.3, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.790,
        y=y_text,
        xanchor="left",
        yanchor="middle",
        text=f"{format_number(record.rpm, 0)} rpm" if record.rpm is not None else "rpm —",
        showarrow=False,
        font=dict(size=12.1, color="#111827"),
        align="left",
    )

    if record.timestamp:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.986,
            y=y_text,
            xanchor="right",
            yanchor="middle",
            text=record.timestamp,
            showarrow=False,
            font=dict(size=11.9, color="#111827"),
            align="right",
        )


def _draw_right_info_box(
    fig: go.Figure,
    rows: List[Tuple[str, str]],
) -> None:
    panel_x0 = 0.836
    panel_x1 = 0.975
    panel_y1 = 0.915
    header_h = 0.034
    row_h = 0.080
    panel_h = header_h + len(rows) * row_h + 0.016
    panel_y0 = panel_y1 - panel_h

    fig.add_shape(
        type="path",
        xref="paper",
        yref="paper",
        path=rounded_rect_path(panel_x0, panel_y0, panel_x1, panel_y1, 0.012),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(255,255,255,0.86)",
        layer="above",
    )

    fig.add_shape(
        type="path",
        xref="paper",
        yref="paper",
        path=rounded_rect_path(panel_x0, panel_y1 - header_h, panel_x1, panel_y1, 0.012),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="#93c5fd",
        layer="above",
    )

    current_top = panel_y1 - header_h - 0.010

    for title, value in rows:
        title_y = current_top - 0.006
        value_y = current_top - 0.042

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=panel_x0 + 0.034,
            y=title_y,
            xanchor="left",
            yanchor="top",
            text=f"<b>{title}</b>",
            showarrow=False,
            font=dict(size=11.2, color="#111827"),
            align="left",
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=panel_x0 + 0.034,
            y=value_y,
            xanchor="left",
            yanchor="top",
            text=value,
            showarrow=False,
            font=dict(size=11.0, color="#111827"),
            align="left",
        )

        current_top -= row_h


def build_waveform_figure(
    record: SignalRecord,
    cursor_a_s: float,
    cursor_b_s: float,
    x_axis_unit: str,
    show_cursor_b: bool,
    show_right_info_box: bool,
    y_scale_mode: str,
    y_limit_abs: Optional[float],
    logo_uri: Optional[str],
    waveform_mode_label: str,
    show_cycle_start_markers: bool,
) -> go.Figure:
    fig = go.Figure()

    x_scale = 1000.0 if x_axis_unit == "ms" else 1.0
    x_title = "Time (ms)" if x_axis_unit == "ms" else "Time (s)"
    amp_unit = record.amplitude_unit or ""
    y_title = f"Amplitude ({amp_unit})" if amp_unit else "Amplitude"

    x = record.time_s * x_scale
    y = record.amplitude

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if x.size < 2 or y.size < 2:
        fig.update_layout(
            height=620,
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#f3f4f6",
            margin=dict(l=46, r=18, t=76, b=40),
            xaxis_title=x_title,
            yaxis_title=y_title,
        )
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No valid waveform samples available",
            showarrow=False,
            font=dict(size=18, color="#6b7280"),
        )
        return fig

    summary = build_measurement_summary(record)
    cursor = cursor_measurements(record, cursor_a_s, cursor_b_s)

    x_min = 0.0
    x_max = float(np.max(x))
    y_data_min = float(np.min(y))
    y_data_max = float(np.max(y))

    hover_template = (
        ("X: %{x:.2f} ms<br>" if x_axis_unit == "ms" else "X: %{x:.2f} s<br>")
        + (f"Y: " + "%{y:.2f} " + amp_unit if amp_unit else "Y: %{y:.2f}")
        + "<extra></extra>"
    )

    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="lines",
            line=dict(width=1.8, color="#5b9cf0"),
            hovertemplate=hover_template,
            showlegend=False,
            connectgaps=False,
            name="waveform_main",
        )
    )

    if show_cycle_start_markers:
        marker_t_s, marker_y = build_cycle_start_markers_from_sync(
            record.time_s,
            record.amplitude,
            record.rpm,
        )
        marker_x = marker_t_s * x_scale

        if marker_x.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=marker_x,
                    y=marker_y,
                    mode="markers",
                    marker=dict(symbol="circle", size=7.0, color="#2f80ed"),
                    hovertemplate=hover_template,
                    showlegend=False,
                    name="cycle_start_markers",
                )
            )

    step = _nice_tick_step_ms(x_max if x_axis_unit == "ms" else x_max * 1000.0)
    if x_axis_unit != "ms":
        step = step / 1000.0

    tickvals = list(np.arange(0.0, x_max + step * 0.5, step))
    for gx in tickvals:
        fig.add_vline(
            x=gx,
            line_width=1,
            line_color="rgba(148, 163, 184, 0.18)",
            layer="below",
        )

    cursor_a = cursor_a_s * x_scale
    cursor_b = cursor_b_s * x_scale

    fig.add_vline(
        x=cursor_a,
        line_width=1.15,
        line_dash="dash",
        line_color="#374151",
        annotation_text="A",
        annotation_position="top left",
        annotation_font_color="#111827",
        annotation_font_size=11,
    )

    if show_cursor_b:
        fig.add_vline(
            x=cursor_b,
            line_width=1.15,
            line_dash="dash",
            line_color="#6b7280",
            annotation_text="B",
            annotation_position="top right",
            annotation_font_color="#111827",
            annotation_font_size=11,
        )

    amp_pp_text = f"{format_number(summary.get('Peak-Peak'), 3)} {amp_unit} p-p".strip()

    _draw_top_strip(
        fig=fig,
        record=record,
        amp_pp_text=amp_pp_text,
        logo_uri=logo_uri,
        waveform_mode_label=waveform_mode_label,
    )

    if show_right_info_box:
        rows = [
            (
                "Waveform Overall (rms)",
                f"{format_number(summary.get('RMS'), 3)} {amp_unit}".strip(),
            ),
            ("Crest Factor", format_number(summary.get("Crest Factor"), 2)),
            (
                "Cursor A (Basic)",
                f"{format_number(cursor.get('Cursor A y'), 3)} {amp_unit}".strip(),
            ),
        ]

        if show_cursor_b:
            rows.append(
                (
                    "Cursor B (Basic)",
                    f"{format_number(cursor.get('Cursor B y'), 3)} {amp_unit}".strip(),
                )
            )

        _draw_right_info_box(fig, rows)

    fig.update_layout(
        height=640,
        margin=dict(l=46, r=18, t=84, b=40),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f3f4f6",
        font=dict(color="#111827"),
        xaxis=dict(
            title=x_title,
            range=[x_min, x_max],
            tickvals=tickvals,
            tickformat=".0f" if x_axis_unit == "ms" else ".2f",
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="#9ca3af",
            mirror=False,
            ticks="outside",
            tickcolor="#6b7280",
            ticklen=4,
            showspikes=True,
            spikecolor="#6b7280",
            spikesnap="cursor",
            spikemode="across",
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.18)",
            zeroline=False,
            showline=True,
            linecolor="#9ca3af",
            mirror=False,
            ticks="outside",
            tickcolor="#6b7280",
            ticklen=4,
        ),
        hovermode="closest",
    )

    if y_scale_mode == "Manual" and y_limit_abs is not None and y_limit_abs > 0:
        fig.update_yaxes(range=[-float(y_limit_abs), float(y_limit_abs)])
    else:
        pad = 0.12 * max(1e-12, (y_data_max - y_data_min))
        fig.update_yaxes(range=[y_data_min - pad, y_data_max + pad])

    return fig


def _build_export_safe_figure(fig: go.Figure) -> go.Figure:
    export_fig = go.Figure()

    for trace in fig.data:
        trace_name = getattr(trace, "name", "") or ""

        if isinstance(trace, go.Scattergl):
            trace_json = trace.to_plotly_json()
            export_fig.add_trace(
                go.Scatter(
                    x=np.array(trace_json.get("x")) if trace_json.get("x") is not None else None,
                    y=np.array(trace_json.get("y")) if trace_json.get("y") is not None else None,
                    mode=trace_json.get("mode"),
                    line=trace_json.get("line"),
                    marker=trace_json.get("marker"),
                    hovertemplate=trace_json.get("hovertemplate"),
                    showlegend=trace_json.get("showlegend"),
                    connectgaps=trace_json.get("connectgaps", False),
                    name=trace_name,
                )
            )
        elif isinstance(trace, go.Scatter):
            export_fig.add_trace(trace)
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


def build_export_png_bytes(
    fig: go.Figure,
) -> Tuple[Optional[bytes], Optional[str]]:
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
if "wm_cursor_a" not in st.session_state:
    st.session_state.wm_cursor_a = 0.0
if "wm_cursor_b" not in st.session_state:
    st.session_state.wm_cursor_b = 0.01
if "wm_selected_signal_ids" not in st.session_state:
    st.session_state.wm_selected_signal_ids = []
if "wm_export_store" not in st.session_state:
    st.session_state.wm_export_store = {}
if "report_items" not in st.session_state:
    st.session_state.report_items = []


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### Input Interpretation")

    input_time_mode = st.selectbox(
        "Incoming time vector",
        ["Auto", "Milliseconds", "Seconds"],
        index=0,
    )


# ------------------------------------------------------------
# Load signals
# ------------------------------------------------------------
records_all = load_signals_from_session(input_time_mode=input_time_mode)

if not records_all:
    st.warning("No se pudieron cargar señales válidas desde `st.session_state['signals']`.")
    st.stop()


def queue_waveform_to_report(record: SignalRecord, fig: go.Figure, panel_title: str, text_diag: Dict[str, str], image_bytes: Optional[bytes] = None) -> None:
    st.session_state.report_items.append(
        {
            "id": make_export_state_key(
                [
                    "report-waveform",
                    record.signal_id,
                    record.timestamp,
                    panel_title,
                    len(st.session_state.report_items),
                ]
            ),
            "type": "waveform",
            "title": panel_title,
            "notes": build_waveform_report_notes(text_diag),
            "signal_id": record.signal_id,
            "figure": go.Figure(fig),
            "image_bytes": image_bytes,
            "machine": record.machine,
            "point": record.point,
            "variable": record.variable,
            "timestamp": record.timestamp,
        }
    )


def render_waveform_panel(
    primary: SignalRecord,
    panel_index: int,
    *,
    t_min: float,
    t_max: float,
    cursor_a: float,
    cursor_b: float,
    x_axis_unit: str,
    show_cursor_b: bool,
    show_right_info_box: bool,
    y_scale_mode: str,
    y_limit_abs: Optional[float],
    logo_uri: Optional[str],
    waveform_view_mode: str,
    normalization_mode: str,
    show_cycle_start_markers: bool,
) -> None:
    mask = (primary.time_s >= t_min) & (primary.time_s <= t_max)
    if not np.any(mask):
        st.warning(f"La ventana seleccionada no contiene datos para {primary.name}.")
        return

    base_y = normalize_signal(primary.amplitude, normalization_mode)[mask]
    base_t = primary.time_s[mask]

    local_view_mode = waveform_view_mode
    if local_view_mode != "Raw" and (primary.rpm is None or primary.rpm <= 0):
        local_view_mode = "Raw"

    processed_y, waveform_mode_label = apply_waveform_view_mode(
        time_s=base_t,
        y=base_y,
        rpm=primary.rpm,
        mode=local_view_mode,
    )

    prepared = SignalRecord(
        signal_id=primary.signal_id,
        name=primary.name,
        machine=primary.machine,
        point=primary.point,
        variable=primary.variable,
        amplitude_unit=primary.amplitude_unit,
        time_s=base_t,
        amplitude=processed_y,
        sample_rate_hz=primary.sample_rate_hz,
        rpm=primary.rpm,
        timestamp=primary.timestamp,
        metadata=primary.metadata,
        source_key=primary.source_key,
        source_time_unit=primary.source_time_unit,
    )

    fig = build_waveform_figure(
        record=prepared,
        cursor_a_s=cursor_a,
        cursor_b_s=cursor_b,
        x_axis_unit=x_axis_unit,
        show_cursor_b=show_cursor_b,
        show_right_info_box=show_right_info_box,
        y_scale_mode=y_scale_mode,
        y_limit_abs=y_limit_abs,
        logo_uri=logo_uri,
        waveform_mode_label=waveform_mode_label,
        show_cycle_start_markers=show_cycle_start_markers,
    )

    export_state_key = make_export_state_key(
        [
            prepared.signal_id,
            prepared.name,
            prepared.machine,
            prepared.point,
            prepared.variable,
            prepared.timestamp,
            panel_index,
            t_min,
            t_max,
            cursor_a,
            cursor_b,
            x_axis_unit,
            normalization_mode,
            local_view_mode,
            y_scale_mode,
            y_limit_abs,
            show_cycle_start_markers,
            show_cursor_b,
            show_right_info_box,
            float(np.nanmax(prepared.amplitude)) if prepared.amplitude.size else 0.0,
            float(np.nanmin(prepared.amplitude)) if prepared.amplitude.size else 0.0,
            prepared.amplitude.size,
            prepared.rpm,
        ]
    )

    if export_state_key not in st.session_state.wm_export_store:
        st.session_state.wm_export_store[export_state_key] = {
            "png_bytes": None,
            "error": None,
        }

    panel_title = f"Waveform {panel_index + 1} — {primary.name}"
    st.markdown(f"### {panel_title}")

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False},
        key=f"wm_waveform_plot_{export_state_key}",
    )

    summary_diag = build_measurement_summary(prepared)
    text_diag = generate_waveform_diagnostic(prepared, summary_diag)

    st.info(text_diag["narrative"])

    st.markdown('<div class="wm-export-actions"></div>', unsafe_allow_html=True)

    left_pad, col_export1, col_export2, col_report, right_pad = st.columns([2.0, 1.2, 1.2, 1.2, 2.0])

    with col_export1:
        if st.button("Prepare PNG HD", key=f"prepare_png_{export_state_key}", use_container_width=True):
            with st.spinner("Generating HD export..."):
                png_bytes, export_error = build_export_png_bytes(fig=fig)
                st.session_state.wm_export_store[export_state_key]["png_bytes"] = png_bytes
                st.session_state.wm_export_store[export_state_key]["error"] = export_error

    with col_export2:
        png_bytes = st.session_state.wm_export_store[export_state_key]["png_bytes"]
        if png_bytes is not None:
            st.download_button(
                "Download PNG HD",
                data=png_bytes,
                file_name=f"{safe_slug(primary.machine)}_{safe_slug(primary.point)}_{safe_slug(primary.variable)}_waveform_hd.png",
                mime="image/png",
                key=f"download_png_{export_state_key}",
                use_container_width=True,
            )
        else:
            st.button(
                "Download PNG HD",
                disabled=True,
                key=f"download_disabled_{export_state_key}",
                use_container_width=True,
            )

    with col_report:
        if st.button("Enviar a Reporte", key=f"send_report_{export_state_key}", use_container_width=True):
            png_bytes_for_report = None
            try:
                png_bytes_for_report, _png_error_for_report = build_export_png_bytes(fig=fig)
            except Exception:
                png_bytes_for_report = None

            queue_waveform_to_report(
                prepared,
                fig,
                panel_title,
                text_diag,
                image_bytes=png_bytes_for_report,
            )
            st.success("Waveform enviado al reporte")

    panel_error = st.session_state.wm_export_store[export_state_key]["error"]
    if panel_error:
        st.warning(f"PNG export error: {panel_error}")


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### Signal Selection")

    signal_name_map = {r.name: r.signal_id for r in records_all}
    signal_names = list(signal_name_map.keys())
    valid_ids = {r.signal_id for r in records_all}

    current_ids = [sid for sid in st.session_state.wm_selected_signal_ids if sid in valid_ids]
    if not current_ids:
        current_ids = [records_all[0].signal_id]
        st.session_state.wm_selected_signal_ids = current_ids

    default_names = [r.name for r in records_all if r.signal_id in current_ids]

    selected_names = st.multiselect(
        "Waveforms to display",
        options=signal_names,
        default=default_names,
    )

    st.session_state.wm_selected_signal_ids = [
        signal_name_map[name] for name in selected_names if name in signal_name_map
    ]

    st.markdown("### View Controls")

    waveform_view_mode = st.selectbox(
        "Waveform view",
        ["Raw", "1X filtered", "2X filtered"],
        index=0,
    )

    normalization_mode = st.selectbox(
        "Normalization",
        ["None", "Remove mean", "Z-score", "Peak normalize"],
        index=1,
    )

    x_axis_unit = st.selectbox(
        "Time display",
        ["ms", "s"],
        index=0,
    )

    y_scale_mode = st.selectbox(
        "Y-axis scale",
        ["Auto", "Manual"],
        index=0,
    )

    show_cycle_start_markers = st.checkbox("Show cycle start markers", value=True)

    default_source_id = (
        st.session_state.wm_selected_signal_ids[0]
        if st.session_state.wm_selected_signal_ids
        else records_all[0].signal_id
    )
    primary_raw_for_unit = next(r for r in records_all if r.signal_id == default_source_id)
    default_y_limit = float(np.max(np.abs(primary_raw_for_unit.amplitude))) if primary_raw_for_unit.amplitude.size else 1.0

    y_limit_abs = None
    if y_scale_mode == "Manual":
        y_limit_abs = st.number_input(
            f"Y limit (± {primary_raw_for_unit.amplitude_unit or 'amp'})",
            value=float(default_y_limit),
            min_value=0.01,
            format="%.2f",
        )

    show_cursor_b = st.checkbox("Show Cursor B", value=True)
    show_right_info_box = st.checkbox("Show info box", value=True)

    st.markdown("### Time Window")

    max_duration = max(primary_raw_for_unit.duration_s, 1e-9)

    window_mode = st.radio("Window mode", ["Full signal", "Custom window"], index=0)

    if window_mode == "Full signal":
        t_min = 0.0
        t_max = max_duration
    else:
        t_min, t_max = st.slider(
            "Time window (s)",
            min_value=0.0,
            max_value=float(max_duration),
            value=(0.0, float(max_duration)),
            step=float(max_duration / 1000.0) if max_duration > 0 else 0.001,
        )

    st.markdown("### Cursors")

    cursor_a = st.number_input(
        "Cursor A (s)",
        min_value=float(t_min),
        max_value=float(t_max),
        value=float(min(max(st.session_state.wm_cursor_a, t_min), t_max)),
        step=float(max((t_max - t_min) / 1000.0, 1e-6)),
        format="%.4f",
    )

    cursor_b = st.number_input(
        "Cursor B (s)",
        min_value=float(t_min),
        max_value=float(t_max),
        value=float(min(max(st.session_state.wm_cursor_b, t_min), t_max)),
        step=float(max((t_max - t_min) / 1000.0, 1e-6)),
        format="%.4f",
    )

    col_ca, col_cb = st.columns(2)
    with col_ca:
        if st.button("A = left", use_container_width=True):
            cursor_a = t_min
    with col_cb:
        if st.button("B = right", use_container_width=True):
            cursor_b = t_max

    st.session_state.wm_cursor_a = cursor_a
    st.session_state.wm_cursor_b = cursor_b


# ------------------------------------------------------------
# Multi-panel render
# ------------------------------------------------------------
selected_ids = [
    signal_id
    for signal_id in st.session_state.wm_selected_signal_ids
    if signal_id in {r.signal_id for r in records_all}
]

if not selected_ids:
    st.info("Selecciona una o más formas de onda en la barra lateral.")
    st.stop()

selected_records = [next(r for r in records_all if r.signal_id == signal_id) for signal_id in selected_ids]
logo_uri = get_logo_data_uri(LOGO_PATH)

for panel_index, primary in enumerate(selected_records):
    render_waveform_panel(
        primary=primary,
        panel_index=panel_index,
        t_min=t_min,
        t_max=t_max,
        cursor_a=cursor_a,
        cursor_b=cursor_b,
        x_axis_unit=x_axis_unit,
        show_cursor_b=show_cursor_b,
        show_right_info_box=show_right_info_box,
        y_scale_mode=y_scale_mode,
        y_limit_abs=y_limit_abs,
        logo_uri=logo_uri,
        waveform_view_mode=waveform_view_mode,
        normalization_mode=normalization_mode,
        show_cycle_start_markers=show_cycle_start_markers,
    )

    if panel_index < len(selected_records) - 1:
        st.markdown("---")
