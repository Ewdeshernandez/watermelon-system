from __future__ import annotations

import base64
import math
import re
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from core.auth import require_login, render_user_menu

st.set_page_config(page_title="Watermelon System | Phase Analysis", layout="wide")

require_login()
render_user_menu()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"


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

        .wm-phase-table-shell {
            background: #ffffff;
            border: 1px solid #dbe5f0;
            border-radius: 22px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
            margin-top: 0.6rem;
        }

        .wm-phase-section-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 12px;
            letter-spacing: -0.01em;
        }

        .wm-phase-table-wrap {
            overflow-x: auto;
            border-radius: 16px;
            border: 1px solid #e5edf7;
        }

        table.wm-phase-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background: #ffffff;
        }

        table.wm-phase-table thead tr:first-child {
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
        }

        table.wm-phase-table thead tr:first-child th {
            border-bottom: 1px solid #d9e8fb;
            color: #1d4ed8;
            font-weight: 800;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 12px 10px;
        }

        table.wm-phase-table thead tr:nth-child(2) {
            background: #f8fbff;
        }

        table.wm-phase-table thead tr:nth-child(2) th {
            border-bottom: 1px solid #e5edf7;
            color: #334155;
            font-weight: 700;
            padding: 10px 8px;
        }

        table.wm-phase-table td {
            border-bottom: 1px solid #edf2f7;
            padding: 10px 8px;
            color: #111827;
            text-align: center;
            white-space: nowrap;
        }

        table.wm-phase-table td:first-child,
        table.wm-phase-table th:first-child {
            text-align: left;
        }

        table.wm-phase-table tbody tr:hover {
            background: #fafcff;
        }

        .wm-badge {
            display: inline-block;
            min-width: 72px;
            padding: 5px 10px;
            border-radius: 999px;
            font-weight: 800;
            font-size: 11px;
            letter-spacing: 0.01em;
        }

        .wm-export-actions {
            margin-top: 0.9rem;
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
        return str(y_axis_unit).strip()

    generic = find_meta(metadata, ["Unit", "unit", "units"])
    if generic:
        return str(generic).strip()

    return ""


def convert_input_time_to_seconds(raw_time: np.ndarray) -> Tuple[np.ndarray, str]:
    if raw_time.size == 0:
        return raw_time, "s"

    raw = raw_time.astype(float, copy=False)
    raw = raw - raw[0]
    duration = float(raw[-1] - raw[0]) if raw.size > 1 else 0.0

    if duration > 5.0:
        return raw / 1000.0, "ms"
    return raw, "s"


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


def signal_object_to_record(
    signal_obj: Any,
    source_key: str,
    display_name: str,
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

    if n < 32:
        return None

    finite_mask = np.isfinite(raw_time) & np.isfinite(amplitude)
    raw_time = raw_time[finite_mask]
    amplitude = amplitude[finite_mask]

    if raw_time.size < 32 or amplitude.size < 32:
        return None

    time_s, detected_unit = convert_input_time_to_seconds(raw_time)

    sort_idx = np.argsort(time_s)
    time_s = time_s[sort_idx]
    amplitude = amplitude[sort_idx]

    unique_mask = np.ones_like(time_s, dtype=bool)
    if time_s.size > 1:
        unique_mask[1:] = np.diff(time_s) > 0

    time_s = time_s[unique_mask]
    amplitude = amplitude[unique_mask]

    if time_s.size < 32 or amplitude.size < 32:
        return None

    sample_rate_hz = infer_sample_rate_from_seconds(time_s)

    metadata = getattr(signal_obj, "metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}

    machine = str(find_meta(metadata, ["Machine", "machine", "Machine Name"]) or "Unknown")
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


def load_signals_from_session() -> List[SignalRecord]:
    records: List[SignalRecord] = []
    signals_dict = st.session_state.get("signals", {})

    if isinstance(signals_dict, dict):
        for key, value in signals_dict.items():
            rec = signal_object_to_record(
                value,
                source_key=f"signals.{key}",
                display_name=str(key),
            )
            if rec is not None:
                records.append(rec)

    return records


def parse_uploaded_csv(file_obj) -> Optional[SignalRecord]:
    try:
        raw_text = file_obj.getvalue().decode("utf-8-sig", errors="ignore")
        df = pd.read_csv(StringIO(raw_text), header=None)
    except Exception:
        return None

    if df.shape[1] < 2 or df.shape[0] < 16:
        return None

    metadata: Dict[str, Any] = {}
    header_row = None

    for i in range(df.shape[0]):
        k = str(df.iloc[i, 0]).strip()
        v = df.iloc[i, 1]
        if k.lower() == "x-axis value":
            header_row = i
            break
        metadata[k] = v

    if header_row is None:
        return None

    data = df.iloc[header_row + 1 :, :2].copy()
    data.columns = ["x", "y"]
    data["x"] = pd.to_numeric(data["x"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data = data.dropna()

    if len(data) < 32:
        return None

    raw_time = data["x"].to_numpy(dtype=float)
    amplitude = data["y"].to_numpy(dtype=float)
    time_s, detected_unit = convert_input_time_to_seconds(raw_time)
    sample_rate_hz = infer_sample_rate_from_seconds(time_s)

    machine = str(metadata.get("Machine Name", "Unknown"))
    point = str(metadata.get("Point Name", file_obj.name))
    variable = str(metadata.get("Variable", "Waveform"))
    rpm = parse_first_float(metadata.get("Sample Speed"))
    amplitude_unit = str(metadata.get("Y-Axis Unit", "")).strip()
    timestamp = metadata.get("Timestamp")

    return SignalRecord(
        signal_id=f"upload.{file_obj.name}",
        name=str(file_obj.name),
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
        source_key=f"upload.{file_obj.name}",
        source_time_unit=detected_unit,
    )


def harmonic_fit_amplitude_phase(
    time_s: np.ndarray,
    y: np.ndarray,
    freq_hz: float,
) -> Tuple[Optional[float], Optional[float]]:
    if freq_hz <= 0 or time_s.size < 16 or y.size < 16:
        return None, None

    n = min(time_s.size, y.size)
    t = time_s[:n]
    x = y[:n].astype(float, copy=True)

    finite_mask = np.isfinite(t) & np.isfinite(x)
    t = t[finite_mask]
    x = x[finite_mask]

    if t.size < 16:
        return None, None

    x = x - np.mean(x)

    omega = 2.0 * np.pi * freq_hz
    c = np.cos(omega * t)
    s = np.sin(omega * t)

    design = np.column_stack([c, s])
    try:
        coeffs, *_ = np.linalg.lstsq(design, x, rcond=None)
    except Exception:
        return None, None

    a, b = coeffs
    amp = float(np.sqrt(a * a + b * b))
    phase_deg = float(np.degrees(np.arctan2(-b, a)) % 360.0)

    if not math.isfinite(amp) or not math.isfinite(phase_deg):
        return None, None

    return amp, phase_deg


def circular_mean_deg(phases_deg: List[float]) -> Optional[float]:
    if not phases_deg:
        return None
    radians = np.deg2rad(phases_deg)
    vec = np.exp(1j * radians)
    mean_vec = np.mean(vec)
    if abs(mean_vec) < 1e-12:
        return None
    angle = float(np.degrees(np.angle(mean_vec)) % 360.0)
    return angle


def phase_stability_percent(
    time_s: np.ndarray,
    y: np.ndarray,
    freq_hz: float,
    segments: int = 6,
) -> Optional[float]:
    if freq_hz <= 0 or time_s.size < max(segments * 16, 64):
        return None

    n = min(time_s.size, y.size)
    t = time_s[:n]
    x = y[:n]

    phases: List[float] = []
    chunks_t = np.array_split(t, segments)
    chunks_x = np.array_split(x, segments)

    for tt, xx in zip(chunks_t, chunks_x):
        amp, ph = harmonic_fit_amplitude_phase(tt, xx, freq_hz)
        if amp is not None and ph is not None and math.isfinite(ph):
            phases.append(ph)

    if len(phases) < 3:
        return None

    radians = np.deg2rad(phases)
    resultant = abs(np.mean(np.exp(1j * radians)))
    stability = float(np.clip(resultant * 100.0, 0.0, 100.0))
    return stability


def get_order_metrics(record: SignalRecord, order: float) -> Dict[str, Any]:
    rpm = record.rpm
    if rpm is None or rpm <= 0:
        return {"amp": None, "phase": None, "stability": None}

    freq_hz = (rpm * order) / 60.0
    amp, phase = harmonic_fit_amplitude_phase(record.time_s, record.amplitude, freq_hz)
    stability = phase_stability_percent(record.time_s, record.amplitude, freq_hz)

    return {
        "amp": amp,
        "phase": phase,
        "stability": stability,
    }


def stability_badge_html(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return '<span class="wm-badge" style="background:#f1f5f9;color:#475569;">—</span>'

    if value >= 85:
        bg = "#dcfce7"
        fg = "#166534"
    elif value >= 65:
        bg = "#fef3c7"
        fg = "#92400e"
    else:
        bg = "#fee2e2"
        fg = "#991b1b"

    return f'<span class="wm-badge" style="background:{bg};color:{fg};">{value:.1f}%</span>'


def build_phase_dataframe(records: List[SignalRecord]) -> pd.DataFrame:
    rows = []

    for rec in records:
        m05 = get_order_metrics(rec, 0.5)
        m10 = get_order_metrics(rec, 1.0)
        m20 = get_order_metrics(rec, 2.0)

        rows.append(
            {
                "Signal": rec.name,
                "Machine": rec.machine,
                "Point": rec.point,
                "RPM": rec.rpm,
                "0.5X Amp": m05["amp"],
                "0.5X Phase": m05["phase"],
                "0.5X Stability": m05["stability"],
                "1X Amp": m10["amp"],
                "1X Phase": m10["phase"],
                "1X Stability": m10["stability"],
                "2X Amp": m20["amp"],
                "2X Phase": m20["phase"],
                "2X Stability": m20["stability"],
                "Timestamp": rec.timestamp,
                "Unit": rec.amplitude_unit,
                "Variable": rec.variable,
            }
        )

    return pd.DataFrame(rows)


def render_top_strip(record: SignalRecord, selected_count: int, logo_uri: Optional[str]) -> None:
    if logo_uri:
        logo_html = f'<img src="{logo_uri}" style="height:42px; width:auto; object-fit:contain;" />'
    else:
        logo_html = """
        <div style="
            width:42px;height:42px;border-radius:12px;background:#1ea7ff;color:white;
            display:flex;align-items:center;justify-content:center;font-weight:800;font-size:14px;
        ">WM</div>
        """

    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #dbe5f0;
            border-radius:20px;
            padding:18px 20px;
            box-shadow:0 12px 28px rgba(15,23,42,0.06);
            margin-bottom:12px;
        ">
            <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
                {logo_html}
                <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:13px;color:#1f2937;">
                    <span><b>{record.machine}</b></span>
                    <span style="color:#94a3b8;">|</span>
                    <span>{record.point}</span>
                    <span style="color:#94a3b8;">|</span>
                    <span>{record.variable} | Phase Dashboard</span>
                    <span style="color:#94a3b8;">|</span>
                    <span><b>RPM:</b> {format_number(record.rpm, 0)}</span>
                    <span style="color:#94a3b8;">|</span>
                    <span><b>Signals:</b> {selected_count}</span>
                    <span style="color:#94a3b8;">|</span>
                    <span>{record.timestamp or "—"}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_phase_table(df: pd.DataFrame) -> None:
    html = []
    html.append('<div class="wm-phase-table-shell">')
    html.append('<div class="wm-phase-section-title">0.5X / 1X / 2X Phase Summary</div>')
    html.append('<div class="wm-phase-table-wrap">')
    html.append('<table class="wm-phase-table">')
    html.append("<thead>")
    html.append(
        """
        <tr>
            <th rowspan="2">Signal</th>
            <th rowspan="2">Machine</th>
            <th rowspan="2">Point</th>
            <th rowspan="2">RPM</th>
            <th colspan="3">0.5X</th>
            <th colspan="3">1X</th>
            <th colspan="3">2X</th>
            <th rowspan="2">Timestamp</th>
        </tr>
        """
    )
    html.append(
        """
        <tr>
            <th>Amp</th>
            <th>Phase</th>
            <th>Stability</th>
            <th>Amp</th>
            <th>Phase</th>
            <th>Stability</th>
            <th>Amp</th>
            <th>Phase</th>
            <th>Stability</th>
        </tr>
        """
    )
    html.append("</thead>")
    html.append("<tbody>")

    for _, row in df.iterrows():
        unit = str(row["Unit"]).strip()
        unit_txt = f" {unit}" if unit else ""

        html.append(
            f"""
            <tr>
                <td>{row["Signal"]}</td>
                <td>{row["Machine"]}</td>
                <td>{row["Point"]}</td>
                <td>{format_number(row["RPM"], 0)}</td>

                <td>{format_number(row["0.5X Amp"], 3)}{unit_txt}</td>
                <td>{format_number(row["0.5X Phase"], 1)}°</td>
                <td>{stability_badge_html(row["0.5X Stability"])}</td>

                <td>{format_number(row["1X Amp"], 3)}{unit_txt}</td>
                <td>{format_number(row["1X Phase"], 1)}°</td>
                <td>{stability_badge_html(row["1X Stability"])}</td>

                <td>{format_number(row["2X Amp"], 3)}{unit_txt}</td>
                <td>{format_number(row["2X Phase"], 1)}°</td>
                <td>{stability_badge_html(row["2X Stability"])}</td>

                <td>{row["Timestamp"] or "—"}</td>
            </tr>
            """
        )

    html.append("</tbody>")
    html.append("</table>")
    html.append("</div>")
    html.append("</div>")

    st.markdown("".join(html), unsafe_allow_html=True)


if "wm_phase_primary_signal_id" not in st.session_state:
    st.session_state.wm_phase_primary_signal_id = None


records_all = load_signals_from_session()

with st.sidebar:
    st.markdown("### Signal Selection")

    uploaded_files = st.file_uploader(
        "Optional CSV upload",
        type=["csv"],
        accept_multiple_files=True,
        key="wm_phase_csv_uploads",
    )

    uploaded_records: List[SignalRecord] = []
    if uploaded_files:
        for f in uploaded_files:
            rec = parse_uploaded_csv(f)
            if rec is not None:
                uploaded_records.append(rec)

    all_records = records_all + uploaded_records

    if not all_records:
        st.warning("No valid signals available.")
        st.stop()

    signal_name_map = {r.name: r.signal_id for r in all_records}
    signal_names = list(signal_name_map.keys())

    if st.session_state.wm_phase_primary_signal_id not in [r.signal_id for r in all_records]:
        st.session_state.wm_phase_primary_signal_id = all_records[0].signal_id

    current_name = next(
        (r.name for r in all_records if r.signal_id == st.session_state.wm_phase_primary_signal_id),
        signal_names[0],
    )

    selected_primary_name = st.selectbox(
        "Primary signal",
        options=signal_names,
        index=signal_names.index(current_name),
    )
    st.session_state.wm_phase_primary_signal_id = signal_name_map[selected_primary_name]

    st.markdown("### Multi-Signal Comparison")

    default_compare = signal_names[: min(6, len(signal_names))]
    selected_compare_names = st.multiselect(
        "Signals to compare",
        options=signal_names,
        default=default_compare,
    )

all_records = records_all + uploaded_records

if not all_records:
    st.warning("No se pudieron cargar señales válidas.")
    st.stop()

selected_records = [r for r in all_records if r.name in selected_compare_names]

if not selected_records:
    st.warning("Selecciona al menos una señal.")
    st.stop()

primary = next(r for r in all_records if r.signal_id == st.session_state.wm_phase_primary_signal_id)
logo_uri = get_logo_data_uri(LOGO_PATH)

render_top_strip(primary, len(selected_records), logo_uri)

df_phase = build_phase_dataframe(selected_records)

if df_phase.empty:
    st.warning("No fue posible calcular métricas de fase.")
    st.stop()

render_phase_table(df_phase)

export_df = df_phase[
    [
        "Signal",
        "Machine",
        "Point",
        "RPM",
        "0.5X Amp",
        "0.5X Phase",
        "0.5X Stability",
        "1X Amp",
        "1X Phase",
        "1X Stability",
        "2X Amp",
        "2X Phase",
        "2X Stability",
        "Timestamp",
        "Unit",
        "Variable",
    ]
].copy()

csv_bytes = export_df.to_csv(index=False).encode("utf-8")

st.markdown('<div class="wm-export-actions"></div>', unsafe_allow_html=True)

left_pad, col_export, right_pad = st.columns([3.1, 1.8, 3.1])

with col_export:
    st.download_button(
        "Export CSV",
        data=csv_bytes,
        file_name="watermelon_phase_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )