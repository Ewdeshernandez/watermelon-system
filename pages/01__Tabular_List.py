from __future__ import annotations

import base64
import math
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from core.auth import require_login, render_user_menu

st.set_page_config(page_title="Watermelon System | Tabular List", layout="wide")

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

        .wm-table-shell {
            background: #ffffff;
            border: 1px solid #dbe5f0;
            border-radius: 22px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
            margin-top: 0.6rem;
        }

        .wm-section-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 12px;
            letter-spacing: -0.01em;
        }

        .wm-table-wrap {
            overflow-x: auto;
            border-radius: 16px;
            border: 1px solid #e5edf7;
        }

        table.wm-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background: #ffffff;
        }

        table.wm-table thead tr:first-child {
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
        }

        table.wm-table thead tr:first-child th {
            border-bottom: 1px solid #d9e8fb;
            color: #1d4ed8;
            font-weight: 800;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 12px 10px;
        }

        table.wm-table td {
            border-bottom: 1px solid #edf2f7;
            padding: 10px 8px;
            color: #111827;
            text-align: center;
            white-space: nowrap;
        }

        table.wm-table td:first-child,
        table.wm-table th:first-child {
            text-align: left;
        }

        table.wm-table tbody tr:hover {
            background: #fafcff;
        }

        .wm-status-badge {
            display: inline-block;
            min-width: 72px;
            padding: 5px 10px;
            border-radius: 999px;
            font-weight: 800;
            font-size: 11px;
            letter-spacing: 0.01em;
        }

        .wm-export-actions {
            margin-top: 0.95rem;
            margin-bottom: 0.15rem;
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

    variable = str(find_meta(metadata, ["Variable", "variable"]) or "").lower()
    if "mil" in variable:
        return "mil"
    if "micra" in variable or "µm" in variable or "um" in variable:
        return "µm"
    if "ips" in variable:
        return "ips"
    if "mm/s" in variable:
        return "mm/s"
    if "g" in variable:
        return "g"
    return ""


def infer_measurement_family(metadata: Dict[str, Any], amplitude_unit: str, variable: str) -> str:
    txt = f"{variable} {amplitude_unit}".lower()

    if any(k in txt for k in ["mil", "µm", "um", "micra", "displacement", "prox", "proxim", "shaft"]):
        return "Proximity"

    if any(k in txt for k in ["ips", "mm/s", "velocity", "velo"]):
        return "Velocity"

    if any(k in txt for k in [" g", "g ", "g's", "acc", "acceleration"]):
        return "Acceleration"

    return "Unknown"


def convert_input_time_to_seconds(raw_time: np.ndarray) -> Tuple[np.ndarray, str]:
    if raw_time.size == 0:
        return raw_time, "s"

    raw = raw_time.astype(float, copy=False)
    raw = raw - raw[0]
    duration = float(raw[-1] - raw[0]) if raw.size > 1 else 0.0

    if duration > 5.0:
        return raw / 1000.0, "ms"
    return raw, "s"


def convert_rms_to_display(value_rms: Optional[float], display_mode: str) -> Optional[float]:
    if value_rms is None or not math.isfinite(value_rms):
        return None

    if display_mode == "RMS":
        return value_rms
    if display_mode == "0-Peak":
        return value_rms * math.sqrt(2.0)
    if display_mode == "Peak-to-Peak":
        return value_rms * 2.0 * math.sqrt(2.0)
    return value_rms


def display_suffix(display_mode: str) -> str:
    if display_mode == "RMS":
        return "rms"
    if display_mode == "0-Peak":
        return "0-pk"
    if display_mode == "Peak-to-Peak":
        return "p-p"
    return ""


@dataclass
class SignalRecord:
    signal_id: str
    name: str
    machine: str = "Unknown"
    point: str = "Point 1"
    variable: str = "Waveform"
    amplitude_unit: str = ""
    measurement_family: str = "Unknown"
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
    measurement_family = infer_measurement_family(metadata, amplitude_unit, variable)

    file_name = getattr(signal_obj, "file_name", display_name)
    signal_name = str(file_name or display_name)

    return SignalRecord(
        signal_id=source_key,
        name=signal_name,
        machine=machine,
        point=point,
        variable=variable,
        amplitude_unit=amplitude_unit,
        measurement_family=measurement_family,
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


def overall_rms(record: SignalRecord) -> Optional[float]:
    x = record.amplitude
    if x.size < 2:
        return None
    finite = x[np.isfinite(x)]
    if finite.size < 2:
        return None
    val = float(np.sqrt(np.mean(np.square(finite))))
    return val if math.isfinite(val) else None


def harmonic_fit_amplitude_pp(
    time_s: np.ndarray,
    y: np.ndarray,
    freq_hz: float,
) -> Optional[float]:
    if freq_hz <= 0 or time_s.size < 16 or y.size < 16:
        return None

    n = min(time_s.size, y.size)
    t = time_s[:n]
    x = y[:n].astype(float, copy=True)

    finite_mask = np.isfinite(t) & np.isfinite(x)
    t = t[finite_mask]
    x = x[finite_mask]

    if t.size < 16:
        return None

    x = x - np.mean(x)

    omega = 2.0 * np.pi * freq_hz
    c = np.cos(omega * t)
    s = np.sin(omega * t)

    design = np.column_stack([c, s])
    try:
        coeffs, *_ = np.linalg.lstsq(design, x, rcond=None)
    except Exception:
        return None

    a, b = coeffs
    amp_peak = float(np.sqrt(a * a + b * b))
    amp_pp = 2.0 * amp_peak

    return amp_pp if math.isfinite(amp_pp) else None


def order_amplitude_pp(record: SignalRecord, order: float) -> Optional[float]:
    rpm = record.rpm
    if rpm is None or rpm <= 0:
        return None
    freq_hz = (rpm * order) / 60.0
    return harmonic_fit_amplitude_pp(record.time_s, record.amplitude, freq_hz)


def overall_status(value: Optional[float], alarm: float, danger: float) -> str:
    if value is None or not math.isfinite(value):
        return "No Data"
    if value >= danger:
        return "Danger"
    if value >= alarm:
        return "Alarm"
    return "Normal"


def status_badge_html(status: str) -> str:
    if status == "Danger":
        return '<span class="wm-status-badge" style="background:#fee2e2;color:#991b1b;">Danger</span>'
    if status == "Alarm":
        return '<span class="wm-status-badge" style="background:#fef3c7;color:#92400e;">Alarm</span>'
    if status == "Normal":
        return '<span class="wm-status-badge" style="background:#dcfce7;color:#166534;">Normal</span>'
    return '<span class="wm-status-badge" style="background:#f1f5f9;color:#475569;">No Data</span>'


def build_table_dataframe(
    records: List[SignalRecord],
    alarm: float,
    danger: float,
    criterion: str,
    family_mode: str,
    overall_mode: str,
) -> pd.DataFrame:
    rows = []

    for rec in records:
        family = rec.measurement_family if family_mode == "Auto" else family_mode
        ov_rms = overall_rms(rec)
        ov_display = convert_rms_to_display(ov_rms, overall_mode)

        a05 = order_amplitude_pp(rec, 0.5)
        a10 = order_amplitude_pp(rec, 1.0)
        a20 = order_amplitude_pp(rec, 2.0)

        rows.append(
            {
                "Machine": rec.machine,
                "Point": rec.point,
                "RPM": rec.rpm,
                "Family": family,
                "Alarm": alarm,
                "Danger": danger,
                "Criterion": criterion,
                "Overall": ov_display,
                "Overall RMS Base": ov_rms,
                "0.5X Amp": a05,
                "1X Amp": a10,
                "2X Amp": a20,
                "Unit": rec.amplitude_unit,
                "Overall Mode": overall_mode,
                "Status": overall_status(ov_display, alarm, danger),
                "_signal_name": rec.name,
                "_timestamp": rec.timestamp,
                "_variable": rec.variable,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["Machine", "Point"], kind="stable").reset_index(drop=True)
    return df


def render_top_strip(sample_record: SignalRecord, total_rows: int, logo_uri: Optional[str], criterion: str, overall_mode: str) -> None:
    if logo_uri:
        logo_html = f'<img src="{logo_uri}" style="height:42px; width:auto; object-fit:contain;" />'
    else:
        logo_html = (
            '<div style="width:42px;height:42px;border-radius:12px;background:#1ea7ff;color:white;'
            'display:flex;align-items:center;justify-content:center;font-weight:800;font-size:14px;">WM</div>'
        )

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
                    <span><b>{sample_record.machine}</b></span>
                    <span style="color:#94a3b8;">|</span>
                    <span>{sample_record.variable} | Tabular List</span>
                    <span style="color:#94a3b8;">|</span>
                    <span><b>Rows:</b> {total_rows}</span>
                    <span style="color:#94a3b8;">|</span>
                    <span><b>Criterion:</b> {criterion}</span>
                    <span style="color:#94a3b8;">|</span>
                    <span><b>Overall:</b> {overall_mode}</span>
                    <span style="color:#94a3b8;">|</span>
                    <span><b>Harmonics:</b> Peak-to-Peak</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_table(df: pd.DataFrame) -> None:
    rows_html = []

    for _, row in df.iterrows():
        unit = str(row["Unit"]).strip()
        overall_suffix = display_suffix(str(row["Overall Mode"]))
        overall_unit = f"{unit} {overall_suffix}".strip() if unit else overall_suffix
        harm_unit = f"{unit} p-p".strip() if unit else "p-p"

        row_html = (
            "<tr>"
            f"<td>{row['Machine']}</td>"
            f"<td>{row['Point']}</td>"
            f"<td>{format_number(row['RPM'], 0)}</td>"
            f"<td>{row['Family']}</td>"
            f"<td>{format_number(row['Alarm'], 3)}</td>"
            f"<td>{format_number(row['Danger'], 3)}</td>"
            f"<td>{row['Criterion']}</td>"
            f"<td>{status_badge_html(row['Status'])}</td>"
            f"<td>{format_number(row['Overall'], 3)} {overall_unit}</td>"
            f"<td>{format_number(row['0.5X Amp'], 3)} {harm_unit}</td>"
            f"<td>{format_number(row['1X Amp'], 3)} {harm_unit}</td>"
            f"<td>{format_number(row['2X Amp'], 3)} {harm_unit}</td>"
            "</tr>"
        )
        rows_html.append(row_html)

    table_html = (
        '<div class="wm-table-shell">'
        '<div class="wm-section-title">Tabular List</div>'
        '<div class="wm-table-wrap">'
        '<table class="wm-table">'
        "<thead>"
        "<tr>"
        "<th>Machine</th>"
        "<th>Point</th>"
        "<th>RPM</th>"
        "<th>Family</th>"
        "<th>Alarm</th>"
        "<th>Danger</th>"
        "<th>Criterion Based</th>"
        "<th>Status</th>"
        "<th>Overall</th>"
        "<th>0.5X Amplitude</th>"
        "<th>1X Amplitude</th>"
        "<th>2X Amplitude</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(rows_html) +
        "</tbody>"
        "</table>"
        "</div>"
        "</div>"
    )

    st.markdown(table_html, unsafe_allow_html=True)


def _load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _status_style(status: str) -> Tuple[str, str]:
    if status == "Danger":
        return "#fee2e2", "#991b1b"
    if status == "Alarm":
        return "#fef3c7", "#92400e"
    if status == "Normal":
        return "#dcfce7", "#166534"
    return "#f1f5f9", "#475569"


def build_png_report(df: pd.DataFrame, sample_record: SignalRecord, criterion: str, overall_mode: str) -> bytes:
    width = 4200
    row_h = 88
    top_h = 180
    title_h = 84
    table_header_h = 72
    n_rows = len(df)
    height = top_h + title_h + table_header_h + n_rows * row_h + 170

    bg = "#f3f4f6"
    white = "#ffffff"
    border = "#dbe5f0"
    blue = "#1d4ed8"
    text = "#111827"
    header_fill = "#eef6ff"

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    font_title = _load_font(52, True)
    font_small = _load_font(26, False)
    font_header = _load_font(23, True)
    font_cell = _load_font(22, False)
    font_badge = _load_font(19, True)

    card_x0 = 60
    card_x1 = width - 60
    top_y0 = 40
    top_y1 = top_y0 + 110

    draw.rounded_rectangle((card_x0, top_y0, card_x1, top_y1), radius=28, fill=white, outline=border, width=2)

    logo_x = card_x0 + 26
    logo_y = top_y0 + 22

    if LOGO_PATH.exists():
        try:
            logo = Image.open(LOGO_PATH).convert("RGBA")
            logo.thumbnail((110, 60))
            img.paste(logo, (logo_x, logo_y), logo)
        except Exception:
            draw.rounded_rectangle((logo_x, logo_y, logo_x + 58, logo_y + 58), radius=12, fill="#1ea7ff")
            draw.text((logo_x + 12, logo_y + 14), "WM", font=font_small, fill="white")
    else:
        draw.rounded_rectangle((logo_x, logo_y, logo_x + 58, logo_y + 58), radius=12, fill="#1ea7ff")
        draw.text((logo_x + 12, logo_y + 14), "WM", font=font_small, fill="white")

    meta_text = (
        f"{sample_record.machine}   |   {sample_record.variable} | Tabular List   |   Rows: {len(df)}   |   "
        f"Criterion: {criterion}   |   Overall: {overall_mode}   |   Harmonics: Peak-to-Peak"
    )
    draw.text((logo_x + 150, top_y0 + 38), meta_text, font=font_small, fill=text)

    shell_y0 = top_y1 + 32
    shell_y1 = height - 50
    draw.rounded_rectangle((card_x0, shell_y0, card_x1, shell_y1), radius=28, fill=white, outline=border, width=2)

    draw.text((card_x0 + 24, shell_y0 + 20), "Tabular List", font=font_title, fill=text)

    table_x0 = card_x0 + 24
    table_x1 = card_x1 - 24
    table_y0 = shell_y0 + 100

    col_widths = [340, 330, 200, 300, 240, 240, 560, 260, 360, 380, 380, 380]
    scale = (table_x1 - table_x0) / sum(col_widths)
    col_widths = [int(w * scale) for w in col_widths]

    col_x = [table_x0]
    for w in col_widths:
        col_x.append(col_x[-1] + w)

    draw.rounded_rectangle(
        (table_x0, table_y0, table_x1, table_y0 + table_header_h + n_rows * row_h),
        radius=22,
        fill=white,
        outline=border,
        width=2,
    )

    headers = [
        "MACHINE", "POINT", "RPM", "FAMILY", "ALARM", "DANGER",
        "CRITERION BASED", "STATUS", "OVERALL", "0.5X AMPLITUDE",
        "1X AMPLITUDE", "2X AMPLITUDE"
    ]

    for i, label in enumerate(headers):
        x0 = col_x[i]
        x1 = col_x[i + 1]
        draw.rectangle((x0, table_y0, x1, table_y0 + table_header_h), fill=header_fill, outline=border, width=1)
        tw = draw.textbbox((0, 0), label, font=font_header)
        tx = x0 + 14
        ty = table_y0 + (table_header_h - (tw[3] - tw[1])) / 2 - 2
        draw.text((tx, ty), label, font=font_header, fill=blue)

    for r, (_, row) in enumerate(df.iterrows()):
        y0 = table_y0 + table_header_h + r * row_h
        y1 = y0 + row_h
        fill = "#ffffff" if r % 2 == 0 else "#fafcff"
        draw.rectangle((table_x0, y0, table_x1, y1), fill=fill, outline=border, width=1)

        unit = str(row["Unit"]).strip()
        overall_suffix = display_suffix(str(row["Overall Mode"]))
        overall_unit = f"{unit} {overall_suffix}".strip() if unit else overall_suffix
        harm_unit = f"{unit} p-p".strip() if unit else "p-p"

        cells = [
            str(row["Machine"]),
            str(row["Point"]),
            format_number(row["RPM"], 0),
            str(row["Family"]),
            format_number(row["Alarm"], 3),
            format_number(row["Danger"], 3),
            str(row["Criterion"]),
            None,
            f"{format_number(row['Overall'], 3)} {overall_unit}",
            f"{format_number(row['0.5X Amp'], 3)} {harm_unit}",
            f"{format_number(row['1X Amp'], 3)} {harm_unit}",
            f"{format_number(row['2X Amp'], 3)} {harm_unit}",
        ]

        for c, cell in enumerate(cells):
            x0 = col_x[c]
            x1 = col_x[c + 1]

            if c == 7:
                status = str(row["Status"])
                bg_badge, fg_badge = _status_style(status)
                badge_w = 130
                badge_h = 36
                bx0 = x0 + (x1 - x0 - badge_w) / 2
                by0 = y0 + (row_h - badge_h) / 2
                bx1 = bx0 + badge_w
                by1 = by0 + badge_h
                draw.rounded_rectangle((bx0, by0, bx1, by1), radius=18, fill=bg_badge)
                tw = draw.textbbox((0, 0), status, font=font_badge)
                tx = bx0 + (badge_w - (tw[2] - tw[0])) / 2
                ty = by0 + (badge_h - (tw[3] - tw[1])) / 2 - 1
                draw.text((tx, ty), status, font=font_badge, fill=fg_badge)
            else:
                if c in [0, 1, 6]:
                    draw.text((x0 + 14, y0 + 28), cell, font=font_cell, fill=text)
                else:
                    tw = draw.textbbox((0, 0), cell, font=font_cell)
                    tx = x0 + (x1 - x0 - (tw[2] - tw[0])) / 2
                    ty = y0 + (row_h - (tw[3] - tw[1])) / 2 - 1
                    draw.text((tx, ty), cell, font=font_cell, fill=text)

    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


if "wm_tabular_export_png_bytes" not in st.session_state:
    st.session_state.wm_tabular_export_png_bytes = None

if "wm_tabular_export_error" not in st.session_state:
    st.session_state.wm_tabular_export_error = None


records_all = load_signals_from_session()

if not records_all:
    st.warning("No se pudieron cargar señales válidas desde st.session_state['signals'].")
    st.stop()

with st.sidebar:
    st.markdown("### Tabular List Setup")

    criterion_options = [
        "ISO 20816-3",
        "Boletín fabricante",
        "API 670",
        "Criterio interno SIGA",
        "Custom",
    ]

    criterion_selected = st.selectbox(
        "Criterion based",
        options=criterion_options,
        index=0,
    )

    criterion_text = criterion_selected
    if criterion_selected == "Custom":
        criterion_text = st.text_input(
            "Custom criterion",
            value="Criterio usuario",
        ).strip() or "Criterio usuario"

    measurement_family = st.selectbox(
        "Measurement family",
        options=["Auto", "Proximity", "Velocity", "Acceleration"],
        index=0,
    )

    if measurement_family in ["Auto", "Proximity"]:
        overall_mode_options = ["Peak-to-Peak"]
    elif measurement_family == "Velocity":
        overall_mode_options = ["RMS", "0-Peak"]
    elif measurement_family == "Acceleration":
        overall_mode_options = ["RMS", "0-Peak"]
    else:
        overall_mode_options = ["RMS"]

    overall_mode = st.selectbox(
        "Overall display mode",
        options=overall_mode_options,
        index=0,
    )

    alarm_value = st.number_input(
        f"Alarm threshold ({overall_mode})",
        min_value=0.0,
        value=4.5,
        step=0.1,
        format="%.3f",
    )

    danger_value = st.number_input(
        f"Danger threshold ({overall_mode})",
        min_value=0.0,
        value=7.1,
        step=0.1,
        format="%.3f",
    )

    if danger_value < alarm_value:
        st.warning("Danger debería ser mayor o igual que Alarm.")

logo_uri = get_logo_data_uri(LOGO_PATH)

df_table = build_table_dataframe(
    records=records_all,
    alarm=float(alarm_value),
    danger=float(danger_value),
    criterion=criterion_text,
    family_mode=measurement_family,
    overall_mode=overall_mode,
)

if df_table.empty:
    st.warning("No fue posible construir la tabla.")
    st.stop()

render_top_strip(records_all[0], len(df_table), logo_uri, criterion_text, overall_mode)
render_table(df_table)

st.markdown('<div class="wm-export-actions"></div>', unsafe_allow_html=True)

left_pad, col_export1, col_export2, right_pad = st.columns([2.4, 1.3, 1.3, 2.4])

with col_export1:
    if st.button("Prepare PNG HD", use_container_width=True):
        try:
            png_bytes = build_png_report(
                df=df_table,
                sample_record=records_all[0],
                criterion=criterion_text,
                overall_mode=overall_mode,
            )
            st.session_state.wm_tabular_export_png_bytes = png_bytes
            st.session_state.wm_tabular_export_error = None
        except Exception as e:
            st.session_state.wm_tabular_export_png_bytes = None
            st.session_state.wm_tabular_export_error = str(e)

with col_export2:
    if st.session_state.wm_tabular_export_png_bytes is not None:
        st.download_button(
            "Download PNG HD",
            data=st.session_state.wm_tabular_export_png_bytes,
            file_name="watermelon_tabular_list_hd.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.button("Download PNG HD", disabled=True, use_container_width=True)

if st.session_state.wm_tabular_export_error:
    st.warning(f"PNG export error: {st.session_state.wm_tabular_export_error}")