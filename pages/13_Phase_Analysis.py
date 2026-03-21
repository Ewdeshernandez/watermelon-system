from __future__ import annotations

import base64
import math
import re
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

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

        .wm-phase-subnote {
            font-size: 0.90rem;
            color: #64748b;
            margin-bottom: 12px;
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
    amp_peak = float(np.sqrt(a * a + b * b))
    amp_pp = 2.0 * amp_peak
    phase_deg = float(np.degrees(np.arctan2(-b, a)) % 360.0)

    if not math.isfinite(amp_pp) or not math.isfinite(phase_deg):
        return None, None

    return amp_pp, phase_deg


def fft_local_amplitude_pp(
    time_s: np.ndarray,
    y: np.ndarray,
    target_hz: float,
) -> Optional[float]:
    if target_hz <= 0 or time_s.size < 32 or y.size < 32:
        return None

    n = min(time_s.size, y.size)
    t = time_s[:n]
    x = y[:n].astype(float, copy=True)

    finite_mask = np.isfinite(t) & np.isfinite(x)
    t = t[finite_mask]
    x = x[finite_mask]
    if t.size < 32:
        return None

    x = x - np.mean(x)

    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if dt.size == 0:
        return None

    fs = 1.0 / float(np.mean(dt))
    nfft = int(2 ** math.ceil(math.log2(len(x))))
    window = np.hanning(len(x))
    gain = float(np.mean(window))
    xw = x * window

    fft_vals = np.fft.rfft(xw, n=nfft)
    freq_hz = np.fft.rfftfreq(nfft, d=1.0 / fs)

    peak_amp = (2.0 / len(x)) * np.abs(fft_vals)
    peak_amp = peak_amp / max(gain, 1e-12)

    if peak_amp.size > 0:
        peak_amp[0] *= 0.5
    if nfft % 2 == 0 and peak_amp.size > 1:
        peak_amp[-1] *= 0.5

    band = max(target_hz * 0.08, 0.2)
    mask = (freq_hz >= target_hz - band) & (freq_hz <= target_hz + band)
    mask &= np.isfinite(peak_amp)

    if not np.any(mask):
        return None

    amp_peak = float(np.max(peak_amp[mask]))
    amp_pp = 2.0 * amp_peak
    return amp_pp if math.isfinite(amp_pp) else None


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


def confidence_score(
    fit_amp_pp: Optional[float],
    fft_amp_pp: Optional[float],
    stability_pct: Optional[float],
) -> Optional[float]:
    if stability_pct is None or not math.isfinite(stability_pct):
        return None

    if fit_amp_pp is None or fft_amp_pp is None or fit_amp_pp <= 0 or fft_amp_pp <= 0:
        return float(np.clip(stability_pct * 0.7, 0.0, 100.0))

    ratio = min(fit_amp_pp, fft_amp_pp) / max(fit_amp_pp, fft_amp_pp)
    conf = 0.65 * stability_pct + 35.0 * ratio
    return float(np.clip(conf, 0.0, 100.0))


def get_order_metrics(record: SignalRecord, order: float) -> Dict[str, Any]:
    rpm = record.rpm
    if rpm is None or rpm <= 0:
        return {
            "freq_cpm": None,
            "fit_amp_pp": None,
            "fft_amp_pp": None,
            "phase": None,
            "stability": None,
            "confidence": None,
        }

    freq_hz = (rpm * order) / 60.0
    freq_cpm = rpm * order

    fit_amp_pp, phase = harmonic_fit_amplitude_phase(record.time_s, record.amplitude, freq_hz)
    fft_amp_pp = fft_local_amplitude_pp(record.time_s, record.amplitude, freq_hz)
    stability = phase_stability_percent(record.time_s, record.amplitude, freq_hz)
    confidence = confidence_score(fit_amp_pp, fft_amp_pp, stability)

    return {
        "freq_cpm": freq_cpm,
        "fit_amp_pp": fit_amp_pp,
        "fft_amp_pp": fft_amp_pp,
        "phase": phase,
        "stability": stability,
        "confidence": confidence,
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


def confidence_badge_html(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return '<span class="wm-badge" style="background:#f1f5f9;color:#475569;">—</span>'

    if value >= 85:
        bg = "#dbeafe"
        fg = "#1d4ed8"
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

                "0.5X Freq": m05["freq_cpm"],
                "0.5X Fit Amp": m05["fit_amp_pp"],
                "0.5X FFT Amp": m05["fft_amp_pp"],
                "0.5X Phase": m05["phase"],
                "0.5X Stability": m05["stability"],
                "0.5X Confidence": m05["confidence"],

                "1X Freq": m10["freq_cpm"],
                "1X Fit Amp": m10["fit_amp_pp"],
                "1X FFT Amp": m10["fft_amp_pp"],
                "1X Phase": m10["phase"],
                "1X Stability": m10["stability"],
                "1X Confidence": m10["confidence"],

                "2X Freq": m20["freq_cpm"],
                "2X Fit Amp": m20["fit_amp_pp"],
                "2X FFT Amp": m20["fft_amp_pp"],
                "2X Phase": m20["phase"],
                "2X Stability": m20["stability"],
                "2X Confidence": m20["confidence"],

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
                    <span><b>{record.machine}</b></span>
                    <span style="color:#94a3b8;">|</span>
                    <span>{record.point}</span>
                    <span style="color:#94a3b8;">|</span>
                    <span>{record.variable} | Phase Dashboard | Amp = Peak-to-Peak</span>
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
    rows_html = []

    for _, row in df.iterrows():
        unit = str(row["Unit"]).strip()
        unit_txt = f" {unit} p-p" if unit else " p-p"

        row_html = (
            "<tr>"
            f"<td>{row['Signal']}</td>"
            f"<td>{row['Machine']}</td>"
            f"<td>{row['Point']}</td>"
            f"<td>{format_number(row['RPM'], 0)}</td>"

            f"<td>{format_number(row['0.5X Freq'], 1)}</td>"
            f"<td>{format_number(row['0.5X Fit Amp'], 3)}{unit_txt}</td>"
            f"<td>{format_number(row['0.5X FFT Amp'], 3)}{unit_txt}</td>"
            f"<td>{format_number(row['0.5X Phase'], 1)}°</td>"
            f"<td>{stability_badge_html(row['0.5X Stability'])}</td>"
            f"<td>{confidence_badge_html(row['0.5X Confidence'])}</td>"

            f"<td>{format_number(row['1X Freq'], 1)}</td>"
            f"<td>{format_number(row['1X Fit Amp'], 3)}{unit_txt}</td>"
            f"<td>{format_number(row['1X FFT Amp'], 3)}{unit_txt}</td>"
            f"<td>{format_number(row['1X Phase'], 1)}°</td>"
            f"<td>{stability_badge_html(row['1X Stability'])}</td>"
            f"<td>{confidence_badge_html(row['1X Confidence'])}</td>"

            f"<td>{format_number(row['2X Freq'], 1)}</td>"
            f"<td>{format_number(row['2X Fit Amp'], 3)}{unit_txt}</td>"
            f"<td>{format_number(row['2X FFT Amp'], 3)}{unit_txt}</td>"
            f"<td>{format_number(row['2X Phase'], 1)}°</td>"
            f"<td>{stability_badge_html(row['2X Stability'])}</td>"
            f"<td>{confidence_badge_html(row['2X Confidence'])}</td>"

            f"<td>{row['Timestamp'] or '—'}</td>"
            "</tr>"
        )
        rows_html.append(row_html)

    table_html = (
        '<div class="wm-phase-table-shell">'
        '<div class="wm-phase-section-title">0.5X / 1X / 2X Phase Summary</div>'
        '<div class="wm-phase-subnote">Amplitude shown as peak-to-peak. Fit Amp = sinusoidal fit result. FFT Amp = local FFT validation. Confidence combines phase stability + agreement between methods.</div>'
        '<div class="wm-phase-table-wrap">'
        '<table class="wm-phase-table">'
        "<thead>"
        "<tr>"
        '<th rowspan="2">Signal</th>'
        '<th rowspan="2">Machine</th>'
        '<th rowspan="2">Point</th>'
        '<th rowspan="2">RPM</th>'

        '<th colspan="6">0.5X</th>'
        '<th colspan="6">1X</th>'
        '<th colspan="6">2X</th>'

        '<th rowspan="2">Timestamp</th>'
        "</tr>"
        "<tr>"
        "<th>Freq CPM</th><th>Fit Amp</th><th>FFT Amp</th><th>Phase</th><th>Stability</th><th>Confidence</th>"
        "<th>Freq CPM</th><th>Fit Amp</th><th>FFT Amp</th><th>Phase</th><th>Stability</th><th>Confidence</th>"
        "<th>Freq CPM</th><th>Fit Amp</th><th>FFT Amp</th><th>Phase</th><th>Stability</th><th>Confidence</th>"
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


def _badge_style_stability(value: Optional[float]) -> Tuple[str, str]:
    if value is None or not math.isfinite(value):
        return "#f1f5f9", "#475569"
    if value >= 85:
        return "#dcfce7", "#166534"
    if value >= 65:
        return "#fef3c7", "#92400e"
    return "#fee2e2", "#991b1b"


def _badge_style_confidence(value: Optional[float]) -> Tuple[str, str]:
    if value is None or not math.isfinite(value):
        return "#f1f5f9", "#475569"
    if value >= 85:
        return "#dbeafe", "#1d4ed8"
    if value >= 65:
        return "#fef3c7", "#92400e"
    return "#fee2e2", "#991b1b"


def build_png_report(df: pd.DataFrame, primary: SignalRecord) -> bytes:
    width = 6200
    row_h = 92
    top_h = 180
    title_h = 84
    table_header_h1 = 72
    table_header_h2 = 62
    n_rows = len(df)
    height = top_h + title_h + table_header_h1 + table_header_h2 + n_rows * row_h + 150

    bg = "#f3f4f6"
    white = "#ffffff"
    border = "#dbe5f0"
    blue = "#1d4ed8"
    text = "#111827"
    header_fill = "#f8fbff"
    subheader_fill = "#eef6ff"

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    font_title = _load_font(54, True)
    font_small = _load_font(28, False)
    font_header = _load_font(24, True)
    font_cell = _load_font(22, False)
    font_badge = _load_font(20, True)

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
        f"{primary.machine}   |   {primary.point}   |   {primary.variable} | Phase Dashboard | Amp = Peak-to-Peak   |   "
        f"RPM: {format_number(primary.rpm, 0)}   |   Signals: {len(df)}   |   {primary.timestamp or '—'}"
    )
    draw.text((logo_x + 150, top_y0 + 38), meta_text, font=font_small, fill=text)

    shell_y0 = top_y1 + 32
    shell_y1 = height - 50
    draw.rounded_rectangle((card_x0, shell_y0, card_x1, shell_y1), radius=28, fill=white, outline=border, width=2)

    draw.text((card_x0 + 24, shell_y0 + 20), "0.5X / 1X / 2X Phase Summary", font=font_title, fill=text)

    table_x0 = card_x0 + 24
    table_x1 = card_x1 - 24
    table_y0 = shell_y0 + 100

    col_widths = [
        420, 420, 340, 220,
        180, 220, 220, 180, 180, 180,
        180, 220, 220, 180, 180, 180,
        180, 220, 220, 180, 180, 180,
        420
    ]
    scale = (table_x1 - table_x0) / sum(col_widths)
    col_widths = [int(w * scale) for w in col_widths]

    col_x = [table_x0]
    for w in col_widths:
        col_x.append(col_x[-1] + w)

    y = table_y0
    draw.rounded_rectangle((table_x0, y, table_x1, y + table_header_h1 + table_header_h2 + n_rows * row_h), radius=22, fill=white, outline=border, width=2)

    group_spans = [
        ("Signal", 0, 1),
        ("Machine", 1, 2),
        ("Point", 2, 3),
        ("RPM", 3, 4),
        ("0.5X", 4, 10),
        ("1X", 10, 16),
        ("2X", 16, 22),
        ("Timestamp", 22, 23),
    ]

    for label, c0, c1 in group_spans:
        x0 = col_x[c0]
        x1 = col_x[c1]
        draw.rectangle((x0, y, x1, y + table_header_h1), fill=subheader_fill, outline=border, width=1)
        tw = draw.textbbox((0, 0), label.upper(), font=font_header)
        tx = x0 + (x1 - x0 - (tw[2] - tw[0])) / 2
        ty = y + (table_header_h1 - (tw[3] - tw[1])) / 2 - 2
        draw.text((tx, ty), label.upper(), font=font_header, fill=blue)

    y2 = y + table_header_h1
    sub_labels = ["Freq", "Fit Amp", "FFT Amp", "Phase", "Stability", "Conf"] * 3
    for i, label in enumerate(sub_labels, start=4):
        x0 = col_x[i]
        x1 = col_x[i + 1]
        draw.rectangle((x0, y2, x1, y2 + table_header_h2), fill=header_fill, outline=border, width=1)
        tw = draw.textbbox((0, 0), label, font=font_cell)
        tx = x0 + (x1 - x0 - (tw[2] - tw[0])) / 2
        ty = y2 + (table_header_h2 - (tw[3] - tw[1])) / 2 - 2
        draw.text((tx, ty), label, font=font_cell, fill="#334155")

    for idx in [0, 1, 2, 3, 22]:
        x0 = col_x[idx]
        x1 = col_x[idx + 1]
        draw.rectangle((x0, y2, x1, y2 + table_header_h2), fill=header_fill, outline=border, width=1)

    row_y = y2 + table_header_h2

    for r, (_, row) in enumerate(df.iterrows()):
        y0 = row_y + r * row_h
        y1 = y0 + row_h
        fill = "#ffffff" if r % 2 == 0 else "#fafcff"
        draw.rectangle((table_x0, y0, table_x1, y1), fill=fill, outline=border, width=1)

        unit = str(row["Unit"]).strip()
        unit_txt = f" {unit} p-p" if unit else " p-p"

        cells = [
            str(row["Signal"]),
            str(row["Machine"]),
            str(row["Point"]),
            format_number(row["RPM"], 0),

            format_number(row["0.5X Freq"], 1),
            f"{format_number(row['0.5X Fit Amp'], 3)}{unit_txt}",
            f"{format_number(row['0.5X FFT Amp'], 3)}{unit_txt}",
            f"{format_number(row['0.5X Phase'], 1)}°",
            None,
            None,

            format_number(row["1X Freq"], 1),
            f"{format_number(row['1X Fit Amp'], 3)}{unit_txt}",
            f"{format_number(row['1X FFT Amp'], 3)}{unit_txt}",
            f"{format_number(row['1X Phase'], 1)}°",
            None,
            None,

            format_number(row["2X Freq"], 1),
            f"{format_number(row['2X Fit Amp'], 3)}{unit_txt}",
            f"{format_number(row['2X FFT Amp'], 3)}{unit_txt}",
            f"{format_number(row['2X Phase'], 1)}°",
            None,
            None,

            str(row["Timestamp"] or "—"),
        ]

        for c, cell in enumerate(cells):
            x0 = col_x[c]
            x1 = col_x[c + 1]

            if c in [8, 14, 20]:
                stability_col = {8: "0.5X Stability", 14: "1X Stability", 20: "2X Stability"}[c]
                val = row[stability_col]
                bg_badge, fg_badge = _badge_style_stability(val)
                badge_text = "—" if val is None or not math.isfinite(val) else f"{val:.1f}%"
                badge_w = 140
                badge_h = 38
                bx0 = x0 + (x1 - x0 - badge_w) / 2
                by0 = y0 + (row_h - badge_h) / 2
                bx1 = bx0 + badge_w
                by1 = by0 + badge_h
                draw.rounded_rectangle((bx0, by0, bx1, by1), radius=20, fill=bg_badge)
                tw = draw.textbbox((0, 0), badge_text, font=font_badge)
                tx = bx0 + (badge_w - (tw[2] - tw[0])) / 2
                ty = by0 + (badge_h - (tw[3] - tw[1])) / 2 - 1
                draw.text((tx, ty), badge_text, font=font_badge, fill=fg_badge)

            elif c in [9, 15, 21]:
                conf_col = {9: "0.5X Confidence", 15: "1X Confidence", 21: "2X Confidence"}[c]
                val = row[conf_col]
                bg_badge, fg_badge = _badge_style_confidence(val)
                badge_text = "—" if val is None or not math.isfinite(val) else f"{val:.1f}%"
                badge_w = 140
                badge_h = 38
                bx0 = x0 + (x1 - x0 - badge_w) / 2
                by0 = y0 + (row_h - badge_h) / 2
                bx1 = bx0 + badge_w
                by1 = by0 + badge_h
                draw.rounded_rectangle((bx0, by0, bx1, by1), radius=20, fill=bg_badge)
                tw = draw.textbbox((0, 0), badge_text, font=font_badge)
                tx = bx0 + (badge_w - (tw[2] - tw[0])) / 2
                ty = by0 + (badge_h - (tw[3] - tw[1])) / 2 - 1
                draw.text((tx, ty), badge_text, font=font_badge, fill=fg_badge)

            else:
                pad_x = 12
                if c in [0, 1, 2, 22]:
                    draw.text((x0 + pad_x, y0 + 30), cell, font=font_cell, fill=text)
                else:
                    tw = draw.textbbox((0, 0), cell, font=font_cell)
                    tx = x0 + (x1 - x0 - (tw[2] - tw[0])) / 2
                    ty = y0 + (row_h - (tw[3] - tw[1])) / 2 - 1
                    draw.text((tx, ty), cell, font=font_cell, fill=text)

    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


if "wm_phase_primary_signal_id" not in st.session_state:
    st.session_state.wm_phase_primary_signal_id = None

if "wm_phase_export_png_bytes" not in st.session_state:
    st.session_state.wm_phase_export_png_bytes = None

if "wm_phase_export_error" not in st.session_state:
    st.session_state.wm_phase_export_error = None


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
        "Signal", "Machine", "Point", "RPM",
        "0.5X Freq", "0.5X Fit Amp", "0.5X FFT Amp", "0.5X Phase", "0.5X Stability", "0.5X Confidence",
        "1X Freq", "1X Fit Amp", "1X FFT Amp", "1X Phase", "1X Stability", "1X Confidence",
        "2X Freq", "2X Fit Amp", "2X FFT Amp", "2X Phase", "2X Stability", "2X Confidence",
        "Timestamp", "Unit", "Variable",
    ]
].copy()

st.markdown('<div class="wm-export-actions"></div>', unsafe_allow_html=True)

left_pad, col_export1, col_export2, right_pad = st.columns([2.4, 1.3, 1.3, 2.4])

with col_export1:
    if st.button("Prepare PNG HD", use_container_width=True):
        try:
            png_bytes = build_png_report(export_df, primary)
            st.session_state.wm_phase_export_png_bytes = png_bytes
            st.session_state.wm_phase_export_error = None
        except Exception as e:
            st.session_state.wm_phase_export_png_bytes = None
            st.session_state.wm_phase_export_error = str(e)

with col_export2:
    if st.session_state.wm_phase_export_png_bytes is not None:
        st.download_button(
            "Download PNG HD",
            data=st.session_state.wm_phase_export_png_bytes,
            file_name="watermelon_phase_hd.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.button("Download PNG HD", disabled=True, use_container_width=True)

if st.session_state.wm_phase_export_error:
    st.warning(f"PNG export error: {st.session_state.wm_phase_export_error}")