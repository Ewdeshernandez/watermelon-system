from __future__ import annotations

import base64
import html
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
from core.spectrum_diagnostics import evaluate_spectrum_diagnostic, build_spectrum_report_notes

st.set_page_config(page_title="Watermelon System | Spectrum", layout="wide")

require_login()
render_user_menu()

# ============================================================
# WATERMELON SYSTEM — SPECTRUM VIEWER
# Premium harmonic annotation line
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


@dataclass
class SpectrumResult:
    freq_cpm: np.ndarray
    amp_peak: np.ndarray
    fs_hz: float
    nfft: int
    resolution_cpm: float
    real_resolution_cpm: float
    peak_freq_cpm: Optional[float]
    peak_amp_peak: Optional[float]
    peak_bin_index: Optional[int]


@dataclass
class HarmonicPoint:
    order: int
    freq_cpm: float
    amp_peak: float


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


def convert_input_time_to_seconds(raw_time: np.ndarray) -> Tuple[np.ndarray, str]:
    if raw_time.size == 0:
        return raw_time, "s"

    raw = raw_time.astype(float, copy=False)
    raw = raw - raw[0]
    duration = float(raw[-1] - raw[0]) if raw.size > 1 else 0.0

    if duration > 5.0:
        return raw / 1000.0, "ms"
    return raw, "s"


def next_pow_2(n: int) -> int:
    return 1 if n <= 1 else 2 ** int(math.ceil(math.log2(n)))


def get_window(name: str, n: int) -> np.ndarray:
    if name == "Hanning":
        return np.hanning(n)
    if name == "Hamming":
        return np.hamming(n)
    if name == "Blackman":
        return np.blackman(n)
    return np.ones(n, dtype=float)


def coherent_gain(window: np.ndarray) -> float:
    if window.size == 0:
        return 1.0
    return float(np.mean(window))


def dominant_peak(freq_cpm: np.ndarray, amp: np.ndarray, min_cpm: float = 1.0) -> Tuple[Optional[float], Optional[float]]:
    if freq_cpm.size == 0 or amp.size == 0:
        return None, None
    mask = np.isfinite(freq_cpm) & np.isfinite(amp) & (freq_cpm >= min_cpm)
    if not np.any(mask):
        return None, None
    freq_sel = freq_cpm[mask]
    amp_sel = amp[mask]
    idx = int(np.argmax(amp_sel))
    return float(freq_sel[idx]), float(amp_sel[idx])


# ------------------------------------------------------------
# Amplitude conversions
# ------------------------------------------------------------
def convert_peak_to_mode(peak_amp: np.ndarray, mode: str) -> np.ndarray:
    if mode == "Peak":
        return peak_amp
    if mode == "Peak-to-Peak":
        return 2.0 * peak_amp
    if mode == "RMS":
        return peak_amp / np.sqrt(2.0)
    return peak_amp


def convert_scalar_peak_to_mode(value: Optional[float], mode: str) -> Optional[float]:
    if value is None:
        return None
    if mode == "Peak":
        return value
    if mode == "Peak-to-Peak":
        return 2.0 * value
    if mode == "RMS":
        return value / math.sqrt(2.0)
    return value


def convert_rms_to_mode(value_rms: Optional[float], mode: str) -> Optional[float]:
    if value_rms is None:
        return None
    if mode == "RMS":
        return value_rms
    if mode == "Peak":
        return value_rms * math.sqrt(2.0)
    if mode == "Peak-to-Peak":
        return value_rms * 2.0 * math.sqrt(2.0)
    return value_rms


def amplitude_mode_suffix(mode: str) -> str:
    if mode == "Peak":
        return "pk"
    if mode == "Peak-to-Peak":
        return "pp"
    if mode == "RMS":
        return "rms"
    return ""


def amplitude_mode_label(mode: str) -> str:
    if mode == "Peak":
        return "Peak"
    if mode == "Peak-to-Peak":
        return "Peak-to-Peak"
    if mode == "RMS":
        return "RMS"
    return mode


def amplitude_unit_text(base_unit: str, mode: str) -> str:
    suffix = amplitude_mode_suffix(mode)
    if base_unit and suffix:
        return f"{base_unit} {suffix}"
    if base_unit:
        return base_unit
    return suffix or ""


# ------------------------------------------------------------
# Load signals
# ------------------------------------------------------------
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

    if n < 2:
        return None

    finite_mask = np.isfinite(raw_time) & np.isfinite(amplitude)
    raw_time = raw_time[finite_mask]
    amplitude = amplitude[finite_mask]

    if raw_time.size < 2 or amplitude.size < 2:
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

    if time_s.size < 2 or amplitude.size < 2:
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


# ------------------------------------------------------------
# Spectrum engine
# ------------------------------------------------------------
def _prepare_signal_for_spectrum(
    time_s: np.ndarray,
    y: np.ndarray,
    remove_dc: bool,
    detrend: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if time_s.size < 2 or y.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float), float("nan")

    n = min(time_s.size, y.size)
    time_s = time_s[:n]
    y = y[:n].astype(float, copy=True)

    finite_mask = np.isfinite(time_s) & np.isfinite(y)
    time_s = time_s[finite_mask]
    y = y[finite_mask]

    if time_s.size < 2 or y.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float), float("nan")

    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if dt.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float), float("nan")

    fs = 1.0 / float(np.mean(dt))

    if detrend and y.size >= 2:
        idx = np.arange(y.size, dtype=float)
        coeffs = np.polyfit(idx, y, 1)
        y = y - np.polyval(coeffs, idx)

    if remove_dc:
        y = y - np.mean(y)

    return time_s, y, float(fs)


def parabolic_peak_interpolation(
    freq_cpm: np.ndarray,
    amp_peak: np.ndarray,
    idx: int,
) -> Tuple[float, float]:
    if idx <= 0 or idx >= (amp_peak.size - 1):
        return float(freq_cpm[idx]), float(amp_peak[idx])

    y1 = float(amp_peak[idx - 1])
    y2 = float(amp_peak[idx])
    y3 = float(amp_peak[idx + 1])

    denom = (y1 - 2.0 * y2 + y3)
    if abs(denom) < 1e-18:
        return float(freq_cpm[idx]), float(amp_peak[idx])

    delta = 0.5 * (y1 - y3) / denom
    delta = float(np.clip(delta, -1.0, 1.0))

    df = float(freq_cpm[idx + 1] - freq_cpm[idx])
    interp_freq = float(freq_cpm[idx] + delta * df)
    interp_amp = float(y2 - 0.25 * (y1 - y3) * delta)

    return interp_freq, interp_amp


def compute_spectrum_peak(
    time_s: np.ndarray,
    y: np.ndarray,
    window_name: str,
    remove_dc: bool,
    detrend: bool,
    zero_padding: bool,
    high_res_factor: int,
    min_peak_cpm: float = 1.0,
) -> SpectrumResult:
    time_s, y, fs = _prepare_signal_for_spectrum(
        time_s=time_s,
        y=y,
        remove_dc=remove_dc,
        detrend=detrend,
    )

    if time_s.size < 2 or y.size < 2 or not math.isfinite(fs):
        return SpectrumResult(
            freq_cpm=np.array([], dtype=float),
            amp_peak=np.array([], dtype=float),
            fs_hz=float("nan"),
            nfft=0,
            resolution_cpm=float("nan"),
            real_resolution_cpm=float("nan"),
            peak_freq_cpm=None,
            peak_amp_peak=None,
            peak_bin_index=None,
        )

    window = get_window(window_name, y.size)
    gain = coherent_gain(window)
    y_win = y * window

    base_nfft = next_pow_2(y.size) if zero_padding else y.size
    factor = max(1, int(high_res_factor))
    nfft = int(base_nfft * factor)

    fft_vals = np.fft.rfft(y_win, n=nfft)
    freq_hz = np.fft.rfftfreq(nfft, d=1.0 / fs)

    peak_amp = (2.0 / y.size) * np.abs(fft_vals)
    peak_amp = peak_amp / max(gain, 1e-12)

    if peak_amp.size > 0:
        peak_amp[0] *= 0.5
    if nfft % 2 == 0 and peak_amp.size > 1:
        peak_amp[-1] *= 0.5

    freq_cpm = freq_hz * 60.0

    resolution_cpm = float((fs / nfft) * 60.0) if nfft > 0 else float("nan")
    real_nfft = int(y.size)
    real_resolution_cpm = float((fs / real_nfft) * 60.0) if real_nfft > 0 else float("nan")

    peak_freq_cpm = None
    peak_amp_peak = None
    peak_bin_index = None

    valid_mask = np.isfinite(freq_cpm) & np.isfinite(peak_amp) & (freq_cpm >= min_peak_cpm)
    if np.any(valid_mask):
        idx_candidates = np.where(valid_mask)[0]
        local_max_index = int(idx_candidates[np.argmax(peak_amp[valid_mask])])
        peak_bin_index = local_max_index
        peak_freq_cpm, peak_amp_peak = parabolic_peak_interpolation(freq_cpm, peak_amp, local_max_index)

    return SpectrumResult(
        freq_cpm=freq_cpm.astype(float),
        amp_peak=peak_amp.astype(float),
        fs_hz=float(fs),
        nfft=int(nfft),
        resolution_cpm=resolution_cpm,
        real_resolution_cpm=real_resolution_cpm,
        peak_freq_cpm=peak_freq_cpm,
        peak_amp_peak=peak_amp_peak,
        peak_bin_index=peak_bin_index,
    )


def compute_spectrum_overall_rms_parseval(
    time_s: np.ndarray,
    y: np.ndarray,
    remove_dc: bool,
    detrend: bool,
    max_cpm: Optional[float] = None,
) -> float:
    time_s, y, fs = _prepare_signal_for_spectrum(
        time_s=time_s,
        y=y,
        remove_dc=remove_dc,
        detrend=detrend,
    )

    if time_s.size < 2 or y.size < 2 or not math.isfinite(fs):
        return float("nan")

    n = int(y.size)
    fft_vals = np.fft.rfft(y, n=n)
    freq_hz = np.fft.rfftfreq(n, d=1.0 / fs)

    mag2 = np.abs(fft_vals) ** 2
    weights = np.ones_like(mag2)

    if mag2.size > 2:
        weights[1:-1] = 2.0

    if n % 2 == 0 and mag2.size > 1:
        weights[-1] = 1.0

    if max_cpm is not None and math.isfinite(max_cpm) and max_cpm > 0:
        max_hz = max_cpm / 60.0
        band_mask = freq_hz <= max_hz
        mag2 = mag2[band_mask]
        weights = weights[band_mask]

    if mag2.size == 0:
        return float("nan")

    mean_square = float(np.sum(weights * mag2) / (n * n))
    mean_square = max(mean_square, 0.0)
    return float(np.sqrt(mean_square))


# ------------------------------------------------------------
# Harmonic estimators
# ------------------------------------------------------------
def estimate_harmonic_from_waveform_peak(
    time_s: np.ndarray,
    y: np.ndarray,
    freq_hz: float,
    remove_mean: bool = True,
) -> Optional[float]:
    if time_s.size < 3 or y.size < 3 or freq_hz <= 0:
        return None

    n = min(time_s.size, y.size)
    t = time_s[:n]
    x = y[:n].astype(float, copy=True)

    finite_mask = np.isfinite(t) & np.isfinite(x)
    t = t[finite_mask]
    x = x[finite_mask]

    if t.size < 3:
        return None

    if remove_mean:
        x = x - np.mean(x)

    omega = 2.0 * np.pi * freq_hz
    c = np.cos(omega * t)
    s = np.sin(omega * t)

    design = np.column_stack([c, s])
    coeffs, *_ = np.linalg.lstsq(design, x, rcond=None)
    a, b = coeffs
    amp_peak = float(np.sqrt(a * a + b * b))
    return amp_peak if math.isfinite(amp_peak) else None


def find_local_peak_near_1x(
    freq_cpm: np.ndarray,
    amp_peak: np.ndarray,
    one_x_cpm: float,
    band_fraction: float = 0.20,
) -> Tuple[Optional[float], Optional[float]]:
    if freq_cpm.size == 0 or amp_peak.size == 0 or one_x_cpm <= 0:
        return None, None

    fmin = one_x_cpm * (1.0 - band_fraction)
    fmax = one_x_cpm * (1.0 + band_fraction)

    mask = np.isfinite(freq_cpm) & np.isfinite(amp_peak) & (freq_cpm >= fmin) & (freq_cpm <= fmax)
    if not np.any(mask):
        return None, None

    idx_candidates = np.where(mask)[0]
    idx = int(idx_candidates[np.argmax(amp_peak[mask])])
    return parabolic_peak_interpolation(freq_cpm, amp_peak, idx)


def collect_harmonic_points(
    freq_cpm: np.ndarray,
    amp_peak: np.ndarray,
    base_rpm: Optional[float],
    harmonic_count: int,
    band_fraction: float,
    max_cpm: float,
) -> List[HarmonicPoint]:
    if base_rpm is None or base_rpm <= 0:
        return []

    points: List[HarmonicPoint] = []
    one_x_cpm = float(base_rpm)

    for order in range(1, harmonic_count + 1):
        target_cpm = one_x_cpm * order
        if target_cpm > max_cpm:
            break

        fmin = target_cpm * (1.0 - band_fraction)
        fmax = target_cpm * (1.0 + band_fraction)

        mask = np.isfinite(freq_cpm) & np.isfinite(amp_peak) & (freq_cpm >= fmin) & (freq_cpm <= fmax)
        if not np.any(mask):
            continue

        idx_candidates = np.where(mask)[0]
        idx = int(idx_candidates[np.argmax(amp_peak[mask])])
        interp_freq, interp_amp = parabolic_peak_interpolation(freq_cpm, amp_peak, idx)

        points.append(
            HarmonicPoint(
                order=order,
                freq_cpm=float(interp_freq),
                amp_peak=float(interp_amp),
            )
        )

    return points


def choose_harmonics_to_annotate(
    harmonic_points: List[HarmonicPoint],
    label_mode: str,
) -> List[HarmonicPoint]:
    if not harmonic_points:
        return []

    if label_mode == "All visible":
        return harmonic_points

    if label_mode == "1X only":
        return [p for p in harmonic_points if p.order == 1]

    if label_mode == "Top 3 amplitudes":
        if len(harmonic_points) <= 3:
            return harmonic_points
        ranked = sorted(harmonic_points, key=lambda p: p.amp_peak, reverse=True)[:3]
        return sorted(ranked, key=lambda p: p.order)

    if label_mode == "1X + Top 3":
        one_x = [p for p in harmonic_points if p.order == 1]
        others = [p for p in harmonic_points if p.order != 1]
        top_others = sorted(others, key=lambda p: p.amp_peak, reverse=True)[:3]
        out = one_x + top_others
        seen = set()
        unique: List[HarmonicPoint] = []
        for p in sorted(out, key=lambda q: q.order):
            if p.order not in seen:
                unique.append(p)
                seen.add(p.order)
        return unique

    return harmonic_points


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------
def _draw_top_strip(
    fig: go.Figure,
    record: SignalRecord,
    peak_amp_text: str,
    logo_uri: Optional[str],
    spectrum_mode_label: str,
    amplitude_mode_text: str,
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
        text=f"{record.variable} | {spectrum_mode_label} | {amplitude_mode_text}",
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
        text=f"Peak Amp: <b>{peak_amp_text}</b>",
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
        fillcolor="rgba(255,255,255,0.68)",
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


def add_harmonic_annotations(
    fig: go.Figure,
    harmonic_points: List[HarmonicPoint],
    amplitude_mode: str,
    base_unit: str,
    y_top: float,
    show_harmonic_amplitudes: bool,
) -> None:
    if not harmonic_points:
        return

    display_unit_text = amplitude_unit_text(base_unit, amplitude_mode)
    y_offset = max(0.02 * y_top, 0.05)

    for point in harmonic_points:
        amp_display = convert_scalar_peak_to_mode(point.amp_peak, amplitude_mode)
        line_color = "#374151" if point.order == 1 else "rgba(107, 114, 128, 0.55)"

        fig.add_vline(
            x=point.freq_cpm,
            line_width=1.15,
            line_dash="dash",
            line_color=line_color,
        )

        if show_harmonic_amplitudes and amp_display is not None:
            label_y = min(amp_display + y_offset, y_top * 0.965 if y_top > 0 else amp_display + y_offset)
            label_text = f"{point.order}X · {format_number(amp_display, 3)} {display_unit_text}".strip()
            fig.add_annotation(
                x=point.freq_cpm,
                y=label_y,
                text=label_text,
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#d1d5db",
                borderwidth=1,
                borderpad=4,
                font=dict(size=10.5, color="#111827"),
            )
        else:
            fig.add_annotation(
                x=point.freq_cpm,
                y=y_top * 0.965 if y_top > 0 else 1.0,
                text=f"{point.order}X",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=10.8, color="#111827"),
            )


def build_spectrum_figure(
    record: SignalRecord,
    freq_cpm: np.ndarray,
    amp_display: np.ndarray,
    amp_peak: np.ndarray,
    amplitude_mode: str,
    max_cpm: float,
    y_axis_mode: str,
    y_axis_manual_max: Optional[float],
    show_harmonics: bool,
    show_harmonic_amplitudes: bool,
    harmonic_points_for_labels: List[HarmonicPoint],
    show_right_info_box: bool,
    fill_area: bool,
    annotate_peak: bool,
    logo_uri: Optional[str],
    spectrum_mode_label: str,
    one_x_display_amp: Optional[float],
    one_x_display_freq_cpm: Optional[float],
    overall_spec_rms: Optional[float],
    resolution_cpm: Optional[float],
    real_resolution_cpm: Optional[float],
    interpolated_peak_freq_cpm: Optional[float],
    interpolated_peak_amp_display: Optional[float],
) -> go.Figure:
    fig = go.Figure()

    display_unit_text = amplitude_unit_text(record.amplitude_unit, amplitude_mode)
    y_title = f"Amplitude ({display_unit_text})" if display_unit_text else "Amplitude"

    mask = np.isfinite(freq_cpm) & np.isfinite(amp_display) & np.isfinite(amp_peak)
    freq_cpm = freq_cpm[mask]
    amp_display = amp_display[mask]
    amp_peak = amp_peak[mask]

    if max_cpm > 0:
        visible_mask = freq_cpm <= max_cpm
        freq_cpm = freq_cpm[visible_mask]
        amp_display = amp_display[visible_mask]
        amp_peak = amp_peak[visible_mask]

    if freq_cpm.size < 2 or amp_display.size < 2:
        fig.update_layout(
            height=940,
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#f3f4f6",
            margin=dict(l=46, r=18, t=84, b=240),
            xaxis_title="Frequency (CPM)",
            yaxis_title=y_title,
        )
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No valid spectrum data available",
            showarrow=False,
            font=dict(size=18, color="#6b7280"),
        )
        return fig

    hover_template = (
        "Frequency: %{x:.2f} CPM<br>"
        + (f"Amplitude: " + "%{y:.4f} " + display_unit_text if display_unit_text else "Amplitude: %{y:.4f}")
        + "<extra></extra>"
    )

    fig.add_trace(
        go.Scattergl(
            x=freq_cpm,
            y=amp_display,
            mode="lines",
            line=dict(width=1.8, color="#5b9cf0"),
            fill="tozeroy" if fill_area else None,
            fillcolor="rgba(91, 156, 240, 0.10)" if fill_area else None,
            hovertemplate=hover_template,
            showlegend=False,
            connectgaps=False,
            name="spectrum_main",
        )
    )

    peak_freq_global, peak_amp_global_display = dominant_peak(freq_cpm, amp_display, min_cpm=1.0)
    peak_freq_to_use = interpolated_peak_freq_cpm if interpolated_peak_freq_cpm is not None else peak_freq_global
    peak_amp_to_use = interpolated_peak_amp_display if interpolated_peak_amp_display is not None else peak_amp_global_display

    if annotate_peak and peak_freq_to_use is not None and peak_amp_to_use is not None:
        fig.add_trace(
            go.Scatter(
                x=[peak_freq_to_use],
                y=[peak_amp_to_use],
                mode="markers",
                marker=dict(symbol="circle", size=8, color="#2f80ed"),
                hovertemplate=hover_template,
                showlegend=False,
                name="peak_marker",
            )
        )
        fig.add_annotation(
            x=peak_freq_to_use,
            y=peak_amp_to_use,
            text=f"Peak {format_number(peak_freq_to_use, 1)} CPM",
            showarrow=True,
            arrowhead=2,
            ax=34,
            ay=-36,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#d1d5db",
            borderwidth=1,
            font=dict(size=11, color="#111827"),
        )

    x_min = 0.0
    x_max = float(np.max(freq_cpm))
    if x_max <= 0:
        x_max = max_cpm if max_cpm > 0 else 1.0

    y_data_max = float(np.max(amp_display))
    auto_top = max(y_data_max * 1.12, y_data_max + 0.05 if y_data_max > 0 else 1.0)

    if y_axis_mode == "Manual" and y_axis_manual_max is not None and y_axis_manual_max > 0:
        y_top = float(y_axis_manual_max)
    else:
        y_top = float(auto_top)

    if show_harmonics and record.rpm is not None and record.rpm > 0:
        add_harmonic_annotations(
            fig=fig,
            harmonic_points=harmonic_points_for_labels,
            amplitude_mode=amplitude_mode,
            base_unit=record.amplitude_unit,
            y_top=y_top,
            show_harmonic_amplitudes=show_harmonic_amplitudes,
        )

    peak_amp_text = (
        f"{format_number(one_x_display_amp, 3)} {display_unit_text}".strip()
        if one_x_display_amp is not None
        else f"{format_number(peak_amp_to_use, 3)} {display_unit_text}".strip()
    )

    overall_spec_display = convert_rms_to_mode(overall_spec_rms, amplitude_mode)

    _draw_top_strip(
        fig=fig,
        record=record,
        peak_amp_text=peak_amp_text,
        logo_uri=logo_uri,
        spectrum_mode_label=spectrum_mode_label,
        amplitude_mode_text=amplitude_mode_label(amplitude_mode),
    )

    if show_right_info_box:
        rows = [
            (
                "1X Frequency",
                f"{format_number(one_x_display_freq_cpm, 1)} CPM" if one_x_display_freq_cpm is not None else "—",
            ),
            (
                f"1X Amplitude ({amplitude_mode_label(amplitude_mode)})",
                f"{format_number(one_x_display_amp, 3)} {display_unit_text}".strip() if one_x_display_amp is not None else "—",
            ),
            (
                f"Spectrum O/All ({amplitude_mode_label(amplitude_mode)})",
                f"{format_number(overall_spec_display, 3)} {display_unit_text}".strip()
                if overall_spec_display is not None
                else "—",
            ),
            (
                "Real Resolution",
                f"{format_number(real_resolution_cpm, 2)} CPM" if real_resolution_cpm is not None else "—",
            ),
            (
                "Display Grid",
                f"{format_number(resolution_cpm, 2)} CPM" if resolution_cpm is not None else "—",
            ),
        ]
        _draw_right_info_box(fig, rows)

    grid_step = 1000.0
    if x_max > 5000:
        grid_step = 5000.0
    if x_max > 20000:
        grid_step = 10000.0
    if x_max > 60000:
        grid_step = 20000.0

    tickvals = list(np.arange(0.0, x_max + grid_step * 0.5, grid_step))
    for gx in tickvals:
        if abs(float(gx)) < 1e-12:
            continue
        fig.add_vline(
            x=gx,
            line_width=1,
            line_color="rgba(148, 163, 184, 0.18)",
            layer="below",
        )

    fig.update_layout(
        height=940,
        margin=dict(l=46, r=18, t=84, b=240),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f3f4f6",
        font=dict(color="#111827"),
        xaxis=dict(
            title="Frequency (CPM)",
            range=[x_min, x_max],
            tickvals=tickvals,
            tickformat=".0f",
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
            range=[0.0, y_top],
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
                    fill=trace_json.get("fill"),
                    fillcolor=trace_json.get("fillcolor"),
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
if "wm_sp_selected_signal_ids" not in st.session_state:
    st.session_state.wm_sp_selected_signal_ids = []
if "wm_sp_export_store" not in st.session_state:
    st.session_state.wm_sp_export_store = {}
if "report_items" not in st.session_state:
    st.session_state.report_items = []


# ------------------------------------------------------------
# Load signals
# ------------------------------------------------------------
records_all = load_signals_from_session()

if not records_all:
    st.warning("No se pudieron cargar señales válidas desde `st.session_state['signals']`.")
    st.stop()


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### Signal Selection")

    signal_name_map = {r.name: r.signal_id for r in records_all}
    signal_names = list(signal_name_map.keys())

    valid_ids = {r.signal_id for r in records_all}
    current_ids = [sid for sid in st.session_state.wm_sp_selected_signal_ids if sid in valid_ids]
    if not current_ids:
        current_ids = [records_all[0].signal_id]
        st.session_state.wm_sp_selected_signal_ids = current_ids

    default_names = [r.name for r in records_all if r.signal_id in current_ids]

    selected_names = st.multiselect(
        "Spectra to display",
        options=signal_names,
        default=default_names,
    )
    st.session_state.wm_sp_selected_signal_ids = [
        signal_name_map[name] for name in selected_names if name in signal_name_map
    ]

    st.markdown("### Spectrum Processing")

    window_name = st.selectbox(
        "Window",
        ["Hanning", "Hamming", "Blackman", "Rectangular"],
        index=0,
    )

    amplitude_mode = st.selectbox(
        "Amplitude mode",
        ["Peak", "Peak-to-Peak", "RMS"],
        index=1,
    )

    remove_dc = st.checkbox("Remove DC", value=True)
    detrend = st.checkbox("Detrend", value=True)
    zero_padding = st.checkbox("Zero padding", value=True)

    high_res_display = st.checkbox("High resolution display", value=True)
    high_res_factor = int(
        st.selectbox(
            "Display interpolation factor",
            options=[1, 2, 4, 8, 16],
            index=3,
            disabled=not high_res_display,
        )
    )
    if not high_res_display:
        high_res_factor = 1

    st.markdown("### Display")

    default_source_id = (
        st.session_state.wm_sp_selected_signal_ids[0]
        if st.session_state.wm_sp_selected_signal_ids
        else records_all[0].signal_id
    )
    primary_for_defaults = next(r for r in records_all if r.signal_id == default_source_id)
    default_max_cpm = float(primary_for_defaults.rpm * 10) if primary_for_defaults.rpm is not None else 60000.0

    max_cpm = st.number_input(
        "Max frequency (CPM)",
        min_value=100.0,
        value=float(max(1000.0, default_max_cpm)),
        step=100.0,
        format="%.0f",
    )

    y_axis_mode = st.selectbox(
        "Y-axis scale",
        ["Auto", "Manual"],
        index=0,
    )

    y_axis_manual_max: Optional[float] = None
    if y_axis_mode == "Manual":
        y_axis_manual_max = float(
            st.number_input(
                "Manual Y max",
                min_value=0.001,
                value=3.0,
                step=0.1,
                format="%.3f",
            )
        )

    fill_area = st.checkbox("Fill area", value=True)
    annotate_peak = st.checkbox("Annotate dominant peak", value=True)
    show_right_info_box = st.checkbox("Show info box", value=True)

    st.markdown("### Harmonics")

    show_harmonics = st.checkbox("Show 1X harmonics", value=True)
    harmonic_count = int(
        st.number_input(
            "Harmonic count",
            min_value=1,
            max_value=30,
            value=8,
            step=1,
            disabled=not show_harmonics,
        )
    )

    harmonic_band_fraction = 0.12

    show_harmonic_amplitudes = st.checkbox(
        "Show harmonic amplitudes",
        value=True,
        disabled=not show_harmonics,
    )

    harmonic_label_mode = st.selectbox(
        "Harmonic label density",
        options=["1X only", "1X + Top 3", "Top 3 amplitudes", "All visible"],
        index=1,
        disabled=not (show_harmonics and show_harmonic_amplitudes),
    )


# ------------------------------------------------------------
# Prepare signals + multi-panel render
# ------------------------------------------------------------
def queue_spectrum_to_report(primary: SignalRecord, fig: go.Figure, panel_title: str, text_diag: Dict[str, str]) -> None:
    st.session_state.report_items.append(
        {
            "id": make_export_state_key(
                [
                    "report-spectrum",
                    primary.signal_id,
                    primary.timestamp,
                    panel_title,
                    len(st.session_state.report_items),
                ]
            ),
            "type": "spectrum",
            "title": panel_title,
            "notes": build_spectrum_report_notes(text_diag),
            "signal_id": primary.signal_id,
            "figure": go.Figure(fig),
            "image_bytes": image_bytes,
            "machine": primary.machine,
            "point": primary.point,
            "variable": primary.variable,
            "timestamp": primary.timestamp,
        }
    )


def render_spectrum_panel(
    primary: SignalRecord,
    panel_index: int,
    *,
    window_name: str,
    amplitude_mode: str,
    remove_dc: bool,
    detrend: bool,
    zero_padding: bool,
    high_res_display: bool,
    high_res_factor: int,
    max_cpm: float,
    y_axis_mode: str,
    y_axis_manual_max: Optional[float],
    fill_area: bool,
    annotate_peak: bool,
    show_harmonics: bool,
    harmonic_count: int,
    harmonic_band_fraction: float,
    show_harmonic_amplitudes: bool,
    harmonic_label_mode: str,
    show_right_info_box: bool,
) -> None:
    spectrum = compute_spectrum_peak(
        time_s=primary.time_s,
        y=primary.amplitude,
        window_name=window_name,
        remove_dc=remove_dc,
        detrend=detrend,
        zero_padding=zero_padding,
        high_res_factor=high_res_factor,
        min_peak_cpm=1.0,
    )

    freq_cpm = spectrum.freq_cpm
    amp_peak = spectrum.amp_peak
    resolution_cpm = spectrum.resolution_cpm
    real_resolution_cpm = spectrum.real_resolution_cpm

    amp_display = convert_peak_to_mode(amp_peak, amplitude_mode)
    interpolated_peak_amp_display = convert_scalar_peak_to_mode(spectrum.peak_amp_peak, amplitude_mode)

    one_x_display_amp: Optional[float] = None
    one_x_display_freq_cpm: Optional[float] = None

    if primary.rpm is not None and primary.rpm > 0:
        one_x_freq_cpm = float(primary.rpm)
        one_x_freq_hz = one_x_freq_cpm / 60.0

        one_x_peak_amp_from_waveform = estimate_harmonic_from_waveform_peak(
            time_s=primary.time_s,
            y=primary.amplitude,
            freq_hz=one_x_freq_hz,
            remove_mean=True,
        )

        one_x_local_freq_cpm, one_x_peak_amp_from_spectrum = find_local_peak_near_1x(
            freq_cpm=freq_cpm,
            amp_peak=amp_peak,
            one_x_cpm=one_x_freq_cpm,
            band_fraction=harmonic_band_fraction,
        )

        if one_x_peak_amp_from_waveform is not None:
            one_x_display_amp = convert_scalar_peak_to_mode(one_x_peak_amp_from_waveform, amplitude_mode)
            one_x_display_freq_cpm = one_x_freq_cpm
        elif one_x_peak_amp_from_spectrum is not None:
            one_x_display_amp = convert_scalar_peak_to_mode(one_x_peak_amp_from_spectrum, amplitude_mode)
            one_x_display_freq_cpm = one_x_local_freq_cpm

    all_harmonic_points = collect_harmonic_points(
        freq_cpm=freq_cpm,
        amp_peak=amp_peak,
        base_rpm=primary.rpm,
        harmonic_count=harmonic_count,
        band_fraction=harmonic_band_fraction,
        max_cpm=max_cpm,
    )

    harmonic_points_for_labels = choose_harmonics_to_annotate(
        harmonic_points=all_harmonic_points,
        label_mode=harmonic_label_mode,
    )

    overall_spec_rms = compute_spectrum_overall_rms_parseval(
        time_s=primary.time_s,
        y=primary.amplitude,
        remove_dc=remove_dc,
        detrend=detrend,
        max_cpm=max_cpm,
    )

    text_diag = evaluate_spectrum_diagnostic(
        one_x_amp=one_x_display_amp,
        harmonics=[{"order": p.order, "freq_cpm": p.freq_cpm, "amp_peak": p.amp_peak} for p in all_harmonic_points],
        overall_spec_rms=overall_spec_rms,
        dominant_peak_freq_cpm=spectrum.peak_freq_cpm,
        dominant_peak_amp=spectrum.peak_amp_peak,
        rpm=primary.rpm,
    )
    semaforo_status = text_diag["status"]
    semaforo_color = text_diag["color"]

    logo_uri = get_logo_data_uri(LOGO_PATH)

    fig = build_spectrum_figure(
        record=primary,
        freq_cpm=freq_cpm,
        amp_display=amp_display,
        amp_peak=amp_peak,
        amplitude_mode=amplitude_mode,
        max_cpm=max_cpm,
        y_axis_mode=y_axis_mode,
        y_axis_manual_max=y_axis_manual_max,
        show_harmonics=show_harmonics,
        show_harmonic_amplitudes=show_harmonic_amplitudes,
        harmonic_points_for_labels=harmonic_points_for_labels if show_harmonics else [],
        show_right_info_box=show_right_info_box,
        fill_area=fill_area,
        annotate_peak=annotate_peak,
        logo_uri=logo_uri,
        spectrum_mode_label=window_name,
        one_x_display_amp=one_x_display_amp,
        one_x_display_freq_cpm=one_x_display_freq_cpm,
        overall_spec_rms=overall_spec_rms,
        resolution_cpm=resolution_cpm,
        real_resolution_cpm=real_resolution_cpm,
        interpolated_peak_freq_cpm=spectrum.peak_freq_cpm,
        interpolated_peak_amp_display=interpolated_peak_amp_display,
    )

    export_state_key = make_export_state_key(
        [
            primary.signal_id,
            primary.name,
            primary.machine,
            primary.point,
            primary.variable,
            primary.timestamp,
            panel_index,
            window_name,
            amplitude_mode,
            remove_dc,
            detrend,
            zero_padding,
            high_res_display,
            high_res_factor,
            max_cpm,
            y_axis_mode,
            y_axis_manual_max,
            fill_area,
            annotate_peak,
            show_harmonics,
            harmonic_count,
            harmonic_band_fraction,
            show_harmonic_amplitudes,
            harmonic_label_mode,
            show_right_info_box,
            primary.rpm,
            float(np.nanmax(amp_display)) if amp_display.size else 0.0,
            float(np.nanmin(amp_display)) if amp_display.size else 0.0,
            amp_display.size,
            overall_spec_rms,
            resolution_cpm,
            real_resolution_cpm,
            spectrum.peak_freq_cpm,
            spectrum.peak_amp_peak,
            len(all_harmonic_points),
        ]
    )

    if export_state_key not in st.session_state.wm_sp_export_store:
        st.session_state.wm_sp_export_store[export_state_key] = {
            "png_bytes": None,
            "error": None,
        }

    panel_title = f"Spectrum {panel_index + 1} — {primary.name}"
    st.markdown(f"### {panel_title}")

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False},
        key=f"wm_spectrum_plot_{export_state_key}",
    )

    helper_title = f"Spectrum Diagnostic Helper · Panel {panel_index + 1}"
    helper_subtitle = text_diag["headline"]

    st.markdown("")

    helper_cols = [
        (f"Semáforo: {semaforo_status}", semaforo_color),
        (f"1X Amp: {format_number(one_x_display_amp, 3)} {amplitude_unit_text(primary.amplitude_unit, amplitude_mode)}".strip(), None),
        (f"Overall: {format_number(overall_spec_rms, 3)} {amplitude_unit_text(primary.amplitude_unit, amplitude_mode)}".strip(), None),
        (f"Peak Freq: {format_number(spectrum.peak_freq_cpm, 1)} CPM", None),
        (f"Harmonics: {len(all_harmonic_points)}", None),
    ]

    from core.module_patterns import helper_card
    helper_card(
        title=helper_title,
        subtitle=helper_subtitle,
        chips=helper_cols,
    )

    st.info(text_diag["narrative"])

    st.markdown('<div class="wm-export-actions"></div>', unsafe_allow_html=True)
    left_pad, col_export1, col_export2, col_report, right_pad = st.columns([2.0, 1.2, 1.2, 1.2, 2.0])

    with col_export1:
        if st.button("Prepare PNG HD", key=f"prepare_png_{export_state_key}", use_container_width=True):
            with st.spinner("Generating HD export..."):
                png_bytes, export_error = build_export_png_bytes(fig=fig)
                st.session_state.wm_sp_export_store[export_state_key]["png_bytes"] = png_bytes
                st.session_state.wm_sp_export_store[export_state_key]["error"] = export_error

    with col_export2:
        png_bytes = st.session_state.wm_sp_export_store[export_state_key]["png_bytes"]
        if png_bytes is not None:
            st.download_button(
                "Download PNG HD",
                data=png_bytes,
                file_name=f"watermelon_spectrum_{panel_index + 1}_hd.png",
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
            queue_spectrum_to_report(primary, fig, panel_title, text_diag)
            st.success("Spectrum enviado al reporte")


selected_ids = [
    signal_id
    for signal_id in st.session_state.wm_sp_selected_signal_ids
    if signal_id in {r.signal_id for r in records_all}
]

if not selected_ids:
    st.info("Selecciona uno o más espectros en la barra lateral.")
    st.stop()

selected_records = [
    next(r for r in records_all if r.signal_id == signal_id)
    for signal_id in selected_ids
]

for panel_index, primary in enumerate(selected_records):
    render_spectrum_panel(
        primary=primary,
        panel_index=panel_index,
        window_name=window_name,
        amplitude_mode=amplitude_mode,
        remove_dc=remove_dc,
        detrend=detrend,
        zero_padding=zero_padding,
        high_res_display=high_res_display,
        high_res_factor=high_res_factor,
        max_cpm=max_cpm,
        y_axis_mode=y_axis_mode,
        y_axis_manual_max=y_axis_manual_max,
        fill_area=fill_area,
        annotate_peak=annotate_peak,
        show_harmonics=show_harmonics,
        harmonic_count=harmonic_count,
        harmonic_band_fraction=harmonic_band_fraction,
        show_harmonic_amplitudes=show_harmonic_amplitudes,
        harmonic_label_mode=harmonic_label_mode,
        show_right_info_box=show_right_info_box,
    )

    if panel_index < len(selected_records) - 1:
        st.markdown("---")
