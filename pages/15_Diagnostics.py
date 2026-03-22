import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.auth import require_login, render_user_menu


# =========================================================
# PAGE CONFIG / AUTH
# =========================================================
st.set_page_config(page_title="Watermelon System | Diagnostic", layout="wide")
require_login()
render_user_menu()


# =========================================================
# STYLES
# =========================================================
st.markdown(
    """
    <style>
    .wm-hero {
        background: linear-gradient(135deg, #061326 0%, #0d2a4a 42%, #163f78 100%);
        padding: 26px 28px;
        border-radius: 24px;
        margin-bottom: 18px;
    }
    .wm-hero h1 {
        color: white;
        margin: 0;
        font-size: 58px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .wm-hero p {
        color: rgba(255,255,255,0.82);
        margin-top: 14px;
        margin-bottom: 0;
        font-size: 16px;
    }
    .wm-kpi {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid rgba(30,167,255,0.16);
        border-radius: 22px;
        padding: 18px 18px 16px 18px;
        min-height: 124px;
    }
    .wm-kpi-title {
        color: #5b6f87;
        font-size: 13px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }
    .wm-kpi-value {
        color: #081326;
        font-size: 40px;
        font-weight: 800;
        margin-top: 10px;
        line-height: 1;
    }
    .wm-kpi-sub {
        color: #687c93;
        font-size: 13px;
        margin-top: 12px;
    }
    .wm-section-card {
        background: white;
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 22px;
        padding: 18px;
        margin-bottom: 18px;
    }
    .wm-section-title {
        color: #081326;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .wm-chip {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 800;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid rgba(0,0,0,0.05);
    }
    .wm-chip-normal { background: #e8f8ee; color: #0d7a3a; }
    .wm-chip-observe { background: #fff8e6; color: #9a6a00; }
    .wm-chip-alert { background: #fff0ea; color: #b44700; }
    .wm-chip-danger { background: #ffe9eb; color: #b42318; }
    .wm-chip-info { background: #eef6ff; color: #185ea9; }

    .wm-verdict {
        background: white;
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 12px;
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }
    .wm-verdict-title {
        font-size: 18px;
        font-weight: 800;
        color: #081326;
        margin-bottom: 8px;
    }
    .wm-verdict-line {
        color: #42566e;
        font-size: 14px;
        margin-bottom: 6px;
    }
    .wm-verdict-left-normal { border-left: 6px solid #18a957; }
    .wm-verdict-left-observe { border-left: 6px solid #d7a600; }
    .wm-verdict-left-alert { border-left: 6px solid #e26f00; }
    .wm-verdict-left-danger { border-left: 6px solid #d92d20; }

    .wm-small-muted {
        color: #6f8298;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="wm-hero">
        <h1>Diagnostic</h1>
        <p>
            Cross-module rule-based diagnostic engine. Prioritizes Phase Analysis data first, then uses hardened fallback logic.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SAFE ACCESS
# =========================================================
def safe_get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_dict_if_possible(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            return {}
    return {}


def to_1d_float_array(arr: Any) -> np.ndarray:
    if arr is None:
        return np.array([], dtype=float)
    try:
        out = np.asarray(arr, dtype=float)
        if out.ndim != 1:
            out = out.flatten()
        return out
    except Exception:
        return np.array([], dtype=float)


def get_signal_bank() -> Dict[str, Any]:
    signals = st.session_state.get("signals", {})
    if isinstance(signals, dict):
        return signals
    if isinstance(signals, list):
        return {f"signal_{i}": s for i, s in enumerate(signals)}
    return {}


# =========================================================
# METADATA / SIGNAL EXTRACTION
# =========================================================
def extract_signal(sig: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], str]:
    t = to_1d_float_array(safe_get(sig, "time", []))
    x = to_1d_float_array(safe_get(sig, "x", []))
    md = to_dict_if_possible(safe_get(sig, "metadata", {}))
    file_name = str(safe_get(sig, "file_name", "") or "")

    n = min(len(t), len(x))
    if n <= 0:
        return np.array([], dtype=float), np.array([], dtype=float), md, file_name

    return t[:n], x[:n], md, file_name


def normalize_metadata(md: Dict[str, Any], fallback_name: str = "") -> Dict[str, Any]:
    raw = {str(k).strip(): v for k, v in (md or {}).items()}

    def pick(keys: List[str], default=""):
        for k in keys:
            if k in raw and raw[k] not in [None, ""]:
                return raw[k]
        return default

    return {
        "machine": str(pick(["Machine Name", "Machine", "MachineName"], "Unknown Machine")),
        "point": str(pick(["Point Name", "Point", "PointName", "Measurement Point"], fallback_name or "Unknown Point")),
        "variable": str(pick(["Variable", "Measurement", "Measure"], "Unknown Variable")),
        "unit": str(pick(["Y-Axis Unit", "Unit", "Amplitude Unit"], "")),
        "timestamp": str(pick(["Timestamp", "Date", "Date Time", "Datetime"], "")),
        "rpm_raw": pick(["RPM", "Sample Speed", "Speed", "Running Speed"], 0),
    }


def infer_family(variable: str, unit: str) -> str:
    v = (variable or "").lower()
    u = (unit or "").lower().strip()

    if any(s in v for s in ["prox", "displ", "shaft"]) or any(s in u for s in ["mil", "um", "µm", "micron"]):
        return "Proximity"
    if any(s in v for s in ["vel"]) or any(s in u for s in ["mm/s", "ips", "in/s"]):
        return "Velocity"
    if any(s in v for s in ["acc", "accel"]) or u == "g" or " g" in u:
        return "Acceleration"
    return "Unknown"


# =========================================================
# SIGNAL ANALYSIS FALLBACK
# =========================================================
def estimate_fs(time_array: np.ndarray) -> float:
    if len(time_array) < 2:
        return 0.0
    dt = np.diff(time_array)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def parse_samples_per_rev(variable: str, file_name: str = "") -> Optional[int]:
    candidates = [variable or "", file_name or ""]
    for text in candidates:
        m = re.search(r"(\d+)\s*x", str(text).lower())
        if m:
            try:
                val = int(m.group(1))
                if val > 1:
                    return val
            except Exception:
                pass
    return None


def compute_order_spectrum_from_sync_waveform(x: np.ndarray, samples_per_rev: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 8 or not samples_per_rev or samples_per_rev < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)

    n = len(x)
    window = np.hanning(n)
    yf = np.fft.rfft(x * window)
    bins = np.arange(len(yf), dtype=float)

    orders = bins * float(samples_per_rev) / float(n)

    coherent_gain = np.sum(window) / n
    amps = (2.0 / n) * np.abs(yf) / max(coherent_gain, 1e-12)

    if len(amps) > 0:
        amps[0] *= 0.5

    return orders, amps


def get_order_amplitude(orders: np.ndarray, amps: np.ndarray, target_order: float, band: float = 0.08) -> float:
    if len(orders) == 0 or target_order <= 0:
        return 0.0
    mask = (orders >= target_order - band) & (orders <= target_order + band)
    if np.sum(mask) == 0:
        return 0.0
    return float(np.max(amps[mask]))


def get_dominant_order(orders: np.ndarray, amps: np.ndarray) -> Tuple[float, float]:
    if len(orders) == 0 or len(amps) == 0:
        return 0.0, 0.0
    mask = orders > 0.1
    if np.sum(mask) == 0:
        return 0.0, 0.0
    o = orders[mask]
    a = amps[mask]
    idx = int(np.argmax(a))
    return float(o[idx]), float(a[idx])


def fft_amplitude_spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 64 or fs <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)

    n = len(x)
    window = np.hanning(n)
    yf = np.fft.rfft(x * window)
    xf = np.fft.rfftfreq(n, d=1.0 / fs)

    coherent_gain = np.sum(window) / n
    amps = (2.0 / n) * np.abs(yf) / max(coherent_gain, 1e-12)

    if len(amps) > 0:
        amps[0] *= 0.5

    return xf, amps


def parse_rpm(rpm_raw: Any, freqs: np.ndarray, amps: np.ndarray) -> float:
    try:
        if isinstance(rpm_raw, str):
            rpm_txt = rpm_raw.lower().replace("rpm", "").replace(",", "").strip()
            rpm = float(rpm_txt)
        else:
            rpm = float(rpm_raw)
        if math.isfinite(rpm) and rpm > 0:
            return rpm
    except Exception:
        pass

    if len(freqs) == 0:
        return 0.0

    mask = (freqs >= 5.0) & (freqs <= 250.0)
    if np.sum(mask) < 3:
        return 0.0

    f = freqs[mask]
    a = amps[mask]
    idx = int(np.argmax(a))
    return float(f[idx]) * 60.0


def overall_value(x: np.ndarray, family: str) -> Tuple[float, str]:
    if len(x) == 0:
        return 0.0, "N/A"

    x = np.asarray(x, dtype=float)

    if family == "Proximity":
        return float(np.max(x) - np.min(x)), "Peak-to-Peak"

    return float(np.sqrt(np.mean(x ** 2))), "RMS"


# =========================================================
# PHASE ANALYSIS INTEGRATION
# =========================================================
def normalize_text_key(s: Any) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def build_match_keys(signal_key: str, machine: str, point: str, variable: str, file_name: str) -> List[str]:
    combos = [
        signal_key,
        point,
        f"{machine}|{point}",
        f"{machine}|{point}|{variable}",
        file_name,
        f"{point}|{variable}",
    ]
    return [normalize_text_key(x) for x in combos if str(x or "").strip()]


def candidate_phase_store_keys() -> List[str]:
    return [
        "phase_results",
        "phase_analysis_results",
        "phase_analysis_df",
        "phase_df",
        "phase_table",
        "phase_data",
        "phase_summary",
        "order_results",
        "order_analysis_results",
    ]


def extract_numeric_from_record(rec: Dict[str, Any], candidates: List[str]) -> Optional[float]:
    for key in candidates:
        if key in rec:
            try:
                val = rec[key]
                if isinstance(val, str):
                    val = val.replace(",", "").strip()
                num = float(val)
                if math.isfinite(num):
                    return num
            except Exception:
                pass
    return None


def flatten_phase_records(store_obj: Any) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    if store_obj is None:
        return records

    if isinstance(store_obj, pd.DataFrame):
        return store_obj.to_dict(orient="records")

    if isinstance(store_obj, list):
        for item in store_obj:
            if isinstance(item, dict):
                records.append(item)
            elif hasattr(item, "__dict__"):
                records.append(dict(item.__dict__))
        return records

    if isinstance(store_obj, dict):
        for k, v in store_obj.items():
            if isinstance(v, dict):
                rec = dict(v)
                if "__source_key__" not in rec:
                    rec["__source_key__"] = k
                records.append(rec)
            elif isinstance(v, pd.DataFrame):
                part = v.to_dict(orient="records")
                for rec in part:
                    rec["__source_key__"] = k
                records.extend(part)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        rec = dict(item)
                        rec["__source_key__"] = k
                        records.append(rec)
            elif hasattr(v, "__dict__"):
                rec = dict(v.__dict__)
                rec["__source_key__"] = k
                records.append(rec)
        return records

    if hasattr(store_obj, "__dict__"):
        return [dict(store_obj.__dict__)]

    return records


def build_record_match_keys(rec: Dict[str, Any]) -> List[str]:
    possible_machine = rec.get("Machine") or rec.get("machine") or rec.get("Machine Name") or ""
    possible_point = rec.get("Point") or rec.get("point") or rec.get("Point Name") or ""
    possible_variable = rec.get("Variable") or rec.get("variable") or rec.get("Measurement") or ""
    possible_key = rec.get("__source_key__") or rec.get("Signal Key") or rec.get("signal_key") or rec.get("Key") or ""

    combos = [
        possible_key,
        possible_point,
        f"{possible_machine}|{possible_point}",
        f"{possible_machine}|{possible_point}|{possible_variable}",
        f"{possible_point}|{possible_variable}",
    ]
    return [normalize_text_key(x) for x in combos if str(x or "").strip()]


def lookup_phase_record(signal_key: str, machine: str, point: str, variable: str, file_name: str) -> Tuple[Optional[Dict[str, Any]], str]:
    match_keys = set(build_match_keys(signal_key, machine, point, variable, file_name))

    for store_key in candidate_phase_store_keys():
        store_obj = st.session_state.get(store_key, None)
        records = flatten_phase_records(store_obj)
        if not records:
            continue

        for rec in records:
            rec_keys = set(build_record_match_keys(rec))
            if match_keys & rec_keys:
                return rec, store_key

        point_norm = normalize_text_key(point)
        for rec in records:
            rec_point = normalize_text_key(
                rec.get("Point") or rec.get("point") or rec.get("Point Name") or ""
            )
            if point_norm and rec_point and point_norm == rec_point:
                return rec, store_key

    return None, ""


def extract_phase_metrics_from_record(rec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not rec:
        return {
            "amp_05x": 0.0,
            "amp_1x": 0.0,
            "amp_2x": 0.0,
            "phase_1x": None,
            "phase_2x": None,
            "stability": None,
            "confidence": None,
            "fit_amp_1x": None,
            "source": "fallback",
        }

    amp_05x = extract_numeric_from_record(
        rec,
        [
            "0.5X Amp",
            "0.5X Amplitude",
            "0.5X FFT Amp",
            "0.5X Fit Amp",
            "0.5X",
            "Amp 0.5X",
        ],
    )
    amp_1x = extract_numeric_from_record(
        rec,
        [
            "1X Amp",
            "1X Amplitude",
            "1X FFT Amp",
            "1X Fit Amp",
            "1X",
            "Amp 1X",
            "FFT Amp",
            "Fit Amp",
        ],
    )
    amp_2x = extract_numeric_from_record(
        rec,
        [
            "2X Amp",
            "2X Amplitude",
            "2X FFT Amp",
            "2X Fit Amp",
            "2X",
            "Amp 2X",
        ],
    )

    phase_1x = extract_numeric_from_record(
        rec,
        ["1X Phase", "Phase 1X", "Phase", "1X_Phase"]
    )
    phase_2x = extract_numeric_from_record(
        rec,
        ["2X Phase", "Phase 2X", "2X_Phase"]
    )
    stability = extract_numeric_from_record(
        rec,
        ["Stability", "stability", "1X Stability", "Phase Stability"]
    )
    confidence = extract_numeric_from_record(
        rec,
        ["Confidence", "Confidence %", "confidence"]
    )
    fit_amp_1x = extract_numeric_from_record(
        rec,
        ["Fit Amp", "1X Fit Amp", "FitAmp"]
    )

    return {
        "amp_05x": float(amp_05x or 0.0),
        "amp_1x": float(amp_1x or 0.0),
        "amp_2x": float(amp_2x or 0.0),
        "phase_1x": phase_1x,
        "phase_2x": phase_2x,
        "stability": stability,
        "confidence": confidence,
        "fit_amp_1x": fit_amp_1x,
        "source": "phase_analysis",
    }


# =========================================================
# RULE ENGINE
# =========================================================
def engineering_limits(family: str, unit: str) -> Tuple[float, float, str]:
    u = (unit or "").lower()

    if family == "Proximity":
        if "um" in u or "µm" in u or "mic" in u:
            return 75.0, 125.0, "µm p-p aprox"
        return 3.0, 5.0, "mil p-p aprox"

    if family == "Velocity":
        if "mm/s" in u:
            return 7.5, 12.5, "mm/s rms aprox"
        return 0.30, 0.50, "ips rms aprox"

    if family == "Acceleration":
        return 0.50, 1.00, "g rms aprox"

    return 1.0, 2.0, "engineering units"


def build_status(overall: float, alarm_v: float, danger_v: float) -> str:
    if overall >= danger_v:
        return "Danger"
    if overall >= alarm_v:
        return "Alert"
    if overall >= alarm_v * 0.60:
        return "Observe"
    return "Normal"


def build_fault_name(
    overall: float,
    amp_05x: float,
    amp_1x: float,
    amp_2x: float,
    dominant_order: float,
    phase_1x: Optional[float] = None,
) -> str:
    ratio_2x = amp_2x / amp_1x if amp_1x > 1e-12 else 0.0

    if amp_1x > 0 and amp_1x >= max(amp_2x * 1.35, 1e-9):
        return "Unbalance tendency"

    if ratio_2x >= 0.8 and amp_2x > 0:
        return "Misalignment tendency"

    if 0.35 <= dominant_order <= 0.65 and amp_05x > max(amp_1x, amp_2x):
        return "Sub-synchronous behavior"

    if overall > 0 and amp_1x < overall * 0.20:
        return "Broadband / looseness tendency"

    return "No dominant abnormal pattern"


def build_recommendation(status: str, fault: str, phase_source: str) -> str:
    if status in ["Danger", "Alert"]:
        if "Unbalance" in fault:
            return f"Review synchronous vibration and balancing condition. Harmonic source: {phase_source}."
        if "Misalignment" in fault:
            return f"Review alignment, coupling and thermal growth condition. Harmonic source: {phase_source}."
        if "Sub-synchronous" in fault:
            return f"Review rotor stability and fluid interaction. Harmonic source: {phase_source}."
        if "Broadband" in fault:
            return f"Review looseness, support condition and non-synchronous excitation. Harmonic source: {phase_source}."
        return f"Immediate engineering review recommended. Harmonic source: {phase_source}."

    if status == "Observe":
        return f"Track trend and validate with Phase Analysis / Spectrum. Harmonic source: {phase_source}."

    return f"Continue normal monitoring. Harmonic source: {phase_source}."


def compute_confidence(
    amp_05x: float,
    amp_1x: float,
    amp_2x: float,
    dom_amp: float,
    phase_confidence: Optional[float] = None,
) -> float:
    if phase_confidence is not None:
        try:
            pc = float(phase_confidence)
            if math.isfinite(pc):
                return float(np.clip(pc, 0.0, 100.0))
        except Exception:
            pass

    if dom_amp <= 1e-12:
        return 25.0

    raw = (amp_05x + amp_1x + amp_2x) / dom_amp * 45.0
    return float(np.clip(raw, 25.0, 100.0))


def status_chip(status: str) -> str:
    cls = {
        "Normal": "wm-chip-normal",
        "Observe": "wm-chip-observe",
        "Alert": "wm-chip-alert",
        "Danger": "wm-chip-danger",
    }.get(status, "wm-chip-info")
    return f'<span class="wm-chip {cls}">{status}</span>'


def info_chip(text: str) -> str:
    return f'<span class="wm-chip wm-chip-info">{text}</span>'


# =========================================================
# PROCESS
# =========================================================
signals = get_signal_bank()

if not signals:
    st.warning("No signals loaded in session_state['signals'].")
    st.stop()

results: List[Dict[str, Any]] = []
debug_rows: List[Dict[str, Any]] = []

for key, sig in signals.items():
    try:
        t, x, md_raw, file_name = extract_signal(sig)
        if len(t) < 64 or len(x) < 64:
            continue

        md = normalize_metadata(md_raw, fallback_name=str(key))
        family = infer_family(md["variable"], md["unit"])

        fs = estimate_fs(t)
        freqs, amps_hz = fft_amplitude_spectrum(x, fs)
        rpm = parse_rpm(md["rpm_raw"], freqs, amps_hz)

        samples_per_rev = parse_samples_per_rev(md["variable"], file_name)
        orders, amps_orders = compute_order_spectrum_from_sync_waveform(x, samples_per_rev or 0)
        dominant_order, dom_amp = get_dominant_order(orders, amps_orders)

        fallback_05x = get_order_amplitude(orders, amps_orders, 0.5)
        fallback_1x = get_order_amplitude(orders, amps_orders, 1.0)
        fallback_2x = get_order_amplitude(orders, amps_orders, 2.0)

        phase_record, phase_store_used = lookup_phase_record(
            signal_key=str(key),
            machine=md["machine"],
            point=md["point"],
            variable=md["variable"],
            file_name=file_name,
        )
        phase_metrics = extract_phase_metrics_from_record(phase_record)

        amp_05x = phase_metrics["amp_05x"] if phase_metrics["amp_05x"] > 0 else fallback_05x
        amp_1x = phase_metrics["amp_1x"] if phase_metrics["amp_1x"] > 0 else fallback_1x
        amp_2x = phase_metrics["amp_2x"] if phase_metrics["amp_2x"] > 0 else fallback_2x

        phase_1x = phase_metrics["phase_1x"]
        phase_2x = phase_metrics["phase_2x"]
        stability = phase_metrics["stability"]
        fit_amp_1x = phase_metrics["fit_amp_1x"]

        source_mode = "Phase Analysis" if phase_store_used else "Internal fallback"

        overall, overall_type = overall_value(x, family)
        alarm_v, danger_v, limit_label = engineering_limits(family, md["unit"])
        status = build_status(overall, alarm_v, danger_v)

        fault = build_fault_name(
            overall=overall,
            amp_05x=amp_05x,
            amp_1x=amp_1x,
            amp_2x=amp_2x,
            dominant_order=dominant_order,
            phase_1x=phase_1x,
        )

        confidence = compute_confidence(
            amp_05x=amp_05x,
            amp_1x=amp_1x,
            amp_2x=amp_2x,
            dom_amp=dom_amp,
            phase_confidence=phase_metrics["confidence"],
        )

        recommendation = build_recommendation(status, fault, source_mode)

        results.append(
            {
                "Signal Key": str(key),
                "Machine": md["machine"],
                "Point": md["point"],
                "Variable": md["variable"],
                "Family": family,
                "Unit": md["unit"],
                "Timestamp": md["timestamp"],
                "RPM": round(rpm, 1),
                "Overall": round(overall, 4),
                "Overall Type": overall_type,
                "0.5X Amp": round(float(amp_05x), 4),
                "1X Amp": round(float(amp_1x), 4),
                "2X Amp": round(float(amp_2x), 4),
                "1X Phase": None if phase_1x is None else round(float(phase_1x), 2),
                "2X Phase": None if phase_2x is None else round(float(phase_2x), 2),
                "Stability": None if stability is None else round(float(stability), 3),
                "Fit Amp 1X": None if fit_amp_1x is None else round(float(fit_amp_1x), 4),
                "Dominant Order": round(dominant_order, 3),
                "Status": status,
                "Fault": fault,
                "Confidence %": round(float(confidence), 1),
                "Alarm Limit": f"{alarm_v:.2f} {limit_label}",
                "Danger Limit": f"{danger_v:.2f} {limit_label}",
                "Recommendation": recommendation,
                "Orders Source": source_mode,
                "Phase Store": phase_store_used if phase_store_used else "",
                "Samples/Rev": samples_per_rev if samples_per_rev else "",
                "Samples": int(len(x)),
                "Fs": round(fs, 2),
            }
        )

        debug_rows.append(
            {
                "key": str(key),
                "sig_type": type(sig).__name__,
                "machine": md["machine"],
                "point": md["point"],
                "phase_store": phase_store_used if phase_store_used else "none",
                "samples_per_rev": samples_per_rev if samples_per_rev else "",
                "fallback_1x": round(float(fallback_1x), 5),
                "used_1x": round(float(amp_1x), 5),
            }
        )

    except Exception as e:
        debug_rows.append(
            {
                "key": str(key),
                "sig_type": type(sig).__name__,
                "machine": "ERROR",
                "point": f"{type(e).__name__}: {e}",
                "phase_store": "error",
                "samples_per_rev": "",
                "fallback_1x": "",
                "used_1x": "",
            }
        )
        continue

df = pd.DataFrame(results)

if df.empty:
    st.error("Diagnostic could not process any valid signals.")
    with st.expander("Debug info"):
        st.dataframe(pd.DataFrame(debug_rows), width="stretch")
        st.write("Session keys:", sorted(list(st.session_state.keys())))
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## Diagnostic Controls")

    family_options = sorted(df["Family"].dropna().unique().tolist())
    status_options = ["Normal", "Observe", "Alert", "Danger"]

    selected_families = st.multiselect("Family", family_options, default=family_options)
    selected_status = st.multiselect("Status", status_options, default=status_options)

    sort_by = st.selectbox(
        "Sort by",
        ["Severity", "Overall", "1X Amp", "2X Amp", "Confidence %", "RPM"],
        index=0,
    )

    source_filter = st.multiselect(
        "Orders Source",
        ["Phase Analysis", "Internal fallback"],
        default=["Phase Analysis", "Internal fallback"],
    )

    show_debug = st.toggle("Show debug panel", value=False)

filtered_df = df.copy()

if selected_families:
    filtered_df = filtered_df[filtered_df["Family"].isin(selected_families)]

if selected_status:
    filtered_df = filtered_df[filtered_df["Status"].isin(selected_status)]

if source_filter:
    filtered_df = filtered_df[filtered_df["Orders Source"].isin(source_filter)]

severity_rank = {"Danger": 3, "Alert": 2, "Observe": 1, "Normal": 0}
filtered_df["Severity Rank"] = filtered_df["Status"].map(severity_rank).fillna(0)

if sort_by == "Severity":
    filtered_df = filtered_df.sort_values(["Severity Rank", "Overall"], ascending=[False, False])
elif sort_by == "Overall":
    filtered_df = filtered_df.sort_values("Overall", ascending=False)
elif sort_by == "1X Amp":
    filtered_df = filtered_df.sort_values("1X Amp", ascending=False)
elif sort_by == "2X Amp":
    filtered_df = filtered_df.sort_values("2X Amp", ascending=False)
elif sort_by == "Confidence %":
    filtered_df = filtered_df.sort_values("Confidence %", ascending=False)
elif sort_by == "RPM":
    filtered_df = filtered_df.sort_values("RPM", ascending=False)

if filtered_df.empty:
    st.warning("No signals match the current filters.")
    st.stop()


# =========================================================
# KPI
# =========================================================
count_danger_alert = int(filtered_df["Status"].isin(["Danger", "Alert"]).sum())
avg_conf = float(filtered_df["Confidence %"].mean()) if not filtered_df.empty else 0.0
top_machine = filtered_df["Machine"].value_counts().idxmax() if not filtered_df.empty else "-"
phase_count = int((filtered_df["Orders Source"] == "Phase Analysis").sum())

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-title">Signals Screened</div>
            <div class="wm-kpi-value">{len(filtered_df)}</div>
            <div class="wm-kpi-sub">Loaded from session signal bank</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-title">Danger + Alert</div>
            <div class="wm-kpi-value">{count_danger_alert}</div>
            <div class="wm-kpi-sub">Immediate engineering review recommended</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-title">Average Confidence</div>
            <div class="wm-kpi-value">{avg_conf:.1f}%</div>
            <div class="wm-kpi-sub">Rule engine confidence layer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-title">Phase-linked Signals</div>
            <div class="wm-kpi-value">{phase_count}</div>
            <div class="wm-kpi-sub">Using specialist module data first</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# MAP + STATUS
# =========================================================
left, right = st.columns([1.45, 0.95])

with left:
    st.markdown('<div class="wm-section-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-section-title">Diagnostic Map</div>', unsafe_allow_html=True)

    fig = go.Figure()
    plot_df = filtered_df.copy()

    fig.add_trace(
        go.Scatter(
            x=plot_df["RPM"],
            y=plot_df["Overall"],
            mode="markers+text",
            text=plot_df["Point"],
            textposition="top center",
            customdata=plot_df[["Status", "Fault", "Confidence %", "Orders Source"]].to_numpy(),
            marker=dict(
                size=np.clip(plot_df["Confidence %"].to_numpy(), 18, 42),
                color=plot_df["Severity Rank"],
                colorscale=[
                    [0.00, "#18a957"],
                    [0.33, "#d7a600"],
                    [0.66, "#e26f00"],
                    [1.00, "#d92d20"],
                ],
                showscale=False,
                line=dict(width=1, color="rgba(0,0,0,0.18)"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "RPM: %{x:.1f}<br>"
                "Overall: %{y:.4f}<br>"
                "Status: %{customdata[0]}<br>"
                "Fault: %{customdata[1]}<br>"
                "Confidence: %{customdata[2]:.1f}%<br>"
                "Orders Source: %{customdata[3]}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=460,
        template="plotly_white",
        margin=dict(l=16, r=16, t=8, b=8),
        xaxis_title="RPM",
        yaxis_title="Overall",
    )
    st.plotly_chart(fig, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="wm-section-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-section-title">Status Breakdown</div>', unsafe_allow_html=True)

    counts = {
        "Normal": int((filtered_df["Status"] == "Normal").sum()),
        "Observe": int((filtered_df["Status"] == "Observe").sum()),
        "Alert": int((filtered_df["Status"] == "Alert").sum()),
        "Danger": int((filtered_df["Status"] == "Danger").sum()),
    }

    st.markdown(
        f"""
        {status_chip("Normal")} {counts["Normal"]}<br>
        {status_chip("Observe")} {counts["Observe"]}<br>
        {status_chip("Alert")} {counts["Alert"]}<br>
        {status_chip("Danger")} {counts["Danger"]}
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="wm-section-title">Top Findings</div>', unsafe_allow_html=True)

    top_faults = filtered_df["Fault"].value_counts().head(5)
    for fault_name, qty in top_faults.items():
        st.markdown(f'{info_chip(f"{qty}x")} {fault_name}', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# VERDICT BOARD
# =========================================================
st.markdown('<div class="wm-section-card">', unsafe_allow_html=True)
st.markdown('<div class="wm-section-title">Diagnostic Verdict Board</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wm-small-muted">Diagnostic interprets Phase Analysis first. If Phase data is not found, it falls back to internal order estimation.</div><br>',
    unsafe_allow_html=True,
)

for _, row in filtered_df.iterrows():
    status_lower = str(row["Status"]).lower()
    border_class = {
        "normal": "wm-verdict-left-normal",
        "observe": "wm-verdict-left-observe",
        "alert": "wm-verdict-left-alert",
        "danger": "wm-verdict-left-danger",
    }.get(status_lower, "wm-verdict-left-normal")

    phase_1x_txt = "-" if pd.isna(row["1X Phase"]) else f'{row["1X Phase"]:.2f}°'
    phase_2x_txt = "-" if pd.isna(row["2X Phase"]) else f'{row["2X Phase"]:.2f}°'
    stability_txt = "-" if pd.isna(row["Stability"]) else f'{row["Stability"]:.3f}'

    st.markdown(
        f"""
        <div class="wm-verdict {border_class}">
            <div class="wm-verdict-title">{row["Machine"]} | {row["Point"]}</div>
            <div class="wm-verdict-line"><b>Fault:</b> {row["Fault"]}</div>
            <div class="wm-verdict-line"><b>Evidence:</b> Overall {row["Overall"]:.4f} ({row["Overall Type"]}), 0.5X {row["0.5X Amp"]:.4f}, 1X {row["1X Amp"]:.4f}, 2X {row["2X Amp"]:.4f}, Dominant Order {row["Dominant Order"]:.3f}</div>
            <div class="wm-verdict-line"><b>Phase / Stability:</b> 1X Phase {phase_1x_txt}, 2X Phase {phase_2x_txt}, Stability {stability_txt}</div>
            <div class="wm-verdict-line"><b>Limits:</b> Alarm {row["Alarm Limit"]} | Danger {row["Danger Limit"]}</div>
            <div class="wm-verdict-line"><b>Recommendation:</b> {row["Recommendation"]}</div>
            <div style="margin-top:10px;">
                {status_chip(row["Status"])}
                {info_chip(row["Family"])}
                {info_chip(f"RPM {row['RPM']:.1f}")}
                {info_chip(f"Confidence {row['Confidence %']:.1f}%")}
                {info_chip(row["Orders Source"])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# EVIDENCE MATRIX
# =========================================================
st.markdown('<div class="wm-section-card">', unsafe_allow_html=True)
st.markdown('<div class="wm-section-title">Evidence Matrix</div>', unsafe_allow_html=True)

evidence_cols = [
    "Machine",
    "Point",
    "Family",
    "Unit",
    "RPM",
    "Overall",
    "Overall Type",
    "0.5X Amp",
    "1X Amp",
    "2X Amp",
    "1X Phase",
    "2X Phase",
    "Stability",
    "Dominant Order",
    "Status",
    "Fault",
    "Confidence %",
    "Orders Source",
]

st.dataframe(
    filtered_df[evidence_cols].reset_index(drop=True),
    width="stretch",
)
st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# INTEGRATION STATUS
# =========================================================
st.markdown('<div class="wm-section-card">', unsafe_allow_html=True)
st.markdown('<div class="wm-section-title">Cross-Module Integration Status</div>', unsafe_allow_html=True)

st.markdown(
    f"""
    {info_chip("Phase Analysis prioritized")}
    {info_chip("Order fallback enabled")}
    {info_chip("Spectrum integration pending")}
    {info_chip("Tabular List comparison pending")}
    {info_chip("AI interpretation layer pending")}
    """,
    unsafe_allow_html=True,
)

st.info(
    "If 0.5X / 1X / 2X still show zeros, the most probable reason is that Phase Analysis is not persisting its result table into st.session_state under any recognized store key. This Diagnostic page is already prepared to consume it once exposed."
)
st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# DEBUG PANEL
# =========================================================
if show_debug:
    st.markdown('<div class="wm-section-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-section-title">Debug Panel</div>', unsafe_allow_html=True)
    st.write("Session keys:", sorted(list(st.session_state.keys())))
    st.dataframe(pd.DataFrame(debug_rows), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
