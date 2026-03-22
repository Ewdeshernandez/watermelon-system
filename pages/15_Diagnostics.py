import io
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

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
        padding: 24px 28px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 14px 34px rgba(0,0,0,0.18);
        margin-bottom: 18px;
    }
    .wm-hero-title {
        color: white;
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.2px;
    }
    .wm-hero-sub {
        color: rgba(255,255,255,0.82);
        font-size: 14px;
        margin-top: 8px;
    }
    .wm-card {
        background: white;
        border-radius: 18px;
        padding: 16px 18px;
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    }
    .wm-kpi {
        background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
        border: 1px solid rgba(30,167,255,0.14);
        border-radius: 18px;
        padding: 14px 16px;
        min-height: 110px;
        box-shadow: 0 8px 22px rgba(30,167,255,0.08);
    }
    .wm-kpi-title {
        color: #50657f;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .wm-kpi-value {
        color: #081326;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.08;
        margin-top: 10px;
    }
    .wm-kpi-sub {
        color: #62768d;
        font-size: 12px;
        margin-top: 8px;
    }
    .wm-chip {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid rgba(0,0,0,0.05);
    }
    .wm-chip-normal { background: #e8f8ee; color: #0d7a3a; }
    .wm-chip-observe { background: #fff8e6; color: #9a6a00; }
    .wm-chip-alert { background: #fff0ea; color: #b44700; }
    .wm-chip-danger { background: #ffe9eb; color: #b42318; }
    .wm-chip-info { background: #eef6ff; color: #185ea9; }
    .wm-section-title {
        font-size: 18px;
        font-weight: 800;
        color: #0b1628;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA MODEL
# =========================================================
@dataclass
class DiagnosticResult:
    signal_key: str
    machine: str
    point: str
    variable: str
    family: str
    unit: str
    timestamp: str
    rpm: float
    overall: float
    overall_label: str
    amp_05x: float
    amp_1x: float
    amp_2x: float
    dominant_order: float
    dominant_amp: float
    confidence: float
    health_status: str
    finding: str
    criterion_based: str
    alarm: str
    danger: str
    n_samples: int
    fs: float


# =========================================================
# HELPERS
# =========================================================
def safe_get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_signal_dict() -> Dict[str, Any]:
    signals = st.session_state.get("signals", {})
    return signals if isinstance(signals, dict) else {}


def normalize_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    md = md or {}
    out = {str(k).strip(): v for k, v in md.items()}

    aliases = {
        "machine": ["Machine Name", "Machine", "MachineName"],
        "point": ["Point Name", "Point", "PointName", "Measurement Point"],
        "variable": ["Variable", "Measurement", "Measure"],
        "sample_speed": ["Sample Speed", "RPM", "Speed", "Running Speed"],
        "timestamp": ["Timestamp", "Date", "Date Time", "Datetime"],
        "unit": ["Y-Axis Unit", "Unit", "Amplitude Unit"],
        "file_name": ["File Name", "Filename"],
    }

    result = {}
    for std_key, keys in aliases.items():
        result[std_key] = ""
        for k in keys:
            if k in out:
                result[std_key] = str(out[k])
                break
    return result


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


def to_1d_float_array(arr: Any) -> np.ndarray:
    if arr is None:
        return np.array([], dtype=float)
    try:
        a = np.asarray(arr, dtype=float)
        if a.ndim != 1:
            a = a.flatten()
        return a
    except Exception:
        return np.array([], dtype=float)


def to_numpy_signal(sig: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], str]:
    t = to_1d_float_array(safe_get(sig, "time", None))
    x = to_1d_float_array(safe_get(sig, "x", None))
    md = safe_get(sig, "metadata", {}) or {}
    file_name = str(safe_get(sig, "file_name", "") or "")

    n = min(len(t), len(x))
    if n <= 0:
        return np.array([], dtype=float), np.array([], dtype=float), md, file_name

    return t[:n], x[:n], md, file_name


def estimate_fs(time_array: np.ndarray) -> float:
    if len(time_array) < 2:
        return 0.0
    dt = np.diff(time_array)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def clean_signal(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x - np.mean(x)


def fft_amplitude_spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 32 or fs <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    n = len(x)
    window = np.hanning(n)
    xw = x * window
    yf = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    coherent_gain = np.sum(window) / n
    amps = (2.0 / n) * np.abs(yf) / max(coherent_gain, 1e-12)

    if len(amps) > 0:
        amps[0] *= 0.5

    return freqs, amps


def parse_rpm(md_norm: Dict[str, Any], freqs: np.ndarray, amps: np.ndarray) -> float:
    raw = str(md_norm.get("sample_speed", "")).strip()
    if raw:
        try:
            txt = raw.lower().replace("rpm", "").replace(",", "").strip()
            return float(txt)
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


def harmonic_amp(freqs: np.ndarray, amps: np.ndarray, target_hz: float, band_ratio: float = 0.04) -> float:
    if len(freqs) == 0 or target_hz <= 0:
        return 0.0
    bw = max(target_hz * band_ratio, 0.5)
    mask = (freqs >= target_hz - bw) & (freqs <= target_hz + bw)
    if np.sum(mask) == 0:
        return 0.0
    return float(np.max(amps[mask]))


def dominant_peak(freqs: np.ndarray, amps: np.ndarray) -> Tuple[float, float]:
    if len(freqs) == 0:
        return 0.0, 0.0
    mask = freqs > 0.5
    if np.sum(mask) == 0:
        return 0.0, 0.0
    f = freqs[mask]
    a = amps[mask]
    idx = int(np.argmax(a))
    return float(f[idx]), float(a[idx])


def overall_value(x_raw: np.ndarray, family: str) -> Tuple[float, str]:
    if len(x_raw) == 0:
        return 0.0, "N/A"

    x_raw = np.asarray(x_raw, dtype=float)

    if family == "Proximity":
        return float(np.max(x_raw) - np.min(x_raw)), "Peak-to-Peak"
    if family == "Velocity":
        return float(np.sqrt(np.mean(x_raw ** 2))), "RMS"
    if family == "Acceleration":
        return float(np.sqrt(np.mean(x_raw ** 2))), "RMS"

    return float(np.sqrt(np.mean(x_raw ** 2))), "RMS"


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


def build_finding(overall: float, alarm_v: float, amp_05x: float, amp_1x: float, amp_2x: float, dominant_order: float) -> str:
    ratio_2x = amp_2x / amp_1x if amp_1x > 1e-12 else 0.0

    if amp_1x > 0 and amp_1x >= max(amp_2x * 1.35, 1e-9):
        return "1X dominant pattern compatible with unbalance tendency"

    if ratio_2x >= 0.8 and amp_2x > 0:
        return "2X relevant pattern compatible with misalignment tendency"

    if 0.35 <= dominant_order <= 0.65 and amp_05x > max(amp_1x, amp_2x):
        return "0.5X dominant content, review sub-synchronous behavior"

    if overall > alarm_v and amp_1x < overall * 0.20:
        return "Overall elevated with low synchronous dominance, review broadband or non-synchronous content"

    return "No dominant abnormal pattern detected"


def build_status(overall: float, alarm_v: float, danger_v: float) -> str:
    if overall >= danger_v:
        return "Danger"
    if overall >= alarm_v:
        return "Alert"
    if overall >= alarm_v * 0.60:
        return "Observe"
    return "Normal"


def compute_confidence(amp_05x: float, amp_1x: float, amp_2x: float, dom_amp: float) -> float:
    if dom_amp <= 1e-12:
        return 25.0
    raw = (amp_05x + amp_1x + amp_2x) / dom_amp * 45.0
    return float(np.clip(raw, 25.0, 100.0))


def build_result(signal_key: str, sig: Any) -> Optional[DiagnosticResult]:
    t, x_raw, md, file_name = to_numpy_signal(sig)
    if len(t) < 32 or len(x_raw) < 32:
        return None

    md_norm = normalize_metadata(md)
    machine = md_norm.get("machine") or "Unknown Machine"
    point = md_norm.get("point") or file_name or signal_key
    variable = md_norm.get("variable") or "Unknown Variable"
    timestamp = md_norm.get("timestamp") or ""
    unit = md_norm.get("unit") or ""
    family = infer_family(variable, unit)

    fs = estimate_fs(t)
    x = clean_signal(x_raw)
    freqs, amps = fft_amplitude_spectrum(x, fs)
    rpm = parse_rpm(md_norm, freqs, amps)

    run_hz = rpm / 60.0 if rpm > 0 else 0.0

    amp_05x = harmonic_amp(freqs, amps, run_hz * 0.5)
    amp_1x = harmonic_amp(freqs, amps, run_hz * 1.0)
    amp_2x = harmonic_amp(freqs, amps, run_hz * 2.0)
    dom_freq, dom_amp = dominant_peak(freqs, amps)
    dominant_order = (dom_freq / run_hz) if run_hz > 0 else 0.0

    overall, overall_label = overall_value(x_raw, family)
    alarm_v, danger_v, limit_label = engineering_limits(family, unit)

    status = build_status(overall, alarm_v, danger_v)
    finding = build_finding(overall, alarm_v, amp_05x, amp_1x, amp_2x, dominant_order)
    confidence = compute_confidence(amp_05x, amp_1x, amp_2x, dom_amp)

    return DiagnosticResult(
        signal_key=str(signal_key),
        machine=str(machine),
        point=str(point),
        variable=str(variable),
        family=str(family),
        unit=str(unit),
        timestamp=str(timestamp),
        rpm=float(rpm),
        overall=float(overall),
        overall_label=str(overall_label),
        amp_05x=float(amp_05x),
        amp_1x=float(amp_1x),
        amp_2x=float(amp_2x),
        dominant_order=float(dominant_order),
        dominant_amp=float(dom_amp),
        confidence=float(confidence),
        health_status=str(status),
        finding=str(finding),
        criterion_based="Auto",
        alarm=f"{alarm_v:.2f} {limit_label}",
        danger=f"{danger_v:.2f} {limit_label}",
        n_samples=int(len(x_raw)),
        fs=float(fs),
    )


def evaluate_signals(signals: Dict[str, Any]) -> List[DiagnosticResult]:
    results: List[DiagnosticResult] = []

    for signal_key, sig in signals.items():
        try:
            result = build_result(signal_key, sig)
            if result is not None:
                results.append(result)
        except Exception:
            continue

    return results


def results_to_df(results: List[DiagnosticResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Machine": r.machine,
                "Point": r.point,
                "Variable": r.variable,
                "Family": r.family,
                "Unit": r.unit,
                "Timestamp": r.timestamp,
                "RPM": round(r.rpm, 1),
                "Overall": round(r.overall, 4),
                "Overall Type": r.overall_label,
                "0.5X Amplitude": round(r.amp_05x, 4),
                "1X Amplitude": round(r.amp_1x, 4),
                "2X Amplitude": round(r.amp_2x, 4),
                "Dominant Order": round(r.dominant_order, 3),
                "Dominant Amp": round(r.dominant_amp, 4),
                "Confidence %": round(r.confidence, 1),
                "Criterion Based": r.criterion_based,
                "Alarm": r.alarm,
                "Danger": r.danger,
                "Status": r.health_status,
                "Finding": r.finding,
                "Samples": r.n_samples,
                "Fs": round(r.fs, 2),
            }
        )
    return pd.DataFrame(rows)


def make_status_counts(results: List[DiagnosticResult]) -> Dict[str, int]:
    counts = {"Normal": 0, "Observe": 0, "Alert": 0, "Danger": 0}
    for r in results:
        counts[r.health_status] = counts.get(r.health_status, 0) + 1
    return counts


def status_chip(status: str) -> str:
    klass = {
        "Normal": "wm-chip-normal",
        "Observe": "wm-chip-observe",
        "Alert": "wm-chip-alert",
        "Danger": "wm-chip-danger",
    }.get(status, "wm-chip-info")
    return f'<span class="wm-chip {klass}">{status}</span>'


def make_diagnostic_chart(results: List[DiagnosticResult]) -> go.Figure:
    fig = go.Figure()

    if not results:
        fig.update_layout(height=500, template="plotly_white")
        return fig

    df = results_to_df(results)
    order_map = {"Normal": 0, "Observe": 1, "Alert": 2, "Danger": 3}
    df["Severity"] = df["Status"].map(order_map).fillna(0)

    fig.add_trace(
        go.Scatter(
            x=df["RPM"],
            y=df["Overall"],
            mode="markers+text",
            text=df["Point"],
            textposition="top center",
            marker=dict(
                size=np.clip(df["Confidence %"].fillna(30).to_numpy(), 18, 42),
                color=df["Severity"],
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
                "Finding: %{customdata[1]}<extra></extra>"
            ),
            customdata=df[["Status", "Finding"]].to_numpy(),
        )
    )

    fig.update_layout(
        height=500,
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="RPM",
        yaxis_title="Overall",
    )
    return fig


def build_png_summary(results: List[DiagnosticResult]) -> bytes:
    width = 1800
    row_h = 54
    top_h = 200
    bottom_pad = 60
    rows_to_draw = min(max(len(results), 1), 18)
    height = top_h + rows_to_draw * row_h + bottom_pad

    img = Image.new("RGB", (width, height), (246, 249, 253))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        sub_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        hdr_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        txt_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
        bold_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    except Exception:
        title_font = sub_font = hdr_font = txt_font = bold_font = ImageFont.load_default()

    draw.rounded_rectangle((40, 30, width - 40, 150), radius=28, fill=(11, 18, 32))
    draw.text((70, 55), "Watermelon System | Diagnostic Executive Summary", font=title_font, fill=(255, 255, 255))
    draw.text((70, 108), f"Signals analyzed: {len(results)}", font=sub_font, fill=(195, 210, 228))

    cols = [
        ("Machine", 50),
        ("Point", 310),
        ("RPM", 560),
        ("Overall", 700),
        ("1X", 860),
        ("2X", 980),
        ("Status", 1110),
        ("Finding", 1270),
    ]

    header_y = 180
    draw.rounded_rectangle((40, header_y, width - 40, header_y + 42), radius=12, fill=(224, 236, 249))
    for text, x in cols:
        draw.text((x, header_y + 10), text, font=hdr_font, fill=(13, 34, 61))

    y = header_y + 52
    for idx, r in enumerate(results[:18]):
        fill = (255, 255, 255) if idx % 2 == 0 else (248, 251, 255)
        draw.rounded_rectangle((40, y, width - 40, y + 42), radius=10, fill=fill)

        color = {
            "Normal": (24, 169, 87),
            "Observe": (215, 166, 0),
            "Alert": (226, 111, 0),
            "Danger": (217, 45, 32),
        }.get(r.health_status, (24, 94, 169))

        draw.text((50, y + 10), str(r.machine)[:26], font=txt_font, fill=(18, 26, 39))
        draw.text((310, y + 10), str(r.point)[:22], font=txt_font, fill=(18, 26, 39))
        draw.text((560, y + 10), f"{r.rpm:.1f}", font=txt_font, fill=(18, 26, 39))
        draw.text((700, y + 10), f"{r.overall:.4f}", font=txt_font, fill=(18, 26, 39))
        draw.text((860, y + 10), f"{r.amp_1x:.4f}", font=txt_font, fill=(18, 26, 39))
        draw.text((980, y + 10), f"{r.amp_2x:.4f}", font=txt_font, fill=(18, 26, 39))
        draw.text((1110, y + 10), r.health_status, font=bold_font, fill=color)
        draw.text((1270, y + 10), str(r.finding)[:48], font=txt_font, fill=(18, 26, 39))
        y += row_h

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="wm-hero">
        <div class="wm-hero-title">Diagnostic</div>
        <div class="wm-hero-sub">
            Executive diagnosis layer for Watermelon System. Fast screening, order-based review,
            commercial clarity and real engineering logic.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD / EVALUATE
# =========================================================
signals = get_signal_dict()

with st.sidebar:
    st.markdown("## Diagnostic Controls")

    if not signals:
        st.info("No signals loaded in session_state['signals'].")

    only_alarm_plus = st.toggle("Show only Observe/Alert/Danger", value=False)

    family_filter = st.multiselect(
        "Measurement Family",
        ["Proximity", "Velocity", "Acceleration", "Unknown"],
        default=["Proximity", "Velocity", "Acceleration", "Unknown"],
    )

    sort_mode = st.selectbox(
        "Sort by",
        ["Severity", "Overall", "1X Amplitude", "2X Amplitude", "RPM", "Confidence"],
        index=0,
    )

if not signals:
    st.warning("No loaded signals found. Load data first to use Diagnostic.")
    st.stop()

results = evaluate_signals(signals)

if not results:
    st.warning("Signals were found, but no valid time waveform arrays could be processed.")
    st.stop()

if family_filter:
    results = [r for r in results if r.family in family_filter]

if only_alarm_plus:
    results = [r for r in results if r.health_status in ["Observe", "Alert", "Danger"]]

severity_rank = {"Danger": 3, "Alert": 2, "Observe": 1, "Normal": 0}

if sort_mode == "Severity":
    results = sorted(results, key=lambda r: (severity_rank.get(r.health_status, 0), r.overall), reverse=True)
elif sort_mode == "Overall":
    results = sorted(results, key=lambda r: r.overall, reverse=True)
elif sort_mode == "1X Amplitude":
    results = sorted(results, key=lambda r: r.amp_1x, reverse=True)
elif sort_mode == "2X Amplitude":
    results = sorted(results, key=lambda r: r.amp_2x, reverse=True)
elif sort_mode == "RPM":
    results = sorted(results, key=lambda r: r.rpm, reverse=True)
elif sort_mode == "Confidence":
    results = sorted(results, key=lambda r: r.confidence, reverse=True)

df = results_to_df(results)
counts = make_status_counts(results)

danger_alert = counts.get("Danger", 0) + counts.get("Alert", 0)
avg_conf = round(float(np.mean([r.confidence for r in results])) if results else 0.0, 1)
top_machine = df["Machine"].value_counts().idxmax() if not df.empty else "-"

# =========================================================
# KPI ROW
# =========================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-title">Signals Screened</div>
            <div class="wm-kpi-value">{len(results)}</div>
            <div class="wm-kpi-sub">Loaded from session signal bank</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-title">Danger + Alert</div>
            <div class="wm-kpi-value">{danger_alert}</div>
            <div class="wm-kpi-sub">Immediate engineering review recommended</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-title">Average Confidence</div>
            <div class="wm-kpi-value">{avg_conf}%</div>
            <div class="wm-kpi-sub">Auto diagnostic confidence layer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-title">Top Machine in View</div>
            <div class="wm-kpi-value" style="font-size:22px;">{top_machine}</div>
            <div class="wm-kpi-sub">Current filtered selection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# MAIN PANELS
# =========================================================
st.markdown("")
left, right = st.columns([1.35, 1.0])

with left:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-section-title">Diagnostic Map</div>', unsafe_allow_html=True)
    st.plotly_chart(make_diagnostic_chart(results), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-section-title">Status Breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        {status_chip('Normal')} {counts.get('Normal', 0)}<br>
        {status_chip('Observe')} {counts.get('Observe', 0)}<br>
        {status_chip('Alert')} {counts.get('Alert', 0)}<br>
        {status_chip('Danger')} {counts.get('Danger', 0)}
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    top_findings = df["Finding"].value_counts().head(5) if not df.empty else pd.Series(dtype=int)

    if len(top_findings) > 0:
        st.markdown('<div class="wm-section-title">Top Findings</div>', unsafe_allow_html=True)
        for finding, count in top_findings.items():
            st.markdown(
                f'<span class="wm-chip wm-chip-info">{count}x</span> {finding}',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TABLE
# =========================================================
st.markdown("")
st.markdown('<div class="wm-card">', unsafe_allow_html=True)
st.markdown('<div class="wm-section-title">Executive Diagnostic Table</div>', unsafe_allow_html=True)
st.dataframe(df, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# DETAIL REVIEW
# =========================================================
st.markdown("")
st.markdown('<div class="wm-card">', unsafe_allow_html=True)
st.markdown('<div class="wm-section-title">Detailed Review</div>', unsafe_allow_html=True)

point_options = [f"{r.machine} | {r.point} | {r.variable}" for r in results]
selected_idx = st.selectbox("Select signal", list(range(len(point_options))), format_func=lambda i: point_options[i])
selected = results[selected_idx]

d1, d2, d3, d4 = st.columns(4)
d1.metric("Machine", selected.machine)
d2.metric("Point", selected.point)
d3.metric("RPM", f"{selected.rpm:.1f}")
d4.metric("Status", selected.health_status)

d5, d6, d7, d8 = st.columns(4)
d5.metric("Overall", f"{selected.overall:.4f}")
d6.metric("0.5X", f"{selected.amp_05x:.4f}")
d7.metric("1X", f"{selected.amp_1x:.4f}")
d8.metric("2X", f"{selected.amp_2x:.4f}")

st.markdown(
    f"""
    {status_chip(selected.health_status)}
    <span class="wm-chip wm-chip-info">{selected.family}</span>
    <span class="wm-chip wm-chip-info">{selected.overall_label}</span>
    <span class="wm-chip wm-chip-info">Confidence {selected.confidence:.1f}%</span>
    """,
    unsafe_allow_html=True,
)

st.info(selected.finding)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# EXPORTS
# =========================================================
st.markdown("")
c1, c2 = st.columns(2)

with c1:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="watermelon_diagnostic_table.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c2:
    png_bytes = build_png_summary(results)
    st.download_button(
        "Download PNG HD",
        data=png_bytes,
        file_name="watermelon_diagnostic_summary.png",
        mime="image/png",
        use_container_width=True,
    )