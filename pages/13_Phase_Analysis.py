# pages/13_Phase_Analysis.py

import base64
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


PRIMARY = "#1ea7ff"
BG = "#f7fafe"
CARD = "#ffffff"
BORDER = "#e6edf5"
TEXT = "#0f172a"
MUTED = "#64748b"
SUCCESS = "#16a34a"
WARNING = "#f59e0b"
DANGER = "#ef4444"


def safe_signals():
    signals = st.session_state.get("signals", [])
    if isinstance(signals, list):
        return [s for s in signals if isinstance(s, dict)]
    return []


def to_array(value):
    if value is None:
        return np.array([], dtype=float)

    if isinstance(value, np.ndarray):
        try:
            return value.astype(float, copy=False).flatten()
        except Exception:
            return np.array([], dtype=float)

    if isinstance(value, (list, tuple, pd.Series)):
        try:
            return np.asarray(value, dtype=float).flatten()
        except Exception:
            return np.array([], dtype=float)

    try:
        return np.array([float(value)], dtype=float)
    except Exception:
        return np.array([], dtype=float)


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            arr = to_array(value)
            if arr.size == 0:
                return default
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return default
            return float(np.nanmean(finite))
        return float(value)
    except Exception:
        return default


def fmt_num(value, decimals=2, suffix=""):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "—"
        return f"{float(value):,.{decimals}f}{suffix}"
    except Exception:
        return "—"


def fmt_text(value, default="—"):
    if value is None:
        return default
    txt = str(value).strip()
    return txt if txt else default


def get_logo_data_uri():
    candidates = [
        "assets/watermelon_logo.png",
        "assets/watermelon.png",
        "assets/logo.png",
        "logo.png",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/png;base64,{data}"
            except Exception:
                pass
    return None


def split_signal_name(name):
    txt = fmt_text(name, "Unnamed Signal")
    parts = [p.strip() for p in txt.replace("|", "-").split("-") if p.strip()]
    if len(parts) >= 3:
        return parts[0], parts[1], " - ".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1], "1X Phase"
    return "Machine", txt, "1X Phase"


def extract_rpm(signal):
    for key in ["rpm", "RPM", "speed", "shaft_rpm"]:
        if key in signal:
            rpm = safe_float(signal.get(key))
            if not np.isnan(rpm):
                return rpm
    return np.nan


def extract_timestamp(signal):
    for key in ["timestamp", "Timestamp", "datetime", "date"]:
        if key in signal:
            return fmt_text(signal.get(key))
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def estimate_1x_amplitude(y):
    if y.size == 0:
        return np.nan
    finite = y[np.isfinite(y)]
    if finite.size < 2:
        return np.nan
    return float((np.max(finite) - np.min(finite)) / 2.0)


def estimate_1x_phase_deg(y):
    if y.size < 8:
        return np.nan

    finite = y[np.isfinite(y)]
    if finite.size < 8:
        return np.nan

    x = finite - np.mean(finite)
    n = len(x)
    if n < 8:
        return np.nan

    fft_vals = np.fft.rfft(x)
    if len(fft_vals) < 2:
        return np.nan

    phase_rad = np.angle(fft_vals[1])
    phase_deg = np.degrees(phase_rad) % 360.0
    return float(phase_deg)


def estimate_peak(y):
    if y.size == 0:
        return np.nan
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return np.nan
    return float(np.max(np.abs(finite)))


def estimate_stability(y, rpm):
    score = 100.0

    finite = y[np.isfinite(y)]
    if finite.size >= 16:
        rms = float(np.sqrt(np.mean(np.square(finite))))
        if rms > 0:
            crest = float(np.max(np.abs(finite)) / rms)
            score -= min(30.0, max(0.0, crest - 1.8) * 18.0)

        halves = np.array_split(finite, 4)
        amps = []
        for h in halves:
            if len(h) >= 4:
                amps.append((np.max(h) - np.min(h)) / 2.0)
        if len(amps) >= 2:
            mean_amp = np.mean(amps)
            if mean_amp > 0:
                drift = np.std(amps) / mean_amp
                score -= min(35.0, drift * 150.0)

    if np.isnan(rpm):
        score -= 10.0

    score = max(0.0, min(100.0, score))

    if score >= 80:
        return "Stable", SUCCESS
    if score >= 55:
        return "Watch", WARNING
    return "Unstable", DANGER


def compute_phase_metrics(signal):
    name = fmt_text(signal.get("name"), "Unnamed Signal")
    y = to_array(signal.get("y"))
    rpm = extract_rpm(signal)
    amp_1x = estimate_1x_amplitude(y)
    phase_deg = estimate_1x_phase_deg(y)
    peak = estimate_peak(y)
    stability, color = estimate_stability(y, rpm)

    return {
        "Signal": name,
        "1X Amplitude": amp_1x,
        "Phase": phase_deg,
        "Stability": stability,
        "Stability Color": color,
        "RPM": rpm,
        "Peak": peak,
        "timestamp": extract_timestamp(signal),
    }


def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {BG};
        }}

        div[data-testid="stSidebar"] {{
            background: #ffffff;
            border-right: 1px solid {BORDER};
        }}

        .wm-header {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 4px 16px rgba(15,23,42,0.04);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }}

        .wm-left {{
            display: flex;
            align-items: center;
            gap: 14px;
            min-width: 280px;
            flex: 1 1 420px;
        }}

        .wm-logo {{
            width: 48px;
            height: 48px;
            border-radius: 12px;
            object-fit: contain;
            background: #eef7ff;
            border: 1px solid {BORDER};
            padding: 4px;
        }}

        .wm-logo-fallback {{
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: {PRIMARY};
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 16px;
        }}

        .wm-title-wrap {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}

        .wm-kicker {{
            color: {PRIMARY};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
        }}

        .wm-title {{
            color: {TEXT};
            font-size: 24px;
            font-weight: 700;
            line-height: 1.15;
        }}

        .wm-subtitle {{
            color: {MUTED};
            font-size: 13px;
        }}

        .wm-right {{
            display: grid;
            grid-template-columns: repeat(3, minmax(130px, 1fr));
            gap: 10px;
            flex: 1 1 560px;
        }}

        .wm-meta {{
            background: #f8fbff;
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 10px 12px;
            min-height: 64px;
        }}

        .wm-meta-label {{
            color: {MUTED};
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: 4px;
        }}

        .wm-meta-value {{
            color: {TEXT};
            font-size: 14px;
            font-weight: 700;
            line-height: 1.2;
            word-break: break-word;
        }}

        .wm-card {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 14px 16px 8px 16px;
            box-shadow: 0 4px 16px rgba(15,23,42,0.04);
        }}

        .wm-card-title {{
            color: {TEXT};
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 10px;
        }}

        .wm-empty {{
            background: {CARD};
            border: 1px dashed {BORDER};
            border-radius: 18px;
            padding: 32px 20px;
            text-align: center;
            color: {MUTED};
        }}

        @media (max-width: 1100px) {{
            .wm-right {{
                grid-template-columns: repeat(2, minmax(130px, 1fr));
            }}
        }}

        @media (max-width: 700px) {{
            .wm-right {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_phase_header(meta):
    logo_uri = get_logo_data_uri()

    if logo_uri:
        logo_html = f'<img src="{logo_uri}" class="wm-logo" />'
    else:
        logo_html = '<div class="wm-logo-fallback">WM</div>'

    st.markdown(
        f"""
        <div class="wm-header">
            <div class="wm-left">
                {logo_html}
                <div class="wm-title-wrap">
                    <div class="wm-kicker">Watermelon System</div>
                    <div class="wm-title">1X Phase Analysis</div>
                    <div class="wm-subtitle">Executive dashboard for direct phase comparison</div>
                </div>
            </div>

            <div class="wm-right">
                <div class="wm-meta">
                    <div class="wm-meta-label">Machine</div>
                    <div class="wm-meta-value">{meta["machine"]}</div>
                </div>
                <div class="wm-meta">
                    <div class="wm-meta-label">Point</div>
                    <div class="wm-meta-value">{meta["point"]}</div>
                </div>
                <div class="wm-meta">
                    <div class="wm-meta-label">Configuration</div>
                    <div class="wm-meta-value">{meta["config"]}</div>
                </div>
                <div class="wm-meta">
                    <div class="wm-meta-label">RPM</div>
                    <div class="wm-meta-value">{meta["rpm"]}</div>
                </div>
                <div class="wm-meta">
                    <div class="wm-meta-label">Peak</div>
                    <div class="wm-meta-value">{meta["peak"]}</div>
                </div>
                <div class="wm-meta">
                    <div class="wm-meta-label">Timestamp</div>
                    <div class="wm-meta-value">{meta["timestamp"]}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_font(size=24, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def build_png(df_export, meta):
    width = 2000
    row_h = 72
    header_h = 320
    table_y = 360
    height = max(900, table_y + 140 + len(df_export) * row_h + 80)

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    font_title = load_font(50, bold=True)
    font_sub = load_font(24, bold=False)
    font_label = load_font(20, bold=False)
    font_value = load_font(24, bold=True)
    font_table_head = load_font(24, bold=True)
    font_table = load_font(22, bold=False)
    font_badge = load_font(20, bold=True)

    draw.rounded_rectangle((40, 40, width - 40, header_h), radius=24, fill=CARD, outline=BORDER, width=2)
    draw.rounded_rectangle((40, table_y, width - 40, height - 40), radius=24, fill=CARD, outline=BORDER, width=2)

    draw.ellipse((80, 92, 172, 184), fill=PRIMARY)
    draw.text((103, 118), "WM", fill="white", font=load_font(30, bold=True))

    draw.text((210, 88), "Watermelon System", fill=PRIMARY, font=font_sub)
    draw.text((210, 122), "1X Phase Analysis", fill=TEXT, font=font_title)
    draw.text((210, 194), "Executive dashboard for direct phase comparison", fill=MUTED, font=font_sub)

    meta_items = [
        ("Machine", meta["machine"]),
        ("Point", meta["point"]),
        ("Configuration", meta["config"]),
        ("RPM", meta["rpm"]),
        ("Peak", meta["peak"]),
        ("Timestamp", meta["timestamp"]),
    ]

    start_x = 1120
    start_y = 84
    card_w = 360
    card_h = 88
    gap_x = 18
    gap_y = 16

    for i, (label, value) in enumerate(meta_items):
        row = i // 2
        col = i % 2
        x1 = start_x + col * (card_w + gap_x)
        y1 = start_y + row * (card_h + gap_y)
        x2 = x1 + card_w
        y2 = y1 + card_h
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill="#f8fbff", outline=BORDER, width=2)
        draw.text((x1 + 18, y1 + 12), label.upper(), fill=MUTED, font=load_font(16, bold=False))
        draw.text((x1 + 18, y1 + 40), str(value), fill=TEXT, font=font_value)

    draw.text((70, table_y + 28), "Phase Summary", fill=TEXT, font=load_font(30, bold=True))

    headers = ["Signal", "1X Amplitude", "Phase", "Stability", "RPM"]
    col_x = [70, 980, 1270, 1470, 1760]
    head_y = table_y + 92

    for x, h in zip(col_x, headers):
        draw.text((x, head_y), h, fill=MUTED, font=font_table_head)

    line_y = head_y + 42
    draw.line((70, line_y, width - 70, line_y), fill=BORDER, width=2)

    for i, row in df_export.iterrows():
        y = line_y + 24 + i * row_h

        if i % 2 == 0:
            draw.rounded_rectangle((56, y - 10, width - 56, y + 48), radius=14, fill="#fbfdff")

        draw.text((col_x[0], y), str(row["Signal"]), fill=TEXT, font=font_table)
        draw.text((col_x[1], y), str(row["1X Amplitude"]), fill=TEXT, font=font_table)
        draw.text((col_x[2], y), str(row["Phase"]), fill=TEXT, font=font_table)
        draw.text((col_x[4], y), str(row["RPM"]), fill=TEXT, font=font_table)

        badge_color = SUCCESS if row["Stability"] == "Stable" else WARNING if row["Stability"] == "Watch" else DANGER
        bx1, by1, bx2, by2 = col_x[3], y - 2, col_x[3] + 180, y + 40
        draw.rounded_rectangle((bx1, by1, bx2, by2), radius=18, fill="white", outline=badge_color, width=2)
        draw.text((bx1 + 18, by1 + 8), str(row["Stability"]), fill=badge_color, font=font_badge)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()


inject_css()

signals = safe_signals()

st.sidebar.markdown("## Phase Analysis")

if not signals:
    st.markdown(
        """
        <div class="wm-empty">
            <h3 style="margin:0 0 8px 0; color:#0f172a;">No signals loaded</h3>
            <div>Load signals first from your acquisition or viewer modules.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

signal_names = [fmt_text(s.get("name"), f"Signal {i+1}") for i, s in enumerate(signals)]

selected_names = st.sidebar.multiselect(
    "Select Signals",
    signal_names,
    default=signal_names[: min(3, len(signal_names))],
)

selected_signals = [s for s in signals if fmt_text(s.get("name")) in selected_names]

if not selected_signals:
    st.markdown(
        """
        <div class="wm-empty">
            <h3 style="margin:0 0 8px 0; color:#0f172a;">No signals selected</h3>
            <div>Select at least one signal from the sidebar.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

metrics = [compute_phase_metrics(s) for s in selected_signals]
df = pd.DataFrame(metrics)

if df.empty:
    st.markdown(
        """
        <div class="wm-empty">
            <h3 style="margin:0 0 8px 0; color:#0f172a;">No valid phase data available</h3>
            <div>The selected signals do not contain usable waveform data.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

primary_name = selected_names[0]
machine, point, config = split_signal_name(primary_name)

meta = {
    "machine": machine,
    "point": point,
    "config": config,
    "rpm": fmt_num(df["RPM"].iloc[0], 0),
    "peak": fmt_num(df["1X Amplitude"].iloc[0], 3),
    "timestamp": fmt_text(df["timestamp"].iloc[0]),
}

render_phase_header(meta)

top_left, top_right = st.columns([1, 1])

with top_left:
    st.caption(f"{len(df)} signal(s) selected")

with top_right:
    export_df = df[["Signal", "1X Amplitude", "Phase", "Stability", "RPM"]].copy()
    export_df["1X Amplitude"] = export_df["1X Amplitude"].apply(lambda v: fmt_num(v, 3))
    export_df["Phase"] = export_df["Phase"].apply(lambda v: fmt_num(v, 1, "°"))
    export_df["RPM"] = export_df["RPM"].apply(lambda v: fmt_num(v, 0))
    png_bytes = build_png(export_df, meta)

    st.download_button(
        "Export PNG HD",
        data=png_bytes,
        file_name="phase_1x_dashboard.png",
        mime="image/png",
        use_container_width=True,
    )

st.markdown('<div class="wm-card"><div class="wm-card-title">Phase Summary</div>', unsafe_allow_html=True)

df_display = df[["Signal", "1X Amplitude", "Phase", "Stability", "RPM"]].copy()
df_display["1X Amplitude"] = df_display["1X Amplitude"].apply(lambda v: fmt_num(v, 3))
df_display["Phase"] = df_display["Phase"].apply(lambda v: fmt_num(v, 1, "°"))
df_display["RPM"] = df_display["RPM"].apply(lambda v: fmt_num(v, 0))

st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True,
    height=min(520, 80 + len(df_display) * 38),
)

st.markdown("</div>", unsafe_allow_html=True)