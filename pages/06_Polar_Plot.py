from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

from core.auth import require_login, render_user_menu


st.set_page_config(page_title="Watermelon System | Polar Plot", layout="wide")
require_login()
render_user_menu()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"


# ============================================================
# STYLE
# ============================================================
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

        .wm-page-title {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.08rem;
            letter-spacing: -0.02em;
        }

        .wm-page-subtitle {
            color: #64748b;
            font-size: 0.96rem;
            margin-bottom: 1rem;
        }

        .wm-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.86));
            border: 1px solid #dbe3ee;
            border-radius: 18px;
            padding: 14px 16px 14px 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            margin-bottom: 12px;
        }

        .wm-card-title {
            font-size: 1.02rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.18rem;
        }

        .wm-card-subtitle {
            color: #64748b;
            font-size: 0.90rem;
            margin-bottom: 0.35rem;
        }

        .wm-meta {
            color: #334155;
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .wm-chip-row {
            display:flex;
            flex-wrap:wrap;
            gap:8px;
            margin-top:10px;
        }

        .wm-chip {
            border:1px solid #dbe3ee;
            background:#f8fafc;
            border-radius:999px;
            padding:5px 10px;
            color:#334155;
            font-size:0.82rem;
            font-weight:600;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_page_style()


# ============================================================
# HELPERS
# ============================================================
def ensure_report_state() -> None:
    if "report_items" not in st.session_state:
        st.session_state["report_items"] = []


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


def get_logo_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


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


def circular_mean_deg(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if vals.empty:
        return float("nan")
    rad = np.deg2rad(vals.to_numpy() % 360.0)
    c = np.mean(np.cos(rad))
    s = np.mean(np.sin(rad))
    ang = np.rad2deg(np.arctan2(s, c))
    return float((ang + 360.0) % 360.0)


def circular_smooth_deg(phase_deg: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return phase_deg.astype(float).copy()
    rad = np.deg2rad(phase_deg.astype(float).to_numpy() % 360.0)
    c = pd.Series(np.cos(rad)).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    s = pd.Series(np.sin(rad)).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    out = np.rad2deg(np.arctan2(s, c))
    out = (out + 360.0) % 360.0
    return pd.Series(out, index=phase_deg.index)


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window is None or window < 2:
        return series.astype(float).copy()

    smoothed = series.astype(float).rolling(window=window, center=True, min_periods=1).mean()
    std = smoothed.std()
    mean = smoothed.mean()

    if pd.notna(std) and pd.notna(mean) and std > 0:
        smoothed = smoothed.clip(lower=mean - 3 * std, upper=mean + 3 * std)

    return smoothed


def nearest_row_for_speed(df: pd.DataFrame, speed_value: float) -> pd.Series:
    idx = int((df["speed"] - speed_value).abs().idxmin())
    return df.loc[idx]


# ============================================================
# CSV LOADER
# ============================================================
def read_polar_csv(file_obj) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame]:
    file_obj.seek(0)
    raw_bytes = file_obj.read()
    text = raw_bytes.decode("utf-8-sig", errors="replace") if isinstance(raw_bytes, bytes) else str(raw_bytes)

    lines = text.splitlines()
    if not lines:
        raise ValueError("Archivo vacío.")

    header_idx = None
    for i, line in enumerate(lines):
        if "Amp" in line and "Phase" in line and "Speed" in line and "Timestamp" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("No se encontró el encabezado real del CSV Polar.")

    meta_lines = lines[:header_idx]
    data_text = "\n".join(lines[header_idx:])

    meta: Dict[str, str] = {}
    for line in meta_lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) == 2:
            meta[parts[0]] = parts[1]

    df = pd.read_csv(io.StringIO(data_text), encoding="utf-8-sig")

    required = ["Amp", "Amp Status", "Phase", "Phase Status", "Speed", "Speed Status", "Timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    df["amp"] = pd.to_numeric(df["Amp"], errors="coerce")
    df["phase"] = pd.to_numeric(df["Phase"], errors="coerce")
    df["speed"] = pd.to_numeric(df["Speed"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.dropna(subset=["amp", "phase", "speed", "Timestamp"]).copy()
    df = df[
        df["Amp Status"].astype(str).str.strip().str.lower().eq("valid")
        & df["Phase Status"].astype(str).str.strip().str.lower().eq("valid")
        & df["Speed Status"].astype(str).str.strip().str.lower().eq("valid")
    ].copy()

    if df.empty:
        raise ValueError("No quedaron filas válidas después del filtrado.")

    raw_df = df.sort_values(["Timestamp", "speed"]).reset_index(drop=True)

    grouped_df = (
        raw_df.groupby("speed", as_index=False)
        .agg(
            amp=("amp", "median"),
            phase=("phase", lambda s: circular_mean_deg(s)),
            samples=("Timestamp", "size"),
            ts_min=("Timestamp", "min"),
            ts_max=("Timestamp", "max"),
        )
        .sort_values("speed", kind="stable")
        .reset_index(drop=True)
    )

    return meta, raw_df, grouped_df


# ============================================================
# POLAR ORIENTATION ENGINE
# ============================================================
def compute_probe_base_angle(axis_label: str, side_label: str, install_angle_deg: float) -> float:
    axis_label = str(axis_label).strip().upper()
    side_label = str(side_label).strip().capitalize()

    base = 0.0
    if axis_label == "X":
        base = 0.0 if side_label == "Right" else 180.0
    elif axis_label == "Y":
        base = 90.0 if side_label == "Right" else 270.0
    else:
        base = 0.0

    return (base + float(install_angle_deg)) % 360.0


def get_polar_axis_rotation_and_direction(
    axis_label: str,
    side_label: str,
    install_angle_deg: float,
    rotation_direction: str,
) -> Tuple[float, str, float]:
    probe_ref = compute_probe_base_angle(axis_label, side_label, install_angle_deg)
    axis_rotation = (90.0 - probe_ref) % 360.0
    angular_direction = "clockwise" if str(rotation_direction).upper() == "CCW" else "counterclockwise"
    return axis_rotation, angular_direction, probe_ref


def compute_polar_display_theta(
    phase_deg: pd.Series,
    axis_label: str,
    side_label: str,
    install_angle_deg: float,
    rotation_direction: str,
) -> pd.Series:
    return phase_deg.astype(float) % 360.0


# ============================================================
# API 684 HEURISTIC FOR POLAR
# ============================================================
def estimate_critical_speeds_api684_style(df: pd.DataFrame, max_count: int = 2) -> List[Dict[str, float]]:
    if df.empty or len(df) < 12:
        return []

    amp = df["amp"].astype(float).to_numpy()
    speed = df["speed"].astype(float).to_numpy()
    phase = df["phase_for_detection"].astype(float).to_numpy()

    candidates: List[Dict[str, float]] = []

    if find_peaks is not None:
        prominence = max(np.nanmax(amp) * 0.08, 0.12)
        distance = max(8, len(df) // 16)
        peaks, props = find_peaks(amp, prominence=prominence, distance=distance)

        for i, p in enumerate(peaks):
            left = max(0, p - 10)
            right = min(len(df) - 1, p + 10)

            amp_peak = float(amp[p])
            prom = float(props["prominences"][i])
            phase_delta = float(phase[right] - phase[left])

            if amp_peak < np.nanmax(amp) * 0.50:
                continue
            if abs(phase_delta) < 10.0:
                continue
            if amp_peak < np.nanmax(amp) * 0.85 and abs(phase_delta) < 20.0:
                continue

            candidates.append(
                {
                    "speed": float(speed[p]),
                    "amp": amp_peak,
                    "phase_delta": phase_delta,
                    "idx": int(p),
                    "prominence": prom,
                }
            )
    else:
        p = int(np.nanargmax(amp))
        left = max(0, p - 10)
        right = min(len(df) - 1, p + 10)
        candidates.append(
            {
                "speed": float(speed[p]),
                "amp": float(amp[p]),
                "phase_delta": float(phase[right] - phase[left]),
                "idx": int(p),
                "prominence": float(amp[p]),
            }
        )

    candidates = sorted(candidates, key=lambda x: (x["prominence"], x["amp"]), reverse=True)

    filtered = []
    for cand in candidates:
        if all(abs(cand["speed"] - kept["speed"]) > 120 for kept in filtered):
            filtered.append(cand)
        if len(filtered) >= max_count:
            break

    filtered = sorted(filtered, key=lambda x: x["speed"])
    return filtered


# ============================================================
# IN-CHART DECORATION
# ============================================================
def _draw_top_strip(
    fig: go.Figure,
    meta: Dict[str, str],
    logo_uri: Optional[str],
    df: pd.DataFrame,
    axis_label: str,
    side_label: str,
    install_angle_deg: float,
    rotation_direction: str,
) -> None:
    x0, x1 = 0.006, 0.994
    y0, y1 = 1.014, 1.104
    radius = 0.014

    fig.add_shape(
        type="path",
        xref="paper",
        yref="paper",
        path=rounded_rect_path(x0, y0, x1, y1, radius),
        line=dict(color="#cfd8e3", width=1.1),
        fillcolor="rgba(255,255,255,0.97)",
        layer="below",
    )

    y_text = (y0 + y1) / 2.0

    machine = meta.get("Machine Name", "")
    point = meta.get("Point Name", "")
    variable = meta.get("Variable", "")
    speed_unit = meta.get("Speed Unit", "rpm") or "rpm"

    if logo_uri:
        fig.add_layout_image(
            dict(
                source=logo_uri,
                xref="paper",
                yref="paper",
                x=0.014,
                y=y1 - 0.008,
                sizex=0.055,
                sizey=0.085,
                xanchor="left",
                yanchor="top",
                layer="above",
                sizing="contain",
                opacity=1.0,
            )
        )
        machine_x = 0.082
    else:
        machine_x = 0.020

    fig.add_annotation(
        xref="paper", yref="paper",
        x=machine_x, y=y_text,
        xanchor="left", yanchor="middle",
        text=f"<b>{machine}</b>",
        showarrow=False,
        font=dict(size=12.3, color="#111827"),
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.205, y=y_text,
        xanchor="left", yanchor="middle",
        text=point,
        showarrow=False,
        font=dict(size=11.7, color="#111827"),
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.355, y=y_text,
        xanchor="left", yanchor="middle",
        text=f"{variable}",
        showarrow=False,
        font=dict(size=11.5, color="#111827"),
    )

    orient_text = f"{axis_label} | {install_angle_deg:.0f}° {side_label} | Rotation {rotation_direction}"
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.47, y=y_text,
        xanchor="left", yanchor="middle",
        text=orient_text,
        showarrow=False,
        font=dict(size=11.3, color="#111827"),
    )

    dt_start = pd.to_datetime(df["ts_min"], errors="coerce").min()
    dt_end = pd.to_datetime(df["ts_max"], errors="coerce").max()
    dt_text = "—"
    if pd.notna(dt_start) and pd.notna(dt_end):
        dt_text = f"{dt_start.strftime('%Y-%m-%d %H:%M:%S')} → {dt_end.strftime('%Y-%m-%d %H:%M:%S')}"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.72, y=y_text,
        xanchor="left", yanchor="middle",
        text=dt_text,
        showarrow=False,
        font=dict(size=10.8, color="#111827"),
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.986, y=y_text,
        xanchor="right", yanchor="middle",
        text=f"{int(round(df['speed'].min()))} - {int(round(df['speed'].max()))} {speed_unit}",
        showarrow=False,
        font=dict(size=10.8, color="#111827"),
    )


def _draw_right_info_box(fig: go.Figure, rows: List[Tuple[str, str]]) -> None:
    panel_x0 = 0.805
    panel_x1 = 0.965
    panel_y1 = 0.915
    header_h = 0.033
    row_h = 0.050
    panel_h = header_h + len(rows) * row_h + 0.016
    panel_y0 = panel_y1 - panel_h

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path=rounded_rect_path(panel_x0, panel_y0, panel_x1, panel_y1, 0.012),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(255,255,255,0.74)",
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
        text="<b>Polar Information</b>",
        showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=11.1, color="#111827"),
    )

    current_top = panel_y1 - header_h - 0.008
    for title, value in rows:
        title_y = current_top - 0.003
        value_y = current_top - 0.026

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.026, y=title_y,
            xanchor="left", yanchor="top",
            text=f"<b>{title}</b>",
            showarrow=False, font=dict(size=10.2, color="#111827"), align="left",
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.026, y=value_y,
            xanchor="left", yanchor="top",
            text=value,
            showarrow=False, font=dict(size=9.9, color="#111827"), align="left",
        )

        current_top -= row_h


def build_info_rows(
    row_a: pd.Series,
    row_b: pd.Series,
    amp_unit: str,
    speed_unit: str,
    axis_label: str,
    side_label: str,
    install_angle_deg: float,
    rotation_direction: str,
    show_rpm_labels: bool,
    marker_stride: int,
    critical_speeds: List[Dict[str, float]],
) -> List[Tuple[str, str]]:
    rows = [
        ("Cursor A", f"{format_number(row_a['amp'],3)} {amp_unit} @ {int(round(row_a['speed']))} {speed_unit} | ∠{format_number(row_a['theta_display'],1)}°"),
        ("Cursor B", f"{format_number(row_b['amp'],3)} {amp_unit} @ {int(round(row_b['speed']))} {speed_unit} | ∠{format_number(row_b['theta_display'],1)}°"),
        ("Probe Orientation", f"{axis_label} | {install_angle_deg:.0f}° {side_label}"),
        ("Rotation", rotation_direction),
        ("RPM Labels", "Enabled" if show_rpm_labels else "Disabled"),
        ("Label Step", f"Every {marker_stride} points"),
    ]

    for i, cs in enumerate(critical_speeds, start=1):
        title = f"Critical Speed {i}" if i == 1 else f"Secondary Candidate {i}"
        rows.append((title, f"{int(round(cs['speed']))} {speed_unit} | {format_number(cs['amp'],3)} {amp_unit}"))
        rows.append((f"Phase Delta {i}", f"{format_number(cs['phase_delta'],1)}°"))

    return rows


# ============================================================
# FIGURE BUILD
# ============================================================
def build_probe_reference_overlay(fig: go.Figure, max_r: float) -> None:
    ref_r0 = max_r * 0.10
    ref_r1 = max_r * 0.98

    fig.add_trace(
        go.Scatterpolar(
            r=[ref_r0, ref_r1],
            theta=[0, 0],
            mode="lines",
            line=dict(color="#111827", width=2.2, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    body_r0 = max_r * 1.02
    body_r1 = max_r * 1.12
    fig.add_trace(
        go.Scatterpolar(
            r=[body_r0, body_r1],
            theta=[0, 0],
            mode="lines",
            line=dict(color="#111827", width=5.0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    tip_r = max_r * 1.145
    fig.add_trace(
        go.Scatterpolar(
            r=[tip_r],
            theta=[0],
            mode="markers",
            marker=dict(size=11, color="#111827", symbol="diamond"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    cone_r = [max_r * 1.00, max_r * 1.06, max_r * 1.00]
    cone_t = [-4, 0, 4]
    fig.add_trace(
        go.Scatterpolar(
            r=cone_r,
            theta=cone_t,
            mode="lines",
            line=dict(color="#111827", width=2.0),
            fill="toself",
            fillcolor="rgba(17,24,39,0.12)",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[max_r * 1.18],
            theta=[0],
            mode="text",
            text=["Probe"],
            textposition="top center",
            textfont=dict(size=10, color="#111827"),
            showlegend=False,
            hoverinfo="skip",
        )
    )


def build_polar_figure(
    df: pd.DataFrame,
    meta: Dict[str, str],
    row_a: pd.Series,
    row_b: pd.Series,
    logo_uri: Optional[str],
    show_info_box: bool,
    show_rpm_labels: bool,
    marker_stride: int,
    axis_label: str,
    side_label: str,
    install_angle_deg: float,
    rotation_direction: str,
    critical_speeds: List[Dict[str, float]],
) -> go.Figure:
    amp_unit = meta.get("Amp Unit", "") or ""
    speed_unit = meta.get("Speed Unit", "rpm") or "rpm"

    axis_rotation, angular_direction, _ = get_polar_axis_rotation_and_direction(
        axis_label=axis_label,
        side_label=side_label,
        install_angle_deg=install_angle_deg,
        rotation_direction=rotation_direction,
    )
    max_r = max(0.1, float(df["amp"].max()) * 1.18)

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=df["amp"],
            theta=df["theta_display"],
            mode="lines",
            line=dict(width=1.35, color="#5b9cf0"),
            hovertemplate=(
                f"Amplitude: %{{r:.3f}} {amp_unit}<br>"
                f"Phase Display: %{{theta:.1f}}°<br>"
                f"Speed: %{{customdata[0]:.0f}} {speed_unit}<extra></extra>"
            ),
            customdata=np.stack([df["speed"]], axis=1),
            showlegend=False,
            name="Polar Path",
        )
    )

    for row, color, name in [
        (row_a, "#efb08c", "Cursor A"),
        (row_b, "#7ac77b", "Cursor B"),
    ]:
        fig.add_trace(
            go.Scatterpolar(
                r=[row["amp"]],
                theta=[row["theta_display"]],
                mode="markers",
                marker=dict(size=10, color=color, line=dict(width=1.2, color="#ffffff")),
                name=name,
                showlegend=False,
                hovertemplate=(
                    f"{name}<br>"
                    f"Amplitude: %{{r:.3f}} {amp_unit}<br>"
                    f"Phase Display: %{{theta:.1f}}°<br>"
                    f"Speed: {int(round(row['speed']))} {speed_unit}<extra></extra>"
                ),
            )
        )

    if show_rpm_labels and len(df) > 0:
        idxs = list(range(0, len(df), max(1, marker_stride)))
        if idxs[-1] != len(df) - 1:
            idxs.append(len(df) - 1)

        fig.add_trace(
            go.Scatterpolar(
                r=df.iloc[idxs]["amp"],
                theta=df.iloc[idxs]["theta_display"],
                mode="text",
                text=[str(int(round(v))) for v in df.iloc[idxs]["speed"]],
                textfont=dict(size=9, color="#6b7280"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    cs_colors = ["#ef4444", "#f59e0b"]
    for idx, cs in enumerate(critical_speeds):
        color = cs_colors[idx % len(cs_colors)]
        cs_row = nearest_row_for_speed(df, cs["speed"])

        fig.add_trace(
            go.Scatterpolar(
                r=[cs_row["amp"]],
                theta=[cs_row["theta_display"]],
                mode="markers+text",
                marker=dict(size=9, color=color, symbol="diamond"),
                text=[f"CS{idx+1} {int(round(cs['speed']))}"],
                textposition="top center",
                textfont=dict(size=10, color=color),
                showlegend=False,
                hovertemplate=(
                    f"Critical Speed {idx+1}<br>"
                    f"Amplitude: %{{r:.3f}} {amp_unit}<br>"
                    f"Phase Display: %{{theta:.1f}}°<br>"
                    f"Speed: {int(round(cs['speed']))} {speed_unit}<extra></extra>"
                ),
            )
        )

    rows = build_info_rows(
        row_a=row_a,
        row_b=row_b,
        amp_unit=amp_unit,
        speed_unit=speed_unit,
        axis_label=axis_label,
        side_label=side_label,
        install_angle_deg=install_angle_deg,
        rotation_direction=rotation_direction,
        show_rpm_labels=show_rpm_labels,
        marker_stride=marker_stride,
        critical_speeds=critical_speeds,
    )

    domain_x = [0.0, 0.78] if show_info_box else [0.0, 1.0]

    fig.update_layout(
        polar=dict(
            domain=dict(x=domain_x, y=[0.05, 0.96]),
            bgcolor="#f8fafc",
            angularaxis=dict(
                rotation=axis_rotation,
                direction=angular_direction,
                tickfont=dict(size=12, color="#111827"),
                gridcolor="rgba(148, 163, 184, 0.18)",
                linecolor="#9ca3af",
                showline=True,
                ticks="outside",
            ),
            radialaxis=dict(
                range=[0, max_r],
                tickfont=dict(size=11, color="#111827"),
                gridcolor="rgba(148, 163, 184, 0.18)",
                linecolor="#9ca3af",
                showline=True,
                ticks="outside",
                angle=225,
            ),
        ),
        height=820,
        margin=dict(l=48, r=20, t=145, b=48),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f3f4f6",
        font=dict(color="#111827"),
        showlegend=False,
    )

    _draw_top_strip(
        fig=fig,
        meta=meta,
        logo_uri=logo_uri,
        df=df,
        axis_label=axis_label,
        side_label=side_label,
        install_angle_deg=install_angle_deg,
        rotation_direction=rotation_direction,
    )

    if show_info_box:
        _draw_right_info_box(fig, rows)

    build_probe_reference_overlay(fig, max_r)

    return fig


# ============================================================
# EXPORT / REPORT
# ============================================================
def _build_export_safe_figure(fig: go.Figure) -> go.Figure:
    return go.Figure(fig.to_dict())


def _scale_export_figure(export_fig: go.Figure) -> go.Figure:
    fig = go.Figure(export_fig)

    for trace in fig.data:
        tj = trace.to_plotly_json()
        mode = tj.get("mode", "") or ""

        if "lines" in mode:
            line = dict(tj.get("line", {}) or {})
            line["width"] = max(3.2, float(line.get("width", 1.0)) * 2.0)
            trace.line = line

        if "markers" in mode:
            marker = dict(tj.get("marker", {}) or {})
            marker["size"] = max(10, float(marker.get("size", 6)) * 1.5)
            trace.marker = marker

        if "text" in mode:
            textfont = dict(tj.get("textfont", {}) or {})
            textfont["size"] = max(16, int(float(textfont.get("size", 10)) * 1.8))
            trace.textfont = textfont

    fig.update_layout(
        width=4300,
        height=2400,
        margin=dict(l=110, r=80, t=320, b=120),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        font=dict(size=26, color="#111827"),
    )

    polar_cfg = dict(fig.layout.polar.to_plotly_json()) if getattr(fig.layout, "polar", None) is not None else {}
    domain_cfg = dict(polar_cfg.get("domain", {}) or {})
    current_x = domain_cfg.get("x", [0.0, 0.78])

    domain_cfg["x"] = [current_x[0], min(0.80, current_x[1])]
    domain_cfg["y"] = [0.06, 0.95]
    polar_cfg["domain"] = domain_cfg

    angular_cfg = dict(polar_cfg.get("angularaxis", {}) or {})
    angular_cfg["tickfont"] = dict(size=22, color="#111827")
    polar_cfg["angularaxis"] = angular_cfg

    radial_cfg = dict(polar_cfg.get("radialaxis", {}) or {})
    radial_cfg["tickfont"] = dict(size=20, color="#111827")
    polar_cfg["radialaxis"] = radial_cfg

    fig.update_layout(polar=polar_cfg)

    for ann in fig.layout.annotations or []:
        if ann.font is not None:
            ann.font.size = max(20, int((ann.font.size or 12) * 1.75))

    for img in fig.layout.images or []:
        sx = getattr(img, "sizex", None)
        sy = getattr(img, "sizey", None)
        if sx is not None:
            img.sizex = sx * 1.10
        if sy is not None:
            img.sizey = sy * 1.10

    return fig


def build_export_png_bytes(fig: go.Figure) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        export_fig = _build_export_safe_figure(fig)
        export_fig = _scale_export_figure(export_fig)
        return export_fig.to_image(format="png", width=4300, height=2200, scale=2), None
    except Exception as e:
        return None, str(e)


def queue_polar_to_report(meta: Dict[str, str], fig: go.Figure, title: str) -> None:
    ensure_report_state()
    st.session_state.report_items.append(
        {
            "id": f"report-polar-{meta.get('Machine Name','')}-{meta.get('Point Name','')}-{title}",
            "type": "polar",
            "title": title,
            "notes": "",
            "signal_id": meta.get("Point Name", ""),
            "figure": go.Figure(fig),
            "machine": meta.get("Machine Name", ""),
            "point": meta.get("Point Name", ""),
            "variable": meta.get("Variable", ""),
            "timestamp": "",
        }
    )


# ============================================================
# MULTI-FILE LOADER
# ============================================================
def uploaded_file_label(file_obj) -> str:
    return Path(getattr(file_obj, "name", "Polar.csv")).name


def uploaded_file_stem(file_obj) -> str:
    return Path(getattr(file_obj, "name", "Polar.csv")).stem


def parse_uploaded_polar_files(files: List[Any]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    parsed_items: List[Dict[str, Any]] = []
    failed_items: List[Tuple[str, str]] = []

    for file_obj in files:
        try:
            meta, raw_df, grouped_df = read_polar_csv(file_obj)
            label = uploaded_file_label(file_obj)
            machine = meta.get("Machine Name", "-")
            point = meta.get("Point Name", label)
            item_id = f"{label}::{machine}::{point}"

            parsed_items.append(
                {
                    "id": item_id,
                    "label": label,
                    "file_name": label,
                    "file_stem": uploaded_file_stem(file_obj),
                    "meta": meta,
                    "raw_df": raw_df,
                    "grouped_df": grouped_df,
                    "machine": machine,
                    "point": point,
                    "variable": meta.get("Variable", "-"),
                }
            )
        except Exception as e:
            failed_items.append((uploaded_file_label(file_obj), str(e)))

    return parsed_items, failed_items


# ============================================================
# PER-PANEL ORIENTATION STATE
# ============================================================
def get_panel_orientation(item_id: str) -> Dict[str, Any]:
    key = f"wm_polar_orientation::{item_id}"
    if key not in st.session_state:
        st.session_state[key] = {
            "axis_label": "Y",
            "side_label": "Left",
            "install_angle_deg": 45,
            "rotation_direction": "CCW",
        }
    return st.session_state[key]


# ============================================================
# PANEL RENDER
# ============================================================
def render_polar_panel(
    item: Dict[str, Any],
    panel_index: int,
    *,
    logo_uri: Optional[str],
    smooth_window: int,
    amp_smooth_window: int,
    show_info_box: bool,
    show_rpm_labels: bool,
    marker_stride: int,
    detect_cs: bool,
    max_critical_speeds: int,
) -> None:
    meta = item["meta"]
    raw_df = item["raw_df"]
    grouped_df = item["grouped_df"]
    orient = get_panel_orientation(item["id"])

    axis_label = orient["axis_label"]
    side_label = orient["side_label"]
    install_angle_deg = float(orient["install_angle_deg"])
    rotation_direction = orient["rotation_direction"]

    plot_df = grouped_df.copy()
    plot_df["amp"] = smooth_series(plot_df["amp"], amp_smooth_window)
    plot_df["phase_smoothed"] = circular_smooth_deg(plot_df["phase"], smooth_window) % 360.0
    plot_df["theta_display"] = compute_polar_display_theta(
        phase_deg=plot_df["phase_smoothed"],
        axis_label=axis_label,
        side_label=side_label,
        install_angle_deg=install_angle_deg,
        rotation_direction=rotation_direction,
    )

    phase_internal = np.rad2deg(np.unwrap(np.deg2rad(plot_df["phase_smoothed"].to_numpy())))
    plot_df["phase_for_detection"] = phase_internal

    speed_min = int(plot_df["speed"].min())
    speed_max = int(plot_df["speed"].max())

    cursor_col1, cursor_col2 = st.columns(2)
    with cursor_col1:
        cursor_a_speed = st.slider(
            f"Cursor A (RPM) · Panel {panel_index + 1}",
            speed_min,
            speed_max,
            speed_min,
            key=f"polar_cursor_a_{panel_index}_{item['id']}",
        )
    with cursor_col2:
        cursor_b_speed = st.slider(
            f"Cursor B (RPM) · Panel {panel_index + 1}",
            speed_min,
            speed_max,
            speed_max,
            key=f"polar_cursor_b_{panel_index}_{item['id']}",
        )

    row_a = nearest_row_for_speed(plot_df, cursor_a_speed)
    row_b = nearest_row_for_speed(plot_df, cursor_b_speed)

    critical_speeds: List[Dict[str, float]] = []
    if detect_cs:
        critical_speeds = estimate_critical_speeds_api684_style(plot_df, max_count=max_critical_speeds)

    machine = meta.get("Machine Name", "-")
    point = meta.get("Point Name", "-")
    variable = meta.get("Variable", "-")
    speed_unit = meta.get("Speed Unit", "rpm")
    amp_unit = meta.get("Amp Unit", "")

    st.markdown(
        f"""
        <div class="wm-card">
            <div class="wm-card-title">Polar {panel_index + 1} · {machine} · {point}</div>
            <div class="wm-card-subtitle">Dynamic polar view</div>
            <div class="wm-meta">
                Variable: <b>{variable}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Orientation: <b>{axis_label} | {install_angle_deg:.0f}° {side_label}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Rotation: <b>{rotation_direction}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Speed Range: <b>{int(plot_df['speed'].min())} - {int(plot_df['speed'].max())} {speed_unit}</b>
            </div>
            <div class="wm-chip-row">
                <div class="wm-chip">File: {item["file_name"]}</div>
                <div class="wm-chip">Raw rows: {len(raw_df):,}</div>
                <div class="wm-chip">Grouped points: {len(plot_df):,}</div>
                <div class="wm-chip">Phase smoothing: {smooth_window}</div>
                <div class="wm-chip">Amplitude smoothing: {amp_smooth_window}</div>
                <div class="wm-chip">Critical speeds: {len(critical_speeds)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = build_polar_figure(
        df=plot_df,
        meta=meta,
        row_a=row_a,
        row_b=row_b,
        logo_uri=logo_uri,
        show_info_box=show_info_box,
        show_rpm_labels=show_rpm_labels,
        marker_stride=marker_stride,
        axis_label=axis_label,
        side_label=side_label,
        install_angle_deg=install_angle_deg,
        rotation_direction=rotation_direction,
        critical_speeds=critical_speeds,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False},
        key=f"wm_polar_plot_{panel_index}_{item['id']}",
    )

    title = f"Polar {panel_index + 1} — {machine} — {point}"

    export_state_key = (
        f"polar::{item['id']}::{panel_index}::{variable}::{smooth_window}::{amp_smooth_window}::"
        f"{show_info_box}::{show_rpm_labels}::{marker_stride}::{axis_label}::{side_label}::"
        f"{install_angle_deg}::{rotation_direction}::{detect_cs}::{max_critical_speeds}::"
        f"{cursor_a_speed}::{cursor_b_speed}"
    )

    if export_state_key not in st.session_state.wm_polar_export_store:
        st.session_state.wm_polar_export_store[export_state_key] = {"png_bytes": None, "error": None}

    left_pad, col_export1, col_export2, col_report, right_pad = st.columns([2.0, 1.2, 1.2, 1.2, 2.0])

    with col_export1:
        if st.button("Prepare PNG HD", key=f"prepare_polar_png_{export_state_key}", use_container_width=True):
            with st.spinner("Generating HD export..."):
                png_bytes, export_error = build_export_png_bytes(fig)
                st.session_state.wm_polar_export_store[export_state_key]["png_bytes"] = png_bytes
                st.session_state.wm_polar_export_store[export_state_key]["error"] = export_error

    with col_export2:
        png_bytes = st.session_state.wm_polar_export_store[export_state_key]["png_bytes"]
        if png_bytes is not None:
            st.download_button(
                "Download PNG HD",
                data=png_bytes,
                file_name=f"{item['file_stem']}_polar_hd.png",
                mime="image/png",
                key=f"download_polar_png_{export_state_key}",
                use_container_width=True,
            )
        else:
            st.button(
                "Download PNG HD",
                disabled=True,
                key=f"download_polar_disabled_{export_state_key}",
                use_container_width=True,
            )

    with col_report:
        if st.button("Enviar a Reporte", key=f"report_polar_{export_state_key}", use_container_width=True):
            queue_polar_to_report(meta, fig, title)
            st.success("Polar enviado al reporte.")


# ============================================================
# MAIN
# ============================================================
if "wm_polar_export_store" not in st.session_state:
    st.session_state.wm_polar_export_store = {}
if "wm_polar_selected_ids" not in st.session_state:
    st.session_state.wm_polar_selected_ids = []
ensure_report_state()

st.markdown('<div class="wm-page-title">Polar Plot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wm-page-subtitle">Dynamic polar trajectory from amplitude, phase and speed.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Upload Polar CSV")
    uploaded_files = st.file_uploader(
        "Upload one or more Polar CSV",
        type=["csv"],
        accept_multiple_files=True,
    )

if not uploaded_files:
    st.info("Carga uno o varios archivos CSV Polar para visualizar trayectorias dinámicas.")
    st.stop()

parsed_items, failed_items = parse_uploaded_polar_files(uploaded_files)

if failed_items:
    for file_name, error_text in failed_items:
        st.warning(f"No pude leer {file_name}: {error_text}")

if not parsed_items:
    st.error("No se pudo cargar ningún archivo Polar válido.")
    st.stop()

id_to_item = {item["id"]: item for item in parsed_items}
label_to_id = {
    f"{item['machine']} · {item['point']} · {item['file_name']}": item["id"]
    for item in parsed_items
}
selection_labels = list(label_to_id.keys())

valid_ids = set(id_to_item.keys())
current_ids = [sid for sid in st.session_state.wm_polar_selected_ids if sid in valid_ids]
if not current_ids:
    current_ids = [parsed_items[0]["id"]]
    st.session_state.wm_polar_selected_ids = current_ids

default_labels = [label for label, sid in label_to_id.items() if sid in current_ids]

with st.sidebar:
    st.markdown("### Polar Selection")
    selected_labels = st.multiselect(
        "Polars to display",
        options=selection_labels,
        default=default_labels,
    )
    st.session_state.wm_polar_selected_ids = [label_to_id[label] for label in selected_labels if label in label_to_id]

    selected_ids_for_sidebar = [sid for sid in st.session_state.wm_polar_selected_ids if sid in id_to_item]

    if selected_ids_for_sidebar:
        st.markdown("### Probe Orientation by Polar")
        for panel_index, sid in enumerate(selected_ids_for_sidebar, start=1):
            item = id_to_item[sid]
            orient_key = f"wm_polar_orientation::{sid}"
            current = get_panel_orientation(sid)

            with st.expander(f"Polar {panel_index} · {item['point']}", expanded=(panel_index == 1)):
                axis_value = st.selectbox(
                    "Probe Axis",
                    ["X", "Y"],
                    index=0 if current["axis_label"] == "X" else 1,
                    key=f"{orient_key}::axis",
                )
                side_value = st.selectbox(
                    "Probe Side",
                    ["Right", "Left"],
                    index=0 if current["side_label"] == "Right" else 1,
                    key=f"{orient_key}::side",
                )
                angle_value = st.slider(
                    "Probe Installation Angle",
                    0,
                    90,
                    int(current["install_angle_deg"]),
                    step=5,
                    key=f"{orient_key}::angle",
                )
                rotation_value = st.selectbox(
                    "Rotation Direction",
                    ["CCW", "CW"],
                    index=0 if current["rotation_direction"] == "CCW" else 1,
                    key=f"{orient_key}::rotation",
                )

                st.session_state[orient_key] = {
                    "axis_label": axis_value,
                    "side_label": side_value,
                    "install_angle_deg": angle_value,
                    "rotation_direction": rotation_value,
                }

    st.markdown("### Polar Controls")
    smooth_window = st.slider("Circular phase smoothing", 1, 11, 3, step=2)
    amp_smooth_window = st.slider("Amplitude smoothing", 1, 11, 3, step=2)
    show_info_box = st.checkbox("Show Polar Information", value=True)
    show_rpm_labels = st.checkbox("Show RPM labels", value=True)
    marker_stride = st.slider("RPM label step", 10, 150, 45, step=5)

    st.markdown("### Critical Speed Detection")
    detect_cs = st.checkbox("Estimate critical speeds (API-684 heuristic)", value=True)
    max_critical_speeds = st.selectbox("Max critical speeds", [1, 2], index=1)

selected_ids = [sid for sid in st.session_state.wm_polar_selected_ids if sid in id_to_item]

if not selected_ids:
    st.info("Selecciona uno o más polares en la barra lateral.")
    st.stop()

selected_items = [id_to_item[sid] for sid in selected_ids]
logo_uri = get_logo_data_uri(LOGO_PATH)

for panel_index, item in enumerate(selected_items):
    render_polar_panel(
        item=item,
        panel_index=panel_index,
        logo_uri=logo_uri,
        smooth_window=smooth_window,
        amp_smooth_window=amp_smooth_window,
        show_info_box=show_info_box,
        show_rpm_labels=show_rpm_labels,
        marker_stride=marker_stride,
        detect_cs=detect_cs,
        max_critical_speeds=max_critical_speeds,
    )

    if panel_index < len(selected_items) - 1:
        st.markdown("---")
