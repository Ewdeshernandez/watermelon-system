from __future__ import annotations

import base64
import html
import io
import math
import re
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
from core.diagnostics import build_polar_text_diagnostics, format_number, get_semaforo_status
from core.module_patterns import export_report_row, helper_card, panel_card
from core.ui_theme import (
    apply_watermelon_page_style,
    draw_info_box,
    draw_top_strip,
    page_header,
)


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Watermelon System | Polar Plot", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"

apply_watermelon_page_style()


# ============================================================
# POLAR FILE PERSISTENCE
# ============================================================
POLAR_UPLOAD_FILES_KEY = "wm_polar_upload_files"

class PolarPersistedUploadedFile:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def read(self):
        return self._data

    def getvalue(self):
        return self._data

    def seek(self, pos):
        return None


def set_polar_persisted_files(files):
    packed = []
    for f in files or []:
        try:
            data = f.getvalue()
        except Exception:
            try:
                f.seek(0)
            except Exception:
                pass
            data = f.read()

        packed.append({
            "name": getattr(f, "name", "Polar.csv"),
            "data": data,
        })

    st.session_state[POLAR_UPLOAD_FILES_KEY] = packed


def get_polar_persisted_files():
    return [
        PolarPersistedUploadedFile(item["name"], item["data"])
        for item in st.session_state.get(POLAR_UPLOAD_FILES_KEY, [])
    ]


def clear_polar_persisted_files():
    st.session_state.pop(POLAR_UPLOAD_FILES_KEY, None)


# ============================================================
# HELPERS
# ============================================================
def ensure_report_state() -> None:
    if "report_items" not in st.session_state:
        st.session_state["report_items"] = []


def get_logo_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


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


def polar_health_status(
    critical_speeds: List[Dict[str, float]],
    amp_series: pd.Series,
) -> Tuple[str, str, Dict[str, float]]:
    max_amp = float(amp_series.max()) if len(amp_series) else 0.0
    candidate_count = len(critical_speeds)

    if candidate_count == 0:
        score = 15.0
    else:
        dominant_amp = max(float(cs["amp"]) for cs in critical_speeds)
        phase_delta = max(abs(float(cs["phase_delta"])) for cs in critical_speeds)
        score = min(100.0, dominant_amp * 10.0 + abs(phase_delta) * 0.35)

    status, color = get_semaforo_status(score, safe_limit=35.0, warning_limit=70.0)
    return status, color, {
        "score": score,
        "max_amp": max_amp,
        "candidate_count": candidate_count,
    }


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
# FIGURE
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
    semaforo_status: str,
    semaforo_color: str,
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
            line=dict(width=1.9, color="#5b9cf0"),
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

    dt_start = pd.to_datetime(df["ts_min"], errors="coerce").min()
    dt_end = pd.to_datetime(df["ts_max"], errors="coerce").max()
    dt_text = "—"
    if pd.notna(dt_start) and pd.notna(dt_end):
        dt_text = f"{dt_start.strftime('%Y-%m-%d %H:%M:%S')} → {dt_end.strftime('%Y-%m-%d %H:%M:%S')}"

    draw_top_strip(
        fig=fig,
        machine=meta.get("Machine Name", ""),
        point_text=meta.get("Point Name", ""),
        variable=f"{meta.get('Variable', '-')} | {axis_label} | {install_angle_deg:.0f}° {side_label} | Rotation {rotation_direction}",
        dt_text=dt_text,
        rpm_text=f"{int(round(df['speed'].min()))} - {int(round(df['speed'].max()))} {speed_unit}",
        logo_uri=logo_uri,
    )

    if show_info_box:
        rows = [
            ("Cursor A", f"{format_number(row_a['amp'],3)} {amp_unit} @ {int(round(row_a['speed']))} {speed_unit} | ∠{format_number(row_a['theta_display'],1)}°"),
            ("Cursor B", f"{format_number(row_b['amp'],3)} {amp_unit} @ {int(round(row_b['speed']))} {speed_unit} | ∠{format_number(row_b['theta_display'],1)}°"),
            ("Probe Orientation", f"{axis_label} | {install_angle_deg:.0f}° {side_label}"),
            ("Rotation", rotation_direction),
            ("RPM Labels", "Enabled" if show_rpm_labels else "Disabled"),
            ("Label Step", f"Every {marker_stride} points"),
            ("Status", f"<span style='color:{semaforo_color};'><b>{semaforo_status}</b></span>"),
        ]

        for i, cs in enumerate(critical_speeds, start=1):
            title = f"Critical Speed {i}" if i == 1 else f"Secondary Candidate {i}"
            rows.append((title, f"{int(round(cs['speed']))} {speed_unit} | {format_number(cs['amp'],3)} {amp_unit}"))
            rows.append((f"Phase Delta {i}", f"{format_number(cs['phase_delta'],1)}°"))

        draw_info_box(fig=fig, title="Polar Information", rows=rows)

    fig.update_layout(
        polar=dict(
            domain=dict(x=[0.0, 0.78] if show_info_box else [0.0, 1.0], y=[0.05, 0.96]),
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
        height=2900,
        margin=dict(l=110, r=80, t=320, b=760),
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


def _build_polar_report_notes(text_diag: Dict[str, str]) -> str:
    headline = str(text_diag.get("headline", "") or "").strip()
    detail = str(text_diag.get("detail", "") or "").strip()
    action = str(text_diag.get("action", "") or "").strip()

    def clean_text(value: str) -> str:
        value = str(value or "")
        value = value.replace("\\n", "\n")
        value = value.replace("\r", "")
        value = re.sub(r"\n{3,}", "\n\n", value)
        value = value.replace("Se recomienda se recomienda:", "Se recomienda:")
        value = value.replace("Se recomienda: Se recomienda:", "Se recomienda:")
        return value.strip()

    headline = clean_text(headline)
    detail = clean_text(detail)
    action = clean_text(action)

    blocks: List[str] = []
    if headline:
        blocks.append(headline)
    if detail:
        blocks.append(detail)

    if action:
        action_clean = action
        action_clean = re.sub(r"^Se recomienda:\s*", "", action_clean, flags=re.IGNORECASE)
        action_clean = action_clean.strip()
        if action_clean:
            blocks.append("Se recomienda:\n" + action_clean)

    return "\n\n".join([b for b in blocks if b]).strip()


def _add_export_diagnostic_footer(fig: go.Figure, text_diag: Dict[str, str]) -> go.Figure:
    headline = str(text_diag.get("headline", "") or "").strip()
    detail = str(text_diag.get("detail", "") or "").strip()
    action = str(text_diag.get("action", "") or "").strip()

    if not any([headline, detail, action]):
        return go.Figure(fig)

    export_fig = go.Figure(fig)

    current_annotations = list(export_fig.layout.annotations) if export_fig.layout.annotations else []
    current_shapes = list(export_fig.layout.shapes) if export_fig.layout.shapes else []

    footer_y0 = -0.285
    footer_y1 = -0.035

    current_shapes.extend(
        [
            dict(
                type="line",
                xref="paper",
                yref="paper",
                x0=0.03,
                x1=0.97,
                y0=-0.008,
                y1=-0.008,
                line=dict(color="rgba(148,163,184,0.55)", width=2),
            ),
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.03,
                x1=0.97,
                y0=footer_y0,
                y1=footer_y1,
                line=dict(color="rgba(148,163,184,0.55)", width=2),
                fillcolor="rgba(255,255,255,0.98)",
                layer="below",
            ),
        ]
    )

    summary_html = (
        f"<b>{html.escape(headline)}</b><br><br>"
        f"<b>Detail:</b> {html.escape(detail)}<br><br>"
        f"<b>Action:</b> {html.escape(action)}"
    )

    current_annotations.extend(
        [
            dict(
                xref="paper",
                yref="paper",
                x=0.05,
                y=-0.055,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                align="left",
                text="<b>DIAGNOSTIC SUMMARY</b>",
                font=dict(size=24, color="#0f172a"),
            ),
            dict(
                xref="paper",
                yref="paper",
                x=0.05,
                y=-0.112,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                align="left",
                text=summary_html,
                font=dict(size=20, color="#111827"),
            ),
        ]
    )

    export_fig.update_layout(
        annotations=current_annotations,
        shapes=current_shapes,
    )
    return export_fig


def build_export_png_bytes(fig: go.Figure, text_diag: Dict[str, str]) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        export_fig = _build_export_safe_figure(fig)
        export_fig = _scale_export_figure(export_fig)
        # Imagen limpia: el diagnóstico va debajo en el reporte, no incrustado en el PNG.
        return export_fig.to_image(format="png", width=4300, height=2900, scale=2), None
    except Exception as e:
        return None, str(e)


def queue_polar_to_report(
    meta: Dict[str, str],
    fig: go.Figure,
    title: str,
    text_diag: Dict[str, str],
    image_bytes: Optional[bytes] = None,
) -> None:
    ensure_report_state()
    st.session_state.report_items.append(
        {
            "id": f"report-polar-{meta.get('Machine Name','')}-{meta.get('Point Name','')}-{title}",
            "type": "polar",
            "title": title,
            "notes": _build_polar_report_notes(text_diag),
            "signal_id": meta.get("Point Name", ""),
            "image_bytes": image_bytes,
            "machine": meta.get("Machine Name", ""),
            "point": meta.get("Point Name", ""),
            "variable": meta.get("Variable", ""),
            "timestamp": "",
        }
    )



# ============================================================
# POLAR PRO OVERRIDES - DIAGNOSTIC + CLEAN HD EXPORT
# ============================================================
def build_polar_text_diagnostics(
    status: str,
    critical_speeds: List[Dict[str, float]],
    max_amp: float,
) -> Dict[str, str]:
    status_up = str(status or "").upper()
    max_amp = float(max_amp or 0.0)

    if critical_speeds:
        dominant = critical_speeds[0]
        cs_speed = float(dominant.get("speed", 0.0) or 0.0)
        cs_amp = float(dominant.get("amp", 0.0) or 0.0)
        phase_delta = float(dominant.get("phase_delta", 0.0) or 0.0)

        if status_up == "DANGER":
            headline = f"Respuesta polar severa compatible con amplificación dinámica cerca de {cs_speed:.0f} rpm"
            detail = (
                f"La trayectoria polar evidencia una respuesta dinámica significativa alrededor de {cs_speed:.0f} rpm, "
                f"con amplitud aproximada de {cs_amp:.3f} y variación de fase de {phase_delta:.1f}°. "
                f"La combinación de incremento de amplitud y cambio de fase sugiere proximidad a una velocidad crítica, "
                f"pérdida de margen dinámico o cambio relevante de rigidez/amortiguamiento del sistema rotor-soporte.\n\n"
                f"Desde el punto de vista de dinámica de rotores, esta condición debe correlacionarse con Bode, órbitas, "
                f"forma de onda, shaft centerline y condiciones reales de carga."
            )
            action = (
                "Se recomienda como acción prioritaria:\n"
                "- Correlacionar el pico polar con Bode de amplitud y fase\n"
                "- Confirmar repetibilidad durante arranque/parada\n"
                "- Verificar alineación, rigidez de soporte, balance y condición de cojinetes\n"
                "- Revisar el cambio de fase alrededor del régimen identificado\n"
                "- Evitar operación sostenida cerca del régimen crítico hasta completar evaluación"
            )
        elif status_up == "WARNING":
            headline = f"Respuesta polar con indicios de amplificación dinámica cerca de {cs_speed:.0f} rpm"
            detail = (
                f"La trayectoria polar muestra una zona de respuesta relevante alrededor de {cs_speed:.0f} rpm, "
                f"con amplitud aproximada de {cs_amp:.3f} y cambio de fase de {phase_delta:.1f}°. "
                f"El comportamiento es consistente con amplificación dinámica moderada, sin evidencia suficiente para clasificarla como severa.\n\n"
                f"Desde el enfoque de análisis de vibraciones, esta condición debe mantenerse bajo seguimiento, especialmente si el pico se repite "
                f"en corridas posteriores o si se acompaña de incremento en 1X, cambio de fase o alteración de órbita."
            )
            action = (
                "Se recomienda:\n"
                "- Comparar contra corridas históricas y condición base\n"
                "- Validar la respuesta con Bode y espectro 1X\n"
                "- Confirmar si existe tendencia creciente de amplitud\n"
                "- Mantener seguimiento durante próximos arranques/paradas"
            )
        else:
            headline = f"Respuesta polar controlada con candidato dinámico cerca de {cs_speed:.0f} rpm"
            detail = (
                f"Se identifica un candidato dinámico alrededor de {cs_speed:.0f} rpm, con amplitud aproximada de {cs_amp:.3f} "
                f"y cambio de fase de {phase_delta:.1f}°. La trayectoria polar no evidencia una respuesta severa en esta condición.\n\n"
                f"El comportamiento es compatible con operación estable, aunque el punto identificado debe conservarse como referencia para comparación futura."
            )
            action = (
                "Se recomienda:\n"
                "- Mantener la corrida como línea base\n"
                "- Comparar con futuras trayectorias polares\n"
                "- Correlacionar con Bode, órbita y tendencia de amplitud 1X"
            )
    else:
        headline = "Respuesta polar sin velocidad crítica dominante claramente identificada"
        detail = (
            f"La trayectoria polar presenta amplitud máxima de {max_amp:.3f} y no muestra un candidato claro de velocidad crítica bajo la heurística aplicada. "
            f"La condición debe correlacionarse con Bode, espectro, órbita y variables operativas antes de concluir el mecanismo dominante."
        )
        action = (
            "Se recomienda:\n"
            "- Mantener seguimiento histórico\n"
            "- Comparar contra futuras corridas\n"
            "- Validar con Bode, espectro y órbita si cambia la condición"
        )

    return {"headline": headline, "detail": detail, "action": action}


def _scale_export_figure(export_fig: go.Figure) -> go.Figure:
    fig = go.Figure(export_fig)

    fig.update_layout(
        width=4300,
        height=2450,
        margin=dict(l=60, r=50, t=220, b=120),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        font=dict(size=25, color="#111827"),
    )

    polar_cfg = dict(fig.layout.polar.to_plotly_json()) if getattr(fig.layout, "polar", None) is not None else {}
    domain_cfg = dict(polar_cfg.get("domain", {}) or {})
    domain_cfg["x"] = [0.01, 0.86]
    domain_cfg["y"] = [0.02, 0.98]
    polar_cfg["domain"] = domain_cfg

    angular_cfg = dict(polar_cfg.get("angularaxis", {}) or {})
    angular_cfg["tickfont"] = dict(size=22, color="#111827")
    angular_cfg["gridcolor"] = "rgba(148, 163, 184, 0.18)"
    polar_cfg["angularaxis"] = angular_cfg

    radial_cfg = dict(polar_cfg.get("radialaxis", {}) or {})
    radial_cfg["tickfont"] = dict(size=20, color="#111827")
    radial_cfg["gridcolor"] = "rgba(148, 163, 184, 0.18)"
    polar_cfg["radialaxis"] = radial_cfg

    fig.update_layout(polar=polar_cfg)

    for trace in fig.data:
        tj = trace.to_plotly_json()
        mode = tj.get("mode", "") or ""
        if "lines" in mode and hasattr(trace, "line"):
            line = dict(tj.get("line", {}) or {})
            line["width"] = max(3.0, float(line.get("width", 1.0)) * 1.8)
            trace.line = line
        if "markers" in mode and hasattr(trace, "marker"):
            marker = dict(tj.get("marker", {}) or {})
            marker["size"] = max(9, float(marker.get("size", 6)) * 1.35)
            trace.marker = marker
        if "text" in mode:
            textfont = dict(tj.get("textfont", {}) or {})
            textfont["size"] = max(15, int(float(textfont.get("size", 10)) * 1.6))
            trace.textfont = textfont

    return fig


def build_export_png_bytes(fig: go.Figure, text_diag: Dict[str, str]) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        export_fig = _build_export_safe_figure(fig)
        export_fig = _scale_export_figure(export_fig)
        return export_fig.to_image(format="png", width=4300, height=2450, scale=2), None
    except Exception as e:
        return None, str(e)



# ============================================================
# POLAR PRO MODAL DIAGNOSTICS + COMPARISON
# ============================================================
def build_polar_text_diagnostics(
    status: str,
    critical_speeds: List[Dict[str, float]],
    max_amp: float,
) -> Dict[str, str]:
    status_up = str(status or "").upper()
    max_amp = float(max_amp or 0.0)

    if critical_speeds:
        dominant = critical_speeds[0]
        cs_speed = float(dominant.get("speed", 0.0) or 0.0)
        cs_amp = float(dominant.get("amp", 0.0) or 0.0)
        phase_delta = float(dominant.get("phase_delta", 0.0) or 0.0)

        if abs(phase_delta) >= 45.0:
            modal_txt = (
                "El cambio de fase es suficientemente representativo para sospechar transición modal marcada. "
                "Antes de la velocidad crítica el rotor tiende a comportarse con respuesta más rígida; después del paso por la zona modal, "
                "la respuesta se vuelve más flexible y la fase evidencia el cambio de relación entre fuerza excitadora y desplazamiento."
            )
        elif abs(phase_delta) >= 15.0:
            modal_txt = (
                "El cambio de fase es moderado y sugiere aproximación a una zona de amplificación dinámica. "
                "La respuesta aún no confirma por sí sola un paso crítico plenamente desarrollado, pero sí muestra una modificación de rigidez dinámica aparente."
            )
        else:
            modal_txt = (
                "El cambio de fase es bajo, por lo que el punto identificado debe tratarse como candidato dinámico y no como velocidad crítica confirmada. "
                "La confirmación requiere correlación con Bode, fase, órbita y repetibilidad entre corridas."
            )

        if status_up == "DANGER":
            headline = f"Respuesta polar severa asociada a posible velocidad crítica cerca de {cs_speed:.0f} rpm"
        elif status_up == "WARNING":
            headline = f"Respuesta polar con indicios de amplificación dinámica cerca de {cs_speed:.0f} rpm"
        else:
            headline = f"Respuesta polar controlada con candidato modal cerca de {cs_speed:.0f} rpm"

        detail = (
            f"La trayectoria polar identifica una zona de interés alrededor de {cs_speed:.0f} rpm, con amplitud aproximada de {cs_amp:.3f} "
            f"y variación de fase de {phase_delta:.1f}°. Esta condición es compatible con una posible aproximación a velocidad crítica o forma modal del sistema rotor-soporte.\\n\\n"
            f"{modal_txt}\\n\\n"
            f"Desde el punto de vista de análisis rotodinámico, la interpretación debe enfocarse en la relación entre amplitud, fase y velocidad. "
            f"Un incremento de amplitud acompañado por cambio de fase consistente puede indicar paso por una forma modal; si la amplitud aumenta sin cambio de fase suficiente, "
            f"la condición puede estar más asociada a desbalance, excentricidad, respuesta forzada o cambios operativos."
        )

        action = (
            "Correlacionar la trayectoria polar con Bode de amplitud y fase.\\n"
            "Verificar repetibilidad de la zona modal entre arranques/paradas.\\n"
            "Comparar contra órbitas filtradas 1X y shaft centerline.\\n"
            "Confirmar si el cambio de fase ocurre antes, durante o después del máximo de amplitud.\\n"
            "Validar condiciones de balance, alineación, rigidez de soporte, lubricación y carga."
        )
    else:
        headline = "Respuesta polar sin velocidad crítica dominante claramente identificada"
        detail = (
            f"La trayectoria polar presenta amplitud máxima de {max_amp:.3f}, sin un candidato modal dominante bajo la heurística aplicada. "
            f"No se observa una combinación suficientemente clara de incremento de amplitud y cambio de fase para confirmar velocidad crítica.\\n\\n"
            f"Esta condición puede representar una respuesta estable, una excitación forzada o una corrida donde el régimen crítico no fue cruzado de forma suficientemente clara."
        )
        action = (
            "Mantener la corrida como referencia histórica.\\n"
            "Comparar contra futuras trayectorias polares.\\n"
            "Correlacionar con Bode, espectro 1X, órbita y variables operativas."
        )

    return {"headline": headline, "detail": detail, "action": action}


def _prepare_polar_compare_df(item: Dict[str, Any], smooth_window: int, amp_smooth_window: int) -> pd.DataFrame:
    orient = get_panel_orientation(item["id"])
    df = item["grouped_df"].copy()

    df["amp"] = smooth_series(df["amp"], amp_smooth_window)
    df["phase_smoothed"] = circular_smooth_deg(df["phase"], smooth_window) % 360.0
    df["theta_display"] = compute_polar_display_theta(
        phase_deg=df["phase_smoothed"],
        axis_label=orient["axis_label"],
        side_label=orient["side_label"],
        install_angle_deg=float(orient["install_angle_deg"]),
        rotation_direction=orient["rotation_direction"],
    )
    df["phase_for_detection"] = np.rad2deg(np.unwrap(np.deg2rad(df["phase_smoothed"].to_numpy())))
    return df


def _polar_compare_metrics(item: Dict[str, Any], smooth_window: int, amp_smooth_window: int, max_critical_speeds: int) -> Dict[str, Any]:
    df = _prepare_polar_compare_df(item, smooth_window, amp_smooth_window)
    cs = estimate_critical_speeds_api684_style(df, max_count=max_critical_speeds)
    max_amp = float(df["amp"].max()) if len(df) else 0.0

    if cs:
        dom = cs[0]
        dom_speed = float(dom.get("speed", 0.0))
        dom_amp = float(dom.get("amp", 0.0))
        dom_phase = float(dom.get("phase_delta", 0.0))
    else:
        idx = int(df["amp"].idxmax()) if len(df) else 0
        dom_speed = float(df.loc[idx, "speed"]) if len(df) else 0.0
        dom_amp = max_amp
        dom_phase = 0.0

    ts_start = pd.to_datetime(df["ts_min"], errors="coerce").min() if "ts_min" in df.columns else None
    ts_end = pd.to_datetime(df["ts_max"], errors="coerce").max() if "ts_max" in df.columns else None

    return {
        "label": item["label"],
        "machine": item["machine"],
        "point": item["point"],
        "df": df,
        "critical_speeds": cs,
        "max_amp": max_amp,
        "dominant_speed": dom_speed,
        "dominant_amp": dom_amp,
        "dominant_phase_delta": dom_phase,
        "ts_start": ts_start,
        "ts_end": ts_end,
    }


def _polar_compare_diagnostic(records: List[Dict[str, Any]]) -> Dict[str, str]:
    ordered = sorted(
        records,
        key=lambda r: pd.Timestamp(r["ts_start"]) if r["ts_start"] is not None else pd.Timestamp.min
    )
    baseline = ordered[0]
    latest = ordered[-1]

    delta_amp = float(latest["dominant_amp"] - baseline["dominant_amp"])
    delta_speed = float(latest["dominant_speed"] - baseline["dominant_speed"])
    delta_phase = float(latest["dominant_phase_delta"] - baseline["dominant_phase_delta"])

    amp_trend = "incremento" if delta_amp > 0.15 else "reducción" if delta_amp < -0.15 else "estabilidad"
    speed_shift = "desplazamiento hacia mayor velocidad" if delta_speed > 100 else "desplazamiento hacia menor velocidad" if delta_speed < -100 else "sin desplazamiento relevante de velocidad"

    headline = "Comparación multi-fecha de trayectoria polar y respuesta modal"

    detail = (
        f"Se compararon {len(ordered)} corridas polares. Entre la corrida base ({baseline['label']}) y la más reciente ({latest['label']}) "
        f"se observa {amp_trend} de la amplitud dominante ({delta_amp:+.3f}), {speed_shift} ({delta_speed:+.0f} rpm) "
        f"y variación de fase dominante de {delta_phase:+.1f}°.\\n\\n"
        f"Desde el punto de vista rotodinámico, la comparación polar permite evaluar si la respuesta del rotor mantiene el mismo patrón modal o si existe migración de la zona crítica. "
        f"Cuando el máximo de amplitud y el cambio de fase se desplazan entre corridas, puede existir modificación de rigidez efectiva, amortiguamiento, condición de soporte, balance o carga.\\n\\n"
        f"Antes de una velocidad crítica el rotor tiende a comportarse como un sistema más rígido; al cruzar una forma modal, la fase y la trayectoria cambian y el rotor manifiesta comportamiento flexible. "
        f"Por eso, la lectura conjunta de amplitud, fase y velocidad es más concluyente que la amplitud por sí sola."
    )

    action = (
        "Correlacionar las corridas polares con Bode de amplitud/fase.\\n"
        "Verificar si la velocidad candidata se repite o migra entre fechas.\\n"
        "Comparar contra órbitas 1X y shaft centerline.\\n"
        "Validar si hubo cambios de balance, alineación, lubricación, temperatura o carga.\\n"
        "Usar la corrida más estable como línea base de aceptación."
    )

    return {"headline": headline, "detail": detail, "action": action}


def render_polar_compare_section(
    items: List[Dict[str, Any]],
    *,
    smooth_window: int,
    amp_smooth_window: int,
    max_critical_speeds: int,
    logo_uri: Optional[str],
) -> None:
    if len(items) < 2:
        return

    records = [
        _polar_compare_metrics(item, smooth_window, amp_smooth_window, max_critical_speeds)
        for item in items
    ]

    st.markdown("---")
    st.markdown("## Comparación multi-fecha · Polar Plot")

    fig = go.Figure()
    palette = ["#2563eb", "#16a34a", "#9333ea", "#ea580c", "#dc2626", "#0891b2", "#7c3aed"]

    for idx, rec in enumerate(records):
        df = rec["df"]
        color = palette[idx % len(palette)]
        date_label = pd.Timestamp(rec["ts_start"]).strftime("%Y-%m-%d %H:%M") if rec["ts_start"] is not None else rec["label"]

        fig.add_trace(
            go.Scatterpolar(
                r=df["amp"],
                theta=df["theta_display"],
                mode="lines",
                name=f"{date_label} · {rec['label']}",
                line=dict(width=2.5, color=color),
                hovertemplate="Amp: %{r:.3f}<br>Phase: %{theta:.1f}°<extra></extra>",
            )
        )

        if rec["critical_speeds"]:
            cs = rec["critical_speeds"][0]
            row = nearest_row_for_speed(df, cs["speed"])
            fig.add_trace(
                go.Scatterpolar(
                    r=[row["amp"]],
                    theta=[row["theta_display"]],
                    mode="markers+text",
                    marker=dict(size=9, color=color, symbol="diamond"),
                    text=[f"{int(round(cs['speed']))} rpm"],
                    textposition="top center",
                    textfont=dict(size=10, color=color),
                    showlegend=False,
                    hovertemplate="Candidate<br>Amp: %{r:.3f}<br>Phase: %{theta:.1f}°<extra></extra>",
                )
            )

    max_r = max([float(rec["df"]["amp"].max()) for rec in records if len(rec["df"])], default=1.0) * 1.18

    base_orient = get_panel_orientation(items[0]["id"])
    axis_label = base_orient["axis_label"]
    side_label = base_orient["side_label"]
    install_angle_deg = float(base_orient["install_angle_deg"])
    rotation_direction = base_orient["rotation_direction"]

    axis_rotation, angular_direction, _ = get_polar_axis_rotation_and_direction(
        axis_label=axis_label,
        side_label=side_label,
        install_angle_deg=install_angle_deg,
        rotation_direction=rotation_direction,
    )

    all_speeds = []
    for rec in records:
        if len(rec["df"]):
            all_speeds.extend(rec["df"]["speed"].astype(float).tolist())

    rpm_text = "—"
    if all_speeds:
        rpm_text = f"{int(min(all_speeds))} - {int(max(all_speeds))} rpm"

    dt_values = [r["ts_start"] for r in records if r["ts_start"] is not None]
    dt_text = "Comparación multi-fecha"
    if dt_values:
        dt_text = " / ".join([pd.Timestamp(v).strftime("%Y-%m-%d") for v in dt_values[:4]])

    draw_top_strip(
        fig=fig,
        machine=items[0].get("machine", ""),
        point_text="Polar Plot · Comparación multi-fecha",
        variable=f"{axis_label} | {install_angle_deg:.0f}° {side_label} | Rotation {rotation_direction}",
        dt_text=dt_text,
        rpm_text=rpm_text,
        logo_uri=logo_uri,
    )

    build_probe_reference_overlay(fig, max_r)

    fig.update_layout(
        title="Polar Plot · Comparación multi-fecha",
        polar=dict(
            bgcolor="#f8fafc",
            domain=dict(x=[0.0, 0.86], y=[0.05, 0.94]),
            radialaxis=dict(
                range=[0, max_r],
                tickfont=dict(size=11, color="#111827"),
                gridcolor="rgba(148,163,184,0.18)",
                linecolor="#9ca3af",
                showline=True,
                ticks="outside",
                angle=225,
            ),
            angularaxis=dict(
                rotation=axis_rotation,
                direction=angular_direction,
                tickfont=dict(size=12, color="#111827"),
                gridcolor="rgba(148,163,184,0.18)",
                linecolor="#9ca3af",
                showline=True,
                ticks="outside",
            ),
        ),
        height=820,
        template="plotly_white",
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0.0),
    )

    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="wm_polar_compare_plot")

    summary = pd.DataFrame([
        {
            "Archivo": r["label"],
            "Fecha inicio": pd.Timestamp(r["ts_start"]).strftime("%Y-%m-%d %H:%M") if r["ts_start"] is not None else "—",
            "Fecha fin": pd.Timestamp(r["ts_end"]).strftime("%Y-%m-%d %H:%M") if r["ts_end"] is not None else "—",
            "Amp dominante": round(r["dominant_amp"], 3),
            "RPM candidata": round(r["dominant_speed"], 0),
            "Delta fase": round(r["dominant_phase_delta"], 1),
            "Max amp": round(r["max_amp"], 3),
        }
        for r in records
    ])

    st.dataframe(summary, width="stretch", hide_index=True)

    diag = _polar_compare_diagnostic(records)
    st.markdown("### Diagnóstico comparativo automático")
    st.markdown(f"**{diag['headline']}**")
    st.write(diag["detail"])
    st.write("Se recomienda:")
    st.write(diag["action"])

    summary_lines = []
    for _, row in summary.iterrows():
        summary_lines.append(
            f"- {row['Archivo']}: candidato {row['RPM candidata']:.0f} rpm, "
            f"amplitud dominante {row['Amp dominante']:.3f}, "
            f"Δfase {row['Delta fase']:.1f}°, máximo {row['Max amp']:.3f}."
        )

    notes = (
        _build_polar_report_notes(diag)
        + "\n\nResumen comparativo de corridas:\n"
        + "\n".join(summary_lines)
    )

    png_bytes, png_error = build_export_png_bytes(fig, diag)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Enviar comparativo Polar a reporte", key="wm_polar_compare_report_btn"):
            ensure_report_state()
            st.session_state.report_items.append(
                {
                    "type": "polar_compare",
                    "title": "Polar Plot · Comparación multi-fecha",
                    "notes": notes,
                    "image_bytes": png_bytes,
                }
            )
            st.success("Comparativo Polar enviado al reporte.")
    with c2:
        if png_bytes is not None:
            st.download_button(
                "Descargar PNG comparativo Polar",
                data=png_bytes,
                file_name="polar_compare_hd.png",
                mime="image/png",
                key="wm_polar_compare_download_btn",
                width="stretch",
            )
        elif png_error:
            st.warning(f"No fue posible generar PNG comparativo: {png_error}")

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

    semaforo_status, semaforo_color, polar_diag = polar_health_status(
        critical_speeds=critical_speeds,
        amp_series=plot_df["amp"],
    )

    text_diag = build_polar_text_diagnostics(
        status=semaforo_status,
        critical_speeds=critical_speeds,
        max_amp=polar_diag["max_amp"],
    )

    machine = meta.get("Machine Name", "-")
    point = meta.get("Point Name", "-")
    variable = meta.get("Variable", "-")
    speed_unit = meta.get("Speed Unit", "rpm")
    amp_unit = meta.get("Amp Unit", "")

    panel_card(
        title=f"Polar {panel_index + 1} · {machine} · {point}",
        subtitle="Dynamic polar view",
        meta_html=(
            f"Variable: <b>{variable}</b> &nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Orientation: <b>{axis_label} | {install_angle_deg:.0f}° {side_label}</b> &nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Rotation: <b>{rotation_direction}</b> &nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Speed Range: <b>{int(plot_df['speed'].min())} - {int(plot_df['speed'].max())} {speed_unit}</b>"
        ),
        chips=[
            f"File: {item['file_name']}",
            f"Raw rows: {len(raw_df):,}",
            f"Grouped points: {len(plot_df):,}",
            f"Phase smoothing: {smooth_window}",
            f"Amplitude smoothing: {amp_smooth_window}",
            f"Critical speeds: {len(critical_speeds)}",
        ],
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
        semaforo_status=semaforo_status,
        semaforo_color=semaforo_color,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False},
        key=f"wm_polar_plot_{panel_index}_{item['id']}",
    )

    helper_card(
        title=f"API 684 Helper · Polar {panel_index + 1}",
        subtitle=text_diag["headline"],
        chips=[
            (f"Semáforo: {semaforo_status}", semaforo_color),
            (f"Health score: {polar_diag['score']:.1f}", None),
            (f"Max amplitude: {polar_diag['max_amp']:.3f} {amp_unit}", None),
            (f"Critical candidates: {polar_diag['candidate_count']}", None),
            (f"Cursor A: {row_a['amp']:.3f} {amp_unit}", None),
            (f"Cursor B: {row_b['amp']:.3f} {amp_unit}", None),
        ],
    )

    st.info(
        f"**Diagnostic detail:** {text_diag['detail']}\n\n"
        f"**Recommended action:** {text_diag['action']}"
    )

    title = f"Polar {panel_index + 1} — {machine} — {point}"

    export_state_key = (
        f"polar::{item['id']}::{panel_index}::{variable}::{smooth_window}::{amp_smooth_window}::"
        f"{show_info_box}::{show_rpm_labels}::{marker_stride}::{axis_label}::{side_label}::"
        f"{install_angle_deg}::{rotation_direction}::{detect_cs}::{max_critical_speeds}::"
        f"{cursor_a_speed}::{cursor_b_speed}"
    )

    export_report_row(
        export_key=export_state_key,
        fig=fig,
        export_builder=lambda export_fig: build_export_png_bytes(export_fig, text_diag),
        report_callback=lambda: queue_polar_to_report(
            meta,
            fig,
            title,
            text_diag,
            image_bytes=build_export_png_bytes(fig, text_diag)[0],
        ),
        file_name=f"{item['file_stem']}_polar_hd.png",
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    require_login()
    ensure_report_state()

    if "wm_polar_selected_ids" not in st.session_state:
        st.session_state.wm_polar_selected_ids = []

    page_header(
        title="Polar Plot",
        subtitle="Dynamic polar trajectory from amplitude, phase and speed.",
    )

    with st.sidebar:
        render_user_menu()
        st.markdown("---")
        st.markdown("### Upload Polar CSV")

        uploaded_files_new = st.file_uploader(
            "Upload one or more Polar CSV",
            type=["csv"],
            accept_multiple_files=True,
            key="wm_polar_file_uploader",
        )

        if uploaded_files_new:
            set_polar_persisted_files(uploaded_files_new)

        active_polar_files = get_polar_persisted_files()

        col1, col2 = st.columns(2)
        with col1:
            if active_polar_files:
                st.caption(f"Archivos Polar activos: {len(active_polar_files)}")
            else:
                st.caption("No hay archivos Polar cargados")

        with col2:
            if st.button("Limpiar archivos Polar", key="wm_polar_clear_files_btn"):
                clear_polar_persisted_files()
                st.rerun()

        uploaded_files = active_polar_files

    if not uploaded_files:
        panel_card(
            title="Carga archivos para comenzar",
            subtitle="Sube uno o varios archivos CSV Polar desde el panel izquierdo.",
            meta_html="",
            chips=[],
        )
        return

    parsed_items, failed_items = parse_uploaded_polar_files(uploaded_files)

    if failed_items:
        for file_name, error_text in failed_items:
            st.warning(f"No pude leer {file_name}: {error_text}")

    if not parsed_items:
        st.error("No se pudo cargar ningún archivo Polar válido.")
        return

    id_to_item = {item["id"]: item for item in parsed_items}
    label_to_id = {
        f"{item['machine']} · {item['point']} · {item['file_name']}": item["id"]
        for item in parsed_items
    }
    selection_labels = list(label_to_id.keys())

    valid_ids = set(id_to_item.keys())
    current_ids = [sid for sid in st.session_state.wm_polar_selected_ids if sid in valid_ids]
    if not current_ids:
        current_ids = [item["id"] for item in parsed_items]
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
        return

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

    if len(selected_items) >= 2:
        render_polar_compare_section(
            selected_items,
            smooth_window=smooth_window,
            amp_smooth_window=amp_smooth_window,
            max_critical_speeds=max_critical_speeds,
            logo_uri=logo_uri,
        )



if __name__ == "__main__":
    main()
