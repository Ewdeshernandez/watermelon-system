from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

from core.auth import require_login, render_user_menu
from core.diagnostics import format_number, get_semaforo_status
from core.module_patterns import export_report_row, helper_card, panel_card
from core.ui_theme import apply_watermelon_page_style, draw_top_strip, page_header


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Watermelon System | Bode Plot", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"

apply_watermelon_page_style()


# ============================================================
# STATE
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


# ============================================================
# LOAD / TRANSFORM
# ============================================================
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
    return series.astype(float).rolling(window=window, center=True, min_periods=1).median()


def unwrap_deg(series: pd.Series) -> pd.Series:
    vals = series.astype(float).to_numpy()
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(vals)))
    return pd.Series(unwrapped, index=series.index)


def read_bode_csv(file_obj) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame]:
    file_obj.seek(0)
    raw_bytes = file_obj.read()
    text = raw_bytes.decode("utf-8-sig", errors="replace") if isinstance(raw_bytes, bytes) else str(raw_bytes)
    lines = text.splitlines()
    if not lines:
        raise ValueError("Archivo vacío.")

    header_idx = None
    for i, line in enumerate(lines):
        if "X-Axis Value" in line and "Y-Axis Value" in line and "Phase" in line and "Timestamp" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("No se encontró el encabezado real del CSV Bode.")

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

    required = ["X-Axis Value", "Y-Axis Value", "Y-Axis Status", "Phase", "Phase Status", "Timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    df["rpm"] = pd.to_numeric(df["X-Axis Value"], errors="coerce")
    df["amp"] = pd.to_numeric(df["Y-Axis Value"], errors="coerce")
    df["phase"] = pd.to_numeric(df["Phase"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.dropna(subset=["rpm", "amp", "phase", "Timestamp"]).copy()
    df = df[
        df["Y-Axis Status"].astype(str).str.strip().str.lower().eq("valid")
        & df["Phase Status"].astype(str).str.strip().str.lower().eq("valid")
    ].copy()

    if df.empty:
        raise ValueError("No quedaron filas válidas después del filtrado.")

    raw_df = df.sort_values(["Timestamp", "rpm"]).reset_index(drop=True)

    grouped_df = (
        raw_df.groupby("rpm", as_index=False)
        .agg(
            amp=("amp", "median"),
            phase=("phase", lambda s: circular_mean_deg(s)),
            samples=("Timestamp", "size"),
            ts_min=("Timestamp", "min"),
            ts_max=("Timestamp", "max"),
        )
        .sort_values("rpm", kind="stable")
        .reset_index(drop=True)
    )

    return meta, raw_df, grouped_df


def uploaded_file_label(file_obj) -> str:
    return Path(getattr(file_obj, "name", "Bode.csv")).name


def uploaded_file_stem(file_obj) -> str:
    return Path(getattr(file_obj, "name", "Bode.csv")).stem


def parse_uploaded_bode_files(files: List[Any]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    parsed_items: List[Dict[str, Any]] = []
    failed_items: List[Tuple[str, str]] = []

    for file_obj in files:
        try:
            meta, raw_df, grouped_df = read_bode_csv(file_obj)
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
# ANALYSIS
# ============================================================
def nearest_row_for_rpm(df: pd.DataFrame, rpm_value: float) -> pd.Series:
    idx = int((df["rpm"] - float(rpm_value)).abs().idxmin())
    return df.loc[idx]


def estimate_critical_speeds_api684_style(df: pd.DataFrame, max_count: int = 2) -> List[Dict[str, float]]:
    if df.empty or len(df) < 12:
        return []

    amp = df["amp"].astype(float).to_numpy()
    rpm = df["rpm"].astype(float).to_numpy()
    phase = df["phase_continuous_internal"].astype(float).to_numpy()

    candidates: List[Dict[str, float]] = []

    if find_peaks is not None:
        prominence = max(np.nanmax(amp) * 0.08, 0.12)
        distance = max(8, len(df) // 16)
        peaks, props = find_peaks(amp, prominence=prominence, distance=distance)

        for i, p in enumerate(peaks):
            left = max(0, p - 8)
            right = min(len(df) - 1, p + 8)

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
                    "rpm": float(rpm[p]),
                    "amp": amp_peak,
                    "phase_delta": phase_delta,
                    "idx": int(p),
                    "prominence": prom,
                }
            )
    else:
        p = int(np.nanargmax(amp))
        left = max(0, p - 8)
        right = min(len(df) - 1, p + 8)
        candidates.append(
            {
                "rpm": float(rpm[p]),
                "amp": float(amp[p]),
                "phase_delta": float(phase[right] - phase[left]),
                "idx": int(p),
                "prominence": float(amp[p]),
            }
        )

    candidates = sorted(candidates, key=lambda x: (x["prominence"], x["amp"]), reverse=True)

    filtered: List[Dict[str, float]] = []
    for cand in candidates:
        if all(abs(cand["rpm"] - kept["rpm"]) > 120 for kept in filtered):
            filtered.append(cand)
        if len(filtered) >= max_count:
            break

    return sorted(filtered, key=lambda x: x["rpm"])


def bode_health_status(
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


def build_bode_text_diagnostics(
    *,
    status: str,
    critical_speeds: List[Dict[str, float]],
    max_amp: float,
) -> Dict[str, str]:
    if not critical_speeds:
        headline = "No strong resonance candidate detected in the current Bode range"
        detail = (
            f"The amplitude-phase response remains relatively controlled in the evaluated speed range. "
            f"Maximum observed amplitude is {max_amp:.3f}."
        )
        action = "Keep trending future run-up/coast-down events and compare repeatability."
        return {"headline": headline, "detail": detail, "action": action}

    cs1 = critical_speeds[0]
    rpm = int(round(float(cs1["rpm"])))
    amp = float(cs1["amp"])
    phase_delta = abs(float(cs1["phase_delta"]))

    if status == "SAFE":
        headline = f"Critical speed candidate near {rpm} rpm, but current response remains controlled"
        detail = (
            f"A candidate appears near {rpm} rpm with amplitude around {amp:.3f} and "
            f"phase change of {phase_delta:.1f}°. The event is present, but not yet severe."
        )
        action = "Track the same speed band during the next startup and compare amplitude growth."
        return {"headline": headline, "detail": detail, "action": action}

    if status == "WARNING":
        headline = f"Possible critical speed proximity near {rpm} rpm"
        detail = (
            f"The Bode response shows amplitude amplification near {rpm} rpm with a phase change of "
            f"{phase_delta:.1f}°. This is consistent with a relevant dynamic amplification zone."
        )
        action = "Compare with Polar and Shaft Centerline behavior, and verify whether the peak repeats consistently."
        return {"headline": headline, "detail": detail, "action": action}

    headline = f"Strong critical speed behavior near {rpm} rpm"
    detail = (
        f"The Bode response shows a dominant amplitude peak near {rpm} rpm with a phase change of "
        f"{phase_delta:.1f}°. This is highly consistent with resonance behavior."
    )
    action = "Treat as high priority: review operating avoidance, correlate with Polar/Shaft behavior, and avoid repeated operation in this speed zone until assessed."
    return {"headline": headline, "detail": detail, "action": action}


# ============================================================
# FIGURE UI
# ============================================================
def rounded_rect_path(x0: float, y0: float, x1: float, y1: float, r: float) -> str:
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
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


def draw_right_info_box(fig: go.Figure, rows: Sequence[Tuple[str, str]]) -> None:
    panel_x0 = 0.805
    panel_x1 = 0.970
    panel_y0 = 0.60
    panel_y1 = 0.94
    header_h = 0.045
    row_h = 0.055

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
        text="<b>Bode Information</b>",
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
            showarrow=False,
            font=dict(size=10.2, color="#111827"),
            align="left",
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.026, y=value_y,
            xanchor="left", yanchor="top",
            text=value,
            showarrow=False,
            font=dict(size=9.9, color="#111827"),
            align="left",
        )
        current_top -= row_h


def build_bode_info_rows(
    row_a: pd.Series,
    row_b: pd.Series,
    phase_mode: str,
    y_unit: str,
    x_unit: str,
    critical_speeds: List[Dict[str, float]],
    semaforo_status: str,
    semaforo_color: str,
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = [
        ("Cursor A", f"{format_number(row_a['amp'],3)} {y_unit} @ {int(round(row_a['rpm']))} {x_unit} | ∠{format_number(row_a['phase_header'],1)}°"),
        ("Cursor B", f"{format_number(row_b['amp'],3)} {y_unit} @ {int(round(row_b['rpm']))} {x_unit} | ∠{format_number(row_b['phase_header'],1)}°"),
        ("Phase Mode", phase_mode),
        ("Status", f"<span style='color:{semaforo_color};'><b>{semaforo_status}</b></span>"),
    ]

    for i, cs in enumerate(critical_speeds, start=1):
        title = f"Critical Speed {i}" if i == 1 else f"Secondary Candidate {i}"
        rows.append((title, f"{int(round(cs['rpm']))} {x_unit} | {format_number(cs['amp'],3)} {y_unit}"))
        rows.append((f"Phase Delta {i}", f"{format_number(cs['phase_delta'],1)}°"))

    return rows


def add_crosshair(fig: go.Figure, rpm_val: float, phase_val: float, amp_val: float, color: str) -> None:
    fig.add_vline(x=rpm_val, line_width=1.3, line_dash="dot", line_color=color, row=1, col=1)
    fig.add_vline(x=rpm_val, line_width=1.3, line_dash="dot", line_color=color, row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=[rpm_val], y=[phase_val], mode="markers",
            marker=dict(size=6, color=color, line=dict(width=1, color="#ffffff")),
            showlegend=False, hoverinfo="skip"
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[rpm_val], y=[amp_val], mode="markers",
            marker=dict(size=6, color=color, line=dict(width=1, color="#ffffff")),
            showlegend=False, hoverinfo="skip"
        ),
        row=2, col=1,
    )


# ============================================================
# FIGURE BUILD
# ============================================================
def build_bode_figure(
    df: pd.DataFrame,
    meta: Dict[str, str],
    row_a: pd.Series,
    row_b: pd.Series,
    x_min: float,
    x_max: float,
    logo_uri: Optional[str],
    phase_mode: str,
    critical_speeds: List[Dict[str, float]],
    show_info_box: bool,
    semaforo_status: str,
    semaforo_color: str,
) -> go.Figure:
    x_unit = meta.get("X-Axis Unit", "rpm") or "rpm"
    y_unit = meta.get("Y-Axis Unit", "") or ""

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        row_heights=[0.48, 0.52],
    )

    fig.add_trace(
        go.Scattergl(
            x=df["rpm"],
            y=df["phase_plot"],
            mode="lines",
            line=dict(width=1.10, color="#5b9cf0"),
            name="Phase",
            hovertemplate=f"Speed: %{{x:.0f}} {x_unit}<br>Phase: %{{y:.1f}}°<extra></extra>",
            showlegend=False,
            connectgaps=False,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df["rpm"],
            y=df["amp"],
            mode="lines",
            line=dict(width=1.35, color="#5b9cf0"),
            name="Amplitude",
            hovertemplate=f"Speed: %{{x:.0f}} {x_unit}<br>Amplitude: %{{y:.3f}} {y_unit}<extra></extra>",
            showlegend=False,
            connectgaps=False,
        ),
        row=2, col=1,
    )

    add_crosshair(fig, float(row_a["rpm"]), float(row_a["phase_plot"]), float(row_a["amp"]), "#efb08c")
    add_crosshair(fig, float(row_b["rpm"]), float(row_b["phase_plot"]), float(row_b["amp"]), "#7ac77b")

    cs_colors = ["#ef4444", "#f59e0b"]
    for idx, cs in enumerate(critical_speeds):
        color = cs_colors[idx % len(cs_colors)]
        cs_rpm = float(cs["rpm"])
        cs_amp = float(cs["amp"])
        cs_phase_row = nearest_row_for_rpm(df, cs_rpm)
        cs_phase = float(cs_phase_row["phase_plot"])

        fig.add_vline(x=cs_rpm, line_width=1.8, line_dash="dash", line_color=color, row=1, col=1)
        fig.add_vline(x=cs_rpm, line_width=1.8, line_dash="dash", line_color=color, row=2, col=1)

        fig.add_annotation(
            x=cs_rpm, y=cs_phase,
            xref="x", yref="y",
            text=f"Critical Speed {idx+1}<br>{int(round(cs_rpm))} rpm",
            showarrow=True, arrowhead=2, arrowcolor=color,
            ax=34, ay=-28,
            font=dict(size=9.6, color="#7f1d1d" if idx == 0 else "#92400e"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#fecaca" if idx == 0 else "#fde68a",
        )

        fig.add_annotation(
            x=cs_rpm, y=cs_amp,
            xref="x2", yref="y2",
            text=f"{format_number(cs_amp,3)} {y_unit}",
            showarrow=True, arrowhead=2, arrowcolor=color,
            ax=35, ay=-26,
            font=dict(size=9.4, color="#7f1d1d" if idx == 0 else "#92400e"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#fecaca" if idx == 0 else "#fde68a",
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
        variable=meta.get("Variable", "-"),
        dt_text=dt_text,
        rpm_text=f"{int(round(df['rpm'].min()))} - {int(round(df['rpm'].max()))} {x_unit}",
        logo_uri=logo_uri,
    )

    if show_info_box:
        rows = build_bode_info_rows(row_a, row_b, phase_mode, y_unit, x_unit, critical_speeds, semaforo_status, semaforo_color)
        draw_right_info_box(fig, rows)

    x_domain = [0.0, 0.77] if show_info_box else [0.0, 1.0]

    fig.update_layout(
        height=820,
        margin=dict(l=48, r=20, t=145, b=48),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f3f4f6",
        font=dict(color="#111827"),
        hovermode="closest",
        dragmode="pan",
        showlegend=False,
    )

    fig.update_xaxes(
        title=f"Speed ({x_unit})",
        range=[x_min, x_max],
        domain=x_domain,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
        row=2, col=1,
    )

    fig.update_xaxes(
        range=[x_min, x_max],
        domain=x_domain,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
        row=1, col=1,
    )

    fig.update_yaxes(
        title="Phase (°)",
        autorange="reversed",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
        row=1, col=1,
    )

    fig.update_yaxes(
        title=f"Amplitude ({y_unit})" if y_unit else "Amplitude",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
        row=2, col=1,
    )

    return fig


# ============================================================
# EXPORT / REPORT
# ============================================================
def _build_export_safe_figure(fig: go.Figure) -> go.Figure:
    export_fig = go.Figure()

    for trace in fig.data:
        if isinstance(trace, go.Scattergl):
            tj = trace.to_plotly_json()
            export_fig.add_trace(
                go.Scatter(
                    x=tj.get("x"),
                    y=tj.get("y"),
                    mode=tj.get("mode"),
                    line=tj.get("line"),
                    marker=tj.get("marker"),
                    hovertemplate=tj.get("hovertemplate"),
                    showlegend=tj.get("showlegend"),
                    name=tj.get("name"),
                    xaxis=tj.get("xaxis"),
                    yaxis=tj.get("yaxis"),
                    connectgaps=tj.get("connectgaps", False),
                )
            )
        else:
            export_fig.add_trace(trace)

    export_fig.update_layout(fig.layout)
    return export_fig


def _scale_export_figure(export_fig: go.Figure) -> go.Figure:
    fig = go.Figure(export_fig)

    scaled = []
    for trace in fig.data:
        tj = trace.to_plotly_json()
        if tj.get("type") == "scatter":
            mode = tj.get("mode", "")
            if "lines" in mode:
                line = dict(tj.get("line", {}) or {})
                line["width"] = max(3.6, float(line.get("width", 1.0)) * 2.25)
                tj["line"] = line
            if "markers" in mode:
                marker = dict(tj.get("marker", {}) or {})
                marker["size"] = max(10, float(marker.get("size", 6)) * 1.6)
                tj["marker"] = marker
        scaled.append(go.Scatter(**tj))

    fig = go.Figure(data=scaled, layout=fig.layout)

    fig.update_layout(
        width=4300,
        height=2200,
        margin=dict(l=110, r=60, t=320, b=110),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        font=dict(size=27, color="#111827"),
    )

    fig.update_xaxes(title_font=dict(size=36), tickfont=dict(size=23))
    fig.update_yaxes(title_font=dict(size=36), tickfont=dict(size=23))

    for shape in fig.layout.shapes or []:
        if shape.line is not None:
            width = getattr(shape.line, "width", 1) or 1
            shape.line.width = max(1.8, width * 1.9)

    for ann in fig.layout.annotations or []:
        if ann.font is not None:
            ann.font.size = max(20, int((ann.font.size or 12) * 1.8))

    for img in fig.layout.images or []:
        sx = getattr(img, "sizex", None)
        sy = getattr(img, "sizey", None)
        if sx is not None:
            img.sizex = sx * 1.12
        if sy is not None:
            img.sizey = sy * 1.12

    return fig


def _build_bode_report_notes(text_diag: Dict[str, str]) -> str:
    return (
        f"Resumen diagnóstico: {text_diag['headline']}\n\n"
        f"Detalle: {text_diag['detail']}\n\n"
        f"Acción recomendada: {text_diag['action']}"
    )


def _add_bode_export_footer(fig: go.Figure, text_diag: Dict[str, str]) -> go.Figure:
    export_fig = go.Figure(fig)

    headline = html.escape(str(text_diag.get("headline", "") or ""))
    detail = html.escape(str(text_diag.get("detail", "") or ""))
    action = html.escape(str(text_diag.get("action", "") or ""))

    existing_shapes = list(export_fig.layout.shapes) if export_fig.layout.shapes else []
    existing_annotations = list(export_fig.layout.annotations) if export_fig.layout.annotations else []

    export_fig.update_xaxes(domain=[0.06, 0.94])
    export_fig.update_yaxes(domain=[0.28, 0.95])

    existing_shapes.extend(
        [
            dict(
                type="line",
                xref="paper",
                yref="paper",
                x0=0.04,
                x1=0.96,
                y0=0.22,
                y1=0.22,
                line=dict(color="#64748b", width=3),
            ),
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.04,
                x1=0.96,
                y0=0.02,
                y1=0.205,
                line=dict(color="rgba(148,163,184,0.75)", width=2),
                fillcolor="rgba(255,255,255,0.97)",
                layer="below",
            ),
        ]
    )

    existing_annotations.extend(
        [
            dict(
                x=0.06,
                y=0.185,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                align="left",
                text="<b>DIAGNOSTIC SUMMARY</b>",
                font=dict(size=26, color="#0f172a"),
            ),
            dict(
                x=0.06,
                y=0.145,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                align="left",
                text=f"<b>{headline}</b>",
                font=dict(size=18, color="#111827"),
            ),
            dict(
                x=0.06,
                y=0.102,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                align="left",
                text=f"<b>Detail:</b> {detail}",
                font=dict(size=16, color="#111827"),
            ),
            dict(
                x=0.06,
                y=0.055,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                align="left",
                text=f"<b>Action:</b> {action}",
                font=dict(size=16, color="#111827"),
            ),
        ]
    )

    export_fig.update_layout(
        height=3000,
        margin=dict(l=110, r=60, t=320, b=170),
        shapes=existing_shapes,
        annotations=existing_annotations,
    )

    return export_fig


def build_export_png_bytes(fig: go.Figure, text_diag: Dict[str, str]) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        export_fig = _build_export_safe_figure(fig)
        export_fig = _scale_export_figure(export_fig)
        export_fig = _add_bode_export_footer(export_fig, text_diag)
        return export_fig.to_image(format="png", width=4300, height=3000, scale=2), None
    except Exception as e:
        return None, str(e)


def queue_bode_to_report(meta: Dict[str, str], fig: go.Figure, title: str, text_diag: Dict[str, str]) -> None:
    ensure_report_state()
    st.session_state.report_items.append(
        {
            "id": f"report-bode-{meta.get('Machine Name','')}-{meta.get('Point Name','')}-{title}",
            "type": "bode",
            "title": title,
            "notes": _build_bode_report_notes(text_diag),
            "signal_id": meta.get("Point Name", ""),
            "figure": go.Figure(fig),
            "machine": meta.get("Machine Name", ""),
            "point": meta.get("Point Name", ""),
            "variable": meta.get("Variable", ""),
            "timestamp": "",
        }
    )


# ============================================================
# PANEL RENDER
# ============================================================
def render_bode_panel(
    item: Dict[str, Any],
    panel_index: int,
    *,
    logo_uri: Optional[str],
    smooth_window: int,
    auto_x: bool,
    x_min_global: float,
    x_max_global: float,
    phase_mode: str,
    detect_cs: bool,
    max_critical_speeds: int,
    show_info_box: bool,
) -> None:
    meta = item["meta"]
    raw_df = item["raw_df"]
    grouped_df = item["grouped_df"]

    plot_df = grouped_df.copy()
    plot_df["amp"] = smooth_series(plot_df["amp"], smooth_window)

    phase_wrapped_raw = plot_df["phase"].astype(float) % 360.0
    phase_wrapped_smooth = circular_smooth_deg(phase_wrapped_raw, min(smooth_window, 5))
    phase_continuous_internal = unwrap_deg(phase_wrapped_smooth)

    plot_df["phase_wrapped_raw"] = phase_wrapped_raw
    plot_df["phase_wrapped_smooth"] = phase_wrapped_smooth
    plot_df["phase_continuous_internal"] = phase_continuous_internal

    if phase_mode == "Wrapped Raw 0-360":
        plot_df["phase_plot"] = plot_df["phase_wrapped_raw"]
        plot_df["phase_header"] = plot_df["phase_wrapped_raw"]
    else:
        plot_df["phase_plot"] = plot_df["phase_wrapped_smooth"]
        plot_df["phase_header"] = plot_df["phase_wrapped_smooth"]

    rpm_min_default = float(plot_df["rpm"].min())
    rpm_max_default = float(plot_df["rpm"].max())

    if auto_x:
        x_min = rpm_min_default
        x_max = rpm_max_default
    else:
        x_min = x_min_global
        x_max = x_max_global

    display_df = plot_df[(plot_df["rpm"] >= x_min) & (plot_df["rpm"] <= x_max)].copy()
    if display_df.empty:
        st.warning(f"Panel {panel_index + 1}: no hay puntos en el rango RPM seleccionado.")
        return

    rpm_min_display = int(display_df["rpm"].min())
    rpm_max_display = int(display_df["rpm"].max())

    c1, c2 = st.columns(2)
    with c1:
        cursor_a_rpm = st.slider(
            f"Cursor A (RPM) · Panel {panel_index + 1}",
            rpm_min_display,
            rpm_max_display,
            rpm_min_display,
            key=f"bode_cursor_a_{panel_index}_{item['id']}",
        )
    with c2:
        cursor_b_rpm = st.slider(
            f"Cursor B (RPM) · Panel {panel_index + 1}",
            rpm_min_display,
            rpm_max_display,
            rpm_max_display,
            key=f"bode_cursor_b_{panel_index}_{item['id']}",
        )

    row_a = nearest_row_for_rpm(display_df, cursor_a_rpm)
    row_b = nearest_row_for_rpm(display_df, cursor_b_rpm)

    critical_speeds: List[Dict[str, float]] = []
    if detect_cs:
        critical_speeds = estimate_critical_speeds_api684_style(display_df, max_count=max_critical_speeds)

    semaforo_status, semaforo_color, bode_diag = bode_health_status(
        critical_speeds=critical_speeds,
        amp_series=display_df["amp"],
    )

    text_diag = build_bode_text_diagnostics(
        status=semaforo_status,
        critical_speeds=critical_speeds,
        max_amp=bode_diag["max_amp"],
    )

    machine = meta.get("Machine Name", "-")
    point = meta.get("Point Name", "-")
    variable = meta.get("Variable", "-")
    probe_angle = meta.get("Probe Angle", "-")
    x_unit = meta.get("X-Axis Unit", "rpm")
    y_unit = meta.get("Y-Axis Unit", "")

    panel_card(
        title=f"Bode {panel_index + 1} · {machine} · {point}",
        subtitle="Run-up / coast-down amplitude and phase view",
        meta_html=(
            f"Variable: <b>{variable}</b> &nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Probe Angle: <b>{probe_angle}</b> &nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Speed Range: <b>{int(display_df['rpm'].min())} - {int(display_df['rpm'].max())} {x_unit}</b>"
        ),
        chips=[
            f"File: {item['file_name']}",
            f"Raw rows: {len(raw_df):,}",
            f"Grouped points: {len(display_df):,}",
            f"Phase mode: {phase_mode}",
            f"Smoothing: {smooth_window}",
            f"Critical speeds: {len(critical_speeds)}",
        ],
    )

    fig = build_bode_figure(
        df=display_df,
        meta=meta,
        row_a=row_a,
        row_b=row_b,
        x_min=float(display_df["rpm"].min()),
        x_max=float(display_df["rpm"].max()),
        logo_uri=logo_uri,
        phase_mode=phase_mode,
        critical_speeds=critical_speeds,
        show_info_box=show_info_box,
        semaforo_status=semaforo_status,
        semaforo_color=semaforo_color,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False},
        key=f"wm_bode_plot_{panel_index}_{item['id']}",
    )

    helper_card(
        title=f"API RP 684 Helper · Bode {panel_index + 1}",
        subtitle=text_diag["headline"],
        chips=[
            (f"Semáforo: {semaforo_status}", semaforo_color),
            (f"Health score: {bode_diag['score']:.1f}", None),
            (f"Max amplitude: {bode_diag['max_amp']:.3f} {y_unit}", None),
            (f"Critical candidates: {bode_diag['candidate_count']}", None),
            (f"Cursor A: {row_a['amp']:.3f} {y_unit}", None),
            (f"Cursor B: {row_b['amp']:.3f} {y_unit}", None),
        ],
    )

    st.info(
        f"**Diagnostic detail:** {text_diag['detail']}\n\n"
        f"**Recommended action:** {text_diag['action']}"
    )

    title = f"Bode {panel_index + 1} — {machine} — {point}"
    export_state_key = (
        f"bode::{item['id']}::{panel_index}::{phase_mode}::{smooth_window}::"
        f"{detect_cs}::{max_critical_speeds}::{show_info_box}::"
        f"{int(display_df['rpm'].min())}::{int(display_df['rpm'].max())}::"
        f"{cursor_a_rpm}::{cursor_b_rpm}"
    )

    export_report_row(
        export_key=export_state_key,
        fig=fig,
        export_builder=lambda export_fig: build_export_png_bytes(export_fig, text_diag),
        report_callback=lambda: queue_bode_to_report(meta, fig, title, text_diag),
        file_name=f"{item['file_stem']}_bode_hd.png",
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    require_login()
    ensure_report_state()

    if "wm_bode_selected_ids" not in st.session_state:
        st.session_state.wm_bode_selected_ids = []

    page_header(
        title="Bode Plot",
        subtitle="Amplitude and phase versus speed from Bode CSV files.",
    )

    with st.sidebar:
        render_user_menu()
        st.markdown("---")
        st.markdown("### Upload Bode CSV")
        uploaded_files = st.file_uploader(
            "Upload one or more Bode CSV",
            type=["csv"],
            accept_multiple_files=True,
        )

    if not uploaded_files:
        panel_card(
            title="Carga archivos para comenzar",
            subtitle="Sube uno o varios archivos CSV Bode desde el panel izquierdo.",
            meta_html="",
            chips=[],
        )
        return

    parsed_items, failed_items = parse_uploaded_bode_files(uploaded_files)

    if failed_items:
        for file_name, error_text in failed_items:
            st.warning(f"No pude leer {file_name}: {error_text}")

    if not parsed_items:
        st.error("No se pudo cargar ningún archivo Bode válido.")
        return

    id_to_item = {item["id"]: item for item in parsed_items}
    label_to_id = {
        f"{item['machine']} · {item['point']} · {item['file_name']}": item["id"]
        for item in parsed_items
    }
    selection_labels = list(label_to_id.keys())

    valid_ids = set(id_to_item.keys())
    current_ids = [sid for sid in st.session_state.wm_bode_selected_ids if sid in valid_ids]
    if not current_ids:
        current_ids = [parsed_items[0]["id"]]
        st.session_state.wm_bode_selected_ids = current_ids

    default_labels = [label for label, sid in label_to_id.items() if sid in current_ids]

    with st.sidebar:
        st.markdown("### Bode Selection")
        selected_labels = st.multiselect(
            "Bodes to display",
            options=selection_labels,
            default=default_labels,
        )
        st.session_state.wm_bode_selected_ids = [label_to_id[label] for label in selected_labels if label in label_to_id]

        selected_ids_for_sidebar = [sid for sid in st.session_state.wm_bode_selected_ids if sid in id_to_item]
        candidate_frames = [id_to_item[sid]["grouped_df"] for sid in selected_ids_for_sidebar]
        candidate_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.concat([parsed_items[0]["grouped_df"]], ignore_index=True)

        st.markdown("### X Axis Control")
        auto_x = st.checkbox("Auto scale X", value=True)
        x_min_default = float(candidate_df["rpm"].min())
        x_max_default = float(candidate_df["rpm"].max())

        if auto_x:
            x_min = x_min_default
            x_max = x_max_default
        else:
            x_min = st.number_input("Min RPM", value=float(x_min_default), step=10.0)
            x_max = st.number_input("Max RPM", value=float(x_max_default), step=10.0)

        st.markdown("### Phase Mode")
        phase_mode = st.selectbox("Phase display", ["Wrapped Raw 0-360", "Wrapped Smoothed"], index=1)

        st.markdown("### Smoothing")
        smooth_window = st.slider("Median smoothing window", 1, 21, 3, step=2)

        st.markdown("### Critical Speed Detection")
        detect_cs = st.checkbox("Estimate critical speeds (API RP 684 heuristic)", value=True)
        max_critical_speeds = st.selectbox("Max critical speeds", [1, 2], index=1)

        st.markdown("### Information Box")
        show_info_box = st.checkbox("Show Bode Information", value=True)

    selected_ids = [sid for sid in st.session_state.wm_bode_selected_ids if sid in id_to_item]
    if not selected_ids:
        st.info("Selecciona uno o más Bodes en la barra lateral.")
        return

    selected_items = [id_to_item[sid] for sid in selected_ids]
    logo_uri = get_logo_data_uri(LOGO_PATH)

    for panel_index, item in enumerate(selected_items):
        render_bode_panel(
            item=item,
            panel_index=panel_index,
            logo_uri=logo_uri,
            smooth_window=smooth_window,
            auto_x=auto_x,
            x_min_global=float(x_min),
            x_max_global=float(x_max),
            phase_mode=phase_mode,
            detect_cs=detect_cs,
            max_critical_speeds=max_critical_speeds,
            show_info_box=show_info_box,
        )

        if panel_index < len(selected_items) - 1:
            st.markdown("---")


if __name__ == "__main__":
    main()
