from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.auth import require_login, render_user_menu


st.set_page_config(page_title="Watermelon System | Bode Plot", layout="wide")
require_login()
render_user_menu()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "watermelon_logo.png"


# =========================
# STYLES
# =========================
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f3f4f6; }
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
            margin-bottom: 0.9rem;
        }
        .wm-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.86));
            border: 1px solid #dbe3ee;
            border-radius: 18px;
            padding: 14px 16px;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# HELPERS
# =========================
def ensure_report_state() -> None:
    if "report_items" not in st.session_state:
        st.session_state["report_items"] = []


def format_number(value: Any, decimals: int = 3) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "—"


def get_logo_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
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


# =========================
# CSV LOADER (ROBUST)
# =========================
def read_bode_csv(file_obj) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame]:
    file_obj.seek(0)
    raw_bytes = file_obj.read()
    if isinstance(raw_bytes, bytes):
        text = raw_bytes.decode("utf-8-sig", errors="replace")
    else:
        text = str(raw_bytes)

    lines = text.splitlines()
    if not lines:
        raise ValueError("Archivo vacío.")

    header_idx = None
    for i, line in enumerate(lines):
        ll = line.strip()
        if "X-Axis Value" in ll and "Y-Axis Value" in ll and "Phase" in ll:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("No se encontró el encabezado real del CSV de Bode.")

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

    required = [
        "X-Axis Value",
        "Y-Axis Value",
        "Y-Axis Status",
        "Phase",
        "Phase Status",
        "Timestamp",
    ]
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
            phase=("phase", "median"),
            samples=("Timestamp", "size"),
            ts_min=("Timestamp", "min"),
            ts_max=("Timestamp", "max"),
        )
        .sort_values("rpm", kind="stable")
        .reset_index(drop=True)
    )

    return meta, raw_df, grouped_df


# =========================
# SIGNAL PROCESSING
# =========================
def phase_unwrap_deg(phase_series: pd.Series) -> pd.Series:
    rad = np.deg2rad(phase_series.astype(float).to_numpy())
    unwrapped = np.unwrap(rad)
    return pd.Series(np.rad2deg(unwrapped), index=phase_series.index)


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series.copy()
    return series.rolling(window=window, center=True, min_periods=1).median()


def estimate_critical_speed_api684_style(df: pd.DataFrame, amp_col: str, phase_col: str) -> Optional[Dict[str, float]]:
    """
    Heurística estilo API 684:
    - toma el máximo principal de amplitud
    - revisa cambio de fase alrededor del pico
    - devuelve velocidad crítica estimada, amplitud y delta de fase
    No sustituye una validación rotodinámica formal.
    """
    if df.empty or len(df) < 9:
        return None

    amp = df[amp_col].astype(float)
    phase = df[phase_col].astype(float)
    rpm = df["rpm"].astype(float)

    peak_idx = int(amp.idxmax())
    if peak_idx <= 2 or peak_idx >= len(df) - 3:
        return None

    left_idx = max(0, peak_idx - 3)
    right_idx = min(len(df) - 1, peak_idx + 3)

    phase_left = float(phase.iloc[left_idx])
    phase_right = float(phase.iloc[right_idx])
    phase_delta = phase_right - phase_left

    return {
        "rpm": float(rpm.iloc[peak_idx]),
        "amp": float(amp.iloc[peak_idx]),
        "phase": float(phase.iloc[peak_idx]),
        "phase_delta": float(phase_delta),
        "idx": peak_idx,
    }


def nearest_row_for_rpm(df: pd.DataFrame, rpm_value: float) -> pd.Series:
    idx = int((df["rpm"] - rpm_value).abs().idxmin())
    return df.loc[idx]


# =========================
# VISUALS
# =========================
def draw_top_strip(
    fig: go.Figure,
    meta: Dict[str, str],
    df: pd.DataFrame,
    logo_uri: Optional[str],
    row_a: pd.Series,
    row_b: pd.Series,
) -> None:
    x0, x1 = 0.006, 0.994
    y0, y1 = 1.015, 1.108
    radius = 0.015

    fig.add_shape(
        type="path",
        xref="paper",
        yref="paper",
        path=rounded_rect_path(x0, y0, x1, y1, radius),
        line=dict(color="#cfd8e3", width=1.1),
        fillcolor="rgba(255,255,255,0.96)",
        layer="below",
    )

    y_text = (y0 + y1) / 2.0

    machine = meta.get("Machine Name", "") or ""
    point = meta.get("Point Name", "") or ""
    variable = meta.get("Variable", "") or ""
    angle = meta.get("Probe Angle", "") or ""
    x_unit = meta.get("X-Axis Unit", "rpm") or "rpm"
    y_unit = meta.get("Y-Axis Unit", "") or ""

    if logo_uri:
        fig.add_layout_image(
            dict(
                source=logo_uri,
                xref="paper",
                yref="paper",
                x=0.014,
                y=y1 - 0.007,
                sizex=0.058,
                sizey=0.082,
                xanchor="left",
                yanchor="top",
                layer="above",
                sizing="contain",
                opacity=1.0,
            )
        )
        machine_x = 0.082
    else:
        machine_x = 0.018

    fig.add_annotation(
        xref="paper", yref="paper",
        x=machine_x, y=y_text,
        xanchor="left", yanchor="middle",
        text=f"<b>{machine}</b>",
        showarrow=False,
        font=dict(size=13.0, color="#111827"),
        align="left",
    )

    left_text = point
    if variable:
        left_text += f" | {variable}"
    if angle:
        left_text += f" | {angle}"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.25, y=y_text,
        xanchor="left", yanchor="middle",
        text=left_text,
        showarrow=False,
        font=dict(size=12.0, color="#111827"),
        align="left",
    )

    a_txt = (
        f"A: <b>{format_number(row_a['amp'],3)} {y_unit}</b> "
        f"∠{format_number(row_a['phase_plot'],1)}° @ {int(round(row_a['rpm']))} {x_unit}"
    )
    b_txt = (
        f"B: <b>{format_number(row_b['amp'],3)} {y_unit}</b> "
        f"∠{format_number(row_b['phase_plot'],1)}° @ {int(round(row_b['rpm']))} {x_unit}"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.52, y=y_text,
        xanchor="left", yanchor="middle",
        text=f"{a_txt} &nbsp;&nbsp;|&nbsp;&nbsp; {b_txt}",
        showarrow=False,
        font=dict(size=11.7, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.988, y=y_text,
        xanchor="right", yanchor="middle",
        text=f"{int(df['rpm'].min())} - {int(df['rpm'].max())} {x_unit}",
        showarrow=False,
        font=dict(size=11.5, color="#111827"),
        align="right",
    )


def add_cursor(fig: go.Figure, rpm: float, phase: float, amp: float, color: str) -> None:
    fig.add_vline(x=rpm, line_width=1.5, line_dash="dot", line_color=color, row=1, col=1)
    fig.add_vline(x=rpm, line_width=1.5, line_dash="dot", line_color=color, row=2, col=1)

    phase_span = 14
    amp_span = 0.18

    fig.add_shape(type="line", x0=rpm, x1=rpm, y0=phase-phase_span, y1=phase+phase_span,
                  line=dict(color=color, width=1.3), row=1, col=1)
    fig.add_shape(type="line", x0=rpm-18, x1=rpm+18, y0=phase, y1=phase,
                  line=dict(color=color, width=1.3), row=1, col=1)

    fig.add_shape(type="line", x0=rpm, x1=rpm, y0=amp-amp_span, y1=amp+amp_span,
                  line=dict(color=color, width=1.3), row=2, col=1)
    fig.add_shape(type="line", x0=rpm-18, x1=rpm+18, y0=amp, y1=amp,
                  line=dict(color=color, width=1.3), row=2, col=1)


def build_bode_figure(
    df: pd.DataFrame,
    meta: Dict[str, str],
    row_a: pd.Series,
    row_b: pd.Series,
    x_min: float,
    x_max: float,
    logo_uri: Optional[str],
    phase_mode: str,
    critical_speed_result: Optional[Dict[str, float]],
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
            line=dict(width=1.35, color="#5B9CF0"),
            name="Phase",
            hovertemplate=f"Speed: %{{x:.0f}} {x_unit}<br>Phase: %{{y:.1f}}°<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df["rpm"],
            y=df["amp"],
            mode="lines",
            line=dict(width=1.55, color="#5B9CF0"),
            name="Amplitude",
            hovertemplate=f"Speed: %{{x:.0f}} {x_unit}<br>Amplitude: %{{y:.3f}} {y_unit}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    add_cursor(fig, float(row_a["rpm"]), float(row_a["phase_plot"]), float(row_a["amp"]), "#efb08c")
    add_cursor(fig, float(row_b["rpm"]), float(row_b["phase_plot"]), float(row_b["amp"]), "#7ac77b")

    if critical_speed_result is not None:
        cs_rpm = critical_speed_result["rpm"]
        cs_amp = critical_speed_result["amp"]
        cs_phase = critical_speed_result["phase"]
        fig.add_vline(x=cs_rpm, line_width=2.0, line_dash="dash", line_color="#ef4444", row=1, col=1)
        fig.add_vline(x=cs_rpm, line_width=2.0, line_dash="dash", line_color="#ef4444", row=2, col=1)
        fig.add_annotation(
            x=cs_rpm,
            y=cs_phase,
            xref="x",
            yref="y",
            text=f"Estimated Critical Speed<br>{int(round(cs_rpm))} rpm",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#ef4444",
            ax=55,
            ay=-40,
            font=dict(size=11, color="#991b1b"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#fecaca",
        )
        fig.add_annotation(
            x=cs_rpm,
            y=cs_amp,
            xref="x2",
            yref="y2",
            text=f"{format_number(cs_amp,3)} {y_unit}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#ef4444",
            ax=45,
            ay=-35,
            font=dict(size=11, color="#991b1b"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#fecaca",
        )

    fig.update_layout(
        template="plotly_white",
        height=820,
        margin=dict(l=48, r=24, t=145, b=48),
        showlegend=False,
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        hovermode="closest",
        dragmode="pan",
    )

    fig.update_xaxes(
        title_text=f"Speed ({x_unit})",
        range=[x_min, x_max],
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        row=2,
        col=1,
    )
    fig.update_xaxes(
        range=[x_min, x_max],
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        row=1,
        col=1,
    )

    phase_title = "Phase (°)" if phase_mode == "Wrapped 0-360" else "Phase Unwrapped (°)"
    fig.update_yaxes(
        title_text=phase_title,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=f"Amplitude ({y_unit})" if y_unit else "Amplitude",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        row=2,
        col=1,
    )

    draw_top_strip(fig, meta, df, logo_uri, row_a, row_b)
    return fig


def build_bode_information(row_a: pd.Series, row_b: pd.Series, critical_speed_result: Optional[Dict[str, float]], y_unit: str) -> str:
    crit_html = ""
    if critical_speed_result is not None:
        crit_html = f"""
        <div class="wm-card" style="margin-top:10px;">
            <div class="wm-card-title">Estimated Critical Speed</div>
            <div class="wm-meta">
                Speed: <b>{int(round(critical_speed_result['rpm']))} rpm</b><br>
                Peak Amplitude: <b>{format_number(critical_speed_result['amp'],3)} {y_unit}</b><br>
                Phase at Peak: <b>{format_number(critical_speed_result['phase'],1)}°</b><br>
                Phase Delta Around Peak: <b>{format_number(critical_speed_result['phase_delta'],1)}°</b>
            </div>
        </div>
        """

    return f"""
    <div class="wm-card">
        <div class="wm-card-title">Bode Information</div>
        <div class="wm-card-subtitle">Cursor summary</div>
        <div class="wm-meta">
            <b>Cursor A</b><br>
            RPM: {int(round(row_a['rpm']))}<br>
            Amplitude: {format_number(row_a['amp'],3)} {y_unit}<br>
            Phase: {format_number(row_a['phase_plot'],1)}°
            <br><br>
            <b>Cursor B</b><br>
            RPM: {int(round(row_b['rpm']))}<br>
            Amplitude: {format_number(row_b['amp'],3)} {y_unit}<br>
            Phase: {format_number(row_b['phase_plot'],1)}°
        </div>
    </div>
    {crit_html}
    """


def export_png(fig: go.Figure, scale: int) -> bytes:
    export_fig = go.Figure(fig.to_dict())
    export_fig.update_layout(width=4600, height=2300, margin=dict(l=110, r=50, t=320, b=120))
    return export_fig.to_image(format="png", scale=scale)


def push_bode_to_report(title: str, fig: go.Figure, meta: Dict[str, str]) -> None:
    ensure_report_state()
    st.session_state["report_items"].append(
        {
            "type": "bode",
            "title": title,
            "figure": go.Figure(fig),
            "image_bytes": None,
            "notes": "",
            "source_module": "07_Bode_Plot",
            "machine": meta.get("Machine Name", ""),
            "point": meta.get("Point Name", ""),
            "variable": meta.get("Variable", ""),
            "timestamp": "",
        }
    )


# =========================
# MAIN
# =========================
def main() -> None:
    inject_styles()
    ensure_report_state()

    st.markdown('<div class="wm-page-title">Bode Plot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="wm-page-subtitle">Amplitude and phase versus speed from Bode CSV files.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Upload Bode CSV")
        uploaded_file = st.file_uploader("Upload Bode CSV", type=["csv"], accept_multiple_files=False)

    if not uploaded_file:
        st.info("Carga un archivo CSV de Bode para visualizar amplitud y fase contra velocidad.")
        return

    try:
        meta, raw_df, grouped_df = read_bode_csv(uploaded_file)
    except Exception as e:
        st.error(f"No pude leer el CSV Bode: {e}")
        return

    if grouped_df.empty:
        st.error("No hay datos válidos para construir el Bode.")
        return

    with st.sidebar:
        st.markdown("### X Axis Control")
        auto_x = st.checkbox("Auto scale X", value=True)
        x_min = float(grouped_df["rpm"].min())
        x_max = float(grouped_df["rpm"].max())

        if not auto_x:
            x_min = st.number_input("Min RPM", value=float(x_min), step=10.0)
            x_max = st.number_input("Max RPM", value=float(x_max), step=10.0)

        st.markdown("### Phase Mode")
        phase_mode = st.selectbox("Phase display", ["Wrapped 0-360", "Unwrapped"])

        st.markdown("### Smoothing")
        smooth_window = st.slider("Median smoothing window", 1, 21, 3, step=2)

        st.markdown("### Critical Speed")
        detect_cs = st.checkbox("Estimate critical speed (API-684 style heuristic)", value=True)

        st.markdown("### Cursors")
        cursor_a_rpm = st.slider("Cursor A (RPM)", int(x_min), int(x_max), int(grouped_df["rpm"].iloc[0]))
        cursor_b_rpm = st.slider("Cursor B (RPM)", int(x_min), int(x_max), int(grouped_df["rpm"].iloc[-1]))

        st.markdown("### Export")
        export_scale = st.selectbox("PNG scale", [2, 3, 4], index=1)

    plot_df = grouped_df.copy()
    plot_df["amp"] = smooth_series(plot_df["amp"], smooth_window)

    if phase_mode == "Wrapped 0-360":
        plot_df["phase_plot"] = plot_df["phase"] % 360.0
    else:
        plot_df["phase_plot"] = phase_unwrap_deg(plot_df["phase"])

    row_a = nearest_row_for_rpm(plot_df, cursor_a_rpm)
    row_b = nearest_row_for_rpm(plot_df, cursor_b_rpm)

    critical_speed_result = None
    if detect_cs:
        critical_speed_result = estimate_critical_speed_api684_style(plot_df, "amp", "phase_plot")

    machine = meta.get("Machine Name", "-")
    point = meta.get("Point Name", "-")
    variable = meta.get("Variable", "-")
    probe_angle = meta.get("Probe Angle", "-")
    x_unit = meta.get("X-Axis Unit", "rpm")
    y_unit = meta.get("Y-Axis Unit", "")

    st.markdown(
        f"""
        <div class="wm-card">
            <div class="wm-card-title">{machine} · {point}</div>
            <div class="wm-card-subtitle">Bode run-up / coast-down view</div>
            <div class="wm-meta">
                Variable: <b>{variable}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Probe Angle: <b>{probe_angle}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Speed Range: <b>{int(plot_df['rpm'].min())} - {int(plot_df['rpm'].max())} {x_unit}</b>
            </div>
            <div class="wm-chip-row">
                <div class="wm-chip">Raw rows: {len(raw_df):,}</div>
                <div class="wm-chip">Grouped points: {len(plot_df):,}</div>
                <div class="wm-chip">Phase mode: {phase_mode}</div>
                <div class="wm-chip">Smoothing: {smooth_window}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    logo_uri = get_logo_data_uri(LOGO_PATH)
    fig = build_bode_figure(
        plot_df,
        meta,
        row_a,
        row_b,
        x_min=x_min,
        x_max=x_max,
        logo_uri=logo_uri,
        phase_mode=phase_mode,
        critical_speed_result=critical_speed_result,
    )

    left, right = st.columns([3.6, 1.25])
    with left:
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    with right:
        st.markdown(build_bode_information(row_a, row_b, critical_speed_result, y_unit), unsafe_allow_html=True)

        title = f"Bode - {machine} - {point} - {variable}".strip(" -")
        try:
            png_bytes = export_png(fig, export_scale)
            st.download_button(
                "Export PNG HD",
                data=png_bytes,
                file_name=f"{Path(uploaded_file.name).stem}_bode_hd.png",
                mime="image/png",
                width="stretch",
            )
        except Exception as e:
            st.warning(f"No pude generar el PNG HD: {e}")

        if st.button("Enviar a Reporte", width="stretch"):
            push_bode_to_report(title=title, fig=fig, meta=meta)
            st.success("Bode enviado al reporte.")

    with st.expander("Grouped Data", expanded=False):
        st.dataframe(plot_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
