from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# PAGE STYLES
# =========================
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f3f4f6;
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
            background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(255,255,255,0.82));
            border: 1px solid #dbe3ee;
            border-radius: 18px;
            padding: 14px 16px 14px 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            margin-bottom: 12px;
        }

        .wm-card-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.15rem;
        }

        .wm-card-subtitle {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }

        .wm-meta {
            color: #334155;
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .wm-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }

        .wm-chip {
            border: 1px solid #dbe3ee;
            background: #f8fafc;
            border-radius: 999px;
            padding: 5px 10px;
            color: #334155;
            font-size: 0.82rem;
            font-weight: 600;
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


def safe_ts(value: Any) -> Optional[pd.Timestamp]:
    try:
        if pd.isna(value):
            return None
        return pd.Timestamp(value)
    except Exception:
        return None


def ts_range_text(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> str:
    if start is None or end is None:
        return "—"
    return f"{start.strftime('%Y-%m-%d %H:%M:%S')} → {end.strftime('%Y-%m-%d %H:%M:%S')}"


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
# CSV LOADER
# =========================
def load_bode_csv(uploaded_file: Any) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame]:
    uploaded_file.seek(0)
    meta = pd.read_csv(
        uploaded_file,
        nrows=7,
        header=None,
        names=["key", "value"],
        encoding="utf-8-sig",
    )

    uploaded_file.seek(0)
    df = pd.read_csv(
        uploaded_file,
        skiprows=7,
        encoding="utf-8-sig",
    )

    meta_dict: Dict[str, str] = {}
    for _, row in meta.iterrows():
        k = str(row["key"]).strip()
        v = "" if pd.isna(row["value"]) else str(row["value"]).strip()
        meta_dict[k] = v

    expected_cols = [
        "X-Axis Value",
        "Y-Axis Value",
        "Y-Axis Status",
        "Phase",
        "Phase Status",
        "Timestamp",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    df["X-Axis Value"] = pd.to_numeric(df["X-Axis Value"], errors="coerce")
    df["Y-Axis Value"] = pd.to_numeric(df["Y-Axis Value"], errors="coerce")
    df["Phase"] = pd.to_numeric(df["Phase"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.dropna(subset=["X-Axis Value", "Y-Axis Value", "Phase", "Timestamp"]).copy()
    df = df[
        df["Y-Axis Status"].astype(str).str.strip().str.lower().eq("valid")
        & df["Phase Status"].astype(str).str.strip().str.lower().eq("valid")
    ].copy()

    if df.empty:
        raise ValueError("No quedaron filas válidas después del filtrado.")

    # Mantener crudo
    raw_df = df.sort_values(["Timestamp", "X-Axis Value"]).reset_index(drop=True)

    # Agrupación robusta por velocidad
    grouped_df = (
        raw_df.groupby("X-Axis Value", as_index=False)
        .agg(
            amplitude=("Y-Axis Value", "median"),
            phase=("Phase", "median"),
            samples=("Timestamp", "size"),
            ts_min=("Timestamp", "min"),
            ts_max=("Timestamp", "max"),
        )
        .sort_values("X-Axis Value", kind="stable")
        .reset_index(drop=True)
    )

    return meta_dict, raw_df, grouped_df


# =========================
# CURSORS / HEADER
# =========================
def build_cursor_labels(grouped_df: pd.DataFrame, x_unit: str, y_unit: str) -> List[str]:
    labels: List[str] = []
    for idx, row in grouped_df.iterrows():
        labels.append(
            f"{idx:04d} | {int(round(row['X-Axis Value']))} {x_unit} | "
            f"{format_number(row['amplitude'],3)} {y_unit} | "
            f"∠{format_number(row['phase'],1)}°"
        )
    return labels


def add_cursor_markers(fig: go.Figure, x_val: float, phase_val: float, amp_val: float, color: str) -> None:
    fig.add_vline(x=x_val, line_width=1.7, line_dash="dot", line_color=color, row=1, col=1)
    fig.add_vline(x=x_val, line_width=1.7, line_dash="dot", line_color=color, row=2, col=1)

    phase_span = 18
    amp_span = 0.22

    fig.add_shape(
        type="line",
        x0=x_val,
        x1=x_val,
        y0=phase_val - phase_span,
        y1=phase_val + phase_span,
        line=dict(color=color, width=1.6),
        row=1,
        col=1,
    )
    fig.add_shape(
        type="line",
        x0=x_val - 25,
        x1=x_val + 25,
        y0=phase_val,
        y1=phase_val,
        line=dict(color=color, width=1.6),
        row=1,
        col=1,
    )

    fig.add_shape(
        type="line",
        x0=x_val,
        x1=x_val,
        y0=amp_val - amp_span,
        y1=amp_val + amp_span,
        line=dict(color=color, width=1.6),
        row=2,
        col=1,
    )
    fig.add_shape(
        type="line",
        x0=x_val - 25,
        x1=x_val + 25,
        y0=amp_val,
        y1=amp_val,
        line=dict(color=color, width=1.6),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[x_val],
            y=[phase_val],
            mode="markers",
            marker=dict(size=7, color=color, line=dict(width=1, color="#ffffff")),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[x_val],
            y=[amp_val],
            mode="markers",
            marker=dict(size=7, color=color, line=dict(width=1, color="#ffffff")),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )


def draw_top_strip(
    fig: go.Figure,
    meta: Dict[str, str],
    grouped_df: pd.DataFrame,
    logo_uri: Optional[str],
    cursor_a_row: Optional[pd.Series],
    cursor_b_row: Optional[pd.Series],
) -> None:
    x_unit = meta.get("X-Axis Unit", "rpm") or "rpm"
    y_unit = meta.get("Y-Axis Unit", "") or ""
    machine = meta.get("Machine Name", "") or ""
    point = meta.get("Point Name", "") or ""
    variable = meta.get("Variable", "") or ""
    angle = meta.get("Probe Angle", "") or ""
    x_min = grouped_df["X-Axis Value"].min()
    x_max = grouped_df["X-Axis Value"].max()

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
        xref="paper",
        yref="paper",
        x=machine_x,
        y=y_text,
        xanchor="left",
        yanchor="middle",
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
        xref="paper",
        yref="paper",
        x=0.23,
        y=y_text,
        xanchor="left",
        yanchor="middle",
        text=left_text,
        showarrow=False,
        font=dict(size=12.0, color="#111827"),
        align="left",
    )

    if cursor_a_row is not None:
        a_txt = (
            f"A: <b>{format_number(cursor_a_row['amplitude'],3)} {y_unit}</b> "
            f"∠{format_number(cursor_a_row['phase'],1)}° @ "
            f"{int(round(cursor_a_row['X-Axis Value']))} {x_unit}"
        )
    else:
        a_txt = "A: —"

    if cursor_b_row is not None:
        b_txt = (
            f"B: <b>{format_number(cursor_b_row['amplitude'],3)} {y_unit}</b> "
            f"∠{format_number(cursor_b_row['phase'],1)}° @ "
            f"{int(round(cursor_b_row['X-Axis Value']))} {x_unit}"
        )
    else:
        b_txt = "B: —"

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.50,
        y=y_text,
        xanchor="left",
        yanchor="middle",
        text=f"{a_txt} &nbsp;&nbsp;|&nbsp;&nbsp; {b_txt}",
        showarrow=False,
        font=dict(size=11.8, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.988,
        y=y_text,
        xanchor="right",
        yanchor="middle",
        text=f"{x_min:.0f} - {x_max:.0f} {x_unit}",
        showarrow=False,
        font=dict(size=11.5, color="#111827"),
        align="right",
    )


# =========================
# FIGURE
# =========================
def build_bode_figure(
    grouped_df: pd.DataFrame,
    meta: Dict[str, str],
    cursor_a_idx: Optional[int],
    cursor_b_idx: Optional[int],
    show_markers: bool,
    logo_uri: Optional[str],
) -> Tuple[go.Figure, Optional[pd.Series], Optional[pd.Series]]:
    x_unit = meta.get("X-Axis Unit", "rpm") or "rpm"
    y_unit = meta.get("Y-Axis Unit", "") or ""
    point_name = meta.get("Point Name", "") or ""

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.48, 0.52],
    )

    mode = "lines+markers" if show_markers else "lines"

    fig.add_trace(
        go.Scattergl(
            x=grouped_df["X-Axis Value"],
            y=grouped_df["phase"],
            mode=mode,
            name="Phase",
            line=dict(width=2.0, color="#5B9CF0"),
            marker=dict(size=5, color="#5B9CF0"),
            customdata=grouped_df[["samples"]],
            hovertemplate=(
                f"<b>{point_name}</b><br>"
                "Speed: %{x:.0f} " + x_unit + "<br>"
                "Phase: %{y:.1f}°<br>"
                "Grouped samples: %{customdata[0]}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=grouped_df["X-Axis Value"],
            y=grouped_df["amplitude"],
            mode=mode,
            name="Amplitude",
            line=dict(width=2.2, color="#5B9CF0"),
            marker=dict(size=5, color="#5B9CF0"),
            customdata=grouped_df[["samples"]],
            hovertemplate=(
                f"<b>{point_name}</b><br>"
                "Speed: %{x:.0f} " + x_unit + "<br>"
                "Amplitude: %{y:.3f} " + y_unit + "<br>"
                "Grouped samples: %{customdata[0]}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    cursor_a_row = grouped_df.iloc[cursor_a_idx] if cursor_a_idx is not None and 0 <= cursor_a_idx < len(grouped_df) else None
    cursor_b_row = grouped_df.iloc[cursor_b_idx] if cursor_b_idx is not None and 0 <= cursor_b_idx < len(grouped_df) else None

    if cursor_a_row is not None:
        add_cursor_markers(
            fig,
            float(cursor_a_row["X-Axis Value"]),
            float(cursor_a_row["phase"]),
            float(cursor_a_row["amplitude"]),
            "#efb08c",
        )
    if cursor_b_row is not None:
        add_cursor_markers(
            fig,
            float(cursor_b_row["X-Axis Value"]),
            float(cursor_b_row["phase"]),
            float(cursor_b_row["amplitude"]),
            "#7ac77b",
        )

    fig.update_layout(
        template="plotly_dark",
        height=820,
        margin=dict(l=50, r=28, t=145, b=50),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        dragmode="pan",
    )

    fig.update_xaxes(
        title_text=f"Speed ({x_unit})",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        row=2,
        col=1,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Phase (°)",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        row=1,
        col=1,
        range=[0, 360],
    )

    fig.update_yaxes(
        title_text=f"Amplitude ({y_unit})" if y_unit else "Amplitude",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        row=2,
        col=1,
    )

    draw_top_strip(fig, meta, grouped_df, logo_uri, cursor_a_row, cursor_b_row)
    return fig, cursor_a_row, cursor_b_row


# =========================
# EXPORT / REPORT
# =========================
def _scale_export_figure(fig: go.Figure) -> go.Figure:
    export_fig = go.Figure(fig.to_dict())

    for i, trace in enumerate(export_fig.data):
        try:
            mode = getattr(trace, "mode", "") or ""
            if "lines" in mode and getattr(trace, "line", None) is not None:
                width = getattr(trace.line, "width", 1.0) or 1.0
                export_fig.data[i].line.width = max(4.8, float(width) * 2.6)
            if "markers" in mode and getattr(trace, "marker", None) is not None:
                size = getattr(trace.marker, "size", 5) or 5
                export_fig.data[i].marker.size = max(12, float(size) * 1.8)
        except Exception:
            pass

    export_fig.update_layout(
        width=4600,
        height=2300,
        margin=dict(l=110, r=50, t=320, b=120),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        font=dict(size=30, color="#111827"),
    )
    export_fig.update_xaxes(title_font=dict(size=40), tickfont=dict(size=24))
    export_fig.update_yaxes(title_font=dict(size=40), tickfont=dict(size=24))

    for ann in export_fig.layout.annotations or []:
        if getattr(ann, "font", None) is not None:
            ann.font.size = max(22, int((ann.font.size or 12) * 2.0))

    for shp in export_fig.layout.shapes or []:
        if getattr(shp, "line", None) is not None:
            width = getattr(shp.line, "width", 1) or 1
            shp.line.width = max(2.0, width * 2.0)

    for img in export_fig.layout.images or []:
        sx = getattr(img, "sizex", None)
        sy = getattr(img, "sizey", None)
        if sx is not None:
            img.sizex = sx * 1.18
        if sy is not None:
            img.sizey = sy * 1.18

    return export_fig


def fig_to_png_bytes(fig: go.Figure, scale: int = 2) -> bytes:
    export_fig = _scale_export_figure(fig)
    return export_fig.to_image(format="png", scale=scale)


def push_bode_to_report(title: str, fig: go.Figure, meta: Dict[str, str]) -> None:
    ensure_report_state()
    item = {
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
    st.session_state["report_items"].append(item)


# =========================
# UI HELPERS
# =========================

def render_linked_analysis_banner():
    st.markdown(
        """
        <div style="
            background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(255,255,255,0.86));
            border: 1px solid #dbe3ee;
            border-radius: 18px;
            padding: 12px 16px;
            margin-bottom: 12px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        ">
            <div style="font-size:1.02rem;font-weight:800;color:#0f172a;margin-bottom:4px;">
                Linked Analysis
            </div>
            <div style="font-size:0.92rem;color:#475569;">
                Bode Plot forma parte del flujo de análisis avanzado de Watermelon. 
                Próximo paso: vinculación con Trends para correlación entre comportamiento temporal y respuesta dinámica del rotor.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_page_header() -> None:
    st.markdown('<div class="wm-page-title">Bode Plot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="wm-page-subtitle">Amplitude and phase versus speed from Bode CSV files.</div>',
        unsafe_allow_html=True,
    )


def render_panel_header(meta: Dict[str, str], raw_df: pd.DataFrame, grouped_df: pd.DataFrame) -> None:
    machine = meta.get("Machine Name", "-")
    point = meta.get("Point Name", "-")
    variable = meta.get("Variable", "-")
    probe_angle = meta.get("Probe Angle", "-")
    x_unit = meta.get("X-Axis Unit", "rpm")
    y_unit = meta.get("Y-Axis Unit", "")
    x_min = grouped_df["X-Axis Value"].min()
    x_max = grouped_df["X-Axis Value"].max()
    y_max = grouped_df["amplitude"].max()

    st.markdown(
        f"""
        <div class="wm-card">
            <div class="wm-card-title">{machine} · {point}</div>
            <div class="wm-card-subtitle">Bode run-up / coast-down view</div>
            <div class="wm-meta">
                Variable: <b>{variable}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Probe Angle: <b>{probe_angle}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Speed Range: <b>{x_min:.0f} - {x_max:.0f} {x_unit}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Peak Amplitude: <b>{y_max:.3f} {y_unit}</b>
            </div>
            <div class="wm-chip-row">
                <div class="wm-chip">Raw rows: {len(raw_df):,}</div>
                <div class="wm-chip">Grouped points: {len(grouped_df):,}</div>
                <div class="wm-chip">X Unit: {x_unit}</div>
                <div class="wm-chip">Y Unit: {y_unit}</div>
                <div class="wm-chip">Phase: degrees</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# MAIN
# =========================
def main() -> None:
    inject_styles()
    ensure_report_state()
    render_page_header()
    render_linked_analysis_banner()

    if "bode_loaded_files" not in st.session_state:
        st.session_state["bode_loaded_files"] = {}

    with st.sidebar:
        st.markdown("### Bode input")
        uploaded_files = st.file_uploader(
            "Upload one or more Bode CSV files",
            type=["csv"],
            accept_multiple_files=True,
            key="wm_bode_uploader",
        )

        st.markdown("### Processing")
        show_raw_table = st.toggle("Show grouped table", value=False)
        export_scale = st.selectbox("PNG scale", [2, 3, 4], index=1)
        show_markers = st.toggle("Show markers", value=True)

    if not uploaded_files:
        st.info("Carga uno o más archivos CSV de Bode para visualizar amplitud y fase contra velocidad.")
        return

    logo_uri = get_logo_data_uri(LOGO_PATH)

    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown("---")

        try:
            meta, raw_df, grouped_df = load_bode_csv(uploaded_file)
        except Exception as e:
            st.error(f"No pude procesar {uploaded_file.name}: {e}")
            continue

        render_panel_header(meta, raw_df, grouped_df)

        x_unit = meta.get("X-Axis Unit", "rpm")
        y_unit = meta.get("Y-Axis Unit", "")

        cursor_labels = build_cursor_labels(grouped_df, x_unit, y_unit)
        default_a_idx = int(grouped_df["amplitude"].idxmax()) if not grouped_df.empty else 0
        default_b_idx = min(len(grouped_df) - 1, default_a_idx + max(1, len(grouped_df) // 8)) if not grouped_df.empty else 0

        csel1, csel2 = st.columns(2)
        with csel1:
            cursor_a_label = st.selectbox(
                f"Cursor A · {uploaded_file.name}",
                options=cursor_labels,
                index=default_a_idx,
                key=f"bode_cursor_a_{idx}",
            )
        with csel2:
            cursor_b_label = st.selectbox(
                f"Cursor B · {uploaded_file.name}",
                options=cursor_labels,
                index=default_b_idx,
                key=f"bode_cursor_b_{idx}",
            )

        cursor_a_idx = cursor_labels.index(cursor_a_label)
        cursor_b_idx = cursor_labels.index(cursor_b_label)

        fig, cursor_a_row, cursor_b_row = build_bode_figure(
            grouped_df,
            meta,
            cursor_a_idx=cursor_a_idx,
            cursor_b_idx=cursor_b_idx,
            show_markers=show_markers,
            logo_uri=logo_uri,
        )
        st.plotly_chart(fig, width="stretch", key=f"bode_fig_{idx}", config={"displaylogo": False})

        title = f"Bode - {meta.get('Machine Name', '')} - {meta.get('Point Name', '')} - {meta.get('Variable', '')}".strip(" -")

        try:
            png_bytes = fig_to_png_bytes(fig, scale=export_scale)
        except Exception as e:
            png_bytes = None
            st.warning(f"No pude generar el PNG HD. Detalle: {e}")

        c1, c2, c3 = st.columns([1.2, 1.2, 4.6])

        with c1:
            if png_bytes is not None:
                st.download_button(
                    "Export PNG HD",
                    data=png_bytes,
                    file_name=f"{Path(uploaded_file.name).stem}_bode_hd.png",
                    mime="image/png",
                    key=f"download_bode_{idx}",
                    width="stretch",
                )

        with c2:
            if st.button("Enviar a Reporte", key=f"send_bode_{idx}", width="stretch"):
                push_bode_to_report(title=title, fig=fig, meta=meta)
                st.success("Bode enviado al reporte.")

        with c3:
            st.caption(
                f"Archivo: {uploaded_file.name} · {len(raw_df):,} filas crudas · {len(grouped_df):,} puntos agrupados por velocidad"
            )

        with st.expander(f"Cursor summary · {uploaded_file.name}", expanded=False):
            if cursor_a_row is not None:
                st.markdown(
                    f"**Cursor A:** {format_number(cursor_a_row['amplitude'],3)} {y_unit} "
                    f"∠{format_number(cursor_a_row['phase'],1)}° @ "
                    f"{int(round(cursor_a_row['X-Axis Value']))} {x_unit}"
                )
            if cursor_b_row is not None:
                st.markdown(
                    f"**Cursor B:** {format_number(cursor_b_row['amplitude'],3)} {y_unit} "
                    f"∠{format_number(cursor_b_row['phase'],1)}° @ "
                    f"{int(round(cursor_b_row['X-Axis Value']))} {x_unit}"
                )

        if show_raw_table:
            with st.expander(f"Grouped data · {uploaded_file.name}", expanded=False):
                st.dataframe(
                    grouped_df.rename(
                        columns={
                            "X-Axis Value": f"Speed ({x_unit})",
                            "amplitude": f"Amplitude ({y_unit})",
                            "phase": "Phase (°)",
                            "samples": "Grouped Samples",
                            "ts_min": "First Timestamp",
                            "ts_max": "Last Timestamp",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
