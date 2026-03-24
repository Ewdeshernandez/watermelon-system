from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.auth import require_login, render_user_menu
from core.orbit import compute_orbit

st.set_page_config(page_title="Watermelon System | Orbit Analysis", layout="wide")

require_login()
render_user_menu()

# ============================================================
# WATERMELON SYSTEM — ORBIT ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"


def apply_page_style():
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
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_page_style()


def get_logo_base64(path: Path):
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def get_logo_data_uri(path: Path):
    b64 = get_logo_base64(path)
    if not b64:
        return None
    return f"data:image/png;base64,{b64}"


def rounded_rect_path(x0, y0, x1, y1, r):
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


def format_number(value, digits=4, fallback="—"):
    if value is None:
        return fallback
    try:
        val = float(value)
        if not np.isfinite(val):
            return fallback
        return f"{val:.{digits}f}"
    except Exception:
        return fallback


def _signals_dict():
    signals = st.session_state.get("signals", {})
    return signals if isinstance(signals, dict) else {}


def _default_signal_pair(signals):
    names = list(signals.keys())
    if len(names) < 2:
        raise ValueError("At least two signals are required.")

    def rank_x(name):
        upper = name.upper()
        return (1 if "X" in upper else 0, 1 if "GEN" in upper else 0)

    def rank_y(name):
        upper = name.upper()
        return (1 if "Y" in upper else 0, 1 if "GEN" in upper else 0)

    x_name = sorted(names, key=lambda n: (-rank_x(n)[0], -rank_x(n)[1], n))[0]
    y_name = sorted(names, key=lambda n: (-rank_y(n)[0], -rank_y(n)[1], n))[0]

    if x_name == y_name:
        for candidate in names:
            if candidate != x_name:
                y_name = candidate
                break

    return x_name, y_name


def make_export_state_key(parts):
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def safe_slug(text: str) -> str:
    text = (text or "").strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "orbit"


def _draw_top_strip(fig, orbit_result, ui_filter_mode, logo_uri):
    machine = orbit_result.probe_state.get("machine_name", "Orbit")
    x_channel = orbit_result.probe_state.get("x_channel", "X")
    y_channel = orbit_result.probe_state.get("y_channel", "Y")
    timestamp = orbit_result.probe_state.get("timestamp", "—")
    rpm_text = f"{format_number(orbit_result.rpm, 0)} rpm" if orbit_result.rpm is not None else "rpm —"

    sentido = orbit_result.traversal
    precession = orbit_result.precession

    mode_label = {
        "Direct": "Orbit Direct",
        "1X": "Orbit 1X",
        "2X": "Orbit 2X",
    }.get(ui_filter_mode, f"Orbit {ui_filter_mode}")

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
        text=f"<b>{machine}</b>",
        showarrow=False,
        font=dict(size=12.8, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.255,
        y=y_text,
        xanchor="left",
        yanchor="middle",
        text=f"{x_channel} + {y_channel} | {mode_label}",
        showarrow=False,
        font=dict(size=12.0, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.535,
        y=y_text,
        xanchor="left",
        yanchor="middle",
        text=f"Sentido de Giro: <b>{sentido}</b>",
        showarrow=False,
        font=dict(size=12.0, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.675,
        y=y_text,
        xanchor="left",
        yanchor="middle",
        text=f"Precesión: <b>{precession}</b>",
        showarrow=False,
        font=dict(size=12.0, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.845,
        y=y_text,
        xanchor="left",
        yanchor="middle",
        text=rpm_text,
        showarrow=False,
        font=dict(size=12.0, color="#111827"),
        align="left",
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.986,
        y=y_text,
        xanchor="right",
        yanchor="middle",
        text=timestamp,
        showarrow=False,
        font=dict(size=11.8, color="#111827"),
        align="right",
    )


def _draw_right_info_box(fig, orbit_result, ui_filter_mode):
    units = orbit_result.probe_state.get("units", "mil")

    rows = [
        (orbit_result.probe_state.get("x_probe_label", "X"), ""),
        (orbit_result.probe_state.get("y_probe_label", "Y"), ""),
    ]

    if ui_filter_mode == "Direct":
        rows.extend(
            [
                ("Amplitud X", f"{format_number(orbit_result.diagnostics.get('x_wf_amp_pkpk'), 3)} {units} pp"),
                ("Amplitud Y", f"{format_number(orbit_result.diagnostics.get('y_wf_amp_pkpk'), 3)} {units} pp"),
                ("Revoluciones", str(int(orbit_result.diagnostics.get("display_revolutions_raw", 1)))),
            ]
        )
    else:
        rows.extend(
            [
                ("Amplitud X", f"{format_number(orbit_result.diagnostics.get('x_harmonic_amplitude_mean'), 3)} {units} pp"),
                ("Amplitud Y", f"{format_number(orbit_result.diagnostics.get('y_harmonic_amplitude_mean'), 3)} {units} pp"),
                ("Revoluciones", str(int(orbit_result.diagnostics.get("displayed_revolutions_filtered", 1)))),
            ]
        )

    panel_x0 = 0.836
    panel_x1 = 0.975
    panel_y1 = 0.915
    header_h = 0.034
    row_h = 0.073
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

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=(panel_x0 + panel_x1) / 2.0,
        y=panel_y1 - header_h / 2.0,
        xanchor="center",
        yanchor="middle",
        text="<b>Orbit Information</b>",
        showarrow=False,
        font=dict(size=11.4, color="#111827"),
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

        if str(value).strip():
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


def build_orbit_figure(orbit_result, ui_filter_mode, logo_uri, scale_mode, manual_scale_value):
    units = orbit_result.probe_state.get("units", "mil")

    fig = go.Figure()
    line_color = "#5b6df0"
    start_color = "#2f80ed"

    for seg_x, seg_y in zip(orbit_result.segment_x_open, orbit_result.segment_y_open):
        fig.add_trace(
            go.Scattergl(
                x=seg_x,
                y=seg_y,
                mode="lines",
                line=dict(width=2.0, color=line_color),
                hovertemplate=f"X: %{{x:.4f}} {units}<br>Y: %{{y:.4f}} {units}<extra></extra>",
                showlegend=False,
                connectgaps=False,
                name="orbit_segment",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[orbit_result.start_point[0]],
            y=[orbit_result.start_point[1]],
            mode="markers",
            marker=dict(symbol="circle", size=8, color=start_color, line=dict(width=1.2, color="#ffffff")),
            showlegend=False,
            hovertemplate=f"Inicio órbita<br>X: %{{x:.4f}} {units}<br>Y: %{{y:.4f}} {units}<extra></extra>",
            name="start_marker",
        )
    )

    all_x = np.concatenate([np.asarray(seg, dtype=float) for seg in orbit_result.segment_x_open])
    all_y = np.concatenate([np.asarray(seg, dtype=float) for seg in orbit_result.segment_y_open])

    finite_mask = np.isfinite(all_x) & np.isfinite(all_y)
    if np.any(finite_mask):
        data = np.concatenate([all_x[finite_mask], all_y[finite_mask]])
        max_abs = float(np.max(np.abs(data)))
    else:
        max_abs = 1.0

    max_abs = max(max_abs, 1e-6)

    if scale_mode == "Manual":
        lim = max(float(manual_scale_value), 1e-6)
    else:
        lim = max_abs * 1.14

    _draw_top_strip(fig, orbit_result, ui_filter_mode, logo_uri)
    _draw_right_info_box(fig, orbit_result, ui_filter_mode)

    fig.update_layout(
        height=700,
        margin=dict(l=46, r=18, t=84, b=40),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f3f4f6",
        font=dict(color="#111827"),
        xaxis=dict(
            title=f"X ({units})",
            range=[-lim, lim],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.18)",
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.35)",
            showline=True,
            linecolor="#9ca3af",
            mirror=False,
            ticks="outside",
            tickcolor="#6b7280",
            ticklen=4,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title=f"Y ({units})",
            range=[-lim, lim],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.18)",
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.35)",
            showline=True,
            linecolor="#9ca3af",
            mirror=False,
            ticks="outside",
            tickcolor="#6b7280",
            ticklen=4,
        ),
        hovermode="closest",
        showlegend=False,
    )

    return fig


def _build_export_safe_figure(fig):
    export_fig = go.Figure()

    for trace in fig.data:
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
                    name=trace_json.get("name", ""),
                )
            )
        else:
            export_fig.add_trace(trace)

    export_fig.update_layout(fig.layout)
    return export_fig


def _scale_export_figure(export_fig):
    fig = go.Figure(export_fig)

    new_data = []
    for trace in fig.data:
        trace_json = trace.to_plotly_json()
        if trace_json.get("type") == "scatter":
            mode = trace_json.get("mode", "")
            if "lines" in mode:
                line = dict(trace_json.get("line", {}) or {})
                line["width"] = max(5.0, float(line.get("width", 1.0)) * 2.7)
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

    fig.update_xaxes(title_font=dict(size=40), tickfont=dict(size=26))
    fig.update_yaxes(title_font=dict(size=40), tickfont=dict(size=26))

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


def build_export_png_bytes(fig):
    try:
        export_fig = _build_export_safe_figure(fig)
        export_fig = _scale_export_figure(export_fig)
        png_bytes = export_fig.to_image(format="png", width=4200, height=2200, scale=2)
        return png_bytes, None
    except Exception as e:
        return None, str(e)


@dataclass
class OrbitPair:
    label: str
    x_name: str
    y_name: str


def build_orbit_pairs(signals: dict) -> List[OrbitPair]:
    names = list(signals.keys())
    if len(names) < 2:
        return []

    used = set()
    pairs: List[OrbitPair] = []

    for name in names:
        upper = name.upper()
        if "X" not in upper or name in used:
            continue

        prefix = upper.replace("X", "")
        candidate_y = None

        for other in names:
            if other == name or other in used:
                continue
            other_upper = other.upper()
            if "Y" in other_upper and other_upper.replace("Y", "") == prefix:
                candidate_y = other
                break

        if candidate_y is None:
            for other in names:
                if other == name or other in used:
                    continue
                if "Y" in other.upper():
                    candidate_y = other
                    break

        if candidate_y is not None:
            used.add(name)
            used.add(candidate_y)
            pairs.append(
                OrbitPair(
                    label=f"{name} + {candidate_y}",
                    x_name=name,
                    y_name=candidate_y,
                )
            )

    if not pairs:
        default_x, default_y = _default_signal_pair(signals)
        pairs.append(OrbitPair(label=f"{default_x} + {default_y}", x_name=default_x, y_name=default_y))

    return pairs


def queue_orbit_to_report(pair: OrbitPair, fig: go.Figure, panel_title: str, result) -> None:
    st.session_state.report_items.append(
        {
            "id": make_export_state_key(
                [
                    "report-orbit",
                    pair.x_name,
                    pair.y_name,
                    result.probe_state.get("timestamp"),
                    panel_title,
                    len(st.session_state.report_items),
                ]
            ),
            "type": "orbit",
            "title": panel_title,
            "notes": "",
            "signal_id": f"{pair.x_name}|{pair.y_name}",
            "figure": go.Figure(fig),
            "machine": result.probe_state.get("machine_name", ""),
            "point": f"{pair.x_name} + {pair.y_name}",
            "variable": "Orbit",
            "timestamp": str(result.probe_state.get("timestamp", "") or ""),
        }
    )


def render_orbit_panel(
    pair: OrbitPair,
    signals: dict,
    panel_index: int,
    *,
    ui_filter_mode: str,
    machine_rotation: str,
    scale_mode: str,
    manual_scale_value: float,
    logo_uri: Optional[str],
) -> None:
    x_signal = signals[pair.x_name]
    y_signal = signals[pair.y_name]

    result = compute_orbit(
        x_signal,
        y_signal,
        filter_mode=ui_filter_mode,
        machine_rotation=machine_rotation,
        x_probe_angle_deg=45.0,
        x_probe_side="Right",
        y_probe_angle_deg=45.0,
        y_probe_side="Left",
        samples_per_rev=None,
        revolution_index=0,
        display_revolutions_raw=None,
        average_revolutions_filtered=None,
        harmonic_plot_samples=720,
        rpm_override=None,
    )

    fig = build_orbit_figure(
        orbit_result=result,
        ui_filter_mode=ui_filter_mode,
        logo_uri=logo_uri,
        scale_mode=scale_mode,
        manual_scale_value=manual_scale_value,
    )

    export_state_key = make_export_state_key(
        [
            pair.x_name,
            pair.y_name,
            panel_index,
            ui_filter_mode,
            machine_rotation,
            result.samples_per_rev,
            result.revolutions_available,
            scale_mode,
            manual_scale_value,
            result.traversal,
            result.precession,
            result.diagnostics.get("x_wf_amp_pkpk"),
            result.diagnostics.get("y_wf_amp_pkpk"),
            result.diagnostics.get("x_harmonic_amplitude_mean"),
            result.diagnostics.get("y_harmonic_amplitude_mean"),
        ]
    )

    if export_state_key not in st.session_state.wm_orbit_export_store:
        st.session_state.wm_orbit_export_store[export_state_key] = {
            "png_bytes": None,
            "error": None,
        }

    panel_title = f"Orbit {panel_index + 1} — {pair.label}"
    st.markdown(f"### {panel_title}")

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False},
        key=f"wm_orbit_plot_{export_state_key}",
    )

    st.markdown('<div class="wm-export-actions"></div>', unsafe_allow_html=True)

    left_pad, col_export1, col_export2, col_report, right_pad = st.columns([2.0, 1.2, 1.2, 1.2, 2.0])

    with col_export1:
        if st.button("Prepare PNG HD", key=f"prepare_png_{export_state_key}", use_container_width=True):
            with st.spinner("Generating HD export..."):
                png_bytes, export_error = build_export_png_bytes(fig)
                st.session_state.wm_orbit_export_store[export_state_key]["png_bytes"] = png_bytes
                st.session_state.wm_orbit_export_store[export_state_key]["error"] = export_error

    with col_export2:
        png_bytes = st.session_state.wm_orbit_export_store[export_state_key]["png_bytes"]
        if png_bytes is not None:
            st.download_button(
                "Download PNG HD",
                data=png_bytes,
                file_name=f"{safe_slug(pair.x_name)}_{safe_slug(pair.y_name)}_orbit_hd.png",
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
            queue_orbit_to_report(pair, fig, panel_title, result)
            st.success("Orbit enviada al reporte")

    panel_error = st.session_state.wm_orbit_export_store[export_state_key]["error"]
    if panel_error:
        st.warning(f"PNG export error: {panel_error}")


if "wm_orbit_selected_labels" not in st.session_state:
    st.session_state.wm_orbit_selected_labels = []
if "wm_orbit_export_store" not in st.session_state:
    st.session_state.wm_orbit_export_store = {}
if "report_items" not in st.session_state:
    st.session_state.report_items = []

signals = _signals_dict()

if not signals:
    st.warning("No se pudieron cargar señales válidas desde `st.session_state['signals']`.")
    st.stop()

if len(signals) < 2:
    st.warning("Orbit necesita mínimo dos señales cargadas.")
    st.stop()

pairs = build_orbit_pairs(signals)
if not pairs:
    st.warning("No fue posible construir pares X/Y para órbitas.")
    st.stop()

pair_label_map = {pair.label: pair for pair in pairs}
pair_labels = list(pair_label_map.keys())

with st.sidebar:
    st.markdown("### Orbit Processing")

    valid_labels = [label for label in st.session_state.wm_orbit_selected_labels if label in pair_label_map]
    if not valid_labels:
        valid_labels = [pair_labels[0]]
        st.session_state.wm_orbit_selected_labels = valid_labels

    selected_labels = st.multiselect(
        "Orbits to display",
        options=pair_labels,
        default=valid_labels,
    )
    st.session_state.wm_orbit_selected_labels = selected_labels

    ui_filter_mode = st.selectbox("Filter", ["Direct", "1X", "2X"], index=0)
    machine_rotation = st.selectbox("Machine rotation", ["CW", "CCW"], index=1)

    scale_mode = st.selectbox("Scale", ["Auto", "Manual"], index=0)
    manual_scale_value = 2.0
    if scale_mode == "Manual":
        manual_scale_value = float(
            st.number_input(
                "Manual symmetric scale",
                min_value=0.001,
                value=2.0,
                step=0.1,
                format="%.3f",
            )
        )

selected_pairs = [pair_label_map[label] for label in st.session_state.wm_orbit_selected_labels if label in pair_label_map]

if not selected_pairs:
    st.info("Selecciona una o más órbitas en la barra lateral.")
    st.stop()

logo_uri = get_logo_data_uri(LOGO_PATH)

for panel_index, pair in enumerate(selected_pairs):
    render_orbit_panel(
        pair=pair,
        signals=signals,
        panel_index=panel_index,
        ui_filter_mode=ui_filter_mode,
        machine_rotation=machine_rotation,
        scale_mode=scale_mode,
        manual_scale_value=manual_scale_value,
        logo_uri=logo_uri,
    )

    if panel_index < len(selected_pairs) - 1:
        st.markdown("---")
