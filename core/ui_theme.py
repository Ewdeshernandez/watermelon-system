from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import plotly.graph_objects as go
import streamlit as st


def apply_watermelon_page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.15rem;
            padding-bottom: 2.0rem;
            max-width: 100%;
        }

        .wm-page-title {
            font-size: 2.15rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.15rem;
            letter-spacing: -0.02em;
        }

        .wm-page-subtitle {
            color: #64748b;
            font-size: 0.98rem;
            margin-bottom: 1.15rem;
        }

        .wm-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.88));
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


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="wm-page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="wm-page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def card(title: str, subtitle: str = "", meta_html: str = "", chips: Optional[Sequence[str]] = None) -> None:
    chips = list(chips or [])
    chip_html = ""
    if chips:
        chip_html = '<div class="wm-chip-row">' + "".join([f'<div class="wm-chip">{c}</div>' for c in chips]) + "</div>"

    subtitle_html = f'<div class="wm-card-subtitle">{subtitle}</div>' if subtitle else ""
    meta_block = f'<div class="wm-meta">{meta_html}</div>' if meta_html else ""

    st.markdown(
        f"""
        <div class="wm-card">
            <div class="wm-card-title">{title}</div>
            {subtitle_html}
            {meta_block}
            {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(title: str, subtitle: str, chips: Sequence[Tuple[str, Optional[str]]]) -> None:
    chip_html = []
    for text, color in chips:
        style = ""
        if color:
            style = f' style="color:{color}; border-color:{color};"'
        chip_html.append(f'<div class="wm-chip"{style}>{text}</div>')

    st.markdown(
        f"""
        <div class="wm-card">
            <div class="wm-card-title">{title}</div>
            <div class="wm-card-subtitle">{subtitle}</div>
            <div class="wm-chip-row">
                {''.join(chip_html)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def draw_top_strip(
    fig: go.Figure,
    machine: str,
    point_text: str,
    variable: str,
    dt_text: str,
    rpm_text: str,
    logo_uri: Optional[str] = None,
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

    machine_x = 0.082 if logo_uri else 0.020

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
        text=point_text,
        showarrow=False,
        font=dict(size=11.7, color="#111827"),
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.43, y=y_text,
        xanchor="left", yanchor="middle",
        text=variable,
        showarrow=False,
        font=dict(size=11.5, color="#111827"),
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.66, y=y_text,
        xanchor="left", yanchor="middle",
        text=dt_text,
        showarrow=False,
        font=dict(size=10.8, color="#111827"),
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.986, y=y_text,
        xanchor="right", yanchor="middle",
        text=rpm_text,
        showarrow=False,
        font=dict(size=10.8, color="#111827"),
    )


def draw_info_box(fig: go.Figure, title: str, rows: Sequence[Tuple[str, str]]) -> None:
    panel_x0 = 0.855
    panel_x1 = 0.970
    panel_y1 = 0.950
    header_h = 0.022
    row_h = 0.033
    body_pad = 0.008
    radius = 0.008

    panel_h = header_h + len(rows) * row_h + body_pad
    panel_y0 = panel_y1 - panel_h

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path=rounded_rect_path(panel_x0, panel_y0, panel_x1, panel_y1, radius),
        line=dict(color="rgba(203,213,225,0.90)", width=1),
        fillcolor="rgba(255,255,255,0.78)",
        layer="above",
    )

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path=rounded_rect_path(panel_x0, panel_y1 - header_h, panel_x1, panel_y1, radius),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(147,197,253,0.88)",
        layer="above",
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=(panel_x0 + panel_x1) / 2.0,
        y=panel_y1 - header_h / 2.0,
        text=f"<b>{title}</b>",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font=dict(size=8.8, color="#111827"),
    )

    current_top = panel_y1 - header_h - 0.004

    for label, value in rows:
        title_y = current_top
        value_y = current_top - 0.013

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.008,
            y=title_y,
            xanchor="left",
            yanchor="top",
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=7.8, color="#111827"),
            align="left",
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.008,
            y=value_y,
            xanchor="left",
            yanchor="top",
            text=value,
            showarrow=False,
            font=dict(size=7.2, color="#111827"),
            align="left",
        )

        current_top -= row_h
