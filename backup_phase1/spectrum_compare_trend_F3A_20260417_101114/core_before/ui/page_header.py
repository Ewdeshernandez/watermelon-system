from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Optional

import streamlit as st


def _get_logo_base64() -> Optional[str]:
    root = Path(__file__).resolve().parents[2]
    logo_path = root / "assets" / "watermelon_logo.png"

    if not logo_path.exists():
        return None

    return base64.b64encode(logo_path.read_bytes()).decode("utf-8")


def _safe(value: str) -> str:
    return html.escape(value, quote=True)


def render_page_header(
    title: str,
    subtitle: str,
    section: str = "Watermelon System",
    badge: Optional[str] = None,
    align: str = "left",
) -> None:
    logo_b64 = _get_logo_base64()

    normalized_align = "center" if align == "center" else "left"
    text_align = "center" if normalized_align == "center" else "left"
    row_justify = "center" if normalized_align == "center" else "flex-start"
    topline_justify = "center" if normalized_align == "center" else "space-between"
    mobile_align = "center" if normalized_align == "center" else "flex-start"
    subtitle_max_width = "920px" if normalized_align == "center" else "1100px"
    subtitle_auto_margin = "margin-left:auto; margin-right:auto;" if normalized_align == "center" else ""

    safe_section = _safe(section)
    safe_title = _safe(title)
    safe_subtitle = _safe(subtitle)

    badge_html = ""
    if badge:
        safe_badge = _safe(badge)
        badge_html = f"""
            <div class="wm-header-badge">
                {safe_badge}
            </div>
        """

    logo_html = ""
    if logo_b64:
        logo_html = f"""
            <div class="wm-header-logo-wrap">
                <img class="wm-header-logo" src="data:image/png;base64,{logo_b64}" alt="Watermelon System logo" />
            </div>
        """

    html_block = f"""
    <style>
    .wm-header-shell {{
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(248,250,252,0.93) 52%, rgba(241,245,249,0.95) 100%);
        border: 1px solid #d8e0ea;
        border-radius: 28px;
        padding: 28px 32px;
        margin-bottom: 1.45rem;
        box-shadow:
            0 18px 45px rgba(15, 23, 42, 0.06),
            inset 0 1px 0 rgba(255,255,255,0.78);
    }}

    .wm-header-shell::before {{
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 6px;
        background: linear-gradient(180deg, #2563eb 0%, #38bdf8 100%);
        border-radius: 28px 0 0 28px;
    }}

    .wm-header-shell::after {{
        content: "";
        position: absolute;
        top: -55px;
        right: -55px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, rgba(59,130,246,0.02) 42%, transparent 72%);
        pointer-events: none;
    }}

    .wm-header-row {{
        display: flex;
        align-items: center;
        justify-content: {row_justify};
        gap: 24px;
    }}

    .wm-header-logo-wrap {{
        flex: 0 0 auto;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 104px;
        height: 104px;
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.95) 100%);
        border: 1px solid #dde5ef;
        box-shadow:
            0 10px 24px rgba(37,99,235,0.06),
            inset 0 1px 0 rgba(255,255,255,0.78);
    }}

    .wm-header-logo {{
        max-height: 74px;
        width: auto;
        display: block;
        filter: saturate(1.05) contrast(1.02);
    }}

    .wm-header-copy {{
        min-width: 0;
        flex: 1 1 auto;
        text-align: {text_align};
    }}

    .wm-header-topline {{
        display: flex;
        align-items: center;
        justify-content: {topline_justify};
        gap: 14px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }}

    .wm-header-kicker {{
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #3b82f6;
        margin: 0;
    }}

    .wm-header-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        border-radius: 999px;
        border: 1px solid #d6e4ff;
        background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
        color: #1d4ed8;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.85);
    }}

    .wm-header-title {{
        font-size: clamp(2.55rem, 5vw, 4.25rem);
        line-height: 0.98;
        font-weight: 900;
        letter-spacing: -0.04em;
        color: #2f3343;
        margin: 0;
    }}

    .wm-header-subtitle {{
        margin-top: 14px;
        font-size: 1.10rem;
        line-height: 1.45;
        color: #526174;
        max-width: {subtitle_max_width};
        {subtitle_auto_margin}
    }}

    .wm-header-divider {{
        margin-top: 18px;
        height: 1px;
        width: 100%;
        background: linear-gradient(90deg, rgba(59,130,246,0.20) 0%, rgba(203,213,225,0.55) 38%, rgba(203,213,225,0.12) 100%);
        border-radius: 999px;
    }}

    @media (max-width: 900px) {{
        .wm-header-row {{
            flex-direction: column;
            align-items: {mobile_align};
        }}

        .wm-header-logo-wrap {{
            width: 92px;
            height: 92px;
        }}

        .wm-header-logo {{
            max-height: 64px;
        }}

        .wm-header-title {{
            font-size: 2.6rem;
        }}

        .wm-header-subtitle {{
            font-size: 1rem;
        }}
    }}
    </style>

    <div class="wm-header-shell">
        <div class="wm-header-row">
            {logo_html}
            <div class="wm-header-copy">
                <div class="wm-header-topline">
                    <div class="wm-header-kicker">{safe_section}</div>
                    {badge_html}
                </div>
                <div class="wm-header-title">{safe_title}</div>
                <div class="wm-header-subtitle">{safe_subtitle}</div>
                <div class="wm-header-divider"></div>
            </div>
        </div>
    </div>
    """

    st.markdown(html_block, unsafe_allow_html=True)


def render_header(title: str) -> None:
    render_page_header(
        title=title,
        subtitle="Industrial Phase Analysis • 1X Extraction • Physical Reference",
        section="Watermelon System",
        badge="PHASE",
        align="left",
    )