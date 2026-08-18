"""
core/balance/ui.py — Kit visual del módulo Balanceo
===================================================

Componentes Streamlit estilo enterprise, consistentes con el resto de
Watermelon (misma paleta y lenguaje que core/modal/ui_components). Nada de
gradients pesados ni emojis abusivos: la calidad "software internacional" viene
de jerarquía visual clara, KPIs grandes, cita normativa inline y estados de
color legibles.

Componentes
-----------
bal_hero_card       — Banner del módulo con activo + modo activo (1p/2p/manual)
bal_section_header  — Header de sección con cita normativa
bal_kpi_row         — Fila de KPI cards (valor grande + label + sublabel)
bal_status_banner   — Banner ok/warning/fail/info
bal_footer_norms    — Footer normativo permanente (ISO 21940 / API 684 + version)
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import streamlit as st


# Paleta — idéntica a core/modal/ui_components para consistencia total.
NAVY = "#0F1E3D"
CYAN = "#1AAEE5"
CYAN_DARK = "#0F7FB0"
AMBER = "#D89B22"
AMBER_LIGHT = "#FEF3C7"
GREEN = "#16a34a"
GREEN_LIGHT = "#DCFCE7"
RED = "#dc2626"
RED_LIGHT = "#FEE2E2"
GRAY = "#6B7280"
GRAY_LIGHT = "#F4F7FB"


def bal_hero_card(asset_name: str = "(sin activo)", client: str = "",
                  site: str = "", mode: str = "—") -> None:
    """Banner del módulo: identidad + activo + modo activo (Manual / 1 plano /
    2 planos). Mismo lenguaje que el hero de Modal."""
    sub = " · ".join([p for p in [f"<b>{client}</b>" if client else "", site] if p]) \
        or "Balanceo de rotores por coeficiente de influencia"
    mode_color = {"1 PLANO": CYAN_DARK, "2 PLANOS": GREEN}.get(mode.upper(), GRAY)
    st.markdown(
        f"""
        <div style="background:{NAVY}; color:white; padding:22px 28px;
             border-radius:14px; margin-bottom:18px; display:flex;
             justify-content:space-between; align-items:center; gap:24px;">
          <div style="flex:1;">
            <div style="font-size:11px; font-weight:700; letter-spacing:0.18em;
                 text-transform:uppercase; color:{CYAN}; margin-bottom:4px;">
                 Módulo Balanceo</div>
            <div style="font-size:24px; font-weight:800; line-height:1.2;
                 margin-bottom:4px;">{asset_name}</div>
            <div style="font-size:13px; color:rgba(226,232,240,0.85);">{sub}</div>
          </div>
          <div style="text-align:right; min-width:170px;">
            <div style="display:inline-block; background:{mode_color}; color:white;
                 font-weight:700; font-size:13px; padding:4px 12px; border-radius:999px;
                 letter-spacing:0.1em;">{mode.upper()}</div>
            <div style="font-size:11px; color:rgba(226,232,240,0.7);
                 font-family:monospace; margin-top:6px;">ISO 21940 · API 684</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bal_section_header(title: str, subtitle: str = "", norm_ref: str = "",
                       icon: str = "") -> None:
    icon_html = (f'<span style="font-size:18px; margin-right:8px;">{icon}</span>'
                 if icon else "")
    norm_html = (
        f'<span style="font-size:11px; color:{GRAY}; font-weight:500; '
        f'font-family:monospace; margin-left:10px; padding:2px 8px; '
        f'background:{GRAY_LIGHT}; border-radius:4px;">{norm_ref}</span>'
        if norm_ref else "")
    st.markdown(
        f"""
        <div style="margin:14px 0 10px 0;">
          <div style="display:flex; align-items:center; color:{NAVY};
               font-size:17px; font-weight:700; line-height:1.3;">
            {icon_html}{title}{norm_html}
          </div>
          {f'<div style="color:{GRAY}; font-size:12.5px; margin-top:4px;">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def bal_kpi_row(metrics: List[Tuple[str, str, str, str]]) -> None:
    """metrics: [(value, label, sublabel, color)] con color en
    cyan|green|amber|red|navy|gray."""
    color_map = {
        "cyan": (CYAN_DARK, "#E0F2FE"), "green": (GREEN, GREEN_LIGHT),
        "amber": (AMBER, AMBER_LIGHT), "red": (RED, RED_LIGHT),
        "navy": (NAVY, GRAY_LIGHT), "gray": (GRAY, GRAY_LIGHT),
    }
    cols = st.columns(len(metrics))
    for col, (value, label, sublabel, color) in zip(cols, metrics):
        fg, bg = color_map.get(color, color_map["navy"])
        with col:
            st.markdown(
                f"""
                <div style="background:{bg}; border-left:4px solid {fg};
                     padding:14px 16px; border-radius:6px; min-height:96px;">
                  <div style="font-size:26px; font-weight:800; color:{fg};
                       line-height:1.05; margin-bottom:6px;">{value}</div>
                  <div style="font-size:12px; font-weight:700; color:{NAVY};
                       text-transform:uppercase; letter-spacing:0.06em;
                       margin-bottom:2px;">{label}</div>
                  <div style="font-size:11px; color:{GRAY}; line-height:1.3;">{sublabel}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def bal_status_banner(title: str, detail: str = "", severity: str = "info") -> None:
    cfg = {
        "ok": (GREEN, GREEN_LIGHT, "#14532D", "✓"),
        "warning": (AMBER, AMBER_LIGHT, "#78350F", "⚠"),
        "fail": (RED, RED_LIGHT, "#7F1D1D", "✗"),
        "info": (CYAN_DARK, "#DBEAFE", "#1E3A8A", "ℹ"),
    }
    border, bg, fg, icon = cfg.get(severity, cfg["info"])
    st.markdown(
        f"""
        <div style="background:{bg}; border:1.5px solid {border}; border-radius:8px;
             padding:13px 18px; margin-bottom:14px; display:flex; gap:14px;
             align-items:flex-start;">
          <div style="font-size:22px; color:{border}; line-height:1;">{icon}</div>
          <div style="flex:1;">
            <div style="font-weight:700; color:{fg}; font-size:14px;
                 margin-bottom:2px;">{title}</div>
            {f'<div style="color:{fg}; font-size:12.5px; line-height:1.5; opacity:0.85;">{detail}</div>' if detail else ''}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bal_footer_norms(version: Optional[str] = None) -> None:
    if version is None:
        try:
            from core.version import get_version_short
            version = get_version_short()
        except Exception:
            version = "v?"
    norms = ["ISO 21940-11 (desbalance residual)",
             "ISO 21940-12 (balanceo multiplano)", "API 684 (peso de prueba)"]
    st.markdown(
        f"""
        <div style="margin-top:30px; padding:14px 18px; background:{GRAY_LIGHT};
             border-top:2px solid {CYAN}; border-radius:6px; font-size:11px;
             color:{GRAY}; line-height:1.6;">
          <div style="font-weight:700; color:{NAVY}; letter-spacing:0.1em;
               text-transform:uppercase; font-size:10px; margin-bottom:6px;">
               Marco normativo aplicado</div>
          <div>{' &nbsp;·&nbsp; '.join(f'<b style="color:{NAVY};">{n}</b>' for n in norms)}</div>
          <div style="margin-top:12px; padding-top:8px; border-top:1px solid #E2E8F0;
               font-size:10px;">
            Watermelon System · Módulo Balanceo {version} · SIGA Group SAS ·
            Coeficiente de influencia con convención de campo (0° en TDC).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


__all__ = [
    "bal_hero_card", "bal_section_header", "bal_kpi_row",
    "bal_status_banner", "bal_footer_norms",
    "NAVY", "CYAN", "CYAN_DARK", "AMBER", "GREEN", "RED", "GRAY", "GRAY_LIGHT",
]
