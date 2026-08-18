"""
core.calibration.ui — Kit visual del módulo Calibración
=======================================================

Hero + footer propios del módulo (API 670), reusando los componentes
compartidos del kit de Balanceo (misma paleta y lenguaje enterprise que Modal):
bal_section_header, bal_kpi_row, bal_status_banner.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from core.balance.ui import (  # noqa: F401  (re-export para la página)
    bal_section_header as cal_section_header,
    bal_kpi_row as cal_kpi_row,
    bal_status_banner as cal_status_banner,
    NAVY, CYAN, CYAN_DARK, AMBER, GREEN, RED, GRAY, GRAY_LIGHT,
)


def cal_hero_card(asset_name: str = "(sin activo)", client: str = "",
                  site: str = "", mode: str = "—") -> None:
    """Banner del módulo Calibración: identidad + activo + tipo de ensayo."""
    sub = " · ".join([p for p in [f"<b>{client}</b>" if client else "", site] if p]) \
        or "Curvas de linealidad de sensores de vibración"
    mode_color = {"PROXIMIDAD": CYAN_DARK, "ACELERÓMETRO": GREEN,
                  "VELOMITOR": AMBER}.get(mode.upper(), GRAY)
    st.markdown(
        f"""
        <div style="background:{NAVY}; color:white; padding:22px 28px;
             border-radius:14px; margin-bottom:18px; display:flex;
             justify-content:space-between; align-items:center; gap:24px;">
          <div style="flex:1;">
            <div style="font-size:11px; font-weight:700; letter-spacing:0.18em;
                 text-transform:uppercase; color:{CYAN}; margin-bottom:4px;">
                 Módulo Calibración</div>
            <div style="font-size:24px; font-weight:800; line-height:1.2;
                 margin-bottom:4px;">{asset_name}</div>
            <div style="font-size:13px; color:rgba(226,232,240,0.85);">{sub}</div>
          </div>
          <div style="text-align:right; min-width:170px;">
            <div style="display:inline-block; background:{mode_color}; color:white;
                 font-weight:700; font-size:13px; padding:4px 12px; border-radius:999px;
                 letter-spacing:0.1em;">{mode.upper()}</div>
            <div style="font-size:11px; color:rgba(226,232,240,0.7);
                 font-family:monospace; margin-top:6px;">API 670 · 5th ed.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cal_footer_norms(version: Optional[str] = None) -> None:
    if version is None:
        try:
            from core.version import get_version_short
            version = get_version_short()
        except Exception:
            version = "v?"
    norms = ["API 670 5.ª ed. — Tabla 1 (precisión) / Fig. 4 (ISF · DSL)",
             "Manual del fabricante (Bently Nevada · Emerson · SKF · Metrix)",
             "Trazabilidad del patrón / shaker de referencia"]
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
            Watermelon System · Módulo Calibración {version} · SIGA Group SAS ·
            Proximidad 200 mV/mil (ISF ±5 % · DSL ±1 mil) · Acelerómetro 100 mV/g.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


__all__ = [
    "cal_hero_card", "cal_footer_norms", "cal_section_header", "cal_kpi_row",
    "cal_status_banner",
    "NAVY", "CYAN", "CYAN_DARK", "AMBER", "GREEN", "RED", "GRAY", "GRAY_LIGHT",
]
