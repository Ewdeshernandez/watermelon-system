"""
core.calibration.ui — Kit visual del módulo Calibración
=======================================================

Hero + footer propios del módulo (API 670), reusando el kit compartido de
Balanceo (core/balance/ui): misma paleta azul acero, tipografía IBM Plex y
lenguaje "software internacional" (Bently/GE/Emerson/SKF). Solo belleza.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from core.balance.ui import (  # noqa: F401  (re-export para la página)
    bal_section_header as cal_section_header,
    bal_kpi_row as cal_kpi_row,
    bal_status_banner as cal_status_banner,
    _ensure_fonts, _SANS, _MONO,
    NAVY, STEEL, CYAN, CYAN_DARK, AMBER, GREEN, RED, GRAY, GRAY_LIGHT, LINE,
)


def cal_hero_card(asset_name: str = "(sin activo)", client: str = "",
                  site: str = "", mode: str = "—") -> None:
    """Banda del módulo Calibración: identidad + activo + tipo de ensayo."""
    _ensure_fonts()
    sub = " · ".join([p for p in [f"<b>{client}</b>" if client else "", site] if p]) \
        or "Curvas de linealidad de sensores de vibración"
    mode_color = {"PROXIMIDAD": CYAN_DARK, "ACELERÓMETRO": GREEN,
                  "VELOMITOR": AMBER}.get(mode.upper(), STEEL)
    st.markdown(
        f"""
        <div style="position:relative; overflow:hidden; border-radius:16px;
             margin-bottom:18px; padding:22px 28px; font-family:{_SANS};
             background:linear-gradient(120deg,#0e2547 0%,{NAVY} 46%,{STEEL} 100%);
             color:#eaf1fb; box-shadow:0 10px 30px rgba(12,32,78,.22);
             display:flex; justify-content:space-between; align-items:center; gap:24px;">
          <div style="position:absolute;inset:0;pointer-events:none;
               background:repeating-linear-gradient(135deg,rgba(255,255,255,.045) 0 2px,transparent 2px 9px);"></div>
          <div style="position:absolute;top:0;right:0;height:100%;width:6px;
               background:linear-gradient(180deg,{AMBER},#c56f05);"></div>
          <div style="flex:1; position:relative;">
            <div style="font:600 10.5px/1 {_MONO}; letter-spacing:.28em;
                 text-transform:uppercase; color:#8fb4e6; margin-bottom:6px;">
                 Módulo Calibración</div>
            <div style="font:800 24px/1.15 {_SANS}; letter-spacing:-.4px;
                 margin-bottom:4px;">{asset_name}</div>
            <div style="font:500 13px/1.4 {_SANS}; color:#b9cbe6;">{sub}</div>
          </div>
          <div style="text-align:right; min-width:170px; position:relative;">
            <div style="display:inline-block; background:{mode_color}; color:white;
                 font:700 12px {_SANS}; padding:5px 13px; border-radius:999px;
                 letter-spacing:0.08em;">{mode.upper()}</div>
            <div style="font:600 10.5px {_MONO}; color:#8fb4e6; margin-top:8px;">
                 API 670 · 5th ed.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cal_footer_norms(version: Optional[str] = None) -> None:
    _ensure_fonts()
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
             border:1px solid {LINE}; border-top:2px solid {AMBER}; border-radius:12px;
             font:500 11px/1.6 {_SANS}; color:{GRAY};">
          <div style="font:700 10px {_MONO}; color:{STEEL}; letter-spacing:0.14em;
               text-transform:uppercase; margin-bottom:6px;">
               Marco normativo aplicado</div>
          <div>{' &nbsp;·&nbsp; '.join(f'<b style="color:{NAVY};">{n}</b>' for n in norms)}</div>
          <div style="margin-top:12px; padding-top:8px; border-top:1px solid {LINE};
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
    "NAVY", "STEEL", "CYAN", "CYAN_DARK", "AMBER", "GREEN", "RED", "GRAY",
    "GRAY_LIGHT", "LINE",
]
