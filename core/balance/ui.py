"""
core/balance/ui.py — Kit visual del módulo Balanceo (compartido con Calibración)
================================================================================

Componentes Streamlit nivel "software internacional" (Bently/GE/Emerson/SKF):
jerarquía clara, KPIs grandes, cita normativa inline, estados con PUNTOS de color
(sin emojis). Alineado al sistema de diseño de la web (core/ui_industrial):
tipografía IBM Plex, banda navy con textura + strip ámbar, azul acero.

Solo BELLEZA — no toca matemática ni lógica. Las firmas son estables (las usan
pages/19_Balanceo.py, pages/21_Calibracion.py y core/calibration/ui.py).

Componentes
-----------
bal_hero_card       — Banda del módulo con activo + modo activo (1p/2p/manual)
bal_section_header  — Header de sección con cita normativa
bal_kpi_row         — Fila de KPI cards (valor grande + label + sublabel)
bal_status_banner   — Banner ok/warning/fail/info con punto de color
bal_footer_norms    — Footer normativo permanente
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import streamlit as st


# Paleta — alineada al sistema de diseño de la web (core/ui_industrial, azul acero).
NAVY = "#12305e"
STEEL = "#274b7d"
CYAN = "#1AAEE5"
CYAN_DARK = "#0F7FB0"
AMBER = "#e8890c"
AMBER_LIGHT = "#FCEED9"
GREEN = "#1f9d55"
GREEN_LIGHT = "#E4F5EC"
RED = "#dc3545"
RED_LIGHT = "#FCE9EB"
GRAY = "#64748b"
GRAY_LIGHT = "#F4F7FB"
LINE = "#e2e8f2"


def _ensure_fonts() -> None:
    """Inyecta IBM Plex una sola vez por sesión (firma visual de la familia)."""
    if st.session_state.get("_wm_ui_fonts"):
        return
    st.session_state["_wm_ui_fonts"] = True
    st.markdown(
        "<style>@import url('https://fonts.googleapis.com/css2?"
        "family=IBM+Plex+Sans:wght@400;500;600;700;800&"
        "family=IBM+Plex+Mono:wght@500;600&display=swap');</style>",
        unsafe_allow_html=True,
    )


_SANS = "'IBM Plex Sans','Segoe UI',system-ui,sans-serif"
_MONO = "'IBM Plex Mono',ui-monospace,monospace"


def bal_hero_card(asset_name: str = "(sin activo)", client: str = "",
                  site: str = "", mode: str = "—") -> None:
    """Banda del módulo: identidad + activo + modo activo. Navy con textura +
    strip ámbar, estilo estación de análisis (System1/AMS)."""
    _ensure_fonts()
    sub = " · ".join([p for p in [f"<b>{client}</b>" if client else "", site] if p]) \
        or "Balanceo de rotores por coeficiente de influencia"
    mode_color = {"1 PLANO": CYAN_DARK, "2 PLANOS": GREEN}.get(mode.upper(), STEEL)
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
                 Módulo Balanceo</div>
            <div style="font:800 24px/1.15 {_SANS}; letter-spacing:-.4px;
                 margin-bottom:4px;">{asset_name}</div>
            <div style="font:500 13px/1.4 {_SANS}; color:#b9cbe6;">{sub}</div>
          </div>
          <div style="text-align:right; min-width:170px; position:relative;">
            <div style="display:inline-block; background:{mode_color}; color:white;
                 font:700 12px {_SANS}; padding:5px 13px; border-radius:999px;
                 letter-spacing:0.08em;">{mode.upper()}</div>
            <div style="font:600 10.5px {_MONO}; color:#8fb4e6; margin-top:8px;">
                 ISO 21940 · API 684</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bal_section_header(title: str, subtitle: str = "", norm_ref: str = "",
                       icon: str = "") -> None:
    _ensure_fonts()
    norm_html = (
        f'<span style="font:600 11px {_MONO}; color:{STEEL}; margin-left:10px; '
        f'padding:2px 9px; background:#eef3fb; border:1px solid {LINE}; '
        f'border-radius:6px; letter-spacing:.02em;">{norm_ref}</span>'
        if norm_ref else "")
    st.markdown(
        f"""
        <div style="margin:16px 0 10px 0; font-family:{_SANS};">
          <div style="display:flex; align-items:center; color:{NAVY};
               font:700 17px/1.3 {_SANS}; letter-spacing:-.2px;">
            <span style="color:{AMBER}; margin-right:9px; font-size:15px;">●</span>
            {title}{norm_html}
          </div>
          {f'<div style="color:{GRAY}; font:500 12.5px {_SANS}; margin-top:4px;">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def bal_kpi_row(metrics: List[Tuple[str, str, str, str]]) -> None:
    """metrics: [(value, label, sublabel, color)] con color en
    cyan|green|amber|red|navy|gray."""
    _ensure_fonts()
    color_map = {
        "cyan": CYAN_DARK, "green": GREEN, "amber": AMBER,
        "red": RED, "navy": STEEL, "gray": GRAY,
    }
    cols = st.columns(len(metrics))
    for col, (value, label, sublabel, color) in zip(cols, metrics):
        fg = color_map.get(color, STEEL)
        with col:
            st.markdown(
                f"""
                <div style="background:#fff; border:1px solid {LINE};
                     border-left:4px solid {fg}; padding:14px 16px; border-radius:12px;
                     min-height:98px; font-family:{_SANS};
                     box-shadow:0 1px 2px rgba(11,31,58,.05),0 6px 18px rgba(11,31,58,.05);">
                  <div style="font:800 26px/1.05 {_SANS}; color:{fg};
                       margin-bottom:6px;">{value}</div>
                  <div style="font:700 11px {_SANS}; color:{NAVY};
                       text-transform:uppercase; letter-spacing:0.06em;
                       margin-bottom:2px;">{label}</div>
                  <div style="font:500 11px/1.3 {_SANS}; color:{GRAY};">{sublabel}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def bal_status_banner(title: str, detail: str = "", severity: str = "info") -> None:
    _ensure_fonts()
    cfg = {
        "ok": (GREEN, GREEN_LIGHT, "#14532D"),
        "warning": (AMBER, AMBER_LIGHT, "#7a4a06"),
        "fail": (RED, RED_LIGHT, "#7F1D1D"),
        "info": (STEEL, "#eef3fb", "#1E3A8A"),
    }
    border, bg, fg = cfg.get(severity, cfg["info"])
    st.markdown(
        f"""
        <div style="background:{bg}; border:1px solid {border}55; border-left:4px solid {border};
             border-radius:12px; padding:13px 18px; margin-bottom:14px; display:flex; gap:13px;
             align-items:flex-start; font-family:{_SANS};">
          <span style="color:{border}; font-size:20px; line-height:1.1;">●</span>
          <div style="flex:1;">
            <div style="font:700 14px {_SANS}; color:{fg}; margin-bottom:2px;">{title}</div>
            {f'<div style="color:{fg}; font:500 12.5px/1.5 {_SANS}; opacity:.9;">{detail}</div>' if detail else ''}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bal_footer_norms(version: Optional[str] = None) -> None:
    _ensure_fonts()
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
             border:1px solid {LINE}; border-top:2px solid {AMBER}; border-radius:12px;
             font:500 11px/1.6 {_SANS}; color:{GRAY};">
          <div style="font:700 10px {_MONO}; color:{STEEL}; letter-spacing:0.14em;
               text-transform:uppercase; margin-bottom:6px;">
               Marco normativo aplicado</div>
          <div>{' &nbsp;·&nbsp; '.join(f'<b style="color:{NAVY};">{n}</b>' for n in norms)}</div>
          <div style="margin-top:12px; padding-top:8px; border-top:1px solid {LINE};
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
    "NAVY", "STEEL", "CYAN", "CYAN_DARK", "AMBER", "GREEN", "RED", "GRAY",
    "GRAY_LIGHT", "LINE",
]
