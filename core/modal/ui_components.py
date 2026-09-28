"""
core/modal/ui_components.py — Componentes visuales del Modal Analysis Module
==============================================================================

Kit Streamlit nivel "software internacional" (Bently Nevada / GE / Emerson / SKF),
alineado al sistema de diseño de la web (core/ui_industrial): tipografía IBM Plex,
banda navy con textura + strip ámbar, azul acero, estados con PUNTOS de color
(sin emojis). Cita normativa inline en cada componente.

Solo BELLEZA — no toca matemática ni lógica. Firmas estables.

Componentes públicos
--------------------
modal_hero_card · modal_footer_norms · modal_kpi_row · modal_section_header
modal_plot_caption · modal_status_banner · modal_empty_state
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import streamlit as st


# =====================================================================
# Paleta — alineada a core/ui_industrial (azul acero)
# =====================================================================
NAVY = "#12305e"
NAVY_LIGHT = "#1B2A4E"
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

_SANS = "'IBM Plex Sans','Segoe UI',system-ui,sans-serif"
_MONO = "'IBM Plex Mono',ui-monospace,monospace"


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


# =====================================================================
# 1. HERO CARD
# =====================================================================
def modal_hero_card(
    asset_name: str = "(sin activo seleccionado)",
    client_name: str = "",
    station_name: str = "",
    method_active: str = "—",
    record_info: str = "",
) -> None:
    """Banda principal: activo + método activo (EMA/OMA). Navy con textura +
    strip ámbar, estilo estación de análisis (System1/AMS)."""
    _ensure_fonts()
    _parts = []
    if client_name:
        _parts.append(f"<b>{client_name}</b>")
    if station_name:
        _parts.append(station_name)
    subtitle = " · ".join(_parts) if _parts else "Modal Analysis"
    _method_color = {"EMA": CYAN_DARK, "OMA": GREEN, "FEA": AMBER}.get(
        method_active.upper(), STEEL)
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
                 Modal Analysis Module</div>
            <div style="font:800 24px/1.15 {_SANS}; letter-spacing:-.4px;
                 margin-bottom:4px;">{asset_name}</div>
            <div style="font:500 13px/1.4 {_SANS}; color:#b9cbe6;">{subtitle}</div>
          </div>
          <div style="text-align:right; min-width:200px; position:relative;">
            <div style="display:inline-block; background:{_method_color}; color:white;
                 font:700 12px {_SANS}; padding:5px 13px; border-radius:999px;
                 letter-spacing:0.08em; margin-bottom:6px;">{method_active.upper()}</div>
            <div style="font:600 10.5px {_MONO}; color:#8fb4e6;">{record_info}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# 2. FOOTER NORMATIVO
# =====================================================================
def modal_footer_norms(
    active_norms: Optional[Sequence[str]] = None,
    algorithms: Optional[Sequence[str]] = None,
    version: Optional[str] = None,
) -> None:
    _ensure_fonts()
    if version is None:
        try:
            from core.version import get_version_short
            version = get_version_short()
        except Exception:
            version = "v?"
    norms = list(active_norms or [
        "ISO 7626-1..6", "ISO 20816", "API 684", "API 618 secc. 7.9.4.2.5.3.2",
    ])
    algos = list(algorithms or [
        "Circle-Fit Nyquist (Kennedy-Pancu 1947)",
        "FDD (Brincker 2001)",
        "Modal Complexity (Pappa & Eishan 1995)",
        "AutoMAC (ISO 7626-6 secc. 6.5)",
    ])
    st.markdown(
        f"""
        <div style="margin-top:32px; padding:14px 18px; background:{GRAY_LIGHT};
             border:1px solid {LINE}; border-top:2px solid {AMBER}; border-radius:12px;
             font:500 11px/1.6 {_SANS}; color:{GRAY};">
          <div style="font:700 10px {_MONO}; color:{STEEL}; letter-spacing:.14em;
               text-transform:uppercase; margin-bottom:6px;">Marco normativo aplicado</div>
          <div style="margin-bottom:6px;">
            {' &nbsp;·&nbsp; '.join(f'<b style="color:{NAVY};">{n}</b>' for n in norms)}</div>
          <div style="font:700 10px {_MONO}; color:{STEEL}; letter-spacing:.14em;
               text-transform:uppercase; margin:10px 0 4px;">Algoritmos implementados</div>
          <div>{' &nbsp;·&nbsp; '.join(algos)}</div>
          <div style="margin-top:12px; padding-top:8px; border-top:1px solid {LINE};
               font-size:10px;">
            Watermelon Modal Module {version} · SIGA Group SAS · Identificación
            modal nativa bajo normas internacionales.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# 3. KPI ROW
# =====================================================================
def modal_kpi_row(metrics: List[Tuple[str, str, str, str]]) -> None:
    """metrics: [(value, label, sublabel, color)] con color cyan|green|amber|red|navy|gray."""
    _ensure_fonts()
    color_map = {"cyan": CYAN_DARK, "green": GREEN, "amber": AMBER,
                 "red": RED, "navy": STEEL, "gray": GRAY}
    cols = st.columns(len(metrics))
    for col, (value, label, sublabel, color_name) in zip(cols, metrics):
        fg = color_map.get(color_name, STEEL)
        with col:
            st.markdown(
                f"""
                <div style="background:#fff; border:1px solid {LINE};
                     border-left:4px solid {fg}; padding:14px 16px; border-radius:12px;
                     min-height:94px; font-family:{_SANS};
                     box-shadow:0 1px 2px rgba(11,31,58,.05),0 6px 18px rgba(11,31,58,.05);">
                  <div style="font:800 27px/1.05 {_SANS}; color:{fg};
                       margin-bottom:6px;">{value}</div>
                  <div style="font:700 11px {_SANS}; color:{NAVY};
                       text-transform:uppercase; letter-spacing:0.06em;
                       margin-bottom:2px;">{label}</div>
                  <div style="font:500 11px/1.3 {_SANS}; color:{GRAY};">{sublabel}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =====================================================================
# 4. SECTION HEADER
# =====================================================================
def modal_section_header(title: str, subtitle: str = "", norm_ref: str = "",
                         icon: str = "") -> None:
    _ensure_fonts()
    norm_html = (
        f'<span style="font:600 11px {_MONO}; color:{STEEL}; margin-left:10px; '
        f'padding:2px 9px; background:#eef3fb; border:1px solid {LINE}; '
        f'border-radius:6px;">{norm_ref}</span>' if norm_ref else "")
    st.markdown(
        f"""
        <div style="margin:18px 0 10px 0; font-family:{_SANS};">
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


# =====================================================================
# 5. PLOT CAPTION
# =====================================================================
def modal_plot_caption(text: str, norm_ref: str = "", algorithm: str = "") -> None:
    _ensure_fonts()
    parts = []
    if norm_ref:
        parts.append(f'<span style="font-family:{_MONO}; color:{STEEL}; '
                     f'font-weight:600;">{norm_ref}</span>')
    if algorithm:
        parts.append(f'<span style="font-style:italic; color:{GRAY};">{algorithm}</span>')
    norms_line = " &nbsp;·&nbsp; ".join(parts)
    st.markdown(
        f"""
        <div style="margin-top:4px; padding:8px 12px; background:#f7f9fc;
             border-left:3px solid {AMBER}; font:500 11.5px/1.5 {_SANS};
             color:{NAVY}; border-radius:0 8px 8px 0;">
          {text}
          {f'<div style="margin-top:4px;">{norms_line}</div>' if norms_line else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# 6. STATUS BANNER (punto de color, sin emojis)
# =====================================================================
def modal_status_banner(title: str, detail: str, severity: str = "info",
                        icon_override: str = "") -> None:
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
             border-radius:12px; padding:14px 18px; margin-bottom:16px; display:flex; gap:13px;
             align-items:flex-start; font-family:{_SANS};">
          <span style="color:{border}; font-size:20px; line-height:1.1;">●</span>
          <div style="flex:1;">
            <div style="font:700 14px {_SANS}; color:{fg}; margin-bottom:4px;">{title}</div>
            <div style="font:500 12.5px/1.5 {_SANS}; color:{fg}; opacity:.9;">{detail}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# 7. EMPTY STATE
# =====================================================================
def modal_empty_state(icon: str, title: str, description: str,
                      cta_label: str = "", norm_ref: str = "") -> None:
    _ensure_fonts()
    st.markdown(
        f"""
        <div style="text-align:center; padding:44px 24px; background:{GRAY_LIGHT};
             border:1px dashed #c4d1e6; border-radius:14px; margin:20px 0; font-family:{_SANS};">
          <div style="font-size:30px; line-height:1; color:{STEEL}; margin-bottom:10px;">●</div>
          <div style="font:700 16px {_SANS}; color:{NAVY}; margin-bottom:8px;">{title}</div>
          <div style="font:500 13px/1.6 {_SANS}; color:{GRAY}; max-width:520px; margin:0 auto;">{description}</div>
          {f'<div style="font:600 11px {_MONO}; color:{STEEL}; margin-top:14px;">{norm_ref}</div>' if norm_ref else ''}
          {f'<div style="margin-top:20px;"><b style="color:{CYAN_DARK};">→ {cta_label}</b></div>' if cta_label else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )
