"""
core/modal/ui_components.py — Componentes visuales del Modal Analysis Module
==============================================================================

Helpers Streamlit estilo enterprise / Bently Nevada / DNV / Stripe.
Paleta consistente con el resto de Watermelon (navy + cyan + ámbar + verde).
Sin gradients pesados, sin shadows decorativos, sin emojis abusivos.

Citas normativas inline en cada componente — la calidad enterprise viene de
mostrar el marco que respalda cada decisión técnica.

Componentes públicos
--------------------
modal_hero_card           — Banner principal del módulo con activo + método activo
modal_footer_norms        — Footer normativo permanente al final de la página
modal_kpi_row             — Row de 3-5 KPIs estilo Live Monitoring
modal_section_header      — Header de sección consistente con cita normativa
modal_plot_caption        — Caption inferior con cita normativa explícita
modal_status_banner       — Banner verde/ámbar/rojo según estado
modal_empty_state         — Empty state profesional para tabs sin data
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import streamlit as st


# =====================================================================
# Paleta Watermelon — consistente con resto del sistema
# =====================================================================
NAVY = "#0F1E3D"
NAVY_LIGHT = "#1B2A4E"
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


# =====================================================================
# 1. HERO CARD — banner principal del módulo
# =====================================================================

def modal_hero_card(
    asset_name: str = "(sin activo seleccionado)",
    client_name: str = "",
    station_name: str = "",
    method_active: str = "—",
    record_info: str = "",
) -> None:
    """
    Hero card principal del Modal Analysis Module.

    Args:
        asset_name: "TES 1 GE LM6000"
        client_name: "MAGNEX" (opcional)
        station_name: "TERMOSURIA" (opcional)
        method_active: "EMA" | "OMA" | "—" si nada cargado
        record_info: "30s @ 5120 Hz · 4 ch" (opcional)
    """
    _subtitle_parts = []
    if client_name:
        _subtitle_parts.append(client_name)
    if station_name:
        _subtitle_parts.append(station_name)
    subtitle = " · ".join(_subtitle_parts) if _subtitle_parts else "Modal Analysis"

    _method_color = {
        "EMA": CYAN_DARK,
        "OMA": GREEN,
        "FEA": AMBER,
    }.get(method_active.upper(), GRAY)

    st.markdown(
        f"""
        <div style="
            background: {NAVY};
            color: white;
            padding: 22px 28px;
            border-radius: 14px;
            margin-bottom: 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 24px;
        ">
            <div style="flex: 1;">
                <div style="
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 0.18em;
                    text-transform: uppercase;
                    color: {CYAN};
                    margin-bottom: 4px;
                ">Modal Analysis Module</div>
                <div style="
                    font-size: 24px;
                    font-weight: 800;
                    line-height: 1.2;
                    margin-bottom: 4px;
                ">{asset_name}</div>
                <div style="
                    font-size: 13px;
                    color: rgba(226,232,240,0.85);
                ">{subtitle}</div>
            </div>
            <div style="text-align: right; min-width: 200px;">
                <div style="
                    display: inline-block;
                    background: {_method_color};
                    color: white;
                    font-weight: 700;
                    font-size: 13px;
                    padding: 4px 12px;
                    border-radius: 999px;
                    letter-spacing: 0.1em;
                    margin-bottom: 6px;
                ">{method_active.upper()}</div>
                <div style="
                    font-size: 11px;
                    color: rgba(226,232,240,0.7);
                    font-family: monospace;
                ">{record_info}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# 2. FOOTER NORMATIVO permanente
# =====================================================================

def modal_footer_norms(
    active_norms: Optional[Sequence[str]] = None,
    algorithms: Optional[Sequence[str]] = None,
    version: str = "v3.31.161",
) -> None:
    """
    Footer permanente al final de la página con marco normativo.

    Args:
        active_norms: ["ISO 7626-5", "ISO 7626-6", "ISO 20816", "API 684"]
        algorithms: ["Circle-Fit Nyquist", "FDD", "MPC"]
        version: versión del módulo
    """
    norms = list(active_norms or [
        "ISO 7626-1..6", "ISO 20816", "API 684", "API 618 §7.9.4.2.5.3.2",
    ])
    algos = list(algorithms or [
        "Circle-Fit Nyquist (Kennedy-Pancu 1947)",
        "FDD (Brincker 2001)",
        "Modal Complexity (Pappa & Eishan 1995)",
        "AutoMAC (ISO 7626-6 §6.5)",
    ])

    st.markdown(
        f"""
        <div style="
            margin-top: 32px;
            padding: 14px 18px;
            background: {GRAY_LIGHT};
            border-top: 2px solid {CYAN};
            border-radius: 6px;
            font-size: 11px;
            color: {GRAY};
            line-height: 1.6;
        ">
            <div style="
                font-weight: 700;
                color: {NAVY};
                letter-spacing: 0.1em;
                text-transform: uppercase;
                font-size: 10px;
                margin-bottom: 6px;
            ">Marco normativo aplicado</div>
            <div style="margin-bottom: 6px;">
                {' &nbsp;·&nbsp; '.join(f'<b style="color:{NAVY};">{n}</b>' for n in norms)}
            </div>
            <div style="
                font-weight: 700;
                color: {NAVY};
                letter-spacing: 0.1em;
                text-transform: uppercase;
                font-size: 10px;
                margin-top: 10px;
                margin-bottom: 4px;
            ">Algoritmos implementados</div>
            <div>{' &nbsp;·&nbsp; '.join(algos)}</div>
            <div style="
                margin-top: 12px;
                padding-top: 8px;
                border-top: 1px solid #E2E8F0;
                color: {GRAY};
                font-size: 10px;
            ">
                Watermelon Modal Module {version} · SIGA Group SAS · Identificación
                modal nativa bajo normas internacionales.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# 3. KPI ROW — métricas resumen estilo Live Monitoring
# =====================================================================

def modal_kpi_row(
    metrics: List[Tuple[str, str, str, str]],
) -> None:
    """
    Renderiza una fila de KPI cards.

    Args:
        metrics: lista de tuples (value, label, sublabel, color_name)
                 color_name: "cyan" | "green" | "amber" | "red" | "navy"
    """
    color_map = {
        "cyan": (CYAN_DARK, "#E0F2FE"),
        "green": (GREEN, GREEN_LIGHT),
        "amber": (AMBER, AMBER_LIGHT),
        "red": (RED, RED_LIGHT),
        "navy": (NAVY, GRAY_LIGHT),
        "gray": (GRAY, GRAY_LIGHT),
    }
    cols = st.columns(len(metrics))
    for col, (value, label, sublabel, color_name) in zip(cols, metrics):
        fg, bg = color_map.get(color_name, color_map["navy"])
        with col:
            st.markdown(
                f"""
                <div style="
                    background: {bg};
                    border-left: 4px solid {fg};
                    padding: 14px 16px;
                    border-radius: 6px;
                    min-height: 92px;
                ">
                    <div style="
                        font-size: 28px;
                        font-weight: 800;
                        color: {fg};
                        line-height: 1;
                        margin-bottom: 6px;
                    ">{value}</div>
                    <div style="
                        font-size: 12px;
                        font-weight: 700;
                        color: {NAVY};
                        text-transform: uppercase;
                        letter-spacing: 0.06em;
                        margin-bottom: 2px;
                    ">{label}</div>
                    <div style="
                        font-size: 11px;
                        color: {GRAY};
                        line-height: 1.3;
                    ">{sublabel}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =====================================================================
# 4. SECTION HEADER consistente
# =====================================================================

def modal_section_header(
    title: str,
    subtitle: str = "",
    norm_ref: str = "",
    icon: str = "",
) -> None:
    """
    Header de sección estandarizado.

    Args:
        title: "Análisis Modal Operacional"
        subtitle: descripción corta
        norm_ref: "ISO 20816 · Brincker 2001"
        icon: emoji opcional al inicio
    """
    icon_html = (
        f'<span style="font-size:18px; margin-right:8px;">{icon}</span>'
        if icon else ""
    )
    norm_html = (
        f'<span style="font-size:11px; color:{GRAY}; font-weight:500; '
        f'font-family:monospace; margin-left:10px; '
        f'padding:2px 8px; background:{GRAY_LIGHT}; border-radius:4px;">'
        f'{norm_ref}</span>'
        if norm_ref else ""
    )
    st.markdown(
        f"""
        <div style="margin: 18px 0 10px 0;">
            <div style="
                display: flex;
                align-items: center;
                color: {NAVY};
                font-size: 17px;
                font-weight: 700;
                line-height: 1.3;
            ">
                {icon_html}{title}{norm_html}
            </div>
            {f'<div style="color:{GRAY}; font-size:12.5px; margin-top:4px;">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# 5. PLOT CAPTION — cita normativa al pie de cada plot
# =====================================================================

def modal_plot_caption(
    text: str,
    norm_ref: str = "",
    algorithm: str = "",
) -> None:
    """
    Caption pequeño al pie del plot con cita normativa y algoritmo.

    Args:
        text: descripción interpretativa
        norm_ref: "ISO 7626-6 §6.3"
        algorithm: "Brincker, Zhang, Andersen 2001"
    """
    parts = []
    if norm_ref:
        parts.append(
            f'<span style="font-family:monospace; color:{CYAN_DARK}; '
            f'font-weight:600;">{norm_ref}</span>'
        )
    if algorithm:
        parts.append(
            f'<span style="font-style:italic; color:{GRAY};">{algorithm}</span>'
        )
    norms_line = " &nbsp;·&nbsp; ".join(parts)
    st.markdown(
        f"""
        <div style="
            margin-top: 4px;
            padding: 8px 12px;
            background: #FAFBFC;
            border-left: 3px solid {CYAN};
            font-size: 11.5px;
            color: {NAVY};
            line-height: 1.5;
            border-radius: 0 4px 4px 0;
        ">
            {text}
            {f'<div style="margin-top:4px;">{norms_line}</div>' if norms_line else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# 6. STATUS BANNER (verde / ámbar / rojo según severity)
# =====================================================================

def modal_status_banner(
    title: str,
    detail: str,
    severity: str = "info",  # "ok" | "warning" | "fail" | "info"
    icon_override: str = "",
) -> None:
    """
    Banner colored según severity con título grande + detail.
    """
    cfg = {
        "ok": (GREEN, GREEN_LIGHT, "#14532D", "✓"),
        "warning": (AMBER, AMBER_LIGHT, "#78350F", "⚠"),
        "fail": (RED, RED_LIGHT, "#7F1D1D", "✗"),
        "info": (CYAN_DARK, "#DBEAFE", "#1E3A8A", "ℹ"),
    }
    border, bg, fg, default_icon = cfg.get(severity, cfg["info"])
    icon = icon_override or default_icon
    st.markdown(
        f"""
        <div style="
            background: {bg};
            border: 1.5px solid {border};
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 16px;
            display: flex;
            gap: 14px;
            align-items: flex-start;
        ">
            <div style="
                font-size: 22px;
                color: {border};
                line-height: 1;
            ">{icon}</div>
            <div style="flex: 1;">
                <div style="
                    font-weight: 700;
                    color: {fg};
                    font-size: 14px;
                    margin-bottom: 4px;
                ">{title}</div>
                <div style="
                    color: {fg};
                    font-size: 12.5px;
                    line-height: 1.5;
                    opacity: 0.85;
                ">{detail}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# 7. EMPTY STATE profesional
# =====================================================================

def modal_empty_state(
    icon: str,
    title: str,
    description: str,
    cta_label: str = "",
    norm_ref: str = "",
) -> None:
    """
    Empty state para tabs sin data — con descripción de qué viene y CTA opcional.
    """
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 48px 24px;
            background: {GRAY_LIGHT};
            border: 1px dashed #CBD5E1;
            border-radius: 12px;
            margin: 20px 0;
        ">
            <div style="font-size:42px; line-height:1; margin-bottom:12px;">{icon}</div>
            <div style="
                font-size: 16px;
                font-weight: 700;
                color: {NAVY};
                margin-bottom: 8px;
            ">{title}</div>
            <div style="
                font-size: 13px;
                color: {GRAY};
                max-width: 520px;
                margin: 0 auto;
                line-height: 1.6;
            ">{description}</div>
            {f'<div style="font-family:monospace; font-size:11px; color:{CYAN_DARK}; margin-top:14px;">{norm_ref}</div>' if norm_ref else ''}
            {f'<div style="margin-top:20px;"><b style="color:{CYAN_DARK};">→ {cta_label}</b></div>' if cta_label else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )
