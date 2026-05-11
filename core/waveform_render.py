"""
core.waveform_render
====================

Renderer de waveform snapshots con estilo idéntico al módulo
`pages/02_Time_Waveforms.py` (Ciclo 23.86).

Se separó del módulo original porque `build_waveform_figure()` ahí
tiene dependencies del UI completo (cursors, cycle markers, session_state).
Acá tenemos una versión **standalone y pura** que matchea el mismo
look visual pero recibe solo un payload de snapshot, sin side effects.

Cuándo usar este vs el módulo original:
  • Este: vista preview en Live Monitoring / cards de últimos análisis
  • Módulo original: análisis completo con zoom, cursors, métricas
    avanzadas, ciclos sincrónicos, export PDF, etc.

API:
    render_snapshot_waveforms(snapshot_payload) → renderiza directo en
        st (no devuelve fig — usa subplots de Plotly + métricas en sidebar)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


# =============================================================
# ESTILO — extraído de pages/02_Time_Waveforms.py para consistencia
# =============================================================

WAVEFORM_LINE_COLOR = "#5b9cf0"        # azul royal soft
PLOT_BG_COLOR = "#f8fafc"              # gris muy claro
PAPER_BG_COLOR = "#f3f4f6"             # gris UI sutil
GRID_COLOR = "#e5e7eb"                 # grid medium gray
AXIS_FONT_COLOR = "#374151"            # text oscuro neutral
TITLE_FONT_COLOR = "#111827"           # header oscuro
META_FONT_COLOR = "#6b7280"            # caption gris

LINE_WIDTH = 1.6
SUBPLOT_HEIGHT_PX = 180                # height por subplot
SUBPLOT_VSPACING = 0.04                # gap vertical entre subplots
LEFT_MARGIN = 60                       # axis Y necesita espacio para units
RIGHT_MARGIN = 12

DEFAULT_AMP_UNIT_PLACEHOLDER = ""


def _safe_label(label: str) -> str:
    return label or "—"


def _format_metric(value: Any, decimals: int = 3) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "—"


def render_snapshot_waveforms(
    snapshot: Dict[str, Any],
    show_metrics_table: bool = True,
    max_sensors: int = 12,
) -> None:
    """Renderiza el contenido de un snapshot waveform con estilo del módulo
    original.

    Layout:
      • Header: corrida_label + RPM + timestamp
      • Multi-subplot Plotly vertical (uno por sensor)
      • Cada subplot: time-series scattergl + Peak/P2P/RMS en title
      • Métricas table compacta al final (opcional)

    Args:
        snapshot: payload dict (output de `load_waveform_snapshot`)
        show_metrics_table: si True, tabla con todas las métricas al final
        max_sensors: máximo a renderizar (corta si hay más)
    """
    sensors = snapshot.get("sensors") or []
    if not sensors:
        st.info("Sin sensores en este snapshot.")
        return

    if len(sensors) > max_sensors:
        st.caption(
            f"⚠ Mostrando primeros {max_sensors} de {len(sensors)} sensores. "
            f"Para ver todos, abrí el módulo Time Waveforms dedicado."
        )
        sensors = sensors[:max_sensors]

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        st.error(f"Plotly no disponible: {e}")
        return

    n = len(sensors)

    # Subplot titles incluyen métricas clave (peak/RMS) — match con look industrial
    subplot_titles = []
    for s in sensors:
        label = _safe_label(s.get("sensor_label", ""))
        unit = s.get("unit", "") or ""
        metrics = s.get("metrics") or {}
        peak = _format_metric(metrics.get("peak"))
        rms = _format_metric(metrics.get("rms"))
        title = (
            f"<b style='color:{TITLE_FONT_COLOR};font-size:13px;'>{label}</b> "
            f"<span style='color:{META_FONT_COLOR};font-size:11px;'>"
            f"&nbsp;&nbsp;Peak {peak} {unit} · RMS {rms} {unit}</span>"
        )
        subplot_titles.append(title)

    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=SUBPLOT_VSPACING,
        shared_xaxes=False,
    )

    for i, s in enumerate(sensors, start=1):
        time_arr = s.get("time") or []
        value_arr = s.get("values") or []
        if not time_arr or not value_arr:
            continue
        unit = s.get("unit", "") or DEFAULT_AMP_UNIT_PLACEHOLDER

        fig.add_trace(
            go.Scattergl(
                x=time_arr,
                y=value_arr,
                mode="lines",
                line=dict(width=LINE_WIDTH, color=WAVEFORM_LINE_COLOR),
                hovertemplate=(
                    f"<b>{_safe_label(s.get('sensor_label'))}</b><br>"
                    "t = %{x:.4f} s<br>"
                    f"y = %{{y:.4f}} {unit}<extra></extra>"
                ),
                name=s.get("sensor_label", ""),
                showlegend=False,
                connectgaps=False,
            ),
            row=i, col=1,
        )

        # Axis labels — solo bottom subplot lleva "Time (s)" para no saturar
        fig.update_yaxes(
            title=dict(
                text=f"Amplitud ({unit})" if unit else "Amplitud",
                font=dict(size=11, color=AXIS_FONT_COLOR),
            ),
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=True,
            zerolinecolor="#cbd5e1",
            zerolinewidth=1,
            tickfont=dict(size=10, color=AXIS_FONT_COLOR),
            row=i, col=1,
        )
        x_title = "Tiempo (s)" if i == n else ""
        fig.update_xaxes(
            title=dict(text=x_title, font=dict(size=11, color=AXIS_FONT_COLOR)),
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            tickfont=dict(size=10, color=AXIS_FONT_COLOR),
            row=i, col=1,
        )

    fig.update_layout(
        height=SUBPLOT_HEIGHT_PX * n + 80,
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=PAPER_BG_COLOR,
        margin=dict(l=LEFT_MARGIN, r=RIGHT_MARGIN, t=40, b=40),
        font=dict(family="-apple-system, 'SF Pro Text', system-ui, sans-serif"),
        hovermode="x unified",
    )
    # Subplot titles styling
    for ann in fig.layout.annotations:
        ann.update(xanchor="left", x=0.005, font=dict(size=12))

    st.plotly_chart(fig, use_container_width=True)

    if show_metrics_table:
        st.markdown(
            f"<div style='font-size:12px;color:{META_FONT_COLOR};margin-top:10px;"
            f"font-weight:600;text-transform:uppercase;letter-spacing:0.08em;'>"
            f"Métricas por sensor</div>",
            unsafe_allow_html=True,
        )
        rows = []
        for s in sensors:
            m = s.get("metrics") or {}
            rows.append({
                "Sensor": _safe_label(s.get("sensor_label", "")),
                "Peak":   _format_metric(m.get("peak")),
                "P2P":    _format_metric(m.get("peak_to_peak")),
                "RMS":    _format_metric(m.get("rms")),
                "Crest":  _format_metric(m.get("crest_factor"), decimals=2),
                "Kurt":   _format_metric(m.get("kurtosis"), decimals=2),
                "Unidad": s.get("unit", "") or "",
                "Fs (Hz)": _format_metric(s.get("sampling_rate_hz"), decimals=0),
            })
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Peak":  st.column_config.TextColumn(width="small"),
                "P2P":   st.column_config.TextColumn(width="small"),
                "RMS":   st.column_config.TextColumn(width="small"),
                "Crest": st.column_config.TextColumn(width="small"),
                "Kurt":  st.column_config.TextColumn(width="small"),
            },
        )


__all__ = ["render_snapshot_waveforms"]
