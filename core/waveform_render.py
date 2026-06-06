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


def _wf_family_global(unit: str, label: str = "") -> str:
    """Familia del canal: vel / acel / prox — por unidad o tokens del label."""
    u = (unit or "").lower()
    if "mm/s" in u or "in/s" in u or "ips" in u:
        return "vel"
    if u.strip().startswith("g") or "m/s2" in u or "m/s²" in u:
        return "acel"
    if "mil" in u or "µm" in u or "um" in u:
        return "prox"
    t = (label or "").upper()
    if "ACEL" in t or "ACC" in t:
        return "acel"
    if "VEL" in t or "VL" in t or "VT" in t:
        return "vel"
    if "VE" in t or "PROX" in t or "DESP" in t:
        return "prox"
    return u or "otro"


def render_snapshot_waveforms(
    snapshot: Dict[str, Any],
    show_metrics_table: bool = True,
    max_sensors: int = 12,
    fam_units: Optional[Dict[str, str]] = None,
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

    # Ciclo 23.160 — Resolver unidad por sensor: la del snapshot, o la de
    # su familia desde datos en vivo (fam_units), o vacío.
    fam_units = fam_units or {}

    def _resolved_unit(s: Dict[str, Any]) -> str:
        u = (s.get("unit", "") or "").strip()
        if u:
            return u
        fam = _wf_family_global(s.get("unit", ""), s.get("sensor_label", ""))
        return fam_units.get(fam, "")

    # Subplot titles incluyen métricas clave (peak/RMS) — match con look industrial
    subplot_titles = []
    for s in sensors:
        label = _safe_label(s.get("sensor_label", ""))
        unit = _resolved_unit(s)
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

    # Ciclo 23.156 — Escala Y COMÚN por familia (vel / acel / prox), estilo
    # System1/AMS: el máximo absoluto de la familia define el rango simétrico
    # de todos sus canales — amplitudes comparables a simple vista.
    _wf_family = _wf_family_global

    _fam_absmax: Dict[str, float] = {}
    for s in sensors:
        vals = s.get("values") or []
        if not vals:
            continue
        fam = _wf_family(s.get("unit", ""), s.get("sensor_label", ""))
        try:
            m = max(abs(float(v)) for v in vals)
        except Exception:
            m = 0.0
        _fam_absmax[fam] = max(_fam_absmax.get(fam, 0.0), m)

    for i, s in enumerate(sensors, start=1):
        time_arr = s.get("time") or []
        value_arr = s.get("values") or []
        if not time_arr or not value_arr:
            continue
        unit = _resolved_unit(s) or DEFAULT_AMP_UNIT_PLACEHOLDER
        # Ciclo 23.159 — Auto-calibración: si la Fs implícita es < 50 Hz es
        # físicamente implausible en vibraciones → el tiempo del CSV venía
        # en ms leído como s. Convertir a segundos reales.
        try:
            _dur = float(time_arr[-1]) - float(time_arr[0])
            if _dur > 0 and (len(time_arr) / _dur) < 50.0:
                time_arr = [float(v) / 1000.0 for v in time_arr]
        except Exception:
            pass

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
        _fm = _fam_absmax.get(_wf_family(unit, s.get("sensor_label", "")), 0.0)
        fig.update_yaxes(
            range=[-_fm * 1.1, _fm * 1.1] if _fm > 0 else None,
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
        # Ciclo 23.158 — Tabla minimalista clase mundial (HTML, sin Kurt/Fs).
        # Pico = amplitud máxima · Pico-Pico = excursión total (máx−mín) ·
        # RMS = energía promedio · Factor cresta = Pico/RMS (impulsividad).
        _fam_color = {"vel": "#2563eb", "acel": "#d97706", "prox": "#7c3aed"}
        _rows_html = ""
        for idx, s in enumerate(sensors):
            m = s.get("metrics") or {}
            lbl = _safe_label(s.get("sensor_label", ""))
            unit = _resolved_unit(s) or "—"
            fam = _wf_family(s.get("unit", ""), s.get("sensor_label", ""))
            dot = _fam_color.get(fam, "#94a3b8")
            bg = "#ffffff" if idx % 2 == 0 else "#f8fafc"
            crest = m.get("crest_factor")
            try:
                crest_hi = crest is not None and float(crest) > 3.5
            except Exception:
                crest_hi = False
            crest_style = "color:#b45309;font-weight:700;" if crest_hi else ""
            _num = ("font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
                    "text-align:right;padding:7px 14px;color:#334155;")
            _rows_html += (
                f"<tr style='background:{bg};'>"
                f"<td style='padding:7px 14px;font-weight:600;color:#0f172a;"
                f"white-space:nowrap;'>"
                f"<span style='display:inline-block;width:8px;height:8px;"
                f"border-radius:50%;background:{dot};margin-right:8px;'></span>"
                f"{lbl}</td>"
                f"<td style='{_num}'>{_format_metric(m.get('peak'))}</td>"
                f"<td style='{_num}'>{_format_metric(m.get('peak_to_peak'))}</td>"
                f"<td style='{_num}'>{_format_metric(m.get('rms'))}</td>"
                f"<td style='{_num}{crest_style}'>"
                f"{_format_metric(crest, decimals=2)}</td>"
                f"<td style='padding:7px 14px;color:#64748b;'>{unit}</td>"
                f"</tr>"
            )
        _th = ("padding:8px 14px;font-size:10px;font-weight:800;"
               "letter-spacing:0.12em;text-transform:uppercase;"
               "color:#64748b;text-align:right;border-bottom:2px solid #e2e8f0;")
        _th_l = _th.replace("text-align:right", "text-align:left")
        st.markdown(
            f"""
<div style='font-size:12px;color:{META_FONT_COLOR};margin:14px 0 6px 0;
font-weight:700;text-transform:uppercase;letter-spacing:0.1em;'>Métricas por sensor</div>
<table style='width:100%;border-collapse:collapse;font-size:13px;
border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;'>
<thead><tr style='background:#f1f5f9;'>
<th style='{_th_l}'>Sensor</th><th style='{_th}'>Pico</th>
<th style='{_th}'>Pico-Pico</th><th style='{_th}'>RMS</th>
<th style='{_th}'>Factor cresta</th><th style='{_th_l}'>Unidad</th>
</tr></thead><tbody>{_rows_html}</tbody></table>
<div style='font-size:11px;color:#94a3b8;margin-top:6px;'>
Pico = amplitud máxima · Pico-Pico = excursión total (máx−mín) ·
RMS = energía vibratoria promedio · Factor cresta = Pico/RMS
(&gt;3.5 sugiere impactos, p. ej. defecto de rodamiento) ·
● <span style='color:#2563eb;'>velocidad</span>
<span style='color:#d97706;'>aceleración</span>
<span style='color:#7c3aed;'>proximidad</span></div>
""",
            unsafe_allow_html=True,
        )


__all__ = ["render_snapshot_waveforms"]
