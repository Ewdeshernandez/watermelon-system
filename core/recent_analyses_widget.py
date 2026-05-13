"""
core.recent_analyses_widget
============================

Sección "📊 Últimos análisis" para Live Monitoring (Ciclo 23.83).

Muestra al cliente final una vista de los últimos snapshots disponibles
del activo activo (Waveform, Spectrum, Orbit, Tabular). Cada tipo
aparece como una card compacta con metadata + botón "Ver detalle"
que despliega el plot completo inline.

El cliente NO necesita ir a Load Data ni a los módulos individuales —
ve directamente lo último que el especialista subió.

API pública:

    render_recent_analyses_section(instance_id)

Estilo: cards en grid 4 columnas con icono, timestamp relativo,
label de corrida, count de sensores, severidad worst-case.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st


# =============================================================
# Tipos de análisis y sus list_X_snapshots
# =============================================================

ANALYSIS_TYPES = [
    {
        "key": "waveform",
        "icon": "📈",
        "label": "Forma de onda",
        "module": "core.waveform_history",
        "list_fn": "list_waveform_snapshots",
        "load_fn": "load_waveform_snapshot",
        "render_fn": "_render_waveform_detail",
        "data_key": "sensors",
    },
    {
        "key": "spectrum",
        "icon": "🔍",
        "label": "Espectro",
        "module": "core.spectrum_history",
        "list_fn": "list_spectrum_snapshots",
        "load_fn": "load_spectrum_snapshot",
        "render_fn": "_render_spectrum_detail",
        "data_key": "sensors",
    },
    {
        "key": "orbit",
        "icon": "🌀",
        "label": "Órbita",
        "module": "core.orbit_history",
        "list_fn": "list_orbit_snapshots",
        "load_fn": "load_orbit_snapshot",
        "render_fn": "_render_orbit_detail",
        "data_key": "bearings",
    },
    {
        "key": "tabular",
        "icon": "📋",
        "label": "Tabular",
        "module": "core.tabular_history",
        "list_fn": "list_tabular_snapshots",
        "load_fn": "load_tabular_snapshot",
        "render_fn": "_render_tabular_detail",
        "data_key": "channels",
    },
]


# =============================================================
# Helpers de tiempo
# =============================================================

def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _format_time_ago(ts_str: str) -> str:
    dt = _parse_timestamp(ts_str)
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta_sec = (datetime.now(timezone.utc) - dt).total_seconds()
    if delta_sec < 60:
        return f"hace {int(delta_sec)} s"
    if delta_sec < 3600:
        return f"hace {int(delta_sec / 60)} min"
    if delta_sec < 86400:
        return f"hace {int(delta_sec / 3600)} h"
    days = int(delta_sec / 86400)
    if days < 30:
        return f"hace {days} d"
    return dt.strftime("%d %b")


# =============================================================
# CSS
# =============================================================

def _inject_css_once():
    if st.session_state.get("_wm_recent_css_injected_v3"):
        return
    st.session_state["_wm_recent_css_injected_v3"] = True
    st.markdown(
        """
        <style>
        /* ── Ciclo 23.124 — Sección "Última data" rediseñada
           Estilo Linear/Notion/Bently System1: minimalista, tipografía clara,
           jerarquía visual real. ── */

        /* Section header (Última data + meta count) */
        .wm-recent-header {
            display: flex; align-items: center; justify-content: space-between;
            gap: 12px; margin: 22px 0 12px 0;
        }
        .wm-recent-title {
            font-size: 13px; font-weight: 800; color: #0f172a;
            letter-spacing: 0.06em; text-transform: uppercase;
        }
        .wm-recent-title-meta {
            font-size: 11px; color: #94a3b8;
            font-family: ui-monospace, SF Mono, "Cascadia Code", monospace;
            letter-spacing: 0.02em;
        }

        /* Cards */
        .wm-recent-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 14px 14px 12px 14px;
            display: flex; flex-direction: column;
            gap: 0;
            transition: all 0.18s cubic-bezier(.4,0,.2,1);
            min-height: 130px;
        }
        .wm-recent-card:hover {
            border-color: #cbd5e1;
            box-shadow:
                0 1px 2px rgba(15,23,42,0.04),
                0 6px 14px rgba(15,23,42,0.06);
            transform: translateY(-1px);
        }
        .wm-recent-card.empty {
            background: #fafbfc;
            border-color: #e2e8f0;
            border-style: dashed;
            min-height: 130px;
            justify-content: center;
            align-items: center;
        }

        /* Card top row: icon + label · ago timestamp */
        .wm-recent-toprow {
            display: flex; align-items: center; justify-content: space-between;
            gap: 8px;
        }
        .wm-recent-label {
            font-size: 13px; font-weight: 700; color: #0f172a;
            letter-spacing: -0.005em;
            display: inline-flex; align-items: center; gap: 6px;
            line-height: 1.2;
        }
        .wm-recent-ago {
            font-size: 10.5px; color: #94a3b8;
            font-family: ui-monospace, SF Mono, "Cascadia Code", monospace;
            letter-spacing: 0.01em;
            white-space: nowrap;
        }

        /* SVG visual */
        .wm-recent-visual {
            display: flex; align-items: center; justify-content: center;
            margin: 12px 0 8px 0;
            opacity: 0.78;
            min-height: 36px;
        }
        .wm-recent-meta {
            font-size: 10.5px; color: #64748b;
            text-align: center;
            margin-bottom: 4px;
        }
        .wm-recent-empty-msg {
            font-size: 11.5px; color: #94a3b8;
            font-style: italic; text-align: center;
        }

        /* Severity chips for tabular */
        .wm-recent-sev-Normal  { color: #15803d; font-weight: 700; font-size: 10.5px; }
        .wm-recent-sev-Alarma  { color: #b45309; font-weight: 700; font-size: 10.5px; }
        .wm-recent-sev-Danger  { color: #b91c1c; font-weight: 700; font-size: 10.5px; }

        /* ── Botones "Abrir" — estilo discreto outlined ── */
        /* Aplica al st.button con type="secondary" dentro del recent block */
        .wm-recent-btn-host + div [data-testid="stButton"] button {
            background: #ffffff !important;
            color: #1e40af !important;
            border: 1px solid #dbeafe !important;
            border-radius: 8px !important;
            font-size: 12px !important;
            font-weight: 700 !important;
            letter-spacing: 0.02em !important;
            padding: 6px 14px !important;
            min-height: 32px !important;
            height: 32px !important;
            box-shadow: 0 1px 0 rgba(30,64,175,0.04) !important;
            transition: all 0.15s ease !important;
        }
        .wm-recent-btn-host + div [data-testid="stButton"] button:hover {
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%) !important;
            border-color: #93c5fd !important;
            color: #1e3a8a !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(30,64,175,0.10) !important;
        }
        .wm-recent-btn-host + div [data-testid="stButton"] button p {
            color: inherit !important;
        }
        .wm-recent-btn-host { display: block; height: 0; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Mini-SVGs por tipo de análisis — visual rápido del contenido
_TYPE_SVG = {
    "waveform": (
        '<svg viewBox="0 0 80 32" width="80" height="32">'
        '<path d="M0 16 Q5 4, 10 16 T20 16 T30 16 T40 16 T50 16 T60 16 T70 16 T80 16" '
        'fill="none" stroke="#2563eb" stroke-width="1.5"/>'
        '</svg>'
    ),
    "spectrum": (
        '<svg viewBox="0 0 80 32" width="80" height="32">'
        '<line x1="5"  y1="32" x2="5"  y2="20" stroke="#2563eb" stroke-width="2"/>'
        '<line x1="12" y1="32" x2="12" y2="8"  stroke="#2563eb" stroke-width="2"/>'
        '<line x1="19" y1="32" x2="19" y2="22" stroke="#2563eb" stroke-width="2"/>'
        '<line x1="26" y1="32" x2="26" y2="14" stroke="#2563eb" stroke-width="2"/>'
        '<line x1="33" y1="32" x2="33" y2="24" stroke="#2563eb" stroke-width="2"/>'
        '<line x1="40" y1="32" x2="40" y2="12" stroke="#2563eb" stroke-width="2"/>'
        '<line x1="47" y1="32" x2="47" y2="26" stroke="#2563eb" stroke-width="2"/>'
        '<line x1="54" y1="32" x2="54" y2="18" stroke="#2563eb" stroke-width="2"/>'
        '<line x1="61" y1="32" x2="61" y2="28" stroke="#2563eb" stroke-width="2"/>'
        '<line x1="68" y1="32" x2="68" y2="22" stroke="#2563eb" stroke-width="2"/>'
        '<line x1="75" y1="32" x2="75" y2="26" stroke="#2563eb" stroke-width="2"/>'
        '</svg>'
    ),
    "orbit": (
        '<svg viewBox="0 0 80 32" width="60" height="32">'
        '<ellipse cx="40" cy="16" rx="22" ry="12" fill="none" '
        'stroke="#2563eb" stroke-width="1.5"/>'
        '<circle cx="40" cy="16" r="2" fill="#2563eb"/>'
        '</svg>'
    ),
    "tabular": (
        '<svg viewBox="0 0 80 32" width="80" height="32">'
        '<rect x="2"  y="6"  width="76" height="5" rx="1" fill="#cbd5e1"/>'
        '<rect x="2"  y="14" width="76" height="3" rx="1" fill="#e2e8f0"/>'
        '<rect x="2"  y="20" width="76" height="3" rx="1" fill="#e2e8f0"/>'
        '<rect x="2"  y="26" width="76" height="3" rx="1" fill="#e2e8f0"/>'
        '</svg>'
    ),
}


# =============================================================
# Render helpers para cada tipo de detalle
# =============================================================

def _render_waveform_detail(payload: Dict[str, Any]) -> None:
    """Plot del waveform snapshot — Ciclo 23.86 usa render reusable
    con estilo idéntico al módulo Time Waveforms (un subplot por sensor)."""
    try:
        from core.waveform_render import render_snapshot_waveforms
        render_snapshot_waveforms(payload)
    except Exception as e:
        st.error(f"Error renderizando waveform: {e}")


def _render_spectrum_detail(payload: Dict[str, Any]) -> None:
    sensors = payload.get("sensors", [])
    if not sensors:
        st.info("Sin sensores en este snapshot.")
        return
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for s in sensors[:6]:  # max 6 traces — espectros se solapan feos con más
            f = s.get("freqs", [])
            a = s.get("amps", [])
            if not f or not a:
                continue
            fig.add_trace(go.Scatter(
                x=f, y=a, mode="lines",
                line=dict(width=1.0),
                name=s.get("sensor_label", ""),
                hovertemplate=f"<b>{s.get('sensor_label', '')}</b><br>"
                              f"%{{x:.1f}} Hz, %{{y:.4f}} {s.get('amp_unit', '')}<extra></extra>",
            ))
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="white",
            xaxis=dict(title="Frecuencia (Hz)", showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title="Amplitud", showgrid=True, gridcolor="#f1f5f9"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top peaks
        for s in sensors[:3]:
            peaks = s.get("peaks", [])
            if not peaks:
                continue
            st.caption(
                f"**{s.get('sensor_label', '')}** — top peaks: " +
                " · ".join([f"{p['freq']:.1f} Hz ({p['amp']:.3f})" for p in peaks[:5]])
            )
    except Exception as e:
        st.error(f"Error renderizando spectrum: {e}")


def _render_orbit_detail(payload: Dict[str, Any]) -> None:
    bearings = payload.get("bearings", [])
    if not bearings:
        st.info("Sin órbitas en este snapshot.")
        return
    try:
        import plotly.graph_objects as go
        cols = st.columns(min(len(bearings), 3))
        for idx, b in enumerate(bearings):
            with cols[idx % len(cols)]:
                fig = go.Figure()
                x = b.get("x_values", [])
                y = b.get("y_values", [])
                if not x or not y:
                    st.warning(f"{b.get('bearing_label', '')}: sin datos")
                    continue
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines",
                    line=dict(width=1.2, color="#2563eb"),
                    name="orbit",
                    hoverinfo="skip",
                ))
                fig.update_layout(
                    title=dict(
                        text=f"<b>{b.get('bearing_label', '')}</b>",
                        font=dict(size=12),
                    ),
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                    plot_bgcolor="white",
                    xaxis=dict(
                        title=f"X ({b.get('x_sensor_label', '')})",
                        showgrid=True, gridcolor="#f1f5f9",
                        zeroline=True, zerolinecolor="#cbd5e1",
                        scaleanchor="y", scaleratio=1,
                    ),
                    yaxis=dict(
                        title=f"Y ({b.get('y_sensor_label', '')})",
                        showgrid=True, gridcolor="#f1f5f9",
                        zeroline=True, zerolinecolor="#cbd5e1",
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
                v1x = b.get("vector_1x", {}) or {}
                if v1x.get("amp_x"):
                    st.caption(
                        f"1X X: {v1x.get('amp_x', 0):.3f} · "
                        f"Y: {v1x.get('amp_y', 0):.3f}"
                    )
    except Exception as e:
        st.error(f"Error renderizando orbit: {e}")


def _render_tabular_detail(payload: Dict[str, Any]) -> None:
    channels = payload.get("channels", [])
    if not channels:
        st.info("Sin canales en este snapshot.")
        return
    rows = []
    for c in channels:
        rows.append({
            "Sensor": c.get("sensor_label", ""),
            "Direct": f"{c.get('direct', 0):.3f}",
            "Unidad": c.get("direct_unit", ""),
            "1X amp": f"{c.get('vector_1x_amp', 0):.3f}",
            "2X amp": f"{c.get('vector_2x_amp', 0):.3f}",
            "Gap (V)": f"{c.get('gap_voltage'):.2f}" if c.get("gap_voltage") is not None else "—",
            "Severidad": c.get("severity", "") or "—",
            "Zona ISO": c.get("iso_zone", "") or "—",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


_RENDER_FUNCTIONS = {
    "_render_waveform_detail": _render_waveform_detail,
    "_render_spectrum_detail": _render_spectrum_detail,
    "_render_orbit_detail":    _render_orbit_detail,
    "_render_tabular_detail":  _render_tabular_detail,
}


# =============================================================
# Función principal
# =============================================================

def render_recent_analyses_section(instance_id: str) -> None:
    """Renderiza la sección '📊 Últimos análisis' debajo del diagrama.

    Muestra 4 cards (Waveform, Spectrum, Orbit, Tabular) con metadata
    del snapshot más reciente. Click "Ver detalle" despliega el plot
    completo inline.
    """
    if not instance_id:
        return

    _inject_css_once()
    import importlib

    # Fetch metadata de los 4 tipos PRIMERO para sacar el conteo total
    metadata_by_type: Dict[str, Optional[Dict[str, Any]]] = {}
    for atype in ANALYSIS_TYPES:
        try:
            mod = importlib.import_module(atype["module"])
            list_fn = getattr(mod, atype["list_fn"])
            snaps = list_fn(instance_id, limit=1)
            metadata_by_type[atype["key"]] = snaps[0] if snaps else None
        except Exception:
            metadata_by_type[atype["key"]] = None

    # Header con conteo discreto a la derecha
    n_with_data = sum(1 for v in metadata_by_type.values() if v is not None)
    st.markdown(
        f"<div class='wm-recent-header'>"
        f"<span class='wm-recent-title'>ÚLTIMA DATA</span>"
        f"<span class='wm-recent-title-meta'>"
        f"{n_with_data} / {len(ANALYSIS_TYPES)} disponibles"
        f"</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Render cards en 4 columnas
    cols = st.columns(4)
    for idx, atype in enumerate(ANALYSIS_TYPES):
        with cols[idx]:
            meta = metadata_by_type[atype["key"]]
            _render_card(atype, meta, instance_id)


def _render_card(atype: Dict[str, Any], meta: Optional[Dict[str, Any]], instance_id: str) -> None:
    """Card minimalista — Ciclo 23.90:
       solo icon + label + ago + mini-SVG + count compact + botón abrir."""
    if meta is None:
        st.markdown(
            f"<div class='wm-recent-card empty'>"
            f"  <div class='wm-recent-toprow'>"
            f"    <span class='wm-recent-label'>{atype['icon']} {atype['label']}</span>"
            f"    <span class='wm-recent-ago'>—</span>"
            f"  </div>"
            f"  <div class='wm-recent-empty-msg'>Sin snapshots todavía</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    ago = _format_time_ago(meta.get("timestamp", ""))

    # Ciclo 23.91 — solo mostrar severidad si es Alarma/Danger (Tabular).
    # Para los otros tipos no mostramos meta — solo el visual SVG habla.
    meta_html = ""
    if atype["key"] == "tabular":
        worst = meta.get("worst_severity", "")
        if worst in ("Alarma", "Danger"):
            meta_html = f"<span class='wm-recent-sev-{worst}'>{worst}</span>"

    svg = _TYPE_SVG.get(atype["key"], "")

    st.markdown(
        f"<div class='wm-recent-card'>"
        f"  <div class='wm-recent-toprow'>"
        f"    <span class='wm-recent-label'>{atype['icon']} {atype['label']}</span>"
        f"    <span class='wm-recent-ago'>{ago}</span>"
        f"  </div>"
        f"  <div class='wm-recent-visual'>{svg}</div>"
        f"  {f'<div class=wm-recent-meta>{meta_html}</div>' if meta_html else ''}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Ciclo 23.90/23.107/23.124 — Waveform & Spectrum & Orbit & Tabular abren
    # módulo dedicado vía st.button + st.switch_page (switch_page mantiene
    # la sesión; `<a href>` con session auth no funciona en Streamlit Cloud).
    _REDIRECT_TARGETS = {
        "waveform": "pages/02_Time_Waveforms.py",
        "spectrum": "pages/03_Spectrum.py",
        "orbit":    "pages/05_Orbit_Analysis.py",
        "tabular":  "pages/01__Tabular_List.py",
    }
    if atype["key"] in _REDIRECT_TARGETS:
        snap_id = meta.get("snapshot_id", "")
        # Span marker para que la CSS enganche el siguiente stButton
        st.markdown(
            '<span class="wm-recent-btn-host"></span>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Abrir  →",
            key=f"wm_open_{atype['key']}_{snap_id}",
            use_container_width=True,
            type="secondary",
        ):
            # Pre-set en session_state — la page lo lee al cargar
            st.session_state["_pending_snapshot_load"] = {
                "snapshot_id": snap_id,
                "instance_id": instance_id,
                "snapshot_type": atype["key"],
            }
            try:
                st.switch_page(_REDIRECT_TARGETS[atype["key"]])
            except Exception:
                st.error(f"No se pudo navegar a {atype['label']}. Refrescá la página.")
    else:
        # Otros tipos: preview inline (próxima version los migra a redirect)
        if st.button(
            f"Ver detalle",
            key=f"wm_recent_detail_{atype['key']}_{instance_id}",
            use_container_width=True,
        ):
            st.session_state[f"_wm_recent_expanded_{atype['key']}"] = True

        if st.session_state.get(f"_wm_recent_expanded_{atype['key']}"):
            _render_detail_expander(atype, meta, instance_id)


def _render_detail_expander(atype: Dict[str, Any], meta: Dict[str, Any], instance_id: str) -> None:
    """Carga el snapshot completo y lo renderiza inline."""
    import importlib
    try:
        mod = importlib.import_module(atype["module"])
        load_fn = getattr(mod, atype["load_fn"])
        payload = load_fn(instance_id, meta["snapshot_id"])
    except Exception as e:
        st.error(f"Error cargando snapshot: {e}")
        return

    if payload is None:
        st.warning("Snapshot no encontrado.")
        return

    st.markdown(f"### {atype['icon']} {atype['label']} — `{meta['snapshot_id']}`")
    if meta.get("corrida_label"):
        st.caption(f"📌 {meta['corrida_label']}")
    if meta.get("notes"):
        st.caption(f"📝 {meta['notes']}")
    if meta.get("operating_speed_rpm"):
        st.caption(f"⚙ {meta['operating_speed_rpm']:.0f} RPM")

    render_fn = _RENDER_FUNCTIONS.get(atype["render_fn"])
    if render_fn:
        render_fn(payload)

    if st.button("✕ Cerrar detalle", key=f"wm_recent_close_{atype['key']}"):
        st.session_state[f"_wm_recent_expanded_{atype['key']}"] = False
        st.rerun()


__all__ = ["render_recent_analyses_section"]
