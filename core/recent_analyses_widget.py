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
    if st.session_state.get("_wm_recent_css_injected_v2"):
        return
    st.session_state["_wm_recent_css_injected_v2"] = True
    st.markdown(
        """
        <style>
        .wm-recent-header {
            display: flex; align-items: baseline; gap: 12px;
            margin: 18px 0 10px 0;
        }
        .wm-recent-title {
            font-size: 16px; font-weight: 800; color: #0f172a;
            letter-spacing: -0.01em;
        }
        /* Ciclo 23.94 — Cards compactas y refinadas */
        .wm-recent-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 10px 12px;
            display: flex; flex-direction: column;
            transition: all 0.15s ease;
        }
        .wm-recent-card:hover {
            border-color: #94a3b8;
            box-shadow: 0 4px 12px rgba(15,23,42,0.06);
        }
        .wm-recent-card.empty {
            background: #f8fafc;
            border-color: #e2e8f0;
            border-style: dashed;
            opacity: 0.6;
        }
        /* Ciclo 23.92 — label + ago en líneas separadas (antes
           se pegaban con flex en cards angostas) */
        .wm-recent-label {
            font-size: 13px; font-weight: 800; color: #0f172a;
            display: block;
            margin-bottom: 2px;
        }
        .wm-recent-ago {
            font-size: 11px; color: #64748b;
            font-family: ui-monospace, SF Mono, monospace;
            display: block;
            margin-bottom: 8px;
        }
        .wm-recent-visual {
            display: flex; align-items: center; justify-content: center;
            margin: 2px 0 4px 0;
            opacity: 0.85;
        }
        .wm-recent-meta {
            font-size: 11px; color: #64748b;
            text-align: center;
            margin-bottom: 4px;
        }
        .wm-recent-empty-msg {
            font-size: 11px; color: #94a3b8;
            font-style: italic; text-align: center;
            padding: 20px 8px;
        }
        .wm-recent-sev-Normal  { color: #15803d; font-weight: 700; }
        .wm-recent-sev-Alarma  { color: #b45309; font-weight: 700; }
        .wm-recent-sev-Danger  { color: #b91c1c; font-weight: 700; }
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

    # Header de sección — Ciclo 23.89: simplificado, menos verboso.
    st.markdown(
        f"<div class='wm-recent-header'>"
        f"<span class='wm-recent-title'>📊 Última data</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Fetch metadata de los 4 tipos
    metadata_by_type: Dict[str, Optional[Dict[str, Any]]] = {}
    for atype in ANALYSIS_TYPES:
        try:
            mod = importlib.import_module(atype["module"])
            list_fn = getattr(mod, atype["list_fn"])
            snaps = list_fn(instance_id, limit=1)
            metadata_by_type[atype["key"]] = snaps[0] if snaps else None
        except Exception:
            metadata_by_type[atype["key"]] = None

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
            f"<div class='wm-recent-row'>"
            f"<span class='wm-recent-label'>{atype['icon']} {atype['label']}</span>"
            f"</div>"
            f"<div class='wm-recent-empty-msg'>Sin snapshots todavía</div>"
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
        f"<span class='wm-recent-label'>{atype['icon']} {atype['label']}</span>"
        f"<span class='wm-recent-ago'>{ago}</span>"
        f"<div class='wm-recent-visual'>{svg}</div>"
        f"{f'<div class=wm-recent-meta>{meta_html}</div>' if meta_html else ''}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Ciclo 23.90 — Waveform abre módulo dedicado vía st.button + st.switch_page.
    # `<a href>` con session auth no funciona en Streamlit Cloud (siempre va a
    # login). switch_page mantiene la sesión.
    if atype["key"] == "waveform":
        snap_id = meta.get("snapshot_id", "")
        if st.button(
            f"📊 Abrir",
            key=f"wm_open_{atype['key']}_{snap_id}",
            use_container_width=True,
            type="primary",
        ):
            # Pre-set en session_state — Time Waveforms lo lee al cargar
            st.session_state["_pending_snapshot_load"] = {
                "snapshot_id": snap_id,
                "instance_id": instance_id,
                "snapshot_type": "waveform",
            }
            try:
                st.switch_page("pages/02_Time_Waveforms.py")
            except Exception:
                # Fallback si switch_page falla (Streamlit muy viejo)
                st.error("No se pudo navegar a Time Waveforms. Refrescá la página.")
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
