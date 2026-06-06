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
        "icon": "",   # Ciclo 23.126 — sin emoji, el SVG visual ya identifica el tipo
        "label": "Forma de Onda",
        "module": "core.waveform_history",
        "list_fn": "list_waveform_snapshots",
        "load_fn": "load_waveform_snapshot",
        "render_fn": "_render_waveform_detail",
        "data_key": "sensors",
    },
    {
        "key": "spectrum",
        "icon": "",
        "label": "Espectro",
        "module": "core.spectrum_history",
        "list_fn": "list_spectrum_snapshots",
        "load_fn": "load_spectrum_snapshot",
        "render_fn": "_render_spectrum_detail",
        "data_key": "sensors",
    },
    {
        "key": "orbit",
        "icon": "",
        "label": "Órbita",
        "module": "core.orbit_history",
        "list_fn": "list_orbit_snapshots",
        "load_fn": "load_orbit_snapshot",
        "render_fn": "_render_orbit_detail",
        "data_key": "bearings",
    },
    # Ciclo 23.148 — "Tabular List" se quitó: es redundante con la tabla
    # "Canales — Overall + vectores 1X/2X (API 670)" del overview.
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

        /* Cards — compactas, sin caja blanca grande */
        .wm-recent-card {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 6px 4px 0 4px;
            display: flex; flex-direction: column;
            gap: 4px;
            transition: opacity 0.15s ease;
        }
        .wm-recent-card:hover {
            opacity: 0.88;
        }
        .wm-recent-card.empty {
            opacity: 0.45;
        }

        /* Ciclo 23.126 — Label row: solo el label, centrado, sin icon/ago */
        .wm-recent-label-row {
            display: flex; align-items: center; justify-content: center;
        }
        .wm-recent-label {
            font-size: 13.5px; font-weight: 700; color: #0f172a;
            letter-spacing: -0.005em;
            text-align: center;
            line-height: 1.2;
        }

        /* SVG visual — compacto, sin margin grande */
        .wm-recent-visual {
            display: flex; align-items: center; justify-content: center;
            margin: 2px 0 6px 0;
            opacity: 0.85;
        }
        .wm-recent-meta {
            font-size: 10.5px; color: #64748b;
            text-align: center;
            margin-bottom: 4px;
        }
        .wm-recent-empty-msg {
            font-size: 11px; color: #94a3b8;
            font-style: italic; text-align: center;
            padding: 6px 0 2px 0;
        }

        /* Severity chips for tabular */
        .wm-recent-sev-Normal  { color: #15803d; font-weight: 700; font-size: 10.5px; }
        .wm-recent-sev-Alarma  { color: #b45309; font-weight: 700; font-size: 10.5px; }
        .wm-recent-sev-Danger  { color: #b91c1c; font-weight: 700; font-size: 10.5px; }

        /* ── Botones "Abrir" — texto link discreto sin caja ── */
        .wm-recent-btn-host + div [data-testid="stButton"] button {
            background: transparent !important;
            color: #1e40af !important;
            border: none !important;
            border-radius: 0 !important;
            font-size: 12px !important;
            font-weight: 700 !important;
            letter-spacing: 0.02em !important;
            padding: 2px 4px !important;
            min-height: 22px !important;
            height: auto !important;
            box-shadow: none !important;
            transition: color 0.15s ease, transform 0.15s ease !important;
        }
        .wm-recent-btn-host + div [data-testid="stButton"] button:hover {
            background: transparent !important;
            color: #1e3a8a !important;
            transform: translateX(2px);
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


# Ciclo 23.150 — Espectro clase mundial (Fase 1): apilado por sensor (NO
# encimado), color consistente por sensor, y cursores de orden 1X/2X/3X.
# Paleta categórica fija (asignada por orden alfabético del label → estable
# entre snapshots de la misma máquina).
_SPEC_PALETTE = [
    "#1D9E75", "#378ADD", "#7F77DD", "#D85A30",
    "#D4537E", "#BA7517", "#0F6E56", "#185FA5",
]


def _spec_color(label: str, ordered_labels: List[str]) -> str:
    try:
        return _SPEC_PALETTE[ordered_labels.index(label) % len(_SPEC_PALETTE)]
    except Exception:
        return _SPEC_PALETTE[0]


def _estimate_running_hz(sensors: List[Dict[str, Any]]) -> Optional[float]:
    """Estima la frecuencia de giro (1X) desde el pico dominante global.
    En máquinas a régimen el 1X suele ser el pico más alto. Devuelve None
    si no hay picos confiables (entonces no se dibujan cursores de orden)."""
    best_amp, best_hz = -1.0, None
    for s in sensors:
        peaks = s.get("peaks") or []
        if not peaks:
            continue
        p = peaks[0]  # ya vienen ordenados por amplitud desc
        try:
            hz = float(p.get("freq", 0))
            amp = float(p.get("amp", 0))
        except Exception:
            continue
        if amp > best_amp and 3.0 <= hz <= 2000.0:
            best_amp, best_hz = amp, hz
    return best_hz


def _render_spectrum_detail(payload: Dict[str, Any]) -> None:
    sensors = [s for s in payload.get("sensors", [])
               if s.get("freqs") and s.get("amps")][:6]
    if not sensors:
        st.info("Sin sensores en este snapshot.")
        return
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        ordered = sorted({s.get("sensor_label", "") for s in sensors})
        run_hz = _estimate_running_hz(sensors)
        n = len(sensors)

        fig = make_subplots(
            rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=[s.get("sensor_label", "") for s in sensors],
        )
        # Recolorear + alinear a la izquierda los títulos (etiqueta-chip por canal)
        for ann, s in zip(fig.layout.annotations, sensors):
            ann.font = dict(size=11,
                            color=_spec_color(s.get("sensor_label", ""), ordered),
                            family="ui-monospace, SFMono-Regular, Menlo, monospace")
            ann.x = 0.0
            ann.xanchor = "left"

        for i, s in enumerate(sensors, start=1):
            lbl = s.get("sensor_label", "")
            unit = s.get("amp_unit", "")
            color = _spec_color(lbl, ordered)
            fig.add_trace(go.Scatter(
                x=s["freqs"], y=s["amps"], mode="lines",
                line=dict(width=1.2, color=color), showlegend=False,
                hovertemplate=(f"<b>{lbl}</b><br>%{{x:.1f}} Hz · "
                               f"%{{y:.4f}} {unit}<extra></extra>"),
            ), row=i, col=1)
            # Cursores de orden 1X / 2X / 3X (label solo en la primera fila)
            if run_hz:
                for k in (1, 2, 3):
                    vkw = dict(x=run_hz * k, row=i, col=1,
                               line=dict(color="#94a3b8", width=1, dash="dot"))
                    if i == 1:
                        vkw.update(annotation_text=f"{k}X",
                                   annotation_position="top",
                                   annotation_font=dict(size=9, color="#64748b"))
                    fig.add_vline(**vkw)

        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
        fig.update_xaxes(title_text="Frecuencia (Hz)", row=n, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#f8fafc", zeroline=False)
        fig.update_layout(
            height=max(150, 118 * n),
            margin=dict(l=46, r=14, t=22, b=34),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="-apple-system, system-ui, sans-serif",
                      size=10, color="#475569"),
            showlegend=False,
        )
        if run_hz:
            st.caption(
                f"Cursores de orden anclados a 1X ≈ {run_hz:.1f} Hz "
                f"(~{run_hz * 60:.0f} rpm), estimado del pico dominante."
            )
        st.plotly_chart(fig, use_container_width=True)
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
                _orb_palette = ["#2563eb", "#059669", "#d97706", "#dc2626",
                                "#7c3aed", "#0891b2"]
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines",
                    line=dict(width=1.2,
                              color=_orb_palette[idx % len(_orb_palette)]),
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

    # Ciclo 23.128 — PERFORMANCE FIX. Las cards solo necesitan saber:
    #   (a) si hay al menos 1 snapshot del tipo (para Abrir/Sin datos)
    #   (b) el snapshot_id del más reciente (para el redirect)
    # No necesitan timestamp/corrida_label/worst_severity (las quitamos en
    # v3.31.126). Antes llamábamos list_*_snapshots(limit=1) que DESCARGA
    # el JSON gzipped completo desde Supabase Storage por cada tipo —
    # 4 downloads secuenciales = 4-8s en Supabase Free tier.
    # Ahora: usamos history_storage.list_snapshots directo que solo lista
    # archivos del bucket (1 API call rápido, sin downloads). Cache en
    # session_state con TTL 60s para reruns y navegación.
    _cache_key = f"_wm_recent_meta_cache_v2_{instance_id}"
    _now_ts = datetime.now(timezone.utc).timestamp()
    _cached = st.session_state.get(_cache_key)
    if _cached and (_now_ts - _cached.get("ts", 0)) < 60:
        metadata_by_type = _cached["data"]
    else:
        from core import history_storage as _hs
        metadata_by_type: Dict[str, Optional[Dict[str, Any]]] = {}
        for atype in ANALYSIS_TYPES:
            try:
                # FAST PATH: solo list de archivos en el bucket
                snaps_list = _hs.list_snapshots(instance_id, atype["key"])
                if snaps_list:
                    # snaps_list[0] es el más reciente (ordenado desc por snapshot_id)
                    metadata_by_type[atype["key"]] = {
                        "snapshot_id": snaps_list[0].get("snapshot_id", ""),
                    }
                else:
                    metadata_by_type[atype["key"]] = None
            except Exception:
                metadata_by_type[atype["key"]] = None
        st.session_state[_cache_key] = {"data": metadata_by_type, "ts": _now_ts}

    # Ciclo 23.154 — PANEL UNIFICADO (pedido Ewdes): en vez de 3 cards con
    # botón "Abrir", se renderizan DIRECTO las tres vistas en orden
    # Espectro → Forma de onda → Órbita. El cliente lo ve todo de una,
    # minimalista; el analista además tiene link al módulo completo.
    # Los payloads se cachean en session_state por snapshot_id para no
    # re-descargar de Storage en cada rerun.
    try:
        from core.auth import get_current_user as _gcu
        _viewer_role = ((_gcu() or {}).get("role") or "").lower()
    except Exception:
        _viewer_role = ""

    _REDIRECT_TARGETS_UNIFIED = {
        "waveform": "pages/02_Time_Waveforms.py",
        "spectrum": "pages/03_Spectrum.py",
        "orbit":    "pages/05_Orbit_Analysis.py",
    }
    _UNIFIED_ORDER = ["spectrum", "waveform", "orbit"]
    _atype_by_key = {a["key"]: a for a in ANALYSIS_TYPES}
    _rendered_any = False
    for _key in _UNIFIED_ORDER:
        atype = _atype_by_key.get(_key)
        meta = metadata_by_type.get(_key)
        if not atype or not meta:
            continue
        _snap_id = meta.get("snapshot_id", "")
        _pcache_key = f"_wm_unified_payload_{_key}_{instance_id}_{_snap_id}"
        payload = st.session_state.get(_pcache_key)
        if payload is None:
            try:
                mod = importlib.import_module(atype["module"])
                load_fn = getattr(mod, atype["load_fn"])
                payload = load_fn(instance_id, _snap_id)
                if payload:
                    st.session_state[_pcache_key] = payload
            except Exception:
                payload = None
        if not payload:
            continue

        # Orden canónico de canales (velocidad → aceleración → proximidad)
        try:
            from core.channel_order import channel_sort_key
            dk = atype.get("data_key")
            items = payload.get(dk) if dk else None
            if isinstance(items, list) and items:
                if dk == "sensors":
                    payload[dk] = sorted(items, key=lambda s: channel_sort_key(
                        s.get("sensor_label", ""), s.get("amp_unit", "")))
                elif dk == "bearings":
                    payload[dk] = sorted(items, key=lambda b: str(b.get("bearing_label", "")))
        except Exception:
            pass

        _rendered_any = True
        st.markdown(
            f"<div style='font-size:11px;font-weight:800;letter-spacing:0.14em;"
            f"text-transform:uppercase;color:#475569;margin:14px 0 2px 0;'>"
            f"{atype['label']}</div>",
            unsafe_allow_html=True,
        )
        render_fn = _RENDER_FUNCTIONS.get(atype["render_fn"])
        if render_fn:
            try:
                render_fn(payload)
            except Exception as e:
                st.warning(f"No se pudo renderizar {atype['label']}: {e}")

        # Link al módulo completo — SOLO analistas (el cliente no lo ve)
        if _viewer_role != "client" and _key in _REDIRECT_TARGETS_UNIFIED:
            if st.button(
                f"Abrir {atype['label']} en módulo completo →",
                key=f"wm_unified_open_{_key}_{_snap_id}",
            ):
                st.session_state["_pending_snapshot_load"] = {
                    "snapshot_id": _snap_id,
                    "instance_id": instance_id,
                    "snapshot_type": _key,
                }
                try:
                    st.switch_page(_REDIRECT_TARGETS_UNIFIED[_key])
                except Exception:
                    st.error("No se pudo navegar al módulo.")

    if not _rendered_any:
        st.info("Aún no hay snapshots de análisis para este activo. "
                "Cargá señales en Load Data para generarlos.")


def _render_card(atype: Dict[str, Any], meta: Optional[Dict[str, Any]], instance_id: str) -> None:
    """Card minimalista — Ciclo 23.90:
       solo icon + label + ago + mini-SVG + count compact + botón abrir."""
    if meta is None:
        st.markdown(
            f"<div class='wm-recent-card empty'>"
            f"  <div class='wm-recent-label-row'>"
            f"    <span class='wm-recent-label'>{atype['label']}</span>"
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
        f"  <div class='wm-recent-label-row'>"
        f"    <span class='wm-recent-label'>{atype['label']}</span>"
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
