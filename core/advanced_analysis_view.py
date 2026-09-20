"""
core/advanced_analysis_view.py
==============================

Render REUSABLE del Análisis Avanzado (Espectro · Forma de onda · Órbita) del
snapshot más reciente de un activo. Extraído de pages/_live_analysis.py para
poder embeberlo COMO EXPANDER dentro de Live Monitoring (pages/02) además de la
página dedicada. No hace set_page_config ni auth (eso lo pone el contenedor).

Estándares de visualización (clase System1/AMS):
  • Espectro: full-scale 60.000 CPM, cursores 1X/2X/3X, escala Y común por familia.
  • Forma de onda: escala Y simétrica común por familia.
  • Órbita: aspecto 1:1 por cojinete.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

_VIEWS = [
    ":material/equalizer: Spectrum",
    ":material/show_chart: Waveform",
    ":material/track_changes: Orbit",
]


def _get_fam_units(iid: str) -> dict:
    """{familia: unidad} desde los datos en vivo del activo. Cache 5 min."""
    from core.recent_analyses_widget import _unit_family
    _ck = f"_wm_la_fam_units_{iid}"
    cached = st.session_state.get(_ck)
    import time as _t
    if cached and (_t.time() - cached.get("ts", 0)) < 300:
        return cached["data"]
    out: dict = {}
    try:
        from core.live_readings import latest_for_instance
        for r in (latest_for_instance(iid) or []):
            u = (r.get("unit") or "").strip()
            if not u:
                continue
            out.setdefault(_unit_family(u), u)
    except Exception:  # noqa: BLE001
        out = {}
    st.session_state[_ck] = {"data": out, "ts": _t.time()}
    return out


def render_advanced_analysis(instance_id: str, tag: Optional[str] = None,
                             show_asset_chip: bool = True) -> None:
    """Dibuja el análisis avanzado del activo. Seguro para embeber en un
    expander (usa `return`, no `st.stop()`)."""
    if not instance_id:
        st.info("Pick an asset first — Advanced analysis uses its latest snapshot.")
        return

    from core.recent_analyses_widget import (
        _render_orbit_detail,
        _render_spectrum_detail,
        _render_waveform_detail,
        list_snapshots_brief,
        load_snapshot_payload,
    )

    _tag = (tag or str(instance_id)).upper()

    c_left, c_right = st.columns([3, 2])
    with c_left:
        if show_asset_chip:
            st.markdown(
                f"<div style='padding-top:6px;'>"
                f"<span style='background:#0f172a;color:#f1f5f9;border-radius:8px;"
                f"padding:4px 12px;font-weight:700;font-size:13px;"
                f"letter-spacing:0.06em;'>{_tag}</span></div>",
                unsafe_allow_html=True,
            )
    with c_right:
        try:
            _view = st.segmented_control(
                "View", _VIEWS, default=_VIEWS[0],
                key=f"wm_la_view_{instance_id}", label_visibility="collapsed",
            )
        except Exception:  # noqa: BLE001
            _view = st.radio(
                "View", _VIEWS, horizontal=True,
                key=f"wm_la_view_radio_{instance_id}", label_visibility="collapsed",
            )
    _view = _view or _VIEWS[0]
    st.markdown("")

    _key_by_view = {
        _VIEWS[0]: ("spectrum", _render_spectrum_detail),
        _VIEWS[1]: ("waveform", _render_waveform_detail),
        _VIEWS[2]: ("orbit", _render_orbit_detail),
    }
    _akey, _render = _key_by_view[_view]

    _snaps = list_snapshots_brief(instance_id, _akey)
    if not _snaps:
        st.info("There is no snapshot of this type for the asset yet. "
                "It is generated from Load Data / analysis modules.")
        return

    _sid_options = [s["snapshot_id"] for s in _snaps]
    _label_by_sid = {s["snapshot_id"]: s["date_label"] for s in _snaps}

    _dc1, _dc2 = st.columns([2, 3])
    with _dc1:
        _sel_sid = st.selectbox(
            "📅 Data date", _sid_options, index=0,
            format_func=lambda sid: _label_by_sid.get(sid, sid),
            key=f"wm_la_snapdate_{_akey}_{instance_id}",
            help="Each date is a saved run with all its channels.",
        )
    with _dc2:
        st.markdown(
            f"<div style='padding-top:30px;color:#475569;font-size:13px;'>"
            f"Showing data from <b>{_label_by_sid.get(_sel_sid, _sel_sid)}</b> "
            f"· {len(_snaps)} date(s) available</div>",
            unsafe_allow_html=True,
        )

    with st.expander("🗑️ Delete this date (if uploaded by mistake)", expanded=False):
        st.caption(f"The snapshot **{_label_by_sid.get(_sel_sid, _sel_sid)}** for this "
                   "asset will be permanently deleted. This cannot be undone.")
        _confirm_del = st.checkbox("I confirm I want to delete this date",
                                   key=f"wm_la_delconfirm_{_akey}_{instance_id}")
        if st.button("Delete permanently", disabled=not _confirm_del,
                     key=f"wm_la_delbtn_{_akey}_{instance_id}"):
            try:
                from core import history_storage as _hs_del
                _ok_del = _hs_del.delete_snapshot(instance_id, _akey, _sel_sid)
            except Exception:  # noqa: BLE001
                _ok_del = False
            if _ok_del:
                try:
                    from core.recent_analyses_widget import _measured_dt_cached
                    _measured_dt_cached.cache_clear()
                except Exception:  # noqa: BLE001
                    pass
                st.success("Snapshot deleted.")
                st.rerun()
            else:
                st.error("Could not delete the snapshot. Try again.")

    with st.spinner("Loading snapshot…"):
        _payload = load_snapshot_payload(instance_id, _akey, _sel_sid)

    if not _payload:
        st.info("Could not load the selected snapshot.")
        return
    _fu = _get_fam_units(instance_id)
    try:
        _render(_payload, fam_units=_fu)
    except TypeError:
        _render(_payload)   # órbita no recibe fam_units


__all__ = ["render_advanced_analysis"]
