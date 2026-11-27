"""
core.rd_live_mirror — Espejo de ROTODINÁMICA dentro de Live Monitoring.
=======================================================================

Flujo: el robot sube la onda cruda (bucket dynamic_raw) → ROTODINÁMICA la
procesa (core.analysis.model.record_from_history + core.rd_figures, el MISMO
motor de pages/02_Analysis) → Live Monitoring PUBLICA esa misma captura
procesada: Forma de onda · Espectro · Órbita. Solo lectura, sin re-implementar
matemática: si Rotodinámica cambia, el espejo cambia igual.

Se actualiza solo: el fragment re-consulta cada 5 min y, cuando entra una
sesión nueva del robot, la procesa y la muestra.

Textos visibles al cliente: sin nombres de proveedores de terceros.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import streamlit as st

from core.portal_i18n import T

REFRESH_S = 300          # re-consulta de sesiones nuevas
STALE_H = 6.0            # más vieja que esto → aviso ámbar
# El robot nombra los archivos en HORA DE PLANTA (Colombia, UTC−5), no UTC.
PLANT_TZ = timezone(timedelta(hours=-5))

# Solo texto (estilo software de monitoreo profesional): sin íconos ni emojis.
_VIEWS = ["Waveform", "Spectrum", "Orbit"]
_VIEW_ES = {"Waveform": "Onda", "Spectrum": "Espectro", "Orbit": "Órbita"}


# ---------------------------------------------------------------- datos
@st.cache_data(ttl=120, show_spinner=False)
def _sessions(source_iid: str):
    from core.analysis.model import session_stamps
    from core.dynamic_raw import list_captures
    caps = list_captures(source_iid, limit=120) or []
    return session_stamps(caps)


@st.cache_data(ttl=900, show_spinner=False, max_entries=8)
def _processed(label: str, keys: Tuple[str, ...]):
    """Captura procesada por el motor de Rotodinámica (cache global por sesión)."""
    from core.analysis.model import record_from_history
    return record_from_history(label, list(keys))


def _age_hours(label: str) -> Optional[float]:
    """Edad de la sesión. El label es la hora de la captura (hora de planta)."""
    try:
        t = datetime.strptime(label.strip(), "%Y-%m-%d  %H:%M:%S").replace(tzinfo=PLANT_TZ)
    except ValueError:
        return None
    return max(0.0, (datetime.now(timezone.utc) - t).total_seconds() / 3600.0)


def _age_txt(h: Optional[float]) -> str:
    if h is None:
        return ""
    if h < 1:
        return f"{int(h * 60)} min ago"
    if h < 48:
        return f"{h:.0f} h ago"
    return f"{h / 24:.0f} days ago"


# ---------------------------------------------------------------- selección
def _groups(rec) -> List[Tuple[str, list]]:
    """[(nombre, [canales])]: un grupo por cojinete (Y antes que X) + canales
    sueltos (acelerómetros / velocímetros)."""
    out, used = [], set()
    for b in rec.bearings:
        if b.has_xy:
            out.append((f"{b.y.label}/{b.x.label}", [b.y, b.x]))
            used.update({b.x.label, b.y.label})
    singles = [c for c in rec.channels if c.label not in used]

    def _yfirst(c):
        m = re.match(r"(\d+)([XYZ])([A-Z]?)", (c.label or "").upper())
        if not m:
            return (99, 9, 9, c.label or "")
        return (int(m.group(1)), {"D": 0, "A": 1, "V": 2}.get(m.group(3), 3),
                {"Y": 0, "X": 1, "Z": 2}.get(m.group(2), 3), c.label)
    for c in sorted(singles, key=_yfirst):
        out.append((c.label, [c]))
    return out


def _pick(groups, key: str):
    names = [g[0] for g in groups]
    try:
        sel = st.segmented_control(T("Punto de medición", "Measurement point"), names, default=names[0],
                                   key=key, label_visibility="collapsed")
    except Exception:  # noqa: BLE001
        sel = st.radio(T("Punto de medición", "Measurement point"), names, horizontal=True, key=key + "_r",
                       label_visibility="collapsed")
    sel = sel if sel in names else names[0]
    return groups[names.index(sel)][1]


# ---------------------------------------------------------------- vistas
def _yx(bb) -> str:
    """Nombre del cojinete con Y antes que X (convención: 1YD/1XD)."""
    return f"{bb.y.label}/{bb.x.label}"


def _view_waveform(rec, k):
    from core.rd_figures import NOBAR, waveform_figure
    from core.ui_industrial import html_table
    chs = _pick(_groups(rec), f"{k}_wf_pt")
    fig, rows, unit = waveform_figure(chs, kph=True, title=T("Forma de onda", "Waveform"), h=430,
                                      time_label=T("Tiempo (ms)", "Time (ms)"),
                                      amp_label=T("Amplitud", "Amplitude"))
    st.plotly_chart(fig, use_container_width=True, config=NOBAR)
    html_table([T("Sensor", "Sensor"), T("Pico a pico", "Peak-to-peak"), "RMS", T("Pico", "Peak"), T("Factor de cresta", "Crest"), "RPM"], rows)
    st.caption(T(f"Puntos = marca de keyphasor (una por vuelta) · amplitud en {unit}.", f"Dots = keyphasor mark (one per revolution) · amplitude in {unit}."))


def _view_spectrum(rec, k):
    from core.rd_figures import NOBAR, spectrum_figure
    from core.ui_industrial import html_table
    chs = _pick(_groups(rec), f"{k}_sp_pt")
    fig, rows = spectrum_figure(chs, xmax=-1, title=T("Espectro (FFT)", "Spectrum (FFT)"), h=430,
                                freq_label=T("Frecuencia (cpm)", "Frequency (cpm)"),
                                amp_label=T("Amplitud", "Amplitude"))
    if fig is None:
        st.info(T("Esta captura no tiene muestras suficientes para el espectro.", "Not enough samples in this capture to compute the spectrum."))
        return
    st.plotly_chart(fig, use_container_width=True, config=NOBAR)
    if rows:
        html_table(["Sensor", "RPM", T("Global", "Overall"), "1X", "2X", "3X", "4X"], rows)
    st.caption("Frequency in cpm · orders 1X–4X marked · displacement in pp, "
               "velocity / acceleration in RMS.")


def _view_orbit(rec, k):
    from core.rd_figures import NOBAR, bearing_angles, orbit_data, orbit_figure, orbit_row
    from core.ui_industrial import html_table
    brgs = [b for b in rec.bearings if b.has_xy]
    if not brgs:
        st.info(T("Esta captura no tiene par de sondas X/Y — la órbita necesita ambas sondas del cojinete.", "This capture has no X/Y probe pair — orbits need both probes of a bearing."))
        return
    try:
        fmode = st.segmented_control(T("Tipo de órbita", "Orbit type"), ["Direct", "1X", "2X"], default="Direct",
                                     key=f"{k}_orb_f", label_visibility="collapsed") or "Direct"
    except Exception:  # noqa: BLE001
        fmode = "Direct"

    def _ang(bb):
        ax, ay, _z = bearing_angles(bb)
        return ax, ay

    datas = [(bb, orbit_data(bb, fmode, *_ang(bb))) for bb in brgs]
    R = max((d["Rp"] for _, d in datas), default=1.0) or 1.0
    if len(datas) == 1:
        bb, d = datas[0]
        fig = orbit_figure(bb, d, R, *_ang(bb), fmode, h=520,
                           title=f"{T('Órbita', 'Orbit')} · {_yx(bb)} · {fmode}")
        st.plotly_chart(fig, use_container_width=True, config=NOBAR)
    else:
        # Una figura por cojinete, MISMA escala R, en columnas de 2: en pantalla
        # angosta (celular) Streamlit las apila solas en vez de apretar una grilla.
        for i in range(0, len(datas), 2):
            cols = st.columns(2)
            for col, (bb, d) in zip(cols, datas[i:i + 2]):
                with col:
                    fig = orbit_figure(bb, d, R, *_ang(bb), fmode, h=380, compact=True,
                                       title=f"{_yx(bb)} · {fmode}")
                    st.plotly_chart(fig, use_container_width=True, config=NOBAR,
                                    key=f"{k}_orb_{bb.name}")
    # Precesión fwd/rev NO se publica al cliente hasta que la captura guarde fs/spr
    # (hoy no es confiable); queda en Rotodinámica para el analista.
    rows = []
    for bb, d in datas:
        r = orbit_row(bb, d)
        rows.append([_yx(bb), r[1]] + r[3:])
    html_table([T("Cojinete", "Bearing"), "RPM", T("Giro (API 670)", "Rotation (API 670)"),
                T("Eje mayor", "Major axis"), T("Eje menor", "Minor axis"),
                T("Elipticidad", "Ellipticity"), "Smax"], rows)
    st.caption("Machine coordinates (vertical ↑ / horizontal →) · 1:1 aspect · "
               "green dot = keyphasor · all orbits on the same scale to compare size.")


# ---------------------------------------------------------------- render
def render_rd_mirror(instance_id: str) -> None:
    """Dibuja el espejo. Seguro para embeber (nunca lanza hacia Live)."""
    try:
        from core.demo_access import source_for
        src = source_for(instance_id)
    except Exception:  # noqa: BLE001
        src = instance_id
    k = f"rdm_{instance_id}"

    @st.fragment(run_every=REFRESH_S)
    def _body():
        try:
            sess = _sessions(src)
        except Exception:  # noqa: BLE001
            sess = []
        if not sess:
            st.info(T("Aún no hay captura dinámica para esta máquina. Apenas el colector suba "
                      "una, se procesa y aparece aquí.",
                      "No dynamic capture yet for this machine. As soon as the data "
                      "collector uploads one, Rotordynamics processes it and it appears here."))
            return
        label, keys = sess[0]
        with st.spinner(T("Procesando la última captura…", "Processing latest capture…")):
            rec = _processed(label, tuple(keys))
        if rec is None or not rec.channels:
            st.info(T("La última captura no se pudo procesar.",
                      "The latest capture could not be processed. Showing nothing rather "
                      "than stale data."))
            return
        age = _age_hours(label)
        stale = age is not None and age > STALE_H
        dot = "#e8890c" if stale else "#1f9d55"
        rpm = rec.rpm
        st.markdown(
            "<div style='display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;"
            "font:13px IBM Plex Sans;color:#3a4c66;margin:2px 0 10px'>"
            f"<span><span style='display:inline-block;width:9px;height:9px;border-radius:50%;"
            f"background:{dot};margin-right:7px'></span><b style='color:#0b1f3a'>"
            f"{T('Captura', 'Capture')} {label.strip()[:-3]} ({T('hora de planta', 'plant time')})</b>"
            f" · {T(_age_txt(age).replace(' ago', '').join(['hace ', '']), _age_txt(age))}</span>"
            f"<span>{len(rec.channels)} {T('canales', 'channels')} · "
            f"{len([b for b in rec.bearings if b.has_xy])}"
            f" {T('cojinetes', 'bearings')}{f' · {rpm:,.0f} rpm' if rpm else ''}</span>"
            f"<span style='color:#8090a6'>{T('Procesado por Rotodinámica · se actualiza solo', 'Processed by Rotordynamics · refreshes automatically')}"
            "</span></div>", unsafe_allow_html=True)
        if stale:
            st.caption(f"No newer capture in the last {_age_txt(age).replace(' ago', '')} — "
                       "showing the latest one available.")
        try:
            view = st.segmented_control(T("Vista", "View"), _VIEWS, default=_VIEWS[0], key=f"{k}_view",
                                        format_func=lambda v: T(_VIEW_ES[v], v),
                                        label_visibility="collapsed") or _VIEWS[0]
        except Exception:  # noqa: BLE001
            view = _VIEWS[0]
        try:
            if view == _VIEWS[0]:
                _view_waveform(rec, k)
            elif view == _VIEWS[1]:
                _view_spectrum(rec, k)
            else:
                _view_orbit(rec, k)
        except Exception as e:  # noqa: BLE001
            import logging
            logging.warning("rd mirror view failed: %s", e)
            st.info(T("Esta vista no está disponible para la última captura.", "This view is not available for the latest capture."))

    _body()


__all__ = ["render_rd_mirror"]
