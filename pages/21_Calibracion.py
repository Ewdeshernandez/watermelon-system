"""
pages/21_Calibracion.py — Módulo Calibración Watermelon
=======================================================

Curvas de linealidad / calibración de sensores de vibración bajo API 670
(5.ª ed.) + manual del fabricante. Motor de cálculo: core.calibration (puro,
validado contra el calibrador portátil Bently/GE y contra la norma). Esta
página es solo UI + wiring; NO contiene la matemática.

Pestañas
--------
1. Proximidad    — linealidad estática gap→voltaje (ISF ±5 % · DSL ±1 mil).
2. Acelerómetro  — linealidad de amplitud (1 %) y respuesta en frecuencia (±3 dB).
3. Velomitor     — linealidad de amplitud y respuesta en frecuencia.
4. Reporte       — resumen + PDF con 1 certificado independiente por lazo (1-50).

Entrada de datos: MANUAL (tabla editable) o AUTOMÁTICA desde un shaker (en
desarrollo — dejará los pares medidos directo en el módulo).

Acceso: analista / admin (gateada; el cliente no la ve).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from core.auth import (
    require_login, render_user_menu, get_current_user, is_page_allowed_for_role,
)
from core.ui_theme import apply_watermelon_page_style
from core.calibration import (
    analyze_proximity_linearity, analyze_amplitude_linearity,
    analyze_frequency_response, get_default_spec, MANUFACTURERS,
)
from core.calibration.curve import linearity_curve_svg
from core.calibration.ui import (
    cal_hero_card, cal_section_header, cal_kpi_row, cal_status_banner,
    cal_footer_norms,
)


# =====================================================================
# Setup + auth
# =====================================================================
st.set_page_config(
    page_title="Watermelon System | Calibración",
    page_icon="📐",
    layout="wide",
)

require_login()
render_user_menu()
apply_watermelon_page_style()

_user = get_current_user() or {}
_role = str(_user.get("role", "")).lower()
if not is_page_allowed_for_role("pages/21_Calibracion.py", _role):
    st.error("Tu rol no tiene acceso a este módulo.")
    st.stop()

st.session_state.setdefault("cal_loops", [])


# =====================================================================
# Hero
# =====================================================================
def _hero() -> None:
    n = len(st.session_state.get("cal_loops", []))
    cal_hero_card(
        asset_name=st.session_state.get("cal_asset") or "(activo sin especificar)",
        client=st.session_state.get("cal_client", ""),
        site=st.session_state.get("cal_location", ""),
        mode=(f"{n} LAZO" + ("S" if n != 1 else "")) if n else "—",
    )


_hero()


# =====================================================================
# Helpers
# =====================================================================
def _parse_xy(df: pd.DataFrame, xcol: str, ycol: str) -> Tuple[List[float], List[float]]:
    """Extrae pares (x, y) numéricos válidos, descartando filas incompletas."""
    xs: List[float] = []
    ys: List[float] = []
    for _, row in df.iterrows():
        try:
            xf = float(row[xcol]); yf = float(row[ycol])
        except (TypeError, ValueError, KeyError):
            continue
        if xf != xf or yf != yf:  # NaN
            continue
        xs.append(xf); ys.append(yf)
    return xs, ys


def _upsert_loop(loop: Dict[str, Any]) -> None:
    """Agrega o actualiza un lazo en el reporte (clave = tag + tipo)."""
    loops = st.session_state["cal_loops"]
    key = (loop["tag"], loop["sensor_type"])
    for i, lp in enumerate(loops):
        if (lp["tag"], lp["sensor_type"]) == key:
            loops[i] = loop
            return
    loops.append(loop)


def _default_df(cols: Dict[str, List[float]]) -> pd.DataFrame:
    return pd.DataFrame(cols)


def _source_selector(prefix: str) -> str:
    src = st.radio("Fuente de datos", ["Manual", "Automático (shaker)"],
                   horizontal=True, key=f"{prefix}_source")
    if src.startswith("Auto"):
        cal_status_banner(
            "Importación automática (shaker) — en desarrollo",
            "Próximamente el shaker de referencia enviará los pares medidos "
            "directo al módulo. Por ahora, ingresa los datos en la tabla.",
            "info")
    return src


# =====================================================================
# PROXIMIDAD
# =====================================================================
def _proximity_tab() -> None:
    cal_section_header(
        "Linealidad estática de proximidad",
        "Gap → voltaje · best-fit vs 200 mV/mil · ISF ±5 % · DSL ±1 mil",
        "API 670 · Tabla 1 / Fig. 4")

    c1, c2, c3 = st.columns(3)
    tag = c1.text_input("Tag / punto", key="px_tag", placeholder="1YD")
    manuf = c2.selectbox("Fabricante", MANUFACTURERS, key="px_manuf")
    xunit_disp = c3.selectbox("Unidad de gap", ["mil", "µm"], key="px_xunit")
    c4, c5, c6 = st.columns(3)
    model = c4.text_input("Modelo", key="px_model", placeholder="3300 XL 8 mm")
    serial = c5.text_input("N.º de serie", key="px_serial")
    idn = c6.text_input("ID / lazo", key="px_id")

    _source_selector("px")

    xunit = "um" if xunit_disp == "µm" else "mil"
    spec = get_default_spec("proximity", manuf, xunit)
    gcol = f"Gap [{xunit_disp}]"
    ocol = "Salida [V]"

    if st.session_state.get("px_grid_unit") != xunit:
        # (re)inicializa la grilla al cambiar de unidad
        st.session_state["px_df"] = _default_df(
            {gcol: spec["grid"], ocol: [None] * len(spec["grid"])})
        st.session_state["px_grid_unit"] = xunit

    st.caption("Ingresa la salida del oscilador-demodulador (V) para cada gap. "
               "Incremento típico API 670: 10 mil / 250 µm.")
    edited = st.data_editor(
        st.session_state.get("px_df", _default_df(
            {gcol: spec["grid"], ocol: [None] * len(spec["grid"])})),
        num_rows="dynamic", use_container_width=True, key="px_editor",
        column_config={
            gcol: st.column_config.NumberColumn(format="%.1f"),
            ocol: st.column_config.NumberColumn(format="%.3f"),
        })

    xs, ys = _parse_xy(edited, gcol, ocol)
    if len(xs) < 2:
        st.info("Ingresa al menos 2 puntos (gap, salida) para ver la curva.")
        return

    try:
        a = analyze_proximity_linearity(
            xs, ys, x_unit=xunit,
            nominal_mv_per_x=spec["nominal_mv_per_x"],
            isf_tol_pct=spec["isf_tol_pct"], dsl_tol_x=spec["dsl_tol_x"],
            min_range_x=spec["min_range_x"])
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo analizar: {e}")
        return

    cal_kpi_row([
        (f"{a['asf_mv_per_x']:.1f}", f"ASF mV/{xunit}",
         f"nominal {spec['nominal_label']}", "cyan"),
        (f"{a['max_isf_err_pct']:.2f} %", "Máx. ISF",
         f"límite ±{spec['isf_tol_pct']:.0f} %", "green" if a["pass_isf"] else "red"),
        (f"{a['max_dsl_x']:.3f}", f"Máx. DSL {xunit}",
         f"límite {spec['dsl_tol_label']}", "green" if a["pass_dsl"] else "red"),
        (a["verdict"], "Veredicto API 670",
         f"rango {a['span_x']:.0f}/{spec['min_range_x']:.0f} {xunit}",
         "green" if a["pass"] else "red"),
    ])

    st.markdown(linearity_curve_svg(
        a["x"], a["y"], a["y_fit"],
        title=f"{tag or 'Lazo'} — Linealidad de proximidad",
        x_label=f"Gap [{xunit_disp}]", y_label="Salida [V]",
        verdict=a["verdict"],
        badge_detail=f"ISF {a['max_isf_err_pct']:.1f}% · DSL {a['max_dsl_x']:.2f} {xunit}"),
        unsafe_allow_html=True)

    _add_button("proximity", "linearity", tag, manuf, model, serial, idn,
                spec["norm"], a)


# =====================================================================
# ACELERÓMETRO / VELOMITOR (amplitud + frecuencia)
# =====================================================================
def _seismic_tab(sensor_type: str, prefix: str) -> None:
    spec = get_default_spec(sensor_type)
    label = "Acelerómetro" if sensor_type == "accelerometer" else "Velomitor"

    c1, c2, c3 = st.columns(3)
    tag = c1.text_input("Tag / punto", key=f"{prefix}_tag")
    manuf = c2.selectbox("Fabricante", MANUFACTURERS, key=f"{prefix}_manuf")
    ensayo = c3.selectbox("Ensayo", ["Linealidad de amplitud", "Respuesta en frecuencia"],
                          key=f"{prefix}_kind")
    c4, c5, c6 = st.columns(3)
    model = c4.text_input("Modelo", key=f"{prefix}_model")
    serial = c5.text_input("N.º de serie", key=f"{prefix}_serial")
    idn = c6.text_input("ID / lazo", key=f"{prefix}_id")

    _source_selector(prefix)

    if ensayo.startswith("Linealidad"):
        _amplitude_section(sensor_type, prefix, spec, tag, manuf, model, serial, idn, label)
    else:
        _frequency_section(sensor_type, prefix, spec, tag, manuf, model, serial, idn, label)


def _amplitude_section(sensor_type, prefix, spec, tag, manuf, model, serial, idn, label):
    cal_section_header(
        f"Linealidad de amplitud — {label}",
        f"Nivel de excitación → salida · best-fit · nominal {spec['nominal_sensitivity']:g} "
        f"{spec['sensitivity_unit']}",
        "API 670 · Tabla 1")
    lu, ou = spec["level_unit"], spec["output_unit"]
    lcol, ocol = f"Nivel [{lu}]", f"Salida [{ou}]"
    cc = st.columns(3)
    nominal = cc[0].number_input(f"Sensibilidad nominal [{spec['sensitivity_unit']}]",
                                 value=float(spec["nominal_sensitivity"]),
                                 key=f"{prefix}_amp_nom", step=1.0, format="%.3f")
    tol = cc[1].number_input("Tolerancia amplitud [%FS]", value=float(spec["ampl_tol_pct"]),
                             key=f"{prefix}_amp_tol", step=0.5, format="%.1f")

    dkey = f"{prefix}_amp_df"
    if dkey not in st.session_state:
        st.session_state[dkey] = _default_df(
            {lcol: spec["levels"], ocol: [None] * len(spec["levels"])})
    edited = st.data_editor(st.session_state[dkey], num_rows="dynamic",
                            use_container_width=True, key=f"{prefix}_amp_editor")
    xs, ys = _parse_xy(edited, lcol, ocol)
    if len(xs) < 2:
        st.info("Ingresa al menos 2 niveles (nivel, salida) para ver la curva.")
        return
    try:
        a = analyze_amplitude_linearity(xs, ys, nominal_sensitivity=nominal,
                                        tol_pct=tol, level_unit=lu, output_unit=ou)
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo analizar: {e}")
        return
    cal_kpi_row([
        (f"{a['sensitivity']:.3f}", spec["sensitivity_unit"], "best-fit", "cyan"),
        (f"{a['sens_err_pct']:.2f} %", "Error vs nominal",
         f"nominal {nominal:g}", "navy"),
        (f"{a['max_dev_pct_fs']:.3f} %", "Máx. desv. amplitud",
         f"límite ±{tol:.1f} %FS", "green" if a["pass"] else "red"),
        (a["verdict"], "Veredicto", "API 670", "green" if a["pass"] else "red"),
    ])
    st.markdown(linearity_curve_svg(
        a["x"], a["y"], a["y_fit"], title=f"{tag or label} — Linealidad de amplitud",
        x_label=f"Nivel [{lu}]", y_label=f"Salida [{ou}]", verdict=a["verdict"],
        badge_detail=f"{a['max_dev_pct_fs']:.2f} %FS"), unsafe_allow_html=True)
    _add_button(sensor_type, "amplitude", tag, manuf, model, serial, idn,
                spec["norm"], a)


def _frequency_section(sensor_type, prefix, spec, tag, manuf, model, serial, idn, label):
    band = spec["freq_band_hz"]
    cal_section_header(
        f"Respuesta en frecuencia — {label}",
        f"Sensibilidad vs frecuencia · desviación dB · banda {band[0]:g}–{band[1]:g} Hz",
        "API 670 · Tabla 1")
    su = spec["sensitivity_unit"]
    fcol, scol = "Frecuencia [Hz]", f"Sensibilidad [{su}]"
    cc = st.columns(3)
    ref = cc[0].number_input("Frecuencia de referencia [Hz]",
                             value=float(spec["freq_ref_hz"]),
                             key=f"{prefix}_fr_ref", step=1.0, format="%.0f")
    tol = cc[1].number_input("Tolerancia [dB]", value=float(spec["freq_tol_db"]),
                             key=f"{prefix}_fr_tol", step=0.5, format="%.1f")

    dkey = f"{prefix}_fr_df"
    if dkey not in st.session_state:
        st.session_state[dkey] = _default_df(
            {fcol: spec["freq_points"], scol: [None] * len(spec["freq_points"])})
    edited = st.data_editor(st.session_state[dkey], num_rows="dynamic",
                            use_container_width=True, key=f"{prefix}_fr_editor")
    xs, ys = _parse_xy(edited, fcol, scol)
    if len(xs) < 2:
        st.info("Ingresa al menos 2 frecuencias (frecuencia, sensibilidad).")
        return
    try:
        a = analyze_frequency_response(xs, ys, ref_freq_hz=ref, tol_db=tol,
                                       band_hz=band, sens_unit=su)
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo analizar: {e}")
        return
    cal_kpi_row([
        (f"{a['ref_sensitivity']:.2f}", su, f"@ {a['ref_freq_hz']:.0f} Hz", "cyan"),
        (f"{a['max_dev_db']:.2f} dB", "Máx. desviación",
         f"límite ±{tol:.1f} dB", "green" if a["pass"] else "red"),
        (a["verdict"], "Veredicto", f"banda {band[0]:g}–{band[1]:g} Hz",
         "green" if a["pass"] else "red"),
    ])
    st.markdown(linearity_curve_svg(
        a["x"], a["y"], None,
        title=f"{tag or label} — Respuesta en frecuencia",
        x_label="Frecuencia [Hz]", y_label=f"Sensibilidad [{su}]",
        verdict=a["verdict"], badge_detail=f"{a['max_dev_db']:.2f} dB"),
        unsafe_allow_html=True)
    _add_button(sensor_type, "frequency", tag, manuf, model, serial, idn,
                spec["norm"], a)


# =====================================================================
# Botón agregar al reporte
# =====================================================================
def _add_button(sensor_type, kind, tag, manuf, model, serial, idn, norm, analysis):
    if not tag:
        st.caption("Asigna un **tag** para poder agregar el lazo al reporte.")
        return
    if st.button(f"➕ Agregar / actualizar «{tag}» en el reporte",
                 key=f"add_{sensor_type}_{kind}", type="primary"):
        _upsert_loop({
            "tag": tag, "sensor_type": sensor_type, "kind": kind,
            "manufacturer": manuf, "model": model, "serial": serial,
            "id_number": idn, "norm": norm, "analysis": analysis,
        })
        st.success(f"«{tag}» agregado al reporte "
                   f"({len(st.session_state['cal_loops'])} lazo(s)).")


# =====================================================================
# REPORTE
# =====================================================================
def _report_tab() -> None:
    from datetime import date as _date
    cal_section_header("Reporte de calibración",
                       "Portada + 1 certificado independiente por lazo (1 a 50)",
                       "WM-CAL · API 670")

    st.session_state.setdefault("cal_asset", "")
    st.session_state.setdefault("cal_client", "")
    st.session_state.setdefault("cal_location", "")
    st.session_state.setdefault("cal_specialist", _user.get("full_name") or "")
    st.session_state.setdefault("cal_date", _date.today().strftime("%d/%m/%Y"))

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_input("Activo", key="cal_asset")
            st.text_input("Cliente", key="cal_client")
        with c2:
            st.text_input("Sitio / ubicación", key="cal_location")
            st.text_input("Especialista", key="cal_specialist")
        with c3:
            st.text_input("Fecha", key="cal_date")
            st.text_area("Notas", key="cal_notes", height=80)

    loops = st.session_state.get("cal_loops", [])
    if not loops:
        st.info("Aún no hay lazos. Agrégalos desde las pestañas de Proximidad, "
                "Acelerómetro o Velomitor.")
        return

    st.markdown(f"**{len(loops)} lazo(s) en el reporte:**")
    for i, lp in enumerate(loops):
        a = lp.get("analysis", {})
        cc = st.columns([3, 2, 2, 2, 1])
        cc[0].markdown(f"**{lp['tag']}** · {lp.get('model') or '—'}")
        cc[1].markdown(lp["sensor_type"].capitalize())
        cc[2].markdown(lp.get("manufacturer", "—"))
        verdict = a.get("verdict", "—")
        color = "#16a34a" if verdict == "PASA" else "#dc2626"
        cc[3].markdown(f"<span style='color:{color};font-weight:700'>{verdict}</span>",
                       unsafe_allow_html=True)
        if cc[4].button("🗑", key=f"del_{i}", help="Quitar del reporte"):
            loops.pop(i)
            st.rerun()

    st.divider()
    if st.button("Generar reporte PDF", key="cal_gen_btn", type="primary"):
        try:
            from core.calibration.report import build_calibration_pdf
            meta = {
                "asset": st.session_state.get("cal_asset"),
                "client": st.session_state.get("cal_client"),
                "location": st.session_state.get("cal_location"),
                "specialist": st.session_state.get("cal_specialist"),
                "report_date": st.session_state.get("cal_date"),
                "notes": st.session_state.get("cal_notes"),
            }
            st.session_state["cal_pdf"] = build_calibration_pdf(meta=meta, loops=loops)
        except Exception as e:  # noqa: BLE001
            st.error(f"Error generando el PDF: {e}")
            st.session_state.pop("cal_pdf", None)
    if st.session_state.get("cal_pdf"):
        import re as _re
        fn = "Calibracion_" + _re.sub(r"[^A-Za-z0-9]+", "_",
                                      (st.session_state.get("cal_asset") or "sensores")
                                      ).strip("_") + ".pdf"
        st.download_button("⬇ Descargar PDF", data=st.session_state["cal_pdf"],
                           file_name=fn, mime="application/pdf")


# =====================================================================
# Tabs
# =====================================================================
tab_px, tab_ac, tab_ve, tab_rep = st.tabs(
    ["🔵  Proximidad", "🟢  Acelerómetro", "🟡  Velomitor", "📄  Reporte"])

with tab_px:
    _proximity_tab()
with tab_ac:
    _seismic_tab("accelerometer", "ac")
with tab_ve:
    _seismic_tab("velomitor", "ve")
with tab_rep:
    _report_tab()

cal_footer_norms()
