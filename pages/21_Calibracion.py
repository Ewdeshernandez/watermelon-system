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
    page_title="Watermelon System | Calibration",
    page_icon="📐",
    layout="wide",
)

require_login()
render_user_menu()
apply_watermelon_page_style()

_user = get_current_user() or {}
_role = str(_user.get("role", "")).lower()
if not is_page_allowed_for_role("pages/21_Calibracion.py", _role):
    st.error("Your role does not have access to this module.")
    st.stop()

st.session_state.setdefault("cal_loops", [])

# =====================================================================
# Autoguardado + recuperación (no perder el reporte si se cae la sesión)
# =====================================================================
from core.reports_ext import drafts as _drafts
from core.reports_ext.common import REVIEWERS, peek_consecutive, commit_consecutive

_CAL_MODULE = "calibracion"
# Claves de estado que se persisten (lo valioso: lazos + metadatos + textos de
# las secciones descriptivas). Las fotos no se persisten (se re-suben).
_CAL_KEYS = ["cal_loops", "cal_asset", "cal_client", "cal_location",
             "cal_specialist", "cal_date", "cal_notes", "cal_hall", "cal_reco",
             "cal_cons", "cal_rev",
             "cal_met_txt", "cal_inf_txt", "cal_dev_txt", "cal_lin_txt", "cal_sat_txt"]


def _cal_capture_state() -> dict:
    return {k: st.session_state.get(k) for k in _CAL_KEYS}


# 1) Aplicar una restauración PENDIENTE (de cargar/duplicar borrador) — se hace
#    ANTES de instanciar cualquier widget para no chocar con Streamlit.
_pending = st.session_state.pop("_cal_pending_restore", None)
if _pending is not None:
    for _k in _CAL_KEYS:
        st.session_state[_k] = _pending.get(_k)
    if not isinstance(st.session_state.get("cal_loops"), list):
        st.session_state["cal_loops"] = _pending.get("cal_loops") or []

# 2) Recuperación AUTOMÁTICA tras caída/reconexión/redeploy: si la sesión está
#    vacía pero el autoguardado tiene CUALQUIER contenido (lazos O metadatos del
#    reporte), se restaura — igual que el reporte del sistema.
if not st.session_state.get("_cal_recovery_checked"):
    st.session_state["_cal_recovery_checked"] = True
    _session_empty = (not st.session_state.get("cal_loops")
                      and not (st.session_state.get("cal_asset") or "").strip())
    if _session_empty:
        _auto = _drafts.load_autosave(_CAL_MODULE)
        _has_content = bool(_auto) and (
            bool(_auto.get("cal_loops"))
            or any(str(_auto.get(_k) or "").strip()
                   for _k in ("cal_asset", "cal_client", "cal_location",
                              "cal_specialist", "cal_notes", "cal_hall", "cal_reco")))
        if _has_content:
            for _k in _CAL_KEYS:
                st.session_state[_k] = _auto.get(_k)
            if not isinstance(st.session_state.get("cal_loops"), list):
                st.session_state["cal_loops"] = _auto.get("cal_loops") or []
            st.session_state["_cal_recovered"] = True


# =====================================================================
# Hero
# =====================================================================
def _hero() -> None:
    n = len(st.session_state.get("cal_loops", []))
    cal_hero_card(
        asset_name=st.session_state.get("cal_asset") or "(unspecified asset)",
        client=st.session_state.get("cal_client", ""),
        site=st.session_state.get("cal_location", ""),
        mode=(f"{n} LOOP" + ("S" if n != 1 else "")) if n else "—",
    )


_hero()

if st.session_state.pop("_cal_recovered", None):
    st.info("♻ Your last automatic draft was recovered (loops and report "
            "data). If you had photos, upload them again.")

_cal_ts = st.session_state.get("_cal_autosave_ts")
st.caption("🟢 Autosave active — the report is recovered if the session drops"
           + (f" · last saved: {_cal_ts}" if _cal_ts else "."))


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
    src = st.radio("Data source", ["Manual", "Automatic (shaker)"],
                   horizontal=True, key=f"{prefix}_source")
    if src.startswith("Auto"):
        cal_status_banner(
            "Automatic import (shaker) — in development",
            "Soon the reference shaker will send the measured pairs straight "
            "into the module. For now, enter the data in the table.",
            "info")
    return src


# =====================================================================
# PROXIMIDAD
# =====================================================================
def _proximity_tab() -> None:
    cal_section_header(
        "Proximity static linearity",
        "Gap → voltage · best-fit vs 200 mV/mil · ISF ±5 % · DSL ±1 mil",
        "API 670 · Table 1 / Fig. 4")

    c1, c2, c3 = st.columns(3)
    tag = c1.text_input("Tag / point", key="px_tag", placeholder="1YD")
    manuf = c2.selectbox("Manufacturer", MANUFACTURERS, key="px_manuf")
    xunit_disp = c3.selectbox("Gap unit", ["mil", "µm"], key="px_xunit")
    c4, c5, c6 = st.columns(3)
    model = c4.text_input("Model", key="px_model", placeholder="3300 XL 8 mm")
    serial = c5.text_input("Serial number", key="px_serial")
    idn = c6.text_input("ID / loop", key="px_id")

    _source_selector("px")

    xunit = "um" if xunit_disp == "µm" else "mil"
    spec = get_default_spec("proximity", manuf, xunit)
    gcol = f"Gap [{xunit_disp}]"
    ocol = "Output [V]"

    if st.session_state.get("px_grid_unit") != xunit:
        # (re)inicializa la grilla al cambiar de unidad
        st.session_state["px_df"] = _default_df(
            {gcol: spec["grid"], ocol: [None] * len(spec["grid"])})
        st.session_state["px_grid_unit"] = xunit

    st.caption("Enter the oscillator-demodulator output (V) for each gap. "
               "Typical API 670 increment: 10 mil / 250 µm.")
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
        st.info("Enter at least 2 points (gap, output) to see the curve.")
        return

    try:
        a = analyze_proximity_linearity(
            xs, ys, x_unit=xunit,
            nominal_mv_per_x=spec["nominal_mv_per_x"],
            isf_tol_pct=spec["isf_tol_pct"], dsl_tol_x=spec["dsl_tol_x"],
            min_range_x=spec["min_range_x"])
    except Exception as e:  # noqa: BLE001
        st.error(f"Analysis failed: {e}")
        return

    cal_kpi_row([
        (f"{a['asf_mv_per_x']:.1f}", f"ASF mV/{xunit}",
         f"nominal {spec['nominal_label']}", "cyan"),
        (f"{a['max_isf_err_pct']:.2f} %", "Max. ISF",
         f"limit ±{spec['isf_tol_pct']:.0f} %", "green" if a["pass_isf"] else "red"),
        (f"{a['max_dsl_x']:.3f}", f"Max. DSL {xunit}",
         f"limit {spec['dsl_tol_label']}", "green" if a["pass_dsl"] else "red"),
        (a["verdict"], "API 670 verdict",
         f"range {a['span_x']:.0f}/{spec['min_range_x']:.0f} {xunit}",
         "green" if a["pass"] else "red"),
    ])

    st.markdown(linearity_curve_svg(
        a["x"], a["y"], a["y_fit"],
        title=f"{tag or 'Loop'} — Proximity linearity",
        x_label=f"Gap [{xunit_disp}]", y_label="Output [V]",
        verdict=a["verdict"],
        badge_detail=f"ISF {a['max_isf_err_pct']:.1f}% · DSL {a['max_dsl_x']:.2f} {xunit}"),
        unsafe_allow_html=True)

    lw = a.get("linear_window")
    if lw:
        sev = "ok" if lw.get("meets_min_range") else "warning"
        cal_status_banner(
            f"Usable linear range (calibratable): {lw['start_x']:.0f}–{lw['end_x']:.0f} {xunit} "
            f"({-abs(lw['start_v']):.2f} to {-abs(lw['end_v']):.2f} Vdc)",
            f"Recommended setpoint ~{lw['center_x']:.0f} {xunit} ({-abs(lw['center_v']):.2f} Vdc) · "
            f"span {lw['span_x']:.0f} {xunit} "
            + ("meets the API 670 minimum of 80 mil."
               if lw.get("meets_min_range")
               else "below the API 670 minimum of 80 mil (usable but out of spec)."),
            sev)
    else:
        cal_status_banner("No linear window meets ISF±5% and DSL±1 mil",
                          "No segment of the curve meets both criteria.", "fail")

    _add_button("proximity", "linearity", tag, manuf, model, serial, idn,
                spec["norm"], a)


# =====================================================================
# ACELERÓMETRO / VELOMITOR (amplitud + frecuencia)
# =====================================================================
def _seismic_tab(sensor_type: str, prefix: str) -> None:
    spec = get_default_spec(sensor_type)
    label = "Accelerometer" if sensor_type == "accelerometer" else "Velomitor"

    c1, c2, c3 = st.columns(3)
    tag = c1.text_input("Tag / point", key=f"{prefix}_tag")
    manuf = c2.selectbox("Manufacturer", MANUFACTURERS, key=f"{prefix}_manuf")
    ensayo = c3.selectbox("Test", ["Amplitude linearity", "Frequency response"],
                          key=f"{prefix}_kind")
    c4, c5, c6 = st.columns(3)
    model = c4.text_input("Model", key=f"{prefix}_model")
    serial = c5.text_input("Serial number", key=f"{prefix}_serial")
    idn = c6.text_input("ID / loop", key=f"{prefix}_id")

    _source_selector(prefix)

    if ensayo.startswith("Amplitude"):
        _amplitude_section(sensor_type, prefix, spec, tag, manuf, model, serial, idn, label)
    else:
        _frequency_section(sensor_type, prefix, spec, tag, manuf, model, serial, idn, label)


def _amplitude_section(sensor_type, prefix, spec, tag, manuf, model, serial, idn, label):
    cal_section_header(
        f"Amplitude linearity — {label}",
        f"Excitation level → output · best-fit · nominal {spec['nominal_sensitivity']:g} "
        f"{spec['sensitivity_unit']}",
        "API 670 · Table 1")
    lu, ou = spec["level_unit"], spec["output_unit"]
    lcol, ocol = f"Level [{lu}]", f"Output [{ou}]"
    cc = st.columns(3)
    nominal = cc[0].number_input(f"Nominal sensitivity [{spec['sensitivity_unit']}]",
                                 value=float(spec["nominal_sensitivity"]),
                                 key=f"{prefix}_amp_nom", step=1.0, format="%.3f")
    tol = cc[1].number_input("Amplitude tolerance [%FS]", value=float(spec["ampl_tol_pct"]),
                             key=f"{prefix}_amp_tol", step=0.5, format="%.1f")

    dkey = f"{prefix}_amp_df"
    if dkey not in st.session_state:
        st.session_state[dkey] = _default_df(
            {lcol: spec["levels"], ocol: [None] * len(spec["levels"])})
    edited = st.data_editor(st.session_state[dkey], num_rows="dynamic",
                            use_container_width=True, key=f"{prefix}_amp_editor")
    xs, ys = _parse_xy(edited, lcol, ocol)
    if len(xs) < 2:
        st.info("Enter at least 2 levels (level, output) to see the curve.")
        return
    try:
        a = analyze_amplitude_linearity(xs, ys, nominal_sensitivity=nominal,
                                        tol_pct=tol, level_unit=lu, output_unit=ou)
    except Exception as e:  # noqa: BLE001
        st.error(f"Analysis failed: {e}")
        return
    cal_kpi_row([
        (f"{a['sensitivity']:.3f}", spec["sensitivity_unit"], "best-fit", "cyan"),
        (f"{a['sens_err_pct']:.2f} %", "Error vs nominal",
         f"nominal {nominal:g}", "navy"),
        (f"{a['max_dev_pct_fs']:.3f} %", "Max. amplitude dev.",
         f"limit ±{tol:.1f} %FS", "green" if a["pass"] else "red"),
        (a["verdict"], "Verdict", "API 670", "green" if a["pass"] else "red"),
    ])
    st.markdown(linearity_curve_svg(
        a["x"], a["y"], a["y_fit"], title=f"{tag or label} — Amplitude linearity",
        x_label=f"Level [{lu}]", y_label=f"Output [{ou}]", verdict=a["verdict"],
        badge_detail=f"{a['max_dev_pct_fs']:.2f} %FS"), unsafe_allow_html=True)
    _add_button(sensor_type, "amplitude", tag, manuf, model, serial, idn,
                spec["norm"], a)


def _frequency_section(sensor_type, prefix, spec, tag, manuf, model, serial, idn, label):
    band = spec["freq_band_hz"]
    cal_section_header(
        f"Frequency response — {label}",
        f"Sensitivity vs frequency · dB deviation · band {band[0]:g}–{band[1]:g} Hz",
        "API 670 · Table 1")
    su = spec["sensitivity_unit"]
    fcol, scol = "Frequency [Hz]", f"Sensitivity [{su}]"
    cc = st.columns(3)
    ref = cc[0].number_input("Reference frequency [Hz]",
                             value=float(spec["freq_ref_hz"]),
                             key=f"{prefix}_fr_ref", step=1.0, format="%.0f")
    tol = cc[1].number_input("Tolerance [dB]", value=float(spec["freq_tol_db"]),
                             key=f"{prefix}_fr_tol", step=0.5, format="%.1f")

    dkey = f"{prefix}_fr_df"
    if dkey not in st.session_state:
        st.session_state[dkey] = _default_df(
            {fcol: spec["freq_points"], scol: [None] * len(spec["freq_points"])})
    edited = st.data_editor(st.session_state[dkey], num_rows="dynamic",
                            use_container_width=True, key=f"{prefix}_fr_editor")
    xs, ys = _parse_xy(edited, fcol, scol)
    if len(xs) < 2:
        st.info("Enter at least 2 frequencies (frequency, sensitivity).")
        return
    try:
        a = analyze_frequency_response(xs, ys, ref_freq_hz=ref, tol_db=tol,
                                       band_hz=band, sens_unit=su)
    except Exception as e:  # noqa: BLE001
        st.error(f"Analysis failed: {e}")
        return
    cal_kpi_row([
        (f"{a['ref_sensitivity']:.2f}", su, f"@ {a['ref_freq_hz']:.0f} Hz", "cyan"),
        (f"{a['max_dev_db']:.2f} dB", "Max. deviation",
         f"limit ±{tol:.1f} dB", "green" if a["pass"] else "red"),
        (a["verdict"], "Verdict", f"band {band[0]:g}–{band[1]:g} Hz",
         "green" if a["pass"] else "red"),
    ])
    st.markdown(linearity_curve_svg(
        a["x"], a["y"], None,
        title=f"{tag or label} — Frequency response",
        x_label="Frequency [Hz]", y_label=f"Sensitivity [{su}]",
        verdict=a["verdict"], badge_detail=f"{a['max_dev_db']:.2f} dB"),
        unsafe_allow_html=True)
    _add_button(sensor_type, "frequency", tag, manuf, model, serial, idn,
                spec["norm"], a)


# =====================================================================
# Botón agregar al reporte
# =====================================================================
def _add_button(sensor_type, kind, tag, manuf, model, serial, idn, norm, analysis):
    if not tag:
        st.caption("Assign a **tag** to add the loop to the report.")
        return
    if st.button(f"➕ Add / update «{tag}» in the report",
                 key=f"add_{sensor_type}_{kind}", type="primary"):
        _upsert_loop({
            "tag": tag, "sensor_type": sensor_type, "kind": kind,
            "manufacturer": manuf, "model": model, "serial": serial,
            "id_number": idn, "norm": norm, "analysis": analysis,
        })
        st.success(f"«{tag}» added to the report "
                   f"({len(st.session_state['cal_loops'])} loop(s)).")


# =====================================================================
# REPORTE
# =====================================================================
def _report_tab() -> None:
    from datetime import date as _date
    cal_section_header("Calibration report",
                       "Cover page + 1 independent certificate per loop (1 to 50)",
                       "WM-CAL · API 670")

    st.session_state.setdefault("cal_asset", "")
    st.session_state.setdefault("cal_client", "")
    st.session_state.setdefault("cal_location", "")
    st.session_state.setdefault("cal_specialist", _user.get("full_name") or "")
    st.session_state.setdefault("cal_date", _date.today().strftime("%d/%m/%Y"))
    st.session_state.setdefault("cal_cons", peek_consecutive("calibracion"))

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_input("Asset", key="cal_asset")
            st.text_input("Client", key="cal_client")
        with c2:
            st.text_input("Site / location", key="cal_location")
            st.text_input("Specialist (prepared by)", key="cal_specialist")
        with c3:
            st.text_input("Date", key="cal_date")
            st.text_area("Notes", key="cal_notes", height=80)
        c4, c5 = st.columns(2)
        with c4:
            st.text_input("Sequential number (automatic · ISO 9001)", key="cal_cons",
                          help="Automatic, editable SIGAGROUP-CAL-YEAR-NNNN code.")
        with c5:
            st.selectbox("Approved by (authority)", list(REVIEWERS.keys()), key="cal_rev")

    # ---- Borradores (autoguardado + guardar/cargar/duplicar/nuevo) --------
    with st.expander("💾 Report drafts", expanded=False):
        st.caption("The report autosaves on its own; if the session drops or "
                   "there is a redeploy, it is recovered when you return. Here "
                   "you can save named versions.")
        d1, d2 = st.columns([3, 1])
        _dname = d1.text_input("Draft name", key="cal_draft_name",
                               placeholder="e.g. Unit 2 — proximity calibration")
        if d2.button("💾 Save draft", key="cal_draft_save",
                     use_container_width=True):
            _nm = (_dname or "").strip() or "draft"
            if _drafts.save_draft(_CAL_MODULE, _nm, _cal_capture_state()):
                st.success(f"Draft «{_nm}» saved.")
            else:
                st.error("Could not save (server disk full?).")
        _existing = _drafts.list_drafts(_CAL_MODULE)
        e1, e2, e3, e4 = st.columns([3, 1, 1, 1])
        _sel = e1.selectbox("Existing drafts", ["—"] + _existing,
                            key="cal_draft_pick")
        _has = _sel != "—"
        if e2.button("Load", key="cal_draft_load", use_container_width=True,
                     disabled=not _has):
            _stt = _drafts.load_draft(_CAL_MODULE, _sel)
            if _stt is not None:
                st.session_state["_cal_pending_restore"] = _stt
                st.rerun()
        if e3.button("Duplicate", key="cal_draft_dup", use_container_width=True,
                     disabled=not _has):
            _stt = _drafts.load_draft(_CAL_MODULE, _sel)
            if _stt is not None:
                _drafts.save_draft(_CAL_MODULE, f"{_sel} (copy)", _stt)
                st.rerun()
        if e4.button("Delete", key="cal_draft_del", use_container_width=True,
                     disabled=not _has):
            _drafts.delete_draft(_CAL_MODULE, _sel)
            st.rerun()
        if st.button("🆕 New report (clear)", key="cal_draft_new"):
            st.session_state["_cal_pending_restore"] = {}
            st.rerun()

    loops = st.session_state.get("cal_loops", [])
    if not loops:
        st.info("No loops yet. Add them from the Proximity, Accelerometer or "
                "Velomitor tabs.")
        return

    st.markdown(f"**{len(loops)} loop(s) in the report:**")
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
        if cc[4].button("🗑", key=f"del_{i}", help="Remove from report"):
            loops.pop(i)
            st.rerun()

    st.divider()
    # Hallazgos / recomendaciones + fotos relevantes (opcionales)
    def _sentences(text: str):
        """Separa por ORACIÓN: un item continúa hasta un punto seguido de
        espacio. Une saltos de línea y no rompe decimales (0.0007 no lleva
        espacio tras el punto)."""
        import re
        t = re.sub(r"\s+", " ", (text or "").strip())
        if not t:
            return []
        return [s.strip() for s in re.split(r"(?<=\.)\s+", t) if s.strip()]

    def _sec_input(key: str, title: str, only_photos: bool = False) -> dict:
        """Sección con párrafo + imágenes (o solo imágenes). Fotos con título
        por figura. El texto se persiste; las fotos se re-suben."""
        with st.expander(title, expanded=False):
            txt = ""
            if not only_photos:
                txt = st.text_area("Paragraph", key=f"{key}_txt", height=100)
            files = st.file_uploader("Images", accept_multiple_files=True,
                                     type=["png", "jpg", "jpeg"], key=f"{key}_imgs")
            phs = []
            for _i, _f in enumerate(files or [], 1):
                _t = st.text_input(f"Figure {_i}", key=f"{key}_figt_{_i}",
                                   placeholder=f"Description of figure {_i}")
                _cap = f"Figure {_i}. {_t}".rstrip(". ") if _t else f"Figure {_i}"
                phs.append({"bytes": _f.getvalue(), "caption": _cap})
        return {"text": txt, "photos": phs}

    cH, cR = st.columns(2)
    hall = cH.text_area("2. Findings (one item per sentence, ends with a period)",
                        key="cal_hall", height=100)
    reco = cR.text_area("3. Recommendations (one item per sentence, ends with a period)",
                        key="cal_reco", height=100)

    st.markdown("**Descriptive sections** (optional — each accepts a paragraph and images):")
    sec_met = _sec_input("cal_met", "4. Methodology")
    sec_inf = _sec_input("cal_inf", "5. Unit information")
    sec_dev = _sec_input("cal_dev", "6. Service execution")
    sec_lin = _sec_input("cal_lin", "6.1 Linearity test")
    sec_sat = _sec_input("cal_sat", "6.2 SAT tests")
    sec_anx = _sec_input("cal_anx", "7. Photographic appendix (images only)", only_photos=True)

    st.divider()
    if st.button("Generate PDF report", key="cal_gen_btn", type="primary"):
        try:
            from core.calibration.report import build_calibration_pdf
            _cons = st.session_state.get("cal_cons")
            _rev = st.session_state.get("cal_rev")
            meta = {
                "asset": st.session_state.get("cal_asset"),
                "client": st.session_state.get("cal_client"),
                "location": st.session_state.get("cal_location"),
                "specialist": st.session_state.get("cal_specialist"),
                "report_date": st.session_state.get("cal_date"),
                "notes": st.session_state.get("cal_notes"),
                "consecutive": _cons,
                "reviewer": _rev, "reviewer_role": REVIEWERS.get(_rev, ""),
                "hallazgos": _sentences(hall),
                "recomendaciones": _sentences(reco),
                "sec_metodologia": sec_met, "sec_info_unidad": sec_inf,
                "sec_desarrollo": sec_dev, "sec_linealidad": sec_lin,
                "sec_sat": sec_sat, "anexo_photos": sec_anx.get("photos", []),
            }
            st.session_state["cal_pdf"] = build_calibration_pdf(meta=meta, loops=loops)
            # Confirmar el consecutivo (una sola vez) si es el automático.
            _done = st.session_state.setdefault("_cal_cons_committed", set())
            if _cons and str(_cons).startswith("SIGAGROUP-") and _cons not in _done:
                commit_consecutive("calibracion")
                _done.add(_cons)
        except Exception as e:  # noqa: BLE001
            st.error(f"Error generating the PDF: {e}")
            st.session_state.pop("cal_pdf", None)
    if st.session_state.get("cal_pdf"):
        import re as _re
        fn = "Calibracion_" + _re.sub(r"[^A-Za-z0-9]+", "_",
                                      (st.session_state.get("cal_asset") or "sensores")
                                      ).strip("_") + ".pdf"
        st.download_button("⬇ Download PDF", data=st.session_state["cal_pdf"],
                           file_name=fn, mime="application/pdf")


# =====================================================================
# Tabs
# =====================================================================
tab_px, tab_ac, tab_ve, tab_rep = st.tabs(
    ["🔵  Proximity", "🟢  Accelerometer", "🟡  Velomitor", "📄  Report"])

with tab_px:
    _proximity_tab()
with tab_ac:
    _seismic_tab("accelerometer", "ac")
with tab_ve:
    _seismic_tab("velomitor", "ve")
with tab_rep:
    _report_tab()

# Autoguardado al final de cada render (tolerante a disco lleno; no crashea).
try:
    if _drafts.autosave(_CAL_MODULE, _cal_capture_state()):
        import datetime as _dt
        st.session_state["_cal_autosave_ts"] = _dt.datetime.now().strftime("%H:%M:%S")
except Exception:
    pass

cal_footer_norms()
