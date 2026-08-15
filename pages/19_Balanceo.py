"""
pages/19_Balanceo.py — Módulo Balanceo Watermelon
=================================================

Balanceo de rotores en campo por coeficiente de influencia, integrado a
Watermelon. Motor de cálculo: core.balance (extraído de ROTORIX, validado en
campo). Esta página es solo UI + wiring; NO contiene matemática de balanceo.

Pestañas
--------
1. Peso de prueba   — recomendación API 684 (Umax = 6350·W/N).
2. Balanceo 1 plano — coeficiente de influencia (ISO 21940-12).
3. Balanceo 2 planos— matriz de coeficientes de influencia (ISO 21940-12).
4. Validación ISO   — desbalance residual permisible y grados (ISO 21940-11).
5. Reporte          — resumen consolidado de la sesión.

Entrada de datos: MANUAL o importada desde LIVE MONITORING. En el modo Live se
selecciona la máquina y las sondas (planos), y se captura el vector 1X
(magnitud + fase) de cada sonda para precargar los vectores de vibración. La
unidad de selección es la sonda/plano; la sección (turbina/generador/compresor/
motor/bomba) sólo agrupa el picker. En 2 planos se usa la MISMA dirección (X/Y)
en ambos planos por defecto.

Marco normativo: ISO 21940-11 / ISO 21940-12 · API 684.
Acceso: analista / admin (gateada; el cliente no la ve).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import streamlit as st

from core.auth import (
    require_login, render_user_menu, get_current_user, is_page_allowed_for_role,
)
from core.ui_theme import page_header
from core.balance import (
    to_complex, to_polar,
    umax_api684_gmm, recommend_trial_weight_g,
    calc_U_trial, pct_reduction,
    evaluate_iso_grades,
    solve_1plane, solve_2plane,
)


# =====================================================================
# Setup + auth
# =====================================================================
st.set_page_config(
    page_title="Watermelon System | Balanceo",
    page_icon="⚖️",
    layout="wide",
)

require_login()
render_user_menu()

_user = get_current_user() or {}
_role = str(_user.get("role", "")).lower()
if not is_page_allowed_for_role("pages/19_Balanceo.py", _role):
    st.error("Tu rol no tiene acceso a este módulo.")
    st.stop()

page_header(
    title="Balanceo",
    subtitle="Balanceo de rotores por coeficiente de influencia — 1 y 2 planos, "
             "peso de prueba y validación bajo ISO 21940 / API 684.",
)

UNITS = ["µm pk-pk", "mil pk-pk", "mm/s RMS"]


# =====================================================================
# Helpers de UI
# =====================================================================
def _num(key: str, label: str, default: float = 0.0, **kw) -> float:
    """number_input persistente por key (sin warning value+key). Permite que
    el import Live pre-cargue el valor vía st.session_state[key]."""
    st.session_state.setdefault(key, float(default))
    return st.number_input(label, key=key, **kw)


def _vector_inputs(prefix: str, title: str, unit: str,
                   mag_default: float = 0.0, ang_default: float = 0.0):
    st.markdown(f"**{title}**")
    c1, c2 = st.columns(2)
    with c1:
        mag = _num(f"{prefix}_mag", f"Magnitud [{unit}]",
                   mag_default, min_value=0.0, step=0.1, format="%.3f")
    with c2:
        ang = _num(f"{prefix}_ang", "Ángulo [°]",
                   ang_default, step=1.0, format="%.1f")
    return mag, ang


def _quality_badge(quality: str) -> None:
    q = (quality or "").upper()
    if q == "GOOD":
        st.success("Calidad del modelo: GOOD — modelo estable.")
    elif q in ("MED", "MEDIUM"):
        st.warning("Calidad del modelo: MED — sensible al ruido, revisar repetibilidad.")
    else:
        st.error("Calidad del modelo: POOR — sensibilidad baja / matriz mal condicionada.")


# =====================================================================
# Import desde Live Monitoring
# =====================================================================
def _instance_options() -> List[Tuple[str, str]]:
    try:
        from core.instance_state import list_instances
        insts = list_instances() or []
    except Exception:
        insts = []
    opts: List[Tuple[str, str]] = []
    for m in insts:
        iid = m.get("instance_id")
        if not iid:
            continue
        tag = m.get("tag") or iid
        opts.append((iid, f"{tag}  ·  {iid}"))
    return opts


def _bal_capture_cb(iid: str, targets: List[Tuple[str, str, str]]) -> None:
    """on_click: captura el 1X de las sondas y precarga los campos.
    targets = [(sensor_label, mag_key, ang_key), ...]. Correr en callback
    permite escribir st.session_state de widgets (patrón Streamlit)."""
    from core.balance.live_source import capture_1x
    labels = [t[0] for t in targets if t[0]]
    res = capture_1x(iid, labels) if labels else {}
    ok, warn = [], []
    for lbl, mag_key, ang_key in targets:
        v = res.get(lbl) if lbl else None
        if v is None:
            warn.append(f"⚠ sin 1X live para {lbl or '—'}")
            continue
        mag, ph, _unit, ts = v
        st.session_state[mag_key] = float(mag)
        st.session_state[ang_key] = float(ph)
        ok.append(f"{lbl}: {mag:.3f} @ {ph:.1f}°")
    parts = ok + warn
    st.session_state["_bal_msg"] = " · ".join(parts) if parts else "Sin datos live."


def _machine_and_planes(key: str):
    """Selector de máquina + carga de planos. Devuelve (iid, planes) o (None, []).
    `key` debe ser único por pestaña (Streamlit renderiza todas las tabs en el
    mismo run, así que dos selectbox con la misma key colisionarían)."""
    opts = _instance_options()
    if not opts:
        st.info("No hay máquinas disponibles.")
        return None, []
    iid = st.selectbox("Máquina", [o[0] for o in opts],
                       format_func=lambda x: dict(opts).get(x, x),
                       key=f"{key}_iid")
    from core.balance.live_source import list_balance_planes
    planes = list_balance_planes(iid)
    if not planes:
        st.warning("Esta máquina no tiene sondas de proximidad radiales configuradas.")
    return iid, planes


def _plane_label(p) -> str:
    return f"[{p['section']}] {p['plane_label']} (plano {p['plane']})"


# =====================================================================
# Tabs
# =====================================================================
tab_tw, tab_1p, tab_2p, tab_iso, tab_rep = st.tabs([
    "Peso de prueba", "1 plano", "2 planos", "Validación ISO", "Reporte",
])


# ---------------------------------------------------------------------
# 1) Peso de prueba (API 684)
# ---------------------------------------------------------------------
with tab_tw:
    st.caption("Recomendación de peso de prueba según API 684 — "
               "Umax = 6350 · W[kg] / N[rpm].")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        W_plane = _num("tw_wplane", "Peso del plano W [kg]", 3500.0,
                       min_value=0.0, step=10.0, format="%.1f")
    with c2:
        rpm = _num("tw_rpm", "Velocidad N [rpm]", 3600.0,
                   min_value=0.0, step=10.0, format="%.0f")
    with c3:
        radius = _num("tw_radius", "Radio de corrección [mm]", 420.0,
                      min_value=0.0, step=1.0, format="%.1f")
    with c4:
        k = _num("tw_k", "Factor k (0.2–2.0)", 1.25,
                 min_value=0.2, max_value=2.0, step=0.05, format="%.2f")

    Wtrial, Utrial = recommend_trial_weight_g(W_plane, rpm, radius, k)
    Umax = umax_api684_gmm(W_plane, rpm)
    m1, m2, m3 = st.columns(3)
    m1.metric("Umax (API 684)", f"{Umax:,.1f} g·mm")
    m2.metric("U de prueba", f"{Utrial:,.1f} g·mm")
    m3.metric("Peso de prueba recomendado", f"{Wtrial:,.2f} g")
    st.caption("Ajustá el peso a la masa/rosca disponible y confirmá que genere "
               "un cambio de vector medible.")


# ---------------------------------------------------------------------
# 2) Balanceo en 1 plano (ISO 21940-12)
# ---------------------------------------------------------------------
with tab_1p:
    st.caption("Coeficiente de influencia en 1 plano: H = (Vt − V0) / Wt · "
               "Wcorr = −V0 / H.")
    unit1 = st.selectbox("Unidad de vibración", UNITS, key="b1_unit")
    source1 = st.radio("Fuente de datos", ["Manual", "Live Monitoring"],
                       key="b1_source", horizontal=True)

    if source1 == "Live Monitoring":
        iid, planes = _machine_and_planes("b1_live")
        if planes:
            cpa, cpb = st.columns([2, 1])
            with cpa:
                p_idx = st.selectbox("Plano a balancear", list(range(len(planes))),
                                     format_func=lambda i: _plane_label(planes[i]),
                                     key="b1_live_plane")
            with cpb:
                direction = st.radio("Dirección", ["Y", "X"], key="b1_live_dir",
                                     horizontal=True)
            from core.balance.live_source import pick_sensor_for_plane
            sensor = pick_sensor_for_plane(planes[p_idx], direction)
            st.caption(f"Sonda seleccionada: **{sensor or '—'}**")
            b1, b2, b3 = st.columns(3)
            b1.button("Capturar V0", key="b1_cap_v0", use_container_width=True,
                      on_click=_bal_capture_cb,
                      args=(iid, [(sensor, "b1_v0_mag", "b1_v0_ang")]))
            b2.button("Capturar Vt", key="b1_cap_vt", use_container_width=True,
                      on_click=_bal_capture_cb,
                      args=(iid, [(sensor, "b1_vt_mag", "b1_vt_ang")]))
            b3.button("Capturar Vf", key="b1_cap_vf", use_container_width=True,
                      on_click=_bal_capture_cb,
                      args=(iid, [(sensor, "b1_vf_mag", "b1_vf_ang")]))
            if st.session_state.get("_bal_msg"):
                st.info(st.session_state["_bal_msg"])
        st.divider()

    colA, colB, colC = st.columns(3)
    with colA:
        v0m, v0a = _vector_inputs("b1_v0", "V0 — vibración inicial", unit1)
    with colB:
        twm, twa = _vector_inputs("b1_tw", "Peso de prueba", "g")
    with colC:
        vtm, vta = _vector_inputs("b1_vt", "Vt — con peso de prueba", unit1)

    if st.button("Calcular balanceo 1 plano", key="b1_calc", type="primary"):
        try:
            st.session_state["bal_r1p"] = solve_1plane(v0m, v0a, vtm, vta, twm, twa)
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("bal_r1p", None)

    r = st.session_state.get("bal_r1p")
    if r:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Peso de corrección", f"{r['corr_mass_g']:,.2f} g")
        c2.metric("Ángulo de corrección", f"{r['corr_ang_deg']:,.1f} °")
        c3.metric("Vibración residual estimada", f"{r['pred_mag']:,.3f} {unit1}")
        _quality_badge(r["quality"])
        st.caption(r["note"])
        with st.expander("Validar contra la medición final (opcional)"):
            vfm, _vfa = _vector_inputs("b1_vf", "Vf — vibración final medida", unit1)
            if vfm > 0:
                st.metric("Reducción de vibración", f"{pct_reduction(v0m, vfm):,.1f} %")


# ---------------------------------------------------------------------
# 3) Balanceo en 2 planos (ISO 21940-12)
# ---------------------------------------------------------------------
with tab_2p:
    st.caption("Coeficiente de influencia en 2 planos (matriz 2×2). Corridas: "
               "0 = inicial · 1 = trial en plano A · 2 = trial en plano B.")
    unit2 = st.selectbox("Unidad de vibración", UNITS, key="b2_unit")
    source2 = st.radio("Fuente de datos", ["Manual", "Live Monitoring"],
                       key="b2_source", horizontal=True)

    if source2 == "Live Monitoring":
        iid, planes = _machine_and_planes("b2_live")
        if planes and len(planes) >= 2:
            c1, c2, c3 = st.columns(3)
            with c1:
                ia = st.selectbox("Plano A (lado acople)", list(range(len(planes))),
                                  format_func=lambda i: _plane_label(planes[i]),
                                  key="b2_live_planeA")
            with c2:
                _def_b = 1 if len(planes) > 1 else 0
                st.session_state.setdefault("b2_live_planeB", _def_b)
                ib = st.selectbox("Plano B (lado libre)", list(range(len(planes))),
                                  format_func=lambda i: _plane_label(planes[i]),
                                  key="b2_live_planeB")
            with c3:
                # Misma dirección en AMBOS planos por defecto (3X → 4X).
                direction = st.radio("Dirección (ambos planos)", ["Y", "X"],
                                     key="b2_live_dir", horizontal=True)
            from core.balance.live_source import pick_sensor_for_plane
            sA = pick_sensor_for_plane(planes[ia], direction)
            sB = pick_sensor_for_plane(planes[ib], direction)
            st.caption(f"Sondas: A = **{sA or '—'}** · B = **{sB or '—'}** "
                       f"(misma dirección {direction})")
            b1, b2, b3 = st.columns(3)
            b1.button("Capturar corrida 0 (A0,B0)", key="b2_cap0",
                      use_container_width=True, on_click=_bal_capture_cb,
                      args=(iid, [(sA, "b2_a0_mag", "b2_a0_ang"),
                                  (sB, "b2_b0_mag", "b2_b0_ang")]))
            b2.button("Capturar corrida 1 (A1,B1)", key="b2_cap1",
                      use_container_width=True, on_click=_bal_capture_cb,
                      args=(iid, [(sA, "b2_a1_mag", "b2_a1_ang"),
                                  (sB, "b2_b1_mag", "b2_b1_ang")]))
            b3.button("Capturar corrida 2 (A2,B2)", key="b2_cap2",
                      use_container_width=True, on_click=_bal_capture_cb,
                      args=(iid, [(sA, "b2_a2_mag", "b2_a2_ang"),
                                  (sB, "b2_b2_mag", "b2_b2_ang")]))
            if st.session_state.get("_bal_msg"):
                st.info(st.session_state["_bal_msg"])
        st.divider()

    st.markdown("##### Corrida 0 — inicial")
    c1, c2 = st.columns(2)
    with c1:
        a0m, a0a = _vector_inputs("b2_a0", "A0 — sonda plano A", unit2)
    with c2:
        b0m, b0a = _vector_inputs("b2_b0", "B0 — sonda plano B", unit2)

    st.markdown("##### Corrida 1 — peso de prueba en plano A")
    c1, c2, c3 = st.columns(3)
    with c1:
        wam, waa = _vector_inputs("b2_wa", "Trial plano A", "g")
    with c2:
        a1m, a1a = _vector_inputs("b2_a1", "A1 — sonda A", unit2)
    with c3:
        b1m, b1a = _vector_inputs("b2_b1", "B1 — sonda B", unit2)

    st.markdown("##### Corrida 2 — peso de prueba en plano B")
    c1, c2, c3 = st.columns(3)
    with c1:
        wbm, wba = _vector_inputs("b2_wb", "Trial plano B", "g")
    with c2:
        a2m, a2a = _vector_inputs("b2_a2", "A2 — sonda A", unit2)
    with c3:
        b2m, b2a = _vector_inputs("b2_b2", "B2 — sonda B", unit2)

    if st.button("Calcular balanceo 2 planos", key="b2_calc", type="primary"):
        try:
            st.session_state["bal_r2p"] = solve_2plane(
                to_complex(a0m, a0a), to_complex(b0m, b0a),
                to_complex(a1m, a1a), to_complex(b1m, b1a),
                to_complex(a2m, a2a), to_complex(b2m, b2a),
                to_complex(wam, waa), to_complex(wbm, wba),
            )
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("bal_r2p", None)

    r = st.session_state.get("bal_r2p")
    if r:
        st.divider()
        wa_mag, wa_ang = to_polar(r["WA_corr"])
        wb_mag, wb_ang = to_polar(r["WB_corr"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Plano A — corrección**")
            st.metric("Peso", f"{wa_mag:,.2f} g")
            st.metric("Ángulo", f"{wa_ang:,.1f} °")
            st.metric("Residual A estimado", f"{abs(r['A_after']):,.3f} {unit2}")
        with c2:
            st.markdown("**Plano B — corrección**")
            st.metric("Peso", f"{wb_mag:,.2f} g")
            st.metric("Ángulo", f"{wb_ang:,.1f} °")
            st.metric("Residual B estimado", f"{abs(r['B_after']):,.3f} {unit2}")
        _quality_badge(r["quality"])
        st.caption(f"{r['note']}  ·  cond(M) = {r['cond']:.1f}")


# ---------------------------------------------------------------------
# 4) Validación ISO 21940-11
# ---------------------------------------------------------------------
with tab_iso:
    st.caption("Desbalance residual permisible: e_per = 9549·G/N · U_per = e_per·W. "
               "Se evalúa el residual contra los grados ISO 21940.")
    c1, c2 = st.columns(2)
    with c1:
        W_iso = _num("iso_w", "Peso del rotor W [kg]", 11000.0,
                     min_value=0.0, step=10.0, format="%.1f")
        rpm_iso = _num("iso_rpm", "Velocidad N [rpm]", 3600.0,
                       min_value=0.0, step=10.0, format="%.0f")
    with c2:
        modo = st.radio("Residual U_res", ["Ingresar U_res [g·mm]",
                                           "Calcular de masa·radio"], key="iso_mode")
        if modo.startswith("Ingresar"):
            U_res = _num("iso_ures", "U_res [g·mm]", 0.0, min_value=0.0,
                         step=1.0, format="%.1f")
        else:
            mr = _num("iso_resmass", "Masa residual [g]", 0.0, min_value=0.0,
                      step=0.1, format="%.2f")
            rr = _num("iso_resrad", "Radio [mm]", 420.0, min_value=0.0,
                      step=1.0, format="%.1f")
            U_res = calc_U_trial(mr, rr)
            st.metric("U_res calculado", f"{U_res:,.1f} g·mm")

    ev = evaluate_iso_grades(W_iso, rpm_iso, U_res)
    st.session_state["bal_iso"] = ev
    if ev["status_code"] == "FAIL":
        st.error(ev["summary_label"])
    elif ev["best_grade"] is not None and ev["best_grade"] <= 2.5:
        st.success(ev["summary_label"])
    else:
        st.warning(ev["summary_label"])

    st.markdown("**Evaluación por grado ISO**")
    rows = []
    for g in ev["results"]:
        rows.append({
            "Grado": f"G{g['G']:g}",
            "e_per [µm]": round(g["e_per"], 3),
            "U_per [g·mm]": round(g["U_per"], 1),
            "U_res/U_per": round(g["ratio"], 2) if g["ratio"] < 900 else "—",
            "Cumple": "✅" if g["pass"] else "❌",
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------
# 5) Reporte (resumen de sesión; PDF branded = siguiente paso)
# ---------------------------------------------------------------------
with tab_rep:
    st.caption("Resumen consolidado de la sesión de balanceo.")
    lines: List[str] = []
    r1 = st.session_state.get("bal_r1p")
    if r1:
        lines.append(
            f"1 plano · corrección {r1['corr_mass_g']:.2f} g @ "
            f"{r1['corr_ang_deg']:.1f}° · residual estimado {r1['pred_mag']:.3f} "
            f"· calidad {r1['quality']}"
        )
    r2 = st.session_state.get("bal_r2p")
    if r2:
        wa_m, wa_a = to_polar(r2["WA_corr"])
        wb_m, wb_a = to_polar(r2["WB_corr"])
        lines.append(
            f"2 planos · A: {wa_m:.2f} g @ {wa_a:.1f}° · B: {wb_m:.2f} g @ "
            f"{wb_a:.1f}° · calidad {r2['quality']}"
        )
    ev = st.session_state.get("bal_iso")
    if ev:
        lines.append(f"Validación ISO · {ev['summary_label']}")

    if lines:
        st.text("\n".join(lines))
        st.download_button(
            "Descargar resumen (.txt)",
            data="\n".join(lines).encode("utf-8"),
            file_name="resumen_balanceo.txt",
            mime="text/plain",
        )
    else:
        st.info("Aún no hay cálculos en esta sesión. Corré un balanceo en las "
                "pestañas anteriores.")

    st.divider()
    st.markdown("##### Datos del reporte")

    # Prefill opcional desde la máquina Live seleccionada (activo/cliente/sitio).
    _live_iid = st.session_state.get("b2_live_iid") or st.session_state.get("b1_live_iid")
    if _live_iid:
        try:
            from core.instance_state import get_instance as _gi
            _inst = _gi(_live_iid)
            if _inst is not None:
                st.session_state.setdefault("rep_asset", _inst.tag or _live_iid)
                st.session_state.setdefault("rep_client", getattr(_inst, "client", "") or "")
                st.session_state.setdefault(
                    "rep_location",
                    getattr(_inst, "site", "") or getattr(_inst, "location", "") or "")
        except Exception:
            pass

    from datetime import date as _date
    st.session_state.setdefault("rep_asset", "")
    st.session_state.setdefault("rep_client", "")
    st.session_state.setdefault("rep_location", "")
    st.session_state.setdefault("rep_specialist", _user.get("full_name") or "")
    st.session_state.setdefault("rep_date", _date.today().strftime("%d/%m/%Y"))

    c1, c2, c3 = st.columns(3)
    with c1:
        rep_asset = st.text_input("Activo", key="rep_asset")
        rep_client = st.text_input("Cliente", key="rep_client")
    with c2:
        rep_location = st.text_input("Sitio / ubicación", key="rep_location")
        rep_specialist = st.text_input("Especialista", key="rep_specialist")
    with c3:
        rep_date = st.text_input("Fecha", key="rep_date")
        rep_notes = st.text_area("Notas", key="rep_notes", height=80)

    def _pair(prefix: str):
        m = st.session_state.get(f"{prefix}_mag")
        if m is None:
            return None
        a = st.session_state.get(f"{prefix}_ang") or 0.0
        return (float(m), float(a))

    one_plane = None
    if r1:
        vf = _pair("b1_vf")
        vf = vf if (vf and vf[0] > 0) else None
        one_plane = {"unit": st.session_state.get("b1_unit", "µm pk-pk"),
                     "v0": _pair("b1_v0"), "trial": _pair("b1_tw"),
                     "vt": _pair("b1_vt"), "vf": vf, "result": r1}
    two_plane = None
    if r2:
        two_plane = {"unit": st.session_state.get("b2_unit", "µm pk-pk"),
                     "a0": _pair("b2_a0"), "b0": _pair("b2_b0"),
                     "a1": _pair("b2_a1"), "b1": _pair("b2_b1"),
                     "a2": _pair("b2_a2"), "b2": _pair("b2_b2"),
                     "wa": _pair("b2_wa"), "wb": _pair("b2_wb"), "result": r2}

    if not (one_plane or two_plane or ev):
        st.info("Corré al menos un balanceo o una validación ISO para generar el PDF.")
    else:
        if st.button("Generar reporte PDF", key="rep_pdf", type="primary"):
            try:
                from core.balance.report import build_balance_pdf
                meta = {
                    "asset": rep_asset, "client": rep_client, "location": rep_location,
                    "specialist": rep_specialist, "report_date": rep_date,
                    "unit": (st.session_state.get("b1_unit")
                             or st.session_state.get("b2_unit") or "µm pk-pk"),
                    "rpm": (st.session_state.get("iso_rpm")
                            or st.session_state.get("tw_rpm")),
                    "notes": rep_notes,
                }
                st.session_state["bal_pdf"] = build_balance_pdf(
                    meta=meta, one_plane=one_plane, two_plane=two_plane, iso=ev)
            except Exception as e:  # noqa: BLE001
                st.error(f"Error generando el PDF: {e}")
                st.session_state.pop("bal_pdf", None)
        if st.session_state.get("bal_pdf"):
            import re as _re
            _fn = "Balanceo_" + _re.sub(r"[^A-Za-z0-9]+", "_",
                                        (rep_asset or "activo")).strip("_") + ".pdf"
            st.download_button("⬇ Descargar PDF", data=st.session_state["bal_pdf"],
                               file_name=_fn, mime="application/pdf")
