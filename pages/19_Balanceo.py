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

Entrada de datos: MANUAL en esta versión. La importación del vector 1X
(magnitud + fase) desde Live Monitoring es el siguiente paso (core.balance.
live_source), respetando la convención de fase del keyphasor.

Marco normativo: ISO 21940-11 / ISO 21940-12 · API 684.
Acceso: analista / admin (gateada; el cliente no la ve).
"""
from __future__ import annotations

import streamlit as st

from core.auth import (
    require_login, render_user_menu, get_current_user, is_page_allowed_for_role,
)
from core.ui_theme import page_header
from core.balance import (
    to_complex, to_polar,
    umax_api684_gmm, recommend_trial_weight_g,
    calc_e_per, calc_U_per, calc_U_trial, pct_reduction, status_from_ratio,
    evaluate_iso_grades, ISO_GRADES,
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
    """number_input persistente por key (sin warning value+key)."""
    st.session_state.setdefault(key, float(default))
    return st.number_input(label, key=key, **kw)


def _vector_inputs(prefix: str, title: str, unit: str,
                   mag_default: float = 0.0, ang_default: float = 0.0):
    """Par (magnitud, ángulo) para un vector de vibración o peso."""
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
    st.caption("El peso de prueba se aplica al radio indicado; ajústalo a la "
               "masa/rosca disponible y confirma que genere cambio de vector medible.")


# ---------------------------------------------------------------------
# 2) Balanceo en 1 plano (ISO 21940-12)
# ---------------------------------------------------------------------
with tab_1p:
    st.caption("Coeficiente de influencia en 1 plano: H = (Vt − V0) / Wt · "
               "Wcorr = −V0 / H.")
    unit1 = st.selectbox("Unidad de vibración", UNITS, key="b1_unit")

    st.radio("Fuente de datos", ["Manual", "Live Monitoring (próximamente)"],
             key="b1_source", horizontal=True,
             help="La importación del 1X (mag+fase) desde Live Monitoring llega "
                  "en el siguiente paso.")

    colA, colB, colC = st.columns(3)
    with colA:
        v0m, v0a = _vector_inputs("b1_v0", "V0 — vibración inicial", unit1)
    with colB:
        twm, twa = _vector_inputs("b1_tw", "Peso de prueba", "g")
    with colC:
        vtm, vta = _vector_inputs("b1_vt", "Vt — con peso de prueba", unit1)

    if st.button("Calcular balanceo 1 plano", key="b1_calc", type="primary"):
        try:
            r = solve_1plane(v0m, v0a, vtm, vta, twm, twa)
            st.session_state["bal_r1p"] = r
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
                red = pct_reduction(v0m, vfm)
                st.metric("Reducción de vibración", f"{red:,.1f} %")


# ---------------------------------------------------------------------
# 3) Balanceo en 2 planos (ISO 21940-12)
# ---------------------------------------------------------------------
with tab_2p:
    st.caption("Coeficiente de influencia en 2 planos (matriz 2×2). Corridas: "
               "0 = inicial · 1 = trial en plano A · 2 = trial en plano B.")
    unit2 = st.selectbox("Unidad de vibración", UNITS, key="b2_unit")

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
            r = solve_2plane(
                to_complex(a0m, a0a), to_complex(b0m, b0a),
                to_complex(a1m, a1a), to_complex(b1m, b1a),
                to_complex(a2m, a2a), to_complex(b2m, b2a),
                to_complex(wam, waa), to_complex(wbm, wba),
            )
            st.session_state["bal_r2p"] = r
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
    lines = []
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
    st.caption("El reporte PDF branded SIGA/Watermelon (reutilizando el "
               "generador de ROTORIX) es el siguiente paso de la integración.")
