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
5. Reporte          — resumen + PDF branded Watermelon/SIGA.

Entrada de datos: MANUAL o importada desde LIVE MONITORING (vector 1X mag+fase
por sonda). Selección por sonda/plano agrupada por sección; en 2 planos la
misma dirección (X/Y) en ambos planos por defecto.

UI: kit enterprise core.balance.ui (mismo lenguaje visual que Modal Analysis).
Acceso: analista / admin (gateada; el cliente no la ve).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import streamlit as st

from core.auth import (
    require_login, render_user_menu, get_current_user, is_page_allowed_for_role,
)
from core.ui_theme import apply_watermelon_page_style
from core.balance import (
    to_complex, to_polar,
    umax_api684_gmm, recommend_trial_weight_g,
    calc_U_trial, pct_reduction,
    evaluate_iso_grades,
    solve_1plane, solve_2plane,
)
from core.balance.ui import (
    bal_hero_card, bal_section_header, bal_kpi_row, bal_status_banner,
    bal_footer_norms,
)
from core.balance.rotor3d import rotor_3d_1plane, rotor_3d_2plane


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
apply_watermelon_page_style()

_user = get_current_user() or {}
_role = str(_user.get("role", "")).lower()
if not is_page_allowed_for_role("pages/19_Balanceo.py", _role):
    st.error("Tu rol no tiene acceso a este módulo.")
    st.stop()

UNITS = ["µm pk-pk", "mil pk-pk", "mm/s RMS"]


# =====================================================================
# Hero del módulo (activo + modo activo, estilo Modal)
# =====================================================================
def _hero() -> None:
    if st.session_state.get("bal_r2p"):
        mode = "2 planos"
    elif st.session_state.get("bal_r1p"):
        mode = "1 plano"
    else:
        mode = "—"
    bal_hero_card(
        asset_name=st.session_state.get("rep_asset") or "(activo sin especificar)",
        client=st.session_state.get("rep_client", ""),
        site=st.session_state.get("rep_location", ""),
        mode=mode,
    )


_hero()


# =====================================================================
# Helpers de UI
# =====================================================================
def _num(key: str, label: str, default: float = 0.0, **kw) -> float:
    st.session_state.setdefault(key, float(default))
    return st.number_input(label, key=key, **kw)


def _vector_inputs(prefix: str, title: str, unit: str):
    st.markdown(f"<div style='font-weight:700;color:#0F1E3D;font-size:13px;"
                f"margin-bottom:2px;'>{title}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        mag = _num(f"{prefix}_mag", f"Magnitud [{unit}]", 0.0,
                   min_value=0.0, step=0.1, format="%.3f")
    with c2:
        ang = _num(f"{prefix}_ang", "Ángulo [°]", 0.0, step=1.0, format="%.1f")
    return mag, ang


def _quality_severity(quality: str) -> Tuple[str, str, str]:
    q = (quality or "").upper()
    if q == "GOOD":
        return ("ok", "Modelo estable", "GOOD")
    if q in ("MED", "MEDIUM"):
        return ("warning", "Sensible al ruido — revisar repetibilidad", "MED")
    return ("fail", "Sensibilidad baja / matriz mal condicionada", "POOR")


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
        opts.append((iid, f"{m.get('tag') or iid}  ·  {iid}"))
    return opts


def _bal_capture_cb(iid: str, targets: List[Tuple[str, str, str]]) -> None:
    from core.balance.live_source import capture_1x
    labels = [t[0] for t in targets if t[0]]
    res = capture_1x(iid, labels) if labels else {}
    ok, warn = [], []
    for lbl, mag_key, ang_key in targets:
        v = res.get(lbl) if lbl else None
        if v is None:
            warn.append(f"⚠ sin 1X live para {lbl or '—'}")
            continue
        mag, ph, _unit, _ts = v
        st.session_state[mag_key] = float(mag)
        st.session_state[ang_key] = float(ph)
        ok.append(f"{lbl}: {mag:.3f} ∠ {ph:.1f}°")
    st.session_state["_bal_msg"] = " · ".join(ok + warn) or "Sin datos live."


def _suggest_trial_cb(w_key: str, n_key: str, r_key: str, k_key: str, mag_key: str) -> None:
    """on_click: calcula el peso de prueba (API 684) y lo rellena en mag_key."""
    W = float(st.session_state.get(w_key) or 0.0)
    N = float(st.session_state.get(n_key) or 0.0)
    R = float(st.session_state.get(r_key) or 0.0)
    k = float(st.session_state.get(k_key) or 1.25)
    Wt, _u = recommend_trial_weight_g(W, N, R, k)
    st.session_state[mag_key] = round(float(Wt), 2)
    st.session_state["_bal_tw_msg"] = f"Peso de prueba sugerido: {Wt:,.2f} g (API 684)"


def _trial_weight_suggester(prefix: str, mag_key: str,
                            title: str = "Sugerir peso de prueba (API 684)") -> None:
    """Panel compacto: W del plano + rpm + radio → sugiere y rellena el peso
    de prueba de esa corrida. W = carga soportada por ESE plano (≈ ½ del rotor
    si está entre dos cojinetes)."""
    with st.expander(title, expanded=False):
        st.caption("W = peso soportado por **este plano** (≈ ½ del rotor entre 2 "
                   "cojinetes). Fórmula API 684: W_prueba = 6350·W·k / (N·radio).")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _num(f"{prefix}_sw", "Peso del plano W [kg]", 3500.0,
                 min_value=0.0, step=10.0, format="%.1f")
        with c2:
            _num(f"{prefix}_sn", "Velocidad N [rpm]",
                 float(st.session_state.get("iso_rpm")
                       or st.session_state.get("tw_rpm") or 3600.0),
                 min_value=0.0, step=10.0, format="%.0f")
        with c3:
            _num(f"{prefix}_sr", "Radio [mm]", 420.0,
                 min_value=0.0, step=1.0, format="%.1f")
        with c4:
            _num(f"{prefix}_sk", "Factor k", 1.25,
                 min_value=0.2, max_value=2.0, step=0.05, format="%.2f")
        Wt, _u = recommend_trial_weight_g(
            st.session_state[f"{prefix}_sw"], st.session_state[f"{prefix}_sn"],
            st.session_state[f"{prefix}_sr"], st.session_state[f"{prefix}_sk"])
        st.button(f"Sugerir → {Wt:,.2f} g  ·  rellena el peso de prueba",
                  key=f"{prefix}_sbtn", on_click=_suggest_trial_cb,
                  args=(f"{prefix}_sw", f"{prefix}_sn", f"{prefix}_sr",
                        f"{prefix}_sk", mag_key))
        if st.session_state.get("_bal_tw_msg"):
            st.caption("✅ " + st.session_state["_bal_tw_msg"])


def _machine_and_planes(key: str):
    opts = _instance_options()
    if not opts:
        st.info("No hay máquinas disponibles.")
        return None, []
    iid = st.selectbox("Máquina", [o[0] for o in opts],
                       format_func=lambda x: dict(opts).get(x, x), key=f"{key}_iid")
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
    bal_section_header("Peso de prueba", "Masa de arranque para provocar un "
                       "cambio de vector medible.", "API 684 · Umax = 6350·W/N", "⚙️")
    with st.container(border=True):
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
    bal_kpi_row([
        (f"{Wtrial:,.2f} g", "Peso de prueba", "recomendado @ radio indicado", "cyan"),
        (f"{Umax:,.0f}", "Umax [g·mm]", "API 684", "navy"),
        (f"{Utrial:,.0f}", "U de prueba [g·mm]", f"k = {k:.2f}", "navy"),
    ])
    st.caption("Ajustá el peso a la masa/rosca disponible y confirmá que genere "
               "un cambio de vector medible antes de calcular la corrección.")


# ---------------------------------------------------------------------
# 2) Balanceo en 1 plano
# ---------------------------------------------------------------------
with tab_1p:
    bal_section_header("Balanceo en 1 plano",
                       "H = (Vt − V0) / Wt  ·  Wcorr = −V0 / H",
                       "ISO 21940-12 · coeficiente de influencia", "🎯")
    # Rotor 3D SIEMPRE visible arriba (neutro al abrir; con el contrapeso al calcular).
    _r1prev = st.session_state.get("bal_r1p")
    st.plotly_chart(
        rotor_3d_1plane(_r1prev["corr_ang_deg"] if _r1prev else None,
                        f"{_r1prev['corr_mass_g']:.1f} g" if _r1prev else "—"),
        use_container_width=True)
    if not _r1prev:
        st.caption("🧭 El rotor muestra la posición del contrapeso apenas calculás el balanceo.")

    top = st.columns([1, 1])
    with top[0]:
        unit1 = st.selectbox("Unidad de vibración", UNITS, key="b1_unit")
    with top[1]:
        source1 = st.radio("Fuente de datos", ["Manual", "Live Monitoring"],
                           key="b1_source", horizontal=True)

    if source1 == "Live Monitoring":
        with st.container(border=True):
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
                st.caption(f"Sonda seleccionada: **{sensor or '—'}**  ·  "
                           "capturá el 1X en cada corrida.")
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
                    st.caption("📡 " + st.session_state["_bal_msg"])

    _trial_weight_suggester("b1_tw", "b1_tw_mag")

    with st.container(border=True):
        colA, colB, colC = st.columns(3)
        with colA:
            v0m, v0a = _vector_inputs("b1_v0", "V0 — vibración inicial", unit1)
        with colB:
            twm, twa = _vector_inputs("b1_tw", "Peso de prueba [g]", "g")
        with colC:
            vtm, vta = _vector_inputs("b1_vt", "Vt — con peso de prueba", unit1)

    if st.button("Calcular balanceo 1 plano", key="b1_calc", type="primary"):
        try:
            st.session_state["bal_r1p"] = solve_1plane(v0m, v0a, vtm, vta, twm, twa)
            st.rerun()
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("bal_r1p", None)

    r = st.session_state.get("bal_r1p")
    if r:
        st.markdown("")
        bal_kpi_row([
            (f"{r['corr_mass_g']:,.2f} g", "Peso de corrección", "masa a instalar", "cyan"),
            (f"{r['corr_ang_deg']:,.1f}°", "Ángulo", "posición angular", "cyan"),
            (f"{r['pred_mag']:,.3f}", f"Residual [{unit1}]", "vibración estimada", "green"),
        ])
        sev, detail, tag = _quality_severity(r["quality"])
        bal_status_banner(f"Calidad del modelo: {tag}", f"{detail}. {r['note']}", sev)
        with st.expander("Validar contra la medición final (opcional)"):
            vfm, _vfa = _vector_inputs("b1_vf", "Vf — vibración final medida", unit1)
            if vfm > 0:
                bal_kpi_row([(f"{pct_reduction(v0m, vfm):,.1f} %",
                              "Reducción de vibración", "V0 → Vf", "green")])


# ---------------------------------------------------------------------
# 3) Balanceo en 2 planos
# ---------------------------------------------------------------------
with tab_2p:
    bal_section_header("Balanceo en 2 planos",
                       "Matriz de coeficientes de influencia 2×2 · corridas "
                       "0 (inicial) · 1 (trial A) · 2 (trial B).",
                       "ISO 21940-12", "🎯")
    # Rotor 3D SIEMPRE visible arriba (neutro al abrir; con contrapesos al calcular).
    _r2prev = st.session_state.get("bal_r2p")
    if _r2prev:
        _wam, _waa = to_polar(_r2prev["WA_corr"])
        _wbm, _wba = to_polar(_r2prev["WB_corr"])
        _fig2 = rotor_3d_2plane(_waa, f"{_wam:.1f} g", _wba, f"{_wbm:.1f} g")
    else:
        _fig2 = rotor_3d_2plane(None, "—", None, "—")
    st.plotly_chart(_fig2, use_container_width=True)
    if not _r2prev:
        st.caption("🧭 El rotor muestra la posición de los contrapesos (planos A y B) al calcular.")

    top = st.columns([1, 1])
    with top[0]:
        unit2 = st.selectbox("Unidad de vibración", UNITS, key="b2_unit")
    with top[1]:
        source2 = st.radio("Fuente de datos", ["Manual", "Live Monitoring"],
                           key="b2_source", horizontal=True)

    if source2 == "Live Monitoring":
        with st.container(border=True):
            iid, planes = _machine_and_planes("b2_live")
            if planes and len(planes) >= 2:
                c1, c2, c3 = st.columns(3)
                with c1:
                    ia = st.selectbox("Plano A (lado acople)", list(range(len(planes))),
                                      format_func=lambda i: _plane_label(planes[i]),
                                      key="b2_live_planeA")
                with c2:
                    st.session_state.setdefault("b2_live_planeB", 1 if len(planes) > 1 else 0)
                    ib = st.selectbox("Plano B (lado libre)", list(range(len(planes))),
                                      format_func=lambda i: _plane_label(planes[i]),
                                      key="b2_live_planeB")
                with c3:
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
                    st.caption("📡 " + st.session_state["_bal_msg"])

    with st.container(border=True):
        st.markdown("**Corrida 0 — inicial**")
        c1, c2 = st.columns(2)
        with c1:
            a0m, a0a = _vector_inputs("b2_a0", "A0 — sonda plano A", unit2)
        with c2:
            b0m, b0a = _vector_inputs("b2_b0", "B0 — sonda plano B", unit2)
    _trial_weight_suggester("b2_wa", "b2_wa_mag", "Sugerir peso de prueba · plano A (API 684)")
    with st.container(border=True):
        st.markdown("**Corrida 1 — peso de prueba en plano A**")
        c1, c2, c3 = st.columns(3)
        with c1:
            wam, waa = _vector_inputs("b2_wa", "Trial plano A [g]", "g")
        with c2:
            a1m, a1a = _vector_inputs("b2_a1", "A1 — sonda A", unit2)
        with c3:
            b1m, b1a = _vector_inputs("b2_b1", "B1 — sonda B", unit2)
    _trial_weight_suggester("b2_wb", "b2_wb_mag", "Sugerir peso de prueba · plano B (API 684)")
    with st.container(border=True):
        st.markdown("**Corrida 2 — peso de prueba en plano B**")
        c1, c2, c3 = st.columns(3)
        with c1:
            wbm, wba = _vector_inputs("b2_wb", "Trial plano B [g]", "g")
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
            st.rerun()
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("bal_r2p", None)

    r = st.session_state.get("bal_r2p")
    if r:
        st.markdown("")
        wa_mag, wa_ang = to_polar(r["WA_corr"])
        wb_mag, wb_ang = to_polar(r["WB_corr"])
        bal_kpi_row([
            (f"{wa_mag:,.2f} g", "Corrección plano A", f"∠ {wa_ang:,.1f}°", "cyan"),
            (f"{wb_mag:,.2f} g", "Corrección plano B", f"∠ {wb_ang:,.1f}°", "cyan"),
            (f"{abs(r['A_after']):,.3f}", f"Residual A [{unit2}]", "estimado", "green"),
            (f"{abs(r['B_after']):,.3f}", f"Residual B [{unit2}]", "estimado", "green"),
        ])
        sev, detail, tag = _quality_severity(r["quality"])
        bal_status_banner(f"Calidad del modelo: {tag}",
                          f"{detail}. cond(M) = {r['cond']:.1f}. {r['note']}", sev)


# ---------------------------------------------------------------------
# 4) Validación ISO 21940-11
# ---------------------------------------------------------------------
with tab_iso:
    bal_section_header("Validación ISO",
                       "e_per = 9549·G/N  ·  U_per = e_per·W. El residual se "
                       "evalúa contra los grados ISO 21940.",
                       "ISO 21940-11", "✅")
    with st.container(border=True):
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
                st.caption(f"U_res calculado = **{U_res:,.1f} g·mm**")

    ev = evaluate_iso_grades(W_iso, rpm_iso, U_res)
    st.session_state["bal_iso"] = ev
    if ev["status_code"] == "FAIL":
        bal_status_banner("No cumple", ev["summary_label"], "fail")
    elif ev["best_grade"] is not None and ev["best_grade"] <= 2.5:
        bal_status_banner("Cumple", ev["summary_label"], "ok")
    else:
        bal_status_banner("Cumple (calidad básica)", ev["summary_label"], "warning")

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
# 5) Reporte
# ---------------------------------------------------------------------
with tab_rep:
    bal_section_header("Reporte", "Resumen de la sesión y PDF branded "
                       "Watermelon/SIGA.", "ISO 21940 · API 684", "⎙")

    r1 = st.session_state.get("bal_r1p")
    r2 = st.session_state.get("bal_r2p")
    ev = st.session_state.get("bal_iso")

    lines: List[str] = []
    if r1:
        lines.append(f"1 plano · corrección {r1['corr_mass_g']:.2f} g @ "
                     f"{r1['corr_ang_deg']:.1f}° · residual {r1['pred_mag']:.3f} "
                     f"· {r1['quality']}")
    if r2:
        wa_m, wa_a = to_polar(r2["WA_corr"])
        wb_m, wb_a = to_polar(r2["WB_corr"])
        lines.append(f"2 planos · A: {wa_m:.2f} g @ {wa_a:.1f}° · "
                     f"B: {wb_m:.2f} g @ {wb_a:.1f}° · {r2['quality']}")
    if ev:
        lines.append(f"Validación ISO · {ev['summary_label']}")

    if lines:
        with st.container(border=True):
            for ln in lines:
                st.markdown(f"- {ln}")
    else:
        st.info("Aún no hay cálculos en esta sesión. Corré un balanceo en las "
                "pestañas anteriores.")

    st.markdown("")
    bal_section_header("Datos del reporte")

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

    with st.container(border=True):
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
        return (float(m), float(st.session_state.get(f"{prefix}_ang") or 0.0))

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


bal_footer_norms()
