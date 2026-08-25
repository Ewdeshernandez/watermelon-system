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
from core.balance.rotorface import rotor_face_svg, build_planes_1p, build_planes_2p


# =====================================================================
# Setup + auth
# =====================================================================
st.set_page_config(
    page_title="Watermelon System | Balancing",
    page_icon="⚖️",
    layout="wide",
)

require_login()
render_user_menu()
apply_watermelon_page_style()

_user = get_current_user() or {}
_role = str(_user.get("role", "")).lower()
if not is_page_allowed_for_role("pages/19_Balanceo.py", _role):
    st.error("Your role does not have access to this module.")
    st.stop()

UNITS = ["µm pk-pk", "mil pk-pk", "mm/s RMS"]


# =====================================================================
# Hero del módulo (activo + modo activo, estilo Modal)
# =====================================================================
def _hero() -> None:
    if st.session_state.get("bal_r2p"):
        mode = "2 planes"
    elif st.session_state.get("bal_r1p"):
        mode = "1 plane"
    else:
        mode = "—"
    bal_hero_card(
        asset_name=st.session_state.get("rep_asset") or "(unspecified asset)",
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
        mag = _num(f"{prefix}_mag", f"Magnitude [{unit}]", 0.0,
                   min_value=0.0, step=0.1, format="%.3f")
    with c2:
        ang = _num(f"{prefix}_ang", "Angle [°]", 0.0, step=1.0, format="%.1f")
    return mag, ang


def _quality_severity(quality: str) -> Tuple[str, str, str]:
    q = (quality or "").upper()
    if q == "GOOD":
        return ("ok", "Stable model", "GOOD")
    if q in ("MED", "MEDIUM"):
        return ("warning", "Noise-sensitive — check repeatability", "MED")
    return ("fail", "Low sensitivity / ill-conditioned matrix", "POOR")


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
            warn.append(f"⚠ no live 1X for {lbl or '—'}")
            continue
        mag, ph, _unit, _ts = v
        st.session_state[mag_key] = float(mag)
        st.session_state[ang_key] = float(ph)
        ok.append(f"{lbl}: {mag:.3f} ∠ {ph:.1f}°")
    st.session_state["_bal_msg"] = " · ".join(ok + warn) or "No live data."


def _suggest_trial_cb(w_key: str, n_key: str, r_key: str, k_key: str, mag_key: str) -> None:
    """on_click: calcula el peso de prueba (API 684) y lo rellena en mag_key."""
    W = float(st.session_state.get(w_key) or 0.0)
    N = float(st.session_state.get(n_key) or 0.0)
    R = float(st.session_state.get(r_key) or 0.0)
    k = float(st.session_state.get(k_key) or 1.25)
    Wt, _u = recommend_trial_weight_g(W, N, R, k)
    st.session_state[mag_key] = round(float(Wt), 2)
    st.session_state["_bal_tw_msg"] = f"Suggested trial weight: {Wt:,.2f} g (API 684)"


def _trial_weight_suggester(prefix: str, mag_key: str,
                            title: str = "Suggest trial weight (API 684)") -> None:
    """Panel compacto: W del plano + rpm + radio → sugiere y rellena el peso
    de prueba de esa corrida. W = carga soportada por ESE plano (≈ ½ del rotor
    si está entre dos cojinetes)."""
    with st.expander(title, expanded=False):
        st.caption("W = weight supported by **this plane** (≈ ½ of the rotor "
                   "between 2 bearings). API 684 formula: W_trial = 6350·W·k / (N·radius).")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _num(f"{prefix}_sw", "Plane weight W [kg]", 3500.0,
                 min_value=0.0, step=10.0, format="%.1f")
        with c2:
            _num(f"{prefix}_sn", "Speed N [rpm]",
                 float(st.session_state.get("iso_rpm")
                       or st.session_state.get("tw_rpm") or 3600.0),
                 min_value=0.0, step=10.0, format="%.0f")
        with c3:
            _num(f"{prefix}_sr", "Radius [mm]", 420.0,
                 min_value=0.0, step=1.0, format="%.1f")
        with c4:
            _num(f"{prefix}_sk", "Factor k", 1.25,
                 min_value=0.2, max_value=2.0, step=0.05, format="%.2f")
        Wt, _u = recommend_trial_weight_g(
            st.session_state[f"{prefix}_sw"], st.session_state[f"{prefix}_sn"],
            st.session_state[f"{prefix}_sr"], st.session_state[f"{prefix}_sk"])
        st.button(f"Suggest → {Wt:,.2f} g  ·  fill in the trial weight",
                  key=f"{prefix}_sbtn", on_click=_suggest_trial_cb,
                  args=(f"{prefix}_sw", f"{prefix}_sn", f"{prefix}_sr",
                        f"{prefix}_sk", mag_key))
        if st.session_state.get("_bal_tw_msg"):
            st.caption("✅ " + st.session_state["_bal_tw_msg"])


def _machine_and_planes(key: str):
    opts = _instance_options()
    if not opts:
        st.info("No machines available.")
        return None, []
    iid = st.selectbox("Machine", [o[0] for o in opts],
                       format_func=lambda x: dict(opts).get(x, x), key=f"{key}_iid")
    from core.balance.live_source import list_balance_planes
    planes = list_balance_planes(iid)
    if not planes:
        st.warning("This machine has no radial proximity probes configured.")
    return iid, planes


def _plane_label(p) -> str:
    return f"[{p['section']}] {p['plane_label']} (plane {p['plane']})"


# =====================================================================
# Tabs
# =====================================================================
tab_tw, tab_1p, tab_2p, tab_iso, tab_rep = st.tabs([
    "Trial weight", "1 plane", "2 planes", "ISO validation", "Report",
])


# ---------------------------------------------------------------------
# 1) Peso de prueba (API 684)
# ---------------------------------------------------------------------
with tab_tw:
    bal_section_header("Trial weight", "Starting mass to produce a measurable "
                       "vector change.", "API 684 · Umax = 6350·W/N", "⚙️")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            W_plane = _num("tw_wplane", "Plane weight W [kg]", 3500.0,
                           min_value=0.0, step=10.0, format="%.1f")
        with c2:
            rpm = _num("tw_rpm", "Speed N [rpm]", 3600.0,
                       min_value=0.0, step=10.0, format="%.0f")
        with c3:
            radius = _num("tw_radius", "Correction radius [mm]", 420.0,
                          min_value=0.0, step=1.0, format="%.1f")
        with c4:
            k = _num("tw_k", "Factor k (0.2–2.0)", 1.25,
                     min_value=0.2, max_value=2.0, step=0.05, format="%.2f")

    Wtrial, Utrial = recommend_trial_weight_g(W_plane, rpm, radius, k)
    Umax = umax_api684_gmm(W_plane, rpm)
    bal_kpi_row([
        (f"{Wtrial:,.2f} g", "Trial weight", "recommended @ given radius", "cyan"),
        (f"{Umax:,.0f}", "Umax [g·mm]", "API 684", "navy"),
        (f"{Utrial:,.0f}", "Trial U [g·mm]", f"k = {k:.2f}", "navy"),
    ])
    st.caption("Adjust the weight to the available mass/thread and confirm it "
               "produces a measurable vector change before computing the correction.")


# ---------------------------------------------------------------------
# 2) Balanceo en 1 plano
# ---------------------------------------------------------------------
with tab_1p:
    bal_section_header("Single-plane balancing",
                       "H = (Vt − V0) / Wt  ·  Wcorr = −V0 / H",
                       "ISO 21940-12 · influence coefficient", "🎯")
    # Rotor 3D fijo (imagen) SIEMPRE arriba: la vibración medida (V0) aparece
    # apenas se carga el dato; el contrapeso, al calcular.
    _r1prev = st.session_state.get("bal_r1p")
    _v0 = st.session_state.get("b1_v0_mag")
    _vib1 = (_v0, st.session_state.get("b1_v0_ang") or 0.0) if _v0 else None
    _planes1 = build_planes_1p(
        _vib1, st.session_state.get("b1_unit", "µm pk-pk"),
        _r1prev["corr_ang_deg"] if _r1prev else None,
        f"{_r1prev['corr_mass_g']:.1f} g" if _r1prev else "")
    st.markdown(rotor_face_svg(_planes1, rotation=st.session_state.get("b1_rot", "CCW")),
                unsafe_allow_html=True)
    st.caption("🔴 Measured vibration (V0)   ·   🔷 Correction weight to install (appears on calculation)")

    top = st.columns([1, 1, 1])
    with top[0]:
        unit1 = st.selectbox("Vibration unit", UNITS, key="b1_unit")
    with top[1]:
        st.selectbox("Rotation direction", ["CCW", "CW"], key="b1_rot",
                     help="Orients the angular scale against rotation (balancing "
                          "convention). Does not affect the calculation.")
    with top[2]:
        source1 = st.radio("Data source", ["Manual", "Live Monitoring"],
                           key="b1_source", horizontal=True)

    if source1 == "Live Monitoring":
        with st.container(border=True):
            iid, planes = _machine_and_planes("b1_live")
            if planes:
                cpa, cpb = st.columns([2, 1])
                with cpa:
                    p_idx = st.selectbox("Plane to balance", list(range(len(planes))),
                                         format_func=lambda i: _plane_label(planes[i]),
                                         key="b1_live_plane")
                with cpb:
                    direction = st.radio("Direction", ["Y", "X"], key="b1_live_dir",
                                         horizontal=True)
                from core.balance.live_source import pick_sensor_for_plane
                sensor = pick_sensor_for_plane(planes[p_idx], direction)
                st.caption(f"Selected probe: **{sensor or '—'}**  ·  "
                           "capture 1X on each run.")
                b1, b2, b3 = st.columns(3)
                b1.button("Capture V0", key="b1_cap_v0", use_container_width=True,
                          on_click=_bal_capture_cb,
                          args=(iid, [(sensor, "b1_v0_mag", "b1_v0_ang")]))
                b2.button("Capture Vt", key="b1_cap_vt", use_container_width=True,
                          on_click=_bal_capture_cb,
                          args=(iid, [(sensor, "b1_vt_mag", "b1_vt_ang")]))
                b3.button("Capture Vf", key="b1_cap_vf", use_container_width=True,
                          on_click=_bal_capture_cb,
                          args=(iid, [(sensor, "b1_vf_mag", "b1_vf_ang")]))
                if st.session_state.get("_bal_msg"):
                    st.caption("📡 " + st.session_state["_bal_msg"])

    _trial_weight_suggester("b1_tw", "b1_tw_mag")

    with st.container(border=True):
        colA, colB, colC = st.columns(3)
        with colA:
            v0m, v0a = _vector_inputs("b1_v0", "V0 — initial vibration", unit1)
        with colB:
            twm, twa = _vector_inputs("b1_tw", "Trial weight [g]", "g")
        with colC:
            vtm, vta = _vector_inputs("b1_vt", "Vt — with trial weight", unit1)

    if st.button("Calculate single-plane balancing", key="b1_calc", type="primary"):
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
            (f"{r['corr_mass_g']:,.2f} g", "Correction weight", "mass to install", "cyan"),
            (f"{r['corr_ang_deg']:,.1f}°", "Angle", "angular position", "cyan"),
            (f"{r['pred_mag']:,.3f}", f"Residual [{unit1}]", "estimated vibration", "green"),
        ])
        sev, detail, tag = _quality_severity(r["quality"])
        bal_status_banner(f"Model quality: {tag}", f"{detail}. {r['note']}", sev)
        with st.expander("Validate against final measurement (optional)"):
            vfm, _vfa = _vector_inputs("b1_vf", "Vf — measured final vibration", unit1)
            if vfm > 0:
                bal_kpi_row([(f"{pct_reduction(v0m, vfm):,.1f} %",
                              "Vibration reduction", "V0 → Vf", "green")])


# ---------------------------------------------------------------------
# 3) Balanceo en 2 planos
# ---------------------------------------------------------------------
with tab_2p:
    bal_section_header("Two-plane balancing",
                       "2×2 influence coefficient matrix · runs "
                       "0 (initial) · 1 (trial A) · 2 (trial B).",
                       "ISO 21940-12", "🎯")
    # Rotor 3D fijo (imagen) SIEMPRE arriba: vibración inicial (A0/B0) aparece al
    # cargar los datos; los contrapesos, al calcular.
    _r2prev = st.session_state.get("bal_r2p")
    _a0 = st.session_state.get("b2_a0_mag")
    _b0 = st.session_state.get("b2_b0_mag")
    _vibA = (_a0, st.session_state.get("b2_a0_ang") or 0.0) if _a0 else None
    _vibB = (_b0, st.session_state.get("b2_b0_ang") or 0.0) if _b0 else None
    _u2 = st.session_state.get("b2_unit", "µm pk-pk")
    if _r2prev:
        _wam, _waa = to_polar(_r2prev["WA_corr"])
        _wbm, _wba = to_polar(_r2prev["WB_corr"])
        _planes2 = build_planes_2p(_vibA, _vibB, _u2, _waa, f"{_wam:.1f} g",
                                   _wba, f"{_wbm:.1f} g")
    else:
        _planes2 = build_planes_2p(_vibA, _vibB, _u2, None, "", None, "")
    st.markdown(rotor_face_svg(_planes2, rotation=st.session_state.get("b2_rot", "CCW")),
                unsafe_allow_html=True)
    st.caption("🔴 Initial vibration (A0/B0)   ·   🔷 Correction weights (appear on calculation)")

    top = st.columns([1, 1, 1])
    with top[0]:
        unit2 = st.selectbox("Vibration unit", UNITS, key="b2_unit")
    with top[1]:
        st.selectbox("Rotation direction", ["CCW", "CW"], key="b2_rot",
                     help="Orients the angular scale against rotation (balancing "
                          "convention). Does not affect the calculation.")
    with top[2]:
        source2 = st.radio("Data source", ["Manual", "Live Monitoring"],
                           key="b2_source", horizontal=True)

    if source2 == "Live Monitoring":
        with st.container(border=True):
            iid, planes = _machine_and_planes("b2_live")
            if planes and len(planes) >= 2:
                c1, c2, c3 = st.columns(3)
                with c1:
                    ia = st.selectbox("Plane A (coupling side)", list(range(len(planes))),
                                      format_func=lambda i: _plane_label(planes[i]),
                                      key="b2_live_planeA")
                with c2:
                    st.session_state.setdefault("b2_live_planeB", 1 if len(planes) > 1 else 0)
                    ib = st.selectbox("Plane B (free side)", list(range(len(planes))),
                                      format_func=lambda i: _plane_label(planes[i]),
                                      key="b2_live_planeB")
                with c3:
                    direction = st.radio("Direction (both planes)", ["Y", "X"],
                                         key="b2_live_dir", horizontal=True)
                from core.balance.live_source import pick_sensor_for_plane
                sA = pick_sensor_for_plane(planes[ia], direction)
                sB = pick_sensor_for_plane(planes[ib], direction)
                st.caption(f"Probes: A = **{sA or '—'}** · B = **{sB or '—'}** "
                           f"(same direction {direction})")
                b1, b2, b3 = st.columns(3)
                b1.button("Capture run 0 (A0,B0)", key="b2_cap0",
                          use_container_width=True, on_click=_bal_capture_cb,
                          args=(iid, [(sA, "b2_a0_mag", "b2_a0_ang"),
                                      (sB, "b2_b0_mag", "b2_b0_ang")]))
                b2.button("Capture run 1 (A1,B1)", key="b2_cap1",
                          use_container_width=True, on_click=_bal_capture_cb,
                          args=(iid, [(sA, "b2_a1_mag", "b2_a1_ang"),
                                      (sB, "b2_b1_mag", "b2_b1_ang")]))
                b3.button("Capture run 2 (A2,B2)", key="b2_cap2",
                          use_container_width=True, on_click=_bal_capture_cb,
                          args=(iid, [(sA, "b2_a2_mag", "b2_a2_ang"),
                                      (sB, "b2_b2_mag", "b2_b2_ang")]))
                if st.session_state.get("_bal_msg"):
                    st.caption("📡 " + st.session_state["_bal_msg"])

    with st.container(border=True):
        st.markdown("**Run 0 — initial**")
        c1, c2 = st.columns(2)
        with c1:
            a0m, a0a = _vector_inputs("b2_a0", "A0 — plane A probe", unit2)
        with c2:
            b0m, b0a = _vector_inputs("b2_b0", "B0 — plane B probe", unit2)
    _trial_weight_suggester("b2_wa", "b2_wa_mag", "Suggest trial weight · plane A (API 684)")
    with st.container(border=True):
        st.markdown("**Run 1 — trial weight on plane A**")
        c1, c2, c3 = st.columns(3)
        with c1:
            wam, waa = _vector_inputs("b2_wa", "Trial plane A [g]", "g")
        with c2:
            a1m, a1a = _vector_inputs("b2_a1", "A1 — probe A", unit2)
        with c3:
            b1m, b1a = _vector_inputs("b2_b1", "B1 — probe B", unit2)
    _trial_weight_suggester("b2_wb", "b2_wb_mag", "Suggest trial weight · plane B (API 684)")
    with st.container(border=True):
        st.markdown("**Run 2 — trial weight on plane B**")
        c1, c2, c3 = st.columns(3)
        with c1:
            wbm, wba = _vector_inputs("b2_wb", "Trial plane B [g]", "g")
        with c2:
            a2m, a2a = _vector_inputs("b2_a2", "A2 — probe A", unit2)
        with c3:
            b2m, b2a = _vector_inputs("b2_b2", "B2 — probe B", unit2)

    if st.button("Calculate two-plane balancing", key="b2_calc", type="primary"):
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
            (f"{wa_mag:,.2f} g", "Plane A correction", f"∠ {wa_ang:,.1f}°", "cyan"),
            (f"{wb_mag:,.2f} g", "Plane B correction", f"∠ {wb_ang:,.1f}°", "cyan"),
            (f"{abs(r['A_after']):,.3f}", f"Residual A [{unit2}]", "estimated", "green"),
            (f"{abs(r['B_after']):,.3f}", f"Residual B [{unit2}]", "estimated", "green"),
        ])
        sev, detail, tag = _quality_severity(r["quality"])
        bal_status_banner(f"Model quality: {tag}",
                          f"{detail}. cond(M) = {r['cond']:.1f}. {r['note']}", sev)


# ---------------------------------------------------------------------
# 4) Validación ISO 21940-11
# ---------------------------------------------------------------------
with tab_iso:
    bal_section_header("ISO validation",
                       "e_per = 9549·G/N  ·  U_per = e_per·W. The residual is "
                       "evaluated against ISO 21940 grades.",
                       "ISO 21940-11", "✅")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            W_iso = _num("iso_w", "Rotor weight W [kg]", 11000.0,
                         min_value=0.0, step=10.0, format="%.1f")
            rpm_iso = _num("iso_rpm", "Speed N [rpm]", 3600.0,
                           min_value=0.0, step=10.0, format="%.0f")
        with c2:
            modo = st.radio("Residual U_res", ["Ingresar U_res [g·mm]",
                                               "Calcular de masa·radio"], key="iso_mode")
            if modo.startswith("Ingresar"):
                U_res = _num("iso_ures", "U_res [g·mm]", 0.0, min_value=0.0,
                             step=1.0, format="%.1f")
            else:
                mr = _num("iso_resmass", "Residual mass [g]", 0.0, min_value=0.0,
                          step=0.1, format="%.2f")
                rr = _num("iso_resrad", "Radius [mm]", 420.0, min_value=0.0,
                          step=1.0, format="%.1f")
                U_res = calc_U_trial(mr, rr)
                st.caption(f"U_res computed = **{U_res:,.1f} g·mm**")

    ev = evaluate_iso_grades(W_iso, rpm_iso, U_res)
    st.session_state["bal_iso"] = ev
    if ev["status_code"] == "FAIL":
        bal_status_banner("Does not comply", ev["summary_label"], "fail")
    elif ev["best_grade"] is not None and ev["best_grade"] <= 2.5:
        bal_status_banner("Complies", ev["summary_label"], "ok")
    else:
        bal_status_banner("Complies (basic quality)", ev["summary_label"], "warning")

    rows = []
    for g in ev["results"]:
        rows.append({
            "Grade": f"G{g['G']:g}",
            "e_per [µm]": round(g["e_per"], 3),
            "U_per [g·mm]": round(g["U_per"], 1),
            "U_res/U_per": round(g["ratio"], 2) if g["ratio"] < 900 else "—",
            "Complies": "✅" if g["pass"] else "❌",
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------
# 5) Reporte
# ---------------------------------------------------------------------
with tab_rep:
    bal_section_header("Report", "Session summary and branded "
                       "Watermelon/SIGA PDF.", "ISO 21940 · API 684", "⎙")

    r1 = st.session_state.get("bal_r1p")
    r2 = st.session_state.get("bal_r2p")
    ev = st.session_state.get("bal_iso")

    lines: List[str] = []
    if r1:
        lines.append(f"1 plane · correction {r1['corr_mass_g']:.2f} g @ "
                     f"{r1['corr_ang_deg']:.1f}° · residual {r1['pred_mag']:.3f} "
                     f"· {r1['quality']}")
    if r2:
        wa_m, wa_a = to_polar(r2["WA_corr"])
        wb_m, wb_a = to_polar(r2["WB_corr"])
        lines.append(f"2 planes · A: {wa_m:.2f} g @ {wa_a:.1f}° · "
                     f"B: {wb_m:.2f} g @ {wb_a:.1f}° · {r2['quality']}")
    if ev:
        lines.append(f"ISO validation · {ev['summary_label']}")

    if lines:
        with st.container(border=True):
            for ln in lines:
                st.markdown(f"- {ln}")
    else:
        st.info("No calculations in this session yet. Run a balancing in the "
                "previous tabs.")

    st.markdown("")
    bal_section_header("Report data")

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
            rep_asset = st.text_input("Asset", key="rep_asset")
            rep_client = st.text_input("Client", key="rep_client")
        with c2:
            rep_location = st.text_input("Site / location", key="rep_location")
            rep_specialist = st.text_input("Specialist", key="rep_specialist")
        with c3:
            rep_date = st.text_input("Date", key="rep_date")
            rep_notes = st.text_area("Notes", key="rep_notes", height=80)

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
        st.info("Run at least one balancing or ISO validation to generate the PDF.")
    else:
        if st.button("Generate PDF report", key="rep_pdf", type="primary"):
            try:
                from core.balance.report import build_balance_pdf
                meta = {
                    "asset": rep_asset, "client": rep_client, "location": rep_location,
                    "specialist": rep_specialist, "report_date": rep_date,
                    "unit": (st.session_state.get("b1_unit")
                             or st.session_state.get("b2_unit") or "µm pk-pk"),
                    "rpm": (st.session_state.get("iso_rpm")
                            or st.session_state.get("tw_rpm")),
                    "rotation": (st.session_state.get("b1_rot")
                                 or st.session_state.get("b2_rot") or "CCW"),
                    "notes": rep_notes,
                }
                st.session_state["bal_pdf"] = build_balance_pdf(
                    meta=meta, one_plane=one_plane, two_plane=two_plane, iso=ev)
            except Exception as e:  # noqa: BLE001
                st.error(f"Error generating the PDF: {e}")
                st.session_state.pop("bal_pdf", None)
        if st.session_state.get("bal_pdf"):
            import re as _re
            _fn = "Balanceo_" + _re.sub(r"[^A-Za-z0-9]+", "_",
                                        (rep_asset or "activo")).strip("_") + ".pdf"
            st.download_button("⬇ Download PDF", data=st.session_state["bal_pdf"],
                               file_name=_fn, mime="application/pdf")


bal_footer_norms()
