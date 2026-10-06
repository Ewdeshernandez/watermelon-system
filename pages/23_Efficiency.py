"""
pages/23_Efficiency.py — Módulo Efficiency Watermelon
=====================================================

Eficiencia de máquinas rotatorias (motores, ventiladores, bombas, compresores,
turbinas) integrada a Watermelon. Motor de cálculo: `core.efficiency.engine`
(numpy puro, el MISMO que usa la app de campo). Esta página es solo UI + wiring;
NO contiene matemática de eficiencia.

Flujo
-----
1. Analizar — manual o cargando una corrida de campo de la nube; calcula
   P_mec, P_elec, η_motor, η_operativa (semáforo por norma) y η de proceso
   según el tipo de máquina.
2. Reporte — PDF branded Watermelon/SIGA.

Normas: IEC 60034-2 · ISO 5801 · ISO 9906 · ASME PTC 10 · IEC 60041 · ISO 20816.
Acceso: analista / admin (gateada; el cliente no la ve, recibe el reporte).
"""
from __future__ import annotations

import streamlit as st

from core.auth import (
    require_login, render_user_menu, get_current_user, is_page_allowed_for_role,
)
from core.ui_theme import apply_watermelon_page_style
from core.ui_industrial import inject_industrial_css, industrial_band, dot, html_table
from core.efficiency.engine import MACHINE_TYPES, EfficiencyInputs, compute

st.set_page_config(page_title="Watermelon System | Efficiency", page_icon="⚡", layout="wide")

require_login()
render_user_menu()
apply_watermelon_page_style()
inject_industrial_css()

st.markdown("""<style>
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"]{display:none !important;}
div[data-testid="stNumberInput"] input{text-align:left;}
</style>""", unsafe_allow_html=True)

_user = get_current_user() or {}
_role = str(_user.get("role", "")).lower()
if not is_page_allowed_for_role("pages/23_Efficiency.py", _role):
    st.error("Your role does not have access to this module.")
    st.stop()

# Mapas de tipo ↔ etiqueta/norma (del motor, única fuente de verdad).
_TYPE_CODES = [c for c, *_ in MACHINE_TYPES]
_TYPE_LABEL = {c: f"{es} · {norm}" for c, en, es, norm in MACHINE_TYPES}
_PROC_FIELDS = {"pump": "flow_head", "hydro": "flow_head", "fan": "flow_dp", "compressor": "comp"}
_SEMA = {"green": "ok", "amber": "warn", "red": "dang"}


industrial_band("WATERMELON · EFFICIENCY",
                "Eficiencia de máquinas rotatorias",
                "Potencia mecánica (TorqueTrak 10K + keyphasor) vs. eléctrica y de diseño. "
                "Semáforo operativo + eficiencia de proceso por tipo. IEC 60034-2 · ISO 5801 · "
                "ISO 9906 · ASME PTC 10 · IEC 60041.")

tab_an, tab_rp = st.tabs(["⚡  Analizar", "⎙  Reporte"])


# =====================================================================
# Carga de corrida de campo desde la nube (opcional)
# =====================================================================
def _load_from_cloud():
    """Selector de corridas de eficiencia subidas por el campo. Pre-llena inputs."""
    try:
        from core.efficiency import cloud
        runs = cloud.list_runs()
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Nube no disponible: {exc}")
        return
    if not runs:
        st.caption("No hay corridas de campo subidas todavía.")
        return
    opts = {f"{r.get('name','run')} · {r.get('updated_at','')[:16]} · {r.get('client','')}": r["id"]
            for r in runs}
    sel = st.selectbox("Corrida de campo (nube)", ["—"] + list(opts.keys()), key="eff_cloud_sel")
    if sel != "—" and st.button("Cargar corrida", key="eff_cloud_load"):
        payload = cloud.load_run(opts[sel])
        if not payload:
            st.warning("No se pudo cargar la corrida.")
            return
        inp = payload.get("inputs") or {}
        setup = payload.get("setup") or {}
        for k, v in {"eff_type": inp.get("machine_type") or setup.get("machine_type") or "motor",
                     "eff_torque": inp.get("torque_nm", 0.0), "eff_rpm": inp.get("rpm", 0.0),
                     "eff_v": inp.get("voltage_v", 0.0), "eff_i": inp.get("current_a", 0.0),
                     "eff_pf": inp.get("power_factor", 0.92), "eff_phases": str(inp.get("phases", 3)),
                     "eff_design": inp.get("design_power_kw", setup.get("design_power_kw", 0.0)),
                     "eff_flow": inp.get("flow_m3s", 0.0), "eff_head": inp.get("head_m", 0.0),
                     "eff_dp": inp.get("dp_pa", 0.0), "eff_rho": inp.get("rho", 1000.0),
                     "eff_mdot": inp.get("mdot_kg_s", 0.0), "eff_cp": inp.get("cp_kj_kgk", 1.005),
                     "eff_tin": inp.get("t_in_k", 293.15), "eff_pr": inp.get("pressure_ratio", 2.0),
                     "eff_k": inp.get("k_ratio", 1.4),
                     "eff_machine": setup.get("machine", ""), "eff_client": setup.get("client", ""),
                     "eff_location": setup.get("location", "")}.items():
            st.session_state[k] = v
        st.success(f"Corrida cargada: {sel}")
        st.rerun()


with tab_an:
    with st.expander("☁ Cargar corrida de campo desde la nube", expanded=False):
        _load_from_cloud()

    c_id1, c_id2, c_id3 = st.columns(3)
    machine = c_id1.text_input("Máquina / activo", key="eff_machine")
    client = c_id2.text_input("Cliente", key="eff_client")
    location = c_id3.text_input("Ubicación", key="eff_location")

    c1, c2 = st.columns([2, 1])
    mtype = c1.selectbox("Tipo de máquina", _TYPE_CODES,
                         format_func=lambda c: _TYPE_LABEL.get(c, c), key="eff_type")
    phases = c2.selectbox("Fases", ["3", "1"], key="eff_phases")

    st.markdown("**Mecánico (eje)**")
    m1, m2, m3 = st.columns(3)
    torque = m1.number_input("Par [N·m]", min_value=0.0, value=0.0, step=1.0, format="%.2f", key="eff_torque")
    rpm = m2.number_input("RPM", min_value=0.0, value=0.0, step=1.0, format="%.1f", key="eff_rpm")
    design = m3.number_input("Potencia de diseño [kW]", min_value=0.0, value=0.0, step=1.0, format="%.1f", key="eff_design")

    st.markdown("**Eléctrico (analizador de red)**")
    e1, e2, e3 = st.columns(3)
    volt = e1.number_input("Tensión V_LL [V]", min_value=0.0, value=0.0, step=1.0, format="%.1f", key="eff_v")
    curr = e2.number_input("Corriente [A]", min_value=0.0, value=0.0, step=1.0, format="%.1f", key="eff_i")
    pf = e3.number_input("cosφ", min_value=0.0, max_value=1.0, value=0.92, step=0.01, format="%.3f", key="eff_pf")

    # Parámetros de proceso según tipo
    proc = _PROC_FIELDS.get(mtype)
    flow = head = dp = rho = mdot = cp = tin = pr = k = 0.0
    rho = 1000.0; cp = 1.005; tin = 293.15; pr = 2.0; k = 1.4
    if proc == "flow_head":
        st.markdown("**Proceso (hidráulico)**")
        p1, p2, p3 = st.columns(3)
        flow = p1.number_input("Caudal Q [m³/s]", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="eff_flow")
        head = p2.number_input("Altura H [m]", min_value=0.0, value=0.0, step=0.1, format="%.2f", key="eff_head")
        rho = p3.number_input("Densidad ρ [kg/m³]", min_value=0.01, value=1000.0, step=1.0, format="%.3f", key="eff_rho")
    elif proc == "flow_dp":
        st.markdown("**Proceso (aire)**")
        p1, p2 = st.columns(2)
        flow = p1.number_input("Caudal Q [m³/s]", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="eff_flow")
        dp = p2.number_input("Presión total Δp [Pa]", min_value=0.0, value=0.0, step=1.0, format="%.1f", key="eff_dp")
    elif proc == "comp":
        st.markdown("**Proceso (compresor isentrópico)**")
        p1, p2, p3 = st.columns(3)
        mdot = p1.number_input("Flujo másico ṁ [kg/s]", min_value=0.0, value=0.0, step=0.1, format="%.3f", key="eff_mdot")
        cp = p2.number_input("Cp [kJ/kg·K]", min_value=0.1, value=1.005, step=0.01, format="%.4f", key="eff_cp")
        tin = p3.number_input("T entrada T1 [K]", min_value=1.0, value=293.15, step=1.0, format="%.2f", key="eff_tin")
        p4, p5 = st.columns(2)
        pr = p4.number_input("Relación de presión π", min_value=1.0, value=2.0, step=0.1, format="%.3f", key="eff_pr")
        k = p5.number_input("k = Cp/Cv", min_value=1.0, max_value=2.0, value=1.4, step=0.01, format="%.3f", key="eff_k")

    if torque > 0 and rpm > 0:
        res = compute(EfficiencyInputs(
            machine_type=mtype, torque_nm=torque, rpm=rpm, voltage_v=volt, current_a=curr,
            power_factor=pf, phases=int(phases), design_power_kw=design, flow_m3s=flow, head_m=head,
            dp_pa=dp, rho=rho, mdot_kg_s=mdot, cp_kj_kgk=cp, t_in_k=tin, pressure_ratio=pr, k_ratio=k))
        st.session_state["eff_result"] = {
            "machine": machine, "client": client, "location": location, "mtype": mtype,
            "torque": torque, "rpm": rpm, "volt": volt, "curr": curr, "pf": pf, "design": design,
            "p_mec": res.p_mec_kw, "p_elec": res.p_elec_kw, "eta_motor": res.eta_motor_pct,
            "eta_op": res.eta_operational_pct, "diag_code": res.diagnosis.code,
            "diag_es": res.diagnosis.label_es, "diag_color": res.diagnosis.color,
            "proc_label": res.process_label, "eta_proc": res.eta_process_pct, "p_proc": res.process_power_kw}

        st.markdown("###  Resultado")
        kc = st.columns(4)
        kc[0].metric("P_mec", f"{res.p_mec_kw:,.1f} kW")
        kc[1].metric("P_elec", f"{res.p_elec_kw:,.1f} kW" if res.p_elec_kw > 0 else "—")
        kc[2].metric("η motor", f"{res.eta_motor_pct:,.1f} %" if res.eta_motor_pct > 0 else "—")
        kc[3].metric("η operativa", f"{res.eta_operational_pct:,.1f} %" if res.eta_operational_pct > 0 else "—")

        if res.eta_operational_pct > 0:
            st.markdown(f"{dot(_SEMA.get(res.diagnosis.color, 'off'))}  "
                        f"**{res.diagnosis.label_es}** — η operativa {res.eta_operational_pct:,.1f}%",
                        unsafe_allow_html=True)

        rows = [["Tipo de máquina", _TYPE_LABEL.get(mtype, mtype)],
                ["Par · RPM", f"{torque:,.2f} N·m · {rpm:,.1f} rpm"],
                ["Potencia mecánica P_mec", f"{res.p_mec_kw:,.2f} kW"]]
        if res.p_elec_kw > 0:
            rows += [["V · I · cosφ", f"{volt:,.0f} V · {curr:,.0f} A · {pf:.3f}"],
                     ["Potencia eléctrica P_elec", f"{res.p_elec_kw:,.2f} kW"],
                     ["η motor (IEC 60034-2)", f"{res.eta_motor_pct:,.1f} %"]]
        if design > 0:
            rows += [["Potencia de diseño", f"{design:,.1f} kW"],
                     ["η operativa", f"{res.eta_operational_pct:,.1f} % ({res.diagnosis.label_es})"]]
        if res.process_label:
            _pl = {"pump": "η bomba (hidráulica)", "fan": "η ventilador (aire)",
                   "hydro": "η turbina", "compressor": "η compresor (isentrópica)"}.get(res.process_label, "η proceso")
            rows += [[_pl, f"{res.eta_process_pct:,.1f} %"]]
            if res.process_power_kw > 0:
                rows += [["Potencia de proceso", f"{res.process_power_kw:,.2f} kW"]]
        html_table(["Ítem", "Valor"], rows)
    else:
        st.info("Ingresa par y RPM (y lo eléctrico/proceso según aplique) para calcular la eficiencia.")


# =====================================================================
# Reporte PDF
# =====================================================================
with tab_rp:
    r = st.session_state.get("eff_result")
    if not r:
        st.info("Calcula una eficiencia en la pestaña **Analizar** primero.")
    else:
        st.markdown(f"**{r['machine'] or '—'}** · {r['client'] or '—'} · {r['location'] or '—'}")
        st.caption(f"{_TYPE_LABEL.get(r['mtype'], r['mtype'])} · η operativa "
                   f"{r['eta_op']:,.1f}% ({r['diag_es']})" if r["eta_op"] > 0
                   else _TYPE_LABEL.get(r["mtype"], r["mtype"]))
        if st.button("📄 Generar reporte (PDF)", type="primary"):
            from datetime import date as _date
            _norm = next((norm for c, en, es, norm in MACHINE_TYPES if c == r["mtype"]), "")
            meta = {"title": "Reporte de Eficiencia", "asset": r["machine"] or "—",
                    "client": r["client"] or "—", "machine_type": _TYPE_LABEL.get(r["mtype"], r["mtype"]),
                    "location": r["location"] or "—", "test_type": "Eficiencia de máquina rotatoria",
                    "rpm": f"{r['rpm']:,.0f}", "technician": _user.get("name", "—"),
                    "reviewer": "—", "date": _date.today().isoformat(), "equipment": "Watermelon Efficiency"}
            _go = "GO" if r["diag_code"] == "normal" else "REVIEW"
            quality = [("Eficiencia operativa", _go,
                        f"{r['eta_op']:,.1f}% — {r['diag_es']}" if r["eta_op"] > 0 else "Sin potencia de diseño")]
            if r["eta_motor"] > 0:
                quality.append(("Eficiencia del motor", "GO" if r["eta_motor"] >= 90 else "REVIEW",
                                f"{r['eta_motor']:,.1f}%"))
            if r["proc_label"]:
                quality.append(("Eficiencia de proceso", "GO" if r["eta_proc"] >= 60 else "REVIEW",
                                f"{r['eta_proc']:,.1f}%"))
            rows = [["Tipo de máquina", _TYPE_LABEL.get(r["mtype"], r["mtype"])],
                    ["Par", f"{r['torque']:,.2f} N·m"], ["RPM", f"{r['rpm']:,.1f}"],
                    ["Potencia mecánica P_mec", f"{r['p_mec']:,.2f} kW"]]
            if r["p_elec"] > 0:
                rows += [["V · I · cosφ", f"{r['volt']:,.0f} V · {r['curr']:,.0f} A · {r['pf']:.3f}"],
                         ["Potencia eléctrica P_elec", f"{r['p_elec']:,.2f} kW"],
                         ["η motor", f"{r['eta_motor']:,.1f} %"]]
            rows += [["Potencia de diseño", f"{r['design']:,.1f} kW"],
                     ["η operativa", f"{r['eta_op']:,.1f} % ({r['diag_es']})" if r["eta_op"] > 0 else "—"]]
            if r["proc_label"]:
                rows += [["η proceso", f"{r['eta_proc']:,.1f} %"]]
                if r["p_proc"] > 0:
                    rows += [["Potencia de proceso", f"{r['p_proc']:,.2f} kW"]]
            sections = [{"title": "Resultado de eficiencia",
                         "table": {"headers": ["Ítem", "Valor"], "rows": rows}}]
            findings = []
            if r["eta_op"] > 0:
                findings.append(f"Eficiencia operativa {r['eta_op']:,.1f}% → {r['diag_es']}.")
            if r["eta_motor"] > 0:
                findings.append(f"Eficiencia del motor {r['eta_motor']:,.1f}% "
                                f"(P_mec {r['p_mec']:,.1f} kW / P_elec {r['p_elec']:,.1f} kW).")
            if r["proc_label"]:
                findings.append(f"Eficiencia de proceso ({r['proc_label']}) {r['eta_proc']:,.1f}%.")
            analysis = ["P_mec = T·ω/1000 · P_elec = √3·V·I·cosφ/1000 · "
                        "η_motor = P_mec/P_elec · η_op = P_mec/P_diseño. Norma: " + _norm + "."]
            recs = ["Comparar contra la eficiencia de placa y la curva de diseño del punto de operación.",
                    "Si está bajo la banda, revisar carga, alineación, ensuciamiento y punto de operación."]
            try:
                from core.modal.preliminary_report import build_preliminary_pdf
                pdf = build_preliminary_pdf(meta=meta, quality=quality, sections=sections,
                                            analysis=analysis, findings=findings, recommendations=recs,
                                            run_id=f"EFF-{meta['asset']}", lang="es")
                st.download_button("⬇ Descargar PDF", data=pdf,
                                   file_name=f"Eficiencia_{r['machine'] or 'equipo'}.pdf",
                                   mime="application/pdf")
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")
