"""
core.balance.engine
===================

Motor de cálculo de balanceo de rotores. Matemática PURA (solo numpy) —
sin Streamlit, sin I/O, sin estado global. Testeable y reutilizable.

Origen: extraído sin cambios de ROTORIX (app propia de balanceo, validada en
campo contra balanceos reales — reportes Termosuria Tes1). Las fórmulas están
marcadas donde corresponde; **no modificar sin re-validar contra campo**.

Normas
------
- ISO 21940-11: desbalance residual permisible (e_per = 9549·G/N; U_per = e_per·W).
- ISO 21940-12: balanceo multiplano por coeficiente de influencia (1 y 2 planos).
- API 684: peso de prueba (Umax = 6350·W/N).

Convención angular: "CAMPO" (0° arriba / TDC, ángulo positivo según el giro
definido por el operador). Los vectores de vibración y de peso se manejan como
números complejos en el plano; magnitud = módulo, ángulo = fase en grados.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


# =========================================================
# Helpers complejos / polar
# =========================================================
def to_complex(mag: float, ang_deg: float) -> complex:
    """Vector polar (magnitud, ángulo°) → número complejo."""
    return float(mag) * np.exp(1j * np.deg2rad(float(ang_deg)))


def to_polar(z: complex) -> Tuple[float, float]:
    """Número complejo → (magnitud, ángulo° en [0, 360))."""
    mag = float(np.abs(z))
    ang = float(np.rad2deg(np.angle(z)) % 360.0)
    return mag, ang


def norm360(a: float) -> float:
    """Normaliza un ángulo a [0, 360)."""
    return float(a) % 360.0


# Alias retro-compat con los nombres usados en ROTORIX (facilita migrar la UI).
polar_to_complex = to_complex
complex_to_polar = to_polar


# =========================================================
# Peso de prueba (API 684)
# =========================================================
def umax_api684_gmm(W_kg: float, N_rpm: float) -> float:
    """Desbalance máximo de arranque API 684 (Umax) en g·mm.

    Umax = 6350 · W[kg] / N[rpm].
    """
    if W_kg <= 0 or N_rpm <= 0:
        return 0.0
    return 6350.0 * W_kg / N_rpm


def recommend_trial_weight_g(
    W_plane_kg: float,
    rpm: float,
    radius_mm: float,
    k_base: float = 1.25,
) -> Tuple[float, float]:
    """Recomienda peso de prueba (g) y su desbalance (g·mm).

    Utrial = Umax · clamp(k_base, 0.2, 2.0);  Wtrial = Utrial / radio[mm].
    Devuelve (Wtrial_g, Utrial_gmm).
    """
    Umax = umax_api684_gmm(W_plane_kg, rpm)
    Utrial = Umax * max(0.2, min(float(k_base), 2.0))
    if radius_mm <= 0:
        return 0.0, 0.0
    Wtrial = Utrial / radius_mm
    return Wtrial, Utrial


# =========================================================
# Desbalance permisible / validación (ISO 21940-11)
# =========================================================
def calc_e_per(G: float, rpm: float) -> float:
    """Excentricidad permisible e_per [mm] = 9549 · G / N (ISO 21940-11)."""
    if rpm <= 0:
        return 0.0
    return 9549.0 * float(G) / float(rpm)


def calc_U_per(e_per_mm: float, W_kg: float) -> float:
    """Desbalance residual permisible U_per [g·mm] = e_per[mm] · W[kg]."""
    return max(0.0, float(e_per_mm) * float(W_kg))


def calc_U_trial(trial_mass_g: float, radius_mm: float) -> float:
    """Desbalance del peso de prueba [g·mm] = masa[g] · radio[mm]."""
    return max(0.0, float(trial_mass_g)) * max(0.0, float(radius_mm))


def calc_U_res_auto(v_initial: float, v_final: float, U_trial: float) -> float:
    """Estima el desbalance residual [g·mm] escalando U_trial por la relación
    de vibración final/inicial: U_res ≈ (V_final / V_inicial) · U_trial."""
    vi = max(1e-12, float(v_initial))
    vf = max(0.0, float(v_final))
    ut = max(0.0, float(U_trial))
    return (vf / vi) * ut


def pct_reduction(before: float, after: float) -> float:
    """% de reducción de vibración: (antes - después) / antes · 100."""
    b = max(1e-12, float(before))
    a = max(0.0, float(after))
    return max(0.0, (b - a) / b * 100.0)


def status_from_ratio(ratio: float) -> Tuple[str, str]:
    """Estado de cumplimiento según U_res/U_per: (etiqueta, tono)."""
    r = float(ratio)
    if r <= 0.90:
        return "CUMPLE", "ok"
    if r <= 1.00:
        return "LÍMITE", "warn"
    return "NO CUMPLE", "bad"


# Grados de calidad ISO 21940 más usados en turbomáquina / generación.
ISO_GRADES: List[float] = [0.4, 1.0, 2.5, 6.3, 16.0]


def evaluate_iso_grades(W_kg: float, N_rpm: float, U_res: float) -> Dict[str, Any]:
    """Evalúa un desbalance residual contra los grados ISO 21940.

    Para cada grado calcula e_per, U_per y si el U_res cumple. Devuelve el
    mejor grado que cumple, el primero que falla y una etiqueta resumen.
    """
    results: List[Dict[str, Any]] = []

    for G in ISO_GRADES:
        e_per = calc_e_per(G, N_rpm)
        U_per = calc_U_per(e_per, W_kg)
        passed = float(U_res) <= float(U_per) + 1e-12
        ratio = (float(U_res) / float(U_per)) if U_per > 1e-12 else 999.0
        results.append({
            "G": float(G),
            "e_per": float(e_per),
            "U_per": float(U_per),
            "pass": bool(passed),
            "ratio": float(ratio),
        })

    passed_grades = [r for r in results if r["pass"]]
    best_grade = min((r["G"] for r in passed_grades), default=None)

    failed_grades = [r for r in results if not r["pass"]]
    first_failed_grade = min((r["G"] for r in failed_grades), default=None)

    if best_grade is None:
        status_code = "FAIL"
        summary_label = "No cumple ningún grado evaluado"
    elif best_grade <= 1.0:
        status_code = "EXCELLENT"
        summary_label = f"Cumple hasta G{best_grade:g} (muy alta calidad)"
    elif best_grade <= 2.5:
        status_code = "GOOD"
        summary_label = f"Cumple hasta G{best_grade:g} (alta calidad)"
    elif best_grade <= 6.3:
        status_code = "ACCEPTABLE"
        summary_label = f"Cumple hasta G{best_grade:g} (aceptable)"
    else:
        status_code = "BASIC"
        summary_label = f"Cumple hasta G{best_grade:g} (calidad básica)"

    return {
        "W_kg": float(W_kg),
        "N_rpm": float(N_rpm),
        "U_res": float(U_res),
        "results": results,
        "best_grade": best_grade,
        "first_failed_grade": first_failed_grade,
        "status_code": status_code,
        "summary_label": summary_label,
    }


# =========================================================
# Balanceo en 1 plano — coeficiente de influencia (ISO 21940-12)
# =========================================================
def solve_1plane(
    V0_mag: float,
    V0_ang: float,
    Vt_mag: float,
    Vt_ang: float,
    trial_mass_g: float,
    trial_ang_deg: float,
) -> Dict[str, Any]:
    """Balanceo en 1 plano por coeficiente de influencia.

    H = (Vt - V0) / Wt ;  Wcorr = -V0 / H ;  Vpred = V0 + H·Wcorr.
    Entrada: vectores de vibración inicial (V0) y con peso de prueba (Vt) en
    (magnitud, ángulo°), y el peso de prueba (masa g, ángulo°).
    """
    V0 = to_complex(V0_mag, V0_ang)
    Vt = to_complex(Vt_mag, Vt_ang)
    Wt = to_complex(trial_mass_g, trial_ang_deg)

    if abs(Wt) < 1e-12:
        raise ValueError("El peso de prueba no puede ser cero.")

    delta = Vt - V0
    if abs(delta) < 1e-12:
        raise ValueError(
            "La respuesta con peso de prueba es prácticamente igual a la "
            "inicial. No hay sensibilidad suficiente."
        )

    H = delta / Wt
    Wcorr = -V0 / H
    Vpred = V0 + H * Wcorr

    corr_mass_g, corr_ang_deg = to_polar(Wcorr)
    pred_mag, pred_ang = to_polar(Vpred)

    influence_abs = abs(H)
    quality = "GOOD"
    note = "Modelo estable."

    if influence_abs < 1e-6:
        quality = "POOR"
        note = ("Sensibilidad extremadamente baja. Revise peso de prueba, "
                "radio o consistencia de medición.")
    elif influence_abs < 1e-3:
        quality = "MED"
        note = "Sensibilidad moderada. La solución puede ser sensible al ruido."

    return {
        "H": H,
        "delta": delta,
        "Wcorr": Wcorr,
        "corr_mass_g": corr_mass_g,
        "corr_ang_deg": corr_ang_deg,
        "pred_mag": pred_mag,
        "pred_ang": pred_ang,
        "quality": quality,
        "note": note,
    }


# =========================================================
# Balanceo en 2 planos — coeficiente de influencia (ISO 21940-12)
# =========================================================
def solve_2plane(
    A0: complex,
    B0: complex,
    A1: complex,
    B1: complex,
    A2: complex,
    B2: complex,
    WA_trial: complex,
    WB_trial: complex,
) -> Dict[str, Any]:
    """Balanceo en 2 planos por coeficiente de influencia (matriz 2x2).

    Corridas: 0 = inicial (A0,B0); 1 = trial en plano A (A1,B1); 2 = trial en
    plano B (A2,B2). Vectores de vibración A/B (complejos) en cada sonda.

    H_ij = ∂(sonda i)/∂(peso plano j) ; se resuelve M·Wcorr = -V0.
    Reporta det y cond de M como control de condicionamiento numérico.
    """
    eps = 1e-12

    dA_A = A1 - A0
    dB_A = B1 - B0
    dA_B = A2 - A0
    dB_B = B2 - B0

    if abs(WA_trial) < eps or abs(WB_trial) < eps:
        raise ValueError("Los pesos de prueba no pueden ser cero.")

    H_AA = dA_A / WA_trial
    H_BA = dB_A / WA_trial
    H_AB = dA_B / WB_trial
    H_BB = dB_B / WB_trial

    M = np.array([[H_AA, H_AB], [H_BA, H_BB]], dtype=complex)
    Y = np.array([[-A0], [-B0]], dtype=complex)

    det = np.linalg.det(M)
    cond = np.linalg.cond(M)

    if abs(det) < 1e-9:
        quality = "POOR"
        note = ("Matriz casi singular: respuesta muy baja o medición "
                "inconsistente (revise pesos/ángulos/ruido).")
    elif cond > 50:
        quality = "MED"
        note = ("Condicionamiento alto: la solución puede ser sensible. "
                "Revisa repetibilidad y magnitud de trial.")
    else:
        quality = "GOOD"
        note = "Modelo estable."

    # Guarda de robustez: matriz singular → no hay solución única de 2 planos.
    # (Antes np.linalg.solve lanzaba LinAlgError cruda, no atrapada por la UI.)
    if abs(det) < 1e-12:
        raise ValueError(
            "Matriz de coeficientes singular: los dos planos responden de forma "
            "casi idéntica. Revisá los pesos/ángulos de prueba y la repetibilidad "
            "de la medición — no hay solución única de balanceo en 2 planos."
        )
    try:
        sol = np.linalg.solve(M, Y)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"No se pudo resolver el sistema de 2 planos (matriz mal condicionada): {exc}"
        ) from exc
    WA_corr = sol[0, 0]
    WB_corr = sol[1, 0]

    A_after = A0 + H_AA * WA_corr + H_AB * WB_corr
    B_after = B0 + H_BA * WA_corr + H_BB * WB_corr

    return {
        "H_AA": H_AA,
        "H_AB": H_AB,
        "H_BA": H_BA,
        "H_BB": H_BB,
        "WA_corr": WA_corr,
        "WB_corr": WB_corr,
        "A_after": A_after,
        "B_after": B_after,
        "det": det,
        "cond": cond,
        "quality": quality,
        "note": note,
    }


# =========================================================
# Diagnóstico de tipo de desbalance (estático / couple / mixto)
# =========================================================
def status_level(o1: float, o2: float, alarm: float, trip: float) -> str:
    """Nivel de estado (PASS/ALARM/TRIP) según el mayor de dos overalls."""
    m = max(float(o1), float(o2))
    if m >= float(trip):
        return "TRIP"
    elif m >= float(alarm):
        return "ALARM"
    return "PASS"


def diagnose_static_couple(S_mag: float, C1_mag: float, C2_mag: float) -> Dict[str, Any]:
    """Diagnostica desbalance ESTÁTICO vs COUPLE (dinámico) vs MIXTO.

    Descompone en componente estática (S, in-phase) y de par (C1/C2). El
    ratio Cavg/S define el tipo; devuelve tipo, dominante, pureza y acciones
    recomendadas de campo.
    """
    Cavg = (float(C1_mag) + float(C2_mag)) / 2.0
    eps = 1e-9
    ratio = Cavg / max(float(S_mag), eps)
    purity = "HIGH" if (min(C1_mag, C2_mag) / max(C1_mag, C2_mag, eps)) > 0.75 else "MED"

    if ratio < 0.75:
        kind = "STATIC"
        dominant = "STATIC"
    elif ratio > 1.35:
        kind = "COUPLE"
        dominant = "COUPLE"
    else:
        kind = "MIXED"
        dominant = "COUPLE" if Cavg >= S_mag else "STATIC"

    actions: List[str] = []
    if kind == "STATIC":
        actions.append("Inicio recomendado: tratar como desbalance ESTÁTICO (componente in-phase dominante).")
        actions.append("Estrategia práctica: aplicar pesos en ambos planos al MISMO ángulo para mover el vector S.")
        actions.append("Prueba típica: trial weight en ambos planos al mismo ángulo, con magnitud similar según radios y límites.")
        actions.append("Iteración: ajustar hasta minimizar S y luego revisar si queda residual couple.")
    elif kind == "COUPLE":
        actions.append("Inicio recomendado: tratar como desbalance DINÁMICO (COUPLE).")
        actions.append("Estrategia práctica: aplicar correcciones en dos planos con ángulos aproximadamente 180° entre planos para atacar C.")
        actions.append("Prueba típica: trial weight en A y B con desfase cercano a 180° entre planos.")
        actions.append("Iteración: minimizar C1/C2 y luego revisar residual estático.")
    else:
        actions.append("Inicio recomendado: caso MIXTO, con presencia importante de S y C.")
        actions.append(f"Dominante: {dominant}. Conviene corregir primero la componente dominante y luego el residual.")
        actions.append("Si domina COUPLE, atacar primero C con dos planos y luego revisar residual estático.")
        actions.append("Si domina STATIC, atacar primero S y luego revisar residual couple.")

    checklist = [
        "Verificar convención angular CAMPO y consistencia con el giro real.",
        "Confirmar referencia de fase y estabilidad del keyphasor.",
        "Si la respuesta cambia fuertemente con RPM o carga, sospechar fenómenos adicionales distintos de desbalance.",
        "Para corrección formal definitiva, usar método por coeficientes de influencia / balanceo formal correspondiente.",
    ]

    return {
        "kind": kind,
        "dominant": dominant,
        "ratio_C_over_S": float(ratio),
        "purity": purity,
        "Cavg": float(Cavg),
        "actions": actions,
        "checklist": checklist,
    }


# =========================================================
# Guardas de validación del balanceo (auditoría — no cambian el cálculo)
# =========================================================
# Umbrales calibrados contra campo (caso Tes#1 Termosuria: el punto dominante
# respondió ~9% al trial y la corrección fue ~11× el peso de prueba → ambos
# debían marcarse). Son escala-conscientes: `cond` solo mide la FORMA de la
# matriz, no la MAGNITUD de la respuesta, por eso se chequea aparte.
TRIAL_RESP_OK = 0.30        # respuesta del punto dominante ≥30% = confiable
TRIAL_RESP_MARGINAL = 0.15  # 15–30% = marginal; <15% = no confiable
CORR_SCALE_WARN = 4.0       # |corrección| / |peso prueba|
CORR_SCALE_CRIT = 8.0
WORSEN_MARGIN = 1.05        # un plano "empeora" si final > 1.05× inicial


def pct_change(before: float, after: float) -> float:
    """% de CAMBIO con signo: + = reducción (mejora), − = aumento (empeoró).
    A diferencia de pct_reduction (que recorta a 0), este NUNCA oculta que un
    plano empeoró."""
    b = max(1e-12, float(before))
    a = max(0.0, float(after))
    return (b - a) / b * 100.0


def _dphase_deg(z0: complex, z1: complex) -> float:
    """Diferencia de fase entre dos vectores, en [0, 180]°."""
    d = abs(float(np.rad2deg(np.angle(z1) - np.angle(z0)))) % 360.0
    return min(d, 360.0 - d)


def _resp_ratio(v0: complex, v1: complex) -> float:
    """|Δvector| / |vector inicial| (sensibilidad relativa del punto)."""
    base = max(1e-9, float(abs(v0)))
    return float(abs(v1 - v0)) / base


def _w(code: str, severity: str, title: str, msg: str) -> Dict[str, str]:
    return {"code": code, "severity": severity, "title": title, "msg": msg}


def diagnose_2plane(
    result: Dict[str, Any],
    A0: complex, B0: complex, A1: complex, B1: complex,
    A2: complex, B2: complex, WA_trial: complex, WB_trial: complex,
    Vf_A: complex = None, Vf_B: complex = None,
) -> List[Dict[str, str]]:
    """Auditoría del balanceo 2 planos. Devuelve lista de avisos
    {code, severity('crit'|'warn'|'info'), title, msg}. NO altera el cálculo."""
    out: List[Dict[str, str]] = []

    # Punto dominante = el de mayor vibración inicial (el que hay que corregir).
    dom, v0_dom = ("B", B0) if abs(B0) >= abs(A0) else ("A", A0)
    if dom == "B":
        dom_resp = max(_resp_ratio(B0, B1), _resp_ratio(B0, B2))
    else:
        dom_resp = max(_resp_ratio(A0, A1), _resp_ratio(A0, A2))

    # G1 — respuesta del punto dominante al peso de prueba
    if dom_resp < TRIAL_RESP_MARGINAL:
        out.append(_w(
            "weak_trial", "crit", "Respuesta al peso de prueba muy baja",
            f"El punto dominante ({dom}, {abs(v0_dom):.3f}) cambió solo "
            f"{dom_resp*100:.0f}% con el peso de prueba (mínimo confiable "
            f"{TRIAL_RESP_OK*100:.0f}%). El coeficiente de influencia NO es "
            f"confiable y la corrección puede ser ficticia. Probablemente el "
            f"problema NO es desbalance másico."))
    elif dom_resp < TRIAL_RESP_OK:
        out.append(_w(
            "marginal_trial", "warn", "Respuesta al peso de prueba marginal",
            f"El punto dominante ({dom}) cambió {dom_resp*100:.0f}% (ideal "
            f"≥{TRIAL_RESP_OK*100:.0f}%). Usar un peso de prueba mayor o revisar "
            f"repetibilidad; la solución es sensible al ruido."))

    # G5 — corrección desproporcionada vs peso de prueba
    for plane, wc_key, wt in (("A", "WA_corr", WA_trial), ("B", "WB_corr", WB_trial)):
        wt_abs = max(1e-9, float(abs(wt)))
        scale = float(abs(result.get(wc_key, 0))) / wt_abs
        if scale >= CORR_SCALE_CRIT:
            out.append(_w(
                "corr_scale", "crit", f"Corrección plano {plane} desproporcionada",
                f"La corrección del plano {plane} es {scale:.0f}× el peso de "
                f"prueba ({float(abs(result.get(wc_key,0))):.0f} vs {wt_abs:.0f} g). "
                f"El trial fue insuficiente o el plano es insensible: no instales "
                f"esa masa a ciegas."))
        elif scale >= CORR_SCALE_WARN:
            out.append(_w(
                "corr_scale", "warn", f"Corrección plano {plane} grande",
                f"La corrección del plano {plane} es {scale:.0f}× el peso de "
                f"prueba. Considera un trial mayor y re-medir."))

    # G2 — empeoramiento por plano (medido si hay Vf; si no, predicho)
    for plane, v0, vf, after_key in (("A", A0, Vf_A, "A_after"),
                                     ("B", B0, Vf_B, "B_after")):
        measured = vf is not None and float(abs(vf)) > 0
        v_final = float(abs(vf)) if measured else float(abs(result.get(after_key, 0)))
        if measured and v_final > float(abs(v0)) * WORSEN_MARGIN:
            out.append(_w(
                "worsened", "crit", f"Plano {plane} empeoró",
                f"El plano {plane} pasó de {abs(v0):.3f} a {v_final:.3f} "
                f"({pct_change(abs(v0), v_final):+.0f}%). El balanceo degradó este "
                f"plano — no reportar como mejora global."))
        elif (not measured) and v_final > float(abs(v0)) * WORSEN_MARGIN:
            out.append(_w(
                "worsened_pred", "warn", f"Plano {plane} empeora (predicho)",
                f"El modelo predice que el plano {plane} sube de {abs(v0):.3f} a "
                f"{v_final:.3f}. Revisa antes de instalar."))

    # G4 — discriminador: síntomas no consistentes con desbalance puro
    weak = any(x["code"] == "weak_trial" for x in out)
    big = any(x["code"] == "corr_scale" and x["severity"] == "crit" for x in out)
    if weak or big:
        out.append(_w(
            "not_unbalance", "info", "¿Es realmente desbalance?",
            "Respuesta débil al trial y/o corrección desproporcionada. Antes de "
            "balancear, verifica alineación (frío/caliente), solturas mecánicas, "
            "2X/armónicos y dependencia con temperatura/tiempo. El desbalance "
            "másico no se corrige con masa si el origen es alineación o térmico."))
    return out


def diagnose_1plane(
    result: Dict[str, Any],
    V0_mag: float, V0_ang: float, Vt_mag: float, Vt_ang: float,
    trial_mass_g: float, trial_ang_deg: float,
    Vf_mag: float = None, Vf_ang: float = None,
) -> List[Dict[str, str]]:
    """Auditoría del balanceo 1 plano. Mismos criterios que 2 planos."""
    out: List[Dict[str, str]] = []
    V0 = to_complex(V0_mag, V0_ang)
    Vt = to_complex(Vt_mag, Vt_ang)
    Wt = to_complex(trial_mass_g, trial_ang_deg)

    resp = _resp_ratio(V0, Vt)
    dph = _dphase_deg(V0, Vt)
    if resp < TRIAL_RESP_MARGINAL and dph < 20.0:
        out.append(_w(
            "weak_trial", "crit", "Respuesta al peso de prueba muy baja",
            f"La vibración cambió solo {resp*100:.0f}% y {dph:.0f}° con el peso "
            f"de prueba (mínimo confiable {TRIAL_RESP_OK*100:.0f}%). Coeficiente "
            f"no confiable; probablemente NO es desbalance másico."))
    elif resp < TRIAL_RESP_OK:
        out.append(_w(
            "marginal_trial", "warn", "Respuesta al peso de prueba marginal",
            f"La vibración cambió {resp*100:.0f}% (ideal ≥{TRIAL_RESP_OK*100:.0f}%). "
            f"Usar un peso mayor o revisar repetibilidad."))

    wt_abs = max(1e-9, float(abs(Wt)))
    scale = float(result.get("corr_mass_g", 0)) / (trial_mass_g if trial_mass_g else 1e-9)
    scale = abs(scale)
    if scale >= CORR_SCALE_CRIT:
        out.append(_w(
            "corr_scale", "crit", "Corrección desproporcionada",
            f"La corrección es {scale:.0f}× el peso de prueba. Trial insuficiente "
            f"o plano insensible."))
    elif scale >= CORR_SCALE_WARN:
        out.append(_w(
            "corr_scale", "warn", "Corrección grande",
            f"La corrección es {scale:.0f}× el peso de prueba. Considera un trial mayor."))

    v0a = float(V0_mag)
    if Vf_mag is not None and float(Vf_mag) > 0:
        if float(Vf_mag) > v0a * WORSEN_MARGIN:
            out.append(_w(
                "worsened", "crit", "El plano empeoró",
                f"La vibración pasó de {v0a:.3f} a {float(Vf_mag):.3f} "
                f"({pct_change(v0a, float(Vf_mag)):+.0f}%)."))
    else:
        if float(result.get("pred_mag", 0)) > v0a * WORSEN_MARGIN:
            out.append(_w(
                "worsened_pred", "warn", "Empeora (predicho)",
                f"El modelo predice subir de {v0a:.3f} a {float(result.get('pred_mag',0)):.3f}."))

    if any(x["code"] == "weak_trial" for x in out) or \
       any(x["code"] == "corr_scale" and x["severity"] == "crit" for x in out):
        out.append(_w(
            "not_unbalance", "info", "¿Es realmente desbalance?",
            "Verifica alineación, solturas, 2X/armónicos y dependencia térmica "
            "antes de balancear con masa."))
    return out


def iso_residual_sanity(
    U_res_used: float, V0_dom: float, Vf_dom: float, U_trial: float
) -> List[Dict[str, str]]:
    """Avisa si el U_res usado para el grado ISO está desacoplado de la vibración
    final real. Compara con el estimado U_res ≈ (Vf/V0)·U_trial."""
    out: List[Dict[str, str]] = []
    if not (V0_dom and V0_dom > 0) or U_trial <= 0:
        return out
    est = calc_U_res_auto(V0_dom, Vf_dom, U_trial)
    used = max(0.0, float(U_res_used))
    # Si el residual usado es mucho menor que el estimado desde la vibración real,
    # el grado ISO queda sobrestimado (demasiado bueno).
    if est > 0 and used < 0.5 * est:
        out.append(_w(
            "iso_optimistic", "warn", "Grado ISO posiblemente sobrestimado",
            f"El U_res usado ({used:.1f} g·mm) es mucho menor que el estimado "
            f"desde la vibración final ({est:.1f} g·mm). Si la máquina sigue "
            f"vibrando, el grado ISO (p.ej. G0.4) no refleja el estado real. "
            f"Usar el residual derivado de los vectores finales o etiquetarlo "
            f"'estimado/manual'."))
    return out


# =========================================================
# Reparto de peso a posiciones fijas / álabes (Angular Mass Split)
# =========================================================
def split_to_positions(W_mag: float, W_ang: float,
                       positions_deg: List[float]) -> List[Dict[str, Any]]:
    """Reparte una corrección W∠θ en las POSICIONES FIJAS disponibles (huecos,
    slots, álabes). Devuelve 1 peso si θ coincide con una posición, o 2 pesos en
    las posiciones que bordean θ cuya suma vectorial = W∠θ.

    Resuelve  C = Wa·e^{iα} + Wb·e^{iβ}  (masas reales; positivas cuando θ está
    en el arco menor entre α y β — caso de posiciones adyacentes)."""
    pos = sorted({round(float(p) % 360.0, 6) for p in positions_deg})
    if not pos:
        return []
    th = float(W_ang) % 360.0
    for p in pos:                                   # coincide con una posición
        if abs(((th - p + 180.0) % 360.0) - 180.0) < 1e-6:
            return [{"angle": p, "mass": float(W_mag)}]
    if len(pos) == 1:                               # una sola posición → todo ahí
        return [{"angle": pos[0], "mass": float(W_mag)}]
    ext = pos + [pos[0] + 360.0]                    # cierre circular
    lo = hi = None
    for i in range(len(pos)):
        a, b = ext[i], ext[i + 1]
        if a <= th < b or a <= th + 360.0 < b:
            lo, hi = a % 360.0, b % 360.0
            break
    if lo is None:                                  # θ cae en el arco de cierre
        lo, hi = pos[-1], pos[0]
    C = to_complex(W_mag, W_ang)
    A = np.array([[np.cos(np.radians(lo)), np.cos(np.radians(hi))],
                  [np.sin(np.radians(lo)), np.sin(np.radians(hi))]])
    try:
        sol = np.linalg.solve(A, np.array([C.real, C.imag]))
    except np.linalg.LinAlgError:
        return [{"angle": float(W_ang) % 360.0, "mass": float(W_mag)}]
    return [{"angle": float(lo), "mass": float(sol[0])},
            {"angle": float(hi), "mass": float(sol[1])}]


def bucket_angles(n: int, offset_deg: float = 0.0,
                  clockwise: bool = False) -> List[Dict[str, Any]]:
    """Ángulos de N álabes/buckets numerados 1..N, igualmente espaciados. El #1 va
    en `offset_deg`; el sentido de numeración puede ser horario o antihorario."""
    n = max(1, int(n))
    step = 360.0 / n
    s = -1.0 if clockwise else 1.0
    return [{"bucket": k + 1, "angle": (offset_deg + s * k * step) % 360.0}
            for k in range(n)]


def split_to_buckets(W_mag: float, W_ang: float, n: int, offset_deg: float = 0.0,
                     clockwise: bool = False) -> List[Dict[str, Any]]:
    """Reparte la corrección W∠θ a los álabes/buckets (posiciones discretas).
    Devuelve [{bucket, angle, mass}] en 1 o 2 buckets."""
    bk = bucket_angles(n, offset_deg, clockwise)
    ang2bk = {round(b["angle"], 6): b["bucket"] for b in bk}
    out = split_to_positions(W_mag, W_ang, [b["angle"] for b in bk])
    for o in out:
        o["bucket"] = ang2bk.get(round(o["angle"], 6))
    return out


def combine_weights(weights: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Resultante (mag, ang°) de varios pesos (masa, ángulo°) — suma vectorial.
    Útil para reemplazar varios pesos por uno solo, o verificar una repartición."""
    z = sum((to_complex(m, a) for m, a in (weights or [])), 0j)
    return to_polar(z)


# =========================================================
# Nomograma ISO 21940-11 (excentricidad permisible vs velocidad)
# =========================================================
def iso_nomogram_lines(grades: List[float] = None,
                       rpm_min: float = 100.0, rpm_max: float = 100000.0,
                       pts: int = 60) -> Dict[str, Any]:
    """Datos para la carta ISO 21940-11: para cada grado G, e_per[µm] = 9549·G/n
    sobre un rango de velocidad (log). Para graficar las líneas de grado."""
    gs = grades or ISO_GRADES
    rpm = list(np.geomspace(max(1.0, rpm_min), max(rpm_min + 1.0, rpm_max), int(pts)))
    lines = {f"G{g:g}": [calc_e_per(g, n) * 1000.0 for n in rpm] for g in gs}  # mm→µm
    return {"rpm": rpm, "lines": lines, "grades": list(gs)}


def iso_operating_point(W_kg: float, U_res_gmm: float, rpm: float) -> Dict[str, Any]:
    """Punto de operación del rotor en la carta: excentricidad real e = U_res/W
    [µm] a la velocidad `rpm`, y el mejor grado que cumple."""
    W = max(1e-9, float(W_kg))
    e_um = float(U_res_gmm) / W            # g·mm/kg = µm
    ev = evaluate_iso_grades(W_kg, rpm, U_res_gmm)
    return {"rpm": float(rpm), "e_um": e_um, "best_grade": ev.get("best_grade"),
            "summary": ev.get("summary_label", "")}
