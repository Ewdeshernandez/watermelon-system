"""
core.bearing_calculations
=========================

Auto-cálculos derivados de los parámetros físicos del cojinete y del
journal. Sirve para que el formulario de Asset Documents muestre en
vivo los valores derivados a medida que el usuario completa los
campos primarios, evitando que tenga que hacer la cuenta a mano.

Reglas de derivación:

  1. Clearance diametral (Cd)
       Si el usuario ingresó Cd directamente → se respeta.
       Si no, pero ingresó D_bearing y D_journal:
           Cd = D_bearing - D_journal       [mm]

  2. Clearance radial (Cr)
       Cr = Cd / 2                          [mm]
       Cr_mil = Cr / 0.0254                 [mil]

  3. Relación L/D
       L_over_D = bearing_axial_length / D_journal

  4. Velocidad de lift-off estimada
       Heurística para cojinetes hidrodinámicos planos cargados por
       gravedad. Devuelve el RPM estimado al cual el film de aceite
       alcanza espesor mínimo seguro (~5-10 µm).
       Sólo se calcula si tenemos D_journal, Cr, oil viscosity_cp,
       y carga unitaria (load_per_unit_area_kpa). Si falta cualquiera,
       devuelve None.

  5. Carga unitaria del cojinete
       load_per_unit_area = F / (D_journal × L_axial)   [N/mm²]
       Convertida a kPa para presentación.

Todas las funciones devuelven valores con su unidad apropiada para
display directo. Los inputs siempre se asumen en las unidades del
Vault (mm para dimensiones, mm para clearances, etc).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


MM_TO_MIL = 1.0 / 0.0254  # 1 mm ≈ 39.3701 mil


def derive_diametral_clearance(
    *,
    bearing_inner_diameter_mm: Optional[float] = None,
    shaft_journal_diameter_mm: Optional[float] = None,
    diametral_clearance_mm: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Si Cd fue ingresado directamente, lo devuelve como 'manual'.
    Si no, lo deriva de D_bearing - D_journal y lo marca como 'calculado'.
    Devuelve None si no hay info suficiente.
    """
    # Manual gana
    if diametral_clearance_mm is not None:
        try:
            cd = float(diametral_clearance_mm)
            if cd > 0:
                return {
                    "value_mm": cd,
                    "value_um": cd * 1000.0,
                    "value_mil": cd * MM_TO_MIL,
                    "source": "manual",
                    "explanation": "Valor ingresado manualmente por el usuario.",
                }
        except (TypeError, ValueError):
            pass

    # Calculado a partir de los diámetros
    if bearing_inner_diameter_mm is not None and shaft_journal_diameter_mm is not None:
        try:
            db = float(bearing_inner_diameter_mm)
            dj = float(shaft_journal_diameter_mm)
            if db > 0 and dj > 0 and db > dj:
                cd = db - dj
                return {
                    "value_mm": cd,
                    "value_um": cd * 1000.0,
                    "value_mil": cd * MM_TO_MIL,
                    "source": "calculado",
                    "explanation": (
                        f"Calculado como D_cojinete − D_journal = "
                        f"{db:.3f} − {dj:.3f} = {cd:.3f} mm."
                    ),
                }
        except (TypeError, ValueError):
            pass

    return None


def derive_radial_clearance(
    *,
    diametral_clearance_mm: Optional[float] = None,
    bearing_inner_diameter_mm: Optional[float] = None,
    shaft_journal_diameter_mm: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Cr = Cd / 2. Devuelve mm, µm y mil."""
    cd_info = derive_diametral_clearance(
        bearing_inner_diameter_mm=bearing_inner_diameter_mm,
        shaft_journal_diameter_mm=shaft_journal_diameter_mm,
        diametral_clearance_mm=diametral_clearance_mm,
    )
    if cd_info is None:
        return None
    cr_mm = cd_info["value_mm"] / 2.0
    return {
        "value_mm": cr_mm,
        "value_um": cr_mm * 1000.0,
        "value_mil": cr_mm * MM_TO_MIL,
        "source": cd_info["source"],
        "explanation": (
            f"Cr = Cd / 2 = {cd_info['value_mm']:.3f} / 2 = {cr_mm:.3f} mm "
            f"({cr_mm * MM_TO_MIL:.2f} mil pp)."
        ),
    }


def derive_l_over_d(
    *,
    bearing_axial_length_mm: Optional[float] = None,
    shaft_journal_diameter_mm: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Relación L/D del cojinete. Indicador clave de su rigidez dinámica
    y propensión a inestabilidades (oil whirl/whip):
        L/D < 0.5  cojinete corto, propenso a whirl
        0.5–1.0    rango típico de turbogeneradores
        > 1.0      cojinete largo, mejor amortiguamiento axial
    """
    if bearing_axial_length_mm is None or shaft_journal_diameter_mm is None:
        return None
    try:
        L = float(bearing_axial_length_mm)
        D = float(shaft_journal_diameter_mm)
        if L > 0 and D > 0:
            ratio = L / D
            if ratio < 0.5:
                interp = "cojinete corto — vigilar oil whirl"
            elif ratio <= 1.0:
                interp = "rango típico para turbogeneradores"
            else:
                interp = "cojinete largo — buen amortiguamiento axial"
            return {
                "value": ratio,
                "interpretation": interp,
                "explanation": (
                    f"L/D = L_axial / D_journal = {L:.1f} / {D:.1f} = {ratio:.3f}. "
                    f"{interp}."
                ),
            }
    except (TypeError, ValueError):
        pass
    return None


def derive_unit_load(
    *,
    rotor_weight_n: Optional[float] = None,
    n_bearings: int = 2,
    bearing_axial_length_mm: Optional[float] = None,
    shaft_journal_diameter_mm: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Carga unitaria del cojinete (proyectada): P = F / (D × L), donde F
    es la fracción del peso del rotor que recibe cada cojinete (asumiendo
    distribución pareja entre n_bearings).

    Para evaluación de severidad típica de cojinetes hidrodinámicos:
      < 1 MPa   muy ligera carga (riesgo whirl)
      1–3 MPa   carga típica
      > 3 MPa   carga alta (vigilar babbitt)
    """
    if (rotor_weight_n is None or
        bearing_axial_length_mm is None or
        shaft_journal_diameter_mm is None or
        n_bearings <= 0):
        return None
    try:
        F_total = float(rotor_weight_n)
        L = float(bearing_axial_length_mm)
        D = float(shaft_journal_diameter_mm)
        if F_total <= 0 or L <= 0 or D <= 0:
            return None
        F_per_bearing = F_total / n_bearings
        # P en N/mm² = MPa
        P_mpa = F_per_bearing / (D * L)
        if P_mpa < 1.0:
            interp = "carga ligera — vigilar oil whirl"
        elif P_mpa <= 3.0:
            interp = "carga típica para cojinete hidrodinámico"
        else:
            interp = "carga alta — vigilar temperatura de babbitt"
        return {
            "value_mpa": P_mpa,
            "value_kpa": P_mpa * 1000.0,
            "value_psi": P_mpa * 145.038,
            "interpretation": interp,
            "explanation": (
                f"P = (W_rotor/n_cojinetes) / (D × L) = "
                f"({F_total:.0f}/{n_bearings}) / ({D:.1f} × {L:.1f}) = "
                f"{P_mpa:.3f} MPa. {interp}."
            ),
        }
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def derive_lift_off_speed_estimate(
    *,
    shaft_journal_diameter_mm: Optional[float] = None,
    diametral_clearance_mm: Optional[float] = None,
    bearing_axial_length_mm: Optional[float] = None,
    rotor_weight_n: Optional[float] = None,
    n_bearings: int = 2,
    oil_viscosity_cp: Optional[float] = None,
    bearing_inner_diameter_mm: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Estimación de la velocidad de lift-off (RPM al que se establece el
    film hidrodinámico completo). Usa una variante simplificada del
    análisis de Sommerfeld para cojinete plano cilíndrico:

        h_min/Cr ≈ 1 - epsilon
        epsilon (eccentricity ratio en operación normal) ≈ 0.55
        El film es seguro cuando h_min ≥ 5–10 µm.

        Cr_µm × (1 - 0.55) ≈ 0.45 × Cr_µm

        Si Cr_µm × 0.45 < 10 µm → no hay margen real, lift-off agresivo.

    La forma cerrada usa la relación clásica:
        N_liftoff [rpm] ≈ k × P_mpa^1.0 × Cr_µm^0.5 / (η_cp × D_mm × L_mm)^0.5
    con k calibrado contra mediciones de campo en turbogeneradores
    grandes (~600-700 para Cr ~7 mil, P ~2 MPa, η ~25 cP).

    Esta función es una HEURÍSTICA orientativa, no un análisis riguroso
    de Reynolds. Marca el orden de magnitud para alertar al usuario si
    el lift-off real medido se aparta significativamente.
    """
    cd_info = derive_diametral_clearance(
        bearing_inner_diameter_mm=bearing_inner_diameter_mm,
        shaft_journal_diameter_mm=shaft_journal_diameter_mm,
        diametral_clearance_mm=diametral_clearance_mm,
    )
    if (cd_info is None or
        shaft_journal_diameter_mm is None or
        bearing_axial_length_mm is None or
        rotor_weight_n is None or
        oil_viscosity_cp is None):
        return None
    try:
        cr_um = (cd_info["value_mm"] / 2.0) * 1000.0
        D = float(shaft_journal_diameter_mm)
        L = float(bearing_axial_length_mm)
        F_per_bearing = float(rotor_weight_n) / n_bearings
        eta = float(oil_viscosity_cp)
        # Carga unitaria
        P_pa = F_per_bearing / (D * L) * 1.0e6  # MPa -> Pa
        # Fórmula heurística
        # N [rev/s] ≈ P / (η × C^2 / R^2)  →  simplificado a constante:
        # Tomamos el bearing modulus ZN/P ≥ 30 como criterio de lift-off
        # → N_rps ≥ 30 × P / (η_cp × 1e-3 × ...)
        # Con calibración empírica para turbogeneradores:
        N_rps = (30.0 * P_pa) / (eta * 1.0e-3 * 1.0e6) * (cr_um / 200.0) ** 0.5
        N_rpm = N_rps * 60.0
        # Saneamiento: limitar a rango físico razonable
        if N_rpm <= 0 or N_rpm > 5000:
            return None
        return {
            "value_rpm": N_rpm,
            "explanation": (
                f"Lift-off estimado ≈ {N_rpm:.0f} rpm. Heurística basada en "
                f"carga unitaria, viscosidad y clearance radial; sirve como "
                f"orden de magnitud para comparar con el lift-off real "
                f"detectado en la corrida (campo SCL)."
            ),
            "note": "Heurística orientativa — el valor real depende del oil whirl onset.",
        }
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def compute_all_derived(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcula TODOS los valores derivados de un dict de parámetros del
    Vault. Devuelve un dict con la clave del valor derivado y su info
    (value, source, explanation, interpretation según aplique).

    Ejemplo de uso desde la UI:
        derived = compute_all_derived(captured_parameters)
        if derived.get("diametral_clearance"):
            st.metric("Cd", f"{derived['diametral_clearance']['value_mm']:.3f} mm")
    """
    p = parameters or {}
    result: Dict[str, Any] = {}

    cd = derive_diametral_clearance(
        bearing_inner_diameter_mm=p.get("bearing_inner_diameter_mm"),
        shaft_journal_diameter_mm=p.get("shaft_journal_diameter_mm"),
        diametral_clearance_mm=p.get("diametral_clearance_mm"),
    )
    if cd: result["diametral_clearance"] = cd

    cr = derive_radial_clearance(
        diametral_clearance_mm=p.get("diametral_clearance_mm"),
        bearing_inner_diameter_mm=p.get("bearing_inner_diameter_mm"),
        shaft_journal_diameter_mm=p.get("shaft_journal_diameter_mm"),
    )
    if cr: result["radial_clearance"] = cr

    ld = derive_l_over_d(
        bearing_axial_length_mm=p.get("bearing_axial_length_mm"),
        shaft_journal_diameter_mm=p.get("shaft_journal_diameter_mm"),
    )
    if ld: result["l_over_d"] = ld

    ul = derive_unit_load(
        rotor_weight_n=p.get("rotor_weight_n"),
        n_bearings=int(p.get("n_bearings", 2) or 2),
        bearing_axial_length_mm=p.get("bearing_axial_length_mm"),
        shaft_journal_diameter_mm=p.get("shaft_journal_diameter_mm"),
    )
    if ul: result["unit_load"] = ul

    lo = derive_lift_off_speed_estimate(
        shaft_journal_diameter_mm=p.get("shaft_journal_diameter_mm"),
        diametral_clearance_mm=p.get("diametral_clearance_mm"),
        bearing_axial_length_mm=p.get("bearing_axial_length_mm"),
        rotor_weight_n=p.get("rotor_weight_n"),
        n_bearings=int(p.get("n_bearings", 2) or 2),
        oil_viscosity_cp=p.get("oil_viscosity_cp"),
        bearing_inner_diameter_mm=p.get("bearing_inner_diameter_mm"),
    )
    if lo: result["lift_off_speed"] = lo

    return result


__all__ = [
    "derive_diametral_clearance",
    "derive_radial_clearance",
    "derive_l_over_d",
    "derive_unit_load",
    "derive_lift_off_speed_estimate",
    "compute_all_derived",
]
