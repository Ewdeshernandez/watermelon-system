"""
core.scl_diagnostics
====================

Cálculos rotodinámicos específicos de Shaft Centerline (posición DC del
muñón dentro del cojinete plano), siguiendo prácticas Cat IV / API 670 /
Bently Nevada.

Métricas clave:

1. **Eccentricity ratio (e/c)**: distancia del centro del muñón al centro
   del cojinete dividida por el clearance radial. Indicador primario de
   condición hidrodinámica:
       e/c < 0.30   → riesgo de oil whirl (sub-amortiguado)
       0.30 ≤ e/c < 0.40 → margen reducido
       0.40 ≤ e/c ≤ 0.70 → ZONA SANA (operación hidrodinámica óptima)
       0.70 < e/c ≤ 0.85 → atención (alta carga / clearance reducido)
       e/c > 0.85   → riesgo de wipe / contacto babbitt

2. **Attitude angle (α)**: ángulo entre la línea de carga estática y la
   línea muñón-centro del cojinete. Para cojinetes planos típicos en
   carga vertical (gravity-loaded), α ≈ 30° a 50° en operación normal.
   Cambios bruscos de α entre fechas indican modificación de carga,
   alineación o desbalance.

3. **Centerline migration**: cambio de posición XY del muñón a operación
   nominal entre fechas. Migración > 30% del clearance es indicador
   directo de degradación del babbitt o cambio de condición operativa.

4. **Lift-off speed**: velocidad mínima a la que se establece régimen
   hidrodinámico completo (la posición XY deja de "barrer" desde el
   bottom-load location hacia su excentricidad de operación).

Pure math — sin Streamlit, sin I/O. Recibe arrays y dataclasses,
devuelve dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np


# =============================================================
# DATACLASSES
# =============================================================

@dataclass
class EccentricityState:
    """Estado de excentricidad para un punto de operación específico."""
    rpm: float
    x_pos: float        # posición X del muñón (mil pp o µm pp)
    y_pos: float
    cx_radial: float    # clearance radial X (misma unidad que posición)
    cy_radial: float
    eccentricity_ratio: float   # e/c (adimensional)
    attitude_angle_deg: float    # α en grados (0 = vertical down, +90 = right)
    classification: Literal["WHIRL_RISK", "MARGINAL_LOW", "HEALTHY", "MARGINAL_HIGH", "WIPE_RISK"]
    classification_text: str


@dataclass
class CenterlineMigration:
    """Resumen de migración entre dos fechas de centerline."""
    rpm_reference: float
    delta_x: float
    delta_y: float
    migration_distance: float
    migration_pct_of_clearance: float
    delta_eccentricity_ratio: float
    delta_attitude_angle_deg: float
    classification: Literal["STABLE", "MINOR_DRIFT", "MODERATE_DRIFT", "MAJOR_DRIFT"]
    narrative: str


# =============================================================
# UTILIDADES UNIDADES
# =============================================================

def mils_to_mm(mils: float) -> float:
    """1 mil = 0.0254 mm."""
    return float(mils) * 0.0254


def mm_to_mils(mm: float) -> float:
    """1 mm = 39.3701 mil."""
    return float(mm) / 0.0254


def derive_radial_clearance_from_vault(
    bearing_inner_diameter_mm: Optional[float],
    shaft_journal_diameter_mm: Optional[float],
    diametral_clearance_mm: Optional[float],
    *,
    target_unit: Literal["mil", "mm", "um"] = "mil",
    severity_estimate: str = "typical",
) -> Tuple[Optional[float], str]:
    """
    Calcula el clearance radial del cojinete plano para uso en SCL.

    Prioridad:
      1. diametral_clearance_mm (dato OEM directo)
      2. bearing_inner_diameter_mm - shaft_journal_diameter_mm (calculado)
      3. estimación heurística desde el diámetro del cojinete

    Args:
        bearing_inner_diameter_mm: diámetro interno del cojinete
        shaft_journal_diameter_mm: diámetro del muñón
        diametral_clearance_mm: clearance diametral OEM si existe
        target_unit: unidad de salida ('mil', 'mm', 'um')
        severity_estimate: 'tight' / 'typical' / 'loose' para fallback

    Returns:
        (clearance_radial, source_description)
    """
    cd_mm = None
    source = ""

    if diametral_clearance_mm is not None and diametral_clearance_mm > 0:
        cd_mm = float(diametral_clearance_mm)
        source = "OEM (campo diametral_clearance_mm capturado en Vault)"
    elif (
        bearing_inner_diameter_mm is not None and bearing_inner_diameter_mm > 0
        and shaft_journal_diameter_mm is not None and shaft_journal_diameter_mm > 0
    ):
        cd_mm = float(bearing_inner_diameter_mm) - float(shaft_journal_diameter_mm)
        source = "calculado (bearing_inner_diameter − shaft_journal_diameter)"
    elif bearing_inner_diameter_mm is not None and bearing_inner_diameter_mm > 0:
        factor_map = {"tight": 0.0010, "typical": 0.0015, "loose": 0.0020}
        factor = factor_map.get(severity_estimate, 0.0015)
        cd_mm = float(bearing_inner_diameter_mm) * factor
        source = f"estimación heurística ({severity_estimate}: Cd ≈ {factor:.4f} × Φ)"
    else:
        return None, "sin datos suficientes"

    # Clearance radial = mitad del diametral
    cr_mm = cd_mm / 2.0

    if target_unit == "mil":
        return mm_to_mils(cr_mm), source
    if target_unit == "um":
        return cr_mm * 1000.0, source
    return cr_mm, source


# =============================================================
# CÁLCULOS DE ECCENTRICITY Y ATTITUDE ANGLE
# =============================================================

def _classify_eccentricity(e_c: float) -> Tuple[str, str]:
    """Clasifica e/c con etiqueta + texto descriptivo Cat IV."""
    if not np.isfinite(e_c):
        return "MARGINAL_LOW", "Eccentricity ratio no determinable."

    if e_c < 0.30:
        return "WHIRL_RISK", (
            "Eccentricity ratio bajo: el muñón opera muy cerca del centro "
            "del cojinete, lo que reduce el amortiguamiento hidrodinámico. "
            "Riesgo de inestabilidad por oil whirl (precesión subsíncrona "
            "alrededor de 0.42–0.48X)."
        )
    if e_c < 0.40:
        return "MARGINAL_LOW", (
            "Eccentricity ratio en margen bajo: la película de aceite es más "
            "gruesa de lo óptimo para máxima rigidez dinámica. Vigilar "
            "subsíncronos en el espectro."
        )
    if e_c <= 0.70:
        return "HEALTHY", (
            "Eccentricity ratio en zona sana (0.40–0.70). El cojinete opera "
            "en régimen hidrodinámico estable con buen amortiguamiento y "
            "rigidez dinámica adecuada."
        )
    if e_c <= 0.85:
        return "MARGINAL_HIGH", (
            "Eccentricity ratio elevado: el muñón opera cerca del límite "
            "del clearance. Indica alta carga, clearance reducido por "
            "desgaste, o presión de aceite insuficiente. Vigilar temperatura "
            "del cojinete."
        )
    return "WIPE_RISK", (
        "Eccentricity ratio crítico (>0.85): el muñón está muy próximo a la "
        "superficie del babbitt. Riesgo inminente de contacto metal-metal "
        "(wipe) y daño por fatiga superficial. Acción técnica requerida."
    )


def compute_eccentricity_state(
    x_pos: float,
    y_pos: float,
    *,
    rpm: float = 0.0,
    cx_radial: float,
    cy_radial: float,
    bearing_center_x: float = 0.0,
    bearing_center_y: float = 0.0,
    load_direction_deg: float = 270.0,
) -> EccentricityState:
    """
    Calcula e/c, attitude angle y clasificación para un punto de operación.

    Args:
        x_pos, y_pos: posición del muñón (mismas unidades que cx/cy)
        rpm: velocidad operativa para el registro
        cx_radial, cy_radial: clearance radial del cojinete
        bearing_center_x, bearing_center_y: centro geométrico del cojinete
        load_direction_deg: dirección de la carga estática (deg, 270 = down)

    Returns:
        EccentricityState con todos los campos calculados.
    """
    dx = float(x_pos) - float(bearing_center_x)
    dy = float(y_pos) - float(bearing_center_y)

    # Excentricidad geométrica normalizada al clearance promedio
    # Para cojinetes con clearance asimétrico se usa el menor (más conservador)
    c_avg = max(1e-9, (float(cx_radial) + float(cy_radial)) / 2.0)
    e_geom = np.sqrt(dx * dx + dy * dy)
    e_c = e_geom / c_avg

    # Attitude angle: ángulo entre línea de carga y línea muñón-centro.
    # Convención: load_direction_deg = 270 (down). Position vector of journal
    # measured CCW from +X. Attitude = angle between -load direction (i.e. up)
    # and journal position vector. Simpler: angle between journal vector and
    # load vector.
    if e_geom < 1e-9:
        attitude_deg = 0.0
    else:
        position_angle_deg = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        # Attitude = angular distance from load direction
        diff = (position_angle_deg - load_direction_deg) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        attitude_deg = float(diff)

    classification, text = _classify_eccentricity(e_c)

    return EccentricityState(
        rpm=float(rpm),
        x_pos=float(x_pos),
        y_pos=float(y_pos),
        cx_radial=float(cx_radial),
        cy_radial=float(cy_radial),
        eccentricity_ratio=float(e_c),
        attitude_angle_deg=attitude_deg,
        classification=classification,
        classification_text=text,
    )


# =============================================================
# MIGRACIÓN MULTI-FECHA
# =============================================================

def _classify_migration(migration_pct: float) -> str:
    if migration_pct < 10.0:
        return "STABLE"
    if migration_pct < 25.0:
        return "MINOR_DRIFT"
    if migration_pct < 50.0:
        return "MODERATE_DRIFT"
    return "MAJOR_DRIFT"


def compare_centerline_migration(
    earlier: EccentricityState,
    later: EccentricityState,
) -> CenterlineMigration:
    """
    Compara dos estados de centerline (típicamente entre dos fechas) y
    cuantifica la migración del muñón normalizada por el clearance.

    Args:
        earlier: estado más antiguo
        later: estado más reciente

    Returns:
        CenterlineMigration con deltas, porcentaje y narrativa.
    """
    dx = later.x_pos - earlier.x_pos
    dy = later.y_pos - earlier.y_pos
    dist = float(np.sqrt(dx * dx + dy * dy))

    c_avg = max(1e-9, (later.cx_radial + later.cy_radial) / 2.0)
    pct = dist / c_avg * 100.0

    de_c = later.eccentricity_ratio - earlier.eccentricity_ratio
    da = later.attitude_angle_deg - earlier.attitude_angle_deg

    classification = _classify_migration(pct)

    if classification == "STABLE":
        narrative = (
            f"La posición del muñón a {later.rpm:.0f} rpm es estable entre las dos "
            f"fechas analizadas (migración {pct:.1f}% del clearance radial). "
            f"La condición rotodinámica del cojinete se considera consistente."
        )
    elif classification == "MINOR_DRIFT":
        narrative = (
            f"Se observa una migración menor del centerline ({pct:.1f}% del "
            f"clearance radial). El factor de eccentricity cambió "
            f"{de_c:+.3f} y el attitude angle {da:+.1f}°. Mantener seguimiento "
            f"para descartar tendencia."
        )
    elif classification == "MODERATE_DRIFT":
        narrative = (
            f"Migración moderada del centerline ({pct:.1f}% del clearance "
            f"radial). El cambio de e/c ({de_c:+.3f}) y de attitude angle "
            f"({da:+.1f}°) sugiere posible modificación de carga, clearance, "
            f"alineación o condición de babbitt. Investigar causas y "
            f"correlacionar con histórico de mantenimiento."
        )
    else:
        narrative = (
            f"Migración importante del centerline ({pct:.1f}% del clearance "
            f"radial). El cambio significativo de e/c ({de_c:+.3f}) y attitude "
            f"angle ({da:+.1f}°) es indicativo de degradación del babbitt, "
            f"cambio sustancial de carga, o desalineación. Acción técnica "
            f"requerida para confirmar el estado del cojinete."
        )

    return CenterlineMigration(
        rpm_reference=later.rpm,
        delta_x=float(dx),
        delta_y=float(dy),
        migration_distance=dist,
        migration_pct_of_clearance=pct,
        delta_eccentricity_ratio=float(de_c),
        delta_attitude_angle_deg=float(da),
        classification=classification,
        narrative=narrative,
    )


# =============================================================
# LIFT-OFF DETECTION
# =============================================================

def detect_lift_off_speed(
    rpms: np.ndarray,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    *,
    cx_radial: float,
    cy_radial: float,
    stable_window_rpm: float = 200.0,
    stable_threshold_pct: float = 5.0,
) -> Optional[float]:
    """
    Estima la velocidad de lift-off detectando cuando la posición del
    muñón se estabiliza (varía menos del threshold% del clearance) durante
    una ventana de stable_window_rpm.

    Args:
        rpms, x_positions, y_positions: arrays alineados de la corrida
        cx_radial, cy_radial: clearance radial
        stable_window_rpm: ventana de RPM para evaluar estabilidad
        stable_threshold_pct: variación máxima de posición (% del clearance)
            para considerar "estable"

    Returns:
        RPM del lift-off o None si no se detecta.
    """
    rpms = np.asarray(rpms, dtype=float)
    xs = np.asarray(x_positions, dtype=float)
    ys = np.asarray(y_positions, dtype=float)

    if rpms.size < 5:
        return None

    order = np.argsort(rpms)
    rpms = rpms[order]
    xs = xs[order]
    ys = ys[order]

    c_avg = max(1e-9, (float(cx_radial) + float(cy_radial)) / 2.0)
    threshold_abs = c_avg * (stable_threshold_pct / 100.0)

    # Para cada índice, mirar la variación XY en la ventana próxima
    for i in range(rpms.size - 3):
        rpm_start = rpms[i]
        # Ventana hasta rpm_start + stable_window_rpm
        idx_window = np.where(
            (rpms >= rpm_start) & (rpms <= rpm_start + stable_window_rpm)
        )[0]
        if idx_window.size < 3:
            continue
        x_win = xs[idx_window]
        y_win = ys[idx_window]
        # Variación máxima respecto a la mediana de la ventana
        x_med = float(np.median(x_win))
        y_med = float(np.median(y_win))
        max_dev = float(np.max(np.sqrt((x_win - x_med) ** 2 + (y_win - y_med) ** 2)))
        if max_dev <= threshold_abs:
            return float(rpm_start)

    return None


__all__ = [
    "EccentricityState",
    "CenterlineMigration",
    "compute_eccentricity_state",
    "compare_centerline_migration",
    "detect_lift_off_speed",
    "derive_radial_clearance_from_vault",
    "mils_to_mm",
    "mm_to_mils",
]
