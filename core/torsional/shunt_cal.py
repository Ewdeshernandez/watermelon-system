"""
core/torsional/shunt_cal.py — Verificación de calibración por shunt (TT10K)
===========================================================================

El TX10K-S tiene dos resistores shunt de precisión a bordo que simulan un
strain conocido, activables con el control remoto RM10K:

  · Reference 1 → 100 µε  (shunt 437,400 Ω ±0.1%)   con 350 Ω / GF 2.0
  · Reference 2 → 500 µε  (shunt  87,370 Ω ±0.1%)   con 350 Ω / GF 2.0

Al activar un shunt, la salida del RX10K debe ir a un voltaje predecible:

    Vs = (εs / εFS) · Z · VFS

donde εs = strain simulado por el shunt, εFS = strain de fondo de escala,
Z = factor de escala (System Gain), VFS = 10 V.

Ejemplo del manual (Appendix B1): εs=100 µε, εFS=500 µε, Z=0.9802
    → Vs = (100/500)·0.9802·10 = 1.9604 V.

Este módulo permite al software de campo:
  1. Calcular el voltaje esperado del shunt (`expected_shunt_voltage`).
  2. Contrastar la lectura real y sugerir corrección (`verify_shunt`),
     replicando el paso 12 del procedimiento de campo del manual de forma
     automatizada — verificación de calibración clase System1/ADRE.

Nota sobre GF ≠ 2.0 o RG ≠ 350 Ω: el strain simulado por el shunt se calcula
con `simulated_strain_from_shunt` usando RC = RG / (N·GF·ε)  (Appendix B1),
siempre con GF = 2.0 para el strain de referencia (ver nota del manual pág. 34).
"""
from __future__ import annotations

from dataclasses import dataclass

from core.torsional.scaling import (
    GageConfig,
    TorqueScaling,
    VFS_DEFAULT,
    full_scale_strain_torque,
)


@dataclass(frozen=True)
class ShuntReference:
    """
    Un shunt de referencia del TX10K-S.

    Attributes:
        name: Etiqueta (p.ej. "Ref 1")
        resistance_ohm: Resistencia del shunt [Ω]
        simulated_ue: Strain simulado nominal [µε] con 350 Ω / GF 2.0
    """
    name: str
    resistance_ohm: float
    simulated_ue: float


# Shunts de fábrica del TX10K-S (Appendix A)
REF1_100UE = ShuntReference(name="Ref 1", resistance_ohm=437_400.0, simulated_ue=100.0)
REF2_500UE = ShuntReference(name="Ref 2", resistance_ohm=87_370.0, simulated_ue=500.0)


def simulated_strain_from_shunt(shunt_resistance_ohm: float,
                                gage_resistance_ohm: float = 350.0,
                                n_active_gages: int = 1,
                                gage_factor: float = 2.0) -> float:
    """
    Strain simulado por un resistor shunt en paralelo con un brazo del puente.

    De RC = RG / (N·GF·ε)  →  ε = RG / (N·GF·RC)   [Appendix B1].

    Args:
        shunt_resistance_ohm: RC, resistencia del shunt [Ω]
        gage_resistance_ohm: RG, resistencia de la galga [Ω]
        n_active_gages: N, número de brazos afectados por el shunt (1 típico)
        gage_factor: GF (usar 2.0 para strain de referencia)

    Returns:
        Strain simulado [µε].
    """
    if shunt_resistance_ohm <= 0:
        raise ValueError("shunt_resistance_ohm debe ser > 0")
    eps_in_in = gage_resistance_ohm / (n_active_gages * gage_factor * shunt_resistance_ohm)
    return eps_in_in * 1.0e6


def expected_shunt_voltage(simulated_ue: float, full_scale_ue: float,
                           scale_factor_z: float = 1.0,
                           vfs: float = VFS_DEFAULT) -> float:
    """
    Voltaje esperado en la salida del RX10K al aplicar un shunt.

        Vs = (εs / εFS) · Z · VFS

    Args:
        simulated_ue: εs, strain simulado por el shunt [µε]
        full_scale_ue: εFS, strain de fondo de escala [µε]
        scale_factor_z: Z, factor de escala System Gain (0.25–4.0)
        vfs: Fondo de escala de voltaje [V]

    Returns:
        Voltaje esperado [V].
    """
    if full_scale_ue <= 0:
        raise ValueError("full_scale_ue debe ser > 0")
    return (simulated_ue / full_scale_ue) * scale_factor_z * vfs


@dataclass
class ShuntCheck:
    """
    Resultado de una verificación de shunt.

    Attributes:
        expected_v: Voltaje esperado [V]
        measured_v: Voltaje medido [V]
        error_pct: Error relativo a fondo de escala [(medido−esperado)/VFS·100]
        passed: True si |error| ≤ tolerancia
        suggested_z: Factor de escala EFECTIVO que revela el shunt
            (Z·medido/esperado). Es el Z real al que está operando el sistema;
            adoptarlo en TorqueScaling da la máxima exactitud. Para en cambio
            ajustar el System Gain del RX10K hasta que el output iguale al
            esperado, multiplica la ganancia por (esperado/medido).
    """
    expected_v: float
    measured_v: float
    error_pct: float
    passed: bool
    suggested_z: float


def verify_shunt(measured_v: float, shunt: ShuntReference, gage: GageConfig,
                 scale_factor_z: float = 1.0, vfs: float = VFS_DEFAULT,
                 tolerance_pct: float = 0.5) -> ShuntCheck:
    """
    Verifica una lectura de shunt contra el valor esperado y sugiere corrección.

    El error se expresa como % del fondo de escala (VFS), que es la convención
    de exactitud del TT10K (±0.16%FS offset, ±0.25%R gain). El ajuste típico
    tras el escalado es <0.5% (manual, Appendix B1).

    Args:
        measured_v: Voltaje leído en el RX10K con el shunt activo [V]
        shunt: Referencia de shunt aplicada (REF1_100UE / REF2_500UE)
        gage: Configuración de galga/transmisor (da εFS)
        scale_factor_z: Z actualmente programado en el RX10K
        vfs: Fondo de escala de voltaje [V]
        tolerance_pct: Tolerancia de aprobación en %FS

    Returns:
        ShuntCheck con esperado, error y Z sugerido.
    """
    eps_fs = full_scale_strain_torque(gage, vfs=vfs)
    expected = expected_shunt_voltage(shunt.simulated_ue, eps_fs, scale_factor_z, vfs)
    error_pct = (measured_v - expected) / vfs * 100.0
    # Factor de escala efectivo que revela el shunt: el output es ∝ Z, así que
    # Z_efectivo = Z_objetivo · (medido/esperado). Adoptarlo en TorqueScaling
    # corrige la desviación medida.
    if expected != 0:
        suggested_z = scale_factor_z * (measured_v / expected)
    else:
        suggested_z = scale_factor_z
    return ShuntCheck(
        expected_v=expected,
        measured_v=measured_v,
        error_pct=error_pct,
        passed=abs(error_pct) <= tolerance_pct,
        suggested_z=suggested_z,
    )
