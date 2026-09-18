"""
core/torsional/scaling.py — Escalado voltaje → torque (Binsfeld TorqueTrak 10K)
==============================================================================

Convierte el voltaje de salida del receptor RX10K (Volts, fondo de escala = 10 V)
a unidades de ingeniería (N·m / ft-lb) usando la geometría del eje, los
parámetros de la galga y la ganancia del transmisor TX10K-S.

Es el equivalente torsional de `core.modal.signal_scaling`: el DAQ (NI 9215)
entrega Volts crudos; aquí se aplica la constante de calibración del sistema.

Base física (Appendix B del User's Guide, verificado vs. ejemplos del manual)
-----------------------------------------------------------------------------
El RX10K entrega VFS = 10 V a fondo de escala. El fondo de escala en strain
depende de la ganancia del transmisor GXMT y del factor de galga GF:

  · Torque, puente completo 4 brazos:   εFS = VFS / (VEXC·GF·GXMT)      [in/in]
  · Axial,  puente completo 2.6 brazos: εFS = VFS·4 / (VEXC·GF·GXMT·2.6)
  · ¼ puente, 1 brazo:                  εFS = VFS·4 / (VEXC·GF·GXMT)     (B3)

Torque de fondo de escala (eje redondo, torsión, γ superficial → T):

  TFS = εFS · π·E·(Do⁴−Di⁴) / (16·(1+ν)·Do)

que para acero sólido (E=30e6 psi, ν=0.30) se reduce a la ecuación simplificada
del manual:  TFS[ft-lb] = 1510.38e3 · Do³ / (GF·GXMT).

Verificaciones (ejemplos del manual)
------------------------------------
  · Torque:  Do=3.000", GF=2.08, GXMT=4000  → TFS = 4,901 ft-lb        (B1)
  · Axial:   Do=2.25",  GF=2.045, GXMT=4000 → PFS = 89,739 lb          (B2)
  · ¼ puente: GF=2.045, GXMT=4000           → εFS = 1956 µε            (B3)

Norma / referencia
------------------
Vishay TN-512 (relación shear strain ↔ torque). El TT10K tiene ancho de banda
0–500 Hz (-3 dB), lo cual acota el análisis de vibración torsional.

Unidades: entradas en sistema imperial (in, psi, ft-lb, lb) — el manual está en
imperial y las galgas se especifican así. Helpers de conversión al final.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# --- Constantes del sistema TT10K ------------------------------------------
VFS_DEFAULT = 10.0    # Fondo de escala de salida del RX10K [V]
VEXC_DEFAULT = 2.5    # Excitación del puente en el TX10K-S [V]

# Propiedades de acero (defaults)
E_STEEL_PSI = 30.0e6      # Módulo de elasticidad acero [psi]
E_STEEL_NMM2 = 206.8e3    # Módulo de elasticidad acero [N/mm²]
POISSON_STEEL = 0.30      # Relación de Poisson acero

# Conversiones
FTLB_PER_NM = 0.737562149
NM_PER_FTLB = 1.0 / FTLB_PER_NM
IN_PER_MM = 1.0 / 25.4

# Ganancias válidas del transmisor TX10K-S y su fondo de escala en microstrain
# (puente completo torque, referenciado a GF = 2.0; datasheet pág. 2 / Appendix A)
TX_GAIN_FULLSCALE_UE = {
    500: 4000.0,
    1000: 2000.0,
    2000: 1000.0,
    4000: 500.0,
    8000: 250.0,
    16000: 125.0,
}


class BridgeType(str, Enum):
    """Tipo de puente de galgas montado en el eje."""
    TORQUE = "torque"        # Puente completo, 4 brazos activos (torsión)
    AXIAL = "axial"          # Puente completo, 2.6 brazos activos (tensión/compresión)
    QUARTER = "quarter"      # ¼ puente, 1 brazo activo (galga simple)


@dataclass
class ShaftGeometry:
    """
    Geometría del eje instrumentado.

    Attributes:
        outer_diameter_in: Diámetro exterior Do [pulgadas]
        inner_diameter_in: Diámetro interior Di [pulgadas] (0.0 para eje sólido)
        modulus_psi: Módulo de elasticidad E [psi] (acero = 30e6)
        poisson: Relación de Poisson ν (acero = 0.30)
    """
    outer_diameter_in: float
    inner_diameter_in: float = 0.0
    modulus_psi: float = E_STEEL_PSI
    poisson: float = POISSON_STEEL

    def __post_init__(self) -> None:
        if self.outer_diameter_in <= 0:
            raise ValueError("outer_diameter_in debe ser > 0")
        if self.inner_diameter_in < 0 or self.inner_diameter_in >= self.outer_diameter_in:
            raise ValueError("inner_diameter_in debe estar en [0, Do)")

    @classmethod
    def from_mm(cls, outer_mm: float, inner_mm: float = 0.0,
                modulus_nmm2: float = E_STEEL_NMM2,
                poisson: float = POISSON_STEEL) -> "ShaftGeometry":
        """Construye desde milímetros (E en N/mm²)."""
        return cls(
            outer_diameter_in=outer_mm * IN_PER_MM,
            inner_diameter_in=inner_mm * IN_PER_MM,
            modulus_psi=modulus_nmm2 / 0.00689476,  # N/mm² → psi
            poisson=poisson,
        )


@dataclass
class GageConfig:
    """
    Parámetros de la galga y del transmisor.

    Attributes:
        gage_factor: Factor de galga GF (típico 2.0–2.1, del empaque)
        transmitter_gain: Ganancia GXMT del TX10K-S (500..16000)
        gage_resistance_ohm: Resistencia del puente RG [Ω] (350 estándar)
        bridge: Tipo de puente
    """
    gage_factor: float = 2.0
    transmitter_gain: int = 4000
    gage_resistance_ohm: float = 350.0
    bridge: BridgeType = BridgeType.TORQUE

    def __post_init__(self) -> None:
        if self.gage_factor <= 0:
            raise ValueError("gage_factor debe ser > 0")
        if self.transmitter_gain <= 0:
            raise ValueError("transmitter_gain debe ser > 0")


# --- Fondo de escala en strain ---------------------------------------------
def full_scale_strain_torque(gage: GageConfig, vfs: float = VFS_DEFAULT,
                             vexc: float = VEXC_DEFAULT) -> float:
    """
    Strain de fondo de escala para puente de torque (4 brazos) [µε].

    εFS[in/in] = VFS / (VEXC·GF·GXMT); ×1e6 → µε.
    Ej.: GF=2.0, GXMT=4000 → 500 µε.
    """
    eps_in_in = vfs / (vexc * gage.gage_factor * gage.transmitter_gain)
    return eps_in_in * 1.0e6


def full_scale_strain_axial(gage: GageConfig, vfs: float = VFS_DEFAULT,
                            vexc: float = VEXC_DEFAULT) -> float:
    """Strain de fondo de escala para puente axial (2.6 brazos) [µε]."""
    eps_in_in = (vfs * 4.0) / (vexc * gage.gage_factor * gage.transmitter_gain * 2.6)
    return eps_in_in * 1.0e6


def full_scale_strain_quarter(gage: GageConfig, vfs: float = VFS_DEFAULT,
                              vexc: float = VEXC_DEFAULT) -> float:
    """
    Strain de fondo de escala para ¼ puente (1 brazo) [µε] (Appendix B3).

    εFS[in/in] = VFS·4 / (VEXC·GF·GXMT).
    Ej.: GF=2.045, GXMT=4000 → 1956 µε.
    """
    eps_in_in = (vfs * 4.0) / (vexc * gage.gage_factor * gage.transmitter_gain)
    return eps_in_in * 1.0e6


# --- Fondo de escala en unidades de ingeniería -----------------------------
def full_scale_torque(shaft: ShaftGeometry, gage: GageConfig,
                      vfs: float = VFS_DEFAULT, vexc: float = VEXC_DEFAULT,
                      units: str = "ftlb") -> float:
    """
    Torque de fondo de escala TFS (correspondiente a VFS = 10 V).

    TFS = εFS · π·E·(Do⁴−Di⁴) / (16·(1+ν)·Do), con εFS en in/in, E en psi,
    Do,Di en pulgadas → resultado en in-lb; /12 → ft-lb.

    Args:
        units: "ftlb" (default) o "nm"

    Returns:
        Torque de fondo de escala en las unidades pedidas.
    """
    eps_fs = full_scale_strain_torque(gage, vfs, vexc) * 1.0e-6  # µε → in/in
    do, di = shaft.outer_diameter_in, shaft.inner_diameter_in
    tfs_inlb = (eps_fs * np.pi * shaft.modulus_psi * (do**4 - di**4)
                / (16.0 * (1.0 + shaft.poisson) * do))
    tfs_ftlb = tfs_inlb / 12.0
    if units == "nm":
        return tfs_ftlb * NM_PER_FTLB
    if units == "ftlb":
        return tfs_ftlb
    raise ValueError(f"units debe ser 'ftlb' o 'nm', se recibió {units!r}")


def full_scale_force_axial(shaft: ShaftGeometry, gage: GageConfig,
                           vfs: float = VFS_DEFAULT, vexc: float = VEXC_DEFAULT) -> float:
    """
    Fuerza axial de fondo de escala PFS [lb] (Appendix B2).

    PFS = VFS·π·E·(Do²−Di²) / (VEXC·GF·GXMT·2·(1+ν)).
    Ej.: Do=2.25", GF=2.045, GXMT=4000 → 89,739 lb.
    """
    do, di = shaft.outer_diameter_in, shaft.inner_diameter_in
    return (vfs * np.pi * shaft.modulus_psi * (do**2 - di**2)
            / (vexc * gage.gage_factor * gage.transmitter_gain
               * 2.0 * (1.0 + shaft.poisson)))


# --- Constante de calibración y conversión runtime -------------------------
@dataclass
class TorqueScaling:
    """
    Constante de calibración lista para convertir Volts → torque en tiempo real.

    El campo clave es `eu_per_volt` (torque por voltio). Se obtiene de la
    geometría (`from_geometry`) o se fija tras una calibración por peso muerto
    (`from_deadweight`).

    Attributes:
        eu_per_volt: Unidades de ingeniería por voltio (N·m/V o ft-lb/V)
        units: "nm" o "ftlb"
        vfs: Fondo de escala de voltaje del RX10K [V]
        scale_factor_z: Factor de escala System Gain aplicado en el RX10K
            (Z = TFS/TREF; 1.0 si System Gain = Transmitter Gain)
        full_scale_torque: TFS a VFS antes de escalar (informativo)
    """
    eu_per_volt: float
    units: str = "nm"
    vfs: float = VFS_DEFAULT
    scale_factor_z: float = 1.0
    full_scale_torque: float = 0.0

    @classmethod
    def from_geometry(cls, shaft: ShaftGeometry, gage: GageConfig,
                      units: str = "nm", scale_factor_z: float = 1.0,
                      vfs: float = VFS_DEFAULT, vexc: float = VEXC_DEFAULT) -> "TorqueScaling":
        """
        Constante de calibración desde geometría del eje.

        Con System Gain = Transmitter Gain (Z=1): torque(10 V) = TFS.
        Si el System Gain se escaló por Z (0.25–4.0), entonces 10 V ↔ TFS/Z,
        y eu_per_volt = TFS / (VFS·Z).
        """
        tfs = full_scale_torque(shaft, gage, vfs, vexc, units)
        if not (0.25 <= scale_factor_z <= 4.0):
            raise ValueError("scale_factor_z debe estar en [0.25, 4.0]")
        return cls(
            eu_per_volt=tfs / (vfs * scale_factor_z),
            units=units,
            vfs=vfs,
            scale_factor_z=scale_factor_z,
            full_scale_torque=tfs,
        )

    @classmethod
    def from_deadweight(cls, applied_torque: float, measured_volts: float,
                        units: str = "nm", vfs: float = VFS_DEFAULT) -> "TorqueScaling":
        """
        Constante desde calibración por peso muerto (el método más preciso):
        se aplica un torque conocido y se lee el voltaje resultante.
        """
        if measured_volts == 0:
            raise ValueError("measured_volts no puede ser 0")
        eupv = applied_torque / measured_volts
        return cls(
            eu_per_volt=eupv,
            units=units,
            vfs=vfs,
            full_scale_torque=eupv * vfs,
        )


def voltage_to_torque(voltage_signal: np.ndarray, scaling: TorqueScaling,
                      remove_dc: bool = False) -> np.ndarray:
    """
    Convierte un vector de Volts del RX10K a torque en unidades de ingeniería.

    Args:
        voltage_signal: Array de voltajes crudos del NI 9215 [V]
        scaling: Constante de calibración (TorqueScaling)
        remove_dc: Si True, resta la media → sólo torque dinámico (rizado /
            vibración torsional). Si False, conserva el torque medio (default).

    Returns:
        Array de torque (N·m o ft-lb según scaling.units).
    """
    v = np.asarray(voltage_signal, dtype=float)
    if remove_dc:
        v = v - np.mean(v)
    return v * scaling.eu_per_volt
