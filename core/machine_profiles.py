"""
core.machine_profiles
=====================

Catálogo de Asset Profiles para Watermelon System. Cada profile describe
una clase de máquina con sus parámetros rotodinámicos típicos y la
estrategia de evaluación de severidad recomendada (norma ISO aplicable
o umbrales personalizados).

Filosofía:
- El usuario selecciona un profile en la sidebar; eso fija
  operating_rpm, machine_group, ISO part, applicable_modules y
  threshold_strategy de manera coherente para todos los módulos
  (Polar, Bode, Shaft Centerline, etc.).
- Si el profile no aplica a un módulo (p. ej. Polar a un compresor
  reciprocante), el módulo muestra una nota técnica explicando por qué
  y sugiere la herramienta correcta.
- Tres fuentes de umbrales: la norma ISO directa, valores del fabricante
  (OEM) ingresados manualmente, o un baseline estadístico de tendencias
  (requiere histórico — placeholder en esta versión).

Esta arquitectura permite que la app sea genuinamente multi-máquina
sin hardcodear ISO 20816-2 group2 (que solo aplica al Brush).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


ThresholdStrategy = Literal["iso", "manufacturer", "data_baseline", "custom"]


@dataclass
class MachineProfile:
    """
    Definición de un perfil de máquina monitoreada.

    Campos:
        key: identificador único interno
        label: nombre legible para UI
        category: clasificación amplia (turbomachinery / motor / pump / reciprocating / custom)
        iso_part: parte aplicable de ISO 20816 ("20816-2", "20816-3", "20816-4", "20816-7", "20816-6", "20816-8")
        machine_group: grupo dentro de la norma cuando aplica ("group1", "group2", "class_iv", etc.)
        operating_rpm: velocidad operativa nominal por defecto
        bearing_type: "plain" / "rolling" / "mixed"
        rated_power_mw: potencia típica
        applicable_modules: lista de páginas donde el análisis es técnicamente relevante
        threshold_strategy: estrategia default ("iso", "manufacturer", "data_baseline", "custom")
        oem_thresholds: umbrales del fabricante en µm pp si threshold_strategy == "manufacturer"
        notes: texto adicional para mostrar en sidebar / reporte
    """
    key: str
    label: str
    category: str
    iso_part: str
    machine_group: str
    operating_rpm: float
    bearing_type: str
    rated_power_mw_min: float
    rated_power_mw_max: float
    applicable_modules: List[str] = field(default_factory=list)
    threshold_strategy: ThresholdStrategy = "iso"
    oem_thresholds_um_pp: Optional[Dict[str, float]] = None  # {"AB":..., "BC":..., "CD":...}
    notes: str = ""


# =============================================================
# CATÁLOGO DE PROFILES
# =============================================================
# Lista ordenada por categoría. El primer profile de la lista es el
# default cuando el usuario aún no eligió ninguno.

_PROFILES_LIST: List[MachineProfile] = [
    # ---------- TURBOMACHINERY ----------
    MachineProfile(
        key="brush_turbogenerator_54mw_3600",
        label="Brush turbogenerator 54 MW (3600 rpm)",
        category="turbomachinery",
        iso_part="20816-2",
        machine_group="group2",
        operating_rpm=3600.0,
        bearing_type="plain",
        rated_power_mw_min=40.0,
        rated_power_mw_max=80.0,
        applicable_modules=["polar", "bode", "shaft_centerline", "spectrum", "trends", "orbit", "phase", "tsa"],
        threshold_strategy="iso",
        notes="Turbogenerador grande con cojinetes planos, 60 Hz 2 polos. ISO 20816-2 grupo 2.",
    ),
    MachineProfile(
        key="steam_turbine_large_3600",
        label="Steam turbine >40 MW (3600 rpm)",
        category="turbomachinery",
        iso_part="20816-2",
        machine_group="group2",
        operating_rpm=3600.0,
        bearing_type="plain",
        rated_power_mw_min=40.0,
        rated_power_mw_max=500.0,
        applicable_modules=["polar", "bode", "shaft_centerline", "spectrum", "trends", "orbit", "phase", "tsa"],
        threshold_strategy="iso",
        notes="Turbina de vapor grande cojinetes planos. ISO 20816-2.",
    ),
    MachineProfile(
        key="steam_turbine_medium_3600",
        label="Steam turbine 15–40 MW (3600 rpm)",
        category="turbomachinery",
        iso_part="20816-3",
        machine_group="class_iv",
        operating_rpm=3600.0,
        bearing_type="plain",
        rated_power_mw_min=15.0,
        rated_power_mw_max=40.0,
        applicable_modules=["polar", "bode", "shaft_centerline", "spectrum", "trends", "orbit", "phase", "tsa"],
        threshold_strategy="iso",
        notes="Turbina de vapor mediana. Cae bajo ISO 20816-3 (15 kW–40 MW).",
    ),
    MachineProfile(
        key="siemens_sgt300",
        label="Siemens SGT-300 industrial gas turbine (~13 MW)",
        category="turbomachinery",
        iso_part="20816-3",
        machine_group="class_iv",
        operating_rpm=14000.0,
        bearing_type="plain",
        rated_power_mw_min=7.0,
        rated_power_mw_max=14.0,
        applicable_modules=["polar", "bode", "shaft_centerline", "spectrum", "trends", "orbit", "phase", "tsa"],
        threshold_strategy="iso",
        notes="Turbina de gas industrial heavy-duty. ISO 20816-3.",
    ),
    MachineProfile(
        key="siemens_sgt400",
        label="Siemens SGT-400 industrial gas turbine (~13–15 MW)",
        category="turbomachinery",
        iso_part="20816-3",
        machine_group="class_iv",
        operating_rpm=14600.0,
        bearing_type="plain",
        rated_power_mw_min=10.0,
        rated_power_mw_max=15.0,
        applicable_modules=["polar", "bode", "shaft_centerline", "spectrum", "trends", "orbit", "phase", "tsa"],
        threshold_strategy="iso",
        notes="Turbina de gas industrial heavy-duty. ISO 20816-3.",
    ),
    MachineProfile(
        key="ge_lm6000",
        label="GE LM6000 aero-derivative (3600 rpm)",
        category="turbomachinery",
        iso_part="20816-4",
        machine_group="rolling_aero",
        operating_rpm=3600.0,
        bearing_type="rolling",
        rated_power_mw_min=40.0,
        rated_power_mw_max=60.0,
        applicable_modules=["polar", "bode", "spectrum", "trends", "orbit", "phase", "tsa", "envelope"],
        threshold_strategy="iso",
        notes="Aero-derivada con cojinetes de rodillos. ISO 20816-4 (envelope spectrum aplicable).",
    ),

    # ---------- MOTORES ELÉCTRICOS ----------
    MachineProfile(
        key="motor_2pole_60hz",
        label="Electric motor 2-pole (3600 rpm, 60 Hz)",
        category="motor",
        iso_part="20816-3",
        machine_group="class_iv",
        operating_rpm=3600.0,
        bearing_type="rolling",
        rated_power_mw_min=0.05,
        rated_power_mw_max=20.0,
        applicable_modules=["polar", "bode", "spectrum", "trends", "orbit", "phase", "tsa", "envelope"],
        threshold_strategy="iso",
        notes="Motor de inducción 2 polos, 60 Hz. ISO 20816-3 clase IV.",
    ),
    MachineProfile(
        key="motor_4pole_60hz",
        label="Electric motor 4-pole (1800 rpm, 60 Hz)",
        category="motor",
        iso_part="20816-3",
        machine_group="class_iv",
        operating_rpm=1800.0,
        bearing_type="rolling",
        rated_power_mw_min=0.05,
        rated_power_mw_max=20.0,
        applicable_modules=["polar", "bode", "spectrum", "trends", "orbit", "phase", "tsa", "envelope"],
        threshold_strategy="iso",
        notes="Motor de inducción 4 polos, 60 Hz. ISO 20816-3 clase IV.",
    ),
    MachineProfile(
        key="motor_6pole_60hz",
        label="Electric motor 6-pole (1200 rpm, 60 Hz)",
        category="motor",
        iso_part="20816-3",
        machine_group="class_iv",
        operating_rpm=1200.0,
        bearing_type="rolling",
        rated_power_mw_min=0.05,
        rated_power_mw_max=20.0,
        applicable_modules=["polar", "bode", "spectrum", "trends", "orbit", "phase", "tsa", "envelope"],
        threshold_strategy="iso",
        notes="Motor de inducción 6 polos, 60 Hz. ISO 20816-3 clase IV.",
    ),
    MachineProfile(
        key="motor_vfd_highspeed",
        label="Electric motor VFD high-speed (variable, >3600 rpm)",
        category="motor",
        iso_part="20816-3",
        machine_group="class_iv",
        operating_rpm=4500.0,
        bearing_type="rolling",
        rated_power_mw_min=0.05,
        rated_power_mw_max=20.0,
        applicable_modules=["polar", "bode", "spectrum", "trends", "orbit", "phase", "tsa", "envelope"],
        threshold_strategy="iso",
        notes="Motor accionado por VFD a alta velocidad. Velocidad operativa variable; ajustar manualmente.",
    ),

    # ---------- BOMBAS CENTRÍFUGAS / MULTIETAPA ----------
    MachineProfile(
        key="pump_horizontal_multistage",
        label="Horizontal multistage centrifugal pump",
        category="pump",
        iso_part="20816-7",
        machine_group="category_i",
        operating_rpm=3600.0,
        bearing_type="rolling",
        rated_power_mw_min=0.05,
        rated_power_mw_max=20.0,
        applicable_modules=["polar", "bode", "spectrum", "trends", "orbit", "phase", "tsa", "envelope"],
        threshold_strategy="iso",
        notes="Bomba centrífuga multietapa horizontal. ISO 20816-7 (rotodinámicas industriales).",
    ),
    MachineProfile(
        key="pump_vertical_multistage",
        label="Vertical multistage pump",
        category="pump",
        iso_part="20816-7",
        machine_group="category_i",
        operating_rpm=3600.0,
        bearing_type="rolling",
        rated_power_mw_min=0.05,
        rated_power_mw_max=20.0,
        applicable_modules=["polar", "bode", "spectrum", "trends", "orbit", "phase", "tsa", "envelope"],
        threshold_strategy="iso",
        notes="Bomba multietapa vertical. ISO 20816-7. Considera modos de carcasa específicos de configuración vertical.",
    ),
    MachineProfile(
        key="pump_vertical_turbine",
        label="Vertical turbine pump (deep well)",
        category="pump",
        iso_part="20816-7",
        machine_group="category_ii",
        operating_rpm=1800.0,
        bearing_type="mixed",
        rated_power_mw_min=0.05,
        rated_power_mw_max=10.0,
        applicable_modules=["spectrum", "trends", "envelope"],
        threshold_strategy="iso",
        notes="Bomba vertical de turbina pozo profundo. Análisis Polar/Bode limitado por configuración multi-etapa larga; preferir Spectrum y Trends.",
    ),

    # ---------- RECIPROCANTES (medición de velocidad mm/s, no rotor dynamics) ----------
    MachineProfile(
        key="reciprocating_compressor",
        label="Reciprocating compressor",
        category="reciprocating",
        iso_part="20816-8",
        machine_group="recip_compressor",
        operating_rpm=600.0,
        bearing_type="plain",
        rated_power_mw_min=0.1,
        rated_power_mw_max=20.0,
        applicable_modules=["spectrum", "trends", "waveform"],
        threshold_strategy="iso",
        notes=(
            "Compresor reciprocante. ISO 20816-8 evalúa velocidad RMS (mm/s) sobre carcasa de "
            "cojinetes. NO aplica análisis rotodinámico clásico de Q factor / críticas — la "
            "vibración inherente por cargas alternantes domina. Usa Spectrum (harmonics 1X/2X/3X "
            "del cigüeñal) y Trends para evaluación correcta."
        ),
    ),
    MachineProfile(
        key="reciprocating_engine",
        label="Reciprocating engine (gas / diesel)",
        category="reciprocating",
        iso_part="20816-6",
        machine_group="recip_engine",
        operating_rpm=900.0,
        bearing_type="plain",
        rated_power_mw_min=0.1,
        rated_power_mw_max=20.0,
        applicable_modules=["spectrum", "trends", "waveform"],
        threshold_strategy="iso",
        notes=(
            "Motor reciprocante. ISO 20816-6 evalúa vibración de carcasa. NO aplica detección "
            "rotodinámica de críticas — la firma vibratoria está dominada por la combustión y "
            "el cigüeñal. Usa Spectrum y Trends para análisis adecuado."
        ),
    ),

    # ---------- CUSTOM / MANUAL ----------
    MachineProfile(
        key="custom_manual",
        label="Custom (manual: OEM / fabricante / data baseline / user-defined)",
        category="custom",
        iso_part="custom",
        machine_group="custom",
        operating_rpm=3600.0,
        bearing_type="plain",
        rated_power_mw_min=0.0,
        rated_power_mw_max=1000.0,
        applicable_modules=["polar", "bode", "shaft_centerline", "spectrum", "trends", "orbit", "phase", "tsa", "envelope"],
        threshold_strategy="custom",
        notes=(
            "Modo manual completo: el usuario define operating_rpm, parte ISO, umbrales A/B/C/D "
            "directamente o usa los valores recomendados por el fabricante (Brush, Siemens, GE, "
            "Bently Nevada API 670, etc.) o un baseline estadístico de tendencias propias del "
            "activo (μ + n·σ)."
        ),
    ),
]


PROFILES: Dict[str, MachineProfile] = {p.key: p for p in _PROFILES_LIST}
DEFAULT_PROFILE_KEY = "brush_turbogenerator_54mw_3600"


def list_profile_options() -> List[Dict[str, str]]:
    """
    Lista de opciones para UI dropdown. Cada item es {key, label, category}.
    Ordenado por categoría natural.
    """
    return [
        {"key": p.key, "label": p.label, "category": p.category}
        for p in _PROFILES_LIST
    ]


def get_profile(key: str) -> MachineProfile:
    """Devuelve el profile por key. Si no existe, retorna el default."""
    return PROFILES.get(key, PROFILES[DEFAULT_PROFILE_KEY])


def is_module_applicable(profile_key: str, module_name: str) -> bool:
    """¿El módulo `module_name` (polar/bode/scl/etc.) es aplicable al profile?"""
    p = get_profile(profile_key)
    return module_name in p.applicable_modules


def module_not_applicable_message(profile_key: str, module_name: str) -> str:
    """
    Mensaje técnico explicando por qué un módulo no aplica al profile actual,
    y qué módulo alternativo recomienda. Para mostrarse cuando el usuario abre
    una página que no corresponde al tipo de máquina elegido.
    """
    p = get_profile(profile_key)
    module_label = {
        "polar": "Polar Plot",
        "bode": "Bode Plot",
        "shaft_centerline": "Shaft Centerline",
        "spectrum": "Spectrum",
        "trends": "Trends",
        "orbit": "Orbit Analysis",
        "phase": "Phase Analysis",
        "tsa": "Time Synchronous Average",
        "envelope": "Envelope Spectrum",
    }.get(module_name, module_name)

    if p.category == "reciprocating":
        return (
            f"El profile '{p.label}' corresponde a una máquina reciprocante. "
            f"El módulo {module_label} se basa en análisis rotodinámico clásico "
            f"(Q factor, velocidades críticas, órbita filtrada), que NO es "
            f"técnicamente apropiado para reciprocantes según ISO 20816-{'6' if 'engine' in p.key else '8'}. "
            f"Recomendamos usar Spectrum (para evaluar harmonics 1X/2X/3X del cigüeñal) y Trends "
            f"(para tendencia de severidad RMS de carcasa). Cambia el profile o navega a Spectrum/Trends."
        )

    if p.category == "pump" and module_name in ("polar", "bode") and "vertical_turbine" in p.key:
        return (
            f"El profile '{p.label}' corresponde a una bomba vertical de turbina (pozo profundo). "
            f"Este tipo de máquina tiene una columna de bombeo larga con múltiples etapas, lo que "
            f"hace que el análisis Polar/Bode de modos rotodinámicos sea limitado. Recomendamos "
            f"Spectrum y Trends como herramientas primarias."
        )

    return (
        f"El profile '{p.label}' no incluye '{module_label}' en sus módulos aplicables. "
        f"Cambia el profile o consulta los módulos recomendados: "
        f"{', '.join(p.applicable_modules)}."
    )


__all__ = [
    "MachineProfile",
    "PROFILES",
    "DEFAULT_PROFILE_KEY",
    "list_profile_options",
    "get_profile",
    "is_module_applicable",
    "module_not_applicable_message",
]
