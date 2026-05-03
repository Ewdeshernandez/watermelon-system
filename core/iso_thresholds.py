"""
core.iso_thresholds
===================

Catálogo central de normas industriales de evaluación de vibración
(ISO / API) y sus tablas de setpoints. Single source of truth para:

  - Sugerencia automática de Warning/Danger según norma + class
  - UI de selección de norma en Machinery Library
  - Cita normativa en el PDF de Reports

Uso típico:

    from core.iso_thresholds import (
        list_norms, list_classes_for_norm, get_thresholds,
    )

    # En el editor de la instancia:
    norms = list_norms()  # [{"code": "ISO_20816_8", "name": ..., ...}, ...]
    classes = list_classes_for_norm("ISO_20816_8")
    info = get_thresholds("ISO_20816_8", "2")
    # info = {
    #     "warning": 7.1, "danger": 17.8, "unit": "mm/s",
    #     "metric": "velocity_rms", "label": "Class 2 ...",
    #     "source_label": "ISO 20816-8 Class 2",
    #     "reference": "Tabla 3 ISO 20816-8:2018 Annex A",
    # }

Convención de campos:

  metric:       'velocity_rms' | 'displacement_pp' | 'acceleration_rms'
  unit:         'mm/s' | 'mil pp' | 'µm pp' | 'g RMS' | 'm/s² RMS'

Las clases se almacenan en orden de display deseado (no alfabético).

Para agregar una norma nueva: añadir entrada en _ISO_NORMS conforme
al schema. La UI y los consumidores leen de aquí automáticamente.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# =============================================================
# CATÁLOGO
# =============================================================
# Normas más usadas en O&G / generación / petroquímica para
# evaluación de severidad de vibración. Las tablas son de las
# ediciones vigentes (2014–2018 según norma).
#
# Para cada norma:
#   name            — display name corto
#   long_name       — nombre formal completo
#   applies_to      — tipo de máquina cubierto
#   metric          — qué se mide (velocity / displacement / accel)
#   unit            — unidad de los setpoints
#   reference       — cita formal para el PDF
#   classes         — dict de clases con sus setpoints

_ISO_NORMS: Dict[str, Dict[str, Any]] = {

    # =========================================================
    # ISO 20816-8 (ex 10816-8) — Compresores reciprocantes
    # =========================================================
    "ISO_20816_8": {
        "name": "ISO 20816-8 — Compresores reciprocantes",
        "long_name": "ISO 20816-8:2018 — Mechanical vibration · "
                     "Reciprocating compressor systems",
        "applies_to": "Compresores reciprocantes (frame + cilindros). "
                      "Aplicable a TODAS las potencias.",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "Tabla 3 ISO 20816-8:2018 Annex A",
        "classes": [
            {
                "code": "1",
                "label": "Class 1 — Pequeños (<200 kW, alta velocidad)",
                "warning": 4.5,
                "danger": 11.2,
            },
            {
                "code": "2",
                "label": "Class 2 — Medios (200-1000 kW, industrial estándar)",
                "warning": 7.1,
                "danger": 17.8,
            },
            {
                "code": "3",
                "label": "Class 3 — Grandes (>1000 kW, gas service)",
                "warning": 11.2,
                "danger": 28.2,
            },
            {
                "code": "4",
                "label": "Class 4 — Especiales (alto torque, baja velocidad)",
                "warning": 17.8,
                "danger": 44.6,
            },
        ],
    },

    # =========================================================
    # ISO 20816-2 (ex 10816-2) — Turbinas de vapor + generadores
    # =========================================================
    "ISO_20816_2": {
        "name": "ISO 20816-2 — Turbinas de vapor + generadores grandes",
        "long_name": "ISO 20816-2:2017 — Land-based steam turbines and "
                     "generators in excess of 50 MW",
        "applies_to": "Turbinas de vapor y generadores acoplados >50 MW "
                      "(ej. Brush turbogenerator).",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "Tablas A.1 / A.2 ISO 20816-2:2017",
        "classes": [
            {
                "code": "Zone_AB",
                "label": "Zona A/B — operación normal sostenida",
                "warning": 2.8,
                "danger": 7.5,
            },
            {
                "code": "Zone_C",
                "label": "Zona C — vigilancia / restricción tiempo",
                "warning": 7.5,
                "danger": 11.8,
            },
        ],
    },

    # =========================================================
    # ISO 20816-3 (ex 10816-3) — Máquinas industriales rotativas
    # =========================================================
    "ISO_20816_3": {
        "name": "ISO 20816-3 — Máquinas industriales 15 kW–50 MW",
        "long_name": "ISO 20816-3:2022 — Industrial machinery with power "
                     "above 15 kW and operating speeds between 120 r/min "
                     "and 30 000 r/min",
        "applies_to": "Bombas, compresores centrífugos, motores, ventiladores, "
                      "etc. con potencia 15 kW – 50 MW.",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "Tabla A.1 ISO 20816-3:2022",
        "classes": [
            {
                "code": "Group1_Rigid",
                "label": "Group 1 (>300 kW) · Soporte rígido",
                "warning": 4.5,
                "danger": 11.2,
            },
            {
                "code": "Group1_Flex",
                "label": "Group 1 (>300 kW) · Soporte flexible",
                "warning": 7.1,
                "danger": 18.0,
            },
            {
                "code": "Group2_Rigid",
                "label": "Group 2 (15-300 kW) · Soporte rígido",
                "warning": 2.8,
                "danger": 7.1,
            },
            {
                "code": "Group2_Flex",
                "label": "Group 2 (15-300 kW) · Soporte flexible",
                "warning": 4.5,
                "danger": 11.2,
            },
        ],
    },

    # =========================================================
    # ISO 20816-4 (ex 10816-4) — Turbinas de gas
    # =========================================================
    "ISO_20816_4": {
        "name": "ISO 20816-4 — Turbinas de gas",
        "long_name": "ISO 20816-4:2018 — Gas turbine sets with fluid-film "
                     "bearings",
        "applies_to": "Turbinas de gas industriales y aero-derivativas "
                      "(LM6000, LM2500, Frame, SGT, etc.).",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "Tabla 1 ISO 20816-4:2018",
        "classes": [
            {
                "code": "Coupled",
                "label": "Driver-Driven coupled (estándar)",
                "warning": 4.5,
                "danger": 9.3,
            },
            {
                "code": "Aero",
                "label": "Aero-derivativo (LM6000, LM2500)",
                "warning": 7.1,
                "danger": 11.0,
            },
        ],
    },

    # =========================================================
    # API 670 — Proximity Probes (cojinetes hidrodinámicos)
    # =========================================================
    "API_670": {
        "name": "API 670 — Proximity Probes (cojinetes planos)",
        "long_name": "API 670 7th edition — Machinery Protection Systems",
        "applies_to": "Vibración relativa al eje medida con sondas de "
                      "proximidad sobre cojinetes hidrodinámicos planos.",
        "metric": "displacement_pp",
        "unit": "mil pp",
        "reference": "API 670 §6.7 + Bently Nevada 3500 default setpoints",
        "classes": [
            {
                "code": "Default",
                "label": "Default Bently 3500 (industrial estándar)",
                "warning": 3.0,
                "danger": 5.0,
            },
            {
                "code": "Conservative",
                "label": "Conservador (cliente exigente, máquina nueva)",
                "warning": 2.5,
                "danger": 4.0,
            },
            {
                "code": "Tolerant",
                "label": "Tolerante (máquina antigua con baseline alto)",
                "warning": 4.0,
                "danger": 7.0,
            },
        ],
    },

    # =========================================================
    # API 618 — Compresores reciprocantes refinería
    # =========================================================
    "API_618": {
        "name": "API 618 — Compresores recip refinería",
        "long_name": "API 618 5th edition — Reciprocating Compressors for "
                     "Petroleum, Chemical, and Gas Industry Services",
        "applies_to": "Compresores reciprocantes para servicio de petróleo, "
                      "química y gas (norma más exigente que ISO 20816-8).",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "API 618 5th ed §7.7 + Annex H",
        "classes": [
            {
                "code": "Frame",
                "label": "Frame top (cuerpo del crankcase)",
                "warning": 9.0,
                "danger": 14.0,
            },
            {
                "code": "Cylinder",
                "label": "Cylinder body (cuerpo del cilindro)",
                "warning": 12.7,
                "danger": 20.0,
            },
            {
                "code": "Pulsation",
                "label": "Pulsation bottle / piping",
                "warning": 14.0,
                "danger": 25.0,
            },
        ],
    },

    # =========================================================
    # ISO 10816-6 (legacy) — Motores diésel
    # =========================================================
    "ISO_10816_6": {
        "name": "ISO 10816-6 — Motores diésel marinos / industriales",
        "long_name": "ISO 10816-6:1995/Amd 1:2015 — Reciprocating machines "
                     "(EXCLUYE compresores reciprocantes desde 2015)",
        "applies_to": "Motores diésel marinos, generadores diésel, bombas "
                      "reciprocantes >100 kW. NO compresores reciprocantes "
                      "(esos van por ISO 20816-8).",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "Tabla A.1 ISO 10816-6:1995/Amd 1:2015",
        "classes": [
            {
                "code": "Class_3",
                "label": "Class 3 — diésel marino mediano",
                "warning": 11.2,
                "danger": 18.0,
            },
            {
                "code": "Class_4",
                "label": "Class 4 — diésel marino grande",
                "warning": 18.0,
                "danger": 28.2,
            },
            {
                "code": "Class_5",
                "label": "Class 5 — diésel industrial estacionario",
                "warning": 28.2,
                "danger": 44.6,
            },
        ],
    },
}


# =============================================================
# API PÚBLICA
# =============================================================

def list_norms() -> List[Dict[str, Any]]:
    """Lista todas las normas disponibles. Devuelve summaries."""
    out = []
    for code, data in _ISO_NORMS.items():
        out.append({
            "code": code,
            "name": data["name"],
            "applies_to": data["applies_to"],
            "metric": data["metric"],
            "unit": data["unit"],
            "n_classes": len(data["classes"]),
        })
    return out


def list_classes_for_norm(norm_code: str) -> List[Dict[str, Any]]:
    """Devuelve las clases de una norma con sus setpoints.
    Cada clase tiene: code, label, warning, danger.
    """
    data = _ISO_NORMS.get(norm_code)
    if not data:
        return []
    return list(data["classes"])


def get_thresholds(norm_code: str, class_code: str) -> Optional[Dict[str, Any]]:
    """Devuelve setpoints + metadata para (norma, clase).

    Args:
        norm_code:  código de norma, ej. 'ISO_20816_8'
        class_code: código de clase dentro de la norma, ej. '2'

    Returns:
        dict {warning, danger, unit, metric, label, source_label,
              reference, applies_to} o None si no existe.
    """
    data = _ISO_NORMS.get(norm_code)
    if not data:
        return None
    cls = next((c for c in data["classes"] if c["code"] == class_code), None)
    if not cls:
        return None

    short_name = data["name"].split("—")[0].strip()
    return {
        "warning": float(cls["warning"]),
        "danger": float(cls["danger"]),
        "unit": data["unit"],
        "metric": data["metric"],
        "label": cls["label"],
        "source_label": f"{short_name} {cls['code']}",
        "reference": data["reference"],
        "applies_to": data["applies_to"],
        "norm_long_name": data["long_name"],
    }


def get_norm_metadata(norm_code: str) -> Optional[Dict[str, Any]]:
    """Devuelve metadata de la norma (sin clases)."""
    data = _ISO_NORMS.get(norm_code)
    if not data:
        return None
    return {
        "code": norm_code,
        "name": data["name"],
        "long_name": data["long_name"],
        "applies_to": data["applies_to"],
        "metric": data["metric"],
        "unit": data["unit"],
        "reference": data["reference"],
    }


def suggest_norm_for_machine(asset_class: str, driver_kind: str = "",
                              driven_kind: str = "") -> Optional[str]:
    """Heurística para sugerir la norma más probable según el tipo
    de activo. Devuelve el norm_code o None si no hay sugerencia
    clara.

    Reglas (en orden):
        - turbogenerador / steam turbine + generator → ISO_20816_2
        - aero turbine (LM6000, LM2500) → ISO_20816_4
        - reciprocating compressor (ARIEL, KBK) → ISO_20816_8
        - motor + bomba / centrifugal compressor → ISO_20816_3
        - proximity probes / cojinetes planos → API_670
        - default → None (usuario elige manualmente)
    """
    txt = " ".join([asset_class or "", driver_kind or "", driven_kind or ""]).lower()
    if not txt.strip():
        return None
    # Recip primero (es el más específico y NO debe caer en otro)
    if "recip" in txt or "ariel" in txt or "kbk" in txt or "dresser" in txt:
        return "ISO_20816_8"
    # Gas turbines aero-derivativas (LM6000, LM2500, etc.) — ANTES de
    # turbogenerador porque LM6000 dentro de un turbogen va por 20816-4.
    if (
        "lm6000" in txt or "lm2500" in txt or "tm2500" in txt
        or "aero" in txt or "frame " in txt or "sgt-" in txt
        or "turbina de gas" in txt or "gas turbine" in txt
    ):
        return "ISO_20816_4"
    # Steam turbine + generador grande (Brush, Siemens vapor, etc.)
    if (
        "vapor" in txt and "turbina" in txt
    ) or ("steam turbine" in txt) or (
        "turbogen" in txt and "vapor" in txt
    ):
        return "ISO_20816_2"
    # Centrífugos / bombas / motores industriales rotativos
    if (
        "centrif" in txt or "bomba" in txt or "pump" in txt
        or "ventilador" in txt or "fan " in txt or "motor" in txt
    ):
        return "ISO_20816_3"
    # Turbogenerador genérico (sin tipo de turbina especificado) → 20816-4
    # como guess más seguro (la mayoría son gas turbines hoy).
    if "turbogen" in txt or "turbina" in txt or "turbine" in txt:
        return "ISO_20816_4"
    return None


def suggest_class_for_machine(norm_code: str, power_kw: float = 0.0,
                               support_type: str = "") -> Optional[str]:
    """Heurística para sugerir la clase dentro de una norma según
    potencia + tipo de soporte. Devuelve class_code o None."""
    if norm_code == "ISO_20816_8":
        if power_kw <= 0:
            return "2"  # default industrial
        if power_kw < 200:
            return "1"
        if power_kw < 1000:
            return "2"
        return "3"
    if norm_code == "ISO_20816_3":
        is_rigid = "rigid" in (support_type or "").lower() or "plano" in (support_type or "").lower()
        if power_kw >= 300:
            return "Group1_Rigid" if is_rigid else "Group1_Flex"
        return "Group2_Rigid" if is_rigid else "Group2_Flex"
    if norm_code == "ISO_20816_2":
        return "Zone_AB"
    if norm_code == "ISO_20816_4":
        return "Coupled"
    if norm_code == "API_670":
        return "Default"
    if norm_code == "API_618":
        return "Frame"
    if norm_code == "ISO_10816_6":
        return "Class_3"
    return None


__all__ = [
    "list_norms",
    "list_classes_for_norm",
    "get_thresholds",
    "get_norm_metadata",
    "suggest_norm_for_machine",
    "suggest_class_for_machine",
]
