"""
core.iso_thresholds
===================

Catálogo central de normas industriales de evaluación de vibración,
rotodinámica y balanceo (ISO / API / IEC / ANSI / VDI / NEMA) y sus
tablas de setpoints. Single source of truth para:

  - Sugerencia automática de Warning/Danger según norma + class
  - UI de selección de norma en Machinery Library
  - Cita normativa en el PDF de Reports

Uso típico:

    from core.iso_thresholds import (
        list_norms, list_classes_for_norm, get_thresholds,
    )

    norms   = list_norms()                    # summaries
    classes = list_classes_for_norm("ISO_20816_8")
    info    = get_thresholds("ISO_20816_8", "2")

Convención de campos:

  metric:       'velocity_rms' | 'velocity_pk' | 'displacement_pp'
                'acceleration_rms' | 'unbalance_grade' |
                'amplification_factor'
  unit:         'mm/s' | 'mm/s pk' | 'mil pp' | 'µm pp' | 'g RMS' |
                'm/s² RMS' | 'G grade (g·mm/kg · ω)' | 'AF (dim.)'

Las clases se almacenan en orden de display deseado.

Para agregar una norma nueva: añadir entrada en _ISO_NORMS conforme
al schema. La UI y los consumidores leen de aquí automáticamente.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# =============================================================
# CATÁLOGO
# =============================================================
# Normas más usadas en O&G / generación / petroquímica / proceso
# para evaluación de severidad de vibración, balanceo de rotores
# y análisis rotodinámico. Tablas de las ediciones vigentes.

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
            {"code": "1", "label": "Class 1 — Pequeños (<200 kW, alta velocidad)",        "warning": 4.5,  "danger": 11.2},
            {"code": "2", "label": "Class 2 — Medios (200-1000 kW, industrial estándar)", "warning": 7.1,  "danger": 17.8},
            {"code": "3", "label": "Class 3 — Grandes (>1000 kW, gas service)",           "warning": 11.2, "danger": 28.2},
            {"code": "4", "label": "Class 4 — Especiales (alto torque, baja velocidad)",  "warning": 17.8, "danger": 44.6},
        ],
    },

    # =========================================================
    # ISO 20816-2 (ex 10816-2) — Turbinas de vapor + generadores
    # =========================================================
    "ISO_20816_2": {
        "name": "ISO 20816-2 — Turbinas de vapor + generadores grandes",
        "long_name": "ISO 20816-2:2017 — Land-based steam turbines and "
                     "generators in excess of 50 MW with normal operating "
                     "speeds of 1500, 1800, 3000 and 3600 r/min",
        "applies_to": "Turbinas de vapor y generadores acoplados >50 MW "
                      "(ej. Brush turbogenerator).",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "Tablas A.1 / A.2 ISO 20816-2:2017",
        "classes": [
            {"code": "Zone_AB", "label": "Zona A/B — operación normal sostenida",      "warning": 2.8, "danger": 7.5},
            {"code": "Zone_C",  "label": "Zona C — vigilancia / restricción tiempo",   "warning": 7.5, "danger": 11.8},
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
        "applies_to": "Bombas, compresores centrífugos, motores, "
                      "ventiladores con potencia 15 kW – 50 MW.",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "Tabla A.1 ISO 20816-3:2022",
        "classes": [
            {"code": "Group1_Rigid", "label": "Group 1 (>300 kW) · Soporte rígido",   "warning": 4.5, "danger": 11.2},
            {"code": "Group1_Flex",  "label": "Group 1 (>300 kW) · Soporte flexible", "warning": 7.1, "danger": 18.0},
            {"code": "Group2_Rigid", "label": "Group 2 (15-300 kW) · Soporte rígido", "warning": 2.8, "danger": 7.1},
            {"code": "Group2_Flex",  "label": "Group 2 (15-300 kW) · Soporte flexible","warning": 4.5, "danger": 11.2},
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
            {"code": "Coupled", "label": "Driver-Driven coupled (estándar)",  "warning": 4.5, "danger": 9.3},
            {"code": "Aero",    "label": "Aero-derivativo (LM6000, LM2500)",  "warning": 7.1, "danger": 11.0},
        ],
    },

    # =========================================================
    # ISO 20816-5 — Bombas hidráulicas / hidroeléctrica
    # =========================================================
    "ISO_20816_5": {
        "name": "ISO 20816-5 — Hidroeléctricas + bombas hidráulicas",
        "long_name": "ISO 20816-5:2018 — Machine sets in hydraulic power "
                     "generating and pump-storage plants",
        "applies_to": "Turbinas hidráulicas (Francis, Kaplan, Pelton), "
                      "bombas-turbina, generadores hidráulicos.",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "Tabla 2 ISO 20816-5:2018",
        "classes": [
            {"code": "Group1_HighSpeed", "label": "Group 1 — >300 r/min (cojinetes guía)",    "warning": 2.5,  "danger": 6.4},
            {"code": "Group2_LowSpeed",  "label": "Group 2 — ≤300 r/min (cojinetes guía)",    "warning": 3.8,  "danger": 9.5},
            {"code": "Vertical_Bracket", "label": "Estructura/araña vertical (housing)",      "warning": 5.0,  "danger": 12.5},
        ],
    },

    # =========================================================
    # ISO 10816-7 — Bombas rotodinámicas
    # =========================================================
    "ISO_10816_7": {
        "name": "ISO 10816-7 — Bombas rotodinámicas (centrífugas)",
        "long_name": "ISO 10816-7:2009 — Rotodynamic pumps for industrial "
                     "applications (input power above 1 kW)",
        "applies_to": "Bombas centrífugas industriales horizontales y "
                      "verticales (>1 kW). Más específica que 20816-3 "
                      "para bombas.",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "Tabla A.2 ISO 10816-7:2009",
        "classes": [
            {"code": "Cat_I_Rigid", "label": "Categoría I (uso crítico) · Soporte rígido",    "warning": 3.5, "danger": 5.0},
            {"code": "Cat_I_Flex",  "label": "Categoría I (uso crítico) · Soporte flexible",  "warning": 5.0, "danger": 7.5},
            {"code": "Cat_II_Rigid","label": "Categoría II (uso general) · Soporte rígido",   "warning": 4.5, "danger": 7.1},
            {"code": "Cat_II_Flex", "label": "Categoría II (uso general) · Soporte flexible", "warning": 6.5, "danger": 9.0},
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
            {"code": "Class_3", "label": "Class 3 — diésel marino mediano",            "warning": 11.2, "danger": 18.0},
            {"code": "Class_4", "label": "Class 4 — diésel marino grande",             "warning": 18.0, "danger": 28.2},
            {"code": "Class_5", "label": "Class 5 — diésel industrial estacionario",   "warning": 28.2, "danger": 44.6},
        ],
    },

    # =========================================================
    # ISO 14694 — Ventiladores industriales (BV grades)
    # =========================================================
    "ISO_14694": {
        "name": "ISO 14694 — Ventiladores industriales",
        "long_name": "ISO 14694:2003 + Amd 1:2010 — Industrial fans · "
                     "Specifications for balance quality and vibration "
                     "levels",
        "applies_to": "Ventiladores axiales, centrífugos, in-line, mixed "
                      "flow para servicio industrial.",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "Tabla 2 ISO 14694:2003 (BV-1 a BV-5)",
        "classes": [
            {"code": "BV_3_Rigid", "label": "BV-3 (proceso general) · Rigid mount",   "warning": 4.5,  "danger": 7.1},
            {"code": "BV_3_Flex",  "label": "BV-3 (proceso general) · Flex mount",    "warning": 6.3,  "danger": 11.8},
            {"code": "BV_4_Rigid", "label": "BV-4 (alta exigencia) · Rigid mount",    "warning": 2.8,  "danger": 4.5},
            {"code": "BV_4_Flex",  "label": "BV-4 (alta exigencia) · Flex mount",     "warning": 3.5,  "danger": 7.1},
            {"code": "BV_5_Rigid", "label": "BV-5 (crítico, sub-marino) · Rigid",     "warning": 1.8,  "danger": 2.8},
            {"code": "BV_5_Flex",  "label": "BV-5 (crítico, sub-marino) · Flex",      "warning": 2.8,  "danger": 4.5},
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
            {"code": "Default",     "label": "Default Bently 3500 (industrial estándar)",   "warning": 3.0, "danger": 5.0},
            {"code": "Conservative","label": "Conservador (cliente exigente, máquina nueva)","warning": 2.5, "danger": 4.0},
            {"code": "Tolerant",    "label": "Tolerante (máquina antigua con baseline alto)","warning": 4.0, "danger": 7.0},
            {"code": "API670_Eq",   "label": "API 670 Eq.1 (12000/N) µm·pp ref 12k rpm",    "warning": 50.0,"danger": 75.0},
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
            {"code": "Frame",     "label": "Frame top (cuerpo del crankcase)",   "warning": 9.0,  "danger": 14.0},
            {"code": "Cylinder",  "label": "Cylinder body (cuerpo del cilindro)","warning": 12.7, "danger": 20.0},
            {"code": "Pulsation", "label": "Pulsation bottle / piping",          "warning": 14.0, "danger": 25.0},
        ],
    },

    # =========================================================
    # API 619 — Compresores rotativos de desplazamiento positivo
    # =========================================================
    "API_619": {
        "name": "API 619 — Compresores rotativos PD (tornillo, lóbulos)",
        "long_name": "API 619 5th edition — Rotary-Type Positive-Displacement "
                     "Compressors for Petroleum, Petrochemical, and Natural "
                     "Gas Industries",
        "applies_to": "Compresores tipo tornillo, lóbulos (Roots), paletas "
                      "para servicio API.",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "API 619 5th ed §6.8 + Annex C",
        "classes": [
            {"code": "Housing_Op", "label": "Housing en operación (FAT/SAT)",     "warning": 2.8, "danger": 4.5},
            {"code": "Shaft_API670","label":"Eje (proximity probe per API 670)",  "warning": 2.0, "danger": 3.0},
        ],
    },

    # =========================================================
    # API 617 — Compresores axiales y centrífugos
    # =========================================================
    "API_617": {
        "name": "API 617 — Compresores axiales y centrífugos",
        "long_name": "API 617 8th edition — Axial and Centrifugal "
                     "Compressors and Expander-compressors",
        "applies_to": "Compresores centrífugos y axiales para servicio "
                      "API (refinería, petroquímica, gas).",
        "metric": "displacement_pp",
        "unit": "µm pp",
        "reference": "API 617 8th ed §2.6 + Annex E (UR test)",
        "classes": [
            {"code": "Vib_API_Limit", "label": "Vibración eje límite API (25.4·√(12000/N))", "warning": 25.4, "danger": 50.8},
            {"code": "Housing_Acc",   "label": "Housing acceptance (FAT)",                  "warning": 2.8,  "danger": 4.5},
            {"code": "Housing_Field", "label": "Housing en campo (alarm)",                  "warning": 4.5,  "danger": 7.1},
        ],
    },

    # =========================================================
    # API 612 — Turbinas de vapor especiales
    # =========================================================
    "API_612": {
        "name": "API 612 — Turbinas de vapor especiales",
        "long_name": "API 612 7th edition — Petroleum, Petrochemical and "
                     "Natural Gas Industries · Steam Turbines · Special-"
                     "purpose Applications",
        "applies_to": "Turbinas de vapor de propósito especial para drive "
                      "de bombas/compresores en servicio API (más exigente "
                      "que ISO 20816-2).",
        "metric": "displacement_pp",
        "unit": "µm pp",
        "reference": "API 612 7th ed §2.8.2 + 6.8",
        "classes": [
            {"code": "Vib_Rotor", "label": "Eje (proximity probe per API 670)",  "warning": 25.4, "danger": 50.8},
            {"code": "Housing",   "label": "Housing/casing (acceptance)",        "warning": 2.8,  "danger": 4.5},
        ],
    },

    # =========================================================
    # API 611 — Turbinas de vapor de propósito general
    # =========================================================
    "API_611": {
        "name": "API 611 — Turbinas vapor propósito general",
        "long_name": "API 611 6th edition — General-Purpose Steam Turbines "
                     "for Petroleum, Chemical, and Gas Industry Services",
        "applies_to": "Turbinas vapor de propósito general (utility, drive "
                      "no crítico). Menos exigente que API 612.",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "API 611 6th ed §2.8",
        "classes": [
            {"code": "Housing_Shop",  "label": "Housing en taller (FAT)",       "warning": 4.5, "danger": 7.1},
            {"code": "Housing_Field", "label": "Housing en campo",              "warning": 7.1, "danger": 11.2},
        ],
    },

    # =========================================================
    # API 610 — Bombas centrífugas (refinería)
    # =========================================================
    "API_610": {
        "name": "API 610 — Bombas centrífugas refinería",
        "long_name": "API 610 12th edition / ISO 13709:2009 — Centrifugal "
                     "Pumps for Petroleum, Petrochemical and Natural Gas "
                     "Industries",
        "applies_to": "Bombas centrífugas API (OH, BB, VS) para servicio "
                      "de refinería y petroquímica.",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "API 610 12th ed Tabla 9 (vibración filtrada/no filtrada)",
        "classes": [
            {"code": "Horizontal_Std",   "label": "Horizontal (OH/BB) · BEP ±10%",  "warning": 3.0,  "danger": 5.0},
            {"code": "Horizontal_Off",   "label": "Horizontal · Fuera de BEP",      "warning": 4.0,  "danger": 6.5},
            {"code": "Vertical_Std",     "label": "Vertical (VS) · BEP ±10%",       "warning": 5.0,  "danger": 8.0},
            {"code": "Vertical_Off",     "label": "Vertical · Fuera de BEP",        "warning": 7.0,  "danger": 10.5},
        ],
    },

    # =========================================================
    # API 671 — Couplings de propósito especial
    # =========================================================
    "API_671": {
        "name": "API 671 — Couplings propósito especial",
        "long_name": "API 671 4th edition / ISO 10441 — Special-purpose "
                     "Couplings for Petroleum, Chemical, and Gas Industry "
                     "Services",
        "applies_to": "Couplings flexibles disco/diafragma para acoples "
                      "críticos (turbo-compresor, turbogen). Define "
                      "límites de unbalance residual del ensamble.",
        "metric": "unbalance_grade",
        "unit": "g·mm",
        "reference": "API 671 4th ed §3.3 + Annex H (W·N²/9550)",
        "classes": [
            {"code": "Standard",   "label": "Standard balance (4 W/N · oz·in)",     "warning": 4.0,  "danger": 8.0},
            {"code": "Precision",  "label": "Precision (1 W/N · oz·in)",            "warning": 1.0,  "danger": 2.0},
        ],
    },

    # =========================================================
    # API 684 — Análisis Rotodinámico (CRÍTICA)
    # =========================================================
    "API_684": {
        "name": "API 684 — Rotordynamic Tutorial (Q factor + SM)",
        "long_name": "API 684 2nd edition — API Standard Paragraphs · "
                     "Rotordynamic Tutorial: Lateral Critical Speeds, "
                     "Unbalance Response, Stability, Train Torsionals, "
                     "and Rotor Balancing",
        "applies_to": "Análisis rotodinámico de máquinas críticas: factor "
                      "de amplificación (AF/Q) en velocidades críticas, "
                      "márgenes de separación (SM) entre crítica y MCOS, "
                      "respuesta a desbalance, estabilidad, balanceo. "
                      "Usado en API 612, 617, 619, 672, etc.",
        "metric": "amplification_factor",
        "unit": "AF (dim.)",
        "reference": "API 684 2nd ed §2.6 + Fig 2.6.1 (AF vs SM chart)",
        "classes": [
            {"code": "AF_Critical_OK",      "label": "AF ≤ 2.5 — críticamente amortiguado (sin SM)", "warning": 2.5,  "danger": 3.55},
            {"code": "AF_Critical_SM",      "label": "AF 2.5–3.55 — requiere SM por chart",          "warning": 3.55, "danger": 5.0},
            {"code": "AF_Critical_Reject",  "label": "AF > 5.0 — rechazo (rediseño)",                "warning": 5.0,  "danger": 8.0},
            {"code": "Vib_UR_Test_Limit",   "label": "Test UR — vibración eje (Av = 25.4·√(12k/N))", "warning": 25.4, "danger": 50.8},
            {"code": "Vib_UR_Test_Margin",  "label": "Test UR — debe ser ≤75% de Av en MCOS",        "warning": 19.0, "danger": 25.4},
            {"code": "Stability_LogDec",    "label": "Log decrement δ — mínimo aceptable >0.1",      "warning": 0.1,  "danger": 0.0},
            {"code": "SM_Below_MCOS",       "label": "Separation margin BELOW MCOS — mínimo 15%",    "warning": 15.0, "danger": 10.0},
            {"code": "SM_Above_MTS",        "label": "Separation margin ABOVE MCOS — mínimo 20%",    "warning": 20.0, "danger": 15.0},
        ],
    },

    # =========================================================
    # ISO 21940-11 — BALANCEO DE ROTORES RÍGIDOS (G grades)
    # =========================================================
    # CRÍTICA según el usuario. Reemplaza ISO 1940-1 (1986).
    # G grade = e_per × ω (mm/s) donde e_per es el desbalance
    # específico permisible en g·mm/kg.
    # =========================================================
    "ISO_21940_11": {
        "name": "ISO 21940-11 — Balanceo rotores RÍGIDOS (G grades)",
        "long_name": "ISO 21940-11:2016 — Mechanical vibration · Rotor "
                     "balancing · Part 11: Procedures and tolerances for "
                     "rotors with rigid behaviour (reemplaza ISO 1940-1)",
        "applies_to": "Balanceo de rotores con comportamiento RÍGIDO "
                      "(rotores que NO cruzan su primera crítica en "
                      "operación). G-grade: G0.4 a G4000 según tipo de "
                      "máquina.",
        "metric": "unbalance_grade",
        "unit": "G grade (mm/s = g·mm/kg · ω)",
        "reference": "Tabla 1 ISO 21940-11:2016",
        "classes": [
            {"code": "G_0_4",   "label": "G 0.4 — Spindles, gyroscopios, óptica precisión", "warning": 0.4,    "danger": 1.0},
            {"code": "G_1_0",   "label": "G 1.0 — Spindles precisión, grinding, audio",     "warning": 1.0,    "danger": 2.5},
            {"code": "G_2_5",   "label": "G 2.5 — Turbinas vapor/gas, turbo-gen, turbo-compr","warning": 2.5,    "danger": 6.3},
            {"code": "G_6_3",   "label": "G 6.3 — Bombas centrif, motores eléctricos, fans", "warning": 6.3,    "danger": 16.0},
            {"code": "G_16",    "label": "G 16  — Drive shafts, crushers, agric, IC parts",  "warning": 16.0,   "danger": 40.0},
            {"code": "G_40",    "label": "G 40  — Car wheels, transmisiones, IC crankshafts","warning": 40.0,   "danger": 100.0},
            {"code": "G_100",   "label": "G 100 — Crankshafts diésel 4-cyl",                 "warning": 100.0,  "danger": 250.0},
            {"code": "G_250",   "label": "G 250 — Crankshafts grandes engines rigid mount",  "warning": 250.0,  "danger": 630.0},
            {"code": "G_630",   "label": "G 630 — Crankshafts large 2-stroke marine",        "warning": 630.0,  "danger": 1600.0},
            {"code": "G_1600",  "label": "G 1600 — Crankshafts large slow marine",           "warning": 1600.0, "danger": 4000.0},
            {"code": "G_4000",  "label": "G 4000 — Largest installations",                   "warning": 4000.0, "danger": 8000.0},
        ],
    },

    # =========================================================
    # ISO 21940-12 — BALANCEO DE ROTORES FLEXIBLES
    # =========================================================
    # CRÍTICA según el usuario. Para rotores que cruzan al menos
    # una velocidad crítica en operación. Metodología modal / por
    # planos de balance múltiples.
    # =========================================================
    "ISO_21940_12": {
        "name": "ISO 21940-12 — Balanceo rotores FLEXIBLES",
        "long_name": "ISO 21940-12:2016 — Mechanical vibration · Rotor "
                     "balancing · Part 12: Procedures and tolerances for "
                     "rotors with flexible behaviour",
        "applies_to": "Balanceo de rotores con comportamiento FLEXIBLE "
                      "(rotores que cruzan ≥1 velocidad crítica). Aplica "
                      "balance modal, low/high speed balance, plano por "
                      "plano. Usado en turbo-compresores, turbogen, etc.",
        "metric": "unbalance_grade",
        "unit": "G grade equivalente (mm/s)",
        "reference": "ISO 21940-12:2016 §6 + Annex A (clases A-D)",
        "classes": [
            {"code": "Class_A_LowSpeed",   "label": "Class A — Low speed balance (rotor rígido sub-crítico)",     "warning": 1.0,  "danger": 2.5},
            {"code": "Class_B_HighSpeed",  "label": "Class B — High speed balance plano por plano",                "warning": 1.0,  "danger": 2.5},
            {"code": "Class_C_Modal",      "label": "Class C — Modal balance (rotores muy flexibles, >1 crítica)", "warning": 0.7,  "danger": 1.5},
            {"code": "Class_D_Combined",   "label": "Class D — Combined low+high speed (turbo-gen grandes)",       "warning": 0.5,  "danger": 1.0},
            {"code": "Test_AtSpeed_OK",    "label": "Test at-speed: vibr eje ≤ 25.4·√(12000/N) µm·pp",             "warning": 25.4, "danger": 50.8},
        ],
    },

    # =========================================================
    # IEC 60034-14 — Vibración de máquinas eléctricas rotativas
    # =========================================================
    "IEC_60034_14": {
        "name": "IEC 60034-14 — Motores y generadores eléctricos",
        "long_name": "IEC 60034-14:2018 — Rotating electrical machines · "
                     "Mechanical vibration of certain machines with shaft "
                     "heights 56 mm and higher",
        "applies_to": "Motores eléctricos y generadores con altura de eje "
                      "≥56 mm. Define grados A (estándar), B (precisión), "
                      "C (alta precisión).",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "Tabla 1 IEC 60034-14:2018 (free-suspended)",
        "classes": [
            {"code": "GradeA_56_132",   "label": "Grade A · H 56-132 mm (motor pequeño)",         "warning": 1.6, "danger": 2.5},
            {"code": "GradeA_132_280",  "label": "Grade A · H 132-280 mm",                        "warning": 2.2, "danger": 3.5},
            {"code": "GradeA_gt280",    "label": "Grade A · H >280 mm (motor grande)",            "warning": 2.8, "danger": 4.5},
            {"code": "GradeB_56_132",   "label": "Grade B · H 56-132 mm (precisión)",             "warning": 0.7, "danger": 1.1},
            {"code": "GradeB_132_280",  "label": "Grade B · H 132-280 mm (precisión)",            "warning": 1.1, "danger": 1.8},
            {"code": "GradeB_gt280",    "label": "Grade B · H >280 mm (precisión)",               "warning": 1.8, "danger": 2.8},
        ],
    },

    # =========================================================
    # API 541 — Motores grandes squirrel cage form-wound
    # =========================================================
    "API_541": {
        "name": "API 541 — Motores grandes form-wound jaula ardilla",
        "long_name": "API 541 5th edition — Form-wound Squirrel-Cage "
                     "Induction Motors · 375 kW (500 hp) and Larger",
        "applies_to": "Motores asíncronos jaula ardilla form-wound "
                      "≥375 kW (500 hp) para servicio API.",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "API 541 5th ed §3.4.6 (vibration limits)",
        "classes": [
            {"code": "Housing_Shop",  "label": "Housing en taller (FAT, no acoplado)",  "warning": 2.3, "danger": 3.5},
            {"code": "Housing_Field", "label": "Housing en campo (instalado)",          "warning": 3.5, "danger": 5.6},
            {"code": "Shaft_Probe",   "label": "Eje con proximity (cojinetes planos)",  "warning": 38.0,"danger": 64.0},
        ],
    },

    # =========================================================
    # API 546 — Generadores síncronos sin escobillas
    # =========================================================
    "API_546": {
        "name": "API 546 — Generadores síncronos brushless",
        "long_name": "API 546 4th edition — Brushless Synchronous "
                     "Machines · 500 kVA and Larger",
        "applies_to": "Generadores y motores síncronos sin escobillas "
                      "≥500 kVA. Aplica al Brush 54 MW (TES1).",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "API 546 4th ed §3.5",
        "classes": [
            {"code": "Housing_Shop",  "label": "Housing taller (FAT)",     "warning": 2.5, "danger": 3.8},
            {"code": "Housing_Field", "label": "Housing campo",            "warning": 3.8, "danger": 6.0},
            {"code": "Shaft_Probe",   "label": "Eje proximity (cojinetes)","warning": 25.0,"danger": 50.0},
        ],
    },

    # =========================================================
    # NEMA MG-1 Part 7 — Vibración de motores eléctricos
    # =========================================================
    "NEMA_MG1_7": {
        "name": "NEMA MG-1 Part 7 — Motores AC integrales",
        "long_name": "NEMA MG-1:2016 Part 7 — Mechanical Vibration of "
                     "Integral Horsepower AC Motors",
        "applies_to": "Motores AC integrales (NEMA frame size). Norma "
                      "norteamericana, similar a IEC 60034-14.",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "Tabla 7-1 NEMA MG-1:2016 Part 7",
        "classes": [
            {"code": "Standard_Resilient", "label": "Standard · Mounting resilient (free)",   "warning": 2.5, "danger": 3.8},
            {"code": "Standard_Rigid",     "label": "Standard · Mounting rígido",              "warning": 1.9, "danger": 3.0},
            {"code": "SpecialPurpose",     "label": "Special purpose (proceso crítico)",       "warning": 1.5, "danger": 2.3},
        ],
    },

    # =========================================================
    # ANSI S2.41 / ASA — Reciprocating IC engines (legacy)
    # =========================================================
    "ANSI_S2_41": {
        "name": "ANSI S2.41 — Motores IC reciprocantes (legacy)",
        "long_name": "ANSI/ASA S2.41-1985 (R2015) — Mechanical Vibration "
                     "of Large Rotating Machines with Speed Range from "
                     "10 to 200 rev/s · Measurement and Evaluation",
        "applies_to": "Máquinas rotativas grandes 600-12000 rpm "
                      "(equivalente a ISO 7919/10816 antiguas).",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "ANSI S2.41-1985 (R2015)",
        "classes": [
            {"code": "Class_I",   "label": "Class I — Pequeñas <15 kW",         "warning": 1.8, "danger": 4.5},
            {"code": "Class_II",  "label": "Class II — Medianas 15-75 kW",      "warning": 2.8, "danger": 7.1},
            {"code": "Class_III", "label": "Class III — Grandes rigid foundation","warning": 4.5, "danger": 11.2},
            {"code": "Class_IV",  "label": "Class IV — Grandes flex foundation",  "warning": 7.1, "danger": 18.0},
        ],
    },

    # =========================================================
    # VDI 2056 — Norma alemana legacy (precursor ISO 10816)
    # =========================================================
    "VDI_2056": {
        "name": "VDI 2056 — Vibración mecánica (legacy DE)",
        "long_name": "VDI 2056:1964 — Beurteilungsmaßstäbe für "
                     "mechanische Schwingungen von Maschinen (precursor "
                     "histórico de ISO 2372 / 10816). Aún citada en "
                     "Europa para máquinas antiguas.",
        "applies_to": "Máquinas industriales legacy. Mantener para "
                      "compatibilidad con contratos viejos donde el "
                      "cliente aún cita VDI 2056.",
        "metric": "velocity_rms",
        "unit": "mm/s",
        "reference": "VDI 2056:1964 Tabla I-IV (Klasse K-T)",
        "classes": [
            {"code": "Klasse_K_Small",  "label": "Klasse K — Pequeñas <15 kW",      "warning": 2.8,  "danger": 7.1},
            {"code": "Klasse_M_Med",    "label": "Klasse M — Medianas 15-75 kW",    "warning": 4.5,  "danger": 11.2},
            {"code": "Klasse_G_Large",  "label": "Klasse G — Grandes rigid (>300 kW)","warning": 7.1,  "danger": 18.0},
            {"code": "Klasse_T_Turbo",  "label": "Klasse T — Turbo-máquinas grandes","warning": 11.2, "danger": 28.2},
        ],
    },

    # =========================================================
    # VDI 2059 — Vibración de eje (precursor ISO 7919)
    # =========================================================
    "VDI_2059": {
        "name": "VDI 2059 — Vibración de eje (legacy DE)",
        "long_name": "VDI 2059 · Wellenschwingungen — Shaft vibration "
                     "measurement and evaluation of large machines "
                     "(precursor ISO 7919, hoy ISO 20816).",
        "applies_to": "Vibración de eje en grandes turbomáquinas "
                      "(turbinas vapor, turbogen). Legacy europeo.",
        "metric": "displacement_pp",
        "unit": "µm pp",
        "reference": "VDI 2059 Blatt 1-5 (1981-1992)",
        "classes": [
            {"code": "Blatt1_Steam",   "label": "Blatt 1 — Turbinas vapor + gen <200 MW",   "warning": 165.0, "danger": 260.0},
            {"code": "Blatt2_Industrial","label": "Blatt 2 — Industriales generales",        "warning": 100.0, "danger": 165.0},
            {"code": "Blatt3_Hydro",   "label": "Blatt 3 — Hidráulicas",                     "warning": 200.0, "danger": 320.0},
        ],
    },
}


# =============================================================
# AGRUPACIONES POR DOMINIO (para UI con secciones)
# =============================================================
_NORM_GROUPS: Dict[str, List[str]] = {
    "Vibración (carcasa)": [
        "ISO_20816_2", "ISO_20816_3", "ISO_20816_4", "ISO_20816_5",
        "ISO_20816_8", "ISO_10816_6", "ISO_10816_7", "ISO_14694",
        "API_618", "API_619", "API_617", "API_612", "API_611",
        "API_610", "API_541", "API_546",
        "IEC_60034_14", "NEMA_MG1_7", "ANSI_S2_41", "VDI_2056",
    ],
    "Vibración de eje (proximity)": [
        "API_670", "VDI_2059",
    ],
    "Balanceo de rotor": [
        "ISO_21940_11", "ISO_21940_12", "API_671",
    ],
    "Análisis rotodinámico": [
        "API_684",
    ],
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


def list_norm_groups() -> Dict[str, List[Dict[str, Any]]]:
    """Devuelve normas agrupadas por dominio (UI con secciones).

    Estructura:
        {
          "Vibración (carcasa)": [{"code": ..., "name": ...}, ...],
          "Vibración de eje (proximity)": [...],
          "Balanceo de rotor": [...],
          "Análisis rotodinámico": [...],
        }
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for group, codes in _NORM_GROUPS.items():
        bucket: List[Dict[str, Any]] = []
        for c in codes:
            data = _ISO_NORMS.get(c)
            if not data:
                continue
            bucket.append({
                "code": c,
                "name": data["name"],
                "metric": data["metric"],
                "unit": data["unit"],
                "n_classes": len(data["classes"]),
            })
        out[group] = bucket
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
              reference, applies_to, norm_long_name} o None si no existe.
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
        - reciprocating compressor (ARIEL, KBK, Dresser) → ISO_20816_8
        - aero turbine (LM6000, LM2500, TM2500) → ISO_20816_4
        - gas turbine industrial (Frame, SGT) → ISO_20816_4
        - steam turbine + generador → ISO_20816_2
        - hidroeléctrica (Francis, Kaplan, Pelton) → ISO_20816_5
        - bomba centrífuga → ISO_10816_7 (más específica que 20816-3)
        - ventilador / fan → ISO_14694
        - motor eléctrico grande (>375 kW) → API_541
        - motor eléctrico estándar → IEC_60034_14
        - generador síncrono brushless → API_546
        - motor diésel → ISO_10816_6
        - centrifugal compressor → API_617 (si nombre API) o ISO_20816_3
        - default → None (usuario elige manualmente)
    """
    txt = " ".join([asset_class or "", driver_kind or "", driven_kind or ""]).lower()
    if not txt.strip():
        return None

    # 1) Compresor reciprocante (más específico, no debe caer en otro)
    if (
        "recip" in txt or "ariel" in txt or "kbk" in txt
        or "dresser" in txt or "ariel kbk" in txt
    ):
        return "ISO_20816_8"

    # 2) Turbinas aero-derivativas / industriales gas
    if (
        "lm6000" in txt or "lm2500" in txt or "tm2500" in txt
        or "aero" in txt or "frame " in txt or "sgt-" in txt
        or "turbina de gas" in txt or "gas turbine" in txt
        or "trent" in txt or "rb211" in txt
    ):
        return "ISO_20816_4"

    # 3) Turbina de vapor + generador grande
    if (
        ("vapor" in txt and "turbina" in txt)
        or "steam turbine" in txt
        or ("turbogen" in txt and "vapor" in txt)
    ):
        return "ISO_20816_2"

    # 4) Hidroeléctricas (Francis, Kaplan, Pelton, Bulb)
    if (
        "francis" in txt or "kaplan" in txt or "pelton" in txt
        or "hidroel" in txt or "hydro" in txt or "bulb" in txt
        or ("turbina" in txt and ("agua" in txt or "hidra" in txt))
    ):
        return "ISO_20816_5"

    # 5) Bomba centrífuga (más específico que 20816-3)
    if (
        "bomba" in txt or "pump" in txt or "centrif" in txt
    ) and (
        "compresor" not in txt and "compressor" not in txt
    ):
        # Si es bomba API (refinería) podríamos sugerir API_610,
        # pero como no lo sabemos aquí, vamos por ISO_10816_7.
        return "ISO_10816_7"

    # 6) Ventilador / fan
    if "ventilador" in txt or "fan " in txt or "blower" in txt:
        return "ISO_14694"

    # 7) Motor diésel
    if "diésel" in txt or "diesel" in txt or "motor diesel" in txt:
        return "ISO_10816_6"

    # 8) Generador síncrono brushless (Brush, Marathon)
    if "brush" in txt or ("generador" in txt and "sincron" in txt):
        return "API_546"

    # 9) Compresor centrífugo
    if "centrif" in txt and ("compresor" in txt or "compressor" in txt):
        return "API_617"

    # 10) Motor eléctrico genérico (sin info de potencia)
    if "motor" in txt:
        return "IEC_60034_14"

    # 11) Turbogenerador genérico (sin tipo) → 20816-4 (gas más probable)
    if "turbogen" in txt or "turbina" in txt or "turbine" in txt:
        return "ISO_20816_4"

    return None


def suggest_class_for_machine(norm_code: str, power_kw: float = 0.0,
                               support_type: str = "") -> Optional[str]:
    """Heurística para sugerir la clase dentro de una norma según
    potencia + tipo de soporte. Devuelve class_code o None."""
    s = (support_type or "").lower()
    is_rigid = "rigid" in s or "plano" in s or "rígido" in s

    if norm_code == "ISO_20816_8":
        if power_kw <= 0:
            return "2"  # default industrial
        if power_kw < 200:
            return "1"
        if power_kw < 1000:
            return "2"
        return "3"

    if norm_code == "ISO_20816_3":
        if power_kw >= 300:
            return "Group1_Rigid" if is_rigid else "Group1_Flex"
        return "Group2_Rigid" if is_rigid else "Group2_Flex"

    if norm_code == "ISO_20816_2":
        return "Zone_AB"

    if norm_code == "ISO_20816_4":
        return "Coupled"

    if norm_code == "ISO_20816_5":
        return "Group1_HighSpeed"

    if norm_code == "ISO_10816_7":
        if power_kw >= 200:  # crítica grande
            return "Cat_I_Rigid" if is_rigid else "Cat_I_Flex"
        return "Cat_II_Rigid" if is_rigid else "Cat_II_Flex"

    if norm_code == "ISO_10816_6":
        return "Class_3"

    if norm_code == "ISO_14694":
        return "BV_3_Rigid" if is_rigid else "BV_3_Flex"

    if norm_code == "API_670":
        return "Default"

    if norm_code == "API_618":
        return "Frame"

    if norm_code == "API_619":
        return "Housing_Op"

    if norm_code == "API_617":
        return "Housing_Acc"

    if norm_code == "API_612":
        return "Housing"

    if norm_code == "API_611":
        return "Housing_Field"

    if norm_code == "API_610":
        return "Horizontal_Std"

    if norm_code == "API_671":
        return "Standard"

    if norm_code == "API_684":
        return "AF_Critical_OK"

    if norm_code == "ISO_21940_11":
        # Heurística por tipo: turbo → G2.5, bomba/motor → G6.3,
        # IC parts → G16, default G6.3.
        return "G_6_3"

    if norm_code == "ISO_21940_12":
        return "Class_B_HighSpeed"

    if norm_code == "IEC_60034_14":
        if power_kw >= 1000:
            return "GradeA_gt280"
        if power_kw >= 100:
            return "GradeA_132_280"
        return "GradeA_56_132"

    if norm_code == "API_541":
        return "Housing_Field"

    if norm_code == "API_546":
        return "Housing_Field"

    if norm_code == "NEMA_MG1_7":
        return "Standard_Rigid" if is_rigid else "Standard_Resilient"

    if norm_code == "ANSI_S2_41":
        if power_kw >= 300:
            return "Class_III" if is_rigid else "Class_IV"
        if power_kw >= 75:
            return "Class_II"
        return "Class_I"

    if norm_code == "VDI_2056":
        if power_kw >= 1000:
            return "Klasse_T_Turbo"
        if power_kw >= 300:
            return "Klasse_G_Large"
        if power_kw >= 75:
            return "Klasse_M_Med"
        return "Klasse_K_Small"

    if norm_code == "VDI_2059":
        return "Blatt2_Industrial"

    return None


def suggest_balance_grade(machine_kind: str) -> Optional[str]:
    """Sugiere un G-grade ISO 21940-11 según el tipo de máquina.
    Heurística específica para balanceo de rotores rígidos.
    """
    t = (machine_kind or "").lower()
    if not t.strip():
        return None
    # Crankshafts grandes — chequear ANTES que "shaft" genérico
    if "crankshaft" in t and ("marine" in t or "naval" in t or "2-stroke" in t):
        return "G_630"
    if "crankshaft" in t and ("4-cyl" in t or "diésel" in t or "diesel" in t):
        return "G_100"
    # Spindles, óptica, instrumentos
    if "spindle" in t or "gyros" in t or "óptic" in t or "precisión" in t:
        return "G_1_0"
    # Turbinas, turbo-compresores, turbogen → G 2.5 (clásico)
    if (
        "turbina" in t or "turbine" in t
        or "turbo" in t or "compresor centr" in t
        or "compresor centríf" in t
    ):
        return "G_2_5"
    # Bombas, motores eléctricos, fans → G 6.3
    if (
        "bomba" in t or "pump" in t or "motor" in t
        or "ventilador" in t or "fan" in t
    ):
        return "G_6_3"
    # Drive shafts genéricos, agro, IC parts → G 16
    if (
        "shaft" in t or "agric" in t or "ic " in t or "crusher" in t
        or "transmis" in t
    ):
        return "G_16"
    return "G_6_3"  # default razonable industrial


__all__ = [
    "list_norms",
    "list_norm_groups",
    "list_classes_for_norm",
    "get_thresholds",
    "get_norm_metadata",
    "suggest_norm_for_machine",
    "suggest_class_for_machine",
    "suggest_balance_grade",
]
