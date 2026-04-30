#!/bin/bash
# =============================================================
# Watermelon — Ciclo 15.1.3 → DEV: hotfix feedback PDF
# =============================================================
# Cuatro arreglos sobre el feedback del primer PDF v2.5:
#
# (1) BUG CRITICO de severidad: el Machine Map clasificaba
# todos los sensores como Normal aunque el Tabular List veia
# 2 ATENCION sobre los mismos CSVs (CRF/TRF ACELL al
# 64%/52% del danger).
#
# Causa: las signals que produce build_signal_from_parsed en
# Load Data son SimpleNamespace(time=..., x=..., metadata=...).
# El helper compute_signal_overall_rms buscaba .amplitude o
# dict["y"], asi que sobre SimpleNamespace lanzaba
# AttributeError silencioso atrapado por el except externo y
# devolvia 0.0 SIEMPRE. Resultado: overall=0 < alarm para
# cualquier umbral → todo Normal.
#
# Fix: el helper ahora prueba en orden .amplitude → .x
# (SimpleNamespace de Load Data) → dict["amplitude"|"y"|"x"].
# Validado vs Tabular real: con CSV real CRF ACELL @ 64%
# del danger ahora clasifica Alarm coherente con Tabular.
#
# (2) Resumen Ejecutivo desaparecido del PDF: si el usuario
# no llenaba el campo manualmente, la seccion entera se
# OMITIA del reporte (tipo "elevator pitch" en blanco). Era
# peor que un draft. Ahora _build_pdf_bytes auto-redacta el
# Resumen Ejecutivo a partir de las figuras cargadas
# (_autodraft_executive_summary) cuando esta vacio. El
# usuario puede sobreescribirlo manualmente como antes.
#
# (3) Diagrama del tren mas turbomachinery, menos cajas:
# core/sensor_diagram.py reemplaza los FancyBboxPatch
# rectangulares por:
#   - Driver: silueta de aero-turbina con inlet vanes,
#     stage rings y cono de exhaust.
#   - Driven: cilindro con end shields + vanes radiales de
#     cooling en el lado outboard.
#   - Coupling: dos discos verticales con tornilleria
#     (4 pernos por disco).
# Generico (no replica un OEM especifico) — coherente con
# nuestra politica de no nombrar competencia.
#
# (4) Sensores de velocity invisibles en el LM6000: el
# diagrama solo mostraba el plane_label del primer sensor
# de cada plano, asi que cuando el TRF tenia ambos VT y
# acelerometro, el label decia "TRF Accel" y la velocity
# parecia que no estaba.
#
# Fix:
#   - _normalize_plane_label quita tokens de tipo (Accel/
#     Vel/Prox/Acelerometro/etc) → label de plano queda
#     "TRF" / "CRF" en lugar de "TRF Accel" / "CRF Accel".
#   - _sensor_types_in_plane lista los tipos presentes,
#     y debajo del numero del cojinete se dibujan chips
#     circulares de color (violeta=prox, cian=vel, rojo=
#     accel, ambar=keyphasor) — uno por cada tipo presente.
#
# Bonus: cojinetes en el lateral del modo FULL ahora
# tambien se colorean por worst-of-plane (antes solo en
# compact). El lateral funciona como mini-heatmap del tren
# y el polar como drill-down por sonda.
#
# Bonus 2: nuevo fallback para detectar driver/driven
# cuando solo un lado tiene tokens "driver"/"driven" en
# plane_label. Antes si el driven traia "Driven NDE" y el
# driver traia "TRF Accel" (sin token), el driver_planes
# quedaba vacio y los cojinetes 1/2 no se dibujaban.
#
# Archivos:
#   - core/machine_severity.py    (fix bug overall)
#   - core/sensor_diagram.py      (turbomachinery silhouette,
#                                   plane label normalization,
#                                   sensor-type chips)
#   - pages/16_Reports.py         (auto-draft Resumen Ejec)
#
# Ejecutar:
#   bash _hotfix_ciclo15_1_3_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add core/machine_severity.py
git add core/sensor_diagram.py
git add pages/16_Reports.py
git status --short | head

git commit -m "fix(machine-map): 4 hotfixes sobre el primer PDF v2.5 (Ciclo 15.1.3)

(1) BUG critico de severidad: signals de Load Data son
SimpleNamespace(.x) y compute_signal_overall_rms solo miraba
.amplitude o dict['y']. AttributeError silencioso → 0.0 →
todo Normal aunque Tabular vea 2 ATENCION. Ahora prueba
.amplitude → .x → dict en orden, robusto a los 3 formatos.

(2) Resumen Ejecutivo perdido: si el campo estaba vacio, la
seccion entera se omitia. Ahora _build_pdf_bytes auto-redacta
con _autodraft_executive_summary a partir de las figuras.

(3) Diagrama mas turbomachinery: silueta aero-turbina con
inlet vanes + stage rings + exhaust cone para el driver,
cilindro con end shields + cooling vanes radiales para el
driven, coupling de dos discos con tornilleria. Generico,
sin nombrar competencia.

(4) Velocity invisible en LM6000: _normalize_plane_label
quita tokens de tipo (Accel/Vel/Prox/Acelerometro) y
_sensor_types_in_plane lista los tipos presentes en cada
plano. Debajo del cojinete se dibujan chips coloreados por
tipo. Asi se ve que en TRF/CRF hay vel + accel.

Bonus: cojinetes del lateral en modo FULL tambien
coloreados por worst-of-plane. Fallback para driver/driven
cuando solo un lado tiene token en plane_label." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 15.1.3 hotfix pusheado a dev"
echo "================================================================"
echo ""
echo "Para verificar (Streamlit Cloud redeploya en 1-2 min):"
echo "  1. Tabular List → el banner Mini Machine Map ahora debe"
echo "     mostrar las 2 ATENCION (CRF y TRF) en lugar de 9 normales."
echo "  2. Generar PDF Reports → ahora aparecen 3 secciones:"
echo "       - RESUMEN EJECUTIVO (auto-redactado)"
echo "       - MAPA DE SENSORES (con chips de tipo + plane labels"
echo "         limpios + diagrama tipo turbomachinery)"
echo "       - 1. FIGURAS Y ANALISIS"
echo "  3. Cojinetes 1 y 2 del LM6000 deben mostrar dot cian"
echo "     (velocity) Y dot rojo (accelerometer) abajo, label TRF/CRF."
echo ""
echo "Cuando lo veas bien en dev, mergeamos a main como v2.5"
echo "definitivo (juntando 15.1.1 + 15.1.2 + 15.1.3)."
echo "================================================================"
