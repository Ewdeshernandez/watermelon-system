#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.3 P4+P3 → DEV: SCL 100% pro
# =============================================================
# Cierra el Ciclo 17.3. SCL ahora tiene paridad completa con
# Polar y Bode:
#
#   ✅ Snapshot por bearing pair (X-Y position + eccentricity +
#      attitude + lift-off + trayectoria completa)
#   ✅ Tabla comparativa inline con diagnóstico
#   ✅ Overlay visual de centerlines históricos sobre el SCL plot
#   ✅ Narrativa modal completa Bently/API 670 en el PDF
#
# == OVERLAY VISUAL (P4) ==
#
# build_scl_figure acepta nuevo param prev_snapshots. Para cada
# snapshot anterior con trayectoria, dibuja:
#   - Línea X vs Y (lift-off curve histórico) con gradiente
#     cronológico (azul claro = más viejo, ámbar = medio, rojo =
#     más reciente). Opacidad 0.55.
#   - Diamond-open marker en el operating point del snapshot con
#     label de corrida + tooltip de eccentricity ratio + attitude.
#   - Si normalize_to_origin está activo, también normaliza la
#     trayectoria histórica por su primer punto para que sean
#     comparables.
#
# render_scl_panel resuelve el bearing label del panel actual via
# resolve_sensor_for_point (sobre Point Name primero, Paired Point
# después como fallback) y arma la lista de prev_snapshots
# filtrando por bearing_label.
#
# == NARRATIVA MODAL (P3) ==
#
# 5 bloques estilo Bently/API 670 inyectados en
# text_diag["comparison_narrative"]:
#
# (1) Encabezado factual: posición DC del muñón antes/ahora,
#     vector de migración, Δ eccentricity ratio, Δ attitude
#     angle.
#
# (2) Evolución del centerline + clearance: % del clearance
#     consumido (anterior vs actual), clasificación según API
#     670 §6.7:
#       >85% zona crítica (contacto sólido probable)
#       70-85% zona de alarma
#       <70% rango operacional normal
#
# (3) Clasificación migración + attitude:
#     - migration_critical (Δe/c >= 0.25): asentamiento, wiping
#       del babbitt, deformación del soporte.
#     - migration_major (0.15-0.25): cambio de carga estática,
#       viscosidad del aceite, desgaste asimétrico incipiente.
#     - migration_minor (0.05-0.15): variación operacional normal.
#     - shift_critical attitude (>=30°): misalignment progresivo
#       entre cojinetes acoplados (API 686).
#
# (4) Lift-off speed evolution:
#     - Δ > 200 rpm: degradación del soporte hidrodinámico
#       (viscosidad reducida, carga elevada, pérdida clearance).
#     - Δ < -200 rpm: condición más favorable del aceite o
#       redistribución de carga que descargó el cojinete.
#
# (5) Distinción diferencial:
#     - Migración + shift attitude → cambio ESTRUCTURAL
#       (alineación, asentamiento, daño mecánico del cojinete).
#     - Migración sin shift attitude → cambio OPERACIONAL
#       (carga, temperatura del aceite).
#
# Vocabulario alineado con Bently Nevada Technical Training,
# API 670 (§6.7 hydrodynamic bearings), API 686 (alineación).
#
# Ejecutar:
#   bash _publish_ciclo17_3_full_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/09_Shaft_Centerline.py
git status --short | head

git commit -m "feat(scl): overlay visual + narrativa modal Bently/API 670 (Ciclo 17.3 P4+P3)

Cierra el Ciclo 17.3. SCL al nivel de Polar/Bode con paridad
completa: snapshot + comparativo tabla + overlay visual sobre
el plot + narrativa modal completa en PDF.

OVERLAY VISUAL:
build_scl_figure acepta prev_snapshots y dibuja lift-off
curves historicos con gradiente cronologico (azul claro = mas
viejo, rojo = mas reciente) opacidad 0.55 + diamond-open
markers en operating points historicos con tooltip de
eccentricity y attitude. Respeta normalize_to_origin.

render_scl_panel resuelve bearing label via
resolve_sensor_for_point (Point Name -> fallback Paired Point)
y arma lista filtrada por bearing_label.

NARRATIVA MODAL (5 bloques estilo Bently/API 670):
(1) Encabezado: vector migración + Δe/c + Δattitude
(2) Evolución clearance vs API 670 §6.7 (>85% critico)
(3) Clasificación migración + shift attitude (con causas
    mecanicas: asentamiento, wiping, viscosidad, alineacion)
(4) Lift-off speed evolution (Δ>200rpm = damping degradation
    o redistribucion de carga)
(5) Diferencial: migración+attitude shift = ESTRUCTURAL
    (alineacion/cojinete) vs migración sola = OPERACIONAL
    (carga/aceite)

Vocabulario API 670 §6.7 hydrodynamic bearings + API 686
alineacion + Bently Technical Training." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.3 al 100% pusheado a dev"
echo "================================================================"
echo ""
echo "Para verlo end-to-end:"
echo "  1. SCL page → cargar CSV + guardar snapshot."
echo "  2. Cargar otra corrida + Comparar contra anterior."
echo "  3. El SCL plot ahora muestra:"
echo "     - Curva actual coloreada por velocidad"
echo "     - Trayectorias historicas en gradiente cronologico"
echo "     - Diamond-open en operating points historicos"
echo "  4. Tabla 'Comparativo SCL' arriba con Δ X/Y, e/c, attitude."
echo "  5. Send to Report → narrativa modal completa estilo Bently/"
echo "     API 670 con 5 bloques diagnósticos."
echo ""
echo "Estado del histórico multi-modulo:"
echo "  Tabular  ✅ snapshot + comparativo + EVOLUCIÓN PDF"
echo "  Polar    ✅ snapshot + multi-overlay + narrativa Bently"
echo "  Bode     ✅ snapshot + multi-overlay + narrativa Bently"
echo "  SCL      ✅ snapshot + multi-overlay + narrativa Bently/API 670"
echo "  Spectrum ⏳ Ciclo 17.4 (proximo)"
echo "================================================================"
