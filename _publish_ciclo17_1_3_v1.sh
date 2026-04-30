#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.1.3 → DEV: Narrativa modal completa Polar
# =============================================================
# Reemplaza la narrativa simple de "comparativo de balance" por
# un analisis rotodinamico al nivel del libro Bently Nevada
# Technical Training y API 684. Cuando se envia una figura
# Polar al PDF Reports con un snapshot anterior elegido, el
# parrafo bajo la figura ahora incluye:
#
# (1) ENCABEZADO FACTUAL — vector change en magnitud y direccion
#     "Análisis comparativo rotodinámico contra «X». A la
#      velocidad operativa (3600 rpm), la respuesta sincrónica
#      1X del sensor evolucionó de 1.545 mil pp @ 178.1° a
#      1.070 mil pp @ 167.4°, lo que representa un vector
#      change de -0.475 mil pp (-30.8%) en magnitud y un shift
#      de fase 1X de -10.8° en arco menor."
#
# (2) CARACTERIZACION DEL MODO — clasifica por phase delta a la
#     critica:
#     ~180° = primer modo translacional / cylindrical
#             (in-phase bending) — clasico para balance shift
#     ~90°  = modo conico/pivotal o segundo modo translacional
#     >210° = segundo modo flexural o respuesta acoplada
#     <90°  = baja deflexion modal — posible resonancia
#             ESTRUCTURAL del soporte/fundacion mas que del
#             rotor (NOTA DIFERENCIAL importante)
#
#     Mas chequeo del separation margin contra API 684 §6
#     (>=15% recomendado).
#
# (3) DIAGNOSTICO DIFERENCIAL DEL SHIFT — segun magnitud:
#     >60° critico: posible crack/perdida de masa/asentamiento
#     30-60° mayor: cambio de balance del rotor (ISO 21940-12)
#     10-30° menor: deriva operacional / vigilar tendencia
#     <10° estable: variacion normal
#
# (4) ANALISIS DE SENSITIVIDAD — segun delta amplitud:
#     >50%: degradacion de damping en soportes hidrodinamicos
#     >=20%: cambio activo en respuesta modal
#     bajada significativa: posible compensatorio (intervencion)
#
# (5) DISTINCION MODAL ROTOR vs ESTRUCTURAL — nota especial
#     cuando el shift es mayor PERO el phase delta a la critica
#     es <90°: puede ser falla en fundacion / grouting / anclajes
#     mas que del rotor — requiere intervencion estructural,
#     no balance.
#
# El tono y vocabulario alineados con Bently Technical Training:
# "vector change", "respuesta sincronica 1X", "deflexion modal",
# "respuesta acoplada rotor-estructura", "damping degradation",
# "in-phase bending", "modo translacional/conico/flexural",
# "separation margin", "stability margin".
#
# Ejecutar:
#   bash _publish_ciclo17_1_3_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/06_Polar_Plot.py
git status --short | head

git commit -m "feat(polar): narrativa modal completa estilo Bently/API 684 (Ciclo 17.1.3)

Reescribe comparison_narrative del PDF Reports con analisis
rotodinamico al nivel del Bently Nevada Technical Training.
Cinco bloques estructurados:

(1) Encabezado factual con vector change.
(2) Caracterizacion del modo por phase delta a la critica:
    ~180° = 1er modo translacional/cylindrical (in-phase bending)
    ~90°  = modo conico/pivotal o 2do modo translacional
    >210° = 2do modo flexural o respuesta acoplada
    <90°  = baja deflexion modal — posible resonancia
            estructural del soporte (no del rotor)
    + chequeo separation margin API 684 §6 (>=15%).
(3) Diagnostico diferencial del shift (critico/mayor/menor/
    stable) con causas mecanicas especificas por categoria.
(4) Analisis de sensitividad vectorial — degradacion de damping
    en soportes hidrodinamicos cuando amplitud crece.
(5) Distincion modal rotor vs estructural — alerta cuando shift
    mayor pero phase delta <90° (posible fundacion/grouting).

Vocabulario Bently/API 684: vector change, respuesta sincronica
1X, deflexion modal, in-phase bending, modo translacional/
conico/flexural, separation margin, stability margin, damping
degradation." || echo "Nothing to commit"

git push origin dev

echo ""
echo "✓ Pusheado a dev. Refrescá y volvé a generar el PDF para ver"
echo "  la narrativa modal completa en la sección Polar."
