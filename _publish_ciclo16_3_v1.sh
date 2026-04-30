#!/bin/bash
# =============================================================
# Watermelon — Ciclo 16.3 → DEV: Trends multi-snapshot en PDF
# =============================================================
# Mini line charts por sensor crítico mostrando los últimos N
# snapshots. Permite ver la trayectoria del sensor en el tiempo,
# no solo el delta vs la corrida anterior.
#
# Aparece en la sección EVOLUCIÓN del PDF, debajo de la tabla
# de cambios significativos. Grid de 2 columnas con hasta 6
# charts (los sensores up_critical / up con mayor delta %).
#
# Cada chart muestra:
#   - Línea conectora con markers redondos coloreados por status
#     de cada snapshot (verde Normal, ámbar Alarm, rojo Danger)
#   - Líneas horizontales discontinuas para Alarm y Danger del
#     setpoint individual del sensor (con etiquetas A / D)
#   - Eje X: timestamps en formato "DD-MMM"
#   - Eje Y: unidad nativa del sensor
#   - Título: "{sensor_label} · {plane_label}"
#
# Cambios:
#
# (1) core/instance_history.py — nueva función
#     get_sensor_history(instance_id, sensor_label, max_snapshots,
#     current_reading=None) que devuelve las últimas N lecturas
#     de un sensor a través de los snapshots, ordenadas
#     cronológicamente. Acepta opcionalmente un current_reading
#     para anexar la corrida actual al final (útil cuando se
#     genera el PDF antes de snapshotear).
#
# (2) core/trend_charts.py NUEVO — render_sensor_trend_chart
#     usa matplotlib (ya en requirements). Robusto a:
#       - Pocos puntos (1-2): markers grandes
#       - Timestamps malformados: descartados silentes
#       - Falta de matplotlib: devuelve None y caller omite
#
# (3) pages/16_Reports.py — en sección EVOLUCIÓN, después de la
#     tabla, render del grid 2×N con los trends de los top
#     sensores up_critical / up por % delta. Caption explicativa.
#     Try/except para no romper el reporte si los trends fallan.
#
# Smoke validado:
#   - render_sensor_trend_chart con 6 puntos sintéticos (1.30
#     verde → 3.85 ámbar) genera PNG 25KB con threshold lines.
#   - Compile OK en los 3 archivos.
#
# Próximas iteraciones (si querés):
#   - 16.2 Parte 3 — trend chips en el render del esquemático
#     (visual sobre foto real / Mini Machine Map).
#   - 16.4 — Mini Machine Map en otras páginas.
#   - 16.5 — Auto-load última sesión.
#
# Ejecutar:
#   bash _publish_ciclo16_3_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add core/instance_history.py
git add core/trend_charts.py
git add pages/16_Reports.py
git status --short | head

git commit -m "feat(reports): trends multi-snapshot en seccion EVOLUCION (Ciclo 16.3)

Mini line charts por sensor critico mostrando los ultimos N
snapshots historicos con threshold lines (Alarm/Danger), markers
coloreados por status, eje X timestamps. Grid 2xN al final de
la seccion EVOLUCION del PDF.

(1) core/instance_history.py — get_sensor_history(instance_id,
sensor_label, max_snapshots, current_reading=None) devuelve las
ultimas N lecturas de un sensor a traves de snapshots ordenadas
cronologicamente. Acepta current_reading para anexar la corrida
actual al final.

(2) core/trend_charts.py NUEVO — render_sensor_trend_chart con
matplotlib. Robusto a pocos puntos, timestamps malformados y
falta de la lib (devuelve None, caller omite).

(3) pages/16_Reports.py — grid 2xN de trends para los top 6
sensores up_critical/up por % delta. Caption explicativa.

Smoke: 6 puntos sinteticos (1.30 Normal -> 3.85 Alarm) genera
PNG 25KB con threshold lines y markers coloreados por status." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 16.3 pusheado a dev"
echo "================================================================"
echo ""
echo "Para verlo en accion:"
echo "  1. Necesitas al menos 2 snapshots distintos guardados"
echo "     para esta instancia (los que ya tenes de Abril 19 y 27"
echo "     sirven)."
echo "  2. Carga la corrida actual."
echo "  3. Genera el PDF Reports."
echo "  4. La seccion EVOLUCION ahora muestra debajo de la tabla"
echo "     un grid de mini line charts por cada sensor con mayor"
echo "     evolucion, mostrando la trayectoria con threshold lines"
echo "     y markers coloreados por status de cada corrida."
echo ""
echo "Cuando confirmes que funciona, podemos juntar 16.1+16.2+16.3"
echo "y publicar v2.6 a main."
echo "================================================================"
