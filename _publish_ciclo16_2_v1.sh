#!/bin/bash
# =============================================================
# Watermelon — Ciclo 16.2 → DEV: histórico persistente + comparativo
# =============================================================
# El sistema guarda snapshots por instancia (Overall, Status,
# Alarm, Danger, Unit por sensor + timestamp) en JSON liviano
# (~5KB por corrida). El ingeniero NO necesita conservar los
# CSVs viejos — el sistema mantiene los readings consolidados.
#
# Implementado en 3 partes (parte 3 de chips visuales pospuesta):
#
# (P1) core/instance_history.py NUEVO
#   Helpers: save_snapshot, list_snapshots, load_snapshot,
#   delete_snapshot, get_previous_snapshot, compare_to_previous,
#   trend_arrow, trend_color. Storage en
#   {INSTANCES_DIR}/{instance_id}/history/snapshot_{ISO}.json.
#   Limita a 24 snapshots por instancia (auto-prune).
#   Trend categorizado por delta % + cambio de status:
#     ▲ up_critical (≥+20% o cruzó hacia Alarm/Danger)
#     ↑ up         (+5 a +20%)
#     → stable     (|Δ| < 5%)
#     ↓ down       (-5 a -20%)
#     ▼ down_good  (≤-20% o regresó de Alarm/Danger a Normal)
#     — no_prev    (sin lectura anterior)
#
# (P2) Tabular List — UI snapshot + comparativo
#   Sidebar "📚 Histórico de la unidad":
#     - Cantidad de snapshots existentes.
#     - Expander "📸 Guardar corrida actual" con etiqueta + notas.
#     - Dropdown "Comparar con corrida anterior".
#     - Expander "🗂️ Gestionar snapshots" con borrar.
#   Cuando hay comparación activa, abajo del Tabular:
#     - Sección "📈 Comparativo con corrida anterior".
#     - Resumen de tendencias (▲ N con alza · → N estables · ...).
#     - Tabla con Anterior, Actual, Δ, Δ %, Tendencia, Status
#       anterior/actual. Ordenada por criticidad.
#
# (P4) Reports PDF — sección EVOLUCIÓN + auto-snapshot
#   Después de MAPA DE SENSORES, antes de Recomendaciones, nueva
#   sección "EVOLUCIÓN DESDE LA CORRIDA ANTERIOR" cuando hay al
#   menos 1 snapshot guardado:
#     - Prosa ingenieril citando la corrida anterior por etiqueta
#       y fecha, distribución por tendencia, mención por nombre
#       del sensor con mayor incremento (con valores absolutos
#       y % delta).
#     - Tabla compacta solo de sensores con cambio significativo
#       (up_critical/up/down/down_good), ordenada por criticidad.
#     - Si todo estable: "Ningún sensor presentó variación
#       significativa entre corridas".
#   AUTO-SNAPSHOT al final del PDF: guarda la corrida actual
#   como snapshot (corrida_label = consecutivo del reporte o
#   "Reporte YYYY-MM-DD") para que la próxima corrida tenga
#   referencia automática sin pedir nada al usuario.
#
# Smoke validado:
#   * Snapshot save/load/list/delete OK.
#   * compare_to_previous detecta correctamente:
#     - +156.7% Normal→Alarm = up_critical ▲
#     - +62.7% Normal→Normal = up_critical ▲ (>+20%)
#     - -1.7% = stable →
#     - sensor nuevo = no_prev —
#   * Compile OK en los 3 archivos editados + 1 nuevo.
#
# Posupuesto para iteración siguiente:
#   * Parte 3 — Trend chips visuales en el render del esquemático
#     (Mini Machine Map + foto real). El comparativo en tabla del
#     Tabular + sección EVOLUCIÓN del PDF ya cubren el valor
#     principal.
#
# Ejecutar:
#   bash _publish_ciclo16_2_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add core/instance_history.py
git add pages/01__Tabular_List.py
git add pages/16_Reports.py
git status --short | head

git commit -m "feat(history): snapshots historicos + comparativo multi-fecha (Ciclo 16.2)

El sistema guarda snapshots por instancia (Overall, Status,
Alarm, Danger por sensor + timestamp) en JSON liviano. El
ingeniero NO necesita conservar los CSVs viejos — el sistema
mantiene los readings consolidados (~5KB por corrida, max 24
por instancia con auto-prune).

(P1) core/instance_history.py NUEVO — save_snapshot,
list_snapshots, load_snapshot, delete_snapshot,
get_previous_snapshot, compare_to_previous, trend_arrow,
trend_color. Storage en INSTANCES_DIR/{id}/history/.

Trend categorizado por delta % + cambio de status:
  ▲ up_critical (≥+20% o cruzó a Alarm/Danger)
  ↑ up (+5 a +20%)
  → stable (|Δ| < 5%)
  ↓ down (-5 a -20%)
  ▼ down_good (≤-20% o salió de Alarm/Danger)
  — no_prev (sin lectura anterior)

(P2) pages/01__Tabular_List.py — sidebar 'Histórico de la
unidad' con: cantidad de snapshots, expander Guardar corrida
actual, dropdown Comparar con, expander Gestionar snapshots
con borrar. Sección inline 'Comparativo con corrida anterior'
con resumen de tendencias y tabla detallada ordenada por
criticidad.

(P4) pages/16_Reports.py — nueva sección EVOLUCIÓN DESDE LA
CORRIDA ANTERIOR en el PDF (entre MAPA DE SENSORES y
Recomendaciones). Prosa ingenieril, mención del sensor con
mayor incremento por nombre, tabla compacta solo de sensores
con cambio significativo. AUTO-SNAPSHOT al generar PDF para
que la próxima corrida tenga referencia automática.

Smoke validado: snapshots save/load/compare con cambios
Normal→Alarm, ascensos +156%, estables, sensores nuevos.

Postpuesto P3 (trend chips visuales en el render del
esquematico) — el comparativo en tabla + EVOLUCIÓN del PDF
ya cubren el valor principal." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 16.2 (Partes 1, 2, 4) pusheado a dev"
echo "================================================================"
echo ""
echo "Para usarlo:"
echo "  1. Ir a Tabular List y cargar la corrida actual."
echo "  2. En la sidebar derecha → '📚 Histórico de la unidad'"
echo "     → '📸 Guardar corrida actual', poner etiqueta y guardar."
echo "  3. Cargar otra corrida (mismos CSVs distintos o cambios)."
echo "  4. En el dropdown 'Comparar con corrida anterior' elegir"
echo "     la primera. Aparece la sección 'Comparativo con corrida"
echo "     anterior' abajo con tendencias por sensor."
echo "  5. Generar el PDF Reports → la sección EVOLUCIÓN aparece"
echo "     automaticamente cuando hay al menos 1 snapshot anterior."
echo "     Cada generación de PDF auto-snapshotea la corrida actual."
echo ""
echo "Próximos pendientes en dev (Parte 3 pospuesta):"
echo "  * Trend chips en el render del esquematico (visual sobre"
echo "    foto real + Mini Machine Map)."
echo "================================================================"
