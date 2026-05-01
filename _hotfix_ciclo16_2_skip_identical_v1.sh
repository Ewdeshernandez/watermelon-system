#!/bin/bash
# =============================================================
# Watermelon — HOTFIX 16.2.1: skip_identical_to en comparativo
# =============================================================
# Caso del usuario: cargo data del 19 abril, guardo snapshot.
# Cargo data del 27 abril, guardo snapshot 'Abril 27 2026'.
# Genera el PDF de la corrida 27 → la sección EVOLUCIÓN dice
# "8 estables, 0% delta" porque comparaba contra el snapshot
# que ACABA DE GUARDAR (que es la misma data actual).
#
# Fix:
#
# (1) get_previous_snapshot acepta nuevo param skip_identical_to
#     que toma el current_df. Si el snapshot mas reciente tiene
#     lecturas casi identicas (variacion media <= 1% del Overall
#     por sensor), lo saltea y devuelve el siguiente. Asi cuando
#     el usuario guarda manual + genera PDF inmediato, el PDF
#     compara contra la corrida REALMENTE anterior.
#
# (2) En el Tabular, el dropdown 'Comparar con corrida anterior'
#     ahora marca con '(corrida actual)' los snapshots cuyas
#     lecturas son identicas a la corrida cargada. El default
#     auto-cae sobre el primer snapshot DIFERENTE — no sobre el
#     que el usuario acaba de guardar.
#
# Smoke validado:
#   - Sin skip → previous = 'Abril 27 2026' (la actual, error)
#   - Con skip → previous = 'Abril 19 2026' (la real)
#   - Comparativo: 1_RAD_A: 1.5 -> 3.85 (+156.7%) up_critical
#                  3X_D:    1.10 -> 1.79 (+62.7%) up_critical
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

git commit -m "fix(history): skip snapshots identicos a corrida actual (Ciclo 16.2.1)

Caso del usuario: guarda manualmente la corrida actual + genera
PDF inmediato. Antes el PDF comparaba contra el snapshot recien
guardado (misma data) y la seccion EVOLUCION decia '8 estables'.

(1) get_previous_snapshot ahora acepta skip_identical_to=current_df.
Saltea snapshots con variacion media de Overall <= 1% (esencialmente
la misma corrida).

(2) Tabular dropdown 'Comparar con' marca los snapshots identicos a
la corrida actual con '(corrida actual)'. Default cae sobre el
primer snapshot DIFERENTE.

Smoke: con dos snapshots Abril 19 y Abril 27 + corrida actual = 27,
ahora previous_snapshot devuelve Abril 19 correctamente y muestra
1_RAD_A +156.7% up_critical, 3X_D +62.7% up_critical." || echo "Nothing to commit"

git push origin dev

echo ""
echo "✓ Hotfix pusheado. Refrescá Streamlit y volvé a generar el PDF."
