#!/bin/bash
# =============================================================
# Watermelon — HOTFIX Ciclo 14b: eliminar override manual del contexto
# =============================================================
# El usuario tenia razon: el expander 'Override manual del contexto'
# dejaba abierta la posibilidad de que los CSVs cargados tuvieran
# metadata que NO matchea la instancia en Machinery Library — eso
# contradice el principio "una sola seleccion, todo el sistema la usa".
#
# Fix: eliminar el expander completamente. El contexto de maquina se
# deriva 100% de Instance.header. Si el ingeniero necesita corregir,
# lo hace en Machinery Library (unica fuente de verdad). Asi
# garantizamos consistencia entre los CSVs cargados y la instancia
# que los identifica.
#
# Ejecutar:
#   bash _hotfix_ciclo14b_no_override_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add pages/01_Load_Data.py
git status --short | head

git commit -m "fix(load-data): eliminar expander 'Override manual del contexto'

El override en Load Data permitia que los CSVs cargados tuvieran
metadata distinta a la instancia activa, lo que contradecia el
principio 'una sola seleccion, todo el sistema la usa'.

Fix: el contexto de maquina se deriva 100% de Instance.header. Si
el ingeniero necesita corregir, lo hace en Machinery Library
(unica fuente de verdad). Garantiza consistencia entre los CSVs
cargados y la instancia que los identifica."

git push origin dev

echo ""
echo "Refrescar — el expander 'Override manual del contexto de máquina'"
echo "ya no aparece. Solo queda el botón 'Generate Time Waveforms'."
echo "Pagina mas limpia, flujo mas claro."
