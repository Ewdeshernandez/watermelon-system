#!/bin/bash
# =============================================================
# Watermelon — HOTFIX: import faltante en Machinery Library
# =============================================================
# El render del Diagrama visual del mapa en Machinery Library
# arrojaba "name 'compose_train_description' is not defined"
# porque la funcion se usaba sin importarla. Otras paginas
# (Tabular List, Machine Map, Reports) sí la importan
# correctamente. Este hotfix solo agrega el import al bloque
# de imports de pages/00_Machinery_Library.py.
#
# Sin cambios funcionales — solo restaura el render del
# diagrama en Machinery Library.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/00_Machinery_Library.py
git status --short | head

git commit -m "fix(machinery-library): importar compose_train_description que faltaba

El render del Diagrama visual del mapa arrojaba
'name compose_train_description is not defined' porque la
funcion se usaba en el bloque de render sin importarla.
Otras paginas que llaman al diagrama ya la importaban.
Sin cambios funcionales — solo restaura el render." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Hotfix import pusheado a dev"
echo "================================================================"
echo ""
echo "Verificacion: ir a Machinery Library, expandir la seccion"
echo "'Diagrama visual del mapa' — debe renderizar el diagrama de la"
echo "instancia activa con el Sensor Map (sin valores, modo de"
echo "configuracion: colores por tipo de sensor)."
echo "================================================================"
