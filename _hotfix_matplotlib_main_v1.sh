#!/bin/bash
# =============================================================
# Watermelon — HOTFIX URGENTE: matplotlib en main v2.5
# =============================================================
# Streamlit Cloud rebuilt el ambiente al deployar v2.5 y
# matplotlib (que antes venia como dep transitiva) ya no se
# instalo. Resultado: el diagrama del Sensor Map (sensor_diagram)
# devuelve None y se muestra "No se pudo renderizar el diagrama"
# en Machinery Library, Machine Map y el PDF.
#
# Fix: agregar matplotlib>=3.7.0 explicitamente a requirements.txt.
#
# Este hotfix va DIRECTO A MAIN para que produccion vuelva
# a renderizar inmediatamente. Tambien queda en dev para
# coherencia.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " STEP 1: Commit hotfix en dev"
echo "================================================================"
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add requirements.txt
git status --short | head

git commit -m "fix(deps): agregar matplotlib explicito a requirements (URGENTE)

Streamlit Cloud rebuilt el env al deployar v2.5 y matplotlib
(antes dep transitiva) ya no se instalo. El render del Sensor
Map fallaba en produccion con 'No se pudo renderizar el diagrama'.

Agregado matplotlib>=3.7.0 explicito. Ya se usa en
core/sensor_diagram.py y core/plot_export.py — solo faltaba
declararlo." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " STEP 2: Cherry-pick a main directo (urgente)"
echo "================================================================"
git fetch origin
PRE_TAG="v2.5.1-pre-main-$(date +%Y%m%d-%H%M%S)"
git tag -a "$PRE_TAG" origin/main -m "Snapshot main antes hotfix matplotlib"
git push origin "$PRE_TAG"

git checkout main
git pull origin main
# Solo el archivo requirements.txt — sin merge completo
git checkout dev -- requirements.txt
git add requirements.txt
git commit -m "fix(deps): agregar matplotlib explicito (hotfix urgente desde dev)

Streamlit Cloud rebuilt el env al deployar v2.5 y matplotlib
ya no se instalo como dep transitiva. El render del Sensor Map
fallaba en produccion. Cherry-pick directo desde dev." || echo "Nothing"

git tag -a "v2.5.1" -m "Watermelon v2.5.1 — hotfix matplotlib explicito"
git push origin main
git push origin v2.5.1

echo ""
echo "================================================================"
echo " STEP 3: Volver a dev"
echo "================================================================"
git checkout dev

echo ""
echo "================================================================"
echo " ✓ HOTFIX PUSHEADO"
echo "================================================================"
echo ""
echo "Streamlit Cloud va a redeployar main en 1-2 min e instalar"
echo "matplotlib. Despues del redeploy, el render del Sensor Map"
echo "vuelve."
echo ""
echo "Tags:"
echo "  - $PRE_TAG (rollback si algo sale mal)"
echo "  - v2.5.1 (release del hotfix)"
echo ""
echo "ROLLBACK si fuera necesario:"
echo "  git checkout main && git reset --hard $PRE_TAG && \\"
echo "  git push --force-with-lease origin main"
echo "================================================================"
