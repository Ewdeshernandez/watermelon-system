#!/bin/bash
# Hotfix: mover helpers _render_*_editor del final al inicio del wizard.
# Las funciones quedaban definidas DESPUÉS del flujo de pasos, así que
# cuando el step 4 las llamaba todavía no existían → NameError.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B hotfix/wizard-helpers-order

echo "v3.25.1" > VERSION

git add pages/_machinery_wizard.py VERSION
git commit -m "fix(wizard): definir helpers _render_*_editor ANTES del flujo de pasos

Sin esto, el bloque elif current==4 disparaba NameError al llamar
_render_recip_visual_editor — la función estaba al final del archivo
y el módulo Streamlit ejecuta top-down.

VERSION → v3.25.1"

git push -u origin hotfix/wizard-helpers-order
git checkout dev
git merge --no-ff hotfix/wizard-helpers-order -m "Merge hotfix/wizard-helpers-order into dev"
git push origin dev

# Direct to main (riesgo bajo, fix de orden)
git checkout main
git pull origin main --ff-only
echo "v3.25.1" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.25.1" --allow-empty

git merge --no-ff dev -m "release(v3.25.1): fix orden de helpers en wizard"
git tag -a "v3.25.1" -m "Release v3.25.1: wizard helpers order fix"
git push origin main
git push origin "v3.25.1"

echo ""
echo "================================================================"
echo " ✅ v3.25.1 en MAIN — wm-test y wm-home redeployan en 1-2 min"
echo " Reintentar: wizard reciprocante → Paso 4 → tab 'Editor visual'"
echo "================================================================"
