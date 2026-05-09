#!/bin/bash
# Hotfix v3.26.3: Machinery Library 'Diagrama visual del mapa' para
# compresores reciprocantes ahora usa core.recip_schematic (boxer)
# en lugar del render genérico turbomáquina.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B hotfix/ml-recip-diagram

echo "v3.26.3" > VERSION

git add pages/00_Machinery_Library.py VERSION
git commit -m "fix(ML): diagrama recip usa el schematic boxer nuevo

Antes: 'Diagrama visual del mapa' en Machinery Library mostraba todos
los sensores en una línea horizontal con cojinetes numerados — render
genérico de turbomáquina aplicado a recip = bug visual.

Ahora: si la instancia es compresor reciprocante (detectado por
driven_kind == recip_compressor o por presencia de 'cilindro' en los
plane_labels), se llama a core.recip_schematic.generate_recip_png()
con N cilindros y N cojinetes motor inferidos de los sensores.

Si falla por cualquier razón, fallback al render genérico (no rompe).

VERSION → v3.26.3"

git push -u origin hotfix/ml-recip-diagram
git checkout dev
git merge --no-ff hotfix/ml-recip-diagram -m "Merge hotfix/ml-recip-diagram into dev"
git push origin dev

git checkout main
git pull origin main --ff-only
echo "v3.26.3" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.26.3" --allow-empty
git merge --no-ff dev -m "release(v3.26.3): ML usa recip_schematic para reciprocantes"
git tag -a "v3.26.3" -m "Release v3.26.3"
git push origin main
git push origin "v3.26.3"

echo ""
echo "================================================================"
echo " ✅ v3.26.3 en MAIN — esperá 1-2 min y reabrí Machinery Library"
echo "    seleccionando un activo reciprocante. Diagrama debe mostrar"
echo "    cilindros opuestos + acople con flanges."
echo "================================================================"
