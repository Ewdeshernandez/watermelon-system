#!/bin/bash
# Release v3.27.1: patterns CSV más permisivos en wizard recip.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B fix/recip-csv-patterns

echo "v3.27.1" > VERSION

git add pages/_machinery_wizard.py VERSION
git commit -m "fix(wizard recip): patterns CSV permisivos (cubre Bently/CSI/ADRE)

Antes: pattern simple '*crosshead*cyl1*' fallaba si el Point del CSV
tenía las palabras en otro orden o abreviaciones distintas (ej:
'Cyl1 Crosshead Acc' o 'C1_CROSSHEAD').

Ahora cada sensor reciprocante (crosshead, rod drop, frame) tiene
patterns OR que cubren los órdenes más comunes y abreviaciones.

Esto reduce mismatch sensor↔CSV y previene el bug de Tabular List
mostrando unidad del CSV en lugar de la configurada en el sensor.

Activos NUEVOS creados con el wizard se benefician inmediatamente.
Activos legacy (como C-200-C) hay que editar patterns en
Machinery Library → Mapa de Sensores manualmente.

VERSION → v3.27.1"

git push -u origin fix/recip-csv-patterns
git checkout dev
git merge --no-ff fix/recip-csv-patterns -m "Merge fix/recip-csv-patterns into dev"
git push origin dev

git checkout main
git pull origin main --ff-only
echo "v3.27.1" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.27.1" --allow-empty
git merge --no-ff dev -m "release(v3.27.1): wizard recip patterns CSV permisivos"
git tag -a "v3.27.1" -m "Release v3.27.1"
git push origin main
git push origin "v3.27.1"

echo ""
echo "================================================================"
echo " ✅ v3.27.1 en MAIN"
echo "================================================================"
