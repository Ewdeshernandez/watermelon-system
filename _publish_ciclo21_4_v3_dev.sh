#!/bin/bash
# Ciclo 21.4 v3: cilindros opuestos boxer + acople con flanges/bulones.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B fix/recip-boxer-flanges

echo "v3.26.1" > VERSION

git add core/recip_schematic.py VERSION
git commit -m "fix(recip schematic): cilindros opuestos boxer + acople con flanges

  ► Cilindros HORIZONTALMENTE OPUESTOS (estilo Ariel KBK / Burckhardt
    real). Convención: impares arriba (C1, C3, C5, C7), pares abajo
    (C2, C4, C6, C8), alineados en pares por posición x.
    Cabeza de válvulas en el lado externo (arriba para cilindros
    superiores, abajo para inferiores).

  ► Acople rediseñado: 2 flanges con bulones marcados + conector
    central (estilo gear coupling industrial real). Reemplaza las
    3 líneas verticales planas anteriores.

  ► sensor_default_position actualizado para coincidir con la nueva
    geometría: cilindros impares→arriba, pares→abajo.

VERSION → v3.26.1"

git push -u origin fix/recip-boxer-flanges
git checkout dev
git merge --no-ff fix/recip-boxer-flanges -m "Merge fix/recip-boxer-flanges into dev"
git push origin dev

# Direct to main
git checkout main
git pull origin main --ff-only
echo "v3.26.1" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.26.1" --allow-empty
git merge --no-ff dev -m "release(v3.26.1): cilindros boxer opuestos + acople flanges"
git tag -a "v3.26.1" -m "Release v3.26.1"
git push origin main
git push origin "v3.26.1"

echo ""
echo "================================================================"
echo " ✅ v3.26.1 en MAIN"
echo " Probar wizard recip 4 cilindros → debe verse:"
echo "   C1↑ C2↓ en x1 (par opuesto)"
echo "   C3↑ C4↓ en x2 (par opuesto)"
echo "   Acople con 2 flanges + bulones"
echo "================================================================"
