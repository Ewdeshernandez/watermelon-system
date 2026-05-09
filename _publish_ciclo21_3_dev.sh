#!/bin/bash
# Ciclo 21.3 → DEV: Soporte para compresores reciprocantes en el wizard.
# Cuando category == reciprocating_compressor:
# - Paso 2: pide # cilindros (2/4/6/8) y rod drop opcional
# - Paso 3: instrumentación driven se fuerza a "reciprocating"
# - Sensor map: frame top+side velocímetros + crosshead accel × cilindros
#               + rod drop × cilindros (opcional) + keyphasor
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo21-3-reciprocating-support

echo "v3.24.0" > VERSION

git add pages/_machinery_wizard.py VERSION
git commit -m "feat(21.3): wizard soporta compresores reciprocantes (API 618 / ISO 20816-8)

  ► En Paso 1, si category=reciprocating_compressor el wizard cambia
    el flujo del lado driven:

  ► Paso 2 driven (cuando es recip):
    - 'Modelo / fabricante' libre (ej: Ariel KBK/4)
    - 'Número de cilindros' (2 / 4 / 6 / 8)
    - 'Incluir sensores de rod drop' (default sí, 1 por cilindro)
    - driven_planes/bearing forzados (en recip los sensores no van
      en cojinetes radiales sino en frame y crossheads)

  ► Paso 3 driven (cuando es recip):
    - Instrumentación forzada 'reciprocating'
    - Info card explicando la dotación API 618:
      · 1 velocímetro frame top
      · 1 velocímetro frame side
      · 1 acelerómetro crosshead × cilindros
      · 1 rod drop × cilindros (opcional)

  ► _build_reciprocating_sensor_map() genera el mapa completo:
    Driver (motor) según su instrumentación normal +
    Frame top+side velocímetros +
    crosshead × N cilindros +
    rod drop × N cilindros (opcional) +
    keyphasor (en motor)

VERSION → v3.24.0 en dev."

git push -u origin feat/ciclo21-3-reciprocating-support
git checkout dev
git merge --no-ff feat/ciclo21-3-reciprocating-support -m "Merge feat/ciclo21-3-reciprocating-support into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 21.3 en DEV — v3.24.0"
echo " Probar: wizard Paso 1 → 'Compresor reciprocante'"
echo "         Paso 2 → 4 cilindros + rod drop activado"
echo "         Paso 4 → ver lista 1 motor + frame + crosshead × 4 + rod × 4"
echo " Si OK → bash _publish_v3_24_0_to_main.sh"
echo "================================================================"
