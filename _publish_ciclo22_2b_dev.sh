#!/bin/bash
# Ciclo 22.2b → DEV: Mapa de Sensores con resumen + validación de coherencia.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo22-2b-sensor-map-validation

echo "v3.29.0" > VERSION

git add pages/00_Machinery_Library.py VERSION
git commit -m "feat(22.2b): Mapa de Sensores con resumen + validación de coherencia

  ► Chips de resumen arriba del editor:
    - 🎯 N proximity, ⚡ N accelerometer, 📊 N velocity, 🔑 N keyphasor
    - 📊 Total

  ► Validación de coherencia tipo↔unidad (el bug histórico C-200-C):
    Detecta sensores donde la unit_native NO corresponde al sensor_type:
      - accelerometer + 'mm/s RMS' → ⚠️ debería ser g RMS / g peak / m/s²
      - velocity + 'g RMS' → ⚠️ debería ser mm/s o in/s
      - proximity + 'g RMS' → ⚠️ debería ser mil pp / µm pp
    Muestra expander con la lista de mismatches y la unidad esperada.

  ► Compatible con sensor_type='velometer' (alias usado por el wizard).

  ► No bloquea el guardado — alerta para que el user corrija. Filosofía
    'safe by default, no surprise restrictions'.

VERSION → v3.29.0"

git push -u origin feat/ciclo22-2b-sensor-map-validation
git checkout dev
git merge --no-ff feat/ciclo22-2b-sensor-map-validation -m "Merge feat/ciclo22-2b-sensor-map-validation into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 22.2b en DEV — v3.29.0"
echo " Probar wm-test → Machinery Library → C-200-C → Mapa de Sensores"
echo " Debe aparecer expander 'X sensor(es) con unidad incoherente'"
echo " que liste los YV/mm/s configurados como accelerometer"
echo " Si OK → bash _publish_v3_29_0_to_main.sh"
echo "================================================================"
