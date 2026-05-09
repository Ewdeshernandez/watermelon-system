#!/bin/bash
# Ciclo 22.1 → DEV: Tabular List respeta unit_native SIEMPRE (bug fix).
# Caso real: C-200-C con sensor_type='accelerometer' + unit_native='mm/s RMS'.
# Antes Family caía a 'Acceleration' y se desincronizaba con la columna Unit.
# Ahora unit_native es la fuente de verdad → Family='Velocity', Unit='mm/s RMS'.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo22-1-tabular-unit-native

git add core/sensor_map.py pages/01__Tabular_List.py VERSION
git commit -m "fix(22.1): Tabular List respeta unit_native SIEMPRE

  ► Bug original
    - Sensor con sensor_type='accelerometer' pero unit_native='mm/s RMS'
      (caso C-200-C en Parex) producía Family='Acceleration' en Tabular,
      pero Unit Full='mm/s RMS' → fila incoherente.
    - Builder caía a CSV legacy intermitentemente porque sensor_unit_family()
      sólo miraba sensor_type.

  ► Fix
    - Nuevo helper core.sensor_map.unit_to_family(unit_native) que infiere
      la familia de medida directamente desde la unidad declarada por el
      sensor (mm/s → Velocity, g → Acceleration, mil/µm → Proximity).
    - sensor_unit_family() ahora prioriza unit_native > sensor_type.
    - build_table_dataframe() en Tabular: cuando hay sensor_match, NUNCA
      cae a CSV legacy. unit_native dicta family + overall_mode + unit_full.
    - Criterion 'API 670' sólo aplica si la unit es realmente displacement
      (mil/µm). Sensor proximity con unit mm/s usa ISO 20816-3.

  ► Smoke
    - tests/run_smoke.py debe pasar (resolver de sensores no cambia).

VERSION → v3.30.1-dev"

git push -u origin feat/ciclo22-1-tabular-unit-native
git checkout dev
git merge --no-ff feat/ciclo22-1-tabular-unit-native -m "Merge feat/ciclo22-1-tabular-unit-native into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 22.1 en DEV — v3.30.1-dev"
echo " Probar wm-test → Tabular List con C-200-C:"
echo "  • Cada fila debe mostrar Family coherente con Unit Full"
echo "  • Sensores accelerometer + mm/s → Family=Velocity"
echo "  • Sensores proximity + mil pp   → Family=Proximity (API 670)"
echo " Si OK → seguimos con Ciclo 22.3 (eliminar form legacy)"
echo "================================================================"
