#!/bin/bash
# Ciclo 21.1 + 21.2 → DEV.
# - 21.1: Editor de sensores generados ANTES de guardar (paso 4 nuevo)
# - 21.2: Soporte para gearbox intermedio (turbina + gearbox + generador)
# Wizard reorganizado: 5 → 6 pasos.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo21-1-2-wizard-gearbox-editor

# Bump VERSION en dev (la lección que aprendimos: dev también bumpea)
echo "v3.23.0" > VERSION

git add pages/_machinery_wizard.py VERSION
git commit -m "feat(21.1+21.2): wizard editor de sensores + gearbox intermedio

  ► 21.1 — Editor de sensores generados (paso 4 nuevo)
    Después de Instrumentación, st.data_editor con la lista completa
    de sensores. User puede editar lados, ángulos, direcciones, tipos,
    unidades, setpoints y patterns CSV antes de crear el activo.
    Botón 'Regenerar desde paso 3' descarta cambios.

  ► 21.2 — Gearbox intermedio
    Checkbox en paso 2 'Incluir gearbox/multiplicador'.
    Si activo, despliega bloque con: tipo, cojinetes (HSS+LSS),
    rodamiento, instrumentación independiente.
    En el paso 3 aparece bloque 'Gearbox — instrumentación'.
    El sensor map auto-generado inserta los sensores del gearbox
    entre driver y driven, renumerando planes correctamente.

  ► Renumeración: 5 → 6 pasos
    1.Tipo / 2.Tren / 3.Instrumentación / 4.Editar sensores /
    5.Unidades / 6.Datos del activo

  ► Helper _build_full_sensor_map() centraliza la generación con/sin gearbox.
    Usa core.sensor_map.new_sensor() para construir gearbox sensors
    consistentes con el formato del resto.

VERSION bumpeada en dev a v3.23.0 (próxima release)."

git push -u origin feat/ciclo21-1-2-wizard-gearbox-editor
git checkout dev
git merge --no-ff feat/ciclo21-1-2-wizard-gearbox-editor -m "Merge feat/ciclo21-1-2-wizard-gearbox-editor into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 21.1 + 21.2 en DEV"
echo " wm-test → footer debe decir v3.23.0 DEVELOPMENT"
echo " Probar: '🧙 Crear activo (wizard)' → 6 pasos, gearbox opcional,"
echo "         editor de sensores antes de guardar"
echo " Si OK → bash _publish_v3_23_0_to_main.sh"
echo "================================================================"
