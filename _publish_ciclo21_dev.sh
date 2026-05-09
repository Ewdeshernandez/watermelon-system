#!/bin/bash
# Ciclo 21 → DEV: Wizard guiado para crear activos (UX tipo System1).
# - pages/_machinery_wizard.py: 5 pasos (Tipo → Tren → Instrumentación → Unidades → Datos)
# - core/auth.py: + entrada NAV "🧙 Crear activo (wizard)"
# Página NUEVA, no toca pages/00_Machinery_Library.py.
set -e
cd "$(dirname "$0")"

PRE_TAG="pre-ciclo21-$(date +%Y%m%d)"
git tag -f "${PRE_TAG}"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo21-machinery-wizard

# Bump VERSION en dev a la próxima release. Sin esto, wm-test mostraría
# la versión vieja en el footer aunque el código deployado sea nuevo.
echo "v3.22.0" > VERSION

git add pages/_machinery_wizard.py core/auth.py VERSION
git commit -m "feat(21): Wizard guiado para crear activos (UX tipo System1)

Página NUEVA pages/_machinery_wizard.py con 5 pasos:
  1. Tipo de máquina + plantilla LATAM (opcional, pre-rellena pasos 5)
  2. Tren mecánico — driver+driven con cojinetes y acople
  3. Instrumentación — proximity_xy / axial_accel / accel+velocity
                       + keyphasor + canales por sensor
  4. Unidades & setpoints — mil pp/µm pp / mm/s pk-rms / g pk-rms
  5. Datos del activo — ID, tag, cliente, sitio, RPM, norma, notas

Backend: usa core.sensor_map.generate_standard_sensor_map() (ya existía)
+ core.instance_state.create_instance + update_instance_header.
Convive con pages/00_Machinery_Library.py (legacy intacto).

Stepper visual con highlight del paso actual. Pre-llenado automático
de RPM/manufacturer/model/norma cuando elegís plantilla LATAM.

Acceso: admin + specialist (NO client). Bloqueada en CLIENT_BLOCKED_PAGES."

git push -u origin feat/ciclo21-machinery-wizard
git checkout dev
git merge --no-ff feat/ciclo21-machinery-wizard -m "Merge feat/ciclo21-machinery-wizard into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 21 en DEV"
echo " Probar en wm-test:"
echo "  1. Sidebar → '🧙 Crear activo (wizard)'"
echo "  2. Elegí plantilla 'Solar Mars 100' o 'Brush turbogen 54MW'"
echo "  3. Avanzá los 5 pasos → Crear"
echo "  4. Verificá que aparezca en Machinery Library"
echo " Si OK → bash _publish_v3_22_0_to_main.sh"
echo "================================================================"
