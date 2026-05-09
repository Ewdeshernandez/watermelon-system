#!/bin/bash
# Ciclo 18.3 → DEV: Crear activo desde plantilla LATAM (1 click).
# - core/machine_templates.py: + suggest_profile_key_for_template()
# - pages/17_Importers.py: expander 'Crear activo desde esta plantilla'
#   con form pre-llenado (profile, RPM, notas) — todo editable.
# - tests/test_template_profile_suggestion.py
set -e
cd "$(dirname "$0")"

PRE_TAG="pre-ciclo18-3-$(date +%Y%m%d)"
git tag -f "${PRE_TAG}"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo18-3-create-from-template

git add core/machine_templates.py pages/17_Importers.py tests/test_template_profile_suggestion.py
git commit -m "feat(18.3): crear activo desde plantilla LATAM (auto-relleno editable)

- core/machine_templates.py:
  + suggest_profile_key_for_template(tid) → mapea categoria + RPM
    a un profile_key de PROFILES (legacy). Heurística simple:
      gas_turbine + RPM>14k → siemens_sgt400
      gas_turbine + RPM>8k  → siemens_sgt300
      turbogenerator        → brush_turbogenerator_54mw_3600
      reciprocating_*       → reciprocating_compressor
      centrifugal_pump      → pump_horizontal/vertical
      electric_motor        → motor_X_pole por RPM
      otros                 → custom_manual

- pages/17_Importers.py (Tab Plantillas):
  Nuevo expander 'Crear activo desde esta plantilla' con form pre-llenado:
    - Profile sugerido (editable en dropdown)
    - Notas pre-cargadas con metadata de la plantilla (RPM, normas,
      rodamientos, fabricante)
    - ID, tag, serial, ubicación: el usuario los completa
  Llama core.instance_state.create_instance() — la misma función
  que usa Machinery Library. NO duplica lógica de persistencia.

- 8 tests nuevos validan que toda plantilla recibe sugerencia
  válida y que el mapeo respeta categoría + RPM.

Tests: 165 passed (157 antes)."

git push -u origin feat/ciclo18-3-create-from-template
git checkout dev
git merge --no-ff feat/ciclo18-3-create-from-template -m "Merge feat/ciclo18-3-create-from-template into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 18.3 en DEV"
echo "================================================================"
echo " Verificar en wm-test:"
echo "  1. Sidebar → 'Importers & Plantillas'"
echo "  2. Tab Plantillas → elegir Solar Mars 100 (por ej)"
echo "  3. Abajo de las notas técnicas: expander '+ Crear activo'"
echo "  4. Llenar ID + click 'Crear activo' → debe crear en ML"
echo " Si OK → bash _publish_v3_17_0_to_main.sh"
echo "================================================================"
