#!/bin/bash
# Ciclo 22.3 → DEV: Wizard como único camino para crear activos.
# Eliminamos el form legacy "Crear nueva instancia" de Machinery Library.
# El asistente garantiza tren coherente + sensor map + parámetros sembrados.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo22-3-wizard-only

git add pages/00_Machinery_Library.py VERSION
git commit -m "feat(22.3): Wizard único camino para crear activos

  ► Cambios
    - Eliminado render_create_instance_section() en su forma de form
      inline. Reemplazado por una CTA card que llama st.switch_page()
      al asistente (pages/_machinery_wizard.py).
    - El wizard de 6 pasos garantiza:
        · profile coherente
        · tren completo (driver + driven + acople)
        · sensor map auto-generado según support_type
        · parámetros sembrados desde el profile
    - Empty-state en main() ahora apunta al wizard.
    - Removed unused imports: create_instance, PROFILES alias.

  ► Por qué
    El form legacy permitía crear máquinas con campos mínimos (id, tag,
    profile) que luego quedaban sin sensores ni parámetros. Esto producía
    el síntoma 'instancia vacía' en Tabular/Polar/Bode/etc. Centralizar
    en el wizard elimina esa fuente de inconsistencia.

VERSION → v3.30.2-dev"

git push -u origin feat/ciclo22-3-wizard-only
git checkout dev
git merge --no-ff feat/ciclo22-3-wizard-only -m "Merge feat/ciclo22-3-wizard-only into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 22.3 en DEV — v3.30.2-dev"
echo " Probar wm-test → Machinery Library:"
echo "  • La sección de crear ahora es una card 🧙 con un botón"
echo "  • Click → te lleva al wizard de 6 pasos"
echo "  • Crear una máquina end-to-end debe terminar con sensor map"
echo "    y parámetros capturados sembrados"
echo " Si OK → consolidamos v3.31.0 a main (22.2c+d + 22.1 + 22.3)"
echo "================================================================"
