#!/bin/bash
# Release v3.27.0: ML recip render + 2 minor fixes (footer text + balloons).
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B fix/footer-balloons

echo "v3.27.0" > VERSION

git add core/recip_schematic.py pages/_machinery_wizard.py VERSION
git commit -m "fix: schematic footer removido + globitos del wizard fix

  ► core/recip_schematic.py: quitamos el texto 'N cilindros · N cojinetes
    motor' del footer del PNG. Se superponía con los cilindros inferiores
    en layouts con muchos cilindros.

  ► pages/_machinery_wizard.py: st.balloons() ahora se llama ANTES del
    _reset_wizard. Antes el reset disparaba un rerun que cancelaba los
    globitos. También capturamos created_id antes del reset para que
    el mensaje de éxito conserve el nombre.

VERSION → v3.27.0"

git push -u origin fix/footer-balloons
git checkout dev
git merge --no-ff fix/footer-balloons -m "Merge fix/footer-balloons into dev"
git push origin dev

# Direct to main
git checkout main
git pull origin main --ff-only
echo "v3.27.0" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.27.0" --allow-empty
git merge --no-ff dev -m "release(v3.27.0): wizard recip + ML recip render + minor fixes"
git tag -a "v3.27.0" -m "Release v3.27.0"
git push origin main
git push origin "v3.27.0"

echo ""
echo "================================================================"
echo " ✅ v3.27.0 en MAIN"
echo " Cambios desde v3.21.1:"
echo "   • Wizard turbomáquina (5→6 pasos) + gearbox + editor sensores"
echo "   • Wizard reciprocante con cilindros opuestos boxer"
echo "   • Schematic recip con flanges + bulones realistas"
echo "   • Editor visual click-to-place"
echo "   • 4 plantillas LATAM nuevas (LM6000, LM5000, TM2500, SGT300)"
echo "   • ML 'Diagrama visual del mapa' usa schematic recip"
echo "   • Multi-tenant ACL + Admin UI clientes"
echo "================================================================"
