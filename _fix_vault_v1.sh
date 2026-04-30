#!/bin/bash
# =============================================================
# Watermelon — Ciclo 7: Vault seeds (persistencia en producción)
# =============================================================
# Sube el cambio que hace que los datos del Brush sobrevivan
# cualquier redeploy de Streamlit Cloud. Después de esto, NO
# hace falta volver a llenar nada en Asset Documents en producción.
#
# Cambios incluidos:
#   - core/vault_seeds.py: dict con datos OEM hardcoded del Brush
#   - core/document_vault.py: get_captured_parameters cae a seed
#     cuando filesystem está vacío
#
# Ejecutar desde el root del repo:
#   bash _fix_vault_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon — Ciclo 7: Vault seeds en producción"
echo "================================================================"
echo ""

# Limpiar locks de git si existen
[ -f .git/index.lock ] && rm -f .git/index.lock

# Confirmar que estamos en dev
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "dev" ]; then
  echo "Cambiando a dev..."
  git checkout dev
fi
git pull origin dev

echo ""
echo "[1] Adoptando vault_seeds.py y modificación de document_vault.py..."
git add core/vault_seeds.py core/document_vault.py
git status --short | grep -E "^(M|A)" | head
echo ""

echo "[2] Commit del fix..."
git commit -m "feat(vault): seed defaults survive ephemeral filesystem (Ciclo 7)

Streamlit Cloud tiene filesystem efimero — los datos persistidos en
data/asset_metadata/ se borran en cada redeploy o reinicio del
container. Eso obligaba al usuario a re-ingresar parametros del
cojinete (diametro, clearance, babbitt) cada vez que la app se
recargaba en produccion.

Solucion: core/vault_seeds.py contiene defaults hardcoded por
profile_key. Arranca poblado con el Brush turbogenerator 54 MW:
  - bearing_inner_diameter_mm: 254.41
  - diametral_clearance_mm: 0.382
  - babbitt_material: ASTM B-23 Grade 2 / BERA 90
  - last_rebabbiting_date: 2018-10-23
  - oil_grade, alarm/trip temps, rated_power_mw, rated_speed_rpm

get_captured_parameters() ahora hace merge: arranca con seed y deja
que lo persistido por el usuario gane campo a campo. Esto significa
que en produccion, despues de cualquier redeploy, el SCL del Brush
muestra automaticamente bearing center en (0, +7.520) mil pp con
clearance circle como CIRCULO real, e/c y attitude angle calculados
correctamente, narrativa citando el reporte de Wersin.

Para sumar mas activos, agregar entrada al dict VAULT_SEEDS con el
profile_key exacto y los campos OEM disponibles."
echo "    OK"
echo ""

echo "[3] Push a dev..."
git push origin dev
echo "    OK"
echo ""

echo "[4] Mergear dev -> main..."
git checkout main
git pull origin main
git merge --no-ff dev -m "merge: dev v1.1 -> main — Vault seeds (Ciclo 7)"
git push origin main
echo "    OK"
echo ""

echo "[5] Tag v1.1 sobre main..."
git tag -a v1.1 -m "Release v1.1 — Vault seeds para activos conocidos.
Brush 54 MW arranca poblado por defecto en cualquier despliegue."
git push origin v1.1
echo "    OK"
echo ""

echo "[6] Volver a dev y alinear..."
git checkout dev
git merge main
git push origin dev
echo "    OK"
echo ""

echo "================================================================"
echo " LISTO — Streamlit Cloud va a redesplegar en 1-2 minutos"
echo "================================================================"
echo ""
echo "Después del redeploy, abrí cualquier despliegue (watermelonsystem"
echo ".app o wm-home-final-2026.streamlit.app) y andá a Shaft Centerline:"
echo ""
echo "  1. Subí los CSVs del Brush"
echo "  2. En la sidebar arriba debería aparecer un mensaje verde:"
echo "     'Vault: clearance radial = 7.52 mil pp"
echo "      (OEM: campo diametral_clearance_mm capturado en Vault)'"
echo "  3. El plot va a mostrar:"
echo "     - Clearance circle como CIRCULO real (no elipse estirada)"
echo "     - BEARING CENTER en (0, +7.52) mil pp"
echo "     - Anillos eccentricity 0.40/0.70/0.85 concentricos"
echo "     - REST marker en (0,0)"
echo "     - Flecha W (load) hacia abajo"
echo "  4. La narrativa va a citar dimensiones del cojinete y babbitt"
echo "  5. e/c y attitude angle van a salir consistentes entre archivos"
echo ""
echo "Si querés agregar otro activo (ej. la GE LM6000 cuando tengas datos"
echo "OEM), edita core/vault_seeds.py y agregá una entrada nueva al dict."
echo "================================================================"
