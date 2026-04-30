#!/bin/bash
# =============================================================
# Watermelon — Fix IBM Plex TTF corruptos (HTML en vez de TTF)
# =============================================================
# Borra los TTF corruptos de IBM Plex que se subieron como
# placeholders (eran HTML de error de GitHub). El sistema seguirá
# funcionando con DejaVu Sans (fallback que ya hoy se usa en
# producción porque IBM Plex no carga).
#
# Si más adelante quieres IBM Plex, hay que descargar los TTF
# correctos desde fonts.google.com/specimen/IBM+Plex+Sans
# manualmente y subirlos.
#
# Ejecutar desde el root del repo:
#   bash _fix_fonts_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon — Limpieza de TTF corruptos"
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
echo "[1] Removiendo IBMPlexSans TTF corruptos del repo..."
git rm assets/fonts/IBMPlexSans-Regular.ttf assets/fonts/IBMPlexSans-Bold.ttf
echo "    OK"
echo ""

echo "[2] Commit del fix..."
git commit -m "fix(report): remove corrupted IBM Plex TTF placeholders

Los archivos IBMPlexSans-Regular.ttf y IBMPlexSans-Bold.ttf no eran
TTF reales sino paginas HTML de error de GitHub (curl bajo el HTML
de redireccion en vez del binario). Producian TTFError en ReportLab,
que caia limpio al fallback DejaVuSans. Por eso el reporte en
produccion siempre uso DejaVuSans.

Removerlos elimina el ruido de logs (warning de TTFError silenciado).
Si en el futuro se quiere IBM Plex, descargarlos manualmente desde
fonts.google.com/specimen/IBM+Plex+Sans y validar magic bytes
00 01 00 00 antes de comitear."
echo "    OK"
echo ""

echo "[3] Push a dev..."
git push origin dev
echo "    OK"
echo ""

echo "[4] Mergear dev -> main..."
git checkout main
git pull origin main
git merge --no-ff dev -m "merge: dev fix(fonts) -> main"
git push origin main
echo "    OK"
echo ""

echo "[5] Volver a dev..."
git checkout dev
git merge main
git push origin dev
echo "    OK"
echo ""

echo "================================================================"
echo " LISTO — los TTF corruptos ya no están en el repo"
echo "================================================================"
echo ""
echo "Streamlit Cloud va a redesplegar en 1-2 minutos. El reporte"
echo "seguirá saliendo en DejaVuSans (que es lo que sale ahora)."
echo ""
echo "El siguiente problema importante es OTRO: los datos del"
echo "Document Vault para el Brush no están en producción porque"
echo ".gitignore excluye data/asset_metadata/. Resuélvelo así:"
echo ""
echo "  1. Abrí watermelonsystem.app"
echo "  2. Menú: 17 Asset Documents"
echo "  3. Perfil: Brush turbogenerator 54 MW (3600 rpm)"
echo "  4. Llená:"
echo "     - Diámetro interno cojinete: 254.41 mm"
echo "     - Clearance diametral Cd: 0.382 mm"
echo "     - Babbitt material: ASTM B-23 Grade 2 / BERA 90"
echo "     - Última fecha rebabbiting: 2018-10-23"
echo "  5. Guardá"
echo "  6. Generá un PDF nuevo desde Shaft Centerline"
echo ""
echo "Eso te dará el bearing center consistente en (0, +7.512) mil"
echo "pp para las tres fechas con e/c y attitude angle comparables."
echo "================================================================"
