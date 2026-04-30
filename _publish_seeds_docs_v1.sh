#!/bin/bash
# =============================================================
# Watermelon — Camino A: Document Vault seeds (manuales fábrica)
# =============================================================
# Sube manuales OEM y reportes históricos como "documentos de
# fábrica" committed al repo. Sobreviven cualquier redeploy de
# Streamlit Cloud y aparecen automáticamente en Asset Documents
# de cualquier despliegue, sin necesidad de re-subir.
#
# Cambios incluidos:
#   - .gitignore: regla de excepción para data/asset_documents_seed/
#   - core/document_vault.py: list/get/delete con union seeds + uploads
#   - pages/17_Asset_Documents.py: UI badge "🔒 Fábrica" + delete bloqueado
#   - data/asset_documents_seed/brush_turbogenerator_54mw_3600/
#       _index.json + wersin_rebabbiting_2018-10-23.pdf (8.5 MB)
#
# Ejecutar desde el root del repo:
#   bash _publish_seeds_docs_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon — Camino A: Document Vault seeds"
echo "================================================================"
echo ""

[ -f .git/index.lock ] && rm -f .git/index.lock

CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "dev" ]; then
  echo "Cambiando a dev..."
  git checkout dev
fi
git pull origin dev

echo ""
echo "[1] Adoptando cambios + PDFs seed (8.5 MB)..."
git add .gitignore core/document_vault.py pages/17_Asset_Documents.py
git add data/asset_documents_seed/
echo "    Files staged:"
git status --short | grep -E "^(M|A|\?\?)" | head -10
echo ""

echo "[2] Commit..."
git commit -m "feat(vault): document seeds — manuales OEM committed survive redeploy

Implementa Camino A para que los PDFs de referencia (manuales del
fabricante, reportes historicos como rebabbiting Wersin) sobrevivan
cualquier redeploy de Streamlit Cloud, complementando las semillas de
parametros estructurales del Ciclo 7.

Arquitectura:
* data/asset_documents_seed/{profile_key}/_index.json describe los
  documentos de fabrica para cada activo
* data/asset_documents_seed/{profile_key}/{filename} contiene los
  PDFs/imagenes en si (el repo absorbe el peso una vez)
* core/document_vault.list_documents() retorna union de seeds + uploads
  del usuario, con flag is_seed=True para que la UI los marque
* core/document_vault.delete_document() bloquea borrado de seeds
* pages/17_Asset_Documents muestra badge '🔒 Fábrica' y deshabilita
  el boton Eliminar para documentos de fabrica
* .gitignore: excepcion explicita para data/asset_documents_seed/
  (anula la regla global *.pdf y la regla data/asset_documents/)

Brush turbogenerator 54 MW arranca poblado con:
* wersin_rebabbiting_2018-10-23.pdf (8.5 MB) — informe Wersin
  rebabbiting de los 4 cojinetes radiales del 23 oct 2018, citado
  en la narrativa Cat IV del SCL como Documento de referencia.

Para sumar mas activos: crear data/asset_documents_seed/{profile}/,
agregar PDFs y declarar entradas en _index.json. Commit y push.

Smoke validado:
* list_documents devuelve seed + user upload sin colision
* get_document_path resuelve PDF de 8.5 MB correctamente
* delete_document(seed_id) bloqueado (devuelve False)
* get_document_bytes lee bytes %PDF magic correcto
* git check-ignore confirma seeds NO excluidos pese a *.pdf global"
echo "    OK"
echo ""

echo "[3] Push a dev..."
git push origin dev
echo "    OK"
echo ""

echo "[4] Mergear dev -> main..."
git checkout main
git pull origin main
git merge --no-ff dev -m "merge: dev v1.2 -> main — Document Vault seeds (Camino A)"
git push origin main
echo "    OK"
echo ""

echo "[5] Tag v1.2..."
git tag -a v1.2 -m "Release v1.2 — Document Vault seeds.
Manuales OEM y reportes historicos committed al repo sobreviven
cualquier redeploy. Brush 54 MW arranca con el reporte Wersin de
rebabbiting de oct 2018 disponible automaticamente."
git push origin v1.2
echo "    OK"
echo ""

echo "[6] Volver a dev..."
git checkout dev
git merge main
git push origin dev
echo "    OK"
echo ""

echo "================================================================"
echo " LISTO — Streamlit Cloud va a redesplegar en 1-3 minutos"
echo "================================================================"
echo ""
echo "Despues del redeploy, abrí cualquier despliegue y andá a:"
echo "  Menú lateral -> 17 Asset Documents"
echo "  Perfil -> Brush turbogenerator 54 MW (3600 rpm)"
echo ""
echo "Verás en la lista de documentos:"
echo "  Origen      | Título"
echo "  🔒 Fábrica  | Reporte de rebabbiting de cojinetes (Wersin)"
echo ""
echo "Al expandir el documento Wersin: se puede DESCARGAR pero el botón"
echo "de Eliminar está bloqueado (🔒 Permanente). Eso garantiza que el"
echo "documento de fábrica nunca se pierde por accion del usuario."
echo ""
echo "Si subes un manual NUEVO (no semilla), aparece en la lista junto"
echo "al documento de fábrica con badge '👤 Usuario' y SI se puede"
echo "borrar. Pero mientras Streamlit Cloud no reinicie el container,"
echo "los uploads persisten."
echo ""
echo "Para sumar manuales del Brush u otros activos al repo:"
echo "  1. Copia el PDF a data/asset_documents_seed/{profile_key}/"
echo "  2. Edita data/asset_documents_seed/{profile_key}/_index.json"
echo "     y agrega una entrada con id, filename, title, document_type,"
echo "     description, tags, uploaded_at"
echo "  3. git add ... ; git commit ; bash este script para publicar"
echo "================================================================"
