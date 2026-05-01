#!/bin/bash
# =============================================================
# Watermelon — HOTFIX 6 Ciclo 14a: filtro permisivo del esquemático (dev)
# =============================================================
# El usuario subio el esquematico antes del hotfix 5, asi que quedo
# con document_type='other' o 'photograph' (porque 'schematic' aun
# no existia en DOCUMENT_TYPES). Despues, el tab Esquematico del
# form filtraba estrictamente document_type in (schematic, esquematico,
# diagram) → no encontraba el doc → dropdown vacio → schematic_png
# de la instancia quedaba en string vacio → auto-fill no rellenaba
# meta.schematic_doc_id → render del esquematico en Resumen Ejecutivo
# se saltaba. Reporte salia sin imagen aunque la narrativa SCL si lo
# referenciaba como 'documento de referencia'.
#
# Fix: filtro permisivo. El dropdown del tab Esquematico ahora
# acepta CUALQUIER documento del Vault que sea imagen por extension
# (.png/.jpg/.jpeg/.gif/.webp/.svg/.bmp/.tiff), ademas del filtro
# por document_type. Asi el doc subido con tipo 'other' aparece
# como opcion sin necesidad de re-subirlo.
#
# Tambien se agrego un st.warning cuando el Vault no tiene imagenes,
# para guiar al usuario a subir una.
#
# Ejecutar:
#   bash _hotfix_ciclo14a_schematic_filter_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add pages/00_Machinery_Library.py
git status --short | head

git commit -m "fix(library): hotfix 6 Ciclo 14a — filtro permisivo del esquematico

El filtro del tab Esquematico exigia document_type estricto
(schematic/esquematico/diagram), excluyendo imagenes subidas con
tipo 'other'/'photograph'/etc. Si el usuario subio el doc antes
del hotfix 5 (o eligio otro tipo), el dropdown del tab quedaba
vacio y schematic_png nunca se setteaba.

Fix: aceptar cualquier documento del Vault con extension de
imagen (.png/.jpg/.jpeg/.gif/.webp/.svg/.bmp/.tiff) en el
dropdown, ademas del filtro por document_type. Sin re-subida.

Tambien se agrega un warning cuando el Vault no tiene imagenes,
guiando al usuario."

git push origin dev

echo ""
echo "Refrescar app, ir a TES1 → Editar metadata completa → tab"
echo "Esquematico → ahora deberia listar tu PNG/JPG aunque haya sido"
echo "subido con tipo 'other'. Seleccionalo, guardas. Generas reporte"
echo "y el esquematico aparece en el Resumen Ejecutivo (pagina 3)."
