#!/bin/bash
# =============================================================
# Watermelon — HOTFIX 5 Ciclo 14a: tipo 'schematic' faltaba en DOCUMENT_TYPES
# =============================================================
# El form Machinery Library tab 'Esquemático' filtraba documents con
# document_type in ('schematic', 'esquematico', 'diagram') para listar
# qué documentos ya cargados podían vincularse como esquemático
# principal de la instancia. Pero ninguno de esos tipos existía en
# core/document_vault.DOCUMENT_TYPES, así que el dropdown de carga
# nunca lo mostraba como opción → el usuario no podía subir el
# esquemático con el tipo correcto → el tab Esquemático nunca lo
# encontraba como opción.
#
# Fix: agregar 'schematic' → 'Esquemático del tren acoplado' al
# catálogo DOCUMENT_TYPES en core/document_vault.py. Lo dejamos
# primero en el dict para que sea la opción más prominente del
# dropdown.
#
# Ejecutar:
#   bash _hotfix_ciclo14a_schematic_type_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add core/document_vault.py
git status --short | head

git commit -m "fix(library): hotfix 5 Ciclo 14a — agregar 'schematic' a DOCUMENT_TYPES

El form Machinery Library tab Esquematico filtraba documents con
type='schematic'/'esquematico'/'diagram' pero ninguno de esos tipos
existia en DOCUMENT_TYPES → el dropdown del uploader nunca lo
ofrecia → el tab Esquematico siempre estaba vacio.

Fix: agregar 'schematic' → 'Esquematico del tren acoplado' al
catalogo. Posicionado primero en el dict para que sea la opcion
mas prominente del uploader."

git push origin dev

echo ""
echo "Refrescar app, ir a 'Cargar nuevo documento', en el dropdown"
echo "'Tipo de documento' ahora aparece 'Esquemático del tren acoplado'."
echo "Subis el PNG, despues vas a 'Editar metadata completa' → tab"
echo "Esquematico → ahora aparece como opcion seleccionable."
