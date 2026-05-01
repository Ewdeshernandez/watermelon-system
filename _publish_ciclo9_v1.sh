#!/bin/bash
# =============================================================
# Watermelon — Ciclo 9: Persistencia real con Supabase
# =============================================================
# Storage backend con dos implementaciones que la app elige sola:
#   - LocalFilesystemRepository: lo que veníamos usando, efímero en
#     Streamlit Cloud
#   - SupabaseRepository: persistencia real, sobrevive cualquier
#     redeploy (Postgres + Storage)
#
# Sin Supabase configurado → Local automático (cero cambio de UX)
# Con Supabase configurado → Supabase automático (badge ☁️ visible)
#
# Cambios incluidos:
#   - core/instance_repository.py: 2 backends + factory + cache
#   - core/instance_state.py: refactor para usar el repository
#   - core/instance_selector.py: badge del backend activo
#   - pages/17_Asset_Documents.py: lectura via get_instance_document_bytes
#   - requirements.txt: +supabase>=2.0.0
#   - .streamlit/secrets.toml.example: template con sección [supabase]
#   - data/supabase_schema.sql: SQL para crear la tabla
#   - docs/supabase_setup.md: guía paso a paso
#
# Ciclo 9 sigue en dev. Cuando lo valides, mergeamos junto con
# Ciclo 8 a main como v2.0.
#
# Ejecutar:
#   bash _publish_ciclo9_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon — Ciclo 9: Persistencia Supabase (dev)"
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
echo "[1] Adoptando archivos del Ciclo 9..."
git add core/instance_repository.py
git add core/instance_state.py
git add core/instance_selector.py
git add pages/17_Asset_Documents.py
git add requirements.txt
git add .streamlit/secrets.toml.example
git add data/supabase_schema.sql
git add docs/supabase_setup.md
git status --short | head -10
echo ""

echo "[2] Commit..."
git commit -m "feat: Ciclo 9 — Supabase persistence backend (dev only)

Capa de abstraccion de storage para Asset Instances con dos
implementaciones que la app selecciona automaticamente:

* LocalFilesystemRepository (default): lo que veniamos usando hasta
  Ciclo 8. Guarda en data/instances/. Funciona sin internet, ideal
  para desarrollo. Pero en Streamlit Cloud el filesystem es efimero
  (datos perdidos en cada redeploy).

* SupabaseRepository (cuando hay credenciales): metadata como JSONB
  en tabla 'instances' de Supabase Postgres + binarios (PDFs) en
  Supabase Storage bucket 'instance-documents'. Sobrevive cualquier
  redeploy / reboot / caida del container.

Seleccion automatica: si st.secrets[supabase] existe y tiene url +
service_key validos -> Supabase. Si no -> Local. Sin tocar codigo
de la UI.

Componentes:
* core/instance_repository: interface + 2 backends + factory + cache
* core/instance_state: refactorizado para delegar al repo activo
  (todas las funciones publicas mantienen su firma, no rompe
  modulos existentes)
* core/instance_selector: badge en sidebar mostrando el backend
  activo (☁️ Supabase persistente vs 💾 local efimero)
* pages/17_Asset_Documents: usa get_instance_document_bytes en
  lugar de path file (compatible con backend Supabase que descarga
  bytes via API en vez de leer del filesystem local)
* requirements.txt: +supabase>=2.0.0 (dependencia opcional, lazy
  import — solo se carga si el backend Supabase se selecciona)
* .streamlit/secrets.toml.example: template extendido con la
  seccion [supabase] documentada y el setup paso a paso
* data/supabase_schema.sql: SQL para crear tabla instances con
  indices apropiados (updated_at desc, GIN sobre metadata) y
  trigger para mantener updated_at sincronizado
* docs/supabase_setup.md: guia paso a paso completa para que el
  usuario configure su proyecto Supabase + bucket + secret

Compatibilidad backwards:
* Sin secret de Supabase, la app se comporta exactamente como
  Ciclo 8 (filesystem local)
* Cualquier instance creada antes en filesystem sigue siendo legible
  por el backend Local (sin migracion)
* Para migrar data Local -> Supabase, hay un script futuro a hacer
  cuando el usuario quiera conservar lo que tenga local

Smoke test:
* Backend Local (sin secret): list/get/save/delete OK
* Backend Supabase (con mock client): save/load/list/upload/download/
  delete OK, badge ☁️ correctamente visible en sidebar mock
* Compile clean en core/* y pages/17"
echo "    OK"
echo ""

echo "[3] Push a dev..."
git push origin dev
echo "    OK"
echo ""

echo "================================================================"
echo " LISTO — Ciclo 9 en dev"
echo "================================================================"
echo ""
echo "Streamlit Cloud (dev) va a redesplegar en 1-3 minutos. Sin"
echo "credenciales Supabase, todo funciona EXACTAMENTE como Ciclo 8"
echo "(backend Local, badge 💾)."
echo ""
echo "Para activar Supabase en algun despliegue:"
echo ""
echo "  1. Seguí docs/supabase_setup.md (5–10 min):"
echo "     - Crear cuenta + proyecto en supabase.com"
echo "     - SQL Editor → pegar data/supabase_schema.sql → Run"
echo "     - Storage → New bucket: 'instance-documents' (privado)"
echo "     - Copiar Project URL + service_role key"
echo "  2. En Streamlit Cloud → tu app → Settings → Secrets, pegá:"
echo ""
echo "     [supabase]"
echo "     url         = \"https://...supabase.co\""
echo "     service_key = \"eyJh...\""
echo "     bucket      = \"instance-documents\""
echo ""
echo "  3. La app se redespliega sola. Vas a ver en la sidebar:"
echo "     ☁️ Persistencia Supabase activa — los datos sobreviven"
echo "        cualquier redeploy."
echo ""
echo "  4. Cualquier instancia / parametro / documento que crees"
echo "     ahora vive en Supabase. No se pierde nunca por reboots."
echo ""
echo "Cuando confirmes que el flujo Supabase anda fino en dev,"
echo "mergeamos Ciclos 8+9 juntos a main como v2.0."
echo "================================================================"
