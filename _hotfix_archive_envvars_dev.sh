#!/bin/bash
# Hotfix: reports_archive.py lee credenciales Supabase tanto de
# st.secrets (Streamlit) como de env vars (API REST en Render).
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B hotfix/archive-supabase-envvars

git add core/reports_archive.py requirements-api-minimal.txt
git commit -m "fix: reports_archive lee Supabase también de env vars

Sin esto, la API REST en Render no podía acceder al archivo de
reportes (st.secrets no existe sin Streamlit). Ahora intenta:
  1. st.secrets[supabase]      (Streamlit Cloud)
  2. SUPABASE_URL + SUPABASE_SERVICE_KEY env vars (Render API)

requirements-api-minimal.txt: + supabase>=2.0.0"

git push -u origin hotfix/archive-supabase-envvars
git checkout dev
git merge --no-ff hotfix/archive-supabase-envvars -m "Merge hotfix/archive-supabase-envvars into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Hotfix en DEV"
echo " Después: bash _hotfix_archive_envvars_to_main.sh"
echo "================================================================"
