#!/bin/bash
# Hotfix v3.18.1: api/app.py — anotación Header() en _api_key_dependency.
# Sin esto FastAPI trataba 'authorization' como query param y nunca leía
# el header → todos los endpoints autenticados devolvían 401.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B hotfix/api-auth-header

git add api/app.py
git commit -m "fix(api): _api_key_dependency necesita Header() explícito

Sin la anotación Header(alias='Authorization'), FastAPI trataba el
parámetro como query string y nunca veía el header. Resultado:
todos los endpoints autenticados respondían 401 'Missing
Authorization header' aunque el cliente sí mandara Bearer key.

Probado con curl: ahora /v1/bearings/overlay y demás funcionan."

git push -u origin hotfix/api-auth-header
git checkout dev
git merge --no-ff hotfix/api-auth-header -m "Merge hotfix/api-auth-header into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Hotfix en DEV"
echo " Render redeploya el watermelon-api en 1-2 min."
echo " Después: curl -H 'Authorization: Bearer watermelon-bot-key-2026' \\"
echo "          https://watermelon-api-bpv4.onrender.com/v1/bearings/overlay?model=SKF%206319&rpm=3600"
echo " Esperado: JSON con familias BPFO/BPFI/BSF/FTF"
echo "================================================================"
