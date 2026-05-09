#!/bin/bash
# Push del hotfix Header() a main (la API en Render deploya de main).
set -e
cd "$(dirname "$0")"

git fetch origin
git checkout dev
git pull origin dev --ff-only
git checkout main
git pull origin main --ff-only

# Bump VERSION patch
echo "v3.18.1" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.18.1" --allow-empty

git merge --no-ff dev -m "release(v3.18.1): fix(api) _api_key_dependency Header() annotation

FastAPI trataba 'authorization' como query param sin la anotación
Header(alias='Authorization'). Resultado: 401 en todos los endpoints
autenticados. Ahora /rodamiento, /plantillas y demás funcionan
end-to-end desde el bot WhatsApp."

git tag -a "v3.18.1" -m "Release v3.18.1: API auth header fix"
git push origin main
git push origin "v3.18.1"

echo ""
echo "================================================================"
echo " ✅ v3.18.1 en MAIN"
echo " watermelon-api redeploya en Render en 1-2 min."
echo "================================================================"
