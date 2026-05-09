#!/bin/bash
set -e
cd "$(dirname "$0")"

VERSION="v3.19.0"

git fetch origin
git checkout dev
git pull origin dev --ff-only
python3 tests/run_smoke.py 2>&1 | tail -3

git checkout main
git pull origin main --ff-only

echo "v3.19.0" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.19.0" --allow-empty

git merge --no-ff dev -m "release(${VERSION}): Asset Query endpoints (WhatsApp integration)

4 endpoints nuevos para que el bot WhatsApp pueda servir PDFs y
listar activos por chat. Capa servicios testeable sin FastAPI.
Sin cambios en core/ ni pages/."

git tag -a "${VERSION}" -m "Release ${VERSION}: Asset Query endpoints"
git push origin main
git push origin "${VERSION}"

echo ""
echo "================================================================"
echo " ✅ ${VERSION} en MAIN"
echo " watermelon-api redeploya en Render en 1-2 min."
echo "================================================================"
