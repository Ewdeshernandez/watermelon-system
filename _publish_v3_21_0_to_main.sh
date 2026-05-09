#!/bin/bash
set -e
cd "$(dirname "$0")"

VERSION="v3.21.0"

git fetch origin
git checkout dev
git pull origin dev --ff-only
python3 tests/run_smoke.py 2>&1 | tail -3

git checkout main
git pull origin main --ff-only
echo "v3.21.0" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.21.0" --allow-empty

git merge --no-ff dev -m "release(${VERSION}): Admin UI para clients.json (Ciclo 20B)"
git tag -a "${VERSION}" -m "Release ${VERSION}: Admin UI multi-tenant"
git push origin main
git push origin "${VERSION}"

echo ""
echo "================================================================"
echo " ✅ ${VERSION} en MAIN"
echo " wm-home-final-2026 redeploya en 1-2 min"
echo " Visible solo para admin: 'Admin · Clientes' en sidebar"
echo "================================================================"
