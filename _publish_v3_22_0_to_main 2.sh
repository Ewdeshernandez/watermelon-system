#!/bin/bash
set -e
cd "$(dirname "$0")"

VERSION="v3.22.0"

git fetch origin
git checkout dev
git pull origin dev --ff-only
python3 tests/run_smoke.py 2>&1 | tail -3

git checkout main
git pull origin main --ff-only
echo "v3.22.0" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.22.0" --allow-empty

git merge --no-ff dev -m "release(${VERSION}): Wizard guiado de activos (Ciclo 21)"
git tag -a "${VERSION}" -m "Release ${VERSION}: Machinery Wizard"
git push origin main
git push origin "${VERSION}"

echo ""
echo "================================================================"
echo " ✅ ${VERSION} en MAIN — wm-home-final-2026 redeploya en 1-2 min"
echo "================================================================"
