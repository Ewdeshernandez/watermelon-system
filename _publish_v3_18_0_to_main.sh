#!/bin/bash
# Release v3.18.0: render-api config (deploy ready).
set -e
cd "$(dirname "$0")"

VERSION="v3.18.0"

git fetch origin
git checkout dev
git pull origin dev --ff-only

echo "▶ Tests..."
python3 tests/run_smoke.py 2>&1 | tail -3

git checkout main
git pull origin main --ff-only

PRE_TAG="pre-${VERSION}-$(date +%Y%m%d)"
git tag -f "${PRE_TAG}"

# Bump VERSION
echo "v3.18.0" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.18.0" --allow-empty

git merge --no-ff dev -m "release(${VERSION}): merge dev -> main · API Render deploy config

Ciclo 18.4 — config para deployar el API REST público en Render free tier.

  ► requirements-api-minimal.txt: deps mínimas (fastapi+uvicorn+
    numpy+pandas, sin streamlit/plotly/kaleido). Cold start <30s.
  ► render-api.yaml: declaración del web service.

Próximo paso manual en Render dashboard (5 min):
  1. New Web Service → connect watermelon-system
  2. Build: pip install -r requirements-api-minimal.txt
  3. Start: uvicorn api.app:app --host 0.0.0.0 --port \$PORT
  4. Env: WATERMELON_API_KEYS = <key>

Sin cambios en producción Streamlit."

git tag -a "${VERSION}" -m "Release ${VERSION}: API Render deploy config"
git push origin main
git push origin "${VERSION}"

echo ""
echo "================================================================"
echo " ✅ RELEASE ${VERSION}"
echo "================================================================"
echo " Streamlit app: SIN cambios visibles."
echo " Próximo paso: crear el web service del API en Render."
echo "================================================================"
