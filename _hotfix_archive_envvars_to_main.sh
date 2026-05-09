#!/bin/bash
set -e
cd "$(dirname "$0")"

VERSION="v3.19.1"

git fetch origin
git checkout dev
git pull origin dev --ff-only
git checkout main
git pull origin main --ff-only

echo "v3.19.1" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.19.1" --allow-empty

git merge --no-ff dev -m "release(${VERSION}): fix Supabase credentials para API en Render"
git tag -a "${VERSION}" -m "Release ${VERSION}: Supabase env vars fallback"
git push origin main
git push origin "${VERSION}"

echo ""
echo "================================================================"
echo " ✅ ${VERSION} en MAIN — watermelon-api redeploya en Render"
echo ""
echo " ► PASO MANUAL: agregar 2 env vars al servicio watermelon-api"
echo "   en Render dashboard → Settings → Environment:"
echo ""
echo "     SUPABASE_URL          = https://<tu-proyecto>.supabase.co"
echo "     SUPABASE_SERVICE_KEY  = <tu service role key>"
echo ""
echo "   (los mismos valores que tenés en wm-test/wm-home secrets"
echo "    bajo [supabase] url + service_key)"
echo "================================================================"
