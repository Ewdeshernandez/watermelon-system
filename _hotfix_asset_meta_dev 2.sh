#!/bin/bash
# Hotfix v3.19.2: list_archived_assets leer report_meta (nested).
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B hotfix/asset-meta-nested

git add api/services.py
git commit -m "fix: list_archived_assets lee asset desde report_meta (nested)

Los sidecars del archivo guardan instance_tag/asset_class/etc dentro
del objeto report_meta, no en el top-level. Mi código v3.19.0 los
buscaba directos y devolvía siempre lista vacía.

Ahora prueba en orden:
  report_meta.instance_tag → report_meta.train_description
  → report_meta.asset_class → top-level asset → instance_id

Mismo fix en list_reports_for_asset y get_latest_report_pdf."

git push -u origin hotfix/asset-meta-nested
git checkout dev
git merge --no-ff hotfix/asset-meta-nested -m "Merge hotfix/asset-meta-nested into dev"
git push origin dev

# Bump VERSION + push main
git checkout main
git pull origin main --ff-only
echo "v3.19.2" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.19.2" --allow-empty
git merge --no-ff dev -m "release(v3.19.2): fix list_archived_assets nested report_meta"
git tag -a "v3.19.2" -m "Release v3.19.2: nested asset metadata fix"
git push origin main
git push origin "v3.19.2"

echo ""
echo "================================================================"
echo " ✅ v3.19.2 en MAIN — watermelon-api redeploya solo en 1-2 min"
echo " Después: probar /activos en WhatsApp"
echo "================================================================"
