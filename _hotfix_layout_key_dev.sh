#!/bin/bash
# Hotfix v3.26.2: KeyError 'cylinder_y_pct' — quedó referencia vieja
# en sensor_default_position tras el rediseño boxer.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B hotfix/recip-layout-key

echo "v3.26.2" > VERSION

git add core/recip_schematic.py VERSION
git commit -m "fix(recip): KeyError 'cylinder_y_pct' tras rediseño boxer

sensor_default_position() todavía leía la key vieja. Ahora lee
cylinder_top_y_pct / cylinder_bottom_y_pct correctamente.

VERSION → v3.26.2"

git push -u origin hotfix/recip-layout-key
git checkout dev
git merge --no-ff hotfix/recip-layout-key -m "Merge hotfix/recip-layout-key into dev"
git push origin dev

# Direct to main
git checkout main
git pull origin main --ff-only
echo "v3.26.2" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.26.2" --allow-empty
git merge --no-ff dev -m "release(v3.26.2): fix KeyError boxer layout"
git tag -a "v3.26.2" -m "Release v3.26.2"
git push origin main
git push origin "v3.26.2"

echo ""
echo "================================================================"
echo " ✅ v3.26.2 en MAIN — esperá 1-2 min y reintentá el wizard"
echo "================================================================"
