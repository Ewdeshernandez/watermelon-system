#!/bin/bash
# =============================================================
# Watermelon — HOTFIX: streamlit_image_coordinates needs PIL.Image
# =============================================================
# El widget streamlit_image_coordinates rechaza bytes crudos:
#   ValueError: Must pass a string, Path, numpy array or object
#                with a save method
# Le pasabamos render_on_schematic() que devuelve bytes. Ahora lo
# decodificamos a PIL.Image antes de pasarlo.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/00_Machinery_Library.py
git status --short | head

git commit -m "fix(ctp): pasar PIL.Image en vez de bytes a streamlit_image_coordinates

ValueError: Must pass a string, Path, numpy array or object with a
save method. El widget rechaza bytes crudos. Decodificamos a PIL.Image
antes de pasarlo." || echo "Nothing to commit"

git push origin dev

echo ""
echo "✓ Hotfix pusheado a dev. Refrescá el browser."
