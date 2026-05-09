#!/bin/bash
# Hotfix v3.21.1: card del archivo de reportes mostraba <div> crudos
# por indentación que Streamlit interpretaba como code block CommonMark.
# Fix: textwrap.dedent + HTML en una sola línea por bloque.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B hotfix/reports-card-html

git add pages/16_Reports.py
git commit -m "fix(reports): card del archivo no renderizaba HTML

El f-string con indentación interna >=4 espacios disparaba el
parser de CommonMark de Streamlit y mostraba los <div> como
bloques de código.

Fix: textwrap.dedent() para normalizar indentación + HTML
estructurado en líneas sin sangría problemática.

NO toca lógica, solo presentación visual del archivo."

git push -u origin hotfix/reports-card-html
git checkout dev
git merge --no-ff hotfix/reports-card-html -m "Merge hotfix/reports-card-html into dev"
git push origin dev

# Direct to main (es bug visual, riesgo bajo)
git checkout main
git pull origin main --ff-only
echo "v3.21.1" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.21.1" --allow-empty

git merge --no-ff dev -m "release(v3.21.1): fix card archivo Reports — HTML no renderizaba"
git tag -a "v3.21.1" -m "Release v3.21.1: Reports card HTML fix"
git push origin main
git push origin "v3.21.1"

echo ""
echo "================================================================"
echo " ✅ v3.21.1 en MAIN — wm-home-final-2026 redeploya en 1-2 min"
echo " Después: refrescá Reports → debe verse el card limpio"
echo "================================================================"
