#!/bin/bash
# Hotfix: invertir prioridad VERSION file > git_latest_tag.
# Streamlit Cloud hace shallow clone y pierde los tags v3.x recientes,
# dejando solo v2.1/v2.5 visibles. Por eso el footer mostraba v2.1
# aunque el commit hash y la fecha eran correctos.
#
# Solución: VERSION file es source of truth declarativa. Lo subimos
# a v3.16.0 y siempre ganará sobre git tags.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B hotfix/version-priority-fix

git add VERSION core/version.py
git commit -m "fix: VERSION file prioridad > git_latest_tag (Streamlit Cloud shallow clone)

Streamlit Cloud hace shallow clone que NO incluye los tags v3.x del
repo. Por eso aunque el commit hash mostraba 2510455c (último), el
footer decía v2.1 (tag más viejo que sí estaba en el clone).

Cambio en core/version.py: prioridad ahora es
  1. WM_VERSION env (override total)
  2. VERSION file  ← PROMOVIDO
  3. git_latest_tag
  4. git_desc
  5. _FALLBACK_VERSION

VERSION file actualizado a v3.16.0.

Workflow nuevo: cada release bump VERSION manual antes de tag git."

git push -u origin hotfix/version-priority-fix
git checkout dev
git merge --no-ff hotfix/version-priority-fix -m "Merge hotfix/version-priority-fix into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Hotfix priority en DEV"
echo "================================================================"
echo " Después de redeploy DEBE mostrar v3.16.0."
echo " Si NO → Reboot manual de la app en Streamlit Cloud:"
echo "   share.streamlit.io → wm-test → ⋮ → Reboot app"
echo "================================================================"
