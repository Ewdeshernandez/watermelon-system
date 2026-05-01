#!/bin/bash
# =============================================================
# Watermelon — HOTFIX: remover expander de debug del Tabular List
# =============================================================
# El expander "🔍 Debug: matching de sensores" ya cumplio su funcion
# (encontramos el bug del falso match cross-tipo). Lo quitamos para
# que la pagina quede limpia para producción. Si en algún caso futuro
# el matcher cae mal, se puede restaurar desde el historial git.
#
# El banner amarillo "Override criterio activo" NO se toca — ese es
# legítimo cuando el usuario tiene valores manuales en el sidebar
# avanzado. Si querés sacarlo también, expandí "⚙️ Override criterio
# para este analisis (avanzado)" en el sidebar y restaurá los
# defaults (alarm/danger del Sensor Map).
#
# Este hotfix queda en dev. Cuando lo apruebes, hacemos otro merge
# rápido a main si querés (o se acumula con el próximo ciclo).
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/01__Tabular_List.py
git status --short | head

git commit -m "fix(tabular): remover expander de debug del matching

Ya cumplio su funcion (encontramos el bug del falso match cross-tipo
con '*4*x*' vs '64x' del oversampling). Removido para que la pagina
de Tabular List quede limpia para produccion. Si en algun caso futuro
el matcher cae mal, restaurar el expander desde el historial git
(commit con BUILD 14c.3-debug-v2)."

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Expander de debug removido en dev"
echo "================================================================"
echo ""
echo "Para que la version sin debug llegue a main:"
echo "  bash _publish_v2_3_to_main.sh"
echo "  (vuelve a hacer commit + merge — el cleanup del debug viaja"
echo "  junto con el merge anterior)"
echo ""
echo "O si querés acumular con próximo ciclo, queda en dev hasta"
echo "el siguiente publish a main."
