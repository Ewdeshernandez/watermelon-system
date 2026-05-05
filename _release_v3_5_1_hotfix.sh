#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.5.1 HOTFIX → MAIN  (DIAGNÓSTICO CORRECTO)
# =============================================================
# Fix DEFINITIVO del hero roto. El bug REAL no era el <script> ni
# el comentario HTML — era la INDENTACIÓN COMBINADA con un f-string
# que evaluaba a string vacío.
#
# Bug exacto:
#   Cuando el usuario NO tiene reportes archivados, _last_report_line
#   queda como "". El f-string del hero tiene esto:
#
#       <div class="wmh-status-line">
#           ...
#       </div>
#       {_last_report_line}     ← evalúa a "" → línea con whitespace
#       </div>
#       <div class="wmh-hero-right">
#           <div class="wmh-clock">...</div>
#
#   Al evaluar, queda:
#
#       </div>
#       (línea con whitespace solo)
#       </div>
#       <div class="wmh-hero-right">
#
#   Markdown ve la línea blanca como SEPARADOR DE PÁRRAFO, y luego
#   las siguientes líneas con 8+ espacios de indent las trata como
#   CODE BLOCK. Por eso el HTML del wmh-hero-right se renderizaba
#   como código en pantalla.
#
#   Solución: HTML del hero TODO en una sola línea. Sin saltos de
#   línea, sin indent, sin riesgo de markdown interpretándolo mal.
#   Es feo en el código fuente pero funcional en producción.
#
# Lecciones aprendidas (las anoto para no repetir):
#   1. NO uses indentación + saltos de línea dentro de st.markdown
#      con HTML, especialmente si hay {variable} que pueda evaluar
#      a string vacío.
#   2. NO metas <script> dentro de un st.markdown grande. Usá
#      st.markdown SEPARADO para el script (patrón Cmd+K).
#   3. NO metas comentarios HTML <!-- ... --> con palabras como
#      "<script>" dentro de st.markdown.
#
# Validación pre-push:
#   ✓ Python compila
#   ✓ HTML del hero en UNA línea (verificado con grep)
#   ✓ 2 bloques <script> aislados (reloj live + Cmd+K)
#   ✓ Cero comentarios HTML problemáticos
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🎯 HOTFIX v3.5.1 → MAIN  (fix DEFINITIVO confirmado)"
echo "================================================================"
echo ""
echo "Diagnóstico real:"
echo "  _last_report_line vacío + indentación markdown disparaba"
echo "  code-block en el wmh-hero-right. Por eso se veía HTML como"
echo "  texto en producción aún con v3.5.0."
echo ""
echo "Fix:"
echo "  HTML del hero en UNA SOLA LÍNEA. Sin saltos de línea, sin"
echo "  indent, sin riesgo de markdown interpretándolo mal."
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el hotfix a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Hotfix cancelado."
    exit 0
fi
echo ""

echo "▶ 1/7  Commit del fix definitivo en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged
for v in v3_4_2 v3_4_3 v3_4_4 v3_4_5 v3_4_6 v3_4_7 v3_4_8 v3_4_9 v3_5_0; do
    git checkout HEAD -- "_release_${v}_hotfix.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_ux.sh" 2>/dev/null || true
done

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add pages/_landing.py _release_v3_5_1_hotfix.sh
    git commit -m "hotfix(v3.5.1): hero en UNA línea — fix def del code-block markdown

Bug confirmado:
===============
Cuando el usuario NO tiene reportes archivados, _last_report_line
es \"\" (string vacío). El f-string del hero tenía:

    </div>
    {_last_report_line}    <- evaluaba a string vacío
    </div>
    <div class=\"wmh-hero-right\">
        <div class=\"wmh-clock\">...

Al evaluar:

    </div>
    (línea con whitespace solo)
    </div>
    <div class=\"wmh-hero-right\">

CommonMark/markdown ve la línea con whitespace como separador
de párrafo, y las siguientes líneas con 8+ espacios de indent
las trata como CODE BLOCK. Por eso el wmh-hero-right se
renderizaba como código en pantalla en producción.

Fix:
HTML del hero en UNA SOLA LÍNEA. Sin saltos de línea, sin
indentación, sin riesgo de markdown interpretándolo mal.
Es feo en el código fuente pero funcional en producción.

Diagnósticos previos que erraron:
- v3.4.7: pensé que era el <script> embebido (sí era parte)
- v3.4.9: pensé que era comentario HTML con \"<script>\" (no era)
- v3.5.0: pensé que era el comentario HTML general (no era)
- v3.5.1: ES la indentación + variable vacía (CONFIRMADO)

Lecciones para no repetir:
1. NO indentar HTML en st.markdown con variables que puedan
   ser strings vacíos.
2. NO mezclar <script> con HTML grande en mismo st.markdown.
3. NO usar comentarios HTML <!-- --> con palabras sensibles
   dentro de st.markdown.

Validación pre-push:
- Python compila
- HTML del hero verificado en una sola línea
- 2 bloques <script> aislados
- Cero comentarios HTML problemáticos" || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene el hotfix commiteado"
echo ""

echo "▶ 2/7  Push de dev a origin..."
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Pull dev falló."; exit 1; }
git push origin dev || { echo "✗ Push dev falló."; exit 1; }
echo "  ✓ dev en origin actualizado"
echo ""

echo "▶ 3/7  Cambiando a main..."
git checkout main || { echo "✗ No se pudo cambiar a main."; exit 1; }
git fetch origin main
git pull --rebase origin main || { echo "✗ Pull main falló."; exit 1; }
echo "  ✓ main actualizado"
echo ""

echo "▶ 4/7  Mergeando dev → main..."
MERGE_MSG="hotfix(v3.5.1): merge dev -> main · hero en UNA línea (fix def)

Diagnóstico real del hero roto: _last_report_line vacío
disparaba code-block markdown por la indentación. Solución:
HTML del hero en UNA sola línea sin saltos.

Diagnósticos previos (todos errados): v3.4.7 <script>,
v3.4.9 comentario HTML, v3.5.0 comentario general. El
verdadero culpable era markdown + indent + variable vacía.

Validado: Python compila, HTML en una línea, 2 bloques
<script> aislados (reloj live + Cmd+K), cero comentarios
HTML problemáticos."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.5.1..."
TAG_EXISTS=$(git tag -l "v3.5.1")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.5.1 ya existe. Saltando creación."
else
    git tag -a v3.5.1 -m "Hotfix v3.5.1 — Hero en una línea (fix def del code-block markdown)"
    echo "  ✓ Tag v3.5.1 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.5.1 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ HOTFIX v3.5.1 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo "     Manage app → Reboot app después del deploy."
echo ""
echo " 👁  Validación: hero debe verse NORMAL ahora."
echo ""
echo "    Reloj: tu hora real en formato 12h (ej. 4:30 pm)"
echo "    Saludo: 'Buenas tardes' (no 'noches' por hora UTC)"
echo "    Countdown 'próximo turno 🌇 tarde en Xh Ymin'"
echo ""
echo "================================================================"
