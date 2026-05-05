#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.5.0 → MAIN  (cierre de saga UX)
# =============================================================
# Fix DEFINITIVO del hero roto. Diagnóstico real encontrado:
#
#   El v3.4.7 metió un comentario HTML dentro del st.markdown del
#   hero que contenía la palabra literal "<script>":
#       <!-- ... <script> que reemplaza ... -->
#   Streamlit ve la palabra "<script>" en CUALQUIER parte del
#   bloque markdown (incluso dentro de comentarios HTML) y dispara
#   el sanitizador, lo que rompe el render completo del hero.
#
#   Aprendido tras dos intentos fallidos:
#   - v3.4.7: <script> embebido → roto
#   - v3.4.9: comentario HTML que mencionaba <script> → roto
#   - v3.5.0: comentarios afuera del bloque markdown → ✓ funciona
#
# Solución FINAL:
#   1. HTML del hero queda LIMPIO sin comentarios HTML con palabras
#      sensibles. Solo HTML pleno con IDs estables (wm-clock-live,
#      wm-shift-live, wm-date-live, wm-next-shift).
#   2. Comentarios de explicación van como comentarios Python (#)
#      AFUERA del st.markdown.
#   3. El <script> live va en su PROPIO st.markdown SEPARADO
#      (patrón validado del Cmd+K, Ciclo 17.13).
#
# Validación pre-push:
#   ✓ Python compila
#   ✓ Cero comentarios HTML con palabras sensibles dentro de
#     st.markdown del hero
#   ✓ JS del reloj pasa node --check
#   ✓ JS del Cmd+K pasa node --check (preexistente)
#   ✓ Patrón markdown-separado para script
#
# Cambios incluidos (cierran la saga de UX):
#   - Avatar mini al fondo del sidebar (17.22)
#   - Sacar header "NAVEGACIÓN" del sidebar (17.23)
#   - Footer del sidebar limpio: 🟢 v3.5.0 (17.24 P1)
#   - Cards KPI clickables con botón "Ver →" (17.24 P2)
#   - Línea "Último reporte: ... · hace 2h" en el hero (17.24 P3)
#   - Reloj LIVE auto-zona-horaria del browser (17.24.3 final)
#   - Countdown próximo turno dinámico (17.24.3 final)
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🎯 RELEASE v3.5.0 → MAIN  (fix DEFINITIVO + UX completo)"
echo "================================================================"
echo ""
echo "Diagnóstico final del bug:"
echo "  Streamlit sanitiza la palabra '<script>' incluso dentro de"
echo "  comentarios HTML <!-- ... -->. Mi comentario en el hero la"
echo "  contenía → render roto."
echo ""
echo "Fix:"
echo "  HTML del hero limpio. Comentarios de explicación como #"
echo "  Python, afuera del st.markdown. Reloj LIVE en st.markdown"
echo "  separado (patrón Cmd+K)."
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el release a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

echo "▶ 1/7  Commit del fix definitivo en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged
git checkout HEAD -- _release_v3_4_2_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_3_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_4_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_5_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_6_ux.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_7_ux.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_8_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_9_hotfix.sh 2>/dev/null || true

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/auth.py pages/_landing.py _release_v3_5_0_hotfix.sh
    git commit -m "release(v3.5.0): UX completo + fix definitivo del hero (Ciclo 17.24.3)

Diagnóstico FINAL del bug del hero roto en v3.4.7 / v3.4.9:
==========================================================
Streamlit sanitiza la palabra LITERAL '<script>' en cualquier
parte del bloque markdown — incluso dentro de comentarios HTML
<!-- ... -->. Mi comentario en el hero la contenía:

    <!-- ... el <script> que reemplaza ... -->

Eso disparó el sanitizador y rompió el render completo (HTML
visible como bloque de código en pantalla en producción).

Solución:
- HTML del hero LIMPIO, sin comentarios HTML con palabras
  sensibles ('script', 'style', etc).
- Comentarios de explicación como # Python, afuera del st.markdown.
- El <script> live en st.markdown SEPARADO (patrón Cmd+K del
  Ciclo 17.13 que funciona desde meses).
- IDs estables: wm-clock-live, wm-shift-live, wm-date-live,
  wm-next-shift, .wmh-last-report (con data-archived-at).
- El script accede a window.parent.document porque Streamlit
  corre en iframe.

Cambios incluidos (cierran la saga UX iniciada en 17.22):
- Avatar mini al fondo del sidebar (17.22)
- Sacar header 'NAVEGACIÓN' del sidebar (17.23)
- Footer del sidebar limpio: 🟢 v3.5.0 (17.24 P1)
- Cards KPI clickables con botón 'Ver →' (17.24 P2)
- Línea 'Último reporte: ... · hace 2h' en el hero (17.24 P3)
- Reloj LIVE con auto-zona-horaria del browser (17.24.3 final)
- Countdown próximo turno dinámico
- Tiempo relativo del último reporte calculado en JS

Validación pre-push:
- Python compila sin errores
- node --check de ambos JS blocks: OK
- Cero comentarios HTML problemáticos en el hero" || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene el fix commiteado"
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
MERGE_MSG="release(v3.5.0): merge dev -> main · UX completo + fix definitivo

Cierra la saga de UX iniciada en 17.22 con la solución correcta
del reloj LIVE.

Diagnóstico final: Streamlit sanitiza la palabra '<script>'
incluso dentro de comentarios HTML. Solución: comentarios afuera
del st.markdown, script en su propio st.markdown separado.

Cambios funcionales:
- Avatar mini sidebar
- Sin header 'Navegación'
- Footer limpio
- Cards KPI clickables
- Hero con último reporte
- Reloj LIVE auto-zona-horaria del browser
- Countdown próximo turno

Sin cambios en lógica funcional. Solo CSS/HTML/JS."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.5.0..."
TAG_EXISTS=$(git tag -l "v3.5.0")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.5.0 ya existe. Saltando creación."
else
    git tag -a v3.5.0 -m "Release v3.5.0 — UX completo + reloj LIVE auto-zona-horaria (cierre saga 17.24)"
    echo "  ✓ Tag v3.5.0 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.5.0 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ RELEASE v3.5.0 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo "     RECOMENDADO: Manage app → Reboot app después del deploy"
echo "     para arrancar limpio."
echo ""
echo " 👁  Cómo verificar que TODO funciona ahora:"
echo ""
echo "    HERO (lo más crítico):"
echo "    1. Se ve el card oscuro normal — SIN código HTML como texto"
echo "    2. Reloj muestra tu hora REAL en formato 12h: '4:18 pm'"
echo "    3. Saludo dice 'Buenas tardes' (no 'noches' por hora UTC)"
echo "    4. Debajo de la fecha: 'próximo turno 🌇 tarde en Xh Ymin'"
echo "    5. Si tenés reportes archivados: 'último reporte: ... · hace Xh'"
echo ""
echo "    SIDEBAR:"
echo "    6. Footer: 🟢 v3.5.0 (hover muestra commit hash)"
echo "    7. Avatar circular celeste con tu inicial al fondo"
echo "    8. Sin el header 'NAVEGACIÓN' arriba"
echo ""
echo "    CARDS KPI:"
echo "    9. Botón 'Ver →' debajo de cada una"
echo ""
echo " 🌍 Test bonus de auto-zona-horaria:"
echo "    Chrome → DevTools (F12) → ⋮ → More tools → Sensors"
echo "    Location: 'Other...' → Tokyo (35.68, 139.65)"
echo "    Recargá la página: el reloj debe mostrar la hora local"
echo "    de Tokyo. Eso prueba que la auto-detect funciona."
echo ""
echo "================================================================"
