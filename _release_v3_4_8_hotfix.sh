#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.4.8 HOTFIX URGENTE → MAIN
# =============================================================
# HOTFIX URGENTE Ciclo 17.24.1 — fix del hero roto en producción.
#
# Bug que matamos:
#   En v3.4.7 metí un <script> JavaScript embebido DENTRO del mismo
#   st.markdown que renderizaba el HTML del hero. Streamlit sanitiza
#   los <script> dentro de st.markdown POR SEGURIDAD, lo cual
#   provocó que TODO el bloque se renderizara como TEXTO PLANO en
#   pantalla. El usuario veía el código HTML/JS literal del hero
#   en lugar del card oscuro con saludo + reloj.
#
#   Notar: hay otro <script> en _landing.py del Cmd+K (Ciclo 17.13)
#   que SÍ funciona, porque está en su PROPIO st.markdown aislado.
#   La regla aprendida: scripts inline solo funcionan en bloques
#   markdown CHICOS y aislados, no embebidos con HTML grande.
#
# Fix:
#   - Eliminado el bloque <script> del hero
#   - Reloj vuelve a server-side (calculado en Python con la hora
#     del server). Pierde la auto-detect de zona del browser que
#     prometía la v3.4.7, pero al menos NO rompe el hero.
#   - Línea "último reporte: <activo> · hace 2h" ahora se calcula
#     completamente en Python (datetime.fromisoformat + diff).
#   - Eliminado el countdown "próximo turno en Xh Ymin" porque
#     era JS-dependiente.
#
# Lo que SÍ se mantiene del paquete v3.4.7:
#   ✓ Avatar mini al fondo del sidebar (17.22)
#   ✓ Sacar header "NAVEGACIÓN" del sidebar (17.23)
#   ✓ Footer del sidebar limpio (17.24 P1) — solo dot + versión
#   ✓ Cards KPI clickables con botón "Ver →" (17.24 P2)
#   ✓ Línea "Último reporte: ..." en el hero (17.24 P3 parcial)
#
# Lo que se elimina:
#   ✗ Reloj live auto-zona-horaria (era el script roto)
#   ✗ Countdown próximo turno
#
# Plan futuro:
#   Si querés re-intentar el reloj live, hay que hacerlo con
#   un st.markdown SEPARADO (igual al patrón del Cmd+K). Lo dejo
#   para otro día con cabeza fresca, no en caliente.
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🚨 HOTFIX v3.4.8 → MAIN URGENTE  (fix hero roto en producción)"
echo "================================================================"
echo ""
echo "Bug que matamos:"
echo "  El <script> embebido en el st.markdown del hero hizo que"
echo "  Streamlit sanitizara TODO el bloque y mostrara el HTML como"
echo "  texto plano en pantalla. Hero completamente roto en main."
echo ""
echo "Fix:"
echo "  Sacar el script. Reloj vuelve a server-side. Mantener todos"
echo "  los otros polish UX (avatar, sidebar, cards, último reporte)."
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

echo "▶ 1/7  Commit del hotfix en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged
git checkout HEAD -- _release_v3_4_2_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_3_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_4_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_5_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_6_ux.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_7_ux.sh 2>/dev/null || true

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/auth.py pages/_landing.py _release_v3_4_8_hotfix.sh
    git commit -m "hotfix(17.24.1): revertir reloj live <script> que rompió hero en main

Bug crítico en v3.4.7:
=====================
El <script> JS embebido dentro del MISMO st.markdown que el HTML
del hero hizo que Streamlit sanitizara todo el bloque (regla de
seguridad). Resultado: el HTML del hero se renderizaba como
TEXTO PLANO en pantalla en producción.

Hay otro <script> en el archivo (Cmd+K del Ciclo 17.13) que SÍ
funciona porque está en su PROPIO st.markdown aislado. La regla
aprendida: scripts inline solo funcionan en bloques markdown
chicos y aislados, no embebidos con HTML grande.

Fix:
- Eliminado el bloque <script> del hero
- Reloj vuelve a server-side (Python con hora del server)
- Línea 'último reporte' calculada Python-side con
  datetime.fromisoformat + diff
- Eliminado el countdown 'próximo turno en Xh Ymin' (era
  JS-dependiente)

Mantenidos del paquete original (v3.4.7):
- Avatar mini al fondo del sidebar (17.22)
- Sacar header 'Navegación' del sidebar (17.23)
- Footer del sidebar limpio (17.24 P1)
- Cards KPI clickables con botón 'Ver →' (17.24 P2)
- Línea 'Último reporte: ...' en el hero (17.24 P3 parcial)

Eliminados (porque eran JS-dependientes):
- Reloj live auto-zona-horaria del browser
- Countdown próximo turno" || echo "  (nada nuevo para commitear)"
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
MERGE_MSG="hotfix(v3.4.8): merge dev -> main · revertir reloj live (rompió hero)

URGENTE — el hero del Home estaba mostrando el código HTML/JS
como texto plano en producción porque el <script> embebido en
st.markdown disparó el sanitizador de Streamlit.

Fix:
  Sacar el <script>, volver al reloj server-side. Mantener
  todos los polish UX que SÍ funcionaron (avatar, sidebar,
  cards, último reporte calculado Python-side).

Lección:
  scripts inline solo funcionan en st.markdown CHICOS y
  AISLADOS (como el Cmd+K del Ciclo 17.13), no mezclados
  con HTML grande."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.4.8..."
TAG_EXISTS=$(git tag -l "v3.4.8")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.4.8 ya existe. Saltando creación."
else
    git tag -a v3.4.8 -m "Hotfix v3.4.8 — Revertir reloj live JS (rompió hero en main)"
    echo "  ✓ Tag v3.4.8 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.4.8 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ HOTFIX v3.4.8 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo "     RECOMENDADO: Manage app → Reboot app después del deploy"
echo "     para arrancar limpio."
echo ""
echo " 👁  Cambios visibles en producción:"
echo ""
echo "    HOME — HERO:"
echo "    - Vuelve a renderizar normal (sin código HTML como texto)"
echo "    - Reloj server-side (formato 24h, hora del server)"
echo "    - Línea 'Último reporte: ...' funcional (Python-side)"
echo ""
echo "    SIDEBAR (mantenido):"
echo "    - Footer limpio: 🟢 v3.4.8"
echo "    - Avatar mini al fondo, sin header 'NAVEGACIÓN'"
echo ""
echo "    HOME — KPI CARDS (mantenido):"
echo "    - Botón 'Ver →' debajo de cada card (clickable)"
echo ""
echo " 📝 Pendiente para otro día:"
echo "    - Reloj live con auto-zona-horaria del browser:"
echo "      requiere usar st.markdown SEPARADO para el script,"
echo "      no embebido con el HTML del hero."
echo ""
echo "================================================================"
