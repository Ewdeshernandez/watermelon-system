#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.4.9 HOTFIX URGENTE → MAIN
# =============================================================
# Fix definitivo del hero roto + reloj LIVE bien implementado.
#
# Historia de los hotfixes:
#   v3.4.7 — metí <script> dentro del st.markdown del hero. ROTO:
#            Streamlit sanitiza scripts dentro de markdown grandes.
#            HTML se renderizó como TEXTO en pantalla en main.
#   v3.4.8 — script se preparó pero nunca llegó a producción.
#            main quedó en v3.4.7 roto.
#   v3.4.9 — implementación CORRECTA: script en st.markdown
#            SEPARADO (igual al patrón del Cmd+K que funciona
#            desde Ciclo 17.13). Validado sintácticamente con
#            node --check. SE PUEDE PUSHEAR.
#
# Solución técnica:
#   - HTML del hero queda CHICO con IDs estables, server-side
#     Python para fallback inicial:
#       <div class="wmh-clock" id="wm-clock-live">{server_time}</div>
#   - Después del st.markdown del hero, OTRO st.markdown DEDICADO
#     al script JS (igual al patrón del Cmd+K, validado meses):
#       st.markdown(\"\"\"<script>...</script>\"\"\", unsafe_allow_html=True)
#   - El script accede a window.parent.document porque Streamlit
#     corre en iframe.
#   - Refresca cada 30s sin necesidad de rerun.
#
# Validación previa al push:
#   ✓ Compila Python sin errores
#   ✓ 2 bloques <script> en el archivo: reloj live (línea 651) y
#     Cmd+K (línea 767, pre-existente y funcional)
#   ✓ node --check del JS del reloj: OK
#   ✓ node --check del JS del Cmd+K: OK (igual que siempre)
#
# Resultado esperado en producción:
#   - Hero vuelve a verse normal (no más HTML como texto)
#   - Reloj muestra hora REAL del browser del usuario:
#       Bogotá → 6:05 am
#       California → 4:05 am
#       Tokyo → 9:05 pm
#   - Saludo, turno, fecha también dinámicos
#   - Countdown "próximo turno 🌙 noche en 4h 22min" dinámico
#   - Línea "último reporte: ... hace 2h" con tiempo relativo
#     calculado en el browser
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🚨 HOTFIX v3.4.9 → MAIN  (fix hero roto + reloj LIVE bien hecho)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.24.2  Reloj LIVE en st.markdown SEPARADO (patrón Cmd+K)"
echo "           - HTML hero limpio con IDs estables + fallback server-side"
echo "           - Script JS aislado en su propio st.markdown"
echo "           - Auto-zona-horaria del browser, formato 12h am/pm"
echo "           - Countdown próximo turno + último reporte dinámicos"
echo ""
echo "Validación previa: Python compila + 2 bloques JS validados node --check"
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
git checkout HEAD -- _release_v3_4_8_hotfix.sh 2>/dev/null || true

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/auth.py pages/_landing.py _release_v3_4_9_hotfix.sh
    git commit -m "hotfix(17.24.2): reloj LIVE bien implementado (script SEPARADO del HTML)

Cierra el ciclo del bug introducido en v3.4.7 (script embebido en
st.markdown del hero) que rompió main mostrando HTML como texto.

Solución técnica:
- HTML del hero queda CHICO con IDs estables (wm-clock-live, etc)
  y fallback server-side Python ({_greet['time_hhmm']}) para que
  haya valor inicial mientras carga el JS.
- Después del st.markdown del hero, OTRO st.markdown SEPARADO
  con SOLO el bloque <script>. Mismo patrón que el Cmd+K del
  Ciclo 17.13 que funciona desde meses sin problemas.
- El script accede a window.parent.document (Streamlit corre en
  iframe, los IDs viven en el documento padre).
- Refresca cada 30s, también re-engancha tras DOMContentLoaded
  y con setTimeout de 800ms para sobrevivir reruns.

Funcionalidad live (todo client-side via new Date()):
- Reloj con hora REAL del browser: Bogotá 6:05 am, California
  4:05 am, Tokyo 9:05 pm. Formato 12h con am/pm.
- Saludo dinámico (TURNO MAÑANA / TARDE / NOCHE según hora)
- Fecha en español
- Countdown 'próximo turno 🌙 noche en 4h 22min'
- Línea 'último reporte: <activo> · hace 2h' con tiempo relativo
  calculado en JS desde data-archived-at

Validación pre-push:
- Python compila OK
- 2 bloques <script> en el archivo: reloj live + Cmd+K
- node --check de ambos bloques: OK

Mantenidos del paquete original:
- Avatar mini al fondo del sidebar (17.22)
- Sacar header 'Navegación' del sidebar (17.23)
- Footer del sidebar limpio (17.24 P1)
- Cards KPI clickables con botón 'Ver →' (17.24 P2)
- Línea 'Último reporte: ...' en el hero (17.24 P3)" || echo "  (nada nuevo para commitear)"
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
MERGE_MSG="hotfix(v3.4.9): merge dev -> main · reloj LIVE bien implementado

Fix definitivo del hero roto + reloj LIVE con auto-zona-horaria.

Solución correcta: script JS en st.markdown SEPARADO del HTML
(patrón del Cmd+K que funciona desde Ciclo 17.13). Validado
sintácticamente con node --check antes de pushear.

Resultado en producción:
- Hero vuelve a verse normal (sin HTML como texto)
- Reloj muestra hora real del browser del usuario
- Saludo, turno, fecha, countdown y último reporte dinámicos
- Refresca cada 30s

Mantiene también: avatar mini sidebar, sacar 'Navegación',
footer limpio, cards KPI clickables, línea 'último reporte'."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.4.9..."
TAG_EXISTS=$(git tag -l "v3.4.9")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.4.9 ya existe. Saltando creación."
else
    git tag -a v3.4.9 -m "Hotfix v3.4.9 — Reloj LIVE bien hecho (script en st.markdown separado)"
    echo "  ✓ Tag v3.4.9 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.4.9 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ HOTFIX v3.4.9 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo "     RECOMENDADO: Manage app → Reboot app después del deploy."
echo ""
echo " 👁  Cómo verificar que TODO funciona:"
echo ""
echo "    1. HERO se ve NORMAL (sin código HTML como texto) — fix del bug"
echo "    2. RELOJ muestra tu hora REAL en formato 12h con am/pm"
echo "       (ej. 4:18 pm si son las 16:18 en Bogotá)"
echo "    3. SALUDO dice 'Buenas tardes' (no 'Buenas noches' por hora UTC)"
echo "    4. FECHA debajo del reloj con día de semana y mes correctos"
echo "    5. COUNTDOWN debajo de fecha: 'próximo turno 🌇 tarde en Xh Ymin'"
echo "    6. ÚLTIMO REPORTE muestra 'hace X min/h' relativo a tu zona"
echo ""
echo " 🌍 Test en otra zona horaria (opcional):"
echo "    Abrí Chrome → DevTools (F12) → ⋮ More tools → Sensors →"
echo "    Location: 'Other...' → poné lat/lon de Tokyo o California."
echo "    Recargá la página: el reloj debe mostrar la hora local de"
echo "    esa ubicación. Eso prueba que la auto-detect funciona."
echo ""
echo "================================================================"
