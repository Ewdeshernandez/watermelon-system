#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.4.7 UX → MAIN
# =============================================================
# Release de polish UX. NO toca lógica funcional. Solo CSS/HTML/JS
# del sidebar y el Home (core/auth.py + pages/_landing.py).
#
# Cambios incluidos en este release:
#
#   CICLO 17.24 — UX polish completo del Home
#   ──────────────────────────────────────────
#
#   P0 (bonus URGENTE): Reloj LIVE auto-zona-horaria
#       Antes: hora del server (UTC), formato 24h, mostraba mal en
#       cualquier huso horario que no sea UTC.
#       Ahora: reloj completamente en JavaScript del browser. Detecta
#       automáticamente la zona del usuario (Bogotá → 6:05 am, California
#       → 4:05 am, Tokyo → 9:05 pm). Formato 12h con am/pm. Saludo,
#       turno y fecha también dinámicos por coherencia. Refresca cada 30s.
#
#   P1: Footer del sidebar limpio
#       Antes: "v3.4.6 production d4d2db45" (info técnica visible)
#       Ahora: dot del entorno + número de versión. Commit hash y env
#       quedan como tooltip al hover. Más discreto, mismo nivel de info
#       para troubleshooting.
#
#   P2: Cards KPI clickables
#       Las 4 cards del Home (Activos / Críticos / En atención / Sin
#       clasificar) ahora tienen un botón "Ver →" debajo que navega a
#       la página correspondiente:
#         - Activos en flota → Machinery Library
#         - Críticos          → Diagnostics (filtro severity=danger)
#         - En atención       → Diagnostics (filtro severity=warning)
#         - Sin clasificar    → Machinery Library (filtro unclassified)
#       Pre-setea filtros via session_state para que la página destino
#       arranque con el filtro aplicado.
#
#   P3: Hero del Home más vivo
#       Agregamos al hero (Buenas tardes + reloj):
#         - Línea "Último reporte: <activo> · hace 2h" debajo del status,
#           con tiempo relativo calculado en JS
#         - Countdown "próximo turno 🌙 noche en 4h 22min" debajo de
#           la fecha, dinámico cada 30s
#       Pequeños toques que dan sensación de panel SCADA vivo.
#
# Archivos modificados:
#   - core/auth.py        (footer del sidebar)
#   - pages/_landing.py   (reloj LIVE + cards clickables + hero vivo)
#
# Nota: este release viene encima de v3.4.6 (avatar mini + sacar
# Navegación). Si v3.4.6 no se merged a main todavía, este merge
# trae ambos en uno. El script lo maneja igual.
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " ✨ RELEASE v3.4.7 UX → MAIN  (polish completo del Home)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.24 P0  Reloj LIVE auto-zona-horaria + formato 12h"
echo "  17.24 P1  Footer del sidebar limpio (dot + versión)"
echo "  17.24 P2  Cards KPI clickables (navegan a página filtrada)"
echo "  17.24 P3  Hero más vivo (último reporte + countdown turno)"
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

echo "▶ 1/7  Commit del 17.24 en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged
git checkout HEAD -- _release_v3_4_2_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_3_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_4_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_5_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_6_ux.sh 2>/dev/null || true

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/auth.py pages/_landing.py _release_v3_4_7_ux.sh
    git commit -m "ux(17.24): polish completo del Home — reloj LIVE + footer + cards + hero

P0 Reloj LIVE auto-zona-horaria del browser
============================================
Antes: hora del server (UTC), formato 24h. Mostraba mal en cualquier
huso horario que no sea UTC.
Ahora: reloj completamente en JavaScript del browser con new Date().
Detecta zona del usuario automáticamente (Bogotá → 6:05 am, California
→ 4:05 am, Tokyo → 9:05 pm). Formato 12h con am/pm. Saludo, turno
y fecha también dinámicos por coherencia. Refresca cada 30s.

P1 Footer del sidebar limpio
============================
Antes: 'v3.4.6 production d4d2db45' (info técnica visible).
Ahora: dot del entorno (verde/ámbar/azul según production/staging/dev)
+ número de versión. Commit hash y env como tooltip al hover. Más
discreto, mismo nivel de info para troubleshooting.

P2 Cards KPI clickables
=======================
Las 4 cards del Home (Activos/Críticos/En atención/Sin clasificar)
tienen botón 'Ver →' debajo que navega a página correspondiente:
- Activos en flota → Machinery Library
- Críticos        → Diagnostics (session: filter_severity=danger)
- En atención     → Diagnostics (session: filter_severity=warning)
- Sin clasificar  → Library (session: filter_status=unclassified)

P3 Hero del Home más vivo
=========================
Dos toques nuevos en el hero:
- Línea 'Último reporte: <activo> · hace 2h' debajo del status,
  con tiempo relativo calculado en JS respecto a la zona del browser
- Countdown 'próximo turno 🌙 noche en 4h 22min' debajo de la fecha,
  dinámico cada 30s

Solo CSS/HTML/JS. Sin cambios funcionales en otras páginas." || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene los cambios commiteados"
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
MERGE_MSG="ux(v3.4.7): merge dev -> main · Ciclo 17.24 polish completo del Home

Release de polish UX. Sin cambios funcionales — solo CSS/HTML/JS
del sidebar y el Home.

P0 Reloj LIVE auto-zona-horaria (browser-side JS)
P1 Footer del sidebar limpio (dot + versión, hash en tooltip)
P2 Cards KPI clickables (navegan con filtro pre-aplicado)
P3 Hero vivo (último reporte + countdown próximo turno)

Si v3.4.6 (avatar mini + sacar Navegación) no estaba mergeado a
main todavía, este merge trae ambos paquetes (17.22 + 17.23 +
17.24) en uno solo.

Archivos: core/auth.py, pages/_landing.py."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.4.7..."
TAG_EXISTS=$(git tag -l "v3.4.7")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.4.7 ya existe. Saltando creación."
else
    git tag -a v3.4.7 -m "Release v3.4.7 — Ciclo 17.24 polish UX (reloj live + footer + cards + hero)"
    echo "  ✓ Tag v3.4.7 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.4.7 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ RELEASE v3.4.7 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo ""
echo " 👁  Cambios visibles en producción:"
echo ""
echo "    SIDEBAR:"
echo "    - Footer minimalista: 🟢 v3.4.7 (hover → tooltip con commit)"
echo ""
echo "    HOME — HERO:"
echo "    - Reloj con tu hora real (Bogotá si estás en Bogotá, etc)"
echo "    - Formato '6:05 am' / '2:30 pm' con am/pm"
echo "    - Saludo y turno se actualizan SOLOS según la hora real"
echo "    - 'Último reporte: <activo> · hace 2h' debajo del status"
echo "    - 'próximo turno 🌙 noche en 4h 22min' debajo de la fecha"
echo ""
echo "    HOME — KPI CARDS:"
echo "    - Botón 'Ver →' debajo de cada una de las 4 cards"
echo "    - Click te lleva a Diagnostics o Library con filtro aplicado"
echo ""
echo " 📊 Sin cambios funcionales:"
echo ""
echo "    Toda la lógica (navegación, auth, generación PDF, cargas,"
echo "    cache Supabase, etc.) sigue igual que en v3.4.6/v3.4.5."
echo ""
echo " 💡 Pendiente en backlog (no urgente):"
echo "    - Rotar Azure client_secret (#86) — seguridad"
echo "    - 5ta Redirect URL de wm-test en Supabase (#36)"
echo ""
echo "================================================================"
