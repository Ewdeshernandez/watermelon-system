#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.5.2 HOTFIX → MAIN
# =============================================================
# Solución FINAL del reloj 12h con auto-zona-horaria.
# SIN JavaScript en el browser. Todo server-side Python.
#
# Bug que matamos:
#   En v3.5.1 el reloj seguía mostrando 24h con hora del server
#   (UTC en Streamlit Cloud). Si el usuario está en Bogotá ve
#   "21:37" en lugar de "4:37 pm" porque el server da UTC.
#
# Por qué los intentos previos con JS fallaron:
#   - st.markdown sanitiza scripts → no se ejecutan
#   - st.components.v1.html crea iframe sandbox → no puede
#     modificar elementos del documento padre por cross-origin
#
# Solución correcta (sin JS):
#   Streamlit 1.31+ expone st.context.timezone, que devuelve un
#   string IANA (tipo "America/Bogota") con la timezone DETECTADA
#   AUTOMÁTICAMENTE del browser del usuario via headers HTTP.
#
#   En _landing.py leemos st.context.timezone y lo pasamos a
#   get_personalized_greeting. La función usa zoneinfo (stdlib
#   Python 3.9+) para convertir UTC del server a la zona del
#   usuario, y formatea el reloj en 12h con am/pm minúsculas.
#
# Validado:
#   - Compila Python OK
#   - Test Bogotá (UTC-5): "4:43 pm" + "Buenas tardes" + "Turno tarde"
#   - Test Tokyo (UTC+9): "6:43 am" (correctamente adelantado 14h
#     respecto a Bogotá)
#
# Beneficios sobre la solución JS:
#   - Sin scripts → sin sanitización → sin riesgo de roturas
#   - Sin iframes → sin cross-origin issues
#   - Calculado al cargar la página (no necesita refresh cada 30s)
#   - Si el usuario cambia de zona, próxima carga refleja la
#     nueva zona automáticamente
#
# Limitación aceptable:
#   El reloj NO se actualiza en tiempo real (no hace tick cada
#   minuto). Solo refleja la hora al momento de cargar la página.
#   Si el usuario está en Reports 30 minutos sin recargar, el
#   reloj del Home sigue mostrando la hora de hace 30 min. Para
#   ver hora actual: F5. Trade-off aceptable.
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🎯 HOTFIX v3.5.2 → MAIN  (reloj 12h auto-TZ sin JS)"
echo "================================================================"
echo ""
echo "Solución final del reloj: server-side via st.context.timezone."
echo "Sin JavaScript, sin iframes, sin riesgo de bugs futuros."
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

echo "▶ 1/7  Commit del fix en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged
for v in v3_4_2 v3_4_3 v3_4_4 v3_4_5 v3_4_6 v3_4_7 v3_4_8 v3_4_9 v3_5_0 v3_5_1; do
    git checkout HEAD -- "_release_${v}_hotfix.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_ux.sh" 2>/dev/null || true
done

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/home_metrics.py pages/_landing.py _release_v3_5_2_hotfix.sh
    git commit -m "hotfix(v3.5.2): reloj 12h auto-zona-horaria via st.context.timezone

Solución FINAL del reloj. Sin JavaScript. Todo server-side Python.

Por qué los intentos previos con JS fallaron:
- v3.4.7: <script> dentro de st.markdown → sanitizado
- v3.5.0: st.markdown SEPARADO → seguía sanitizado
- v3.5.1: st.components.v1.html() → iframe sandbox impide
  acceder al document padre por cross-origin

Solución correcta (sin JS):
Streamlit 1.31+ expone st.context.timezone que devuelve un
string IANA (ej. 'America/Bogota') con la timezone detectada
AUTOMÁTICAMENTE del browser via headers HTTP.

Cambios:
- core/home_metrics.py: get_personalized_greeting acepta
  parámetro tz_name. Si lo recibe, usa zoneinfo.ZoneInfo(tz_name)
  para convertir datetime.now(UTC) a la zona del usuario.
  Formato del reloj cambia a 12h con am/pm en minúsculas
  ('4:40 pm' en lugar de '16:40').
- pages/_landing.py: lee st.context.timezone y lo pasa a
  get_personalized_greeting. Eliminado todo el código JS muerto
  del intento de reloj live (DEAD_CODE_PRESERVED block, ~100
  líneas borradas).

Validado:
- Bogotá (UTC-5): '4:43 pm' + 'Buenas tardes' + 'Turno tarde' ✓
- Tokyo (UTC+9): '6:43 am' ✓
- Python compila

Limitación aceptada:
El reloj no actualiza en tiempo real (no tick cada min). Solo
refleja la hora al cargar la página. F5 para ver hora actual.
Trade-off vale por la simplicidad y robustez." || echo "  (nada nuevo para commitear)"
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
MERGE_MSG="hotfix(v3.5.2): merge dev -> main · reloj 12h auto-TZ server-side

Cierre definitivo de la saga del reloj. Sin JS. Server-side
con st.context.timezone que detecta la TZ del browser.

Validado: Bogotá '4:43 pm', Tokyo '6:43 am'.

Limitación: no live tick (refresca al recargar). Aceptable."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.5.2..."
TAG_EXISTS=$(git tag -l "v3.5.2")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.5.2 ya existe. Saltando creación."
else
    git tag -a v3.5.2 -m "Hotfix v3.5.2 — Reloj 12h auto-TZ via st.context.timezone (sin JS)"
    echo "  ✓ Tag v3.5.2 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.5.2 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ HOTFIX v3.5.2 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo "     Manage app → Reboot app después del deploy."
echo ""
echo " 👁  Validación esperada:"
echo "    - Reloj muestra '4:40 pm' (o tu hora actual real en Bogotá)"
echo "    - Saludo: 'Buenas tardes' (correcto para esa hora)"
echo "    - Turno: '☀️ Turno tarde'"
echo "    - Si recargás la página después, el reloj se actualiza"
echo ""
echo "================================================================"
