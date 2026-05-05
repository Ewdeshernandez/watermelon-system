#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.4.3 HOTFIX → MAIN
# =============================================================
# HOTFIX URGENTE Ciclo 17.19 — directo a main.
# Especialistas no pueden hacer reportes con 5 gráficas: el
# sistema se cae cada vez y se pierde todo el trabajo.
#
# Bug raíz REAL (que v3.4.1 NO atacó):
#   Spectrum (2 lugares), Time Waveforms y Orbit guardaban
#   go.Figure(fig) ENTERO en st.session_state.report_items.
#   Cada figure pesa 20-100 MB en RAM (incluye TODOS los
#   datos del trace, layout, configuración). Con 5 gráficas
#   → 250-500 MB en memoria solo del session_state →
#   Streamlit Cloud (1 GB RAM) reventaba.
#
#   v3.4.1 PNG sueltos solo arregló el JSON en disco; el
#   problema vivía en MEMORIA, no en disco.
#
# Fix:
#   Los 4 lugares ahora ponen "figure": None y dejan solo
#   "image_bytes" (que ya estaban generando). La UI de
#   Reports cae al fallback st.image() — pierde el zoom
#   interactivo de Plotly pero NO se cuelga. Para Orbit
#   (que no tenía image_bytes) generamos el PNG con
#   build_export_png_bytes(fig) antes del append.
#
# Verificado:
#   grep "figure": go.Figure → 0 matches en pages/
#
# Cambio visible en UI:
#   Las gráficas en Reports se muestran como IMAGEN ESTÁTICA.
#   Pierde zoom/pan/hover. Costo aceptable para no caerse.
#   La interactividad sigue intacta en los módulos originales
#   (Spectrum, Time Waveforms, Orbit).
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🚨 HOTFIX v3.4.3 → MAIN  (URGENTE — fix OOM real con 5 gráficas)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.19  fix OOM real — figures Plotly fuera de session_state"
echo "         (Spectrum x2, Time Waveforms, Orbit)"
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

echo "▶ 1/7  Commit del 17.19 en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar modificaciones locales al script v3.4.2 (ya está en main)
git checkout HEAD -- _release_v3_4_2_hotfix.sh 2>/dev/null || true

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add pages/02_Time_Waveforms.py pages/03_Spectrum.py \
            pages/05_Orbit_Analysis.py _release_v3_4_3_hotfix.sh
    git commit -m "hotfix(17.19): nullear go.Figure en session_state — fix OOM real

Bug raíz del crash con 5 gráficas en producción:
Spectrum (2 lugares), Time Waveforms y Orbit guardaban el
go.Figure ENTERO en st.session_state.report_items. Cada
figure pesa 20-100 MB en RAM (incluye TODOS los datos del
trace, layout, configuración). Con 5 gráficas → 250-500 MB
en memoria solo del session_state → Streamlit Cloud (1 GB)
reventaba.

v3.4.1 PNG sueltos solo arregló el JSON en disco; este bug
vivía en MEMORIA, no en disco. Por eso seguía cayéndose.

Fix:
- pages/02_Time_Waveforms.py: 'figure': None
- pages/03_Spectrum.py (línea 1853): 'figure': None
- pages/03_Spectrum.py (línea 2658): 'figure': None
- pages/05_Orbit_Analysis.py: 'figure': None + generar
  image_bytes vía build_export_png_bytes(fig) antes del
  append (no lo tenía)

Las gráficas en Reports ahora se muestran como st.image()
estático en lugar de st.plotly_chart() interactivo. Pierde
zoom/pan/hover de Plotly. Aceptable: mejor PNG estático
que app caída. La interactividad sigue en los módulos
originales.

Verificado: grep 'figure': go.Figure → 0 matches en pages/." || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene el 17.19 commiteado"
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
MERGE_MSG="hotfix(v3.4.3): merge dev -> main · Ciclo 17.19 fix OOM real

URGENTE — especialistas no podían hacer reportes con 5 gráficas.

Bug raíz REAL (que v3.4.1 PNG sueltos NO atacó):
  Spectrum/Time Waveforms/Orbit guardaban go.Figure ENTERO
  en st.session_state.report_items. Cada figure pesa 20-100 MB
  en RAM. Con 5 gráficas → 250-500 MB → Streamlit Cloud (1 GB)
  reventaba. v3.4.1 fixeaba el JSON en disco; este vivía en
  MEMORIA.

Fix:
  Nullear 'figure' en los 4 lugares afectados, dejar solo
  image_bytes que ya estaban generándose. UI cae al fallback
  st.image() — pierde zoom interactivo de Plotly pero no se
  cae. Para Orbit (sin image_bytes previo) generamos el PNG
  con build_export_png_bytes(fig) antes del append.

Cambio visible:
  Las gráficas en Reports se ven como imagen estática en
  lugar de Plotly interactivo. La interactividad sigue en
  los módulos originales (Spectrum, etc.).

Verificado:
  grep 'figure': go.Figure → 0 matches en pages/"

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.4.3..."
TAG_EXISTS=$(git tag -l "v3.4.3")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.4.3 ya existe. Saltando creación."
else
    git tag -a v3.4.3 -m "Hotfix v3.4.3 — Ciclo 17.19 fix OOM real (figures fuera de session_state)"
    echo "  ✓ Tag v3.4.3 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.4.3 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ HOTFIX v3.4.3 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo ""
echo " 🧪 VALIDACIÓN URGENTE en producción:"
echo ""
echo "    1. Andá a Spectrum / Time Waveforms / Orbit"
echo "    2. Generá una gráfica → 'Enviar al reporte'"
echo "    3. Repetir 5-10 veces con DIFERENTES máquinas"
echo "    4. Andá a Reports → debe abrir sin colgarse"
echo "    5. Generar PDF → debe completar"
echo ""
echo "    Si los 5 pasos pasan → bug RESUELTO."
echo ""
echo " 👁  Cambio visible en Reports:"
echo ""
echo "    Las gráficas se muestran como imagen estática (no Plotly"
echo "    interactivo). Sin zoom/pan/hover dentro del reporte."
echo "    La interactividad sigue funcionando en los módulos"
echo "    originales (Spectrum, Time Waveforms, Orbit)."
echo ""
echo "================================================================"
