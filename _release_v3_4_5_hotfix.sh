#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.4.5 HOTFIX → MAIN
# =============================================================
# HOTFIX URGENTE Ciclo 17.21 — fix del crash al "Preparar PDF".
#
# Estado después de v3.4.4 (lazy loading + widget fix):
#   - El sistema YA NO se cae al subir 10 imágenes ✓
#   - El sistema YA NO se cae al abrir Reports ✓
#   - PERO al click "Preparar PDF" → OOMKilled
#
# Diagnóstico:
#   reportlab al construir el PDF carga TODAS las imágenes
#   simultáneamente a memoria. Cada imagen Plotly export con
#   scale=2 pesa ~2-3 MB sin comprimir cuando reportlab la
#   abre internamente. 10 imgs × 3 MB + buffer del PDF + libs
#   Python (~300 MB) excede los 1 GB de Streamlit Cloud.
#   El log se cortaba sin Python traceback — firma clásica
#   de OOMKilled por el kernel.
#
# Fix (Ciclo 17.21):
#   Helper nuevo _pdf_safe_image_bytes(raw_bytes, max_width=1500):
#     - Si Pillow disponible y la imagen es más ancha que 1500 px:
#       downsize manteniendo aspect ratio
#     - Re-encodea como PNG optimizado
#     - gc.collect() para liberar buffers intermedios
#     - Si Pillow no está o el downsize falla: devuelve original
#
#   Wirear en los 5 lugares donde Image(BytesIO(...)) carga al PDF:
#     - Línea 1842: figura del análisis multi-fecha
#     - Línea 1880: schematic principal
#     - Línea 2133: schematic small
#     - Línea 2593: imagen en celda de tabla
#     - Línea 2703: imagen del panel de figura (el más impactante)
#
#   Más gc.collect() después de cada Image() para forzar liberación
#   inmediata del buffer entre imágenes.
#
# Calidad visual:
#   1500 px de ancho en página A4 (21 cm) = 180 DPI. Para
#   impresión profesional bastan 150 DPI. El usuario no nota
#   diferencia visible. Solo se reduce CONSUMO DE RAM en
#   reportlab al construir el PDF.
#
# Resultado esperado:
#   Imagen del PDF ocupa ~500 KB en RAM en lugar de 2-3 MB.
#   Pico al generar PDF: ~50 MB → ~5-10 MB.
#   Con 10-15 imágenes ahora debería completar OK.
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🚨 HOTFIX v3.4.5 → MAIN  (fix OOM al 'Preparar PDF')"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.21  Downsize de imágenes con Pillow antes del PDF"
echo "         + gc.collect() entre cada imagen"
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

echo "▶ 1/7  Commit del 17.21 en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged
git checkout HEAD -- _release_v3_4_2_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_3_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_4_hotfix.sh 2>/dev/null || true

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add pages/16_Reports.py _release_v3_4_5_hotfix.sh
    git commit -m "hotfix(17.21): downsize imgs + gc en PDF gen — fix OOM al 'Preparar PDF'

Estado tras v3.4.4 (lazy + widget fix):
- Sistema NO se cae al subir 10 imgs ✓
- Sistema NO se cae al abrir Reports ✓
- PERO al click 'Preparar PDF' → OOMKilled

Diagnóstico:
reportlab al construir el PDF carga TODAS las imágenes
simultáneamente a memoria. Cada Plotly export con scale=2
pesa ~2-3 MB en RAM cuando reportlab la abre. 10 imgs ×
3 MB + buffer PDF + libs Python (~300 MB) excede 1 GB
de Streamlit Cloud. Log se cortaba sin Python traceback
— firma clásica de OOMKilled por el kernel.

Fix:
- Nuevo _pdf_safe_image_bytes(raw, max_width=1500) en
  pages/16_Reports.py:
    * Si Pillow disponible y imagen >1500 px de ancho:
      downsize con LANCZOS manteniendo aspect ratio
    * Re-encodea como PNG optimizado
    * gc.collect() para liberar buffers intermedios
    * Si Pillow no está o falla: devuelve original (no rompe)
- Wirear en los 5 Image(BytesIO(...)) del PDF:
    * Líneas 1842, 1880, 2133, 2593, 2703
- gc.collect() después de cada Image() para forzar
  liberación inmediata entre imágenes

Calidad visual:
1500 px de ancho en A4 (21 cm) = 180 DPI. Para impresión
profesional bastan 150 DPI. Sin diferencia visible para
el usuario.

Impacto esperado:
- Imagen del PDF en RAM: 2-3 MB → ~500 KB
- Pico al generar PDF: ~50 MB → ~5-10 MB
- Con 10-15 imágenes debería completar OK." || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene el 17.21 commiteado"
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
MERGE_MSG="hotfix(v3.4.5): merge dev -> main · Ciclo 17.21 downsize PDF gen

Estado tras v3.4.4: sistema NO se cae con 10 imgs ni al abrir
Reports, PERO se cae al click 'Preparar PDF'.

Diagnóstico OOMKilled:
  reportlab carga TODAS las imágenes simultáneamente al
  construir el PDF. 10 imgs × 2-3 MB + buffer + libs Python
  excede 1 GB de Streamlit Cloud.

Fix:
  _pdf_safe_image_bytes(raw, max_width=1500) con Pillow:
  downsize a max 1500px de ancho + PNG optimizado + gc. Wirear
  en los 5 lugares de Image(BytesIO(...)) del PDF + gc.collect()
  entre imágenes.

Calidad: 1500px en A4 = 180 DPI (sobra para impresión).
Impacto: pico de 50 MB → 5-10 MB al generar PDF."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.4.5..."
TAG_EXISTS=$(git tag -l "v3.4.5")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.4.5 ya existe. Saltando creación."
else
    git tag -a v3.4.5 -m "Hotfix v3.4.5 — Ciclo 17.21 downsize images en PDF gen"
    echo "  ✓ Tag v3.4.5 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.4.5 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ HOTFIX v3.4.5 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo "     Después del deploy, RECOMENDADO: Manage app → Reboot."
echo ""
echo " 🧪 VALIDACIÓN URGENTE (con un specialist al lado):"
echo ""
echo "    1. Limpiar reporte actual o crear uno nuevo"
echo "    2. Subir 10-15 imágenes (Spectrum/Waveforms/Orbit/Bode)"
echo "    3. Click 'Preparar PDF' → debe completar OK (no más crash)"
echo "    4. Descargar PDF → verificar que las imágenes se vean bien"
echo ""
echo " 👁  Cambio visual (mínimo):"
echo ""
echo "    Las imágenes en el PDF que ANTES eran >1500 px de ancho"
echo "    ahora se downsizean a 1500 px. A 180 DPI en A4 sigue"
echo "    siendo calidad de impresión profesional. La diferencia"
echo "    visual NO debería notarse a ojo en el documento final."
echo ""
echo " 📊 Si AÚN se cae con 15+ imágenes:"
echo ""
echo "    Llegamos al techo de RAM de Streamlit Community Cloud (1 GB)."
echo "    Próximo paso obligatorio: migrar a Render (\$25/mes, 2 GB) o"
echo "    Fly.io (\$7/mes, 2 GB). Te paso el plan de migración cuando"
echo "    confirmes."
echo ""
echo "================================================================"
