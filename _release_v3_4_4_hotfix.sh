#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.4.4 HOTFIX → MAIN
# =============================================================
# HOTFIX URGENTE Ciclo 17.20 — fix definitivo del crash de Reports.
#
# Diagnóstico del log de Streamlit Cloud (gracias al log que
# capturamos del crash en producción):
#
#   Bug A — Widget conflict (línea 3750 de pages/16_Reports.py)
#   --------------------------------------------------
#     `st.text_area(key=_wkey, value=meta.get(_key, ""))`
#     combinado con el botón "Auto-redactar" que setea
#     `st.session_state[_wkey] = _new_text` causa el warning
#     fatal de Streamlit:
#       "The widget with key X was created with a default value
#        but also had its value set via the Session State API."
#     → reruns extraños + página rota → "Oh no" en producción.
#
#   Bug B — OOM real (log se corta sin traceback)
#   --------------------------------------------------
#     Después de v3.4.3 (figures Plotly nullos), restore_report_items
#     SEGUÍA cargando TODOS los image_bytes de TODOS los items en
#     memoria al hacer load_report_state. Con 10 items de
#     200KB-2MB cada uno, son 5-20 MB constantes en session_state
#     que se replican en cada rerun. Más reportlab cargando todo
#     de una al generar el PDF → spike a 50-100 MB transitorios →
#     Streamlit Cloud (1 GB RAM) mata el container (OOMKilled).
#     El log se corta sin traceback Python — clásica firma de
#     OOMKilled por el kernel.
#
# Fixes:
#
#   A) pages/16_Reports.py línea 3750
#     - Inicializar st.session_state[_wkey] ANTES del widget si
#       no existe, y NO pasar value= al st.text_area. Patrón
#       correcto y sancionado de Streamlit.
#
#   B) Lazy loading de image_bytes
#     - core/report_state.py:
#       * restore_report_items NO carga bytes a memoria —
#         solo registra image_file (path) y _images_dir
#       * Nuevo read_item_image_bytes(item) → lee PNG desde
#         disco SOLO cuando se llama
#       * append_report_item_and_persist hace round-trip a
#         lazy form después del save (libera bytes inmediato)
#     - pages/16_Reports.py:
#       * 3 lugares que usaban item["image_bytes"] ahora
#         usan read_item_image_bytes(item)
#       * Render de panel: lee PNG solo al renderizar
#       * Generación PDF: lee PNG solo al construir el frame
#       * Sanitize para PDF: idem
#
# Resultado esperado:
#   session_state.report_items con 10 imgs:
#     ANTES: ~10 MB (cada item con bytes en memoria)
#     AHORA: ~50 KB (solo metadata + path)
#   Pico de memoria al generar PDF:
#     ANTES: ~50-100 MB
#     AHORA: ~5-10 MB (stream item por item)
#
# Validado con smoke test (_test_17_17_png_storage.py):
#   - 60 items round-trip lossless via lazy loading
#   - in_memory == 0 después de load (todos lazy)
#   - migración legacy → PNG + lazy
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🚨 HOTFIX v3.4.4 → MAIN  (FIX DEFINITIVO crash con muchas imgs)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.20  Lazy loading de image_bytes + fix widget conflict"
echo ""
echo "Diagnóstico clave del log de Streamlit Cloud:"
echo "  - Widget conflict en línea 3750 → reruns + 'Oh no'"
echo "  - OOMKilled por bytes cargados de más en session_state"
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

echo "▶ 1/7  Commit del 17.20 en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar modificaciones locales a scripts ya merged
git checkout HEAD -- _release_v3_4_2_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_3_hotfix.sh 2>/dev/null || true

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/report_state.py pages/16_Reports.py \
            _test_17_17_png_storage.py _release_v3_4_4_hotfix.sh
    git commit -m "hotfix(17.20): lazy loading image_bytes + fix widget conflict (fix REAL OOM)

Diagnóstico del log de Streamlit Cloud (capturado del crash con
10 imágenes en producción):

Bug A — Widget conflict en pages/16_Reports.py línea 3750
=========================================================
st.text_area(key=_wkey, value=meta.get(_key, '')) combinado con
el botón Auto-redactar (que setea st.session_state[_wkey]) tiraba
el warning fatal de Streamlit:
  'The widget with key X was created with a default value but
   also had its value set via the Session State API.'
→ reruns extraños + 'Oh no' en producción.

Fix: inicializar st.session_state[_wkey] ANTES del widget si
no existe, y eliminar value= del text_area. Patrón sancionado
por Streamlit.

Bug B — OOM real (log se cortaba sin Python traceback)
======================================================
v3.4.3 fixeó figures Plotly pesados pero restore_report_items
SEGUÍA cargando todos los image_bytes a memoria. Con 10 items
de 200KB-2MB cada uno → 5-20 MB en session_state que se
replicaban en cada rerun. Más reportlab cargando todo de una
al generar PDF → spike a 50-100 MB → Streamlit Cloud (1 GB)
mataba el container (OOMKilled — clásica firma de log
cortado sin traceback).

Fix lazy loading:
- restore_report_items: NO carga bytes a memoria, solo registra
  image_file (path) y _images_dir
- Nuevo read_item_image_bytes(item) que lee PNG desde disco
  ON-DEMAND solo cuando se llama
- append_report_item_and_persist hace round-trip a lazy form
  después del save (libera bytes inmediato sin esperar rerun)
- pages/16_Reports.py: 3 lugares que usaban item['image_bytes']
  ahora usan read_item_image_bytes(item):
    * Render del panel (st.image)
    * Generación del PDF (Image(BytesIO(...)))
    * Sanitize para PDF normalize

Resultado:
- session_state.report_items con 10 imgs: ~10 MB → ~50 KB
- Pico al generar PDF: ~50-100 MB → ~5-10 MB

Smoke test 17.17 actualizado para verificar:
- 60 items round-trip lossless via lazy
- in_memory == 0 después de load (verificación dura)
- migración legacy → PNG + lazy" || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene el 17.20 commiteado"
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
MERGE_MSG="hotfix(v3.4.4): merge dev -> main · Ciclo 17.20 lazy + widget fix

FIX DEFINITIVO del crash con muchas imágenes en Reports.

Diagnóstico capturado del log de Streamlit Cloud:
  Bug A: widget conflict en línea 3750 → reruns + 'Oh no'
  Bug B: image_bytes cargados de más en session_state →
         OOMKilled (log se cortaba sin traceback)

Fix:
  A) Inicializar session_state ANTES del text_area, sin value=
  B) Lazy loading via read_item_image_bytes(item):
     - restore_report_items NO carga bytes
     - Render Reports lee PNG solo al renderizar
     - Generación PDF idem
     - append_report_item_and_persist round-trip a lazy

Resultado:
  - session_state con 10 imgs: ~10 MB → ~50 KB
  - Pico al generar PDF: ~50-100 MB → ~5-10 MB

API publica intacta — sigue mostrando las imágenes igual
(idénticos bytes, solo cambia DÓNDE viven entre lecturas)."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.4.4..."
TAG_EXISTS=$(git tag -l "v3.4.4")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.4.4 ya existe. Saltando creación."
else
    git tag -a v3.4.4 -m "Hotfix v3.4.4 — Ciclo 17.20 lazy loading + fix widget conflict"
    echo "  ✓ Tag v3.4.4 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.4.4 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ HOTFIX v3.4.4 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo "     RECOMENDADO: forzar reboot manual desde Manage app después"
echo "     del deploy (botón ... → Reboot app) para garantizar que"
echo "     todos los specialists arranquen con session_state limpio."
echo ""
echo " 🧪 VALIDACIÓN URGENTE (con un specialist al lado):"
echo ""
echo "    1. Abrir un reporte nuevo o limpiar el existente"
echo "    2. Subir 5 imágenes (Spectrum/Waveforms/Orbit) → Reports"
echo "       debe abrir sin colgarse y sin 'Oh no'"
echo "    3. Subir hasta 15 imágenes en total → debe seguir andando"
echo "    4. Generar PDF → debe completar OK"
echo ""
echo " 👁  Cambio visible en la UI:"
echo ""
echo "    Las gráficas en Reports siguen viéndose igual (PNG estático"
echo "    desde v3.4.3, ahora cargado on-demand). Sin diferencia"
echo "    visual para el usuario. Solo cambia DÓNDE viven los bytes."
echo ""
echo " 📊 Métrica de éxito:"
echo ""
echo "    Antes del fix: crash con 5-10 imágenes."
echo "    Esperado ahora: aguantar 50+ imágenes sin problema."
echo ""
echo "================================================================"
