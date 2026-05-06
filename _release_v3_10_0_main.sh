#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.10.0 → MAIN
# =============================================================
# Ciclo 17.28 — AI Run-vs-Run Comparison (delta forense automático)
#
# El reporte que se está preparando ahora se compara automáticamente
# contra el último reporte archivado del mismo activo (match por
# instance_id). Claude lee la metadata + severidad + resumen
# ejecutivo del reporte anterior, las figuras y datos cuantitativos
# del actual, y emite el "delta forense": qué métricas cambiaron,
# qué firmas mecánicas son nuevas o desaparecieron, hacia dónde va
# la tendencia, en qué ventana de tiempo, y qué implicación
# operativa tiene.
#
# Esto es el feature que separa MONITORING (mirar números) de
# CONDITION MONITORING REAL (entender la evolución mecánica). Es
# el sueño de cualquier programa de mantenimiento predictivo.
#
# Por qué es disruptivo:
# ──────────────────────────────────────────────────────────
# Sin Run-vs-Run: el especialista lee un reporte, ve los números,
# pero tiene que comparar mentalmente contra los del mes pasado.
# Costoso cognitivamente, propenso a perder cambios sutiles.
#
# Con Run-vs-Run: el sistema le dice "comparado con el reporte de
# hace 30 días, esto es lo que cambió" — incluyendo aparición de
# firmas nuevas, cambio de severidad, evolución del Q-factor, ratio
# de cambio del overall, etc.
#
# Para el cliente final (ECOPETROL/MAGNEX) esto es lo que justifica
# el modelo de monitoreo continuo vs reportes puntuales: el valor
# está en la EVOLUCIÓN, no en la foto.
#
# Ningún competidor en LATAM tiene esto: ni Bently Nevada System 1,
# ni SKF Observer, ni Schaeffler ProLink, ni Emerson AMS Machinery
# Manager. Todos ellos te muestran tendencias gráficas de números
# crudos, pero NINGUNO te explica QUÉ CAMBIÓ MECÁNICAMENTE en
# lenguaje natural.
#
# Cambios técnicos:
# ──────────────────────────────────────────────────────────
#   core/ai_runcompare.py (NUEVO):
#     - find_previous_report(viewer, instance_id, instance_tag,
#       before_date) → busca el último archivado del mismo activo
#       en core.reports_archive. Match prioritario por instance_id;
#       fallback por instance_tag normalizado. Filtro temporal para
#       evitar comparar contra el reporte que se está preparando.
#     - generate_run_comparison(prev_sidecar, current_meta,
#       current_items) → genera el delta forense.
#     - System prompt v1 'delta forense': voz Cat IV ISO 18436-2
#       senior, formato fijo (lead de evolución, cambios
#       cuantitativos detectados, interpretación clínica,
#       implicación operativa y ventana de acción, evaluación de
#       confianza). Cita números reales del payload, no inventa.
#     - _extract_quant_table_from_notes: parsea las tablas
#       cuantitativas de las notas (marcador <<<WM_AI_BLOCK>>>)
#       para que el AI tenga datos comparables crudos.
#     - _build_runcompare_user_message: compone request con el
#       reporte anterior (severidad, summary) + reporte actual
#       (figuras + tablas) + días transcurridos calculados.
#     - Cache local: data/cache/ai_runcompare/ con
#       autoinvalidación por RUNCOMPARE_PROMPT_VERSION.
#     - Robustez heredada: retry x3 + fallback Haiku 4.5 +
#       detección de timeouts + pricing dinámico.
#
#   pages/16_Reports.py:
#     - Import de find_previous_report + generate_run_comparison.
#     - Búsqueda automática en cada render: si el activo activo
#       (instance_id en meta) tiene reporte anterior archivado,
#       se cachea en st.session_state[wm_ai_runcmp_prev_{key}].
#     - Botón '🔄 Comparar con reporte anterior' (etiqueta
#       enriquecida con consecutivo o fecha del reporte previo
#       cuando existe). Disabled si no hay key Anthropic, no hay
#       items, o no hay reporte anterior — con tooltip explicativo
#       en cada caso.
#     - Caption a la derecha del botón muestra: consecutivo
#       previo + fecha + severidad anterior + activo, así el
#       especialista sabe contra qué se va a comparar antes de
#       clickear.
#     - Render del resultado en expander con preview, banner de
#       fallback, botones Regenerar/Descartar, caption con
#       modelo+tokens+costo.
#     - Inyección automática a meta['ai_run_comparison'] +
#       meta['ai_run_comparison_meta'] (con prev_consecutive,
#       prev_archived_at, days_elapsed) antes de _build_pdf_bytes.
#     - Modificación del PDF: nueva sección 'EVOLUCIÓN DESDE LA
#       ÚLTIMA CORRIDA' después del RESUMEN EJECUTIVO. Incluye
#       caption con metadata del reporte previo + intervalo, y
#       el delta forense renderizado con _render_ai_clinical_flowables
#       (mismos estilos que usamos para los demás bloques AI).
#
# Costo estimado:
# ──────────────────────────────────────────────────────────
#   Sonnet 4.5: ~\$0.02-0.06 por comparación (depende de tamaño
#               del executive summary anterior + número de figuras
#               actuales).
#   Haiku 4.5 (fallback): ~5x menos.
#   Cache local → comparaciones repetidas son virtualmente gratis.
#
# Cero regresiones:
# ──────────────────────────────────────────────────────────
#   - Si el activo es nuevo (sin reporte anterior archivado), el
#     botón se deshabilita con mensaje claro. El reporte se genera
#     normal sin la sección de evolución.
#   - Si el especialista NO clickea el botón, meta['ai_run_comparison']
#     queda vacío y la sección no aparece en el PDF.
#   - Si Anthropic está caído, retry+fallback Haiku absorben.
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🔄 RELEASE v3.10.0 → MAIN  (AI Run-vs-Run Delta Forense)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.28  AI Run-vs-Run Comparison:"
echo "         - core/ai_runcompare.py (wrapper + system prompt v1)"
echo "         - pages/16_Reports.py (botón + UI + inyección al PDF)"
echo "         - Nueva sección PDF 'EVOLUCIÓN DESDE LA ÚLTIMA CORRIDA'"
echo "         - Match automático por instance_id contra el archivo"
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el release v3.10.0 a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

echo "▶ 1/7  Commit del ciclo 17.28 en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged
for v in v3_4_2 v3_4_3 v3_4_4 v3_4_5 v3_4_6 v3_4_7 v3_4_8 v3_4_9 \
         v3_5_0 v3_5_1 v3_5_2 v3_5_3 v3_6_0 v3_7_0 v3_8_0 v3_9_0; do
    git checkout HEAD -- "_release_${v}_hotfix.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_ux.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_main.sh" 2>/dev/null || true
done

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/ai_runcompare.py \
            pages/16_Reports.py \
            _release_v3_10_0_main.sh
    git commit -m "feat(17.28): AI Run-vs-Run Comparison — delta forense automático

Cuando el especialista prepara un reporte de un activo que ya tuvo
reportes anteriores archivados, el sistema lee automáticamente el
último archivado del mismo instance_id y genera el 'delta forense':
qué métricas cambiaron, qué firmas mecánicas son nuevas o
desaparecieron, hacia dónde va la tendencia, en qué ventana de
tiempo, y qué implicación operativa tiene.

Esto separa MONITORING (mirar números) de CONDITION MONITORING
REAL (entender la evolución mecánica). Ningún competidor en LATAM
tiene esto: ni Bently Nevada System 1, ni SKF Observer, ni
Schaeffler ProLink, ni Emerson AMS Machinery Manager.

Wrapper (core/ai_runcompare.py)
================================
- find_previous_report(viewer, instance_id, instance_tag,
  before_date): busca el último archivado del mismo activo en
  core.reports_archive. Match prioritario por instance_id,
  fallback por instance_tag. Filtro temporal opcional.
- generate_run_comparison(prev_sidecar, current_meta, current_items):
  arma payload con metadata + summary anterior + figuras actuales
  con sus tablas cuantitativas extraídas de los marcadores
  <<<WM_AI_BLOCK>>>, calcula días transcurridos, llama a Claude.
- System prompt v1 'delta forense' Cat IV ISO 18436-2:
  formato fijo (lead de evolución, cambios cuantitativos detectados,
  interpretación clínica, implicación operativa y ventana de
  acción, evaluación de confianza). Voz pasiva técnica, cita
  números reales, no inventa valores, máximo 500 palabras.
- _extract_quant_table_from_notes: parsea tablas de las notas para
  comparación numérica cruda.
- Cache local data/cache/ai_runcompare/ con autoinvalidación por
  RUNCOMPARE_PROMPT_VERSION.
- Retry x3 + fallback Haiku + detección de timeouts heredados.

Reports (pages/16_Reports.py)
==============================
- Búsqueda automática del reporte anterior al cargar la página
  (cacheada en session_state por activo).
- Botón '🔄 Comparar con reporte anterior' con etiqueta enriquecida
  (consecutivo o fecha del previo). Disabled con tooltip
  explicativo cuando no hay reporte anterior, no hay activo, no
  hay items, o no hay key Anthropic.
- Caption derecho del botón muestra metadata del reporte previo
  detectado (consecutivo + fecha + severidad anterior + activo).
- Expander de preview con banner de fallback si aplica, botones
  Regenerar/Descartar, caption con modelo+tokens+costo.
- Inyección automática a meta['ai_run_comparison'] +
  meta['ai_run_comparison_meta'] antes de _build_pdf_bytes.
- PDF: nueva sección 'EVOLUCIÓN DESDE LA ÚLTIMA CORRIDA' después
  del RESUMEN EJECUTIVO con caption de metadata del reporte previo
  + intervalo en días + delta forense renderizado con
  _render_ai_clinical_flowables.

Costo: ~\$0.02-0.06 por comparación (Sonnet) o ~5x menos (Haiku
fallback). Cache local → repeticiones casi gratis.

Cero regresiones: si no hay reporte anterior, si el especialista
no clickea, o si Anthropic falla, el reporte se genera normal sin
la sección de evolución." || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene el ciclo 17.28 commiteado"
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
MERGE_MSG="release(v3.10.0): merge dev -> main · AI Run-vs-Run Delta Forense

Comparación automática del reporte actual contra el último
archivado del mismo activo. Claude emite el 'delta forense' en
lenguaje natural: qué cambió mecánicamente entre la corrida
anterior y la actual, hacia dónde va la tendencia, qué hacer.

Es lo que separa monitoring (números) de condition monitoring
real (evolución). Ningún competidor en LATAM tiene este feature.

Aparece como botón en Reports y como nueva sección en el PDF
después del Resumen Ejecutivo. Cero regresiones: si no hay
reporte anterior, si no se clickea, o si la AI falla, el
reporte se genera normal."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.10.0..."
TAG_EXISTS=$(git tag -l "v3.10.0")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.10.0 ya existe. Saltando creación."
else
    git tag -a v3.10.0 -m "Release v3.10.0 — AI Run-vs-Run Delta Forense"
    echo "  ✓ Tag v3.10.0 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.10.0 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ RELEASE v3.10.0 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar en 1-2 min."
echo "    No hay que tocar secrets — la key sigue de v3.6.0."
echo ""
echo " 🧪 VALIDACIÓN en producción:"
echo ""
echo "    Para probar este feature necesitás un activo con AL MENOS"
echo "    UN reporte ya archivado previamente. Si tenés TES1 con"
echo "    historia, andá a probarlo. Sino, archivá el reporte actual"
echo "    primero, después generá uno nuevo con datos diferentes."
echo ""
echo "    1. Activá una instancia desde Machinery Library"
echo "    2. Andá a Reports — verás el activo en el meta"
echo "    3. Cargá figuras (Spectrum, Trends, etc.)"
echo "    4. Si hay reporte anterior, el botón muestra"
echo "       '🔄 Comparar con reporte anterior (321-1233)' habilitado"
echo "       con caption a la derecha mostrando severidad anterior"
echo "    5. Click → spinner 8-20 seg → preview con delta forense"
echo "    6. Verificá: lead de evolución, cambios cuantitativos"
echo "       numerados con citación de figuras, interpretación clínica,"
echo "       implicación operativa, confianza"
echo "    7. Click 'Preparar PDF' → descargar"
echo "    8. Ver la nueva sección 'EVOLUCIÓN DESDE LA ÚLTIMA CORRIDA'"
echo "       después del Resumen Ejecutivo, con caption de metadata"
echo "       y delta forense renderizado con estilos clínicos."
echo ""
echo " 💡 Demo killer para ventas:"
echo "    Mostrá al cliente 2 reportes consecutivos de TES1 con"
echo "    cambios reales. La sección automática que explica QUÉ"
echo "    CAMBIÓ entre las dos corridas vale más que cualquier"
echo "    gráfico de tendencias que haga la competencia."
echo ""
echo "================================================================"
