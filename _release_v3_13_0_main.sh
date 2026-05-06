#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.13.0 → MAIN
# =============================================================
# Ciclo 17.34 — Pattern Memory (memoria institucional con AI)
#
# Cuando se prepara un reporte nuevo, el sistema busca
# automáticamente en el archivo histórico patrones mecánicos
# similares en CUALQUIER activo del cliente, no solo el mismo.
# El archivo deja de ser un repositorio pasivo y se convierte
# en cerebro colectivo: cada reporte que se archiva suma valor
# a TODOS los próximos análisis.
#
# Por qué es disruptivo:
# ──────────────────────────────────────────────────────
# Bently Nevada System 1, SKF Observer, Schaeffler ProLink,
# Emerson AMS — NINGUNO puede tener esto, porque NO TIENEN TUS
# DATOS. Ese es el moat competitivo definitivo de SIGA: cuanto
# más reportes se archiven, más inteligente se hace el sistema.
# A los 6 meses tu archivo es un activo intelectual irreplicable.
#
# Demo killer para ventas:
# ──────────────────────────────────────────────────────
# "Mostrame qué casos parecidos a este se vieron antes en TU
# archivo." → Watermelon lee el reporte actual, busca en los
# históricos del cliente, y devuelve 5 casos similares con score
# de similitud, explicación de POR QUÉ son similares, y la
# resolución que se aplicó en cada caso histórico (si está
# documentada). Cierra ventas con C-level porque demuestra valor
# acumulativo del programa de monitoreo.
#
# Cambios técnicos:
# ──────────────────────────────────────────────────────
#   core/ai_patterns.py (NUEVO):
#     - find_similar_patterns(current_meta, current_items, viewer)
#       → busca matches en el archivo histórico accesible.
#     - System prompt v1 'matcher institucional Cat IV': identifica
#       similitud MECÁNICA (mismas firmas, frecuencias, patrones
#       de evolución) — no por cliente, no por fecha, no por
#       activo. Solo patrón mecánico.
#     - compute_fingerprint extrae huella mecánica de un sidecar
#       (asset, severidad, executive_summary).
#     - compute_fingerprint_for_current arma huella del reporte
#       en preparación con tablas cuantitativas extraídas de los
#       marcadores WM_AI_BLOCK + narrativa Cat IV/AI de los items.
#     - Salida JSON estricta del modelo, parser tolerante con
#       enriquecimiento automático desde sidecars (filtra
#       alucinaciones de archive_ids inexistentes).
#     - Bandas de similarity score con colores: 85+ rojo intenso,
#       70+ naranja, 55+ amarillo, 40+ verde-lima, <40 marginal.
#     - Cache local TTL 14 días con autoinvalidación por
#       PATTERNS_PROMPT_VERSION.
#     - Reusa retry x3 + fallback Haiku 4.5 + detección de
#       timeouts heredados de ai_diagnostic.
#
#   pages/16_Reports.py:
#     - Import de find_similar_patterns.
#     - Botón 'Buscar patrones similares en archivo histórico'
#       después de RUL. Disabled si no hay key Anthropic o no hay
#       items.
#     - Caption explicativo del feature al lado del botón.
#     - Spinner 10-25 seg durante la búsqueda.
#     - Render de cada match en expander: badge coloreado de
#       similarity score + band, identificadores del caso
#       histórico (consecutivo + fecha + activo + severidad),
#       'Por qué son similares' (rationale del AI), 'Resolución
#       del caso histórico' (si está documentada), 'Aplicabilidad
#       al caso actual', botón de descarga directa del PDF citado.
#     - Botones Regenerar / Descartar + caption con metadata
#       técnica (modelo, tokens, costo).
#     - Inyección a meta['ai_patterns_matches'] +
#       meta['ai_patterns_meta'] antes de _build_pdf_bytes.
#     - PDF: nueva sección 'PATRONES RECONOCIDOS EN ARCHIVO
#       HISTÓRICO' después de RUL. Cada match con header
#       coloreado por band + rationale + resolución +
#       aplicabilidad. Renderizado con estilos clínicos nativos.
#
# Costo estimado:
# ──────────────────────────────────────────────────────
#   ~\$0.04-0.10 por búsqueda (Sonnet) o ~5x menos (Haiku
#   fallback). Cache TTL 14 días → repeticiones gratis. Para
#   un programa de monitoreo activo con 10 búsquedas/mes:
#   ~\$0.50/mes adicionales.
#
# Cero regresiones:
# ──────────────────────────────────────────────────────
#   - Si el archivo histórico está vacío, mensaje informativo
#     y el reporte se genera normal sin la sección.
#   - Si no se encuentran patrones >40% similitud, mensaje claro
#     ("patrón sin antecedentes") y el reporte se genera normal.
#   - Si Anthropic está caído, retry+fallback Haiku absorben.
#   - Si no se clickea el botón, la sección no aparece en el PDF.
#   - Es feature opcional puro; no afecta otros flujos.
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " RELEASE v3.13.0 → MAIN  (Pattern Memory — memoria institucional)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.34  Pattern Memory con AI:"
echo "         - core/ai_patterns.py (matcher Cat IV con Claude)"
echo "         - Botón 'Buscar patrones similares' en Reports"
echo "         - Sección PDF 'PATRONES RECONOCIDOS EN ARCHIVO"
echo "           HISTÓRICO' con score, rationale, resolución del"
echo "           caso histórico, aplicabilidad al caso actual"
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el release v3.13.0 a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

echo "▶ 1/7  Commit del ciclo 17.34 en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged
for v in v3_4_2 v3_4_3 v3_4_4 v3_4_5 v3_4_6 v3_4_7 v3_4_8 v3_4_9 \
         v3_5_0 v3_5_1 v3_5_2 v3_5_3 v3_6_0 v3_7_0 v3_8_0 v3_9_0 \
         v3_10_0 v3_11_0 v3_12_0; do
    git checkout HEAD -- "_release_${v}_hotfix.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_ux.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_main.sh" 2>/dev/null || true
done

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/ai_patterns.py \
            pages/16_Reports.py \
            _release_v3_13_0_main.sh
    git commit -m "feat(17.34): Pattern Memory — memoria institucional con AI

Cuando se prepara un reporte nuevo, el sistema busca automáticamente
en el archivo histórico patrones mecánicos similares en CUALQUIER
activo del cliente, no solo el mismo. El archivo deja de ser un
repositorio pasivo de PDFs y se convierte en cerebro colectivo:
cada reporte que se archiva suma valor a todos los próximos
análisis.

Wrapper (core/ai_patterns.py)
==============================
- find_similar_patterns(current_meta, current_items, viewer):
  busca matches en el archivo accesible. Pre-corta al N más
  reciente (default 60) para presupuesto de tokens.
- System prompt v1 'matcher institucional Cat IV': identifica
  similitud MECÁNICA — mismas firmas, frecuencias, patrones de
  evolución. NO por cliente, fecha o activo.
- compute_fingerprint extrae huella de cada sidecar (asset,
  severidad, executive_summary).
- compute_fingerprint_for_current arma huella del reporte en
  preparación con tablas cuantitativas + narrativa de items.
- Salida JSON estricta del modelo, parser tolerante con
  enriquecimiento desde sidecars y filtro de alucinaciones.
- Bandas de score con colores (85+ rojo, 70+ naranja, etc.).
- Cache TTL 14 días con autoinvalidación por
  PATTERNS_PROMPT_VERSION.
- Retry x3 + fallback Haiku heredados.

Reports (pages/16_Reports.py)
==============================
- Botón 'Buscar patrones similares en archivo histórico' después
  del bloque RUL. Disabled con tooltip si no hay AI o no hay
  items.
- Caption explicativo: 'Reconoce patrones mecánicos similares en
  otros reportes archivados (mismo cliente, otros activos).
  Memoria institucional.'
- Render de matches: badge coloreado por band, identificadores,
  rationale del AI, resolución del caso histórico, aplicabilidad
  al caso actual, descarga del PDF citado.
- Inyección a meta['ai_patterns_matches'] +
  meta['ai_patterns_meta'] antes del PDF gen.
- PDF: sección 'PATRONES RECONOCIDOS EN ARCHIVO HISTÓRICO'
  después de RUL. Cada match con header coloreado +
  rationale + resolución + aplicabilidad.

Por qué es el moat competitivo definitivo de SIGA:
====================================================
Bently System 1, SKF Observer, Schaeffler ProLink, Emerson AMS
— ninguno puede tener esto. NO tienen los datos. A los 6 meses
de uso del programa de monitoreo, tu archivo es un activo
intelectual literalmente irreplicable.

Costo: ~\$0.04-0.10 por búsqueda (Sonnet) o ~5x menos (Haiku
fallback). Cache TTL 14 días → repeticiones gratis.

Cero regresiones: si archivo vacío, sin matches, sin items o sin
clickear, el reporte se genera normal sin la sección." || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene el ciclo 17.34 commiteado"
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
MERGE_MSG="release(v3.13.0): merge dev -> main · Pattern Memory

Memoria institucional con AI: cuando se prepara un reporte
nuevo, el sistema busca automáticamente patrones mecánicos
similares en TODO el archivo histórico accesible.

El archivo se vuelve cerebro colectivo. Cada reporte que se
archiva suma valor a los próximos análisis. Moat competitivo
definitivo de SIGA — ningún sistema de monitoreo industrial
(Bently, SKF, Schaeffler, Emerson) puede replicar esto sin
los datos del cliente.

Cero regresiones: feature opcional puro. Si no se clickea, no
afecta el reporte."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.13.0..."
TAG_EXISTS=$(git tag -l "v3.13.0")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.13.0 ya existe. Saltando creación."
else
    git tag -a v3.13.0 -m "Release v3.13.0 — Pattern Memory (memoria institucional con AI)"
    echo "  ✓ Tag v3.13.0 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.13.0 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ RELEASE v3.13.0 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar en 1-2 min."
echo ""
echo " VALIDACIÓN en producción:"
echo ""
echo "    Para que Pattern Memory devuelva resultados útiles, "
echo "    necesitás archivo histórico con varios reportes (mín 3)."
echo "    Si tenés <3 archivados, el feature va a funcionar pero"
echo "    devolver 'patrón sin antecedentes mecánicos similares'."
echo ""
echo "    1. Login en wm-home-final-2026.streamlit.app"
echo "    2. Activá un activo cualquiera, cargá data, andá a Reports"
echo "    3. Cargá las figuras del reporte"
echo "    4. (Opcional) Generá Síntesis Ejecutiva AI / Run-vs-Run / RUL"
echo "    5. Click 'Buscar patrones similares en archivo histórico'"
echo "    6. Spinner 10-25 seg → preview con los TOP 5 matches"
echo "    7. Cada match: badge de score + band, datos del caso,"
echo "       rationale, resolución previa, aplicabilidad, PDF descargable"
echo "    8. Click 'Preparar PDF' → sección 'PATRONES RECONOCIDOS EN"
echo "       ARCHIVO HISTÓRICO' después de la de RUL"
echo ""
echo " Demo killer para ventas a próximos clientes:"
echo ""
echo "    Tomá un reporte real con un caso de oil whip o BPFO etapa"
echo "    III y mostrá cómo el sistema reconoce que ese mismo patrón"
echo "    se vio hace meses en otro activo y dice cómo se resolvió."
echo "    Eso es lo que separa una herramienta de un cerebro."
echo ""
echo "================================================================"
