#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.9.0 → MAIN
# =============================================================
# Ciclo 17.27 — AI Assistant: Q&A sobre el archivo histórico
#
# Nueva página tipo chat donde el usuario hace preguntas en
# lenguaje natural sobre los reportes archivados, y Claude
# responde con citaciones precisas a los reportes específicos
# (consecutivo, fecha, activo, severidad). Cada citación
# incluye un botón directo de descarga del PDF.
#
# Este es el feature que convierte el archivo histórico de un
# PDF dump en una BASE DE CONOCIMIENTO consultable. Demo killer
# para ventas: "Mostrame todos los activos con oil whip de los
# últimos 6 meses" → respuesta con 4 reportes citados, todos
# del cliente, descargables al click.
#
# Cambios técnicos:
# ──────────────────────────────────────────────────────────
#   core/ai_qa.py (NUEVO):
#     - query_archive(question, viewer_email, viewer_role) →
#       respuesta con citaciones precisas + tokens + costo.
#     - System prompt v1: asistente técnico Cat IV ISO 18436-2
#       que SOLO responde con base en los reportes provistos,
#       cita cada afirmación con [REPORT:archive_id], detecta
#       cuándo la pregunta no se puede responder con la info
#       disponible.
#     - _infer_filters_from_question: heurísticas baratas que
#       detectan año, rango temporal ("últimos 6 meses",
#       "este año"), cliente nombrado (ECOPETROL, MAGNEX, etc.),
#       reduciendo el contexto enviado a Claude.
#     - _parse_citations: extrae [REPORT:archive_id] del
#       markdown de respuesta y filtra alucinaciones de IDs
#       (solo cita los que efectivamente están en el contexto).
#     - extract_pdf_text con cache local (TTL 30 días por
#       hash de path+mtime+size). Solo se invoca si el usuario
#       pide "análisis profundo".
#     - Cache de respuestas en data/cache/ai_qa/ con
#       autoinvalidación al cambiar QA_PROMPT_VERSION.
#     - Robustez heredada de ai_diagnostic: retry x3 con
#       backoff + fallback Haiku 4.5 + detección de timeouts +
#       mensajes user-friendly.
#     - Pricing dinámico según modelo usado.
#
#   pages/_ai_assistant.py (NUEVO):
#     - UI tipo chat con st.chat_message + st.chat_input.
#     - Sidebar con 6 preguntas sugeridas comunes (clickeables).
#     - Toggle "Análisis profundo" que activa extract_pdf_text.
#     - Render de cada respuesta:
#         * Banner si se usó modelo de respaldo
#         * Markdown con citaciones [REPORT:...]
#         * Sección expandible "Reportes citados" con cada uno
#           teniendo título + cliente + activo + fecha +
#           severidad + botón "⬇ PDF" descargable
#         * Caption con modelo + reportes en contexto + tokens +
#           costo de esa consulta
#     - Conversación persistida en st.session_state.
#     - Acumulador de tokens y costo de la sesión.
#     - Botones de "Limpiar conversación" + "Limpiar cache" (admin).
#     - Expander "Archivo accesible para tu rol" mostrando
#       cuántos reportes ve el usuario, agrupados por cliente y
#       severidad (vista rápida del scope).
#
#   core/auth.py:
#     - NAV_ITEMS: agregado "🧠 AI Assistant" como última entrada.
#     - CLIENT_BLOCKED_PAGES: pages/_ai_assistant.py incluida
#       (solo admin + specialist en v1; client se evalúa
#       cuando se exponga el feature al cliente externo).
#
# Permisos por role:
# ──────────────────────────────────────────────────────────
#   admin:      ve TODOS los reportes archivados → puede
#               consultar el archivo completo.
#   specialist: ve los suyos + de otros @sigasas.com → puede
#               consultar el archivo de su equipo.
#   client:     bloqueado en v1. Cuando se libere, usaremos
#               filtro shared_with_client=True para que solo
#               vea los marcados como compartidos.
#
# Costo estimado por consulta:
# ──────────────────────────────────────────────────────────
#   Sonnet 4.5: ~\$0.03-0.10 según tamaño del archivo
#               consultado (más reportes en contexto = más
#               tokens IN). Output ~500-800 tokens típico.
#   Haiku (fallback): ~5x más barato que Sonnet.
#   Cache local de respuestas → consultas repetidas ~ \$0.
#   Estimado realista: 50 consultas/mes ≈ \$2-3/mes.
#
# Cero regresiones:
# ──────────────────────────────────────────────────────────
#   - Es una página completamente nueva. No toca ningún flujo
#     existente.
#   - Si la key de Anthropic no está, la página muestra mensaje
#     informativo y st.stop(). El resto del sistema funciona.
#   - Si Anthropic está overloaded, retry+fallback heredados.
#   - El archivo histórico tradicional (Reports → Archivo
#     histórico) sigue siendo opción válida con filtros manuales.
#
# Validación previa:
# ──────────────────────────────────────────────────────────
#   - Compila sin warnings (Python 3.13)
#   - Smoke test del filter inferer: detecta cliente, año,
#     rangos temporales correctamente
#   - Smoke test del citation parser: extrae IDs limpiamente,
#     filtra alucinaciones
#   - Verificación del NAV_ITEMS: AI Assistant aparece como
#     última entrada, bloqueada para client
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🧠 RELEASE v3.9.0 → MAIN  (AI Assistant: Q&A sobre archivo)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.27  AI Assistant — Q&A sobre archivo histórico"
echo "         - core/ai_qa.py (wrapper + system prompt + parser)"
echo "         - pages/_ai_assistant.py (UI tipo chat)"
echo "         - NAV_ITEMS + CLIENT_BLOCKED_PAGES en core/auth.py"
echo ""
echo "  Nueva página visible para admin + specialist con:"
echo "  - Chat input + 6 sugerencias clickeables"
echo "  - Toggle 'Análisis profundo' (extrae texto del PDF)"
echo "  - Citaciones precisas con descarga directa del PDF"
echo "  - Acumulador de tokens y costo de la sesión"
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el release v3.9.0 a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

echo "▶ 1/7  Commit del ciclo 17.27 en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged
for v in v3_4_2 v3_4_3 v3_4_4 v3_4_5 v3_4_6 v3_4_7 v3_4_8 v3_4_9 \
         v3_5_0 v3_5_1 v3_5_2 v3_5_3 v3_6_0 v3_7_0 v3_8_0; do
    git checkout HEAD -- "_release_${v}_hotfix.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_ux.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_main.sh" 2>/dev/null || true
done

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/ai_qa.py \
            pages/_ai_assistant.py \
            core/auth.py \
            _release_v3_9_0_main.sh
    git commit -m "feat(17.27): AI Assistant — Q&A sobre archivo histórico de reportes

Nueva página tipo chat donde el especialista o admin pregunta en
lenguaje natural sobre los reportes archivados. Claude responde
con citaciones precisas que incluyen botón de descarga del PDF
correspondiente. Convierte el archivo histórico de un PDF dump
en una base de conocimiento consultable.

Wrapper (core/ai_qa.py)
=========================
- query_archive(question, viewer_email, viewer_role) con
  retorno estructurado: markdown + reports_referenced +
  tokens + costo + flags de fallback.
- System prompt v1 'asistente técnico Cat IV ISO 18436-2':
  responde solo con base en reportes provistos, cita con
  [REPORT:archive_id], declara cuándo no puede responder.
- _infer_filters_from_question: heurísticas baratas que
  detectan año, rangos temporales y cliente nombrado para
  reducir el contexto enviado.
- _parse_citations: extrae IDs citados y filtra alucinaciones.
- extract_pdf_text con cache local TTL 30d (sólo si el usuario
  activa 'análisis profundo').
- Cache de respuestas en data/cache/ai_qa/ con
  autoinvalidación por QA_PROMPT_VERSION.
- Reusa retry x3 + fallback Haiku 4.5 + detección de timeouts
  + pricing dinámico de core/ai_diagnostic.py.

Página (pages/_ai_assistant.py)
=================================
- UI nativa de chat con st.chat_message + st.chat_input.
- Sidebar con 6 preguntas sugeridas clickeables, toggle de
  análisis profundo, contador de tokens y costo de la sesión,
  botones de limpiar conversación y limpiar cache (admin).
- Render de cada respuesta: banner de fallback si aplica,
  markdown con citaciones, sección expandible 'Reportes citados'
  con título + cliente + activo + fecha + severidad + botón
  descargar PDF, caption con modelo y costo de la consulta.
- Expander 'Archivo accesible para tu rol' que muestra cuántos
  reportes ve el usuario agrupados por cliente y severidad.
- Conversación persistida en st.session_state['wm_aiq_history'].

NAV (core/auth.py)
====================
- NAV_ITEMS: agregada entrada '🧠 AI Assistant' al final.
- CLIENT_BLOCKED_PAGES: pages/_ai_assistant.py añadida en v1
  (solo admin + specialist). Para liberar al client en una
  iteración futura, removerla de la lista y filtrar consultas
  a shared_with_client=True dentro de la página.

Costo estimado: ~\$0.03-0.10 por consulta (Sonnet) o ~5x
menos en Haiku (fallback). Cache local de respuestas → las
consultas repetidas son virtualmente gratis.

Cero regresiones: es una página nueva que no toca ningún flujo
existente. El archivo histórico tradicional (Reports → Archivo
histórico con filtros manuales) sigue siendo opción válida.

Validado: smoke tests de filter inferer + citation parser
pasan limpios. Compila sin warnings en Python 3.13." || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene el ciclo 17.27 commiteado"
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
MERGE_MSG="release(v3.9.0): merge dev -> main · AI Assistant Q&A sobre archivo

Nueva página tipo chat para hacer preguntas en lenguaje natural
sobre el archivo histórico de reportes. El sistema responde con
citaciones precisas que incluyen descarga directa del PDF.

Convierte el archivo histórico en base de conocimiento
consultable. Demo killer para ventas: 'Mostrame todos los
activos con oil whip de los últimos 6 meses' → respuesta con
N reportes citados del cliente, descargables al click.

Solo admin + specialist por ahora; el cliente externo se
expondrá en una iteración futura con filtro shared_with_client.

Costo: ~\$0.03-0.10 por consulta. Robustez heredada de
ai_diagnostic (retry+fallback Haiku+timeouts).

Cero regresiones: página nueva que no toca el flujo existente."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.9.0..."
TAG_EXISTS=$(git tag -l "v3.9.0")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.9.0 ya existe. Saltando creación."
else
    git tag -a v3.9.0 -m "Release v3.9.0 — AI Assistant: Q&A sobre archivo histórico"
    echo "  ✓ Tag v3.9.0 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.9.0 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ RELEASE v3.9.0 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo "    No hay que tocar secrets — la key [anthropic] sigue de v3.6.0."
echo ""
echo " 🧪 VALIDACIÓN en producción:"
echo ""
echo "    1. Login en wm-home-final-2026.streamlit.app"
echo "    2. En el sidebar verás la nueva entrada '🧠 AI Assistant'"
echo "       como última de la lista (después de Reports)"
echo "    3. Click → llegás a la página chat"
echo "    4. En el sidebar derecho hay 6 preguntas sugeridas:"
echo "       - '¿Cuáles activos tienen severidad CRÍTICA?'"
echo "       - 'Mostrame reportes con oil whip de los últimos 6 meses'"
echo "       - etc."
echo "    5. Click en una sugerencia → spinner 5-30 seg → respuesta"
echo "    6. Verificá:"
echo "       - Markdown bien renderizado, citaciones claras"
echo "       - Sección 'Reportes citados' con botones ⬇ PDF"
echo "       - Click en ⬇ PDF descarga el reporte real"
echo "       - Caption con modelo, tokens, costo de la consulta"
echo "       - Sidebar acumula tokens y costo de la sesión"
echo "    7. Probá una pregunta libre escribiéndola en el chat input"
echo ""
echo " 💡 Demo poderoso para ventas:"
echo "    Pedí ante ECOPETROL/MAGNEX: 'Mostrame todos los reportes"
echo "    de TES1 con tendencia ascendente' y dejá que el cliente"
echo "    vea cómo cita reportes específicos con descarga directa."
echo ""
echo "================================================================"
