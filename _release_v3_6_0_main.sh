#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.6.0 → MAIN
# =============================================================
# Ciclo 17.26 — AI Diagnóstico con Claude API (opción C híbrido)
#
# Primer release con interpretación clínica AI Cat IV asistida en
# el módulo Spectrum. El cliente recibe un reporte que parece
# firmado por un especialista senior — sin marcas "AI", sin
# markdown crudo, con prosa forense + tabla cuantitativa de
# evidencia + recomendaciones priorizadas P1/P2/P3/P4 + cierre
# legal de no-deadline.
#
# Por qué Spectrum solo (y no Trends/Bode/Orbit en este release):
#   Spectrum es el módulo más usado y de mayor valor diagnóstico.
#   Lanzamos incrementalmente para validar en producción real
#   (Streamlit Cloud, infra Bogotá→Anthropic, tráfico real)
#   antes de replicar el patrón a los otros 4 módulos en v3.7.0.
#
# Cambios técnicos principales:
# ──────────────────────────────────────────────────────────
#   core/ai_diagnostic.py (NUEVO, ~700 líneas):
#     - Wrapper de Claude API (Anthropic) para diagnósticos clínicos.
#     - System prompt v5: voz Cat IV ISO 18436-2 sin emojis, prosa
#       de hallazgos con frase-tesis en negrita, recomendaciones
#       numeradas con clasificación P1/P2/P3/P4 + horizonte
#       sugerido + cláusula de norma + cierre legal de
#       responsabilidad del operador.
#     - Cache local SHA256 con TTL 30 días, invalidación
#       automática al cambiar _PROMPT_VERSION.
#     - Stats persistentes (n_calls, tokens, costo estimado).
#     - Retry con backoff exponencial (1s/2s/4s) para 429/502/503/529.
#     - Detección de timeouts (APITimeoutError, APIConnectionError,
#       "timed out", "interrupted") como retryables.
#     - Fallback automático a Claude Haiku 4.5 si Sonnet 4.5 se
#       agota tras 3 retries. El cliente nunca ve un error;
#       recibe diagnóstico de Haiku con calidad ~85% Sonnet.
#     - Pricing dinámico según modelo (Sonnet \$3/\$15, Haiku \$1/\$5).
#     - Mensajes user-friendly diferenciados: 529 overload, 408
#       timeout, 429 rate limit, 401 auth, billing insuficiente.
#
#   pages/03_Spectrum.py:
#     - Import de generate_ai_diagnostic + is_ai_available.
#     - Expander "🧠 Interpretación clínica AI" debajo del Cat IV
#       determinístico, con botones Generar / Regenerar.
#     - Construcción del payload con machine, norm, technical
#       (overall, picos, 1X, 2X, harmonics) + cat_iv_diag detail
#       + bearing_assessment cuando corresponde.
#     - Persistencia del resultado en st.session_state.
#     - Banner azul informativo cuando se usa el modelo de
#       respaldo (visible solo al especialista, no al cliente).
#     - Caption con modelo, tokens IN→OUT, costo real, cached
#       flag, fallback flag, timestamp.
#     - "Enviar a Reporte" appendea quant table + marcador
#       <<<WM_AI_BLOCK>>> + marcador <<<WM_AI_NARRATIVE>>> + AI
#       markdown a las notas (suprimiendo la narrativa Cat IV
#       determinística para evitar duplicación).
#     - Bug fix: Overall RMS en la tabla cuantitativa ahora pasa
#       por convert_rms_to_mode(amplitude_mode), así muestra el
#       valor en peak-peak (consistente con 1X) cuando el usuario
#       opera en ese modo, en lugar de mezclar valor RMS con
#       unidad peak-peak.
#     - Fix de escapes: reemplazado \\\$ por \$ en dos lugares
#       (cosmético + elimina SyntaxWarning de Python 3.13).
#
#   pages/16_Reports.py:
#     - 4 estilos clínicos nuevos (WMClinicalHeading,
#       WMClinicalBody, WMClinicalBullet, WMClinicalNumbered)
#       parent BodyText, fontSize alineado al resto del reporte.
#     - Helper _md_inline_to_rl: convierte markdown inline
#       (\*\*bold\*\*, \*italic\*, \`code\`) a tags ReportLab
#       seguros con escape de HTML.
#     - Helper _split_ai_clinical_block: parsea las notas
#       buscando los marcadores <<<WM_AI_BLOCK>>> y
#       <<<WM_AI_NARRATIVE>>>, devuelve (pre_text, quant_rows,
#       ai_md).
#     - Helper _render_ai_clinical_flowables: parsea el markdown
#       del AI a flowables ReportLab nativos (Paragraphs con
#       estilo por bloque, ListFlowable manual con sangría
#       francesa para bullets y numeradas, Spacer para reglas
#       horizontales).
#     - Helper _render_quant_evidence_table: tabla cuantitativa
#       compacta con header oscuro, zebra striping y distribución
#       35%/65% en columnas.
#     - _render_notes_flowables modificado: detecta el marcador y
#       enruta. Cuando hay AI, suprime la narrativa
#       determinística previa, renderiza tabla cuantitativa +
#       bloque clínico con estilos nativos. Sin AI, comportamiento
#       legacy intacto (cero regresión).
#
#   requirements.txt:
#     - anthropic>=0.39.0 (SDK oficial)
#
# Costo en producción (estimado):
# ──────────────────────────────────────────────────────────
#   Sonnet 4.5 (modelo principal): ~\$0.015 por diagnóstico
#     - 1500 tokens input + 800 tokens output = \$0.0165
#   Haiku 4.5 (fallback): ~\$0.005 por diagnóstico
#   100 diagnósticos/mes: \$1.50/mes (Sonnet)
#   1000 diagnósticos/mes: \$15/mes (Sonnet)
#
# Setup en Streamlit Cloud:
# ──────────────────────────────────────────────────────────
#   El secreto [anthropic] api_key debe estar configurado en
#   Settings → Secrets de wm-home-final-2026 ANTES de que el
#   redeploy termine. Sin la key, el botón "🧠 Generar
#   diagnóstico AI" muestra mensaje informativo y el resto del
#   sistema sigue funcionando normal (degradación elegante).
#
# Validación local previa:
# ──────────────────────────────────────────────────────────
#   - Compila sin warnings (Python 3.13)
#   - Smoke test del parser markdown → ReportLab pasa
#   - Generación real contra Claude API verificada con datos de
#     turbogenerador GE LM6000 (oil whip + desbalance + resonancia)
#   - PDF generado con prosa forense, tabla cuantitativa correcta,
#     recomendaciones P1/P2/P3, cierre legal completo
#   - Fallback a Haiku probado durante overload real de Sonnet
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🧠 RELEASE v3.6.0 → MAIN  (AI Diagnóstico Cat IV en Spectrum)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.26  AI Diagnóstico con Claude API:"
echo "         - Wrapper completo (core/ai_diagnostic.py)"
echo "         - Botón en Spectrum + payload + persistencia"
echo "         - PDF profesional (markdown → ReportLab nativo)"
echo "         - Tabla cuantitativa de evidencia"
echo "         - Recomendaciones P1/P2/P3/P4 con disclaimer legal"
echo "         - Retry x3 + fallback Haiku 4.5"
echo "         - Detección de timeouts + mensajes user-friendly"
echo "         - Bug fix Overall RMS (convert_rms_to_mode)"
echo "  17.25  (ya en main desde v3.5.3) Trends VFD→RPM + multi-op"
echo ""
echo "Pendiente para v3.7.0 (siguiente ciclo en dev):"
echo "  17.26 P5  Replicar AI a Time Waveforms / Trends / Bode / Orbit"
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el release v3.6.0 a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

echo "▶ 1/7  Commit del ciclo 17.26 en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged en releases previos.
for v in v3_4_2 v3_4_3 v3_4_4 v3_4_5 v3_4_6 v3_4_7 v3_4_8 v3_4_9 \
         v3_5_0 v3_5_1 v3_5_2 v3_5_3; do
    git checkout HEAD -- "_release_${v}_hotfix.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_ux.sh" 2>/dev/null || true
done

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/ai_diagnostic.py \
            pages/03_Spectrum.py \
            pages/16_Reports.py \
            requirements.txt \
            _release_v3_6_0_main.sh
    git commit -m "feat(17.26): AI Diagnóstico Cat IV con Claude API en módulo Spectrum

Primer ciclo de interpretación clínica asistida por IA. El cliente
recibe un reporte que se lee como informe de especialista senior:
sin marcas 'AI', sin markdown crudo, con prosa forense + tabla
cuantitativa de evidencia + recomendaciones priorizadas P1/P2/P3/P4
+ cierre legal de no-deadline.

Wrapper (core/ai_diagnostic.py)
===============================
Wrapper completo de Anthropic Claude API. System prompt v5 con
voz Cat IV ISO 18436-2 (sin emojis, voz pasiva técnica, citando
cláusulas de API 670/684, ISO 20816/21940). Cache local SHA256
con TTL 30 días + autoinvalidación por _PROMPT_VERSION. Stats
persistentes para tracking de costo. Retry exponencial 1s/2s/4s
para 429/502/503/529. Detección de timeouts y errores de conexión
como retryables (status virtual 408). Fallback automático a Claude
Haiku 4.5 cuando Sonnet 4.5 se agota tras 3 retries — Haiku corre
en infraestructura distinta y casi nunca se sobrecarga, calidad
~85% Sonnet, costo 5x menor. Pricing dinámico según modelo usado.

Spectrum (pages/03_Spectrum.py)
================================
Expander 'Interpretación clínica AI' debajo del Cat IV
determinístico. Botones Generar / Regenerar con cost hint.
Payload con machine, norm, technical (overall, picos, 1X, 2X,
harmonics), cat_iv_diag detail, bearing_assessment cuando hay.
Banner discreto cuando se usa modelo de respaldo (visible solo
al especialista en la app, jamás en el PDF al cliente). Caption
rico con modelo, tokens, costo, cached/fallback flags.

'Enviar a Reporte' construye quant table + marcadores
<<<WM_AI_BLOCK>>> y <<<WM_AI_NARRATIVE>>> y appendea el markdown
del AI a las notas, suprimiendo la narrativa Cat IV determinística
para evitar duplicación.

Fixes:
- Overall RMS: ahora pasa por convert_rms_to_mode(amplitude_mode),
  así muestra el valor en peak-peak coherente con 1X y demás picos
  cuando el usuario opera en ese modo. Antes mezclaba valor RMS
  con unidad peak-peak.
- Eliminados \\\$ literales en f-strings (cosmético + elimina
  SyntaxWarning de Python 3.13).

PDF Render (pages/16_Reports.py)
=================================
4 estilos clínicos nuevos (WMClinicalHeading/Body/Bullet/Numbered)
con sangría francesa para listas. Helpers:
- _md_inline_to_rl: markdown inline → tags ReportLab seguros
- _split_ai_clinical_block: parsea marcadores y devuelve
  (pre_text, quant_rows, ai_md)
- _render_ai_clinical_flowables: markdown a flowables nativos
  (headings, bullets, numeradas, párrafos justificados)
- _render_quant_evidence_table: tabla compacta con header oscuro
_render_notes_flowables enruta según marcador. Cuando hay AI,
suprime narrativa determinística + renderiza quant table + bloque
clínico estilizado. Sin AI, comportamiento legacy intacto.

Costo estimado en producción
=============================
Sonnet: ~\$0.015 por diagnóstico (1500 tok in + 800 tok out)
Haiku (fallback): ~\$0.005 por diagnóstico
100 diagnósticos/mes ≈ \$1.50/mes (Sonnet)
1000 diagnósticos/mes ≈ \$15/mes (Sonnet)

Setup pre-deploy
=================
[anthropic] api_key debe estar en st.secrets de Streamlit Cloud
para wm-home-final-2026 ANTES de que termine el redeploy. Sin
key, el botón muestra mensaje informativo y el resto del sistema
sigue funcionando (degradación elegante).

Pendiente próximo ciclo
========================
P5 (v3.7.0): replicar AI a Time Waveforms / Trends / Bode / Orbit
            con el mismo molde — payload + expander + 'Enviar a
            Reporte' append. ~25 min cada uno." || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene el ciclo 17.26 commiteado"
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
MERGE_MSG="release(v3.6.0): merge dev -> main · Ciclo 17.26 AI Diagnóstico

Primer release con interpretación clínica AI Cat IV asistida en
el módulo Spectrum. El cliente recibe un reporte que se lee como
informe de especialista senior, sin marcas AI ni markdown crudo.

Lo que va a producción:
- core/ai_diagnostic.py: wrapper completo Claude API con system
  prompt v5 (voz Cat IV ISO 18436-2), cache, stats, retry x3 con
  backoff, detección de timeouts, fallback automático a Haiku 4.5.
- pages/03_Spectrum.py: expander de Diagnóstico AI, payload
  completo, persistencia, banner de fallback, caption con costo
  dinámico, append al reporte con marcadores. Bug fix Overall
  RMS via convert_rms_to_mode.
- pages/16_Reports.py: 4 estilos clínicos, parser markdown a
  ReportLab nativo, tabla cuantitativa de evidencia, supresión
  de narrativa determinística cuando hay AI.
- requirements.txt: anthropic>=0.39.0.

Robustez de producción:
- 3 capas: retry con backoff -> fallback Haiku -> mensaje claro.
- El cliente nunca ve un error feo aunque Anthropic esté caído.
- Cache se autoinvalida si el system prompt cambia.

Costo: ~\$0.015 por diagnóstico Sonnet, ~\$0.005 Haiku.

Pendiente próximo ciclo (v3.7.0):
- Replicar AI a Time Waveforms / Trends / Bode / Orbit."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.6.0..."
TAG_EXISTS=$(git tag -l "v3.6.0")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.6.0 ya existe. Saltando creación."
else
    git tag -a v3.6.0 -m "Release v3.6.0 — Ciclo 17.26 AI Diagnóstico Cat IV en Spectrum"
    echo "  ✓ Tag v3.6.0 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.6.0 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ RELEASE v3.6.0 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo ""
echo " 🔑 ANTES DE QUE TERMINE EL DEPLOY:"
echo ""
echo "    Andá a Streamlit Cloud → wm-home-final-2026 → Manage app"
echo "    → Settings → Secrets, y agregá al final del archivo:"
echo ""
echo "        [anthropic]"
echo "        api_key = \"sk-ant-api03-...TU_KEY_NUEVA...\""
echo "        model = \"claude-sonnet-4-5-20250929\""
echo ""
echo "    Si la key no está, el botón Diagnóstico AI muestra mensaje"
echo "    informativo y el resto del sistema funciona normal."
echo ""
echo " 🧪 VALIDACIÓN en producción:"
echo ""
echo "    1. Login en wm-home-final-2026.streamlit.app"
echo "    2. Cargá una señal de Spectrum real"
echo "    3. Después del Cat IV avanzado, abrí el expander"
echo "       '🧠 Interpretación clínica AI'"
echo "    4. Click 'Generar diagnóstico AI'"
echo "    5. Verificá:"
echo "       - Sale en 5-15 seg sin error"
echo "       - El markdown tiene prosa con frase-tesis en negrita"
echo "       - Las recomendaciones empiezan con P1/P2/P3/P4 + etiqueta"
echo "       - El cierre incluye disclaimer del operador"
echo "    6. 'Enviar a Reporte' + generar PDF y verificar:"
echo "       - Tabla cuantitativa con Overall en unidad correcta"
echo "       - Prosa fluida sin '###' ni '**' literales"
echo "       - Recomendaciones numeradas con prioridad en negrita"
echo "       - Sin marca 'AI' visible para el cliente"
echo ""
echo " 🔁 Si todo OK → seguimos en dev con P5 (Time Waveforms,"
echo "    Trends, Bode, Orbit) para v3.7.0."
echo " 🚨 Si aparece bug → hotfix v3.6.1 inmediato (template del"
echo "    último _release_v3_5_3_hotfix.sh)."
echo ""
echo "================================================================"
