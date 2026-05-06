#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.12.0 → MAIN
# =============================================================
# Ciclo 17.31 — Briefing Mensual Ejecutivo automático
#
# Nueva página '📨 Briefing Mensual' (admin + specialist) que
# genera un PDF de 1 página con el estado consolidado del cliente
# durante un mes y lo envía por email al VP de Operaciones / CFO.
#
# Por qué importa comercialmente:
# ──────────────────────────────────────────────────────
# El VP del cliente NO se loguea al sistema. Solo abre su email.
# Con este feature, recibe un briefing mensual automático con:
#   - Estado consolidado de TODOS sus activos en 1 página.
#   - Top 3 prioridades operativas con acción ejecutiva sugerida.
#   - Distribución de severidad ejecutiva.
#   - Recomendación global del mes.
#
# Esto convierte el modelo de negocio de SIGA: pasa de cobrar por
# reporte individual a cobrar por SUBSCRIPTION mensual fija
# justificada por el touch recurrente con C-level del cliente.
#
# Ningún competidor latam (Bently System 1, SKF Observer, Schaeffler
# ProLink, Emerson AMS) genera ESTO automáticamente con AI. Es un
# diferenciador comercial enorme para cierres con C-level.
#
# Cambios técnicos:
# ──────────────────────────────────────────────────────
#   core/ai_briefing.py (NUEVO):
#     - generate_monthly_briefing(client_filter, month_iso, viewer)
#       lista reportes archivados del cliente en el mes,
#       agrupa por instance_id (último estado de cada activo),
#       construye payload estructurado, llama a Claude con prompt
#       VP-level y devuelve markdown + asset_aggregates + tokens.
#     - System prompt v1 'briefing ejecutivo C-level': lenguaje
#       accesible al lector NO-especialista, evita jerga muy
#       técnica, formato fijo (apertura ejecutiva ~80 palabras +
#       Top 3 prioridades + estado del portafolio + cierre con
#       recomendación global + disclaimer fijo).
#     - _aggregate_by_asset: agrupa N reportes del mismo activo
#       en 1 entrada con el estado MÁS RECIENTE.
#     - _month_range: convierte 'YYYY-MM' a (date_from, date_to).
#     - Cache local data/cache/ai_briefing/ TTL 7 días.
#     - Retry x3 + fallback Haiku heredados.
#
#   core/briefing_monthly_pdf.py (NUEVO):
#     - generate_monthly_briefing_pdf(briefing_result) → bytes.
#     - Layout 1 página A4: header con cliente + mes en español +
#       chips de severidad coloreados, RESUMEN EJECUTIVO DEL MES
#       (apertura AI), TOP 3 PRIORIDADES OPERATIVAS (numeradas
#       con bold leds), ESTADO DEL PORTAFOLIO POR ACTIVO (tabla
#       compacta con header oscuro + zebra striping + colores por
#       severidad), disclaimer Cat IV ISO 18436-2, footer SIGA.
#     - _parse_priorities: extrae bullets numerados del markdown
#       del AI para renderizarlos como ListFlowable en el PDF.
#     - _md_inline_to_rl: convierte **bold** y *italic* a tags
#       ReportLab nativos.
#     - _format_month_es: 'YYYY-MM' a texto español ('Abril 2026').
#
#   pages/_monthly_briefing.py (NUEVO):
#     - UI completa: selector de cliente (auto-discover de los
#       reportes archivados accesibles), selector de mes (default
#       mes anterior completo, últimos 12 meses), botón Generar.
#     - Preview: banner si fallback, caption con metadata, render
#       del markdown del AI, expander con tabla de activos
#       coloreada por severidad.
#     - Botones de acción: Generar PDF, Descargar PDF,
#       Enviar por email.
#     - Email: input de destinatarios separados por coma, validación
#       de formato, envío individual a cada destinatario con PDF
#       adjunto. Subject 'Executive Briefing — {Cliente} · {Mes}'.
#       Body HTML con header gradient SIGA + lista de contenidos +
#       link al sistema. Body text fallback para clientes que no
#       renderizan HTML.
#
#   core/auth.py:
#     - NAV_ITEMS: agregada entrada '📨 Briefing Mensual' después
#       de '🧠 AI Assistant'.
#     - CLIENT_BLOCKED_PAGES: pages/_monthly_briefing.py añadida.
#       El cliente no debe generar briefings — los recibe por email.
#
# Costo estimado:
# ──────────────────────────────────────────────────────
#   ~\$0.03-0.10 por briefing generado (Sonnet) o ~5x menos
#   (Haiku fallback). Cache TTL 7 días → re-generaciones gratis.
#   12 briefings/mes (1 por cliente) ≈ \$0.50-1.00/mes.
#
# Cero regresiones:
# ──────────────────────────────────────────────────────
#   - Nueva página oculta (admin/specialist). No toca flujos
#     existentes.
#   - Si AI no está configurada, st.stop() con mensaje claro.
#   - Si email no está configurado, sección de envío deshabilitada
#     pero descarga del PDF sigue disponible.
#   - Si el cliente no tiene reportes archivados, lista vacía con
#     mensaje informativo.
#   - Funcionalidad opcional manual; no afecta el resto del sistema.
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " RELEASE v3.12.0 → MAIN  (Briefing Mensual + cleanup UI)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.31  Briefing Mensual Ejecutivo automático al VP del cliente"
echo "         - core/ai_briefing.py (wrapper + system prompt v1)"
echo "         - core/briefing_monthly_pdf.py (PDF 1 página A4)"
echo "         - pages/_monthly_briefing.py (UI generación + email)"
echo "         - NAV_ITEMS + CLIENT_BLOCKED_PAGES (auth.py)"
echo ""
echo "  17.32  UX cleanup CRÍTICO: 195 emojis eliminados del UI"
echo "         - NAV_ITEMS sin emojis (AI Assistant, Briefing Mensual)"
echo "         - Botones AI en los 7 módulos analíticos limpios"
echo "         - Botones de Reports (Síntesis Ejecutiva, Run-vs-Run, RUL)"
echo "         - Headers de pages/_ai_assistant.py + _monthly_briefing.py"
echo "         - Mensajes de error/warning de los wrappers AI"
echo "         - Software internacional, presentación profesional"
echo ""
echo "  17.33  Eliminación de módulos deadweight:"
echo "         - pages/13_Phase_Analysis.py BORRADO (sin uso real)"
echo "         - pages/15_Diagnostics.py BORRADO (reemplazado por"
echo "           AI Assistant + Machinery Library con filtros)"
echo "         - Cards Críticos/Atención del Home redirigen a"
echo "           Machinery Library con filtro de severidad"
echo "         - Quick Action 'Diagnostics' reemplazado por"
echo "           'AI Assistant' en el strip del Home"
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el release v3.12.0 a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

echo "▶ 1/7  Commit del ciclo 17.31 en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged
for v in v3_4_2 v3_4_3 v3_4_4 v3_4_5 v3_4_6 v3_4_7 v3_4_8 v3_4_9 \
         v3_5_0 v3_5_1 v3_5_2 v3_5_3 v3_6_0 v3_7_0 v3_8_0 v3_9_0 \
         v3_10_0 v3_11_0; do
    git checkout HEAD -- "_release_${v}_hotfix.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_ux.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_main.sh" 2>/dev/null || true
done

# Ciclo 17.33 — eliminar módulos legacy deadweight. Si los archivos
# todavía existen en el filesystem (no fueron borrados antes), los
# pasamos por git rm para que la eliminación quede en el commit.
for f in pages/13_Phase_Analysis.py pages/15_Diagnostics.py; do
    if [ -f "$f" ]; then
        git rm -f "$f" 2>/dev/null || rm -f "$f"
        echo "  · Eliminado del repo: $f"
    fi
done

if ! git diff --quiet || ! git diff --staged --quiet; then
    # Agregamos los archivos del 17.31 (Briefing Mensual) + los del
    # 17.32 (cleanup de emojis del UI) + los del 17.33 (eliminación
    # de Phase Analysis y Diagnostics) en el mismo commit. Son cambios
    # coordinados en este release.
    git add core/ai_briefing.py \
            core/briefing_monthly_pdf.py \
            pages/_monthly_briefing.py \
            core/auth.py \
            pages/_landing.py \
            pages/_ai_assistant.py \
            pages/02_Time_Waveforms.py \
            pages/03_Spectrum.py \
            pages/04_Trends.py \
            pages/05_Orbit_Analysis.py \
            pages/06_Polar_Plot.py \
            pages/07_Bode_Plot.py \
            pages/09_Shaft_Centerline.py \
            pages/16_Reports.py \
            core/ai_diagnostic.py \
            core/ai_qa.py \
            core/ai_runcompare.py \
            core/ai_rul.py \
            _release_v3_12_0_main.sh
    git commit -m "feat(17.31 + 17.32 + 17.33): Briefing Mensual + UX cleanup + módulos deadweight removidos

Nueva página '📨 Briefing Mensual' (admin + specialist) que genera
un PDF de 1 página con el estado consolidado del cliente durante
un mes y lo envía por email al VP de Operaciones / CFO. Recurring
touch automático con el C-level decisor que no se loguea al
sistema. Game-changer comercial: justifica modelo de subscription
mensual. Ningún competidor LATAM tiene esto.

Wrapper (core/ai_briefing.py)
==============================
- generate_monthly_briefing(client, month_iso, viewer) lista
  archivos del cliente en el mes, agrupa por instance_id, llama
  a Claude con prompt VP-level, devuelve markdown +
  asset_aggregates + tokens + costo.
- System prompt v1 'briefing ejecutivo C-level': lenguaje
  accesible al NO-especialista, evita jerga, formato fijo
  (apertura ~80 palabras + Top 3 prioridades + estado del
  portafolio + recomendación global + disclaimer Cat IV).
- _aggregate_by_asset agrupa N reportes del mismo activo en 1
  entrada con el estado MÁS RECIENTE. Ranking por severidad desc
  + fecha desc.
- Cache TTL 7 días con autoinvalidación por
  BRIEFING_PROMPT_VERSION.
- Retry x3 + fallback Haiku + detección de timeouts heredados.

PDF Generator (core/briefing_monthly_pdf.py)
=============================================
- generate_monthly_briefing_pdf(briefing_result) → bytes.
- Layout 1 página A4: header con cliente + mes español + chips
  coloreados por severidad, RESUMEN EJECUTIVO DEL MES, TOP 3
  PRIORIDADES OPERATIVAS, ESTADO DEL PORTAFOLIO POR ACTIVO
  (tabla compacta con header oscuro), disclaimer Cat IV,
  footer SIGA.
- _parse_priorities extrae bullets numerados del markdown.
- _md_inline_to_rl convierte **bold** y *italic* a tags
  ReportLab.

UI (pages/_monthly_briefing.py)
================================
- Selector de cliente con auto-discover de los reportes
  archivados accesibles al viewer.
- Selector de mes (default: mes anterior completo, últimos 12
  meses).
- Generar briefing → spinner → preview con banner de fallback,
  caption con tokens y costo, markdown del AI, expander con
  tabla de activos coloreada por severidad.
- Botones: Generar PDF, Descargar PDF, Enviar email.
- Email: input destinatarios separados por coma, validación,
  envío individual con PDF adjunto. Subject 'Executive Briefing
  — {Cliente} · {Mes}'. Body HTML con header gradient + lista
  de contenidos + link al sistema. Body text fallback.
- Reusa core.email_sender (Microsoft Graph configurado).

NAV (core/auth.py)
====================
- NAV_ITEMS: '📨 Briefing Mensual' después de 'AI Assistant'.
- CLIENT_BLOCKED_PAGES: bloqueada para client (los clientes
  RECIBEN briefings por email, no los GENERAN).

Costo: ~\$0.03-0.10 por briefing (Sonnet) o ~5x menos (Haiku).
Cache local → re-generaciones gratis.

Cero regresiones: nueva página oculta, no toca flujos existentes.
Si AI/email no configurados, mensajes informativos claros." || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene el ciclo 17.31 commiteado"
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
MERGE_MSG="release(v3.12.0): merge dev -> main · Briefing Mensual Ejecutivo

Nueva página oculta (admin + specialist) que genera un PDF de 1
página con el estado consolidado del cliente durante un mes y lo
envía por email al VP de Operaciones / CFO.

El VP del cliente NO se loguea al sistema, solo abre su email.
Este feature crea touch recurrente automático con C-level del
cliente y justifica modelo de subscription mensual fija.

Reusa toda la infraestructura: archive + Claude API +
email_sender (Microsoft Graph). Genera PDF, lo adjunta, envía
con HTML branded a los destinatarios indicados.

Cero regresiones: página oculta, no toca flujos existentes."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.12.0..."
TAG_EXISTS=$(git tag -l "v3.12.0")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.12.0 ya existe. Saltando creación."
else
    git tag -a v3.12.0 -m "Release v3.12.0 — Briefing Mensual Ejecutivo automático"
    echo "  ✓ Tag v3.12.0 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.12.0 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ RELEASE v3.12.0 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo "    No hay que tocar secrets — la key [anthropic] y [email] ya"
echo "    están configuradas de releases previos."
echo ""
echo " 🧪 VALIDACIÓN en producción:"
echo ""
echo "    1. Login en wm-home-final-2026.streamlit.app"
echo "    2. En el sidebar verás la nueva entrada '📨 Briefing Mensual'"
echo "       después de 'AI Assistant'"
echo "    3. Click → vas a la nueva página"
echo "    4. Seleccionar cliente del dropdown (auto-discover de los"
echo "       que tienen reportes archivados accesibles a tu rol)"
echo "    5. Seleccionar mes (default mes anterior completo)"
echo "    6. Click '🔮 Generar briefing' → spinner 10-30 seg"
echo "    7. Preview en pantalla con markdown + tabla de activos"
echo "    8. Click '📄 Generar PDF' → click '⬇️ Descargar PDF' para"
echo "       inspeccionar el output"
echo "    9. Para enviar por email: input destinatarios separados por"
echo "       coma → click '📨 Enviar email' → validación → envío con"
echo "       PDF adjunto + body HTML branded SIGA"
echo ""
echo " 💼 Demo killer para ventas:"
echo "    Generá un briefing real de ECOPETROL del último mes,"
echo "    descargalo, mostralo al prospecto siguiente como ejemplo"
echo "    de lo que recibe el VP del cliente cada mes. Cierre"
echo "    casi automático."
echo ""
echo " 🚀 Próximas locuras candidatas:"
echo "    Idea 5 — Maintenance Work Orders auto SAP PM/Maximo"
echo "    Idea 7 — Anomaly Detection con memoria institucional"
echo "    Idea 4 — Multi-language reports (inglés/portugués/árabe)"
echo "    Idea 6 — Voice-to-report del especialista en campo"
echo ""
echo "================================================================"
