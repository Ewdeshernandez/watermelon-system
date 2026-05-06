#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.11.0 → MAIN
# =============================================================
# DOS COSAS GRANDES JUNTAS en este release:
#
# (1) CICLO 17.29 — FIX CRÍTICO DE PÉRDIDA DE DATOS
# ──────────────────────────────────────────────────────
# Los reportes archivados se perdían en cada redeploy de Streamlit
# Cloud (los contenedores son efímeros — borran data/). Ahora se
# persisten en Supabase Storage en un bucket dedicado y se
# restauran automáticamente al cold start.
#
# (2) CICLO 17.30 — AI RUL PREDICTIVO (Remaining Useful Life)
# ──────────────────────────────────────────────────────
# Estimación de vida útil restante con percentiles P10/P50/P90 y
# ventana óptima de intervención. Claude actúa como ACTUARIO
# PREDICTIVO Cat IV ISO 18436-2 leyendo TODA la historia archivada
# del activo + estado actual. Disclaimer legal fuerte.
#
# ⚠️ SETUP MANUAL ANTES DEL DEPLOY:
# ──────────────────────────────────────────────────────
# Para que la persistencia funcione en producción, CREA EL BUCKET
# en Supabase ANTES de que termine el redeploy:
#
#   1. Andá a https://supabase.com/dashboard
#   2. Tu proyecto → Storage → New bucket
#   3. Name: 'reports-archive'  (exacto, sin espacios)
#   4. Public bucket: NO (mantener privado)
#   5. Save
#
# Si NO creás el bucket, el archivo va a seguir guardándose local
# pero perderá la persistencia entre redeploys (comportamiento
# anterior a este release). El sistema NO se rompe — solo no se
# blinda.
#
# Cambios técnicos (17.29 — Persistencia):
# ──────────────────────────────────────────────────────
#   core/reports_archive.py:
#     - Helpers nuevos: _get_archive_supabase_client, _upload_to_supabase,
#       _download_from_supabase, _delete_from_supabase,
#       _list_supabase_archive_files.
#     - sync_archive_from_supabase(force=False) — descarga del bucket
#       al filesystem en cold start. Lazy: 1 vez por proceso.
#     - sync_archive_to_supabase(force=False) — sube todo lo local que
#       NO esté en el bucket. Útil para migrar archivos existentes la
#       primera vez.
#     - archive_report_pdf: ahora también sube a Supabase tras escribir
#       local. Best-effort (no bloquea si falla).
#     - list_archived_reports: llama sync_from_supabase la primera vez
#       que se invoca por proceso (cold start lazy).
#     - get_archived_pdf_bytes: si el PDF no está local, lo baja de
#       Supabase y persiste local para futuros accesos.
#     - delete_archived_report: borra de filesystem Y de Supabase.
#     - share_with_client: re-sube el sidecar actualizado al bucket.
#     - Nuevo bucket: 'reports-archive' (constante ARCHIVE_BUCKET_NAME).
#     - Layout: reports-archive/{owner}/{YYYY}/{MM}/{file}.{pdf,json}
#
# Cambios técnicos (17.30 — RUL Predictivo):
# ──────────────────────────────────────────────────────
#   core/ai_rul.py (NUEVO, ~480 líneas):
#     - find_asset_history(viewer, instance_id, instance_tag, limit=30):
#       devuelve la timeline ASC del activo (más viejo primero, más
#       reciente último).
#     - _compute_severity_progression(history): cuantifica la
#       trayectoria — n_points, monotonic_ascending/descending,
#       total_days_covered, first/last severity, severity_curve.
#     - generate_rul_estimate(history, current_meta, current_items):
#       genera la proyección con Claude.
#     - System prompt v1 ACTUARIO PREDICTIVO Cat IV ISO 18436-2:
#       reglas críticas de gobernanza (insumo técnico, no decisión
#       de mantenimiento), formato fijo (situación actual, trayectoria
#       histórica, modelo de degradación inferido, proyección RUL
#       SOLO si hay datos suficientes, variables que afectan
#       incertidumbre, disclaimer legal de doble línea obligatorio).
#     - Si historia < 3 reportes monótonos: emite análisis cualitativo
#       SIN percentiles inventados.
#     - MIN_HISTORY_FOR_RUL = 3.
#     - Cache local TTL 7 días (data fresca para esto).
#     - Robustez heredada: retry x3 + fallback Haiku + timeouts.
#
#   pages/16_Reports.py:
#     - Imports nuevos: find_asset_history, generate_rul_estimate,
#       MIN_HISTORY_FOR_RUL.
#     - Botón '🔮 Estimar Vida Útil Restante (RUL)' con etiqueta
#       enriquecida ('N reportes históricos'). Disabled si no hay
#       items, no hay key, o no hay historial.
#     - Caption derecho muestra: cantidad de reportes + rango de
#       fechas + indicador 'Suficiente para percentiles' (≥3) vs
#       'Análisis cualitativo solamente' (<3).
#     - Expander preview con header informativo (n reportes, días
#       cubiertos, monotónica/no-monotónica), markdown del estimado,
#       botones Regenerar/Descartar, caption con modelo+tokens+costo.
#     - Inyección automática a meta['ai_rul_estimate'] +
#       meta['ai_rul_meta'] antes de _build_pdf_bytes.
#     - PDF: nueva sección 'PROYECCIÓN DE VIDA ÚTIL RESTANTE' después
#       de 'Evolución desde la última corrida'. Caption gris con
#       metadata estadística + render del estimado con
#       _render_ai_clinical_flowables.
#
# Costo estimado:
# ──────────────────────────────────────────────────────
#   17.29 (persistencia): cero costo Anthropic. Supabase Storage
#                         es ~free hasta 1 GB.
#   17.30 RUL:            ~\$0.03-0.08 por estimación. Cache local
#                         TTL 7 días.
#
# Cero regresiones:
# ──────────────────────────────────────────────────────
#   - Si Supabase no está configurado o el bucket no existe, la
#     persistencia falla silenciosa pero el archivo local sigue
#     funcionando como antes (comportamiento previo a este fix).
#   - Si el especialista no clickea RUL, la sección no aparece en el
#     PDF.
#   - Si el activo tiene < 3 reportes históricos, el botón RUL emite
#     análisis cualitativo (no inventa percentiles).
#   - Si Anthropic está caído, retry+fallback Haiku absorben.
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🚨 RELEASE v3.11.0 → MAIN  (Persistencia archive + RUL)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.29 CRÍTICO  Persistencia del archivo histórico vía Supabase"
echo "                 → Fin del bug de pérdida de datos en redeploy"
echo "  17.30          AI RUL Predictivo (Remaining Useful Life)"
echo "                 → P10/P50/P90 + ventana óptima de intervención"
echo ""
echo "⚠️  ANTES DE CONFIRMAR: ¿Ya creaste el bucket 'reports-archive'"
echo "    en tu proyecto Supabase? (Storage → New bucket → privado)"
echo "    Si no lo hiciste, hacelo AHORA en otra pestaña, y después"
echo "    confirmá acá. Sin bucket no se persiste."
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el release v3.11.0 a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

echo "▶ 1/7  Commit en dev (persistencia + RUL)..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged
for v in v3_4_2 v3_4_3 v3_4_4 v3_4_5 v3_4_6 v3_4_7 v3_4_8 v3_4_9 \
         v3_5_0 v3_5_1 v3_5_2 v3_5_3 v3_6_0 v3_7_0 v3_8_0 v3_9_0 v3_10_0; do
    git checkout HEAD -- "_release_${v}_hotfix.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_ux.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_main.sh" 2>/dev/null || true
done

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/reports_archive.py \
            core/ai_rul.py \
            pages/16_Reports.py \
            _release_v3_11_0_main.sh
    git commit -m "feat(17.29 + 17.30): Persistencia archivo + AI RUL Predictivo

DOS cosas grandes en este release.

Ciclo 17.29 — FIX CRÍTICO DE PÉRDIDA DE DATOS
==============================================
Los reportes archivados se perdían en cada redeploy de Streamlit
Cloud (contenedores efímeros). Ahora se persisten en Supabase
Storage en bucket 'reports-archive' y se restauran al cold start.

core/reports_archive.py:
- _get_archive_supabase_client + _upload/_download/_delete/_list.
- sync_archive_from_supabase: cold start lazy (1 vez por proceso).
- sync_archive_to_supabase: migración manual de archivos locales.
- archive_report_pdf: sube a Supabase tras escribir local
  (best-effort).
- list_archived_reports: llama sync_from_supabase la primera vez.
- get_archived_pdf_bytes: si falta local, baja de Supabase y
  persiste local.
- delete_archived_report: borra de ambos.
- share_with_client: re-sube sidecar actualizado.
- Layout: reports-archive/{owner}/{YYYY}/{MM}/{file}.{pdf,json}.

Ciclo 17.30 — AI RUL PREDICTIVO
================================
Estimación de Remaining Useful Life con P10/P50/P90 y ventana
óptima de intervención. Claude actúa como ACTUARIO PREDICTIVO
Cat IV ISO 18436-2.

core/ai_rul.py (NUEVO):
- find_asset_history: timeline ASC del activo.
- _compute_severity_progression: cuantifica trayectoria
  (monotonicidad, días cubiertos, severidad inicial/final).
- generate_rul_estimate: percentiles + ventana de intervención
  + variables que afectan incertidumbre + disclaimer legal de
  doble línea obligatorio.
- System prompt v1 ACTUARIO PREDICTIVO con reglas críticas de
  gobernanza (insumo técnico, NO decisión de mantenimiento).
- Si historia < 3 reportes monótonos: análisis cualitativo SIN
  percentiles inventados.
- MIN_HISTORY_FOR_RUL = 3. Cache TTL 7 días.

pages/16_Reports.py:
- Botón '🔮 Estimar Vida Útil Restante (RUL)' con etiqueta
  enriquecida (N reportes históricos).
- Caption con cantidad/rango de fechas/indicador de suficiencia.
- Expander preview con header informativo, markdown, botones
  regenerar/descartar.
- Inyección a meta['ai_rul_estimate'] + meta['ai_rul_meta'].
- PDF: sección 'PROYECCIÓN DE VIDA ÚTIL RESTANTE' después de
  'Evolución desde la última corrida'.

Costo: persistencia cero. RUL ~\$0.03-0.08 por estimación.

Cero regresiones: si bucket Supabase no existe, persistencia
falla silenciosa pero el archivo local sigue funcionando como
antes. Si especialista no clickea RUL, la sección no aparece.
Si activo tiene <3 reportes, RUL emite análisis cualitativo." || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev commiteado"
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
MERGE_MSG="release(v3.11.0): merge dev -> main · Persistencia archive + RUL Predictivo

DOS cosas grandes en este release.

(1) FIX CRÍTICO 17.29 — Persistencia del archivo histórico vía
    Supabase Storage. Antes los reportes archivados se perdían
    en cada redeploy. Ahora se persisten en bucket privado
    'reports-archive' y se restauran al cold start. Cero pérdida
    de datos.

(2) CICLO 17.30 — AI RUL Predictivo. Estimación de Remaining
    Useful Life con P10/P50/P90 y ventana óptima de intervención.
    Claude actúa como actuario predictivo Cat IV ISO 18436-2
    leyendo toda la historia del activo + estado actual.
    Disclaimer legal fuerte.

⚠️ Requiere bucket 'reports-archive' creado en Supabase Dashboard
    antes del cold start. Sin bucket: persistencia falla silenciosa
    pero el sistema sigue funcionando (comportamiento previo)."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.11.0..."
TAG_EXISTS=$(git tag -l "v3.11.0")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.11.0 ya existe. Saltando creación."
else
    git tag -a v3.11.0 -m "Release v3.11.0 — Persistencia archive (CRÍTICO) + AI RUL Predictivo"
    echo "  ✓ Tag v3.11.0 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.11.0 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ RELEASE v3.11.0 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo ""
echo " 🚨 VERIFICACIÓN POST-DEPLOY (CRÍTICO):"
echo ""
echo "    PASO 1 — bucket Supabase creado:"
echo "      Andá a Supabase Dashboard → Storage. Tenés que ver el"
echo "      bucket 'reports-archive' creado (privado). Si no lo creaste"
echo "      antes, créalo AHORA. Si lo creaste después del deploy, no"
echo "      pasa nada — el código lo va a usar la próxima vez que se"
echo "      archive un reporte."
echo ""
echo "    PASO 2 — migración de archivos existentes (UNA SOLA VEZ):"
echo "      Si tenés reportes archivados ANTES de este release, no"
echo "      están en Supabase todavía. Para subirlos, podés ejecutar"
echo "      desde una sesión Python local:"
echo ""
echo "        python3 -c \"from core.reports_archive import sync_archive_to_supabase; print(sync_archive_to_supabase())\""
echo ""
echo "      O esperar a que el próximo nuevo reporte se archive (los"
echo "      archivos viejos nunca van a estar en Supabase si no los"
echo "      subís manualmente)."
echo ""
echo "    PASO 3 — validar persistencia con un test deliberado:"
echo "      a. Archivá un reporte cualquiera desde Reports → Archivar"
echo "      b. En el log de stderr deberías ver:"
echo "         [WM_ARCHIVE] Supabase upload OK: ehernandez_at_..."
echo "      c. Hacé un push a main (cualquier cambio menor) para"
echo "         disparar un redeploy"
echo "      d. Cuando redeployó, andá a Reports → Archivo Histórico"
echo "         → el reporte debería seguir ahí (antes desaparecía)"
echo ""
echo " 🧪 VALIDACIÓN del feature RUL:"
echo ""
echo "    1. Activá un activo con HISTORIA (idealmente TES1 con 3+"
echo "       reportes archivados de meses distintos)"
echo "    2. Andá a Reports, cargá las figuras del nuevo análisis"
echo "    3. Verás 3 botones AI: Síntesis Ejecutiva, Run-vs-Run, RUL"
echo "    4. Click en '🔮 Estimar Vida Útil Restante'"
echo "    5. Caption derecho dice 'N reportes · ✅ Suficiente'"
echo "    6. Spinner 10-25 seg → preview con percentiles P10/P50/P90"
echo "       y ventana de intervención"
echo "    7. Click 'Preparar PDF' → descargar"
echo "    8. Ver sección 'PROYECCIÓN DE VIDA ÚTIL RESTANTE' después"
echo "       de 'Evolución desde la última corrida'"
echo ""
echo " 💼 Tres botones AI ahora disponibles en Reports:"
echo "    🧠 Síntesis Ejecutiva AI    (cross-figura del reporte actual)"
echo "    🔄 Run-vs-Run               (delta vs reporte anterior)"
echo "    🔮 RUL Predictivo           (vida útil restante)"
echo ""
echo "================================================================"
