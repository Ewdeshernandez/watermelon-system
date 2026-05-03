#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.13 → DEV: Briefing diario + Severidad
#                                  ejecutiva real + Cmd+K
# =============================================================
# Última pieza del Nivel 3. Convierte el sistema de "dashboard
# administrativo" a "monitoreo continuo automatizado".
#
# Las 3 piezas:
#
#   ★ BRIEFING DIARIO PDF (1 página)
#     - core/briefing.py: snapshot del fleet + comparativo vs ayer
#       + top 3 atención + vencimientos + próximos pasos
#     - PDF A4 generado con reportlab (ya en requirements.txt)
#     - Snapshot histórico en data/briefings/snapshots/YYYY-MM-DD.json
#       para poder computar deltas día a día
#     - Botón "📰 Briefing día" en el quick action strip del Home
#       que genera el PDF al instante y muestra download_button
#     - Script standalone scripts/generate_daily_briefing.py para
#       cron / GitHub Actions con soporte SMTP opcional para envío
#       automático por email
#
#   ★ SEVERIDAD EJECUTIVA REAL (no más heurística)
#     - Instance dataclass extendido con 3 campos:
#         last_executive_severity   ("CRÍTICA" / "ATENCIÓN" / etc.)
#         last_executive_summary    (frase ejecutiva)
#         last_report_date          (ISO timestamp)
#     - Helper update_instance_executive_severity(id, severity, summary)
#     - Cableado en pages/16_Reports.py: después del botón
#       "Preparar PDF", recomputa severity_live via _global_severity()
#       + _extract_findings_from_items(items) y persiste en metadata
#     - core/home_metrics.py mapea las 5 etiquetas del PDF a las
#       4 bandas del Home (CRÍTICA/ACCIÓN REQUERIDA → danger,
#       ATENCIÓN/VIGILANCIA → warning, CONDICIÓN ACEPTABLE → healthy)
#     - Si la instancia tiene severity ejecutiva persistida, el Home
#       USA ESE valor (estado real). Si no, cae a la heurística previa
#       de configuración (norma + baseline + docs)
#
#   ★ Cmd+K REAL para enfocar el omnibox
#     - JS inline en _landing.py (sin streamlit-shortcuts dep)
#     - Escucha keydown en window + window.parent (Streamlit iframe)
#     - Cmd+K en Mac, Ctrl+K en Windows/Linux
#     - Selector estable por placeholder del input (empieza con 🔍)
#     - Re-engancha tras cada rerun de Streamlit
#     - Hint visible en el omnibox: "Cmd+K (o Ctrl+K) para enfocar"
#
# Cambios técnicos:
#
# (NUEVO) core/briefing.py
#   - compute_fleet_snapshot()       — captura estado serializable
#   - save_daily_snapshot()          — JSON en snapshots/YYYY-MM-DD
#   - load_yesterday_snapshot()      — busca el snapshot más reciente
#                                      anterior a hoy
#   - compute_deltas_vs_yesterday()  — detecta cambios de banda +
#                                      score promedio
#   - compute_upcoming_items()       — sin norma, sin baseline,
#                                      baseline >90d
#   - suggest_next_actions()         — pasos sugeridos según severity
#   - generate_briefing_pdf()        — PDF reportlab con layout 1 pág
#   - generate_and_save_briefing()   — wrapper end-to-end
#
# (NUEVO) scripts/generate_daily_briefing.py
#   - Standalone CLI para cron/GitHub Actions
#   - Args: --output, --email, --smtp-host/port/user/pass, --quiet
#   - Vars de entorno: WM_SMTP_HOST/PORT/USER/PASS
#   - Crontab ejemplo:
#       30 6 * * * cd /path/to/WatermelonSystem && \\
#                  python3 scripts/generate_daily_briefing.py
#
# (MODIFICADO) core/instance_state.py
#   - Instance: 3 campos nuevos (last_executive_severity/summary/date)
#   - from_dict: parse retro-compatible
#   - allowed set en update_instance_header extendido
#   - update_instance_executive_severity(id, severity, summary, date=None)
#     helper específico para cableado desde el PDF generator
#
# (MODIFICADO) core/home_metrics.py
#   - _EXEC_SEVERITY_TO_BAND: mapeo 5 → 4 bandas
#   - _norm_no_accents: normaliza para tolerar mayús/acentos
#   - _heuristic_severity: si hay last_executive_severity, USA ESE
#     valor (no la heurística de config). Si no, cae al cálculo previo.
#
# (MODIFICADO) pages/16_Reports.py
#   - Después del botón "Preparar PDF", al success path:
#     recomputa severity_live + persiste en metadata via
#     update_instance_executive_severity(). Try/except para no
#     romper el flujo del PDF si falla.
#
# (MODIFICADO) pages/_landing.py
#   - Quick action strip: 5 → 6 botones, agregado "📰 Briefing día"
#   - Botón briefing genera PDF al instante + muestra download_button
#   - JS inline para Cmd+K que enfoca el omnibox
#   - Hint del omnibox actualizado para mencionar Cmd+K
#
# Resultado de negocio:
#   El cliente final pasa de "tengo que entrar al sistema cada tanto
#   para ver cómo va" a "todas las mañanas a las 6:30 AM me llega
#   un briefing en mi correo con el estado del día". Eso justifica
#   el contrato mensual de monitoreo.
#
# Solo push a DEV.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo ""
echo "▶ Stageando 17.13..."
git add core/briefing.py
git add core/instance_state.py
git add core/home_metrics.py
git add pages/16_Reports.py
git add pages/_landing.py
git add scripts/generate_daily_briefing.py
git add _publish_ciclo17_13_dev.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando 17.13..."
    git commit -m "feat(briefing+severity+hotkey): Nivel 3 final (17.13)

Tres piezas que cierran el rediseno del Home:

NUEVO core/briefing.py:
- generate_briefing_pdf(): PDF A4 1 pagina con header + KPIs +
  top 3 atencion + cambios vs ayer + vencimientos + proximos pasos
- compute_fleet_snapshot() / save_daily_snapshot() persisten
  estado en data/briefings/snapshots/YYYY-MM-DD.json para deltas
- compute_deltas_vs_yesterday() detecta cambios de banda + score
- generate_and_save_briefing() wrapper end-to-end

NUEVO scripts/generate_daily_briefing.py:
- Standalone CLI para cron/GitHub Actions
- Soporte SMTP opcional con --email + --smtp-* (o vars WM_SMTP_*)
- Genera PDF y opcionalmente lo envia por email

MODIFICADO core/instance_state.py:
- Instance dataclass +3 campos: last_executive_severity,
  last_executive_summary, last_report_date
- update_instance_executive_severity() helper

MODIFICADO core/home_metrics.py:
- _EXEC_SEVERITY_TO_BAND mapea 5 etiquetas del PDF (CRITICA,
  ACCION REQUERIDA, ATENCION, VIGILANCIA, CONDICION ACEPTABLE) a
  las 4 bandas del Home
- _heuristic_severity prioriza last_executive_severity persistido
  sobre la heuristica de config

MODIFICADO pages/16_Reports.py:
- Despues de Preparar PDF, recomputa severity_live + persiste en
  metadata del activo. Try/except para no romper el flujo del PDF.

MODIFICADO pages/_landing.py:
- Quick action strip: +6to boton 'Briefing dia' que genera y
  ofrece download del PDF
- JS inline para Cmd+K (Mac) / Ctrl+K (Win/Linux) que enfoca el
  omnibox. Sin streamlit-shortcuts dependency.
- Re-engancha tras cada rerun

Beneficio de negocio: el cliente pasa de 'entrar al sistema' a
'recibir briefing diario por correo'. Justifica contrato mensual.

Solo push a DEV." || echo "  (sin cambios)"
else
    echo "  (no hay cambios)"
fi

echo ""
echo "▶ Reconciliando..."
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Rebase falló."; exit 1; }

echo "▶ Push dev..."
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.13 pusheado a DEV (main intacto)"
echo "================================================================"
echo ""
echo " Para probar:"
echo "  1. Recargar Home → ver el 6to boton 📰 Briefing dia."
echo "     Click → genera PDF al instante → boton de descarga."
echo "  2. Abrir el PDF: deberia tener header con fecha, KPIs de"
echo "     flota, top 3 activos en atencion, cambios vs ayer (si hay"
echo "     snapshot del dia anterior), vencimientos, proximos pasos."
echo "  3. Probar Cmd+K (Mac) o Ctrl+K (Win/Linux) — el cursor"
echo "     debe saltar al omnibox sin importar donde estes en el Home."
echo "  4. Generar un PDF de Reports normal:"
echo "     → ir a Machinery Library, asignar instancia activa"
echo "     → ir a Reports, agregar items, click 'Preparar PDF'"
echo "     → volver al Home → el dot de ese activo debe reflejar"
echo "       AHORA la severidad real del PDF (no heuristica)."
echo ""
echo " Para automatizar el briefing por correo:"
echo "  Crontab (todos los dias a las 6:30 AM hora servidor):"
echo "    30 6 * * * cd ~/Documents/WatermelonSystem && \\"
echo "               python3 scripts/generate_daily_briefing.py \\"
echo "                 --email cliente@empresa.com \\"
echo "                 --smtp-host smtp.gmail.com --smtp-port 587 \\"
echo "                 --smtp-user TU_USER --smtp-pass APP_PASSWORD"
echo ""
echo " Si todo OK → ciclo siguiente decidir publish a main con"
echo " tag v3.2.0 (17.10 + 17.11 + 17.12 + 17.13 acumulados)."
echo "================================================================"
