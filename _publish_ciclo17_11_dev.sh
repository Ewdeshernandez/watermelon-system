#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.11 → DEV: Home rediseño Nivel 1+2
#                                  (de brochure → HMI premium)
# =============================================================
# Refit total del Home/landing. Pasamos de "página de brochure
# con AI stock-art y título medio invisible" a un HMI tipo
# centro de control con datos reales del Vault.
#
# Lo que cambia visualmente al usuario:
#
#   ► HERO COMPACTO (ya no ocupa media pantalla)
#     - Saludo personalizado por hora ("Buenos días/tardes/noches,
#       Ewdes") + nombre del usuario logueado
#     - Reloj grande HH:MM + turno (Mañana/Tarde/Noche con emoji)
#     - Fecha legible en español (domingo · 03 may 2026)
#     - Línea de status que prioriza la peor noticia ("X críticos",
#       "Y en atención" o "Z monitoreados sin alertas")
#     - Título con gradient + alto contraste (el viejo era
#       semi-transparente sobre navy → invisible)
#     - Stock-art de la turbina ELIMINADA del hero
#
#   ► KPI BAND con 4 cards
#     - Total activos · Críticos · En atención · Saludables
#     - Cada card con sparkline SVG de últimos 7 días
#     - Border-top de color según severidad (azul/rojo/ámbar/verde)
#
#   ► QUICK ACTIONS strip
#     - 5 botones grandes hacia las páginas más usadas:
#       Cargar CSV · Trends · Machinery Library · Diagnostics ·
#       Reports
#     - Click → switch_page directo (cero clics extra)
#
#   ► ACTIVE ASSETS GRID (2/3 width)
#     - Card por instancia con dot health 🟢🟡🔴⚪
#     - Tag, asset class, ubicación, n_documentos, "hace X tiempo"
#     - Botón Trends (carga la instancia activa) y Editar
#     - Hover effect (lift + shadow)
#
#   ► ACTIVITY FEED (1/3 width)
#     - Últimas 10 acciones combinando edits de metadata + drafts
#       de reportes + report_state.json
#     - Cada evento con icono, título, subtítulo monoespaciado,
#       "hace X tiempo"
#
#   ► SCADA STATUS BAR (footer dark)
#     - ENV pill (PRODUCTION/DEVELOPMENT con dot de color)
#     - Versión + commit corto
#     - Vault status + n activos + última sync
#     - Sesión del usuario activo
#     - Estilo monoespaciado tipo HMI industrial
#
# Cambios técnicos:
#
# (NUEVO) core/home_metrics.py
#   - get_personalized_greeting(name, now) → saludo + turno + fecha
#   - compute_fleet_status() → total + by_severity + List[InstanceHealth]
#     con heurística inicial 17.11:
#       healthy = norma asignada + ≥1 doc + last_balance_date
#       warning = norma asignada pero falta info
#       danger  = (reservado para 17.12 con persistencia ejecutiva)
#       unknown = sin norma asignada
#   - list_recent_activity(limit) → ActivityEvent[] combinando
#     edits a instances/*/metadata.json + report_drafts/*.json +
#     report_state.json. Ordenado por mtime desc.
#   - activity_sparkline(days) → conteo eventos por día
#   - severity_sparkline(severity, days) → trend approx por severity
#     (placeholder hasta tener histórico real en 17.12)
#   - get_system_health() → env, version, commit, vault_n,
#     vault_status, last_data_age (para SCADA bar)
#   - Sin dependencia de Streamlit (deterministicamente testeable)
#
# (REWRITE) pages/_landing.py
#   - 425 líneas (vs 130 originales)
#   - CSS embebido con prefijo .wmh- para no chocar con tema global
#   - Sparkline SVG inline (sin Plotly para no agregar peso)
#   - Switch_page directo desde quick actions y asset cards
#   - Responsive (media query <900px ajusta hero)
#
# IMPORTANTE — limitaciones conocidas (a refinar en 17.12):
#   1. La severidad "danger" todavía no se calcula desde reportes
#      históricos. La heurística actual da healthy/warning/unknown.
#      Para danger necesitamos persistir last_executive_severity
#      en metadata cuando se genera el PDF.
#   2. Sparklines de severidad usan placeholder determinístico.
#      En 17.12 será trend real desde histórico de severidades.
#   3. No hay omnibox Cmd+K todavía (Nivel 3, ciclo siguiente).
#   4. No hay briefing diario PDF schedulable (Nivel 3).
#
# Solo push a DEV. Cliente revisa y decide main + Nivel 3.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo ""
echo "▶ Stageando 17.11..."
git add core/home_metrics.py
git add pages/_landing.py
git add _publish_ciclo17_11_dev.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando 17.11..."
    git commit -m "feat(home): rediseno Nivel 1+2 - HMI premium con fleet status (17.11)

De brochure a HMI de centro de control.

NUEVO core/home_metrics.py (deterministicamente testeable, sin
Streamlit deps):
- get_personalized_greeting: saludo + turno + fecha en espanol
- compute_fleet_status: total + by_severity + InstanceHealth[]
  con heuristica healthy/warning/unknown segun norma+docs+balance
- list_recent_activity: combina edits metadata + report drafts
  + report_state.json ordenado por mtime
- activity_sparkline / severity_sparkline para mini-charts 7d
- get_system_health: env + version + commit + vault status

REWRITE pages/_landing.py:
- HERO compacto con saludo personalizado dinamico, reloj HH:MM,
  turno (Manana/Tarde/Noche), fecha en espanol, status pill que
  prioriza peor noticia
- Stock-art de turbina ELIMINADA
- KPI BAND: 4 cards (Total/Criticos/Atencion/Saludables) con
  sparkline SVG inline 7d
- QUICK ACTIONS strip: 5 botones grandes a Load/Trends/Library/
  Diagnostics/Reports
- ACTIVE ASSETS GRID 2/3 width: card por instancia con dot
  health, tag, asset class, ubicacion, n_docs, age, botones
  Trends y Editar
- ACTIVITY FEED 1/3 width: ultimas 10 acciones del usuario
- SCADA STATUS BAR footer: env pill, version+commit, vault
  status, last sync, sesion (estilo monoespaciado)

Limitaciones conocidas (a refinar en 17.12):
- severity 'danger' aun no se calcula desde reportes historicos
  (heuristica actual: healthy/warning/unknown)
- sparklines de severidad usan placeholder
- sin omnibox Cmd+K (Nivel 3, proximo ciclo)
- sin briefing diario PDF schedulable (Nivel 3)

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
echo " ✓ Ciclo 17.11 pusheado a DEV (main intacto)"
echo "================================================================"
echo ""
echo " Para probar en https://wm-home-final-2026.streamlit.app/landing :"
echo "  1. El stock-art desaparece. En su lugar: hero compacto con"
echo "     saludo personalizado y reloj."
echo "  2. Banda de 4 KPI cards (Total / Críticos / Atención / OK)"
echo "     con sparklines de actividad de los últimos 7 días."
echo "  3. Strip de 5 quick actions hacia las páginas más usadas."
echo "  4. Grid de cards por cada instancia del Vault con dot de"
echo "     salud y botones Trends + Editar (que cargan la instancia"
echo "     directamente)."
echo "  5. Feed lateral derecho con tu actividad reciente."
echo "  6. Footer SCADA con env + version + vault + sync."
echo ""
echo " Si te gusta → siguiente ciclo (17.12) Nivel 3:"
echo "  - omnibox Cmd+K (busqueda global)"
echo "  - briefing diario PDF schedulable"
echo "  - severidad real desde reportes historicos (no heuristica)"
echo "  - mini-mapa del site con dots animados sobre activos"
echo "================================================================"
