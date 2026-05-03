#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.12 → DEV: Home Nivel 3
#                  (Health Score + Omnibox + Modo turno)
# =============================================================
# Tercer y último nivel del rediseño del Home. Las tres piezas
# pedidas:
#
#   ★ HEALTH SCORE 0-100 por activo
#     - Algoritmo determinístico en core/health_score.py
#     - 100 base; penalidades por norma faltante (-35), baseline
#       ausente (-20), sin docs en Vault (-15), baseline >90 días
#       (-10), override sin justificar (-10); bonus por reporte
#       reciente (+5) y combo norma+baseline+reporte (+10).
#     - 4 bandas: ÓPTIMO (90+), BUENO (70-89), ATENCIÓN (40-69),
#       CRÍTICO (<40)
#     - Gauge SVG semicircular tipo Bently embebido en cada
#       asset card del Home (sin Plotly, sin libs externas)
#     - Breakdown explicable por activo (qué se restó y por qué)
#
#   ★ OMNIBOX global (búsqueda fuzzy)
#     - Caja arriba del KPI band, placeholder con sugerencias
#     - Indexa contra: instancias del Vault (tag/asset_class/
#       location/profile/id), drafts de reportes (filename), y
#       las 26 normas ISO/API (código + nombre + applies_to)
#     - Ranking por substring + tokens + match exacto
#     - Resultados con icono por tipo + pill de kind + botón
#       "Abrir →" que hace switch_page directo y setea la
#       instancia activa en session_state
#     - Sin atajo Cmd+K verdadero (Streamlit no lo soporta nativo,
#       requeriría componente custom). Pero el omnibox queda muy
#       visible y a un foco del usuario.
#
#   ★ MODO TURNO automático
#     - Detecta hora local: 22:00–06:00 = "Turno noche"
#     - Si turno noche, el hero vira de gradient verde-azul a
#       rojo-púrpura (señal visual "guardia nocturna")
#     - Accent del borde izquierdo del hero pasa de #10b981
#       (verde watermelon) a #ef4444 (rojo)
#     - Sutil pero claro
#
# Cambios técnicos:
#
# (NUEVO) core/health_score.py
#   - HealthScore dataclass {score, band, color, breakdown,
#     one_liner}
#   - compute_health_score(instance_data) - cálculo principal
#   - compute_health_score_for_instance_id(id) - wrapper que
#     carga la instancia
#   - render_score_gauge(score, color, size) - SVG inline
#     semicircular con número central
#   - render_score_pill(...) - alternativa compacta para listas
#
# (NUEVO) core/omnibox_search.py
#   - OmniHit dataclass
#   - omnibox_search(query, limit) - busca contra 3 fuentes
#   - kind_label / kind_color helpers para UI
#   - Sin dependencias externas, ranking heurístico simple
#
# (MODIFICADO) pages/_landing.py
#   - Nuevo bloque CSS para .wmh-omni-* y .wmh-gauge-*
#   - Nuevo bloque CSS DINÁMICO inyectado f-string aparte
#     que pinta el background del hero según turno
#   - Sección OMNIBOX entre hero y KPI band con st.text_input
#     + render de hits con st.button por hit (switch_page)
#   - Asset card REWRITE: ahora muestra el gauge 0-100 al
#     centro + band label coloreada + one-liner explicativo
#     (en lugar del simple dot)
#   - Lifecycle: detecta _is_night al cargar, define _HERO_BG
#     y _ACCENT_COLOR variables que el CSS dinámico aplica
#
# Limitaciones conocidas:
#   - Cmd+K real no funciona (requiere componente JS custom).
#     Si querés el atajo verdadero en 17.13, agregamos
#     `streamlit-shortcuts` a requirements.txt.
#   - Severity "danger" del KPI band sigue siendo placeholder
#     hasta persistir last_executive_severity en metadata
#     desde el PDF generator (planeado para 17.13).
#   - Sparklines de severidad siguen placeholder.
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
echo "▶ Stageando 17.12..."
git add core/health_score.py
git add core/omnibox_search.py
git add pages/_landing.py
git add _publish_ciclo17_12_dev.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando 17.12..."
    git commit -m "feat(home): Nivel 3 - Health Score 0-100 + Omnibox + Modo turno (17.12)

NUEVO core/health_score.py (deterministic, sin Streamlit/Plotly):
- HealthScore dataclass {score, band, color, breakdown, one_liner}
- Algoritmo 0-100 con penalidades fuertes por config incompleta:
  base 100, -35 sin norma, -20 sin baseline, -15 sin docs,
  -10 baseline viejo, -10 override sin justificar, +5 reporte
  reciente, +10 combo norma+baseline+reporte
- 4 bandas: OPTIMO 90+ (verde), BUENO 70-89 (lima), ATENCION
  40-69 (ambar), CRITICO <40 (rojo)
- render_score_gauge(): SVG semicircular tipo Bently inline
- render_score_pill(): variante compacta

NUEVO core/omnibox_search.py:
- OmniHit dataclass + omnibox_search(query, limit)
- Indexa instancias (tag/class/location/profile/id), drafts de
  reportes, y 26 normas ISO/API
- Ranking heuristico: prefijo +50, substring +30, tokens *12,
  match exacto +80
- kind_color / kind_label helpers

MODIFICADO pages/_landing.py:
- Sección OMNIBOX entre hero y KPI band con text_input +
  render de hits live (boton 'Abrir' por hit -> switch_page)
- Asset card ahora con HEALTH SCORE GAUGE 0-100 al centro
  (SVG semicircular) + band label coloreada + one-liner
- MODO TURNO: detecta hora actual, si 22-06 vira hero a
  gradient rojo/purpura (en vez de verde/azul). Accent del
  border-left pasa de verde a rojo. Sin invertir todo el
  tema (eso requeriria theme.toml).
- CSS adicional .wmh-omni-* y .wmh-gauge-*
- CSS DINAMICO en bloque f-string aparte para no escapar
  el bloque grande estatico

Limitaciones: Cmd+K real requiere streamlit-shortcuts (opcional
en 17.13). Severity 'danger' aun placeholder hasta persistir
last_executive_severity desde el PDF generator.

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
echo " ✓ Ciclo 17.12 pusheado a DEV (main intacto)"
echo "================================================================"
echo ""
echo " Para probar:"
echo "  1. Recargar https://wm-home-final-2026.streamlit.app/landing"
echo "  2. Ver omnibox arriba — tipear 'TES1', '20816', 'balanceo'"
echo "     'API 684', 'rotor', '684' — todos deberían dar resultados"
echo "  3. Ver gauges semicirculares en cada asset card"
echo "     - Si tu C-200C tiene norma asignada y baseline → score >90"
echo "     - Si sólo tiene tag y nada más → score ~30 (CRÍTICO)"
echo "  4. Si entrás entre 22:00 y 06:00 (server time o local) →"
echo "     hero vira a tonos rojo/púrpura. Accent del border-left"
echo "     pasa de verde a rojo (modo guardia nocturna)."
echo ""
echo " Ciclo 17.13 (siguiente) traería:"
echo "  - Briefing diario PDF schedulable (envío automático 7am)"
echo "  - Persistencia de severidad ejecutiva real (no heurística)"
echo "  - Cmd+K real con streamlit-shortcuts (opcional)"
echo "================================================================"
