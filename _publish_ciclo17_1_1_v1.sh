#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.1.1 → DEV: Polar multi-snapshot overlay
# =============================================================
# Cambia el selector "Comparar contra corrida anterior"
# (single-select) a "Corridas a superponer en el polar"
# (multi-select). El usuario puede elegir:
#
#   - 0 corridas → solo se muestra la corrida actual
#   - 1 corrida  → comparativo simple (igual que antes)
#   - N corridas → superposición histórica con gradiente
#                  cronológico
#
# Cada snapshot anterior se dibuja con marker ghost + linea
# conectora dotted al actual, en color según posición
# cronológica:
#
#   Más viejo  → azul claro (#7dd3fc)
#   Medio      → ámbar       (#f59e0b)
#   Más reciente → rojo      (#dc2626)
#
# Asi el ingeniero ve de un vistazo la trayectoria del balance
# del rotor a lo largo del tiempo: si el cluster de markers
# anteriores está concentrado y el actual se aleja, eso indica
# cambio reciente. Si los markers forman una línea progresiva,
# eso es degradación gradual.
#
# Cambios en pages/06_Polar_Plot.py:
#
# (1) Sidebar 'Histórico Polar' → selectbox cambia a multiselect
#     con el primer snapshot no-actual preseleccionado por
#     default. Caption indica cuántas corridas se superponen.
#     Persistencia en wm_polar_compare_snapshot_ids (lista).
#
# (2) Comparativo Polar inline → tabla con UNA fila por
#     (sensor × snapshot). Caption se adapta a 1 vs N corridas.
#
# (3) build_polar_figure acepta nuevo param prev_snapshots
#     (lista de dicts {amp, phase, label, op_speed, timestamp})
#     que prevalece sobre el legacy single-snapshot. Iterar
#     ordenado por timestamp asc y dibujar marker ghost + linea
#     conectora en color por posición cronológica.
#
# (4) render_polar_panel itera todos los snapshot ids elegidos,
#     resuelve el sensor matched al panel, y arma la lista de
#     prev_snapshots para pasar a build_polar_figure.
#
# Compatibilidad: el primer snapshot de la lista sigue
# alimentando comparison_narrative del PDF para no romper el
# texto del reporte (si querés narrativa de los N en el PDF lo
# hacemos en proxima iteracion).
#
# Ejecutar:
#   bash _publish_ciclo17_1_1_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/06_Polar_Plot.py
git status --short | head

git commit -m "feat(polar): multi-snapshot overlay con gradiente cronologico (Ciclo 17.1.1)

Cambia el selector single-select del Polar history a multiselect.
Ahora el usuario puede:
- 0 corridas: ver solo la actual
- 1 corrida: comparativo simple (legacy)
- N corridas: superposicion historica con gradiente cronologico
  (mas viejo = azul claro, medio = ambar, mas reciente = rojo)

Cambios:
(1) Sidebar selectbox -> multiselect, persistencia en
wm_polar_compare_snapshot_ids (lista). Default = primer snap
no-actual.
(2) Tabla comparativa inline ahora con UNA fila por
(sensor x snapshot). Caption se adapta a 1 vs N corridas.
(3) build_polar_figure acepta prev_snapshots (lista) que
prevalece sobre single legacy. Itera ordenado por timestamp asc,
dibuja marker ghost + linea conectora en color por posicion
cronologica via interpolacion azul-ambar-rojo.
(4) render_polar_panel itera todos los snap ids elegidos y
arma lista de prev_snapshots por sensor matched.

Backward-compat con narrativa PDF: el primer snapshot de la
lista sigue alimentando comparison_narrative." || echo "Nothing to commit"

git push origin dev

echo ""
echo "✓ Ciclo 17.1.1 pusheado a dev. Refrescá el browser."
echo ""
echo "Para verlo: en la sidebar del Polar Plot ahora hay un"
echo "multiselect 'Corridas a superponer en el polar'. Elegí 0,"
echo "1, o varias. Cada una se ve sobre el polar con color"
echo "cronologico (azul claro = mas vieja, rojo = mas reciente)."
