#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.10 → DEV: Catálogo ISO/API expandido
#                                  (26 normas, +API 684, +bal.)
# =============================================================
# Expansión masiva del catálogo de iso_thresholds.py de 7 a 26
# normas. Incluye específicamente lo que el cliente pidió como
# "vital":
#
#   ★ API 684 — Rotordynamic Tutorial (Q factor / AF, separation
#               margins, log decrement, vibración eje en UR test).
#               8 clases. Es la referencia del análisis lateral.
#
#   ★ ISO 21940-11 — Balanceo de rotores RÍGIDOS. Tabla completa
#                     de G-grades de G 0.4 (spindles precisión)
#                     hasta G 4000 (largest installations).
#                     11 clases.
#
#   ★ ISO 21940-12 — Balanceo de rotores FLEXIBLES (Class A/B/
#                     C/D + at-speed test).
#
# Otras normas agregadas en este ciclo:
#
#   Vibración (carcasa):
#   - ISO 20816-5 (hidroeléctricas: Francis/Kaplan/Pelton)
#   - ISO 10816-7 (bombas rotodinámicas — más específica que -3)
#   - ISO 14694   (ventiladores industriales BV-1..5)
#   - API 619     (compresores rotativos PD: tornillo, lóbulos)
#   - API 617     (compresores axiales y centrífugos)
#   - API 612     (turbinas vapor especiales — más estricta -2)
#   - API 611     (turbinas vapor propósito general)
#   - API 610     (bombas centrífugas refinería)
#   - API 541     (motores grandes form-wound jaula ardilla)
#   - API 546     (generadores síncronos brushless — Brush!)
#   - IEC 60034-14(motores y generadores eléctricos)
#   - NEMA MG-1 P7(motores AC integrales norteamericanos)
#   - ANSI S2.41  (legacy USA)
#   - VDI 2056    (legacy DE — precursor ISO 2372)
#
#   Vibración eje:
#   - VDI 2059    (legacy DE — precursor ISO 7919/20816)
#
#   Couplings:
#   - API 671     (couplings propósito especial)
#
# Total: 26 normas registradas.
#
# Cambios complementarios:
#
# (P2) iso_thresholds.py:
#      - list_norm_groups() NUEVO — devuelve normas agrupadas
#        en 4 dominios (Vibración carcasa, Vibración eje,
#        Balanceo, Rotordynamics).
#      - suggest_norm_for_machine() — heurísticas extendidas
#        para hidro/Francis/Kaplan, fan/blower, motor electrico,
#        Brush, compresor centrífugo, motor diésel.
#      - suggest_class_for_machine() — clase default para todas
#        las nuevas normas.
#      - suggest_balance_grade() NUEVO — G-grade ISO 21940-11
#        según tipo de máquina (turbo→G2.5, bomba/motor→G6.3,
#        crankshaft→G100/G630, etc.).
#
# (P3) pages/00_Machinery_Library.py:
#      - Tab "Norma ISO" ahora muestra selectbox AGRUPADO con
#        prefijos por dominio (📳 VIB / 🛡️ EJE / ⚖️ BAL / 🔬 ROT)
#      - Caption con dominio + métrica + unidad cuando se
#        selecciona la norma (ayuda al usuario a saber si está
#        eligiendo vibración, balanceo o rotordynamics).
#
# IMPORTANTE — limitación conocida:
#   Para normas con metric != velocity_rms / displacement_pp
#   (ej. unbalance_grade en ISO 21940, amplification_factor en
#   API 684), el flow Trend → comparación numérica contra setpoint
#   asume que el usuario sabe qué está midiendo. El "warning" y
#   "danger" se siguen mostrando como referencia normativa en el
#   PDF, pero el chequeo Warning/Danger sólo tiene sentido
#   directamente para velocity_rms (G-grade es ω·e ≈ vel rms 1×).
#   Para API 684 son criterios de DISEÑO no de monitoreo.
#
# IMPORTANTE — flujo solicitado:
#   "primero dev, luego revisamos y vemos si nos vamos a main"
#   → Este script SÓLO pushea a dev. Main intacto.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo ""
echo "▶ Stageando 17.10..."
git add core/iso_thresholds.py
git add pages/00_Machinery_Library.py
git add _publish_ciclo17_10_dev.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando 17.10..."
    git commit -m "feat(iso-norms): catalogo expandido 26 normas + API 684 + balanceo (17.10)

Expansion masiva del catalogo industrial: 7 -> 26 normas.

CRITICAS pedidas por el cliente:
- API 684 Rotordynamic Tutorial: AF/Q factor, separation margins,
  log decrement, vibracion eje UR test (8 clases).
- ISO 21940-11: Balanceo rotores RIGIDOS, tabla G-grades completa
  G 0.4 a G 4000 (11 clases por tipo de maquinaria).
- ISO 21940-12: Balanceo rotores FLEXIBLES, classes A/B/C/D
  modal + low/high speed + at-speed test.

Otras normas agregadas:
Vibracion carcasa: ISO 20816-5 (hidro), ISO 10816-7 (bombas),
ISO 14694 (fans BV grades), API 619 (recip PD), API 617 (axiales+
centrifugos), API 612 (vapor especiales), API 611 (vapor general),
API 610 (bombas refineria), API 541 (motores grandes), API 546
(generadores sincronos brushless), IEC 60034-14 (motores
electricos), NEMA MG-1 Part 7, ANSI S2.41, VDI 2056.
Vibracion eje: VDI 2059.
Couplings: API 671.

P2 - iso_thresholds.py:
- list_norm_groups() nuevo: devuelve normas en 4 dominios
- suggest_norm_for_machine: heuristicas extendidas (hidro,
  fan, motor electrico, Brush, centrifugo, diesel)
- suggest_class_for_machine: default para 19 normas nuevas
- suggest_balance_grade() nuevo: G-grade por tipo de maquina
  (turbo->G2.5, bomba/motor->G6.3, crankshaft->G100/G630)

P3 - pages/00_Machinery_Library.py:
- Tab Norma ISO ahora con selectbox agrupado por dominio
  (prefijos VIB/EJE/BAL/ROT)
- Caption con dominio + metrica + unidad on-select

Solo push a DEV (main intacto). Cliente revisara antes de main." || echo "  (sin cambios)"
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
echo " ✓ Ciclo 17.10 pusheado a DEV (main intacto)"
echo "================================================================"
echo ""
echo " Total catálogo: 26 normas en 4 dominios:"
echo "   📳 Vibración (carcasa) ........ 20 normas"
echo "   🛡️  Vibración eje (proximity) ..  2 normas"
echo "   ⚖️  Balanceo de rotor ..........  3 normas (incl. ISO 21940-11/-12)"
echo "   🔬 Análisis rotodinámico ......  1 norma  (API 684)"
echo ""
echo " Para probar:"
echo "  1. Machinery Library → instancia C-200C → Editar metadata →"
echo "     Tab 'Norma ISO' → selectbox debe mostrar 26 opciones"
echo "     agrupadas por VIB/EJE/BAL/ROT."
echo "  2. Para C-200C debería seguir auto-sugiriendo ISO 20816-8."
echo "  3. Para TES1 (LM6000+Brush) debería sugerir ISO 20816-4."
echo "  4. Para Brush solo (sin LM6000) → API 546."
echo "  5. Probar seleccionar API 684 → ver 8 clases AF/SM/UR."
echo "  6. Probar ISO 21940-11 → ver 11 G-grades (G 0.4 → G 4000)."
echo ""
echo " Si OK → siguiente ciclo decidir publish a main con tag v3.2.0"
echo "================================================================"
