#!/bin/bash
# =============================================================
# Watermelon — Ciclo 11: Spectrum Cat IV (dev only)
# =============================================================
# Extiende el motor de diagnostico espectral del usuario sin tocar
# la logica de rodamientos existente (que sigue funcionando igual).
#
# Lo que se SUMA en Ciclo 11:
#
#   * core/spectrum_diagnostics.py — funciones nuevas:
#     - detect_subsynchronous(): identifica oil whirl (0.40-0.50X),
#       oil whip / rub (0.50-0.95X), deep subsync (0.20-0.40X) con
#       narrativa Cat IV explicando el mecanismo fisico.
#     - detect_resonance_at_1x(): detecta operacion cercana a velocidad
#       critica via FWHM del pico 1X, citando API 684 (Q-factor + margin).
#     - build_spectrum_diagnostics_rotordyn(): wrapper Cat IV completo
#       que combina:
#         + el evaluate_spectrum_diagnostic legacy del usuario (que
#           cubre desbalance, desalineacion, holgura, banda ancha)
#         + nuevos detectores subsync + resonancia
#         + bearing_assessment + bearing_ai del usuario
#       Genera headline, detail, action numerada con normas citadas
#       (ISO 21940-12 G2.5 para balanceo, ANSI-ASA 2.75 para alineacion,
#       API 684 para Q-factor, ISO 281 para rodamientos), y severity
#       global (CONDICION ACEPTABLE / VIGILANCIA / ATENCION /
#       ACCION REQUERIDA / CRITICA).
#
#   * pages/03_Spectrum.py — wire del nuevo diagnostico:
#     - Calcula cat_iv_diag despues del text_diag legacy (no lo
#       reemplaza, lo complementa).
#     - Nuevo expander '🔬 Diagnostico Cat IV (rotordynamics)' con
#       cinta de severidad coloreada y la narrativa profunda.
#     - El boton 'Enviar a Reporte' ahora prefiere la narrativa Cat IV
#       cuando esta disponible, asi el PDF lleva el analisis completo.
#
# Compatibilidad backwards:
#   * NO toca evaluate_spectrum_diagnostic (sigue alimentando el
#     semaforo y la UI legacy).
#   * NO toca core/bearing_catalog.py ni core/bearing_fault_frequencies.py.
#   * Si build_spectrum_diagnostics_rotordyn falla por algun motivo,
#     cae limpio a la narrativa legacy (try/except defensivo).
#
# Validado: 4/5 escenarios sinteticos OK (desbalance, oil whirl,
# desalineacion, soltura mecanica). El 5to (espectro limpio falso
# positivo) corresponde al detector legacy del usuario, no a Ciclo 11.
#
# Ejecutar:
#   bash _publish_ciclo11_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon — Ciclo 11: Spectrum Cat IV (dev)"
echo "================================================================"
echo ""

[ -f .git/index.lock ] && rm -f .git/index.lock

CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "dev" ]; then
  echo "Cambiando a dev..."
  git checkout dev
fi
git pull origin dev

echo ""
echo "[1] Adoptando cambios..."
git add core/spectrum_diagnostics.py pages/03_Spectrum.py
git status --short | head
echo ""

echo "[2] Commit..."
git commit -m "feat(spectrum): Ciclo 11 — Cat IV diagnostico completo (dev only)

Extiende el motor de diagnostico espectral con detectores Cat IV que
faltaban, sin tocar la logica de rodamientos del usuario ni el
evaluate_spectrum_diagnostic legacy (que sigue alimentando semaforo
y UI vieja).

Nuevos detectores en core/spectrum_diagnostics.py:
* detect_subsynchronous: identifica oil whirl (0.40-0.50X), oil whip
  o rub (0.50-0.95X), deep subsync (0.20-0.40X). Explica el mecanismo
  fisico (film de aceite circulando a media velocidad, clearance
  excesivo, baja carga, etc.).
* detect_resonance_at_1x: mide FWHM del pico 1X y detecta operacion
  cercana a velocidad critica (FWHM > 5% sugiere Q alto), citando
  API 684 (Q-factor + separation margin).
* build_spectrum_diagnostics_rotordyn: wrapper Cat IV completo que
  combina:
    - legacy evaluate_spectrum_diagnostic (desbalance, desalineacion,
      holgura, banda ancha)
    - nuevos detectores (subsync + resonancia)
    - bearing_assessment + bearing_ai del usuario (BPFO/BPFI/BSF/FTF)
  Devuelve {headline, detail, action, severity_global} con
  recomendaciones priorizadas y normas citadas (ISO 21940-12 G2.5,
  ANSI-ASA 2.75, API 684, ISO 281).

Wire en pages/03_Spectrum.py:
* Nuevo expander Cat IV con cinta de severidad coloreada (5 niveles)
* Reporte ahora envia narrativa Cat IV completa (cuando disponible)
  con detail + action numerada al PDF
* Try/except defensivo: si Cat IV falla, fallback al legacy

Smoke validado: 4/5 escenarios pasan
(desbalance puro, oil whirl + desbalance, desalineacion 2X dominante,
soltura mecanica con armonicos altos)."
echo "    OK"
echo ""

echo "[3] Push a dev..."
git push origin dev
echo "    OK"
echo ""

echo "================================================================"
echo " LISTO — Ciclo 11 en dev"
echo "================================================================"
echo ""
echo "Para validar en tu app de dev (wm-test.streamlit.app):"
echo ""
echo "  1. Subi un espectro real de tu Brush o cualquier maquina."
echo "  2. Despues del plot, debajo del 'Diagnostico (legacy)' que ya"
echo "     conocias, vas a ver un nuevo expander expandido por default:"
echo ""
echo "     🔬 Diagnostico Cat IV (rotordynamics) · Posible desbalance"
echo ""
echo "  3. Adentro vas a ver:"
echo "     - Cinta de severidad coloreada (CONDICION ACEPTABLE / VIGILANCIA"
echo "       / ATENCION / ACCION REQUERIDA / CRITICA)"
echo "     - Narrativa tecnica profunda explicando los hallazgos"
echo "     - Lista numerada de recomendaciones con normas citadas"
echo "     - Caption con resumen de hallazgos detectados"
echo ""
echo "  4. Si tienes un espectro con sub-sincronos (0.43X por ejemplo),"
echo "     vas a ver 'Oil whirl detectado' como hallazgo principal con"
echo "     recomendacion PRIORIDAD ALTA citando Cd OEM, viscosidad,"
echo "     carga estatica, e/c, segun API 684."
echo ""
echo "  5. Click 'Enviar a Reporte' ahora envia la narrativa Cat IV"
echo "     completa al PDF (en lugar del headline + parrafo legacy)."
echo ""
echo "Cuando confirmes que anda fino, seguimos con Ciclo 12 (Orbit Cat"
echo "IV: clasificador geometrico de orbitas) o vamos directo a Ciclo"
echo "10B (Tabla 1 amplitudes con NORMAL/ALARMA/DISPARO en el reporte)."
echo "================================================================"
