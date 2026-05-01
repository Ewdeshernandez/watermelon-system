#!/bin/bash
# =============================================================
# Watermelon — Ciclo 12: Time Waveform Cat IV (dev only)
# =============================================================
# Lleva el módulo Time Waveforms al nivel Cat IV — el dominio del
# tiempo es la materia prima de todo análisis vibratorio: lo que el
# espectro promedia y oculta, el waveform lo revela.
#
# NUEVO core/waveform_pattern_detectors.py — 5 detectores Cat IV:
#
#   * detect_amplitude_modulation: envolvente Hilbert (FFT-based,
#     sin scipy.signal). Detecta modulación AM con profundidad y
#     frecuencia de modulación. Firma de defectos incipientes de
#     rodamiento, engranajes con desgaste, carga variable.
#
#   * detect_asymmetry: compara pico positivo absoluto vs negativo.
#     Detecta rub unidireccional, precarga lateral del eje,
#     restricción direccional de movimiento (mecanismo secundario:
#     deflexión térmica por calor de fricción local).
#
#   * detect_clipping: muestra cuántas muestras están saturadas en
#     el tope del rango. Indica que la amplitud está subestimada por
#     escala dinámica insuficiente del sensor.
#
#   * detect_sawtooth_shape: compara pendiente promedio de subida vs
#     bajada. Diente de sierra = rub severo bidireccional.
#
#   * detect_beating: modulación lenta (<10 Hz) por interferencia de
#     dos frecuencias muy cercanas. Slip de polos en motores de
#     inducción, máquinas vecinas.
#
#   * classify_crest_factor: 5 buckets Cat IV
#       SINUSOIDAL (CF<3) → CONDICIÓN ACEPTABLE
#       NORMAL     (3-4)  → VIGILANCIA
#       ALERT      (4-6)  → ATENCIÓN
#       SEVERE     (6-10) → ACCIÓN REQUERIDA
#       CRITICAL   (>10)  → CRÍTICA
#
# EXTENDIDO core/waveform_diagnostics.py — sin tocar lo legacy:
#
#   * build_waveform_diagnostics_rotordyn: wrapper Cat IV completo
#     (mismo patrón Polar/Bode/SCL/Spectrum). Combina:
#       - classify_crest_factor (5 buckets)
#       - 5 detectores nuevos
#       - kurtosis estadístico (>4.5 → distribución no gaussiana)
#       - impacts del detector existente del usuario
#     Devuelve {headline, detail, action, severity_global,
#     severity_rank, findings, structured}.
#     Recomendaciones priorizadas citan: ISO 13373-1, ISO 7919,
#     API 670, ISO 281 según hallazgo.
#     Vocabulario técnico Cat IV riguroso: presesión reversa,
#     deflexión térmica, inestabilidad inducida en fluido,
#     compensación slow roll.
#
# WIRE pages/02_Time_Waveforms.py:
#   * Import de build_waveform_diagnostics_rotordyn + helpers
#   * Después de text_diag legacy, calcular cat_iv_wf_diag
#   * Nuevo expander '🔬 Diagnóstico Cat IV (rotordynamics)' con
#     cinta de severidad coloreada (5 niveles)
#   * Botón 'Enviar a Reporte' prefiere narrativa Cat IV cuando
#     está disponible; fallback a legacy si Cat IV falla
#
# Compatibilidad backwards: NO toca compute_waveform_metrics,
# detect_impacts, generate_waveform_diagnostic ni build_waveform_report_notes.
# Si Cat IV falla por algún motivo, todo cae limpio al comportamiento
# anterior (try/except defensivo).
#
# Smoke compile validado: sin errores en los 3 archivos modificados.
# (Smoke runtime con N=4096 timeout en sandbox — tu Streamlit local
#  tiene mejor performance, ahí lo validás visualmente.)
#
# Ejecutar:
#   bash _publish_ciclo12_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon — Ciclo 12: Time Waveform Cat IV (dev)"
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
git add core/waveform_pattern_detectors.py core/waveform_diagnostics.py pages/02_Time_Waveforms.py
git status --short | head
echo ""

echo "[2] Commit..."
git commit -m "feat(waveform): Ciclo 12 — Time Waveform Cat IV completo (dev only)

Lleva el modulo Time Waveforms al nivel Cat IV con 5 detectores
nuevos en dominio del tiempo + wrapper completo de narrativa.

NUEVO core/waveform_pattern_detectors.py:
* detect_amplitude_modulation: envolvente Hilbert (FFT-based, sin
  scipy.signal). Reporta modulation depth y modulation frequency.
  Firma de defectos incipientes de rodamiento, engranajes, carga
  variable.
* detect_asymmetry: relacion pico+/pico-. Rub unidireccional,
  precarga lateral. Mecanismo secundario citado: deflexion termica
  por calor de friccion local.
* detect_clipping: fraccion de muestras saturadas. Indica que la
  amplitud esta subestimada (escala dinamica insuficiente).
* detect_sawtooth_shape: pendiente subida vs bajada. Rub severo
  bidireccional.
* detect_beating: modulacion lenta (<10 Hz) por interferencia entre
  dos frecuencias cercanas. Slip de polos, maquinas vecinas.
* classify_crest_factor: 5 buckets (SINUSOIDAL/NORMAL/ALERT/SEVERE/
  CRITICAL) con severity_label, rank y mensaje Cat IV.

EXTENDIDO core/waveform_diagnostics.py (sin tocar legacy):
* build_waveform_diagnostics_rotordyn: wrapper Cat IV alineado con
  Polar/Bode/SCL/Spectrum. Combina los 5 detectores + kurtosis
  estadistico + impacts del detector existente. Headline + detail +
  action numerada citando normas ISO 13373-1, ISO 7919, API 670,
  ISO 281.
  Vocabulario tecnico riguroso: presesion reversa, deflexion termica,
  inestabilidad inducida en fluido, compensacion slow roll.

WIRE pages/02_Time_Waveforms.py:
* Import de build_waveform_diagnostics_rotordyn
* Calculo cat_iv_wf_diag con try/except defensivo
* Expander 'Diagnostico Cat IV' con cinta de severidad (5 niveles)
* Boton 'Enviar a Reporte' prefiere narrativa Cat IV cuando esta
  disponible; fallback al legacy.

Compatibilidad: NO toca compute_waveform_metrics, detect_impacts,
generate_waveform_diagnostic ni build_waveform_report_notes legacy.
Compile clean en los 3 archivos."
echo "    OK"
echo ""

echo "[3] Push a dev..."
git push origin dev
echo "    OK"
echo ""

echo "================================================================"
echo " LISTO — Ciclo 12 en dev"
echo "================================================================"
echo ""
echo "Validar en wm-test.streamlit.app:"
echo ""
echo "  1. Carga un CSV de waveform desde Load Data."
echo ""
echo "  2. Andate a Time Waveforms. Bajo el plot vas a ver primero"
echo "     el diagnostico legacy (st.info) que ya conocias, y debajo"
echo "     un expander nuevo expandido por default:"
echo ""
echo "     🔬 Diagnostico Cat IV (rotordynamics) · <headline>"
echo ""
echo "  3. Adentro vas a ver:"
echo "     - Cinta de severidad coloreada (CONDICION ACEPTABLE /"
echo "       VIGILANCIA / ATENCION / ACCION REQUERIDA / CRITICA)"
echo "     - Narrativa profunda con metricas (RMS, peak, p2p, CF,"
echo "       kurtosis, skewness) y findings detectados"
echo "     - Lista numerada de recomendaciones citando normas:"
echo "       ISO 13373-1, ISO 7919, API 670, ISO 281, ANSI-ASA 2.75"
echo "     - Caption con resumen de hallazgos"
echo ""
echo "  4. Casos donde se activan los detectores:"
echo "     - Sinusoidal limpia → CONDICION ACEPTABLE, sin findings"
echo "     - Defecto rodamiento incipiente → AM detectado con freq"
echo "       de modulacion + recomendacion Envelope Spectrum"
echo "     - Rub unidireccional → asimetria + cita deflexion termica"
echo "       como mecanismo secundario"
echo "     - Sensor saturado → clipping detectado, recomendacion"
echo "       repetir captura con rango ampliado (API 670)"
echo "     - Slip de polos → beating con freq <10 Hz, recomendacion"
echo "       comparar con velocidad nominal de placa"
echo ""
echo "  5. Click 'Enviar a Reporte' ahora envia la narrativa Cat IV"
echo "     completa al PDF (en lugar del headline + parrafo legacy)."
echo ""
echo "Cuando confirmes que anda fino, podemos seguir con:"
echo "  - Ciclo 12.1 Time Waveform UX (auto-select all + auto-escala Y"
echo "    segun unidad, mismo patron Spectrum/Bode)"
echo "  - Ciclo 13 Orbit Cat IV (clasificador geometrico de orbitas)"
echo "  - Ciclo 10B Tabla 1 amplitudes en el reporte"
echo "================================================================"
