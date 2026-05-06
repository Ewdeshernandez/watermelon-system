#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.7.0 → MAIN
# =============================================================
# Ciclo 17.26 P5 — AI Diagnóstico extendido a 6 módulos clínicos
#
# Continuación de v3.6.0 (que llevó AI a Spectrum). Ahora la
# interpretación clínica AI Cat IV ISO 18436-2 está disponible en
# TODOS los módulos analíticos del flujo diario del especialista:
#
#   ✅ Spectrum (ya en prod desde v3.6.0)
#   ✅ Time Waveforms       (NUEVO)
#   ✅ Trends               (NUEVO)
#   ✅ Orbit Analysis       (NUEVO)
#   ✅ Polar Plot           (NUEVO)
#   ✅ Bode Plot            (NUEVO)
#   ✅ Shaft Centerline     (NUEVO)
#
# El cliente final ve el mismo formato profesional en cualquier
# tipo de figura: tabla cuantitativa de evidencia + prosa forense
# con frase-tesis en negrita + recomendaciones P1/P2/P3/P4 con
# horizonte sugerido + cláusulas normativas + cierre legal de
# responsabilidad del operador. Sin marcas "AI" en el PDF.
#
# Patrón replicado consistentemente:
# ────────────────────────────────────────────────────────
# Cada módulo agrega:
#   1. Import de generate_ai_diagnostic + is_ai_available
#   2. queue_*_to_report con parámetro opcional notes_override
#      (cuando viene con contenido, reemplaza la narrativa
#      determinística — el bloque AI manda)
#   3. Expander "Interpretación clínica AI" con botones
#      Generar / Regenerar y caption de costo
#   4. Payload específico al tipo de análisis (cada módulo
#      expone sus métricas relevantes a Claude)
#   5. Persistencia del resultado en st.session_state con key
#      por panel (so multi-panel works sin colisión)
#   6. Banner discreto si se usó modelo de respaldo (Haiku 4.5)
#   7. Caption con modelo, tokens, costo dinámico, fallback flag
#   8. "Enviar a Reporte" arma quant table + marcadores
#      <<<WM_AI_BLOCK>>> + <<<WM_AI_NARRATIVE>>> y los pasa al
#      reporte como notes_override
#
# Payloads específicos por módulo:
# ────────────────────────────────────────────────────────
#   Time Waveforms: RMS, peak, peak-to-peak, crest factor,
#                   kurtosis, skewness, transitorios detectados
#                   y threshold dinámico, narrativa Cat IV.
#   Trends:         stats por record (n_samples/min/max/mean/std/
#                   last_value), autodiag headline+status+prose+
#                   recomendaciones, thresholds W/D, fuente de
#                   setpoints, n_records y n_operacionales.
#   Orbit:          amplitudes peak-to-peak X/Y waveform,
#                   amplitudes harmónicas X/Y filtradas, ratio
#                   Y/X, dirección de precesión, traversal,
#                   samples_per_revolution, revoluciones
#                   disponibles, modo de filtro.
#   Polar:          max amplitude, health score, candidate count,
#                   velocidades críticas con RPM/amp/Q-factor/
#                   phase_delta, semáforo, RPM operativa,
#                   rotation direction, diagnostic detail+action.
#   Bode:           rango RPM (min/max), max amplitude, health
#                   score, candidate count, velocidades críticas
#                   con Q-factor y phase delta, modo de fase,
#                   diagnostic detail+action.
#   Shaft Centerline: clearance center mode, puntos X/Y
#                   acoplados, máquina, Cat IV detail+action,
#                   notes legacy de utilization.
#
# Tabla cuantitativa de evidencia (en el PDF):
# ────────────────────────────────────────────────────────
# Cada módulo arma su propia tabla con los valores que importan
# al cliente para entender el diagnóstico AI:
#   Waveforms: RPM · RMS · Peak · Peak-to-Peak · Crest Factor
#              · Kurtosis · Transitorios · Severidad · Punto
#   Trends:    Métrica · N señales · N operativas · Umbrales
#              W/D · Fuente de setpoints
#   Orbit:     RPM · Amplitud X p-p · Amplitud Y p-p · Precesión
#              · Traversal · Modo filtro · Punto
#   Polar:     RPM operativa · Amplitud máxima · Health score ·
#              Velocidades críticas detectadas · Semáforo · Punto
#   Bode:      Rango RPM · Amplitud máxima · Health score ·
#              Velocidades críticas · Modo fase · Punto
#   SCL:       Máquina · Punto X · Punto Y · Modo referencia ·
#              Diagnóstico Cat IV
#
# Compilación validada:
# ────────────────────────────────────────────────────────
#   - pages/02_Time_Waveforms.py  · OK
#   - pages/04_Trends.py          · OK
#   - pages/05_Orbit_Analysis.py  · OK
#   - pages/06_Polar_Plot.py      · OK
#   - pages/07_Bode_Plot.py       · OK
#   - pages/09_Shaft_Centerline.py · OK
#
# Cero regresiones esperadas:
# ────────────────────────────────────────────────────────
# Cada módulo cae a comportamiento legacy si:
#   - El especialista no genera diagnóstico AI (botón no clickeado)
#   - La key de Anthropic no está configurada en secrets
#   - Anthropic está caído (mensaje user-friendly + retry/fallback)
# El parámetro notes_override es opcional con default None; si
# viene None, los queue_*_to_report siguen usando la narrativa
# determinística existente. Cero breaking changes.
#
# Costo en producción (estimado):
# ────────────────────────────────────────────────────────
#   Sonnet 4.5: ~\$0.015 por diagnóstico (idéntico a v3.6.0)
#   Haiku 4.5 (fallback): ~\$0.005 por diagnóstico
#   100 diagnósticos/mes: \$1.50/mes
#   Si la suite completa se usa en 5 módulos por reporte:
#   500 generaciones/mes ≈ \$7.50/mes
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🧠 RELEASE v3.7.0 → MAIN  (AI Diagnóstico en TODA la suite)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción 6 módulos con AI Diagnóstico:"
echo "  ✓ Time Waveforms      (métricas tiempo-dominio + transitorios)"
echo "  ✓ Trends              (correlación operativa + autodiag)"
echo "  ✓ Orbit Analysis      (precesión + ratio X/Y + traversal)"
echo "  ✓ Polar Plot          (Q-factor + velocidades críticas)"
echo "  ✓ Bode Plot           (rango RPM + amplitud + Q-factor)"
echo "  ✓ Shaft Centerline    (clearance + posición del muñón)"
echo ""
echo "Spectrum ya estaba en main desde v3.6.0."
echo "Total: 7 módulos analíticos con interpretación clínica AI."
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el release v3.7.0 a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

echo "▶ 1/7  Commit de los 6 módulos en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged en releases previos
for v in v3_4_2 v3_4_3 v3_4_4 v3_4_5 v3_4_6 v3_4_7 v3_4_8 v3_4_9 \
         v3_5_0 v3_5_1 v3_5_2 v3_5_3 v3_6_0; do
    git checkout HEAD -- "_release_${v}_hotfix.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_ux.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_main.sh" 2>/dev/null || true
done

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add pages/02_Time_Waveforms.py \
            pages/04_Trends.py \
            pages/05_Orbit_Analysis.py \
            pages/06_Polar_Plot.py \
            pages/07_Bode_Plot.py \
            pages/09_Shaft_Centerline.py \
            _release_v3_7_0_main.sh
    git commit -m "feat(17.26 P5): AI Diagnóstico extendido a 6 módulos clínicos

Continuación de v3.6.0 (Spectrum). Ahora la suite Cat IV completa
tiene interpretación clínica AI con el mismo formato profesional:
tabla cuantitativa de evidencia + prosa forense con frase-tesis +
recomendaciones P1/P2/P3/P4 + cierre legal del operador.

Módulos integrados (mismo patrón de Spectrum):
- pages/02_Time_Waveforms.py: payload con RMS/peak/CF/kurtosis/
  skewness/transitorios. Tabla cuantitativa con métricas
  tiempo-dominio. queue_waveform_to_report con notes_override.
- pages/04_Trends.py: payload con stats por record + autodiag +
  thresholds. Tabla con métrica + N señales/operativas + umbrales
  + fuente de setpoints. queue_trend_to_report con notes_override.
- pages/05_Orbit_Analysis.py: payload con amplitudes X/Y peak-to-peak,
  ratio Y/X, precesión, traversal. Tabla con RPM + amplitudes +
  precesión + filtro. queue_orbit_to_report con notes_override.
- pages/06_Polar_Plot.py: payload con velocidades críticas +
  Q-factor + phase delta + health score. Tabla con RPM operativa
  + amplitud máxima + criticas + semáforo. queue_polar_to_report
  con notes_override (vía closure helper).
- pages/07_Bode_Plot.py: payload con rango RPM + criticas +
  Q-factor + modo fase. Tabla con rango RPM + amplitud + criticas
  + modo fase. queue_bode_to_report con notes_override.
- pages/09_Shaft_Centerline.py: payload con clearance mode +
  Cat IV detail/action. Tabla con máquina + puntos X/Y + modo +
  Cat IV headline. push_report_item con final_notes overrides
  inline.

Patrón consistente en todos:
- Expander \"Interpretación clínica AI\" con botones
  Generar/Regenerar y caption de costo (~\$0.015 Sonnet,
  ~\$0.005 Haiku).
- Persistencia en st.session_state con key por panel.
- Banner discreto si se usó modelo de respaldo (visible solo al
  especialista, NUNCA en el PDF al cliente).
- Caption rico con modelo, tokens, costo dinámico, fallback flag.
- 'Enviar a Reporte' arma quant table + marcadores y los pasa
  como notes_override.

Cero regresiones:
- notes_override es Optional con default None. Cuando es None,
  los queue_*_to_report siguen usando la narrativa determinística
  existente.
- Si Anthropic cae, retry x3 + fallback Haiku absorben el evento.
- Si Anthropic key no está configurada, el botón AI muestra
  mensaje informativo y el módulo funciona normal (legacy).

Cobertura final de la suite Cat IV ISO 18436-2:
- Spectrum (firmas mecánicas, BPFx, sub-síncrónicas)
- Time Waveforms (transitorios, modulación)
- Trends (correlación operativa, regímenes de daño)
- Orbit (forma de órbita, precesión, rub vs whirl)
- Polar (estado de balance, drift, estabilidad)
- Bode (Q-factor, velocidades críticas, separation margin)
- Shaft Centerline (eccentricity, attitude angle, lift-off)" || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene los 6 módulos commiteados"
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
MERGE_MSG="release(v3.7.0): merge dev -> main · AI Diagnóstico en 6 módulos clínicos

Continuación de v3.6.0 (Spectrum). Ahora la suite analítica completa
del especialista tiene interpretación clínica AI Cat IV ISO 18436-2:

Módulos NUEVOS con AI:
- Time Waveforms (transitorios, modulación, asimetría)
- Trends (correlación operativa, regímenes de daño)
- Orbit (forma de órbita, precesión, rub vs whirl)
- Polar (Q-factor, velocidades críticas, balance)
- Bode (separation margin, response peaks)
- Shaft Centerline (eccentricity, attitude angle)

Spectrum ya estaba desde v3.6.0. Total: 7 módulos cubiertos.

El cliente recibe el mismo formato profesional en TODAS las
figuras del reporte: tabla cuantitativa + prosa forense con
frase-tesis + recomendaciones P1/P2/P3/P4 + cierre legal del
operador. Sin marcas 'AI' en el PDF.

Robustez heredada de v3.6.0:
- Retry x3 con backoff exponencial para 429/502/503/529.
- Fallback automático a Haiku 4.5 si Sonnet 4.5 falla.
- Detección de timeouts y errores de conexión.
- Degradación elegante si la key no está configurada.

Cero regresiones: cada módulo cae a comportamiento legacy si el
especialista no genera AI o si la key de Anthropic no está
configurada."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.7.0..."
TAG_EXISTS=$(git tag -l "v3.7.0")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.7.0 ya existe. Saltando creación."
else
    git tag -a v3.7.0 -m "Release v3.7.0 — AI Diagnóstico Cat IV en suite completa (Waveforms/Trends/Orbit/Polar/Bode/SCL)"
    echo "  ✓ Tag v3.7.0 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.7.0 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ RELEASE v3.7.0 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo ""
echo " 🔑 NO HACE FALTA tocar secrets — la key [anthropic] ya está"
echo "    configurada desde v3.6.0. Si la borraste por accidente, el"
echo "    botón AI muestra mensaje informativo y el resto del sistema"
echo "    sigue funcionando."
echo ""
echo " 🧪 VALIDACIÓN en producción:"
echo ""
echo "    Probá generar un PDF que combine 3+ módulos distintos para"
echo "    verificar que la interpretación clínica AI luce consistente:"
echo ""
echo "    1. Login en wm-home-final-2026.streamlit.app"
echo "    2. Cargá señales de tu activo de prueba"
echo "    3. Andá a Spectrum → Generar AI → Enviar a Reporte"
echo "    4. Andá a Time Waveforms → Generar AI → Enviar a Reporte"
echo "    5. Andá a Trends → Generar AI → Enviar a Reporte"
echo "    6. (Opcional) Polar / Bode / Orbit / SCL"
echo "    7. Reports → Generar PDF"
echo "    8. Verificá que las 3+ figuras tengan:"
echo "       - Tabla cuantitativa con datos correctos del módulo"
echo "       - Prosa de hallazgos (sin '###' ni '**' literales)"
echo "       - Recomendaciones P1/P2/P3/P4"
echo "       - Cierre legal del operador"
echo "       - Sin marca 'AI' visible al cliente"
echo ""
echo " 🚀 Próximos pasos sugeridos:"
echo ""
echo "    Algunas ideas para v3.8.0 (en dev, mientras la 3.7 corre):"
echo "    - Síntesis ejecutiva AI: 'Resumen del reporte completo'"
echo "      en el módulo Reports (Claude lee TODOS los hallazgos y"
echo "      emite el executive summary del PDF)."
echo "    - Q&A AI sobre archivo histórico: 'Mostrame todos los"
echo "      reportes de TES1 con desbalance en los últimos 6 meses'."
echo "    - Auto-fix de gramática en notas del especialista."
echo "    - AI ranking de activos por riesgo en el Home dashboard."
echo "    - Briefing diario AI con cambios desde el reporte anterior."
echo ""
echo "================================================================"
