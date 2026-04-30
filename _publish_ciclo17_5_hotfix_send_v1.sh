#!/bin/bash
# =============================================================
# Watermelon — Hotfix 17.5.6: send-to-report global + autodiag sobrio
# =============================================================
# Dos correcciones tras revisar el flujo end-to-end del Trend
# y el envío al reporte:
#
# (1) BUG GLOBAL: "envío al reporte no aparece la primera vez"
#
#     Reporte del usuario: "siempre me presenta como un bug que
#     lazo el grafico al reporte y no aparece en todos los modulos
#     debo primero ir a reporte volver y enviarlo y hay si carga".
#
#     Causa raíz en pages/16_Reports.py: la inicialización
#     cargaba report_state.json desde disco a session_state SOLO
#     la primera vez (gated por flag report_state_loaded), y al
#     cargar SOBRESCRIBÍA st.session_state["report_items"] con la
#     lista de disco. Si el usuario añadía items desde Trends/
#     Polar/Bode/SCL/Spectrum/TimeWave/Tabular ANTES de visitar
#     Reports, esos items se perdían en la primera entrada a
#     Reports — había que volver al módulo y reenviar.
#
#     Fix arquitectónico:
#
#     A) core/report_state.py — dos helpers compartidos:
#        - ensure_report_state_loaded(): idempotente, hace MERGE
#          de items en memoria con disco priorizando memoria por
#          id. Ya no sobreescribe.
#        - append_report_item_and_persist(item): asegura load +
#          appende item + persiste a disco. Así disco siempre
#          queda al día y aunque el usuario nunca abra Reports el
#          reporte queda armado.
#
#     B) pages/16_Reports.py — usa ensure_report_state_loaded()
#        en vez de la lógica manual de overwrite.
#
#     C) Los 7 módulos que envían al reporte ahora usan
#        append_report_item_and_persist():
#          - 01__Tabular_List.py     (1 call)
#          - 02_Time_Waveforms.py    (1 call)
#          - 03_Spectrum.py          (3 calls)
#          - 04_Trends.py            (1 call)
#          - 05_Orbit_Analysis.py    (1 call)
#          - 06_Polar_Plot.py        (2 calls)
#          - 07_Bode_Plot.py         (3 calls)
#
#     Total 12 llamadas migradas, fix global del bug.
#
# (2) AUTODIAGNÓSTICO TREND MÁS SOBRIO
#
#     Reporte del usuario: "dejar eso tan marcado como escribir
#     diagnostico ejecutivo eso no mas tipo ingenieria como los
#     otros modulos mas el texto sin tanto subtitulo marcado".
#
#     Antes: chip HTML con border-left coloreado por status,
#     emoji estetoscopio, "Autodiagnóstico ejecutivo" en negrita
#     grande con background.
#
#     Ahora (alineado con Polar / Bode / SCL):
#       st.markdown("### Diagnóstico ejecutivo")
#       st.markdown(f"**{headline}**")
#       st.write(parrafo_1)
#       st.write(parrafo_2)
#       ...
#       st.write("Acciones recomendadas:")
#       st.write("1. ...")
#
#     En el PDF: cambia "AUTODIAGNÓSTICO EJECUTIVO\n{headline}"
#     a "Diagnóstico ejecutivo: {headline}" — sin caps shouty.
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add core/report_state.py
git add pages/16_Reports.py
git add pages/01__Tabular_List.py
git add pages/02_Time_Waveforms.py
git add pages/03_Spectrum.py
git add pages/04_Trends.py
git add pages/05_Orbit_Analysis.py
git add pages/06_Polar_Plot.py
git add pages/07_Bode_Plot.py
git add _publish_ciclo17_5_hotfix_send_v1.sh
git status --short | head -20

git commit -m "fix(reports): bug global send-to-report + autodiag Trend mas sobrio (17.5.6)

(1) Bug global resuelto. Antes Reports cargaba report_state.json
desde disco SOBREESCRIBIENDO st.session_state['report_items'] la
primera vez que se visitaba la pagina, perdiendo items que ya
hubieran sido enviados desde otros modulos en la misma sesion.
La UX era 'envio al reporte no aparece la primera vez, hay que
ir a Reports y volver y reenviar'.

Fix arquitectonico en core/report_state.py:
  - ensure_report_state_loaded(): idempotente, hace MERGE de
    items en memoria con disco priorizando memoria por id.
  - append_report_item_and_persist(item): load + append +
    persiste a disco. Disco siempre al dia.

Los 7 modulos que envian al reporte (Tabular, Time Waveforms,
Spectrum, Trends, Orbit, Polar, Bode) ahora usan el helper
compartido. 12 llamadas migradas. pages/16_Reports.py usa
ensure_report_state_loaded() en vez de la logica manual de
overwrite.

(2) Autodiagnostico Trend mas sobrio, alineado con Polar/Bode/
SCL. Antes: chip HTML con border-left coloreado por status +
emoji estetoscopio + 'Autodiagnostico ejecutivo' destacado.
Ahora: 'Diagnostico ejecutivo' como header markdown simple +
headline en bold + prosa con st.write + recomendaciones
enumeradas. En el PDF se quita el ALL CAPS." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Hotfix 17.5.6 pusheado a dev"
echo "================================================================"
echo ""
echo "Para verificar el bug send-to-report:"
echo "  1. Reiniciar Streamlit (sesion limpia)."
echo "  2. NO entrar a Reports — ir directo a Trends/Polar/Bode/etc."
echo "  3. Cargar CSV + click 'Enviar a Reporte'."
echo "  4. Ir a Reports → el item DEBE aparecer la PRIMERA vez."
echo ""
echo "Para verificar el autodiag sobrio:"
echo "  1. Trends → cargar CSV con thresholds Warning/Danger."
echo "  2. El bloque 'Diagnostico ejecutivo' ahora es solo header"
echo "     markdown + bold + prosa, sin chip colorido."
echo "  3. PNG HD del PDF tampoco tiene 'AUTODIAGNOSTICO EJECUTIVO'"
echo "     en mayusculas — usa 'Diagnostico ejecutivo: ...'."
echo "================================================================"
