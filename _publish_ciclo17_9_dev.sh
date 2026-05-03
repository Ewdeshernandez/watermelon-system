#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.9 → DEV: Catálogo ISO/API + Override
# =============================================================
# Sistema completo de normas industriales de evaluación de
# vibración. El usuario puede asignar una norma + clase a cada
# instancia de activo, los setpoints Warning/Danger se derivan
# automáticamente de la tabla de la norma, y el especialista
# puede overridear con justificación que queda en el reporte.
#
# Implementación en 4 partes:
#
# (P1) core/iso_thresholds.py NUEVO — Catálogo central
#      7 normas con tablas completas:
#      - ISO 20816-2 (turbinas vapor + generadores >50MW)
#      - ISO 20816-3 (industriales 15kW-50MW: bombas, centrífugos)
#      - ISO 20816-4 (turbinas de gas: LM6000, LM2500, Frame)
#      - ISO 20816-8 (compresores reciprocantes: ARIEL KBK)
#      - API 670 (proximity probes mil pp)
#      - API 618 (compresores recip refinería)
#      - ISO 10816-6 (legacy, motores diésel)
#
#      API: list_norms(), list_classes_for_norm(),
#      get_thresholds(), suggest_norm_for_machine() con
#      heurística por keywords del activo.
#
# (P2) core/instance_state.py — 5 campos nuevos en Instance
#      iso_norm_code, iso_norm_class,
#      setpoint_warning_override, setpoint_danger_override,
#      override_justification
#      Se persisten en JSON, retro-compatibles (default vacíos).
#
# (P3) pages/00_Machinery_Library.py — Tab "Norma ISO" nuevo
#      Dentro de "Editar metadata completa de esta instancia"
#      (ahora con 9 tabs en lugar de 8). Contiene:
#      - Selectbox de norma con auto-sugerencia inteligente
#        según asset_class + driver_kind + driven_kind
#      - Selectbox de clase con auto-sugerencia según potencia
#        y tipo de soporte
#      - Setpoints sugeridos en vivo + cita normativa
#      - Override del especialista con campos numéricos +
#        justificación obligatoria si difiere de la norma
#      - Cálculo de Δ% override vs norma
#
# (P4) Cableado en consumidores existentes
#      - pages/04_Trends.py: suggest_trend_thresholds() acepta
#        nuevo param `instance` y la chequea PRIMERO (antes que
#        el Sensor Map). Override del especialista respetado.
#      - threshold_source en session_state ahora incluye
#        norm_reference + norm_code + override_justification.
#      - Bloque de prosa del PDF agrega línea "Referencia
#        normativa: Tabla 3 ISO 20816-8:2018 Annex A" + cita
#        del override justificado.
#
# Resultado en flujo end-to-end:
#   1. Cargás CSV en Trends.
#   2. La sidebar de Alarms muestra "Setpoints sugeridos:
#      ISO 20816-8 Class 2 (Override)" (en lugar del genérico).
#   3. Generás PDF → el reporte cita la norma + class + override
#      con justificación en la sección Diagnóstico ejecutivo.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo ""
echo "▶ Stageando 17.9..."
git add core/iso_thresholds.py
git add core/instance_state.py
git add pages/00_Machinery_Library.py
git add pages/04_Trends.py
git add _publish_ciclo17_9_dev.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando 17.9..."
    git commit -m "feat(iso-norms): catalogo ISO/API thresholds + UI seleccion + override (17.9)

Sistema completo de normas industriales para Warning/Danger.

(P1) core/iso_thresholds.py NUEVO con 7 normas:
- ISO 20816-2 (turbinas vapor + generadores grandes)
- ISO 20816-3 (industriales 15kW-50MW)
- ISO 20816-4 (turbinas de gas)
- ISO 20816-8 (compresores reciprocantes)
- API 670 (proximity probes)
- API 618 (compresores recip refineria)
- ISO 10816-6 (legacy motores diesel)
API: list_norms, list_classes_for_norm, get_thresholds,
suggest_norm_for_machine con heuristica por keywords.

(P2) core/instance_state.py — 5 campos nuevos en Instance:
iso_norm_code, iso_norm_class, setpoint_warning_override,
setpoint_danger_override, override_justification.
Persistidos en JSON, retro-compatibles.

(P3) pages/00_Machinery_Library.py — tab 'Norma ISO' nuevo
con selectbox de norma (auto-sugerencia segun activo) +
class + setpoints sugeridos + override del especialista
con justificacion + delta% vs norma.

(P4) Cableado en consumidores:
- suggest_trend_thresholds acepta param 'instance' y la
  chequea PRIMERO (antes Sensor Map y fallback ISO generico).
- threshold_source en session_state incluye norm_reference,
  norm_code, override_justification.
- PDF cita 'Referencia normativa: Tabla X ISO ...' + override
  justificado en seccion Diagnostico ejecutivo." || echo "  (sin cambios)"
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
echo " ✓ Ciclo 17.9 pusheado a DEV (main intacto)"
echo "================================================================"
echo ""
echo " Para probar:"
echo "  1. Machinery Library → instancia C200C → Editar metadata →"
echo "     Tab 'Norma ISO' → debe sugerir ISO 20816-8."
echo "     Click Class 2, ver Warning 7.1 / Danger 17.8 sugeridos."
echo "     Probar override: Warning 5.0 + justificación."
echo "     Guardar."
echo "  2. Ir a Trends, cargar CSV. Sidebar Alarms ahora muestra"
echo "     'Setpoints sugeridos: ISO 20816-8 Class 2 (Override)'."
echo "  3. Send to Report → PDF debe citar la norma + override."
echo ""
echo "  Probar también con TES1 (LM6000+Brush) → debe sugerir"
echo "  ISO 20816-4 automáticamente."
echo "================================================================"
