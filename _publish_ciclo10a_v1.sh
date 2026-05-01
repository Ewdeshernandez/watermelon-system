#!/bin/bash
# =============================================================
# Watermelon — Ciclo 10A: Reports estructura SIGA
# =============================================================
# Reorganiza el PDF para que parezca el reporte SIGA-REP-TEC original:
#
#   - Portada con bloque GRANDE del activo:
#       TURBOGENERADOR TES1
#       LM5000
#       VILLAVICENCIO
#       TERMOSURIA
#
#   - Header en cada pagina interna con codigo de formato controlado:
#       WMS-FMT-001 | Version 1 | Fecha 2026-04-28 | REPORTE DE...
#
#   - Recomendaciones AL INICIO (seccion 1), no al final.
#   - Resumen Ejecutivo con cinta severidad sigue justo despues de portada.
#   - Objetivo y Desarrollo del servicio quedan despues de Recomendaciones.
#
#   - Nuevos campos en formulario UI: 'Clase de activo' y 'Modelo'.
#
# Validado: 11/11 checks pasan en smoke contra meta SIGA-style.
#
# Ejecutar:
#   bash _publish_ciclo10a_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon — Ciclo 10A: Reports estructura SIGA (dev)"
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
git add pages/16_Reports.py
git status --short | head
echo ""

echo "[2] Commit..."
git commit -m "feat(report): Ciclo 10A + 10A.1 — estructura SIGA + polish portada (dev only)

Reorganiza el PDF para que coincida visualmente con el formato
SIGA-REP-TEC original que el cliente conocia:

* Portada: bloque grande del activo en mayusculas, jerarquia tipo:
    TURBOGENERADOR TES1   (font 22pt, primer linea — clase + tag/unidad)
    LM5000                (font 17pt — modelo)
    VILLAVICENCIO         (font 17pt — ubicacion)
    TERMOSURIA            (font 17pt — cliente)
  Si esta el train_description tambien se imprime debajo en regular.

* Header de cada pagina interna: codigo de formato controlado tipo
  'WMS-FMT-001 | Version 1 | Fecha 2026-04-28 | REPORTE...' que es
  el equivalente al SIGA-FMT-178 que el cliente esperaba ver.

* Reordenamiento de secciones SIGA-style:
    Portada
    Resumen Ejecutivo (con cinta de severidad)
    1. RECOMENDACIONES (antes era al final)
    2. OBJETIVO DEL SERVICIO
    3. DESARROLLO DEL SERVICIO
    4. FIGURAS Y ANALISIS

  La logica es: el cliente abre el PDF y lee Recomendaciones primero
  para saber que tiene que hacer; el detalle viene despues.

* Defaults profesionales:
    prepared_role: 'Junior Condition Monitoring Engineer'
    reviewed_role: 'Machinery Diagnostic Champion'

* Nuevos campos meta:
    asset_class    -> ej. 'TURBOGENERADOR'
    asset_model    -> ej. 'LM5000'
    format_code    -> ej. 'WMS-FMT-001'
    format_version -> ej. '1'
    format_date    -> ej. '2026-04-28'
    prepared_city, reviewed_city (preparados para 10B)

* UI: dos inputs nuevos 'Clase de activo (portada)' y 'Modelo /
  configuracion (portada)' debajo de Unidad/Ubicacion/Consecutivo.

Validado: smoke 11/11 OK contra meta del Brush 54MW estilo SIGA
(TURBOGENERADOR TES1 / LM5000 / VILLAVICENCIO / TERMOSURIA + recos
con normas ISO 21940-12 y ANSI-ASA 2.75)."
echo "    OK"
echo ""

echo "[3] Push a dev..."
git push origin dev
echo "    OK"
echo ""

echo "================================================================"
echo " LISTO — Ciclo 10A en dev"
echo "================================================================"
echo ""
echo "Para validar visualmente:"
echo ""
echo "  1. Abri tu app de dev (wm-test.streamlit.app)"
echo "  2. Andate a Reports"
echo "  3. En 'Metadatos del reporte' completa los nuevos campos:"
echo "     - Clase de activo: TURBOGENERADOR"
echo "     - Modelo: LM5000  (o el de tu maquina real)"
echo "     - Activo: TES1 (que ya existe — va a la portada como tag)"
echo "     - Ubicacion: VILLAVICENCIO"
echo "     - Cliente: TERMOSURIA"
echo "     - Preparado por / Revisado por: tus nombres"
echo "  4. Carga al menos una figura desde Spectrum / SCL / Polar / Bode"
echo "  5. Click 'Auto-redactar resumen ejecutivo' (boton conocido)"
echo "  6. Click 'Auto-redactar 1/2/3 desde figuras' (boton conocido)"
echo "  7. Click 'Preparar PDF' y descarga"
echo ""
echo "Vas a ver el PDF con el bloque grande SIGA-style en la portada,"
echo "header con WMS-FMT-001 en cada pagina, y Recomendaciones primero."
echo ""
echo "Despues de validar, seguimos con 10B (Tabla 1 amplitudes con"
echo "clasificacion NORMAL/ALARMA/DISPARO automatica)."
echo "================================================================"
