#!/bin/bash
# =============================================================
# Watermelon — Ciclo 14a: Machinery Library cockpit (dev)
# =============================================================
# Promueve el sistema a un modelo "máquina-céntrico" como las
# plataformas industriales serias. Cada máquina física se registra
# UNA vez con su perfil técnico completo, y todos los módulos de
# análisis + el reporte la consumen automáticamente.
#
# CAMBIOS:
#
# 1) core/instance_state.py — Instance.header EXTENDIDO (Ciclo 14a)
#    Mantiene back-compat con instancias existentes (campos nuevos
#    con default vacío). Nuevas categorías:
#
#    Identificación: client, site, asset_class
#    Tren acoplado: driver_manufacturer/model/serial,
#                   driven_manufacturer/model/serial, nominal_power_mw
#    Operación: nominal_rpm, min_rpm, max_rpm, trip_rpm, iso_group
#    Soportes: support_type (fluid_film/rolling_element/magnetic/mixed),
#              support_count, support_detail (texto libre)
#    Sondas: probe_x_orientation_deg, probe_y_orientation_deg
#    Setpoints: alert_level, danger_level, trip_level, setpoint_unit
#    Acople: coupling_class (rigid/flexible/fluid)
#    Esquemático: schematic_png (doc_id en el Vault)
#    Mantenimiento (oculto del reporte): last_balance/alignment/
#                   overhaul_date, commissioning_date
#    Trazabilidad (oculto del reporte): driver_serial, driven_serial
#
#    NUEVA función compose_train_description(inst): arma el string
#    "Turbogenerador GE LM6000 acoplado a Generador Brush 54 MW"
#    a partir de los campos driver/driven, sin duplicar palabras.
#
#    update_instance_header(**kwargs) ahora acepta TODOS los campos
#    nuevos vía kwargs.
#
# 2) pages/00_Machinery_Library.py (renombrado desde 17_Asset_Documents.py)
#    * Promovida a la primera página después del Login.
#    * Título "Machinery Library" + subtitle de cockpit.
#    * NUEVA función render_machinery_grid(): grilla de cards 3-col
#      con preview del esquemático (si está cargado), tag, driver,
#      driven, potencia, rpm, cliente, sitio, n° docs. Botón
#      "Activar" por card → setea wm_active_instance_id y rerun.
#      Card de la instancia activa muestra "✓ activa".
#    * render_instance_header() reemplazado por form completo en 8
#      tabs: Identificación · Tren acoplado · Operación · Soportes ·
#      Sondas · Setpoints · Mantenimiento · Esquemático.
#      El selector de esquemático filtra documentos del Vault con
#      document_type='schematic' o similares.
#    * Preview del esquemático visible en la cabecera cuando hay uno.
#
# 3) pages/16_Reports.py — Auto-fill desde Asset Instance activa
#    * NUEVO _autofill_report_meta_from_active_instance(): si hay
#      instancia activa, pre-llena meta[client/asset_class/asset_model/
#      location/asset/unit/train_description/schematic_doc_id/
#      schematic_instance_id] SIN sobrescribir lo que el ingeniero
#      ya escribió (back-fill no destructivo).
#    * DEFAULT_REPORT_META extendido con schematic_doc_id +
#      schematic_instance_id.
#    * RENDER del esquemático en el Resumen Ejecutivo:
#      Después del badge "Estado global: ...", si hay un schematic
#      cargado en el Vault de la instancia activa, se renderiza
#      centrado (12.5cm × 6cm máx, ajuste proporcional) con caption
#      "Esquemático del tren · {train_description}".
#      Try/except defensivo: si falla por cualquier motivo
#      (instancia borrada, doc roto, imagen inválida), omite limpio
#      sin bloquear el resto del reporte.
#
# RESULTADO ESPERADO:
#
# Flujo de trabajo del ingeniero (TES1 ya existe en el sistema):
#   1. Abrir Watermelon → Machinery Library (primera página post-login)
#   2. Ver grid de máquinas — click "TES1 Brush 54 MW" → activa
#   3. Ir a Load Data → subir CSVs (no auto-asociados todavía: 14b)
#   4. Polar / Bode / SCL / Spectrum / Waveform: análisis normal
#   5. Reports → los campos cliente, sitio, clase, modelo, train_desc
#      ya vienen pre-llenados de TES1. El reporte tiene el esquemático
#      del tren centrado bajo "Estado global" en página 3 (Resumen
#      Ejecutivo).
#
# Flujo de alta de máquina nueva (e.g. Parex con TM2500 nuevo):
#   1. Machinery Library → "Crear nueva instancia"
#   2. Llenar tabs Identificación + Tren acoplado + Operación
#   3. Subir esquemático PNG en sección "Cargar nuevo documento"
#      con tipo='schematic', y seleccionarlo en tab Esquemático
#   4. Listo — esa máquina ya está en el grid y todos los reportes
#      futuros la auto-llenan
#
# Compatibilidad: NO toca core/instance_repository, core/instance_selector,
# core/document_vault. Las instancias creadas en Ciclos 8/9 siguen
# siendo válidas (Instance.from_dict resiliente a campos nuevos).
# Compile clean en core/instance_state.py + pages/16_Reports.py +
# pages/00_Machinery_Library.py. Smoke runtime del PDF con TOC +
# esquemático embebido validado en sandbox.
#
# Ejecutar:
#   bash _publish_ciclo14a_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon — Ciclo 14a: Machinery Library cockpit (dev)"
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
git add core/instance_state.py \
        pages/16_Reports.py \
        pages/00_Machinery_Library.py
# git mv detecta el rename automaticamente; igual lo confirmamos
git status --short
echo ""

echo "[2] Commit..."
git commit -m "feat(library): Ciclo 14a — Machinery Library cockpit + auto-fill reportes (dev)

Promueve el sistema a modelo maquina-centrico estilo plataformas
industriales serias. Cada maquina fisica se registra UNA vez con
perfil tecnico completo y todos los reportes futuros la auto-llenan.

core/instance_state.py — Instance.header EXTENDIDO:
* Identificacion: client, site, asset_class
* Tren acoplado: driver_manufacturer/model/serial,
  driven_manufacturer/model/serial, nominal_power_mw
* Operacion: nominal_rpm, min_rpm, max_rpm, trip_rpm, iso_group
* Soportes: support_type (fluid_film/rolling_element/magnetic/mixed),
  support_count, support_detail
* Sondas: probe_x_orientation_deg, probe_y_orientation_deg
* Setpoints: alert/danger/trip_level + setpoint_unit
* Acople: coupling_class (rigid/flexible/fluid)
* Esquematico: schematic_png (doc_id en Vault)
* Mantenimiento + trazabilidad (ocultos del reporte)
NUEVA compose_train_description(): arma 'Turbogenerador GE LM6000
acoplado a Generador Brush 54 MW' sin duplicar palabras.
update_instance_header(**kwargs) acepta todos los campos nuevos.
Back-compat: from_dict resiliente a campos faltantes.

pages/00_Machinery_Library.py (renombrado desde 17_Asset_Documents.py):
* Promovida a primera pagina despues del Login.
* render_machinery_grid(): cards 3-col con preview del esquematico,
  tag, driver, driven, potencia, rpm, cliente, sitio, n docs.
* Form completo en 8 tabs: Identificacion / Tren acoplado / Operacion
  / Soportes / Sondas / Setpoints / Mantenimiento / Esquematico.
* Selector de esquematico filtra docs con type=schematic.

pages/16_Reports.py — Auto-fill desde instancia activa:
* _autofill_report_meta_from_active_instance(): pre-llena meta
  (client/asset_class/asset_model/location/asset/train_description/
  schematic_doc_id) sin sobrescribir lo que el ingeniero ya escribio.
* DEFAULT_REPORT_META extendido con schematic_doc_id + _instance_id.
* Render del esquematico en Resumen Ejecutivo: 12.5x6cm centrado bajo
  el badge 'Estado global', con caption 'Esquematico del tren · ...'.
  Try/except defensivo: si falla, omite limpio sin bloquear PDF.

Compatibilidad: NO toca core/instance_repository, core/instance_selector,
core/document_vault. Instancias previas siguen validas.
Compile clean. Smoke runtime de PDF con TOC + esquematico OK."
echo "    OK"
echo ""

echo "[3] Push a dev..."
git push origin dev
echo "    OK"
echo ""

echo "================================================================"
echo " LISTO — Ciclo 14a en dev"
echo "================================================================"
echo ""
echo "Plan de pruebas en wm-test.streamlit.app:"
echo ""
echo "  ===== A. Verificar promocion de Library a pagina 0 ====="
echo "  1. Login → la PRIMERA pagina post-login es 'Machinery Library'"
echo "     (antes era Load Data; ahora Library va primero)."
echo "  2. Ver el grid de cards: si ya tenias instancias creadas en"
echo "     Ciclo 8/9, deben aparecer todas con los campos minimos"
echo "     que tenian (tag, profile_label, n docs)."
echo ""
echo "  ===== B. Editar metadata extendida de una maquina ====="
echo "  3. Click 'Activar' en una card (ej. TES1 si ya existe)."
echo "  4. Click 'Editar metadata completa de esta instancia'."
echo "  5. Llenar las 8 tabs:"
echo "     - Identificacion: tag=TES1, client=ECOPETROL - MAGNEX,"
echo "       site=TERMOSURIA - VILLAVICENCIO, asset_class=TURBOGENERADOR"
echo "     - Tren acoplado: driver GE/LM6000, driven Brush/Generador 54 MW,"
echo "       potencia=54, coupling_class=flexible"
echo "     - Operacion: nominal_rpm=3600, min_rpm=3500, max_rpm=3700,"
echo "       trip_rpm=3960, iso_group=rigid"
echo "     - Soportes: support_type=fluid_film, support_count=4,"
echo "       support_detail='4 cojinetes planos tilting pad 5 zapatas...'"
echo "     - Sondas: probe_x=45, probe_y=-45 (XL/YR estandar)"
echo "     - Setpoints (opcional): alert=4, danger=6, trip=8, unit='mil pp'"
echo "     - Mantenimiento: dejar vacio o llenar fechas reales"
echo "     - Esquematico: dejar 'sin esquematico' por ahora"
echo "  6. Click 'Actualizar metadata completa' → mensaje 'Metadata actualizada'."
echo ""
echo "  ===== C. Subir esquematico PNG y vincularlo ====="
echo "  7. En la misma pagina, ir a 'Cargar nuevo documento'."
echo "  8. Subir una imagen PNG/JPG del tren TES1, type='schematic',"
echo "     title='Esquematico TES1 Brush 54 MW'."
echo "  9. Despues, volver a 'Editar metadata' → tab 'Esquematico' →"
echo "     ahora aparece el documento que subiste como opcion."
echo " 10. Seleccionarlo y guardar."
echo " 11. Volver al header de la instancia: ahora deberia mostrar el"
echo "     preview del esquematico (480px de ancho)."
echo " 12. Volver al grid de cards (sidebar): la card de TES1 ahora"
echo "     muestra el esquematico embebido."
echo ""
echo "  ===== D. Auto-fill en Reports ====="
echo " 13. Asegurarte que TES1 esta activa (badge verde en su card)."
echo " 14. Ir a Reports."
echo " 15. Sin tipear nada, ver que los campos cliente, ubicacion,"
echo "     asset_class, asset_model, train_description ya vienen"
echo "     pre-llenados desde TES1."
echo " 16. (Opcional) modificar manualmente algun campo → no se debe"
echo "     pisar en el siguiente rerun (auto-fill solo back-fill)."
echo " 17. Generar el PDF."
echo " 18. Pagina 1: portada con tu maquina (sin cambios)."
echo " 19. Pagina 2: TOC (sin cambios)."
echo " 20. Pagina 3: 'RESUMEN EJECUTIVO' + badge severidad +"
echo "     ESQUEMATICO DEL TREN centrado + caption 'Esquematico del"
echo "     tren · Turbogenerador GE LM6000 acoplado a Generador..."
echo "     ¡Eso es lo nuevo de este ciclo!"
echo " 21. Si la maquina activa NO tiene esquematico cargado, el PDF"
echo "     se genera igual que antes (sin imagen, omite limpio)."
echo ""
echo "  ===== E. Crear maquina nueva (caso TES3 TM2500) ====="
echo " 22. En el grid, click 'Crear nueva instancia'."
echo " 23. Tag=TES3, profile=el que aplique, llenar driver=GE/TM2500,"
echo "     driven=Brush/Generador X MW, etc."
echo " 24. Confirmar que aparece como nueva card en el grid."
echo " 25. Activar TES3 → ir a Reports → confirmar que ahora los"
echo "     campos auto-llenados son los de TES3 (no de TES1)."
echo ""
echo "Cuando confirmes que todo OK, los siguientes pasos quedaran:"
echo "  - Ciclo 14b: wire del selector en Load Data + Spectrum +"
echo "    Time Waveforms + Trend (los CSVs quedan asociados a la"
echo "    instancia activa al subirlos)"
echo "  - Ciclo 14c: polish del wizard de alta de maquina nueva"
echo "  - Ciclo 13: Orbit avanzado"
echo "  - Ciclo 15: Machine Map (esquematico con heatmap de severidad)"
echo "================================================================"
