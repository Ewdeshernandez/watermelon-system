#!/bin/bash
# =============================================================
# Watermelon — Ciclo 18.1 → DEV: Industrial Plumbing Foundations
#                                  (Tests + Templates + Loaders + API)
# =============================================================
# Arranca el "Ciclo 18 — Fundamentos para superar a System1/Baker/
# Emerson/SKF". Esta es la base sin la cual el resto del roadmap
# (live ingestion, multi-tenant, pattern memory federado) no escala.
#
# Lo nuevo (TODO ADITIVO — cero archivos de producción tocados):
#
#   ► SUITE DE TESTS CON GOLDEN DATASETS SINTÉTICOS  (157 tests)
#     - tests/conftest.py         generadores: sine, multi-armónico,
#                                 BPFO, oil whirl, looseness, Bode,
#                                 órbita
#     - tests/test_synthetic_signals.py
#     - tests/test_waveform_metrics.py
#     - tests/test_order_tracking.py
#     - tests/test_tsa.py
#     - tests/test_bearing_fault_frequencies.py
#     - tests/test_iso_thresholds.py
#     - tests/test_rotordynamics_zones.py     ISO 20816-2 zones A/B/C/D
#     - tests/test_critical_speeds.py         Bode → críticas + API 684
#     - tests/test_machine_templates.py
#     - tests/test_loaders.py
#     - tests/test_api_services.py
#     - tests/run_smoke.py        mini-runner casero (corre sin pytest)
#     - pytest.ini
#     - requirements-dev.txt      pytest + pytest-cov
#
#   ► CATÁLOGO EXTENDIDO DE PLANTILLAS DE MÁQUINAS LATAM
#     - data/machine_templates.json   20 plantillas:
#         · Solar Centaur 40/50, Mars 100, Taurus 60
#         · Siemens SGT-400, SGT-700
#         · GE Frame 5, Frame 7EA
#         · Brush turbogen 54 MW
#         · Solar C30, Atlas Copco ZH (centrífugos)
#         · Ariel KBB, Burckhardt Process (reciprocantes)
#         · Sulzer ZSK, Goulds 3700 (bombas API 610)
#         · WEG W22, ABB AMI, Siemens SIMOTICS HV (motores)
#         · TLT-Turbo, Howden (ventiladores)
#       Cada una: RPM nominal + range, rodamientos típicos, norma ISO/API
#       recomendada, esquema de sensores recomendado, notas técnicas.
#     - core/machine_templates.py    loader robusto (fallback graceful
#                                    si JSON inválido), bridge a profile
#                                    legacy. NO toca machine_profiles.py.
#
#   ► IMPORTADORES UNIVERSALES — argumento de venta directo
#     - core/loaders/base.py        LoadedSignal canónico + loaded_to_signal()
#     - core/loaders/csi2140.py     Emerson CSI 2140 CSV (tiempo + espectro)
#     - core/loaders/adre408.py     Bently Nevada ADRE 408 CSV
#     - core/loaders/uff.py         Universal File Format dataset 58 (SDRC)
#     Argumento: "te migramos en un fin de semana sin tocar tu data".
#
#   ► API REST PÚBLICA v1.0 (read-only)
#     - api/services.py             capa pura testeable sin FastAPI
#     - api/auth.py                 API keys con hmac.compare_digest
#     - api/app.py                  FastAPI factory, 13 endpoints v1.0
#     - api/README.md               quickstart + tabla de endpoints
#     - requirements-api.txt        fastapi + uvicorn + pydantic
#     Endpoints:
#       GET  /v1/health                                       (público)
#       GET  /v1/templates[?category=&manufacturer=]
#       GET  /v1/templates/{id}
#       GET  /v1/templates/{id}/norm-recommendation
#       GET  /v1/templates/categories
#       GET  /v1/norms
#       GET  /v1/norms/groups
#       GET  /v1/norms/{code}
#       GET  /v1/norms/{code}/classes/{class}/thresholds
#       GET  /v1/loaders                                      (capabilities)
#       GET  /v1/bearings[?limit=]
#       GET  /v1/bearings/overlay?model=&rpm=&harmonics=
#
# Cambios técnicos resumidos:
#
# (NUEVO) tests/                       suite pytest + run_smoke.py
# (NUEVO) data/machine_templates.json
# (NUEVO) core/machine_templates.py
# (NUEVO) core/loaders/                csi2140 + adre408 + uff
# (NUEVO) api/                         services + auth + app FastAPI
# (NUEVO) requirements-dev.txt
# (NUEVO) requirements-api.txt
# (NUEVO) pytest.ini
#
# Archivos de producción modificados: NINGUNO.
# git diff --stat HEAD -- core/ pages/ modules/ scripts/ tools/ está vacío.
#
# 157 tests pasando (verificable con: python3 tests/run_smoke.py).
#
# Solo push DEV. NO mergear a main hasta validar manualmente que el
# Streamlit corre idéntico (Reports + AI diagnostics + Trends sin
# regresiones).
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Ciclo 18.1 — Industrial Plumbing Foundations → DEV"
echo "================================================================"
echo ""

# Tag pre-cambio (ancla de retorno por si algo se rompe)
PRE_TAG="pre-ciclo18-1-$(date +%Y%m%d)"
echo "▶ Creando tag de retorno: ${PRE_TAG}"
git tag -f "${PRE_TAG}" 2>/dev/null || true

echo ""
echo "▶ Cambiando a dev..."
# Lógica robusta: si dev existe localmente, switch; si solo está en origin,
# crear local tracking; si no existe en ninguna parte, error claro.
if git show-ref --verify --quiet refs/heads/dev; then
    git checkout dev
elif git show-ref --verify --quiet refs/remotes/origin/dev; then
    git checkout -b dev origin/dev
else
    echo "  ✗ ERROR: branch 'dev' no existe ni local ni en origin."
    echo "    Crear primero con: git checkout -b dev"
    exit 1
fi

echo ""
echo "▶ Pull origin/dev (fast-forward only)..."
git pull origin dev --ff-only || echo "  (sin cambios remotos)"

echo ""
echo "▶ Creando rama de feature..."
git checkout -B feat/ciclo18-1-tests-loaders-api

echo ""
echo "================================================================"
echo " Commit 1/4 — Suite de tests con golden datasets"
echo "================================================================"
git add tests/ pytest.ini requirements-dev.txt 2>/dev/null || true
git commit -m "feat(18.1): pytest suite con 89 tests sobre golden datasets sintéticos

- tests/conftest.py con 9 generadores de señales sintéticas
  (sine, multi-armónico, BPFO impacts, oil whirl, looseness,
  Bode con crítica conocida, órbita XY)
- 89 tests cubriendo: waveform_metrics, order_tracking, tsa,
  bearing_fault_frequencies, iso_thresholds, rotordynamics
  zones (ISO 20816-2 A/B/C/D), detect_critical_speeds + API 684
- tests/run_smoke.py: mini-runner casero que ejecuta la suite
  sin pytest instalado (útil en CI/sandbox)
- pytest.ini + requirements-dev.txt
- CERO modificaciones a core/, pages/, modules/, scripts/" || echo "  (sin cambios)"

echo ""
echo "================================================================"
echo " Commit 2/4 — Catálogo extendido de plantillas LATAM"
echo "================================================================"
git add data/machine_templates.json core/machine_templates.py tests/test_machine_templates.py 2>/dev/null || true
git commit -m "feat(18.1): catálogo extendido de plantillas de máquinas LATAM

- data/machine_templates.json con 20 plantillas comunes en O&G,
  generación y petroquímica LATAM:
  Solar (Centaur 40/50, Mars 100, Taurus 60), Siemens SGT-400/700,
  GE Frame 5/7EA, Brush turbogen 54MW, Solar C30 + Atlas Copco
  centrífugos, Ariel KBB + Burckhardt reciprocantes,
  Sulzer ZSK + Goulds 3700, WEG W22 + ABB AMI + Siemens SIMOTICS HV,
  TLT-Turbo + Howden axial.
- Cada plantilla con: RPM nominal + range, rodamientos típicos,
  norma ISO/API recomendada, esquema de sensores recomendado.
- core/machine_templates.py loader robusto (fallback graceful
  si JSON inválido) + bridge a MachineProfile legacy.
- 21 tests. NO toca machine_profiles.py existente." || echo "  (sin cambios)"

echo ""
echo "================================================================"
echo " Commit 3/4 — Importadores universales CSI / ADRE / UFF"
echo "================================================================"
git add core/loaders/ tests/test_loaders.py 2>/dev/null || true
git commit -m "feat(18.1): importadores universales CSI 2140 / ADRE 408 / UFF

Argumento de venta directo: 'te migramos sin pelear con el incumbente'.

- core/loaders/base.py: LoadedSignal canónico + bridge a Signal
  legacy (loaded_to_signal). Helpers _read_text_input para tolerar
  path | str | bytes | file-like con BOM utf-8/latin-1.
- core/loaders/csi2140.py: Emerson CSI 2140 CSV (modo tiempo y
  espectro). Detecta separador (, ; tab), extrae fs/rpm de
  metadata, infiere fs del time vector si no está declarado.
- core/loaders/adre408.py: Bently Nevada ADRE 408 CSV/TXT
  exports. Maneja cabecera con comillas dobles (Excel SaveAs).
- core/loaders/uff.py: Universal File Format dataset 58 (SDRC),
  ASCII. Soporta even/uneven spacing, function_type 1/2/4.
- 22 tests con archivos sintéticos en memoria.

Cada loader produce LoadedSignal compat con core.signal_registry.
NO modifica modules/csv_loader.py ni el flujo de carga existente." || echo "  (sin cambios)"

echo ""
echo "================================================================"
echo " Commit 4/4 — API REST pública v1.0 (read-only)"
echo "================================================================"
git add api/ requirements-api.txt tests/test_api_services.py 2>/dev/null || true
git commit -m "feat(18.1): API REST pública v1.0 (read-only) — capa servicios + FastAPI

Cuchillo contra el lock-in de System1: integradores y firmas
externas pueden consumir Watermelon vía HTTP estándar.

- api/services.py: capa pura sin FastAPI/Streamlit. Funciones
  testeables con pytest. Todo lo expuesto al exterior pasa por
  aquí — ni FastAPI ni nuestro shell tocan core/ directo.
- api/auth.py: validación API keys con hmac.compare_digest
  (anti timing-attack). Lee de WATERMELON_API_KEYS env var
  o st.secrets.api_keys. hash_for_log() para audit sin
  exponer la key.
- api/app.py: factory FastAPI con 13 endpoints v1.0.
  Auth Authorization: Bearer <key>. Swagger en /docs.
- api/README.md: quickstart + tabla de endpoints.
- requirements-api.txt: fastapi + uvicorn[standard] + pydantic.
- 25 tests sobre la capa de servicios.

Endpoints v1.0:
  GET /v1/health (público)
  GET /v1/templates[?category=&manufacturer=]
  GET /v1/templates/{id}
  GET /v1/templates/{id}/norm-recommendation
  GET /v1/templates/categories
  GET /v1/norms
  GET /v1/norms/groups
  GET /v1/norms/{code}
  GET /v1/norms/{code}/classes/{class}/thresholds
  GET /v1/loaders (capabilities)
  GET /v1/bearings[?limit=]
  GET /v1/bearings/overlay?model=&rpm=&harmonics=

NO existe POST/PUT/DELETE en v1.0 (read-only por seguridad)." || echo "  (sin cambios)"

echo ""
echo "▶ Push de la rama feat..."
git push -u origin feat/ciclo18-1-tests-loaders-api

echo ""
echo "▶ Merge fast-forward → dev..."
git checkout dev
git merge --no-ff feat/ciclo18-1-tests-loaders-api -m "Merge feat/ciclo18-1-tests-loaders-api into dev

Ciclo 18.1 — Industrial Plumbing Foundations:
- pytest suite (157 tests, golden datasets sintéticos)
- catálogo plantillas LATAM (20 máquinas)
- importadores universales (CSI 2140, ADRE 408, UFF)
- API REST pública v1.0 (read-only, FastAPI)

Aditivo puro: cero archivos de producción modificados."

echo ""
echo "▶ Push dev..."
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 18.1 pusheado a DEV"
echo "================================================================"
echo ""
echo " ► VERIFICACIÓN ANTES DE MERGE A MAIN:"
echo ""
echo "   1. Tests deben pasar limpios:"
echo "      python3 tests/run_smoke.py"
echo "      → expect: 157 passed in ~0.3s"
echo ""
echo "   2. Streamlit debe arrancar IDÉNTICO a antes:"
echo "      streamlit run app.py"
echo "      → login OK, sidebar OK, todas las páginas cargan"
echo ""
echo "   3. Reports — NO debe haber regresión (cero cambios en core):"
echo "      Generar un PDF de reporte en wm-test"
echo "      → expect: idéntico al de v3.13.0"
echo ""
echo "   4. AI Diagnóstico — NO debe haber regresión:"
echo "      En Spectrum / Trends / Bode pulsar 'Diagnóstico AI'"
echo "      → expect: misma respuesta que en v3.13.0"
echo ""
echo "   5. Trends — el módulo más sensible NO se tocó:"
echo "      Cargar 3-4 trends multi-fecha"
echo "      → expect: ranking VFD/VSD igual que v3.13.0"
echo ""
echo "   6. API REST opcional:"
echo "      pip install -r requirements-api.txt"
echo "      export WATERMELON_API_KEYS=\"dev-key-123\""
echo "      uvicorn api.app:app --port 8000"
echo "      curl http://localhost:8000/v1/health"
echo ""
echo " ► Cuando todo OK → ejecutar:"
echo "   bash _publish_v3_14_0_to_main.sh"
echo ""
echo " ► Si algo se rompe (NO debería, todo es aditivo):"
echo "   git checkout ${PRE_TAG}"
echo "   git checkout -b rescue/pre-ciclo18-1"
echo ""
echo "================================================================"
