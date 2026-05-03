#!/bin/bash
# =============================================================
# Watermelon — v3.1.0 → MAIN: Lote acumulado de dev
# =============================================================
# Junta TODO el trabajo desde v3.0.8 (último tag estable):
#
# 17.6   — Login premium internacional (paleta corporativa
#          azul + tipografía sobria + trust chips + glassmorphism)
#        — Editor de Reports UNIFICADO (4 secciones full-width
#          con auto-redactar individual cada una)
#
# 17.6.1 — Login fix recuadro rojo + cajas blancas fantasma
#          (override BaseWeb total + :has() en columnas en
#          lugar de wrappers div)
#
# 17.6.2 — Login fix doble borde en inputs (solo el wrapper
#          OUTER de BaseWeb pinta borde, INNER transparente)
#
# 17.7   — Versión del sistema visible cross-app:
#          - core/version.py auto-deriva del tag semver más alto
#            del repo (sin importar branch — clave porque tags
#            v3.x viven en main sobre merge commits)
#          - Login footer: chip de environment coloreado +
#            commit SHA + fecha
#          - Sidebar: caption monoespaciado debajo del logout
#          - PDF Reports: footer trazabilidad en portada y
#            páginas internas
#
# 17.8   — Trend module operational parser CLASE MUNDIAL:
#          - Sort + dedup automático de timestamps (DCS exports
#            interleaves retries, antes daban resultados raros)
#          - Date M/D/YYYY (US Honeywell/Emerson) con fallback
#            a EU + ISO + auto-infer
#          - Familias extendidas: pressure, flow, frequency,
#            speed, vibration (antes solo power/temperature)
#          - Humanización de labels DCS: '[C200C]TIT_200AXPV' →
#            'Temp 200 (TIT)', '[BL1_BPCS]VFD_*VSD_Freq' →
#            'VFD Frecuencia'
#          - Data quality banner post-load: N variables · M
#            muestras · ventana T · familias detectadas
#          - Auto-select primeras 3 variables si no hay
#            selección previa
#          - Quick-pick por familia: botones para cargar todas
#            las temperaturas / presiones / flujos / VFD de un
#            click. Práctico cuando el CSV tiene 12+ variables.
#          - Validado contra CSV oficial C-200C 18-25 ABR 2026
#            (Hyundai HNP2 + Ariel KBK/4): 12/12 columnas
#            clasificadas correctamente.
#
# Tag v3.1.0 (no patch porque trae features nuevas: parser
# extendido + UI rediseñada).
#
# Ejecutar:
#   bash _publish_v3_1_0_to_main.sh
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.1.0"
RELEASE_TITLE="Login premium + Versión visible + Trend operational clase mundial"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

# ---------- 0) Asegurar branch dev ----------
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# ---------- 1) Stage TODO lo modificado en pages/ y core/ ----------
echo ""
echo "▶ Stageando cambios pendientes..."
git add core/version.py
git add core/auth.py
git add pages/00_Login.py
git add pages/04_Trends.py
git add pages/16_Reports.py
git add _publish_v3_1_0_to_main.sh 2>/dev/null || true
git add _publish_ciclo17_6_dev.sh 2>/dev/null || true
git add _publish_ciclo17_6_1_dev.sh 2>/dev/null || true
git add _publish_ciclo17_6_2_dev.sh 2>/dev/null || true
git add _publish_ciclo17_7_dev.sh 2>/dev/null || true

# ---------- 2) Commit consolidado (si hay cambios staged) ----------
if ! git diff --staged --quiet; then
    echo "▶ Commiteando lote 17.6+17.7+17.8..."
    git commit -m "feat(login+trend+version): lote acumulado para v3.1.0

17.6   Login premium internacional (paleta corporativa azul +
       tipografia sobria + trust chips + glassmorphism). Editor
       de Reports UNIFICADO con 4 secciones full-width y auto-
       redactar individual.

17.6.1 Login fix recuadro rojo + cajas blancas fantasma
       (override BaseWeb total + :has() en columnas en lugar
       de wrappers div).

17.6.2 Login fix doble borde input (solo OUTER pinta borde,
       INNER transparente).

17.7   Version visible cross-app: core/version.py auto-deriva
       del tag semver mas alto del repo (sin importar branch).
       Login footer + sidebar + PDF Reports muestran version +
       environment + commit SHA + fecha.

17.8   Trend module operational parser CLASE MUNDIAL:
       - Sort+dedup timestamps (DCS exports interleave retries)
       - Date M/D/YYYY US + EU + ISO auto-infer
       - Familias extendidas: pressure, flow, frequency, speed,
         vibration (antes solo power/temperature)
       - Humanizacion labels DCS: TIT_200AXPV -> Temp 200 (TIT)
       - Data quality banner post-load
       - Auto-select primeras 3 variables
       - Quick-pick chips por familia
       - Validado contra CSV oficial C-200C: 12/12 OK" || echo "  (sin cambios)"
else
    echo "  (no hay cambios staged)"
fi

# ---------- 3) Reconciliar contra origin/dev ----------
echo ""
echo "▶ Reconciliando contra origin/dev..."
git fetch origin dev
git pull --rebase origin dev || {
    echo ""
    echo "✗ Rebase falló. Resolvelos a mano y re-ejecutá."
    exit 1
}

# ---------- 4) Push dev ----------
echo ""
echo "▶ Pusheando dev..."
git push origin dev

# ---------- 5) Switch a main + pull ----------
echo ""
echo "▶ Cambiando a main..."
git checkout main
git pull origin main

# ---------- 6) Merge dev → main ----------
echo ""
echo "▶ Mergeando dev → main..."
git merge dev --no-ff -m "release(${VERSION}): ${RELEASE_TITLE}

Lote acumulado desde v3.0.8:

== LOGIN PREMIUM INTERNACIONAL (17.6 + 17.6.1 + 17.6.2) ==
Paleta corporativa azul (#21478c, no cyan brillante).
Trust chips (SSO-ready, API 670/684, ISO 20816, Multi-instance).
Glassmorphism sutil en la card. Hero con accent gradient.
Tagline corporativa explicita.

Fixes: recuadro rojo persistente killed (override agresivo
BaseWeb [data-baseweb=input], focus-visible, autofill, invalid).
Cajas blancas fantasma killed (técnica :has() para detectar
columna y aplicar card style sin div wrapper). Doble borde en
inputs killed (solo wrapper OUTER pinta borde, INNER
transparente).

== EDITOR DE REPORTS UNIFICADO (17.6) ==
4 secciones (Resumen ejecutivo + Objetivo + Desarrollo +
Recomendaciones) ahora full-width consistentes con auto-
redactar individual por seccion. Helper
_autodraft_single_section() permite regenerar 1 seccion sin
tocar las otras.

== VERSION VISIBLE (17.7) ==
core/version.py NUEVO. Single source of truth con resolucion
por prioridad: WM_VERSION env -> latest semver tag (regardless
of branch) -> git describe -> VERSION file -> fallback. lru_cache
para no fork-ear git en cada rerun. Visible en login footer,
sidebar inferior y PDF Reports footer (trazabilidad para
auditoria).

== TREND OPERATIONAL CLASE MUNDIAL (17.8) ==
Parser DCS-grade del CSV operacional:
- Sort + dedup automatico (DCS exports interleave retries,
  daban resultados desordenados antes)
- Date format M/D/YYYY (US Honeywell/Emerson) + EU + ISO con
  auto-infer
- Familias extendidas: pressure, flow, frequency, speed,
  vibration (antes solo power/temperature)
- Humanizacion de labels DCS:
    [C200C]TIT_200AXPV -> Temp 200 (TIT)
    [BL1_BPCS]AGA3_FIT_*Flow -> Flow (FIT)
    [BL1_BPCS]VFD_*VSD_Freq -> VFD Frecuencia
- Data quality banner post-load (N vars + N muestras + ventana
  + familias detectadas)
- Auto-select primeras 3 variables si no hay seleccion previa
- Quick-pick chips por familia (cargar todas las temperaturas
  o presiones de un click)
- Validado contra CSV oficial C-200C 18-25 ABR 2026: 12/12
  columnas clasificadas correctamente."

# ---------- 7) Tag de release ----------
echo ""
echo "▶ Creando tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Lote acumulado sobre v3.0.8:

  Login premium internacional con paleta corporativa azul,
  trust chips, glassmorphism sutil. Bordes rojos eliminados,
  cajas blancas fantasma resueltas, doble borde de inputs
  arreglado.

  Editor de Reports unificado: 4 secciones full-width con
  auto-redactar individual.

  Version visible cross-app (login + sidebar + PDF) auto-
  derivada del tag semver mas alto del repo.

  Trend module operational parser DCS-grade: sort + dedup,
  date M/D/YYYY, familias pressure/flow/frequency/speed,
  labels humanizados, data quality banner, quick-pick por
  familia. Validado contra CSV oficial C-200C."

# ---------- 8) Push main + tag ----------
echo ""
echo "▶ Pusheando main + ${VERSION}..."
git push origin main
git push origin "${VERSION}"

# ---------- 9) Volver a dev ----------
echo ""
echo "▶ Volviendo a dev..."
git checkout dev

echo ""
echo "================================================================"
echo " ✓ ${VERSION} publicado a main"
echo "================================================================"
echo ""
echo " Branch actual: dev (lista para próximos ciclos)"
echo " Tag creado:    ${VERSION}"
echo ""
echo " Streamlit Cloud auto-redeploya en ~2 min."
echo " Después del deploy, refrescá Cmd+Shift+R y verificá:"
echo "   1. Login: paleta azul corporativa, sin recuadros rojos,"
echo "      sin cajas blancas, footer con 'v3.1.0 [PRODUCTION]'"
echo "   2. Sidebar: caption 'v3.1.0 ● production' al pie"
echo "   3. Reports editor: 4 secciones full-width con auto-redactar"
echo "      individual cada una"
echo "   4. Trends: subir CSV oficial del C-200C → ver banner"
echo "      verde con 12 variables, familias detectadas, 3"
echo "      auto-seleccionadas, botones quick-pick por familia"
echo "   5. PDF Reports generado → footer dice 'Watermelon System"
echo "      v3.1.0' en gris tenue"
echo "================================================================"
