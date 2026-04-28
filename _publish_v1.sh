#!/bin/bash
# =============================================================
# Watermelon System — Publicación v1.0 a main
# =============================================================
# Este script hace TODO el trabajo local (commit + merge + tag)
# en feat/scl-cat-iv → dev → main, con tags v1.0-pre-main y v1.0.
#
# NO hace push. Los pushes están en _push_v1.sh para que requieran
# tu confirmación de credenciales por separado.
#
# Ejecutar desde el root del repo:
#   bash _publish_v1.sh
#
# Si algo falla, el script se detiene y muestra el error.
# Tu working directory NO se pierde porque cada paso valida
# antes de continuar.
# =============================================================

set -e  # detenerse al primer error

cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " WATERMELON SYSTEM — Publicación v1.0"
echo "================================================================"
echo ""

# --------------------------------------------------------------
# 0. Sanity checks
# --------------------------------------------------------------
echo "[0] Verificando estado inicial..."
CURRENT_BRANCH=$(git branch --show-current)
echo "    Rama actual: $CURRENT_BRANCH"

if [ "$CURRENT_BRANCH" != "feat/scl-cat-iv" ]; then
  echo ""
  echo "  ! Atención: NO estás en feat/scl-cat-iv (estás en $CURRENT_BRANCH)."
  echo "  ! Cambiando a feat/scl-cat-iv automáticamente..."
  git checkout feat/scl-cat-iv
fi

# Limpiar lock files si existen (de procesos git anteriores que se cortaron)
if [ -f .git/index.lock ]; then
  echo "    Removiendo .git/index.lock huérfano..."
  rm -f .git/index.lock
fi
if [ -f .git/HEAD.lock ]; then
  rm -f .git/HEAD.lock
fi

echo "    OK"
echo ""

# --------------------------------------------------------------
# 1. Commit todo lo pendiente en feat/scl-cat-iv
# --------------------------------------------------------------
echo "[1] Adoptando archivos modificados y nuevos en feat/scl-cat-iv..."
git add -A

# Verificar si hay algo para commitear
if git diff --cached --quiet; then
  echo "    No hay cambios pendientes para commitear (commit anterior ya hecho)."
else
  echo "    Creando commit con todo el trabajo de los Ciclos 2-6 + sanitización..."
  git commit -m "feat: SCL Cat IV + reporte clase mundial + sanitizacion v1.0

Ciclo 3 - Asset Profiles:
* core/machine_profiles.py: 16 perfiles (Brush 54MW, GE LM6000, Siemens SGT,
  motores 2/4/6 polos, bombas multietapas H/V, recip motors/compressors)
* core/profile_state.py: selector de perfil persistente por sesion
* iso_part / machine_group / threshold_strategy propagados a todos los modulos

Ciclo 4 - Document Vault:
* core/document_vault.py: 15 tipos de documento, 26 parametros capturados
* pages/17_Asset_Documents.py: subida de manuales OEM, reportes historicos
* Storage en data/asset_documents/{profile}/ + metadata json

Ciclo 5 - Shaft Centerline Cat IV:
* core/scl_diagnostics.py: EccentricityState, CenterlineMigration,
  compute_eccentricity_state, compare_centerline_migration, lift-off detect
* pages/09_Shaft_Centerline.py: integracion Vault + perfil + override manual
* core/diagnostics.py: build_scl_diagnostics_rotordyn (narrativa Cat IV)

Bottom load reference (API 670 / practica estandar):
* Bearing center en (0, +Cr) por convencion industrial estandar
* Eccentricity rings 0.40/0.70/0.85, REST/BEARING markers, load arrow
* Validado contra Brush real: alpha=34.4 textbook (vs frame Origin alpha=155)

Comparativo SCL multi-fecha enriquecido:
* Markers de punto operativo por fecha, vectores de migracion delta+%c
* Linea de attitude angle, narrativa Cat IV multi-fecha

Ciclo 6 - Reporte clase mundial:
* assets/fonts/IBMPlexSans-{Regular,Bold}.ttf + DejaVuSans fallback
* Fontfamily registrada para que <b> resuelva al peso bold
* Secciones 1/2/3 ocultas si vienen vacias (renumeracion automatica)
* Tabla nativa para bloques --- RESUMEN ---
* Boton Auto-redactar 1/2/3 desde figuras

Camino A - Resumen Ejecutivo:
* _extract_findings_from_items: parsea narrativas y extrae e/c, alpha,
  migracion, Q-factor, ISO zones, lift-off, prioridad alta
* _global_severity: 5 niveles (CONDICION ACEPTABLE -> CRITICA)
* _compose_executive_summary: 4 parrafos en prosa Cat IV
* PDF inserta RESUMEN EJECUTIVO con cinta de severidad coloreada

Sanitizacion comercial:
* Cero menciones a Bently Nevada / Bently/Adapt / CSV en narrativas
* Frame citado como API 670 / practica estandar (sin marca)
* Nuevo campo train_description para trenes acoplados (turbina+generador)
* Auto-draft con lenguaje pro: API 670/684, ISO 20816/21940, Cat IV VI,
  monitoreo en linea Watermelon System, DCS, Document Vault
* _paragraph_safe whitelist <b><i><sub><sup>"
  echo "    Commit creado."
fi
echo ""

# --------------------------------------------------------------
# 2. Limpiar tags mal puestos (estaban sobre feat/scl-cat-iv en vez de main)
# --------------------------------------------------------------
echo "[2] Limpiando tags v1.0 / v1.0-pre-main mal posicionados (si existen)..."
git tag -d v1.0 2>/dev/null && echo "    Tag v1.0 local borrado." || echo "    Tag v1.0 local no existía (OK)."
git tag -d v1.0-pre-main 2>/dev/null && echo "    Tag v1.0-pre-main local borrado." || echo "    Tag v1.0-pre-main local no existía (OK)."
echo ""

# --------------------------------------------------------------
# 3. Mergear feat/scl-cat-iv -> dev
# --------------------------------------------------------------
echo "[3] Mergeando feat/scl-cat-iv -> dev..."
git checkout dev
echo "    En dev. Mergeando..."
git merge --no-ff feat/scl-cat-iv -m "merge: feat/scl-cat-iv -> dev — Watermelon v1.0 (SCL Cat IV + reporte + sanitizacion)"
echo "    Merge a dev OK."
echo ""

# --------------------------------------------------------------
# 4. Tag de retorno v1.0-pre-main sobre dev
# --------------------------------------------------------------
echo "[4] Creando tag v1.0-pre-main sobre dev..."
git tag -a v1.0-pre-main -m "Estado de dev justo antes de la primera publicacion v1.0 a main"
echo "    Tag creado."
echo ""

# --------------------------------------------------------------
# 5. Mergear dev -> main
# --------------------------------------------------------------
echo "[5] Mergeando dev -> main..."
git checkout main
echo "    En main. Mergeando..."
git merge --no-ff dev -m "merge: dev v1.0 -> main — Watermelon System reporte clase mundial"
echo "    Merge a main OK."
echo ""

# --------------------------------------------------------------
# 6. Tag de release v1.0 sobre main
# --------------------------------------------------------------
echo "[6] Creando tag v1.0 sobre main..."
git tag -a v1.0 -m "Release v1.0 — Watermelon System reporte clase mundial:
* SCL Cat IV con convencion API 670 (bottom load reference)
* Eccentricity ratio + attitude angle + lift-off + migracion multi-fecha
* Polar y Bode con criticas (factor Q, FWHM) y zonas ISO 20816
* Asset Profiles (16 tipos) + Document Vault del activo
* Reporte PDF con IBM Plex Sans, Resumen Ejecutivo + cinta de severidad
* Tabla nativa, secciones inteligentes, soporte de trenes acoplados
* Narrativas sanitizadas, lenguaje normativo API/ISO/Cat IV"
echo "    Tag v1.0 creado."
echo ""

# --------------------------------------------------------------
# 7. Volver a dev y reincorporar el merge para mantener lineage
# --------------------------------------------------------------
echo "[7] Volviendo a dev y alineando con main..."
git checkout dev
git merge main
echo "    dev al dia con main."
echo ""

# --------------------------------------------------------------
# 8. Resumen final
# --------------------------------------------------------------
echo "================================================================"
echo " TODO LISTO LOCALMENTE — falta hacer el push a GitHub"
echo "================================================================"
echo ""
echo "Estado actual del repo:"
git log --oneline --all --decorate -8
echo ""
echo "Próximo paso: ejecutar el push a GitHub:"
echo ""
echo "  bash _push_v1.sh"
echo ""
echo "Eso empuja dev, main y los dos tags. Después ya verás los"
echo "cambios en watermelonsystem.app a los pocos minutos."
echo "================================================================"
