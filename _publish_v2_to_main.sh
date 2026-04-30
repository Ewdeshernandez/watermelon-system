#!/bin/bash
# =============================================================
# Watermelon — v2.0: Asset Instances + Auto-cálculos + Supabase
# =============================================================
# Cierra Ciclos 8 + 9 mergeando dev → main como v2.0.
# Después vuelve a dev para arrancar Ciclo 10 (Reports clase mundial).
#
# Cambios incluidos en v2.0 (todo lo que está en dev y NO en main):
#   - Ciclo 8: Asset Instances (data por máquina física específica)
#       * core/instance_state.py
#       * core/instance_selector.py
#       * core/bearing_calculations.py (auto-cálculos vivos)
#       * pages/17_Asset_Documents.py reescrita
#       * pages/06/07/09 usan instance selector
#   - Ciclo 9: Persistencia real Supabase
#       * core/instance_repository.py (Local + Supabase backends)
#       * data/supabase_schema.sql
#       * docs/supabase_setup.md
#       * requirements.txt: +supabase
#       * .streamlit/secrets.toml.example actualizado
#
# Validado en producción (wm-test.streamlit.app):
#   * Badge ☁️ Persistencia Supabase activa visible
#   * Auto-bootstrap creó brush_default con 13 parámetros del seed
#   * Auto-cálculos en vivo: Cd=0.382 mm, Cr=7.52 mil pp
#   * Persistencia confirmada: cambios sobreviven reboot del container
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon v2.0 → main"
echo "================================================================"
echo ""

[ -f .git/index.lock ] && rm -f .git/index.lock

# 1. Asegurar que estamos en dev limpio
git checkout dev
git pull origin dev
git status --short

# 2. Tag de retorno antes del merge
echo ""
echo "[1] Creando tag v2.0-pre-main sobre dev..."
git tag -a v2.0-pre-main -m "Estado de dev justo antes de mergear v2.0 a main.
Incluye Ciclos 8 (Asset Instances + Auto-calculos) + 9 (Supabase persistence).
Validado en wm-test.streamlit.app con badge supabase activo."
git push origin v2.0-pre-main
echo "    OK"
echo ""

# 3. Merge dev → main
echo "[2] Mergeando dev → main..."
git checkout main
git pull origin main
git merge --no-ff dev -m "merge: dev v2.0 → main — Asset Instances + Supabase persistence

Release mayor v2.0. Cambios arquitecturales:

* Asset Instances: cada maquina fisica tiene su propio ID con sus
  parametros, documentos e historico. Antes la asociacion era por
  profile_key (familia/tipo) lo que mezclaba data entre maquinas
  distintas del mismo modelo.
* Auto-calculos vivos en el formulario de parametros: Cd, Cr, L/D,
  carga unitaria, lift-off speed estimate.
* Persistencia real con Supabase Postgres + Storage. La app elige
  automaticamente el backend segun st.secrets[supabase]:
    - Sin secret: filesystem local (efimero, igual que v1.x)
    - Con secret: Supabase (sobrevive cualquier redeploy)
* Backwards compatibility: v1.x sigue funcionando si no se configura
  Supabase. Las semillas del Brush 54MW siguen poblando brush_default
  automaticamente al arrancar."
git push origin main
echo "    OK"
echo ""

# 4. Tag v2.0
echo "[3] Tag v2.0 sobre main..."
git tag -a v2.0 -m "Release v2.0 — Asset Instances + Persistencia real.

Cambios mayores:
* Modelo de datos refactorizado: cada maquina fisica = una Instance
  con su propio Vault. No mas mezcla entre activos.
* Auto-calculos derivados (Cd, Cr, L/D, carga unitaria) en vivo en
  el formulario de parametros, con interpretacion textual.
* Persistencia Supabase optional — la app sobrevive cualquier
  redeploy de Streamlit Cloud cuando esta configurado.
* Lift-off speed estimate, unit load, L/D ratio con thresholds
  interpretables (textbook hidrodinamico).

Validado en produccion (Brush 54 MW TES1):
* badge supabase activo, datos persistidos, auto-bootstrap del seed
  funcional, parametros editables que sobreviven reboot."
git push origin v2.0
echo "    OK"
echo ""

# 5. Volver a dev y reincorporar el merge
echo "[4] Volviendo a dev..."
git checkout dev
git merge main
git push origin dev
echo "    OK"
echo ""

# 6. Verificacion
echo "================================================================"
echo " v2.0 publicado a main"
echo "================================================================"
echo ""
git log --oneline --decorate -10
echo ""
echo "Estas de vuelta en dev. Listo para arrancar Ciclo 10:"
echo "Reports clase mundial — analisis automatico Cat IV."
echo "================================================================"
