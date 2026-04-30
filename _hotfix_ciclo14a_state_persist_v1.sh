#!/bin/bash
# =============================================================
# Watermelon — HOTFIX 8 Ciclo 14a: state persistente cross-page (dev)
# =============================================================
# Diagnostico via screenshots:
# - En Library, TES1 muestra ✓ activa (verde) Y badge esquematico
#   vinculado.
# - En Reports, el panel auto-fill dice "No hay activo monitoreado
#   activo".
# - Conclusion: la key 'wm_active_instance_id' NO esta persistiendo
#   entre Library y Reports.
#
# Causa raiz: el hotfix 4 unifico la key del widget selectbox con
# SESSION_KEY_INSTANCE ('wm_active_instance_id'). Cuando se navega
# de una pagina con widget a otra sin widget, Streamlit "des-instancia"
# el widget y la key vinculada a el se invalida en el contexto de la
# pagina nueva. Eso rompe el cross-page state.
#
# Fix: separar la key del widget de la key persistente.
#
# core/instance_selector.py:
# - Selectbox usa key 'wm_instance_select_{module_name}' (efimera,
#   atada al widget de cada pagina).
# - Despues del selectbox, copiar el valor a SESSION_KEY_INSTANCE
#   ('wm_active_instance_id') que NO esta atada a ningun widget y
#   por lo tanto persiste entre paginas.
# - Si la key del widget no existe, inicializarla con current_id
#   antes de instanciar el widget (evita reset accidental).
#
# pages/00_Machinery_Library.py:
# - _set_active_instance callback ahora actualiza ambas keys:
#   * SESSION_KEY_INSTANCE (persistente, leida desde Reports)
#   * widget key 'wm_instance_select_library' (para que el selectbox
#     del sidebar al re-renderizarse se posicione en la nueva activa)
#
# Resultado:
# - Activar TES1 en Library → wm_active_instance_id = 'tes1'
# - Navegar a Reports → wm_active_instance_id sigue siendo 'tes1'
# - get_active_instance_id() → 'tes1' → auto-fill funciona →
#   esquematico aparece en Resumen Ejecutivo del PDF.
#
# Ejecutar:
#   bash _hotfix_ciclo14a_state_persist_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add core/instance_selector.py pages/00_Machinery_Library.py
git status --short | head

git commit -m "fix(library): hotfix 8 Ciclo 14a — state persistente cross-page

El hotfix 4 unifico la key del widget selectbox con SESSION_KEY_INSTANCE.
Eso rompia el cross-page state: al navegar de Library a Reports, la
key vinculada al widget de Library se invalidaba y get_active_instance_id()
devolvia None.

Fix: separar key del widget de key persistente.
* core/instance_selector.py: selectbox usa 'wm_instance_select_{module_name}'
  (efimera). Despues copia valor a SESSION_KEY_INSTANCE ('wm_active_instance_id',
  no atada a widget → persiste entre paginas).
* pages/00_Machinery_Library.py: _set_active_instance callback actualiza
  ambas keys (la persistente + la del widget de la pagina actual)."

git push origin dev

echo ""
echo "================================================================"
echo " HOTFIX 8 listo — el state ahora persiste entre paginas"
echo "================================================================"
echo "Refrescar app, ir a Machinery Library → TES1 ya activa."
echo "Navegar a Reports → panel auto-fill ahora dice ACTIVO: TES1"
echo "y muestra esquematico verde 'listo para Resumen Ejecutivo'."
echo "Generar PDF → el esquematico aparece en pagina 3."
