#!/bin/bash
# =============================================================
# Watermelon — HOTFIX 4 Ciclo 14a: 'cannot be modified after widget' (dev)
# =============================================================
# El hotfix anterior intentaba sincronizar dos state keys cuando se
# clickeaba "Activar", pero Streamlit prohibe modificar una state key
# que ya es widget key fuera de un callback:
#
#   StreamlitAPIException: st.session_state.wm_instance_select_documents
#   cannot be modified after the widget with key
#   wm_instance_select_documents is instantiated.
#
# Causa: el sidebar selectbox y el grid escriben a la MISMA key, pero
# el grid lo hace después de que el widget se instancie. La unica
# manera legal en Streamlit es via on_click callback (corre en una
# fase especial antes del próximo render).
#
# Fix correcto + de raíz:
#
# 1) core/instance_selector.py
#    Cambia la key del selectbox de 'wm_instance_select_{module_name}'
#    (una distinta por módulo) a SESSION_KEY_INSTANCE = 'wm_active_instance_id'
#    (única, compartida entre todos los módulos). Eso elimina el
#    desfase entre la key del widget y la key principal de state, que
#    era la raíz del bug original. El selectbox y el resto del sistema
#    leen y escriben a la misma key.
#
# 2) pages/00_Machinery_Library.py
#    El botón "Activar" usa on_click=_set_active_instance, args=(inst_id,).
#    El callback corre en la fase pre-render donde se permite modificar
#    cualquier key (incluso keys de widgets ya instanciados en el
#    script anterior). Después Streamlit hace rerun automático y el
#    selectbox del sidebar lee el nuevo valor de SESSION_KEY_INSTANCE
#    y se sincroniza solo.
#
# Ejecutar:
#   bash _hotfix_ciclo14a_activar_v2.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add core/instance_selector.py pages/00_Machinery_Library.py
git status --short | head

git commit -m "fix(library): hotfix 4 Ciclo 14a — boton Activar via on_click callback + key unificada

El hotfix anterior intentaba sincronizar dos state keys directamente,
pero Streamlit prohibe modificar widget keys fuera de un callback.

Fix de raiz:
* core/instance_selector.py: la key del selectbox pasa de
  'wm_instance_select_{module_name}' a SESSION_KEY_INSTANCE
  ('wm_active_instance_id'). Una sola fuente de verdad para todos
  los modulos (Polar/Bode/SCL/Library/...).
* pages/00_Machinery_Library.py: boton 'Activar' usa on_click callback
  (_set_active_instance) que corre en la fase pre-render donde
  session_state se puede escribir libremente. Despues Streamlit hace
  rerun automatico y el selectbox lee el nuevo valor."

git push origin dev

echo ""
echo "Refrescar app — ahora click 'Activar' en cualquier card debe"
echo "cambiar instantaneamente la maquina activa, sin error."
