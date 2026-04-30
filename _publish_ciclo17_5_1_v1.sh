#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.5.1 → DEV: Limpieza Trend
# =============================================================
# Hotfix sobre 17.5 con tres limpiezas pequeñas pedidas tras
# probar el módulo:
#
# (1) Fix module_name: render_instance_selector("trend") →
#     "trends". El profile registra applicable_modules = [...,
#     "trends", ...] así que con la clave en singular siempre
#     mostraba el warning amarillo "El profile no incluye trend".
#
# (2) Asset context heredado de la instancia activa. Antes el
#     usuario tenía que reescribir asset_type, configuración
#     mecánica, primary/secondary equipment y descripción
#     técnica desde la sidebar de Trends — todo eso ya está en
#     Machinery Library. Se eliminó el bloque "Machine
#     Diagnostic Context" entero y se construye `asset_context`
#     vía _build_trend_asset_context_from_instance(state) usando
#     profile_label, machine_group, tag, location, notes. Las
#     legacy session keys (wm_tr_asset_type, wm_tr_machine_*) se
#     siguen sincronizando para no romper consumidores.
#
#     Las validaciones de "Asset type is required..." que
#     bloqueaban el envío al reporte se eliminaron — el contexto
#     siempre está poblado mientras haya una instancia activa.
#
# (3) Marcadores de anomalías sutiles. Antes eran X rojos
#     tamaño 11 que saturaban el gráfico. Ahora son círculos
#     huecos coloreados por severidad:
#       - High   → rojo opacidad 0.85, size 9
#       - Medium → ámbar opacidad 0.70, size 7
#       - Low    → gris opacidad 0.55, size 6
#     El trend line vuelve a ser el elemento dominante y las
#     anomalías quedan como anotaciones discretas sobre la
#     curva.
#
# Ejecutar:
#   bash _publish_ciclo17_5_1_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/04_Trends.py
git add _publish_ciclo17_5_1_v1.sh
git status --short | head

git commit -m "fix(trend): Ciclo 17.5.1 — limpieza módulo Trend (asset_context heredado + anomalías sutiles)

(1) Fix render_instance_selector module_name 'trend' -> 'trends'.
El profile registra 'trends' (plural) en applicable_modules, asi
que con la clave en singular siempre mostraba el warning amarillo
'El profile no incluye trend'.

(2) Asset context heredado automaticamente de la instancia activa
(Machinery Library). Se elimina el bloque 'Machine Diagnostic
Context' entero (asset_type, configuracion, primary/secondary
equipment, descripcion tecnica) — todos esos campos ya estan en
la instancia. _build_trend_asset_context_from_instance() deriva
asset_type por heuristica sobre profile_label/machine_group y
machine_description concatenando profile_label + tag + location +
notes. Legacy session keys (wm_tr_asset_type, wm_tr_machine_*) se
siguen sincronizando. Validaciones bloqueantes en _send_to_report
eliminadas (el contexto siempre esta poblado).

(3) Marcadores de anomalias sutiles. Antes: X rojos size 11
saturando el grafico. Ahora: circulos huecos por severidad
(High=rojo 0.85 size 9, Medium=ambar 0.70 size 7, Low=gris 0.55
size 6). El trend line vuelve a ser dominante visual." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.5.1 (limpieza Trend) pusheado a dev"
echo "================================================================"
echo ""
echo "Cambios visibles:"
echo "  • Sidebar: ya no pide asset_type ni descripcion (vienen de"
echo "    la instancia activa) — sidebar mucho mas corta."
echo "  • Plot: anomalias mas discretas, no compiten con la curva."
echo "  • Sin warning amarillo de 'profile no incluye trend'."
echo "================================================================"
