#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.4.2 HOTFIX → MAIN
# =============================================================
# HOTFIX URGENTE Ciclo 17.18 — directo a main.
#
# Bug que matamos:
#   El plan Free de Supabase muestra "Exceeding usage limits" con
#   Cached Egress 8.6 GB / 5 GB (173% del cap mensual). El bucket
#   instance-documents en sí pesa apenas 2 MB — el problema NO es
#   storage, es que la app descargaba el mismo archivo MUCHAS veces.
#
# Causa raíz:
#   get_instance_document_bytes() llamaba directo a
#   repo.download_document_bytes() en cada invocación. En Streamlit,
#   cada interacción del usuario (slider, selectbox, click) dispara
#   un rerun, y cada rerun re-descargaba los archivos completos del
#   bucket. Con specialists trabajando todo el día → 8.6 GB en 4 días.
#
# Fix:
#   Cache local en disco en data/cache/instance_documents/ con TTL
#   de 30 días. La primera vez que se pide un archivo se baja del
#   bucket; después sale del disco. Sobrevive reruns y redeploys de
#   Streamlit Cloud. Invalidación automática en upload/remove.
#
# Validado con smoke test (_test_17_18_document_cache.py):
#   - HIT/MISS funcionan correctamente
#   - invalidate_document(specific) y invalidate_document(instancia)
#   - TTL: archivos viejos se re-bajan, nuevos sirven desde cache
#   - clear_all_cache() limpia todo
#   - None responses no se cachean (no contamina cache)
#   - Simulación 30 reruns: 96.7% reducción de egress
#
# Impacto esperado en producción:
#   8.6 GB / 4 días → estimado <500 MB / 4 días (>95% reducción).
#   Sale del red del cap del Free tier.
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🚨 HOTFIX v3.4.2 → MAIN  (fix Supabase Cached Egress 173%)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.18  Cache local de Supabase Storage (data/cache/...)"
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el release a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

echo "▶ 1/7  Commit del cache en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/document_cache.py core/instance_state.py \
            _test_17_18_document_cache.py _release_v3_4_2_hotfix.sh
    git commit -m "feat(17.18): cache local de Supabase Storage — fix Cached Egress 173%

Bug: el plan Free de Supabase mostraba 'Exceeding usage limits' con
Cached Egress 8.6 GB / 5 GB. El bucket pesa apenas 2 MB — el problema
era que get_instance_document_bytes() llamaba directo al repo en cada
rerun de Streamlit, multiplicando el egress por cada interacción del
usuario.

Fix:
- Nuevo módulo core/document_cache.py con cache en disco
  (data/cache/instance_documents/), TTL 30 días, write atómico
- get_instance_document_bytes y get_instance_document_path ahora
  usan cached_download_bytes (lazy import, sin circulares)
- Invalidación automática en upload (add_instance_document_from_streamlit)
  y remove (remove_instance_document)
- Sobrevive reruns Y redeploys de Streamlit Cloud (cache en disco
  no en memoria)
- API publica intacta — los 8+ callers no cambian

Smoke test (_test_17_18_document_cache.py):
- HIT/MISS funcionan correctamente
- invalidate_document por archivo y por instancia
- TTL expirado fuerza MISS
- clear_all_cache limpia todo
- None responses no se cachean
- Simulación 30 reruns: 96.7% reducción de egress

Impacto esperado en producción:
8.6 GB/4 días → <500 MB/4 días (>95% reducción), saliendo del cap
del plan Free." || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene el cache commiteado"
echo ""

echo "▶ 2/7  Push de dev a origin..."
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Pull dev falló."; exit 1; }
git push origin dev || { echo "✗ Push dev falló."; exit 1; }
echo "  ✓ dev en origin actualizado"
echo ""

echo "▶ 3/7  Cambiando a main..."
git checkout main || { echo "✗ No se pudo cambiar a main."; exit 1; }
git fetch origin main
git pull --rebase origin main || { echo "✗ Pull main falló."; exit 1; }
echo "  ✓ main actualizado"
echo ""

echo "▶ 4/7  Mergeando dev → main..."
MERGE_MSG="hotfix(v3.4.2): merge dev -> main · Ciclo 17.18 cache Supabase Storage

URGENTE — fix Cached Egress 173% del cap (8.6 GB / 5 GB).

Bug:
  Banner 'Exceeding usage limits' en Supabase. El bucket
  pesa 2 MB pero el egress mensual estaba excedido porque
  get_instance_document_bytes re-descargaba el mismo archivo
  en cada rerun de Streamlit.

Fix:
  Nuevo core/document_cache.py con cache en disco
  (data/cache/instance_documents/), TTL 30 días, write
  atómico (tmp + os.replace). Lazy import desde
  instance_state para evitar circulares. Invalidación
  automática en upload/remove.

Garantías:
  - API publica intacta (8+ callers sin cambios)
  - Cache sobrevive reruns y redeploys de Streamlit Cloud
  - Bytes idénticos a los del bucket (lossless, sin transform)
  - None responses no se cachean (no contamina)
  - TTL configurable, default 30d, archivos viejos se refrescan

Validado con smoke test:
  - HIT/MISS, invalidación específica, invalidación por instancia,
    TTL expiration, clear_all, None handling
  - Simulación: 30 interacciones → 96.7% egress reducido

Impacto:
  8.6 GB/4 días → <500 MB/4 días estimado. Sale del cap del Free."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.4.2..."
TAG_EXISTS=$(git tag -l "v3.4.2")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.4.2 ya existe. Saltando creación."
else
    git tag -a v3.4.2 -m "Hotfix v3.4.2 — Ciclo 17.18 cache local de Supabase Storage"
    echo "  ✓ Tag v3.4.2 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.4.2 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ HOTFIX v3.4.2 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en"
echo "    1-2 minutos."
echo ""
echo " 📊 Cómo verificar que funciona:"
echo ""
echo "    1. Después del redeploy, abrí cualquier instancia que"
echo "       tenga schematic o documentos. Mové sliders / cambiá"
echo "       selectboxes varias veces."
echo ""
echo "    2. Andá al dashboard de Supabase → Settings → Usage"
echo "       y observá 'Cached Egress' en los próximos días."
echo "       Debería crecer mucho más despacio (estimado <500 MB"
echo "       por semana en lugar de >2 GB)."
echo ""
echo "    3. El ciclo de billing actual termina el 28 May. El"
echo "       contador se resetea ese día. Hasta entonces sigue"
echo "       en 8.6/5 GB pero ya NO va a seguir creciendo a"
echo "       ritmo anterior."
echo ""
echo " ⚠  Aviso: el cache vive en data/cache/instance_documents/"
echo "    Si en algún momento el bucket se modifica MANUALMENTE"
echo "    (no vía la app), correr en local:"
echo "      from core.document_cache import clear_all_cache"
echo "      clear_all_cache()"
echo ""
echo "================================================================"
