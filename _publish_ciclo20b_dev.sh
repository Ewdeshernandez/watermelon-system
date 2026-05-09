#!/bin/bash
# Ciclo 20B → DEV: Admin UI para clients.json.
# - core/clients.py: + save_registry() atómico con cache invalidation
# - pages/_admin_clients.py: 4 tabs (Clientes / Specialists / Admins / Raw JSON)
# - core/auth.py: NAV 'Admin · Clientes' + CLIENT_BLOCKED_PAGES
# - tests/test_save_registry.py
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo20b-admin-ui-clients

git add core/clients.py pages/_admin_clients.py core/auth.py tests/test_save_registry.py
git commit -m "feat(20B): Admin UI para gestionar clients.json desde Streamlit

  ► core/clients.py: + save_registry(data)
    Escritura atómica (tmp + rename), preserva _meta.version,
    bumpea last_updated, invalida lru_cache automáticamente.

  ► pages/_admin_clients.py (NUEVO, admin only)
    4 tabs:
      🏭 Clientes     — CRUD: id, match_strings, asset_tags,
                         whatsapp_numbers, owner_emails
      🛠️ Specialists  — CRUD: name, email, whatsapp_numbers
      🔑 Admins       — Add (eliminar a uno mismo está restringido
                         visualmente con warning, no técnicamente)
      📄 Raw JSON     — Vista de auditoría con api_keys enmascaradas

  ► core/auth.py
    + NAV_ITEMS entry 'Admin · Clientes' → pages/_admin_clients.py
    + CLIENT_BLOCKED_PAGES include _admin_clients.py
    (specialists tampoco entran porque require_role(allowed=admin))

  ► tests/test_save_registry.py — 3 tests con tempdir aislado

Sin cambios en core/reports_archive ni en api/."

git push -u origin feat/ciclo20b-admin-ui-clients
git checkout dev
git merge --no-ff feat/ciclo20b-admin-ui-clients -m "Merge feat/ciclo20b-admin-ui-clients into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 20B en DEV"
echo " Refresh wm-test → debe aparecer 'Admin · Clientes' en sidebar (solo admin)"
echo " Después: bash _publish_v3_21_0_to_main.sh"
echo "================================================================"
