#!/bin/bash
# Ciclo 20A → DEV: Multi-tenant ACL.
# - data/clients.json: registry de admins/specialists/clientes
# - core/clients.py: CallerScope, resolve_by_phone/api_key, filter_matches
# - api/services.py: list_archived_*, get_latest_report_pdf aceptan scope
# - api/app.py: _scope_dependency con X-Caller-Phone
# - bot main.py: pasa from_number como X-Caller-Phone
# - tests/test_clients_acl.py
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo20a-multi-tenant-acl

git add data/clients.json core/clients.py api/services.py api/app.py tests/test_clients_acl.py
git commit -m "feat(20A): multi-tenant ACL — registry de clientes + filtrado por scope

  ► data/clients.json (NUEVO)
    Registry single-source-of-truth de:
      - admins:      [{Ewdes, ehernandez@sigasas.com, +573008888883}]
      - specialists: [Jessica Suarez, Natalia Lopez]
      - clients:     [Ecopetrol-Magnex, Parex, Refoenergy] (precargados)

  ► core/clients.py (NUEVO)
    CallerScope dataclass + resolvers:
      resolve_by_phone(num)  → CallerScope (admin/specialist/client/unknown)
      resolve_by_api_key(k)  → idem (admin global vs client api_key)
      filter_matches(rm, s)  → True si el sidecar es visible para el scope
        admin/specialist     → True para todo
        client               → match_strings substring sobre report_meta
        unknown              → False

  ► api/services.py (modificado)
    list_archived_assets(scope), list_reports_for_asset(scope),
    get_latest_report_pdf(scope) aplican el filtro.
    None scope = compat retro (admin).

  ► api/app.py (modificado)
    _scope_dependency: lee Authorization + X-Caller-Phone:
      - admin key + phone admin/specialist/client → ese scope
      - admin key + phone unknown → 403
      - client api_key → ese client (X-Caller-Phone se ignora)
      - key inválida → 401

  ► tests/ — 24 tests nuevos validan ACL completo

NO toca core/auth.py ni reports_archive ni pages existentes."

git push -u origin feat/ciclo20a-multi-tenant-acl
git checkout dev
git merge --no-ff feat/ciclo20a-multi-tenant-acl -m "Merge feat/ciclo20a-multi-tenant-acl into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 20A en DEV"
echo " Después: bash _publish_v3_20_0_to_main.sh"
echo "================================================================"
