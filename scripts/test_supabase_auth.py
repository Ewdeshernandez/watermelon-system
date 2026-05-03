#!/usr/bin/env python3
"""
scripts/test_supabase_auth.py
=============================

Smoke test del setup de Supabase Auth (pre Ciclo 17.14).

Verifica:
  1. Que .streamlit/secrets.toml tenga sección [supabase] con url+service_key
  2. Que supabase-py esté instalado
  3. Que las credenciales sean válidas (conecta + lista usuarios)
  4. Que Auth admin API responda

Uso:
    cd ~/Documents/WatermelonSystem
    python3 scripts/test_supabase_auth.py

Salida esperada:
  ✓ Todos los chequeos pasaron — listos para codear el 17.14
"""

from __future__ import annotations

import sys
from pathlib import Path


def _read_secrets() -> dict:
    """Lee .streamlit/secrets.toml usando tomllib (3.11+) o toml fallback."""
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / ".streamlit" / "secrets.toml"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    text = path.read_text(encoding="utf-8")
    try:
        import tomllib
        return tomllib.loads(text)
    except ImportError:
        try:
            import toml
            return toml.loads(text)
        except ImportError:
            print("✗ No tengo ni tomllib (Python 3.11+) ni el paquete 'toml'.")
            print("  Instalá toml con: pip install toml")
            sys.exit(2)


def main() -> int:
    print("═" * 60)
    print("  SUPABASE AUTH — SMOKE TEST")
    print("═" * 60)
    print()

    # ─── 1. Leer secrets ───
    print("► 1. Leyendo .streamlit/secrets.toml…")
    try:
        secrets = _read_secrets()
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return 2

    sb = secrets.get("supabase", {})
    if not sb:
        print("  ✗ No hay sección [supabase] en secrets.toml")
        return 2

    url = sb.get("url", "").strip()
    key = sb.get("service_key", "").strip()
    bucket = sb.get("bucket", "")

    if not url:
        print("  ✗ Falta 'url' en [supabase]")
        return 2
    if not key or key == "PEGAR_AQUI_TU_SERVICE_ROLE_KEY_DE_SUPABASE":
        print("  ✗ Falta 'service_key' (o todavía es el placeholder)")
        return 2

    key_kind = "sb_secret (formato nuevo)" if key.startswith("sb_secret_") else \
               "JWT legacy" if key.startswith("eyJ") else "desconocido"
    print(f"  ✓ url        = {url}")
    print(f"  ✓ service_key = {key[:14]}…{key[-6:]} ({key_kind}, {len(key)} chars)")
    print(f"  ✓ bucket     = {bucket}")
    print()

    # ─── 2. Verificar supabase-py ───
    print("► 2. Verificando supabase-py instalado…")
    try:
        import supabase
        print(f"  ✓ supabase-py version: {supabase.__version__}")
    except ImportError:
        print("  ✗ supabase-py NO instalado")
        print("    Instalá con: pip install supabase")
        return 2
    print()

    # ─── 3. Conectar ───
    print("► 3. Conectando a Supabase…")
    try:
        from supabase import create_client
        client = create_client(url, key)
        print("  ✓ Cliente creado correctamente")
    except Exception as e:
        print(f"  ✗ Falló al crear cliente: {e}")
        return 3
    print()

    # ─── 4. Auth admin API: listar usuarios ───
    print("► 4. Probando Auth admin API (list_users)…")
    try:
        # supabase-py >= 2.x: client.auth.admin.list_users()
        users = client.auth.admin.list_users()
        # Algunos versions devuelven lista directa, otros un objeto con .users
        n = len(users) if isinstance(users, list) else len(getattr(users, "users", []) or [])
        print(f"  ✓ list_users() OK — {n} usuario(s) en Auth")
    except Exception as e:
        print(f"  ✗ list_users() falló: {e}")
        print()
        print("  POSIBLES CAUSAS:")
        print("  - service_key incorrecta o expirada")
        print("  - URL del proyecto incorrecta")
        print("  - service_key es realmente la 'anon' por error")
        print("  - Email auth no está habilitado en el proyecto")
        return 4
    print()

    # ─── 5. Vault DB (opcional, sanity check) ───
    print("► 5. Sanity check: listar tabla 'instances' del Vault…")
    try:
        res = client.table("instances").select("id", count="exact").limit(1).execute()
        n_inst = getattr(res, "count", None) or len(getattr(res, "data", []) or [])
        print(f"  ✓ Tabla 'instances' accesible — {n_inst} instancia(s) registrada(s)")
    except Exception as e:
        print(f"  ⚠ list 'instances' falló (no bloquea Auth, solo Vault): {e}")
    print()

    # ─── Final ───
    print("═" * 60)
    print("  ✅ TODOS LOS CHEQUEOS PASARON")
    print("  Setup Supabase Auth está listo para codear el Ciclo 17.14")
    print("═" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
