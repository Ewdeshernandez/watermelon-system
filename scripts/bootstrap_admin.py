#!/usr/bin/env python3
"""
scripts/bootstrap_admin.py
==========================

Setup inicial del administrador único del sistema (Ciclo 17.14).

Se corre UNA SOLA VEZ después de configurar Supabase Auth, para
crear el usuario `ehernandez@sigasas.com` con role=admin en la tabla
`auth.users` de Supabase. Sin esto, no hay forma de hacer el primer
login con el sistema nuevo.

Flujo:
  1. Verifica conexión a Supabase Auth
  2. Verifica si el admin único YA existe → si sí, ofrece resetear password
  3. Si no existe, lo crea con una password aleatoria segura
  4. Imprime la password en la terminal (UNA sola vez)
  5. Recomienda que el admin cambie su password después del primer login

Uso:
    cd ~/Documents/WatermelonSystem
    python scripts/bootstrap_admin.py

    # Si querés especificar la password manualmente:
    python scripts/bootstrap_admin.py --password "MiPassWord123"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _read_secrets() -> dict:
    """Lee .streamlit/secrets.toml (igual que el test_supabase_auth.py)."""
    path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib
        return tomllib.loads(text)
    except ImportError:
        import toml
        return toml.loads(text)


def _make_streamlit_shim(secrets_dict: dict):
    """Crea un módulo 'streamlit' falso que solo expone st.secrets,
    para que core.supabase_auth funcione sin Streamlit corriendo.
    """
    import types

    class _SecretsShim:
        def __init__(self, d): self._d = d
        def get(self, k, default=None): return self._d.get(k, default)
        def __getitem__(self, k): return self._d[k]
        def __contains__(self, k): return k in self._d

    class _SessionShim(dict):
        def get(self, k, default=None): return dict.get(self, k, default)

    fake = types.ModuleType("streamlit")
    fake.secrets = _SecretsShim(secrets_dict)
    fake.session_state = _SessionShim()
    sys.modules["streamlit"] = fake


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap del admin único en Supabase Auth (17.14)."
    )
    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="Password para el admin (si vacío, se genera una aleatoria).",
    )
    parser.add_argument(
        "--force-reset",
        action="store_true",
        help="Si el admin ya existe, resetea su password al valor especificado/aleatorio.",
    )
    args = parser.parse_args()

    print("═" * 60)
    print("  BOOTSTRAP ADMIN ÚNICO — Watermelon System")
    print("═" * 60)
    print()

    # 1. Leer secrets
    print("► 1. Leyendo .streamlit/secrets.toml…")
    try:
        secrets = _read_secrets()
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return 2
    if "supabase" not in secrets:
        print("  ✗ No hay sección [supabase]. Corré primero scripts/test_supabase_auth.py")
        return 2
    print("  ✓ secrets cargados")

    # 2. Shim Streamlit y cargar core.supabase_auth
    _make_streamlit_shim(secrets)
    try:
        from core.supabase_auth import (
            ADMIN_EMAIL,
            create_user,
            generate_temp_password,
            get_user_by_email,
            reset_user_password,
        )
    except Exception as e:
        print(f"  ✗ No se pudo cargar core.supabase_auth: {e}")
        return 2
    print(f"  ✓ Admin email objetivo: {ADMIN_EMAIL}")
    print()

    # 3. Chequear si ya existe
    print(f"► 2. Buscando si {ADMIN_EMAIL} ya existe en Supabase Auth…")
    existing = get_user_by_email(ADMIN_EMAIL)
    if existing:
        print(f"  ⚠ Ya existe (id={existing.get('id', '')[:8]}…)")
        print(f"    role: {existing.get('role')}")
        print(f"    creado: {existing.get('created_at', '')[:19]}")

        if not args.force_reset:
            print()
            print("  Si querés resetear su password, corré:")
            print("    python scripts/bootstrap_admin.py --force-reset")
            print()
            print("  Si no, podés iniciar sesión normalmente con la password "
                  "que ya conocés.")
            return 0

        # Reset password
        new_pwd = (args.password or generate_temp_password(14)).strip()
        print(f"\n► 3. Reseteando password a un valor temporal…")
        res = reset_user_password(existing["id"], new_pwd)
        if not res.get("ok"):
            print(f"  ✗ Falló: {res.get('error')}")
            return 3
        print(f"  ✓ Password reseteada.")
        print()
        print("═" * 60)
        print(f"  📌 NUEVA PASSWORD (entrégate y guardala en password manager):")
        print()
        print(f"     Email:    {ADMIN_EMAIL}")
        print(f"     Password: {new_pwd}")
        print()
        print("  📌 Esta password sólo se muestra UNA vez. Cambiala en tu primer")
        print("     login desde el Admin Panel.")
        print("═" * 60)
        return 0

    print("  ✓ No existe — vamos a crearlo")
    print()

    # 4. Crear
    new_pwd = (args.password or generate_temp_password(14)).strip()
    if len(new_pwd) < 8:
        print("  ✗ La password debe tener al menos 8 caracteres.")
        return 2

    print(f"► 3. Creando usuario {ADMIN_EMAIL} como admin…")
    res = create_user(
        email=ADMIN_EMAIL,
        password=new_pwd,
        full_name="Ewdes Hernández",
        role="admin",
    )
    if not res.get("ok"):
        print(f"  ✗ Falló: {res.get('error')}")
        return 3

    user = res["user"]
    print(f"  ✓ Creado: id={user.get('id', '')[:8]}…  role={user.get('role')}")
    print()

    # 5. Imprimir credenciales (UNA SOLA VEZ)
    print("═" * 60)
    print("  ✅ ADMIN CREADO CORRECTAMENTE")
    print()
    print("  📌 GUARDÁ ESTOS DATOS — NO se vuelven a mostrar:")
    print()
    print(f"     Email:    {ADMIN_EMAIL}")
    print(f"     Password: {new_pwd}")
    print()
    print("  📌 Cómo seguir:")
    print("     1. Andá a https://wm-home-final-2026.streamlit.app/Login")
    print("        (o tu local si estás desarrollando)")
    print("     2. Iniciá sesión con esos datos")
    print("     3. Vas a ver un nuevo botón 'Admin · Usuarios' en el sidebar")
    print("     4. Desde ahí podés crear más usuarios @sigasas.com")
    print()
    print("  📌 Consejo: cambiá la password después del primer login (próximo")
    print("     ciclo agregamos UI de cambio de password propia).")
    print("═" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
