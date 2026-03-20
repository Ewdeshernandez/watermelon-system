from __future__ import annotations

import getpass
from core.auth import make_password_hash


def main() -> None:
    print("Watermelon System - Password Hash Generator")
    print("-" * 44)

    while True:
        username = input("Username (admin/demo): ").strip()
        if not username:
            print("Username vacío. Intenta de nuevo.\n")
            continue

        password = getpass.getpass(f"Password for {username}: ").strip()
        if not password:
            print("Password vacía. Intenta de nuevo.\n")
            continue

        confirm = getpass.getpass("Confirm password: ").strip()
        if password != confirm:
            print("Las contraseñas no coinciden.\n")
            continue

        password_hash = make_password_hash(password)

        print("\nPega esto en Secrets:\n")
        print(f'[auth.users.{username}]')
        print(f'email = "{username}@watermelon.local"')
        print(f'full_name = "{username.capitalize()} User"')
        print('role = "admin"' if username.lower() == "admin" else 'role = "viewer"')
        print(f'password_hash = "{password_hash}"')
        print("\n" + "-" * 44 + "\n")

        again = input("¿Generar otro? [y/N]: ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()