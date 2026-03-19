from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = PROJECT_ROOT / "pages"

ROOT_CANDIDATES = [
    "00_Home.py",
    "Home.py",
    "app.py",
    "main.py",
]

SKIP_FILES = {
    "00_Login.py",
    "Login.py",
}

GUARD_IMPORT = "from core.auth import require_login, render_user_menu"
GUARD_CALLS = "require_login()\nrender_user_menu()"


def already_protected(content: str) -> bool:
    return "require_login()" in content


def make_backup(file_path: Path) -> None:
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")


def insert_after_header(content: str) -> str:
    stripped = content.lstrip()
    leading = content[: len(content) - len(stripped)]

    if stripped.startswith('"""') or stripped.startswith("'''"):
        quote = stripped[:3]
        end_idx = stripped.find(quote, 3)
        if end_idx != -1:
            docstring_block = stripped[: end_idx + 3]
            rest = stripped[end_idx + 3 :].lstrip("\n")
            return (
                f"{leading}{docstring_block}\n\n"
                f"{GUARD_IMPORT}\n\n"
                f"{GUARD_CALLS}\n\n"
                f"{rest}"
            )

    lines = stripped.splitlines()
    if lines and lines[0].startswith("from __future__ import"):
        first = lines[0]
        rest = "\n".join(lines[1:]).lstrip("\n")
        return (
            f"{leading}{first}\n\n"
            f"{GUARD_IMPORT}\n\n"
            f"{GUARD_CALLS}\n\n"
            f"{rest}"
        )

    return (
        f"{leading}{GUARD_IMPORT}\n\n"
        f"{GUARD_CALLS}\n\n"
        f"{stripped}"
    )


def protect_file(file_path: Path) -> bool:
    if not file_path.exists():
        return False

    original = file_path.read_text(encoding="utf-8")

    if already_protected(original):
        return False

    make_backup(file_path)
    updated = insert_after_header(original)
    file_path.write_text(updated, encoding="utf-8")
    return True


def protect_root_files() -> list[str]:
    updated = []

    for filename in ROOT_CANDIDATES:
        file_path = PROJECT_ROOT / filename
        if protect_file(file_path):
            updated.append(filename)

    return updated


def protect_pages() -> list[str]:
    updated = []

    if not PAGES_DIR.exists():
        return updated

    for file_path in sorted(PAGES_DIR.glob("*.py")):
        if file_path.name in SKIP_FILES:
            continue

        if protect_file(file_path):
            updated.append(str(file_path.relative_to(PROJECT_ROOT)))

    return updated


def main() -> None:
    updated_root = protect_root_files()
    updated_pages = protect_pages()
    updated_all = updated_root + updated_pages

    if not updated_all:
        print("No hubo cambios. Todo ya estaba protegido.")
        return

    print("Protección aplicada en:")
    for item in updated_all:
        print(f" - {item}")

    print("\nListo. Ahora ninguna página protegida abrirá sin login.")
    print("Se creó backup .bak de cada archivo modificado.")


if __name__ == "__main__":
    main()