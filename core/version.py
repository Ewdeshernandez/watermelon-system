"""
core.version
============

Single source of truth para la información de versión del Watermelon System.

Diseñado para que cualquier parte de la app (login, sidebar, PDF reports,
About modal) consulte la misma data, derivada automáticamente de los git
tags. Esto significa que cada vez que pusheamos un tag nuevo (v3.0.8,
v3.1.0, etc.), TODO el sistema lo refleja sin tocar código.

Estrategia de resolución (en orden de prioridad):

    1. Variable de entorno WM_VERSION (override en producción cuando no
       hay git disponible, e.g. en un container Docker stripped).
    2. `git describe --tags --abbrev=8 --dirty` — el modo normal en
       desarrollo y en deploys con git.
    3. Archivo VERSION en la raíz del repo (fallback si git no existe).
    4. Constante hardcodeada _FALLBACK_VERSION (último recurso).

Lo mismo para environment (production / staging / development) — se
infiere de la branch git si no viene la variable WM_ENVIRONMENT.

Uso típico:

    from core.version import get_version_info, get_version_short

    info = get_version_info()
    # info = {
    #     "version":      "v3.0.8",
    #     "commit":       "ab12cd34",
    #     "branch":       "main",
    #     "date":         "2026-05-03",
    #     "environment":  "production",
    #     "full_label":   "v3.0.8 · production · ab12cd34",
    #     "release_name": "Trend module clase mundial",
    # }

    short = get_version_short()      # "v3.0.8"
    label = get_version_full_label() # "v3.0.8 · production · ab12cd34"

Subprocess timeouts cortos (2s) para no bloquear el render de Streamlit
si git está lento o no existe.
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = PROJECT_ROOT / "VERSION"

# Hardcoded fallback — bump al hacer release si no hay git disponible.
# Ciclo 18.2: subido a v3.15.0 (estaba en v3.0.8 desde release inicial,
# lo que combinado con un override WM_VERSION viejo en Streamlit Cloud
# mostraba versiones obsoletas/incorrectas).
_FALLBACK_VERSION = "v3.15.0"
_FALLBACK_RELEASE_NAME = "Industrial Plumbing + Importers UI Hub"
_GIT_TIMEOUT_SEC = 2.0


# =============================================================
# GIT HELPERS — todos defensivos, devuelven None si git no existe
# =============================================================

def _run_git(args: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git"] + args,
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SEC,
        )
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _git_describe() -> Optional[str]:
    """
    Devuelve algo como:
      - 'v3.0.8'                         (estamos exactamente en el tag)
      - 'v3.0.8-2-gab12cd34'             (2 commits desde el tag, hash)
      - 'v3.0.8-2-gab12cd34-dirty'       (igual + working tree dirty)
    """
    return _run_git(["describe", "--tags", "--always", "--abbrev=8", "--dirty"])


def _git_latest_semver_tag() -> Optional[str]:
    """
    Devuelve el tag SEMVER más alto del repo (ej. 'v3.0.8'), sin
    importar si es ancestro de la branch actual.

    Esto es importante porque nuestros publish scripts crean los
    tags en MAIN sobre merge commits — desde DEV el `git describe`
    no los ve (encuentra una versión vieja). Para mostrar la
    versión real del producto al usuario, queremos siempre el tag
    más alto que existe en el repo.
    """
    out = _run_git(["tag", "--list", "v*", "--sort=-v:refname"])
    if not out:
        return None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("v") and any(c.isdigit() for c in line):
            return line
    return None


def _git_commit_sha() -> Optional[str]:
    return _run_git(["rev-parse", "--short=8", "HEAD"])


def _git_branch() -> Optional[str]:
    """Branch actual; en CI/detached-HEAD puede devolver 'HEAD'."""
    name = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if name == "HEAD":
        # detached: intentar la primera branch que apunte al commit
        names = _run_git(["branch", "--contains", "HEAD"])
        if names:
            for ln in names.splitlines():
                ln = ln.strip().lstrip("*").strip()
                if ln and ln != "HEAD":
                    return ln
    return name


def _git_commit_date() -> Optional[str]:
    """Devuelve la fecha del último commit en formato YYYY-MM-DD."""
    return _run_git(["log", "-1", "--format=%cs"])


def _read_version_file() -> Optional[str]:
    if VERSION_FILE.exists():
        try:
            content = VERSION_FILE.read_text(encoding="utf-8").strip()
            if content:
                return content.splitlines()[0].strip()
        except Exception:
            return None
    return None


def _normalize_version(raw: str) -> str:
    """Limpia 'v3.0.8-2-gab12cd34-dirty' → 'v3.0.8'.
    Acepta también '3.0.8' sin la 'v' y la añade."""
    if not raw:
        return _FALLBACK_VERSION
    raw = raw.strip()
    base = raw.split("-")[0]
    if base and not base.startswith("v") and base[0].isdigit():
        base = "v" + base
    return base or _FALLBACK_VERSION


# =============================================================
# API PÚBLICA
# =============================================================

@functools.lru_cache(maxsize=1)
def get_version_info() -> Dict[str, str]:
    """
    Devuelve un dict con toda la información de versión.

    Cacheado durante la vida del proceso (lru_cache) para no fork-ear
    git en cada rerun de Streamlit. Se invalida al reiniciar el server.
    """
    env_version = os.environ.get("WM_VERSION", "").strip()
    env_environment = os.environ.get("WM_ENVIRONMENT", "").strip()

    git_desc = _git_describe()
    git_sha = _git_commit_sha()
    git_branch = _git_branch()
    git_date = _git_commit_date()
    # Latest semver tag (regardless of ancestry — para el caso típico
    # donde estamos en dev pero los tags v3.0.x están en main)
    git_latest_tag = _git_latest_semver_tag()

    file_version = _read_version_file()

    # Compute is_dirty / commits_ahead first (they vienen de git_desc)
    if git_desc:
        parts = git_desc.split("-")
        commits_ahead = parts[1] if len(parts) >= 3 and parts[1].isdigit() else ""
        is_dirty = git_desc.endswith("-dirty")
    else:
        commits_ahead = ""
        is_dirty = False

    # Resolver versión por prioridad
    if env_version:
        version = _normalize_version(env_version)
    elif git_latest_tag:
        # Tag semver más alto del repo (ej. v3.0.8 aunque estemos en dev)
        version = _normalize_version(git_latest_tag)
    elif git_desc:
        version = _normalize_version(git_desc)
    elif file_version:
        version = _normalize_version(file_version)
    else:
        version = _FALLBACK_VERSION

    if env_environment:
        environment = env_environment.lower()
    elif git_branch == "main":
        environment = "production"
    elif git_branch == "dev":
        environment = "development"
    elif git_branch:
        environment = git_branch
    else:
        environment = "unknown"

    label_parts = [version, environment]
    if git_sha:
        label_parts.append(git_sha)
    full_label = " · ".join(label_parts)
    if commits_ahead:
        full_label += f" (+{commits_ahead})"
    if is_dirty:
        full_label += " ⚠ dirty"

    return {
        "version": version,
        "commit": git_sha or "",
        "branch": git_branch or "",
        "date": git_date or "",
        "environment": environment,
        "commits_ahead": commits_ahead,
        "is_dirty": "true" if is_dirty else "false",
        "full_label": full_label,
        "release_name": _FALLBACK_RELEASE_NAME,
    }


def get_version_short() -> str:
    """Solo la versión, e.g. 'v3.0.8'."""
    return get_version_info()["version"]


def get_version_full_label() -> str:
    """Label completo, e.g. 'v3.0.8 · production · ab12cd34'."""
    return get_version_info()["full_label"]


def get_environment() -> str:
    """'production' / 'development' / branch name."""
    return get_version_info()["environment"]


def get_commit_sha() -> str:
    """SHA corto del commit actual (8 chars)."""
    return get_version_info()["commit"]


# Compat con la línea original del archivo (`VERSION = "..."`)
VERSION = get_version_short() if PROJECT_ROOT.exists() else _FALLBACK_VERSION


__all__ = [
    "get_version_info",
    "get_version_short",
    "get_version_full_label",
    "get_environment",
    "get_commit_sha",
    "VERSION",
]
