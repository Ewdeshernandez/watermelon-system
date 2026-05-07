"""
tests.run_smoke
===============

Mini-runner casero para correr la suite de tests SIN pytest instalado.

¿Por qué existe?
    - En entornos restringidos (sandbox, contenedor mínimo, CI sin pip),
      no siempre podés instalar pytest.
    - Este script imita la ergonomía esencial de pytest:
        * @pytest.fixture
        * pytest.approx
        * pytest.raises
        * pytest.skip
        * pytest.mark.parametrize
      y descubre/ejecuta cualquier `def test_*` en tests/test_*.py.

Uso:
    python tests/run_smoke.py                # corre todo
    python tests/run_smoke.py waveform       # filtra por substring
    python tests/run_smoke.py -v             # verbose

Diferencias con pytest:
    - No soporta fixtures complejos con scope, conftest plugins,
      monkeypatch, capsys, etc. Es un runner mínimo.
    - Para la suite real, en tu máquina seguí usando pytest:
        pip install -r requirements-dev.txt
        pytest -q
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Tuple


# =============================================================
# Bootstrap path
# =============================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================
# Mini pytest shim — sólo las features que usa esta suite
# =============================================================

class _Skipped(Exception):
    """Marca un test como saltado."""


class _Approx:
    """Imita pytest.approx para comparaciones float con tolerancia."""

    __slots__ = ("expected", "rel", "abs_")

    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = rel
        self.abs_ = abs

    def __eq__(self, actual):
        try:
            actual_f = float(actual)
            expected_f = float(self.expected)
        except (TypeError, ValueError):
            return False
        if self.abs_ is not None:
            return abs(actual_f - expected_f) <= float(self.abs_)
        rel = self.rel if self.rel is not None else 1e-6
        if expected_f == 0:
            return abs(actual_f) <= max(rel, 1e-12)
        return abs(actual_f - expected_f) / abs(expected_f) <= float(rel)

    def __repr__(self):
        if self.abs_ is not None:
            return f"approx({self.expected}, abs={self.abs_})"
        return f"approx({self.expected}, rel={self.rel})"


class _Raises:
    """Imita pytest.raises como context manager."""

    def __init__(self, expected_exc):
        if isinstance(expected_exc, tuple):
            self.expected = expected_exc
        else:
            self.expected = (expected_exc,)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(
                f"DID NOT RAISE {tuple(e.__name__ for e in self.expected)}"
            )
        if not issubclass(exc_type, self.expected):
            return False  # propagate
        return True  # swallow


class _Mark:
    """Soporta @pytest.mark.parametrize y otros marks como no-op."""

    def __init__(self):
        pass

    def __getattr__(self, name):
        if name == "parametrize":
            def parametrize(arg_names, arg_values):
                def decorator(fn):
                    fn.__pytest_parametrize__ = (arg_names, list(arg_values))
                    return fn
                return decorator
            return parametrize
        # Otros marks (slow, integration, etc.) → no-op
        def noop(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
        return noop


def _fixture(*args, **kwargs):
    """Decorador que marca una función como fixture."""
    def _decorator(fn):
        fn.__is_fixture__ = True
        return fn
    if len(args) == 1 and callable(args[0]) and not kwargs:
        # @pytest.fixture sin paréntesis
        return _decorator(args[0])
    return _decorator


def _skip(msg=""):
    raise _Skipped(msg)


def _make_pytest_module() -> ModuleType:
    mod = ModuleType("pytest")
    mod.fixture = _fixture
    mod.approx = lambda expected, rel=None, abs=None: _Approx(expected, rel=rel, abs=abs)
    mod.raises = _Raises
    mod.skip = _skip
    mod.mark = _Mark()
    mod.Skipped = _Skipped
    return mod


# Inyectar antes de importar conftest/tests
sys.modules["pytest"] = _make_pytest_module()
import pytest as _pytest  # noqa: E402  (alias amigable, mismo objeto)


# =============================================================
# Discovery
# =============================================================

def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _collect_tests(test_dir: Path, filter_substr: str = "") -> List[Tuple[str, str, Callable]]:
    """
    Devuelve lista de (modulo_nombre, test_nombre, callable).
    Expande tests parametrizados.
    """
    out: List[Tuple[str, str, Callable]] = []
    test_files = sorted(test_dir.glob("test_*.py"))
    for f in test_files:
        if filter_substr and filter_substr not in f.stem:
            # filtro a nivel de archivo solo si el substring no está en
            # ningún test del archivo. Por simplicidad, primero cargamos.
            pass
        mod_name = f"tests.{f.stem}"
        try:
            mod = _load_module(f, mod_name)
        except _Skipped as e:
            print(f"  [skipped module] {f.name}: {e}")
            continue
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            if filter_substr and (filter_substr not in name and filter_substr not in f.stem):
                continue
            params = getattr(fn, "__pytest_parametrize__", None)
            if params is None:
                out.append((mod_name, name, fn))
            else:
                arg_names, arg_values = params
                if isinstance(arg_names, str):
                    names = [a.strip() for a in arg_names.split(",")]
                else:
                    names = list(arg_names)
                for vals in arg_values:
                    if not isinstance(vals, (list, tuple)):
                        vals = (vals,)
                    label = "[" + "-".join(repr(v) for v in vals) + "]"
                    bound = _bind_parametrize(fn, names, vals)
                    out.append((mod_name, name + label, bound))
    return out


def _bind_parametrize(fn, names, vals):
    def wrapper(**fixture_kwargs):
        kwargs = dict(zip(names, vals))
        # Si la función pide más argumentos (fixtures), el runner los inyectará
        kwargs.update(fixture_kwargs)
        return fn(**kwargs)
    wrapper.__name__ = fn.__name__
    wrapper.__signature__ = _signature_minus(fn, names)
    wrapper.__module__ = fn.__module__
    return wrapper


def _signature_minus(fn, removed_names):
    sig = inspect.signature(fn)
    new_params = [p for p in sig.parameters.values() if p.name not in set(removed_names)]
    return sig.replace(parameters=new_params)


def _resolve_fixtures(test_fn: Callable, fixtures: Dict[str, Callable]) -> Dict[str, Any]:
    sig = inspect.signature(test_fn)
    kwargs = {}
    for name in sig.parameters:
        if name in fixtures:
            kwargs[name] = fixtures[name]()  # llamada simple, sin caching
    return kwargs


# =============================================================
# Runner
# =============================================================

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _colored(text: str, color: str) -> str:
    if not _supports_color():
        return text
    return f"{color}{text}{RESET}"


def main(argv: List[str]) -> int:
    verbose = "-v" in argv
    argv = [a for a in argv if a != "-v"]
    filter_substr = argv[0] if argv else ""

    test_dir = THIS_DIR

    # Cargar conftest para registrar fixtures
    conftest_path = test_dir / "conftest.py"
    fixtures: Dict[str, Callable] = {}
    if conftest_path.exists():
        try:
            cmod = _load_module(conftest_path, "tests.conftest")
            for name, fn in inspect.getmembers(cmod, inspect.isfunction):
                if getattr(fn, "__is_fixture__", False):
                    fixtures[name] = fn
        except Exception as e:
            print(_colored(f"ERROR cargando conftest: {e}", RED))
            traceback.print_exc()
            return 2

    # Collect tests
    tests = _collect_tests(test_dir, filter_substr=filter_substr)

    if not tests:
        print(_colored("Ningún test encontrado.", YELLOW))
        return 0

    print(_colored(f"\nEjecutando {len(tests)} tests "
                   f"({len(fixtures)} fixtures registradas)\n", DIM))

    passed: List[str] = []
    failed: List[Tuple[str, str, str]] = []
    skipped: List[Tuple[str, str]] = []
    t0 = time.perf_counter()

    for mod_name, test_name, fn in tests:
        full = f"{mod_name}::{test_name}"
        try:
            kwargs = _resolve_fixtures(fn, fixtures)
            fn(**kwargs)
        except _Skipped as e:
            skipped.append((full, str(e)))
            print(_colored("S", YELLOW), end="", flush=True)
        except AssertionError as e:
            tb = traceback.format_exc()
            failed.append((full, str(e), tb))
            print(_colored("F", RED), end="", flush=True)
        except Exception as e:
            tb = traceback.format_exc()
            failed.append((full, f"{type(e).__name__}: {e}", tb))
            print(_colored("E", RED), end="", flush=True)
        else:
            passed.append(full)
            print(_colored(".", GREEN), end="", flush=True)

    elapsed = time.perf_counter() - t0
    print("\n")

    # Reporte de fallos
    if failed:
        print(_colored("=" * 70, RED))
        print(_colored("FAILURES", RED))
        print(_colored("=" * 70, RED))
        for full, msg, tb in failed:
            print(_colored(f"\n[FAIL] {full}", RED))
            print(_colored(f"       {msg}", RED))
            if verbose:
                print(_colored(tb, DIM))

    if skipped and verbose:
        print(_colored("\n[skipped]", YELLOW))
        for full, msg in skipped:
            print(_colored(f"  {full} — {msg or 'no reason'}", YELLOW))

    # Resumen
    total = len(passed) + len(failed) + len(skipped)
    print(_colored("=" * 70, DIM))
    summary = (
        f"{len(passed)} passed"
        + (f", {len(failed)} failed" if failed else "")
        + (f", {len(skipped)} skipped" if skipped else "")
        + f" in {elapsed:.2f}s ({total} total)"
    )
    color = RED if failed else GREEN
    print(_colored(summary, color))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
