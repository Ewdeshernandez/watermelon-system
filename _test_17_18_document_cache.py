"""
_test_17_18_document_cache.py
=============================
Smoke test del cache local de Supabase Storage (Ciclo 17.18).

Estrategia:
  Inyectamos un FakeRepo con un counter de descargas. La idea es que,
  con el cache activo, una segunda llamada al mismo (instance_id,
  filename) NO debe incrementar el counter (porque sale de disco).
  Después de invalidar, sí debe volver a llamar al repo.

Verifica:
  1) Primera descarga → MISS → counter +1, archivo en cache
  2) Segunda descarga (mismo path) → HIT → counter sigue igual
  3) Después de invalidate → MISS → counter +1
  4) TTL expirado → MISS → counter +1
  5) Después de clear_all_cache → MISS → counter +1
  6) Repo devuelve None → cache no guarda nada, counter +1
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

WM_ROOT = Path("/sessions/wonderful-blissful-pascal/mnt/WatermelonSystem")
sys.path.insert(0, str(WM_ROOT))

from core import document_cache as dc  # noqa: E402

# Redirigir el cache a un tmp para no escribir en data/cache real
TMP_CACHE = Path(tempfile.mkdtemp(prefix="wm_doc_cache_test_"))
dc.CACHE_DIR = TMP_CACHE


# =============================================================
# FakeRepo con counter — inyectado vía monkey-patch
# =============================================================

class FakeRepo:
    backend_name = "fake"

    def __init__(self):
        self.download_calls = 0
        # Diccionario que simula el bucket: (instance_id, filename) → bytes
        self.bucket = {
            ("inst_001", "schematic_main.png"): b"FAKE_PNG_HEADER_001" + b"\x00" * 1024,
            ("inst_001", "datasheet.pdf"):       b"FAKE_PDF_HEADER_001" + b"\x00" * 2048,
            ("inst_002", "bently.bn"):           b"FAKE_BN_HEADER_002" + b"\x00" * 512,
        }

    def download_document_bytes(self, instance_id: str, filename: str):
        self.download_calls += 1
        return self.bucket.get((instance_id, filename))


_fake = FakeRepo()


# Monkey-patch para que cached_download_bytes use nuestro FakeRepo en
# vez de import "from core.instance_state import get_active_repository"
def _fake_get_repo():
    return _fake


import core.instance_state as inst_state  # noqa: E402
inst_state.get_active_repository = _fake_get_repo


# =============================================================
# TESTS
# =============================================================

def t1_first_download_misses() -> None:
    print("=" * 60)
    print("TEST 1 — Primera descarga es MISS, segunda es HIT")
    print("=" * 60)
    dc.clear_all_cache()
    _fake.download_calls = 0

    data = dc.cached_download_bytes("inst_001", "schematic_main.png")
    assert data is not None and len(data) > 0, "primera descarga devolvió None"
    assert _fake.download_calls == 1, (
        f"esperaba 1 llamada al repo, hubo {_fake.download_calls}"
    )
    print(f"  ✓ Primera llamada: counter={_fake.download_calls} (MISS)")

    # Segunda llamada — debe servir desde cache
    data2 = dc.cached_download_bytes("inst_001", "schematic_main.png")
    assert data2 == data, "los bytes del cache no coinciden con la descarga"
    assert _fake.download_calls == 1, (
        f"segunda llamada disparó re-descarga (counter={_fake.download_calls}). "
        f"El cache no está funcionando."
    )
    print(f"  ✓ Segunda llamada: counter={_fake.download_calls} (HIT — sin re-descarga)")
    print()


def t2_invalidate_specific_file() -> None:
    print("=" * 60)
    print("TEST 2 — invalidate_document(specific) → próxima MISS")
    print("=" * 60)
    dc.clear_all_cache()
    _fake.download_calls = 0

    # Primer fetch — MISS
    dc.cached_download_bytes("inst_001", "schematic_main.png")
    # Segundo fetch — HIT
    dc.cached_download_bytes("inst_001", "schematic_main.png")
    assert _fake.download_calls == 1

    # Invalidar SOLO ese archivo
    n = dc.invalidate_document("inst_001", "schematic_main.png")
    assert n == 1, f"esperaba invalidar 1 archivo, invalidé {n}"
    print(f"  ✓ Invalidados {n} archivos del cache")

    # Tercer fetch — debe ser MISS de nuevo
    dc.cached_download_bytes("inst_001", "schematic_main.png")
    assert _fake.download_calls == 2, (
        f"después de invalidar esperaba 2 llamadas totales, hubo {_fake.download_calls}"
    )
    print(f"  ✓ Después de invalidar: counter={_fake.download_calls} (MISS forzado)")
    print()


def t3_invalidate_whole_instance() -> None:
    print("=" * 60)
    print("TEST 3 — invalidate_document(toda la instancia)")
    print("=" * 60)
    dc.clear_all_cache()
    _fake.download_calls = 0

    # Cachear 2 archivos de inst_001 + 1 de inst_002
    dc.cached_download_bytes("inst_001", "schematic_main.png")
    dc.cached_download_bytes("inst_001", "datasheet.pdf")
    dc.cached_download_bytes("inst_002", "bently.bn")
    assert _fake.download_calls == 3

    # Invalidar TODA inst_001 (sin pasar storage_filename)
    n = dc.invalidate_document("inst_001")
    assert n == 2, f"esperaba 2 archivos invalidados de inst_001, fueron {n}"
    print(f"  ✓ Invalidados {n} archivos de inst_001 (esperado: 2)")

    # inst_002 debería seguir cacheado
    dc.cached_download_bytes("inst_002", "bently.bn")  # debería ser HIT
    assert _fake.download_calls == 3, "inst_002 cache se rompió por invalidar inst_001"
    print(f"  ✓ inst_002 sigue cacheado (counter={_fake.download_calls})")

    # Re-fetch de inst_001 → 2 MISS más
    dc.cached_download_bytes("inst_001", "schematic_main.png")
    dc.cached_download_bytes("inst_001", "datasheet.pdf")
    assert _fake.download_calls == 5
    print(f"  ✓ inst_001 re-cacheada: counter={_fake.download_calls} (esperado 5)")
    print()


def t4_ttl_expiration() -> None:
    print("=" * 60)
    print("TEST 4 — TTL expirado fuerza MISS")
    print("=" * 60)
    dc.clear_all_cache()
    _fake.download_calls = 0

    # Cachear con TTL muy chico (1 segundo)
    dc.cached_download_bytes("inst_001", "schematic_main.png", ttl_seconds=1)
    assert _fake.download_calls == 1

    # Mientras el TTL esté vigente, HIT
    dc.cached_download_bytes("inst_001", "schematic_main.png", ttl_seconds=1)
    assert _fake.download_calls == 1
    print("  ✓ Dentro del TTL: HIT (counter sigue en 1)")

    # Esperar que expire
    time.sleep(1.5)

    # Ahora debe ser MISS
    dc.cached_download_bytes("inst_001", "schematic_main.png", ttl_seconds=1)
    assert _fake.download_calls == 2
    print(f"  ✓ Después del TTL: MISS forzado (counter={_fake.download_calls})")
    print()


def t5_clear_all() -> None:
    print("=" * 60)
    print("TEST 5 — clear_all_cache borra todo")
    print("=" * 60)
    dc.clear_all_cache()
    _fake.download_calls = 0

    dc.cached_download_bytes("inst_001", "schematic_main.png")
    dc.cached_download_bytes("inst_001", "datasheet.pdf")
    dc.cached_download_bytes("inst_002", "bently.bn")

    n = dc.clear_all_cache()
    assert n == 3, f"esperaba 3 archivos borrados, fueron {n}"
    print(f"  ✓ clear_all_cache borró {n} archivos")

    stats = dc.get_cache_stats()
    assert stats["n_files"] == 0
    print(f"  ✓ stats post-clear: n_files={stats['n_files']}, "
          f"size={stats['total_size_human']}")
    print()


def t6_none_response_not_cached() -> None:
    print("=" * 60)
    print("TEST 6 — repo devuelve None (archivo no existe) → no cachea")
    print("=" * 60)
    dc.clear_all_cache()
    _fake.download_calls = 0

    data = dc.cached_download_bytes("inst_999", "no_existe.png")
    assert data is None
    assert _fake.download_calls == 1

    # Próxima llamada NO debería tener cache (no se guardó nada para None)
    data2 = dc.cached_download_bytes("inst_999", "no_existe.png")
    assert data2 is None
    assert _fake.download_calls == 2, (
        f"esperaba 2 calls (None no debe cachearse), hubo {_fake.download_calls}"
    )
    print(f"  ✓ None no se cachea — counter sube cada llamada ({_fake.download_calls})")
    print()


def t7_egress_savings_simulation() -> None:
    print("=" * 60)
    print("TEST 7 — simulación de ahorro de egress en sesión real")
    print("=" * 60)
    dc.clear_all_cache()
    _fake.download_calls = 0

    # Simulamos un user que en una sesión interactúa 30 veces con un módulo
    # que necesita el schematic. Con cache: 1 download. Sin cache: 30.
    file_size = len(_fake.bucket[("inst_001", "schematic_main.png")])

    n_interactions = 30
    for _ in range(n_interactions):
        dc.cached_download_bytes("inst_001", "schematic_main.png")

    bytes_with_cache = _fake.download_calls * file_size
    bytes_without_cache = n_interactions * file_size
    saved = bytes_without_cache - bytes_with_cache

    print(f"  Interacciones simuladas:    {n_interactions}")
    print(f"  Llamadas reales al repo:    {_fake.download_calls}")
    print(f"  Bytes egress CON cache:     {bytes_with_cache:,}")
    print(f"  Bytes egress SIN cache:     {bytes_without_cache:,}")
    print(f"  Bytes ahorrados:            {saved:,}  "
          f"(reducción {100*saved/bytes_without_cache:.1f}%)")

    assert _fake.download_calls == 1
    print()


def main() -> int:
    print()
    print("########################################################")
    print("# SMOKE TEST — Ciclo 17.18 document_cache")
    print(f"# (cache test path: {TMP_CACHE})")
    print("########################################################")
    print()
    try:
        t1_first_download_misses()
        t2_invalidate_specific_file()
        t3_invalidate_whole_instance()
        t4_ttl_expiration()
        t5_clear_all()
        t6_none_response_not_cached()
        t7_egress_savings_simulation()
        print("=" * 60)
        print(" ✅ TODOS LOS SMOKE TESTS PASARON")
        print("=" * 60)
        print()
        shutil.rmtree(TMP_CACHE, ignore_errors=True)
        return 0
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f" ✗ FALLÓ: {e}")
        print("=" * 60)
        shutil.rmtree(TMP_CACHE, ignore_errors=True)
        return 1
    except Exception as e:
        print()
        print("=" * 60)
        print(f" ✗ ERROR INESPERADO: {type(e).__name__}: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        shutil.rmtree(TMP_CACHE, ignore_errors=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
