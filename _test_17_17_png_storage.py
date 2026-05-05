"""
_test_17_17_png_storage.py
==========================
Smoke test del refactor del Ciclo 17.17 (PNG sueltos).

Valida 3 cosas:

1) BAJA DE PESO DEL JSON
   60 items con imagen de 100KB cada uno antes pesaban
   ~67MB en el JSON (base64 inline). Ahora deben pesar
   pocos KB porque solo guardan `image_file`.

2) ROUND-TRIP DE BYTES INTACTOS
   save_report_state → load_report_state debe devolver
   los mismos bytes para cada item (hash check).

3) MIGRACIÓN LEGACY
   Forjamos un JSON con image_bytes_b64 inline (formato
   viejo), lo cargamos y verificamos:
     - image_bytes está poblado
     - se creó el PNG en disco
     - el dict devuelto tiene image_file apuntándolo
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path

# Importar core/report_state desde WatermelonSystem
WM_ROOT = Path("/sessions/wonderful-blissful-pascal/mnt/WatermelonSystem")
sys.path.insert(0, str(WM_ROOT))

from core import report_state as rs  # noqa: E402

# Redirigir DATA_DIR al tmp ANTES de cualquier llamada — así el test
# no escribe nada en WatermelonSystem/data/ (que no podemos limpiar
# por permisos del mount).
TMP_DATA = Path(tempfile.mkdtemp(prefix="wm_smoke_17_17_"))
rs.DATA_DIR = TMP_DATA
rs._LEGACY_REPORT_STATE_FILE = TMP_DATA / "report_state.json"
rs._LEGACY_REPORT_DRAFTS_DIR = TMP_DATA / "report_drafts"
rs.REPORT_STATE_FILE = rs._LEGACY_REPORT_STATE_FILE
rs.REPORT_DRAFTS_DIR = rs._LEGACY_REPORT_DRAFTS_DIR

# Email de prueba — slug diferente del usuario real para no tocar nada
TEST_EMAIL = "test_17_17@example.com"


def _hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} TB"


def _make_fake_png(size: int) -> bytes:
    """Bytes random — el smoke test no necesita PNG real, solo que
    los bytes sobrevivan el round-trip intactos."""
    return secrets.token_bytes(size)


def _cleanup_test_user() -> None:
    user_dir = TMP_DATA / "users" / rs._email_slug(TEST_EMAIL)
    if user_dir.exists():
        try:
            shutil.rmtree(user_dir)
        except Exception:
            pass


def t1_json_size_drop() -> None:
    print("=" * 60)
    print("TEST 1 — BAJA DE PESO DEL JSON con 60 imágenes de 100KB")
    print("=" * 60)
    _cleanup_test_user()

    n_items = 60
    img_size = 100 * 1024  # 100 KB cada una

    items = []
    expected_hashes = {}
    for i in range(n_items):
        item_id = f"smoke_item_{i:03d}"
        b = _make_fake_png(img_size)
        expected_hashes[item_id] = _hash(b)
        items.append({
            "id":          item_id,
            "type":        "figure",
            "title":       f"Figura smoke {i}",
            "notes":       "",
            "signal_id":   f"SIG-{i:03d}",
            "machine":     "TEST-MACHINE",
            "point":       "P1",
            "variable":    "Vel",
            "timestamp":   "2026-05-04T10:00:00",
            "image_bytes": b,
        })

    rs.save_report_state(items=items, meta={"smoke": True}, email=TEST_EMAIL)

    state_file = rs.get_user_state_file(TEST_EMAIL)
    json_size = state_file.stat().st_size
    print(f"  JSON pesa: {_human(json_size)}  (path: {state_file.name})")

    images_dir = rs.get_user_images_dir(TEST_EMAIL)
    pngs = list(images_dir.glob("*.png"))
    pngs_total = sum(p.stat().st_size for p in pngs)
    print(f"  PNGs en disco: {len(pngs)} archivos, total {_human(pngs_total)}")

    # Esperamos que el JSON pese KB, no MB.  Antes (b64 inline) eran ~9MB.
    assert json_size < 200 * 1024, (
        f"JSON pesó {_human(json_size)} — debería ser <200KB con PNG sueltos. "
        f"¿Sigue cayendo a base64 inline?"
    )
    assert len(pngs) == n_items, (
        f"Esperaba {n_items} PNGs en disco, encontré {len(pngs)}"
    )

    # 2) round-trip
    print()
    print("  Verificando round-trip save→load...")
    loaded = rs.load_report_state(email=TEST_EMAIL)
    loaded_items = loaded.get("items", [])
    assert len(loaded_items) == n_items, (
        f"Esperaba {n_items} items cargados, encontré {len(loaded_items)}"
    )

    # Ciclo 17.20: ahora image_bytes se lee LAZY via read_item_image_bytes
    mismatches = 0
    for it in loaded_items:
        iid = it["id"]
        # Verificar que el item tenga image_file (formato lazy)
        if not it.get("image_file"):
            print(f"  ✗ item {iid} sin image_file en disco")
            mismatches += 1
            continue
        # Leer bytes lazy desde disco
        ib = rs.read_item_image_bytes(it)
        if not ib:
            print(f"  ✗ item {iid} read_item_image_bytes devolvió None")
            mismatches += 1
            continue
        h = _hash(ib)
        if h != expected_hashes.get(iid):
            print(f"  ✗ item {iid} hash mismatch: {h} != {expected_hashes.get(iid)}")
            mismatches += 1

    # Verificación adicional 17.20: image_bytes NO debería estar cargado
    # en memoria del item (debe ser lazy)
    in_memory = sum(1 for it in loaded_items if it.get("image_bytes") is not None)
    assert in_memory == 0, (
        f"{in_memory} items tienen image_bytes en memoria — el lazy loading "
        f"del 17.20 no está funcionando, todos deberían cargarse desde disco "
        f"on-demand vía read_item_image_bytes()"
    )

    assert mismatches == 0, f"{mismatches} items con bytes corruptos en el round-trip"
    print(f"  ✓ {n_items}/{n_items} items con bytes idénticos al original")
    print()
    print(f"  RESULTADO: JSON {_human(json_size)} + {len(pngs)} PNGs separados")
    print(f"  (vs. ~{_human(n_items * img_size * 4 // 3)} de JSON con b64 inline)")
    print()


def t2_legacy_migration() -> None:
    print("=" * 60)
    print("TEST 2 — MIGRACIÓN LEGACY image_bytes_b64 → PNG en disco")
    print("=" * 60)
    _cleanup_test_user()

    # Forjamos manualmente un JSON con formato VIEJO (b64 inline)
    fake_bytes = _make_fake_png(50 * 1024)
    fake_b64 = base64.b64encode(fake_bytes).decode("ascii")
    expected_hash = _hash(fake_bytes)

    legacy_payload = {
        "items": [
            {
                "id":         "legacy_item_001",
                "type":       "figure",
                "title":      "Figura legacy",
                "notes":      "",
                "signal_id":  "OLD-SIG",
                "machine":    "OLD-M",
                "point":      "P1",
                "variable":   "Vel",
                "timestamp":  "2026-04-01T10:00:00",
                "image_bytes_b64": fake_b64,   # <-- formato viejo
            }
        ],
        "meta": {},
    }

    state_file = rs.get_user_state_file(TEST_EMAIL)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(legacy_payload, indent=2), encoding="utf-8")

    print(f"  Forjado JSON legacy en: {state_file.name}")
    print(f"  JSON legacy pesa: {_human(state_file.stat().st_size)}")

    # Cargamos — debería migrar transparentemente
    loaded = rs.load_report_state(email=TEST_EMAIL)
    items = loaded.get("items", [])
    assert len(items) == 1, f"esperaba 1 item, encontré {len(items)}"

    it = items[0]
    # Ciclo 17.20: la migración legacy ahora va directo a image_file (lazy)
    # — no carga los bytes en memoria si pudo escribir el PNG.
    assert it.get("image_file"), (
        "después de la carga legacy debería estar poblado image_file (migración)"
    )
    # Leer los bytes lazy y verificar hash
    bytes_lazy = rs.read_item_image_bytes(it)
    assert bytes_lazy, "read_item_image_bytes no devolvió bytes después de migración"
    assert _hash(bytes_lazy) == expected_hash, "los bytes del legacy migrado no matchean"

    # Verificar que el PNG existe en disco
    images_dir = rs.get_user_images_dir(TEST_EMAIL)
    png_path = images_dir / it["image_file"]
    assert png_path.exists(), f"el PNG migrado no existe en disco: {png_path}"
    assert _hash(png_path.read_bytes()) == expected_hash, "PNG en disco corrupto"
    print(f"  ✓ Migración OK: {it['image_file']} ({_human(png_path.stat().st_size)})")

    # Y al re-guardar, el JSON debería quedar SIN b64 (solo image_file)
    rs.save_report_state(items=items, meta=loaded.get("meta", {}), email=TEST_EMAIL)
    re_raw = json.loads(state_file.read_text(encoding="utf-8"))
    re_item = re_raw["items"][0]
    assert "image_file" in re_item, "re-save no preservó image_file"
    assert not re_item.get("image_bytes_b64"), (
        "re-save debería haber eliminado el b64 legacy del JSON"
    )
    print(f"  ✓ Re-save limpió el b64 legacy. JSON ahora pesa: "
          f"{_human(state_file.stat().st_size)}")
    print()


def t3_idempotence_no_image() -> None:
    print("=" * 60)
    print("TEST 3 — Items SIN imagen no rompen y no crean PNGs basura")
    print("=" * 60)
    _cleanup_test_user()

    items = [
        {
            "id":        "text_only_001",
            "type":      "note",
            "title":     "Solo texto",
            "notes":     "Item sin imagen — no debería crear PNG",
            "signal_id": "",
        },
        {
            "id":          "with_image_001",
            "type":        "figure",
            "title":       "Con imagen",
            "image_bytes": _make_fake_png(20 * 1024),
        },
    ]
    rs.save_report_state(items=items, meta={}, email=TEST_EMAIL)

    images_dir = rs.get_user_images_dir(TEST_EMAIL)
    pngs = list(images_dir.glob("*.png"))
    assert len(pngs) == 1, f"esperaba 1 PNG, encontré {len(pngs)}: {[p.name for p in pngs]}"
    assert pngs[0].name.startswith("with_image_001"), (
        f"el PNG no usa el item_id esperado: {pngs[0].name}"
    )
    print(f"  ✓ Solo se creó 1 PNG ({pngs[0].name}) para el item con imagen")

    loaded = rs.load_report_state(email=TEST_EMAIL)
    its = loaded["items"]
    assert len(its) == 2
    text_it = next(i for i in its if i["id"] == "text_only_001")
    img_it  = next(i for i in its if i["id"] == "with_image_001")
    # Ciclo 17.20: image_bytes es lazy, verificamos image_file
    assert not text_it.get("image_file"), "item de texto no debería tener image_file"
    assert img_it.get("image_file"), "item de figura debería tener image_file"
    assert rs.read_item_image_bytes(text_it) is None
    assert rs.read_item_image_bytes(img_it) is not None
    print("  ✓ Round-trip preserva diferenciación texto vs figura (lazy load)")
    print()


def main() -> int:
    print()
    print("########################################################")
    print("# SMOKE TEST — Ciclo 17.17 PNG storage refactor")
    print("########################################################")
    print()
    try:
        t1_json_size_drop()
        t2_legacy_migration()
        t3_idempotence_no_image()
        print("=" * 60)
        print(" ✅ TODOS LOS SMOKE TESTS PASARON")
        print("=" * 60)
        print()
        _cleanup_test_user()
        return 0
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f" ✗ FALLÓ: {e}")
        print("=" * 60)
        _cleanup_test_user()
        return 1
    except Exception as e:
        print()
        print("=" * 60)
        print(f" ✗ ERROR INESPERADO: {type(e).__name__}: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        _cleanup_test_user()
        return 2


if __name__ == "__main__":
    sys.exit(main())
