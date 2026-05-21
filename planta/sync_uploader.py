"""
planta/sync_uploader.py — Upload de TDMS al Watermelon Cloud
==============================================================

Detecta los archivos .tdms en planta/data/captures/ que NO han sido
subidos al Cloud y los sube al bucket Supabase Storage `modal-captures`.

Tracking de subidas:
SQLite local en planta/data/sync_state.db con tabla `uploads` que registra:
- file_path: path local absoluto del .tdms
- uploaded_at: ISO datetime
- remote_path: ruta en el bucket (e.g. user@email.com/2026/05/capture.tdms)
- file_size: bytes
- status: 'uploaded' | 'failed'
- error_msg: detalle del fallo si aplica

Estructura en el bucket:
  modal-captures/
    user1@example.com/
      2026/
        05/
          planta_ema_20260520_134255.tdms
          planta_oma_20260520_145812.tdms
        06/
          ...
    user2@example.com/
      ...

Protección por RLS (configurada en Supabase manualmente):
- Cada user solo puede INSERT objetos bajo su propio folder (su email)
- Cada user solo puede SELECT/READ objetos bajo su propio folder
- Nadie puede modificar/borrar objetos de otros users
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable

_SYNC_DB = Path(__file__).parent / "data" / "sync_state.db"
_BUCKET = "modal-captures"


def _open_db() -> sqlite3.Connection:
    """Abre la BD local y asegura el schema."""
    _SYNC_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_SYNC_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            file_path   TEXT PRIMARY KEY,
            uploaded_at TEXT,
            remote_path TEXT,
            file_size   INTEGER,
            status      TEXT,
            error_msg   TEXT
        )
    """)
    conn.commit()
    return conn


def list_pending(captures_dir: Path) -> List[Path]:
    """
    Lista los .tdms en captures_dir que NO están marcados como 'uploaded' OK.

    Returns:
        Lista de Path absolutos ordenados por fecha de creación (más viejo primero).
    """
    captures_dir = Path(captures_dir)
    if not captures_dir.exists():
        return []

    all_tdms = sorted(captures_dir.glob("*.tdms"), key=lambda p: p.stat().st_mtime)

    conn = _open_db()
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM uploads WHERE status='uploaded'")
    uploaded = {row[0] for row in cur.fetchall()}
    conn.close()

    return [t for t in all_tdms if str(t.resolve()) not in uploaded]


def list_uploaded(captures_dir: Path) -> List[Dict]:
    """
    Lista los .tdms que ya están subidos al Cloud con su info.

    Returns:
        Lista de dicts: {file_name, uploaded_at, remote_path, file_size}
    """
    conn = _open_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT file_path, uploaded_at, remote_path, file_size
        FROM uploads WHERE status='uploaded'
        ORDER BY uploaded_at DESC
    """)
    rows = []
    for fp, ts, rp, sz in cur.fetchall():
        rows.append({
            "file_name": Path(fp).name,
            "uploaded_at": ts,
            "remote_path": rp,
            "file_size": sz,
        })
    conn.close()
    return rows


def upload_one(
    tdms_path: Path,
    user_email: str,
    access_token: str,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Sube UN archivo TDMS al bucket Supabase y registra en SQLite.

    Args:
        tdms_path: ruta local al .tdms
        user_email: email del user logueado
        access_token: JWT válido del user
        on_progress: callback opcional (progress 0-1, mensaje)

    Returns:
        remote_path en el bucket (e.g. "user@email.com/2026/05/file.tdms")
    Raises:
        RuntimeError si el upload falla
    """
    try:
        from supabase import create_client
    except ImportError as exc:
        raise ImportError("supabase-py no instalado: pip install supabase") from exc

    # Import absoluto — Streamlit no carga el folder como package
    try:
        from auth_planta import _get_supabase_credentials  # type: ignore
    except ImportError:
        # Fallback si se llama desde otro contexto donde el path es distinto
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from auth_planta import _get_supabase_credentials  # type: ignore
    url, key = _get_supabase_credentials()

    # v3.31.210 fix — Para que las RLS policies de Storage reconozcan al user
    # logueado (no como anon), el cliente debe inicializarse con headers
    # custom que incluyan tanto la apikey como el Bearer token del JWT.
    # La forma vieja (client.storage.session.headers[...]) NO funciona con las
    # nuevas publishable keys (sb_publishable_*) ni con el storage3 moderno.
    from supabase.client import ClientOptions  # type: ignore
    options = ClientOptions(
        headers={
            "apikey": key,
            "Authorization": f"Bearer {access_token}",
        }
    )
    client = create_client(url, key, options=options)
    # Doble seguridad: también setear el session de auth del cliente para que
    # postgrest y demás endpoints reconozcan al user
    try:
        client.auth.set_session(access_token, refresh_token="")
    except Exception:
        # set_session puede requerir refresh_token válido; si falla, los
        # headers custom de ClientOptions siguen vigentes
        pass
    try:
        client.postgrest.auth(access_token)
    except Exception:
        pass

    tdms_path = Path(tdms_path).resolve()
    if not tdms_path.exists():
        raise FileNotFoundError(f"No existe: {tdms_path}")

    now = datetime.now()
    safe_email = user_email.replace("/", "_").replace("\\", "_")
    remote_path = f"{safe_email}/{now.year}/{now.month:02d}/{tdms_path.name}"

    if on_progress:
        on_progress(0.1, f"Leyendo {tdms_path.name}...")

    file_bytes = tdms_path.read_bytes()
    file_size = len(file_bytes)

    if on_progress:
        on_progress(0.3, f"Subiendo {file_size / (1024*1024):.1f} MB...")

    try:
        client.storage.from_(_BUCKET).upload(
            remote_path,
            file_bytes,
            file_options={
                "content-type": "application/octet-stream",
                "x-upsert": "true",  # sobreescribe si ya existe
            },
        )
    except Exception as exc:
        # Registrar fallo en SQLite
        conn = _open_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO uploads
                (file_path, uploaded_at, remote_path, file_size, status, error_msg)
            VALUES (?, ?, ?, ?, 'failed', ?)
            ON CONFLICT(file_path) DO UPDATE SET
                uploaded_at=excluded.uploaded_at,
                status='failed',
                error_msg=excluded.error_msg
        """, (str(tdms_path), now.isoformat(), remote_path, file_size, str(exc)))
        conn.commit()
        conn.close()
        raise RuntimeError(f"Upload falló: {exc}") from exc

    if on_progress:
        on_progress(0.95, f"Registrando en BD local...")

    # Registrar exito
    conn = _open_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO uploads
            (file_path, uploaded_at, remote_path, file_size, status, error_msg)
        VALUES (?, ?, ?, ?, 'uploaded', NULL)
        ON CONFLICT(file_path) DO UPDATE SET
            uploaded_at=excluded.uploaded_at,
            remote_path=excluded.remote_path,
            file_size=excluded.file_size,
            status='uploaded',
            error_msg=NULL
    """, (str(tdms_path), now.isoformat(), remote_path, file_size))
    conn.commit()
    conn.close()

    if on_progress:
        on_progress(1.0, f"✓ {tdms_path.name} subido")

    return remote_path


def sync_all(
    captures_dir: Path,
    user_email: str,
    access_token: str,
    on_file_done: Optional[Callable[[int, int, str, str], None]] = None,
) -> Dict:
    """
    Sube TODOS los TDMS pendientes en captures_dir.

    Args:
        captures_dir: carpeta con .tdms
        user_email, access_token: credenciales del user
        on_file_done: callback (current_idx, total, file_name, status_msg)

    Returns:
        dict {"total": N, "uploaded": M, "failed": K, "errors": [...]}
    """
    pending = list_pending(captures_dir)
    total = len(pending)
    uploaded = 0
    failed = 0
    errors = []

    for i, tdms in enumerate(pending):
        try:
            upload_one(tdms, user_email, access_token)
            uploaded += 1
            if on_file_done:
                on_file_done(i + 1, total, tdms.name, "✓ subido")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            errors.append({"file": tdms.name, "error": str(exc)})
            if on_file_done:
                on_file_done(i + 1, total, tdms.name, f"✗ {exc}")

    return {
        "total": total,
        "uploaded": uploaded,
        "failed": failed,
        "errors": errors,
    }


def get_sync_stats(captures_dir: Path) -> Dict:
    """Stats rápidas para mostrar en la UI sin tocar internet."""
    pending = list_pending(captures_dir)
    uploaded = list_uploaded(captures_dir)
    return {
        "pending": len(pending),
        "uploaded": len(uploaded),
        "pending_mb": sum(p.stat().st_size for p in pending) / (1024 * 1024),
    }
