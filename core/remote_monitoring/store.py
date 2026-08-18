"""
core/remote_monitoring/store.py — Store local offline de snapshots live
=======================================================================

Persiste ventanas de datos en el PC de sitio SIN depender de red. Cuando
vuelve internet, `pending_sync()` lista lo no subido para que el uploader
lo empuje a Supabase (mismo espíritu que planta/sync_uploader.py).

Diseño:
  · Índice en SQLite (metadata + estado de sync).
  · Payload de cada snapshot en .npz (numpy portable, sin dependencias):
    array (n_channels, n_samples) + metadata de canales.
  · Dir durable: WM_PERSIST_DIR (Render/sitio) o relativo (dev).
    Mismo patrón que core/modal/modal_session.py.

npz vs TDMS: para el buffer offline usamos npz (rápido, cero deps). El
export a TDMS (formato de intercambio) se hace en el paso de sync a nube.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _store_dir() -> Path:
    pd = os.environ.get("WM_PERSIST_DIR")
    if pd:
        return Path(pd) / "remote_monitoring"
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "data" / "remote_monitoring"
    return Path("data/remote_monitoring")


@dataclass
class SnapshotMeta:
    snapshot_id: str
    instance_id: str
    captured_at: str
    fs: float
    rpm: Optional[float]
    n_channels: int
    n_samples: int
    synced: bool
    path: str


class LocalStore:
    """Store local de snapshots live, offline-first."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else _store_dir()
        self.snap_dir = self.root / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.sqlite"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    instance_id TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    fs REAL NOT NULL,
                    rpm REAL,
                    n_channels INTEGER NOT NULL,
                    n_samples INTEGER NOT NULL,
                    synced INTEGER NOT NULL DEFAULT 0,
                    path TEXT NOT NULL
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS ix_snap_instance ON snapshots(instance_id, captured_at)"
            )

    # --- escritura ---
    def save_snapshot(
        self,
        instance_id: str,
        data: np.ndarray,
        channels: List[Dict[str, Any]],
        fs: float,
        rpm: Optional[float] = None,
        captured_at: Optional[str] = None,
    ) -> SnapshotMeta:
        """Guarda una ventana (n_channels, n_samples) + metadata de canales.

        channels: lista de dicts serializables (name, bnc_port, coupling,
        sensitivity_mv_per_eu, units, role).
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"data debe ser 2D (n_ch, n); recibí {data.shape}")
        if data.shape[0] != len(channels):
            raise ValueError("data.shape[0] != len(channels)")

        captured_at = captured_at or datetime.now(timezone.utc).isoformat()
        # id cronológico-ordenable
        stamp = captured_at.replace(":", "").replace("-", "").replace(".", "").replace("+", "Z")[:20]
        snapshot_id = f"{instance_id}_{stamp}"
        path = self.snap_dir / f"{snapshot_id}.npz"

        np.savez_compressed(
            path,
            data=data,
            channels_json=json.dumps(channels),
            fs=float(fs),
            rpm=(float(rpm) if rpm is not None else np.nan),
            captured_at=captured_at,
            instance_id=instance_id,
        )

        meta = SnapshotMeta(
            snapshot_id=snapshot_id,
            instance_id=instance_id,
            captured_at=captured_at,
            fs=float(fs),
            rpm=(float(rpm) if rpm is not None else None),
            n_channels=data.shape[0],
            n_samples=data.shape[1],
            synced=False,
            path=str(path),
        )
        with self._connect() as con:
            con.execute(
                """INSERT OR REPLACE INTO snapshots
                   (snapshot_id, instance_id, captured_at, fs, rpm, n_channels, n_samples, synced, path)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (meta.snapshot_id, meta.instance_id, meta.captured_at, meta.fs,
                 meta.rpm, meta.n_channels, meta.n_samples, 0, meta.path),
            )
        return meta

    # --- lectura ---
    def _row_to_meta(self, r: sqlite3.Row) -> SnapshotMeta:
        return SnapshotMeta(
            snapshot_id=r["snapshot_id"], instance_id=r["instance_id"],
            captured_at=r["captured_at"], fs=r["fs"], rpm=r["rpm"],
            n_channels=r["n_channels"], n_samples=r["n_samples"],
            synced=bool(r["synced"]), path=r["path"],
        )

    def list_snapshots(self, instance_id: Optional[str] = None,
                       limit: int = 100) -> List[SnapshotMeta]:
        with self._connect() as con:
            if instance_id:
                rows = con.execute(
                    "SELECT * FROM snapshots WHERE instance_id=? ORDER BY captured_at DESC LIMIT ?",
                    (instance_id, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM snapshots ORDER BY captured_at DESC LIMIT ?", (limit,),
                ).fetchall()
        return [self._row_to_meta(r) for r in rows]

    def load_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
        if not row:
            return None
        npz = np.load(row["path"], allow_pickle=False)
        rpm = float(npz["rpm"]) if not np.isnan(npz["rpm"]) else None
        return {
            "snapshot_id": snapshot_id,
            "data": npz["data"],
            "channels": json.loads(str(npz["channels_json"])),
            "fs": float(npz["fs"]),
            "rpm": rpm,
            "captured_at": str(npz["captured_at"]),
            "instance_id": str(npz["instance_id"]),
        }

    # --- sync ---
    def pending_sync(self, limit: int = 100) -> List[SnapshotMeta]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM snapshots WHERE synced=0 ORDER BY captured_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_meta(r) for r in rows]

    def mark_synced(self, snapshot_id: str) -> None:
        with self._connect() as con:
            con.execute("UPDATE snapshots SET synced=1 WHERE snapshot_id=?", (snapshot_id,))

    def count(self, only_pending: bool = False) -> int:
        q = "SELECT COUNT(*) c FROM snapshots" + (" WHERE synced=0" if only_pending else "")
        with self._connect() as con:
            return int(con.execute(q).fetchone()["c"])
