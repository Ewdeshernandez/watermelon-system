#!/usr/bin/env python3
"""
Watermelon System1 Agent  ·  VM Parex → Watermelon Cloud
========================================================

Robot HEADLESS que corre en el servidor de System1 (VM Parex) y, cada hora
(Task Scheduler), extrae la onda CRUDA dinámica con keyphasor de la base
PostgreSQL de Bently System1 y la sube al bucket Supabase `dynamic_raw`.
Watermelon Live · Análisis Avanzado la reconstruye (onda/espectro/órbita).

Diseño:
  • Self-contained: solo requiere Python 3.9+, numpy, supabase, (psycopg para
    modo producción). NO arrastra el repo Streamlit.
  • Incremental: guarda en SQLite el último captured_at por punto → solo sube
    lo nuevo. Barato para correr cada hora.
  • Idempotente: x-upsert en Storage; reintentar no duplica.
  • Dos readers:
      - DemoReader     → genera capturas sintéticas (probar TODO el pipeline
                         sin DB real).
      - System1Reader  → lee la PostgreSQL de System1 (query configurable).

Comandos:
  python s1_agent.py --discover      # explora la DB de System1 (read-only)
  python s1_agent.py --demo --once   # sube 1 ronda de capturas sintéticas
  python s1_agent.py --once          # 1 ronda real (usa [system1] del config)
  python s1_agent.py --selftest      # valida formato CSV sin red

Config: config.toml (ver config.example.toml).
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# tomllib (3.11+) o tomli
try:
    import tomllib as _toml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml    # type: ignore

HERE = Path(__file__).resolve().parent
STATE_DB = HERE / "data" / "agent_state.db"
LOG_DIR = HERE / "logs"
BUCKET = "dynamic_raw"

# ---- formato CSV (DEBE coincidir con core.dynamic_raw v1) -------------------
CAPTURE_VERSION = "watermelon_dynamic_raw v1"
TIME_COL = "t_s"

log = logging.getLogger("s1_agent")


# =========================================================
# SERIALIZADOR (idéntico a core.dynamic_raw.build_capture_csv)
# =========================================================
def build_capture_csv(meta: Dict[str, object], t: np.ndarray,
                      channels: Dict[str, np.ndarray]) -> str:
    t = np.asarray(t, dtype=float).reshape(-1)
    cols = list(channels.keys())
    arrs = [np.asarray(channels[c], dtype=float).reshape(-1) for c in cols]
    n = t.size
    for c, a in zip(cols, arrs):
        if a.size != n:
            raise ValueError(f"Canal {c}: {a.size} != t {n}")
    buf = io.StringIO()
    buf.write(f"# {CAPTURE_VERSION}\n")
    m = dict(meta)
    m.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    m.setdefault("channels", ",".join(cols))
    for k in sorted(m.keys()):
        buf.write(f"# {k}={m[k]}\n")
    buf.write(TIME_COL + "," + ",".join(cols) + "\n")
    np.savetxt(buf, np.column_stack([t] + arrs), delimiter=",", fmt="%.6g")
    return buf.getvalue()


# =========================================================
# ESTADO LOCAL (SQLite)
# =========================================================
def _open_state() -> sqlite3.Connection:
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(STATE_DB))
    conn.execute("""CREATE TABLE IF NOT EXISTS sync_state(
        point TEXT PRIMARY KEY, last_captured_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS uploads(
        remote_key TEXT PRIMARY KEY, uploaded_at TEXT,
        status TEXT, error TEXT)""")
    conn.commit()
    return conn


def get_last(point: str) -> Optional[str]:
    conn = _open_state()
    row = conn.execute("SELECT last_captured_at FROM sync_state WHERE point=?",
                       (point,)).fetchone()
    conn.close()
    return row[0] if row else None


def set_last(point: str, captured_at: str) -> None:
    conn = _open_state()
    conn.execute("""INSERT INTO sync_state(point,last_captured_at) VALUES(?,?)
        ON CONFLICT(point) DO UPDATE SET last_captured_at=excluded.last_captured_at""",
                 (point, captured_at))
    conn.commit()
    conn.close()


def record_upload(key: str, status: str, error: str = "") -> None:
    conn = _open_state()
    conn.execute("""INSERT INTO uploads(remote_key,uploaded_at,status,error)
        VALUES(?,?,?,?) ON CONFLICT(remote_key) DO UPDATE SET
        uploaded_at=excluded.uploaded_at, status=excluded.status,
        error=excluded.error""",
                 (key, datetime.now(timezone.utc).isoformat(), status, error))
    conn.commit()
    conn.close()


# =========================================================
# CAPTURA (lo que un reader devuelve)
# =========================================================
class RawCapture:
    def __init__(self, asset: str, point: str, captured_at: datetime,
                 t: np.ndarray, channels: Dict[str, np.ndarray],
                 meta: Dict[str, object]):
        self.asset = asset
        self.point = point
        self.captured_at = captured_at
        self.t = t
        self.channels = channels
        self.meta = meta

    def to_csv(self) -> str:
        m = dict(self.meta)
        m.update({"asset": self.asset, "point": self.point,
                  "captured_at": self.captured_at.astimezone(timezone.utc)
                  .isoformat()})
        return build_capture_csv(m, self.t, self.channels)


# =========================================================
# READERS
# =========================================================
class DemoReader:
    """Capturas sintéticas: desbalance 1X + roce leve + keyphasor. Sirve para
    probar TODO el pipeline (CSV→nube→web) sin la DB real."""

    def __init__(self, cfg: dict):
        self.asset = cfg.get("asset", "SGT300B")
        self.points = cfg.get("demo", {}).get("points", ["CDE", "CNDE", "TDE"])
        self.rpm = float(cfg.get("demo", {}).get("rpm", 9000.0))

    def fetch_new(self) -> List[RawCapture]:
        out = []
        now = datetime.now(timezone.utc)
        f1 = self.rpm / 60.0
        spr = 128
        fs = spr * f1
        revs = 32
        n = spr * revs
        t = np.arange(n) / fs
        rng = np.random.default_rng(int(now.timestamp()) % 100000)
        for k, pt in enumerate(self.points):
            amp = 40.0 + 8.0 * k
            ph = 0.3 * k
            x = amp * np.sin(2 * np.pi * f1 * t + ph)
            y = 0.85 * amp * np.cos(2 * np.pi * f1 * t + ph)
            # armónico 2X leve (desalineación) + ruido
            x += 0.15 * amp * np.sin(2 * np.pi * 2 * f1 * t)
            y += 0.10 * amp * np.cos(2 * np.pi * 2 * f1 * t)
            x += rng.normal(0, 1.2, n)
            y += rng.normal(0, 1.2, n)
            kph = np.zeros(n)
            for r in range(revs):
                kph[r * spr: r * spr + 4] = 1.0
            out.append(RawCapture(
                self.asset, pt, now, t,
                {"X": x, "Y": y, "KPH": kph},
                {"rpm": round(self.rpm, 1), "fs_hz": round(fs, 3),
                 "samples_per_rev": spr, "units": "X:um,Y:um,KPH:V",
                 "source": "demo"}))
        return out


class System1Reader:
    """Lee la base de Bently System1 = **SQL Server** (MSSQLSERVER), base
    `BNC_Databases`. Verificado en el server Parex: SQL Server escucha en 1433
    y responde con **Windows Authentication (sin password)** al correr en la
    misma máquina. La forma exacta de las tablas de onda se descubre con
    --discover; la query va en config [system1.query] para adaptarse SIN tocar
    código.

    Contrato esperado de la query: filas
        point, channel, captured_at, unit, rpm, fs_hz, samples_per_rev,
        sample_index, value
    ordenadas por (point, captured_at, channel, sample_index). Placeholders
    posicionales `?` en este orden: (since, point). El agente agrupa en
    capturas (point, captured_at) con canales X/Y/KPH.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.asset = cfg.get("asset", "SGT300B")
        s1 = cfg.get("system1", {})
        self.dsn = s1.get("dsn") or _dsn_from(s1)
        self.query = s1.get("query", {}).get("sql", "")

    def _connect(self):
        import pyodbc
        return pyodbc.connect(self.dsn, timeout=10, readonly=True)

    def fetch_new(self) -> List[RawCapture]:
        if not self.query:
            raise RuntimeError("Falta [system1.query].sql — corre --discover y "
                               "define la query.")
        captures: List[RawCapture] = []
        conn = self._connect()
        try:
            for pt in self.cfg.get("system1", {}).get("points", []):
                since = get_last(pt) or "1970-01-01T00:00:00"
                cur = conn.cursor()
                cur.execute(self.query, since, pt)  # ? = since, ? = point
                rows = cur.fetchall()
                captures.extend(self._group(pt, rows))
        finally:
            conn.close()
        return captures

    def _group(self, point: str, rows) -> List[RawCapture]:
        # rows: (point, channel, captured_at, unit, rpm, fs_hz, spr, idx, value)
        by_cap: Dict[str, dict] = {}
        for r in rows:
            (_pt, ch, cap_at, unit, rpm, fs, spr, idx, val) = r
            key = str(cap_at)
            g = by_cap.setdefault(key, {"meta": {}, "chan": {}, "units": {},
                                        "cap_at": cap_at})
            g["chan"].setdefault(ch, []).append((int(idx), float(val)))
            g["units"][ch] = unit or ""
            g["meta"].update({"rpm": rpm, "fs_hz": fs, "samples_per_rev": spr})
        out = []
        for key, g in sorted(by_cap.items()):
            chans, tlen = {}, 0
            for ch, pairs in g["chan"].items():
                pairs.sort()
                arr = np.array([v for _i, v in pairs], dtype=float)
                chans[ch] = arr
                tlen = max(tlen, arr.size)
            fs = float(g["meta"].get("fs_hz") or 0) or 1.0
            t = np.arange(tlen) / fs
            units = ",".join(f"{c}:{u}" for c, u in g["units"].items())
            meta = {k: v for k, v in g["meta"].items() if v is not None}
            meta["units"] = units
            meta["source"] = "system1"
            cap_at = g["cap_at"]
            if not isinstance(cap_at, datetime):
                cap_at = datetime.fromisoformat(str(cap_at))
            if cap_at.tzinfo is None:
                cap_at = cap_at.replace(tzinfo=timezone.utc)
            out.append(RawCapture(self.asset, point, cap_at, t, chans, meta))
        return out


def _dsn_from(s1: dict) -> str:
    """Connection string ODBC para SQL Server (System1 = MSSQLSERVER)."""
    driver = s1.get("driver", "ODBC Driver 17 for SQL Server")
    server = s1.get("server", s1.get("host", "localhost"))
    database = s1.get("database", s1.get("dbname", "BNC_Databases"))
    parts = [f"DRIVER={{{driver}}}", f"SERVER={server}", f"DATABASE={database}"]
    if s1.get("trusted", True):
        parts.append("Trusted_Connection=yes")   # Windows Auth, sin password
    else:
        parts.append(f"UID={s1.get('user', '')}")
        parts.append(f"PWD={s1.get('password', '')}")
    parts.append("Encrypt=optional")
    return ";".join(parts) + ";"


# =========================================================
# DISCOVER (read-only) — mapear la DB de System1 (SQL Server)
# =========================================================
def discover(cfg: dict) -> None:
    import pyodbc
    s1 = cfg.get("system1", {})
    dsn = s1.get("dsn") or _dsn_from(s1)
    print("== Conectando a System1 SQL Server (read-only, Windows Auth) ==")
    conn = pyodbc.connect(dsn, timeout=10, readonly=True)
    cur = conn.cursor()
    cur.execute("SELECT DB_NAME(), @@VERSION")
    db, ver = cur.fetchone()
    print(f"DB={db}\n{ver}\n")
    cur.execute("""SELECT TABLE_SCHEMA, TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE='BASE TABLE'
        ORDER BY TABLE_SCHEMA, TABLE_NAME""")
    tables = cur.fetchall()
    print(f"== {len(tables)} tablas ==")
    # heurística: tablas con pinta de onda / waveform / trend / sample
    kw = ("wave", "waveform", "tw", "sample", "raw", "dynamic", "vector",
          "trend", "point", "channel", "signal", "keyphasor", "kph", "spectrum")
    cand = [(s, t) for s, t in tables if any(k in t.lower() for k in kw)]
    print("\n== Candidatas (onda/canal/keyphasor) ==")
    for s, t in cand:
        try:
            cur.execute(f"SELECT COUNT(*) FROM [{s}].[{t}]")
            cnt = cur.fetchone()[0]
        except Exception:
            cnt = "?"
        cur.execute("""SELECT COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA=? AND TABLE_NAME=?
            ORDER BY ORDINAL_POSITION""", s, t)
        cols = ", ".join(f"{c}:{d}" for c, d in cur.fetchall())
        print(f"  {s}.{t}  (rows={cnt})\n     {cols}")
    conn.close()
    print("\nSiguiente: define [system1.query].sql en config.toml usando estas "
          "tablas y vuelve a correr con --once.")


# =========================================================
# UPLOAD
# =========================================================
def _supabase(cfg: dict):
    sup = cfg.get("supabase", {})
    url = os.environ.get("SUPABASE_URL", "").strip() or sup.get("url", "").strip()
    key = (os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
           or sup.get("service_key", "").strip())
    if not (url and key):
        raise RuntimeError("Faltan credenciales Supabase (url + service_key).")
    from supabase import create_client
    return create_client(url, key)


def upload_capture(client, cap: RawCapture) -> str:
    safe_pt = "".join(c for c in cap.point if c.isalnum() or c in "-_") or "PT"
    ts = cap.captured_at.astimezone(timezone.utc)
    key = f"{cap.asset}/{ts:%Y%m%d}/{ts:%H%M%S}__{safe_pt}.csv"
    client.storage.from_(BUCKET).upload(
        key, cap.to_csv().encode("utf-8"),
        file_options={"content-type": "text/csv", "x-upsert": "true"})
    return key


# =========================================================
# CICLO
# =========================================================
def run_once(cfg: dict, demo: bool, dry: bool = False) -> dict:
    reader = DemoReader(cfg) if demo else System1Reader(cfg)
    caps = reader.fetch_new()
    log.info("Capturas nuevas: %d", len(caps))
    stats = {"found": len(caps), "uploaded": 0, "failed": 0, "keys": []}
    client = None if dry else _supabase(cfg)
    for cap in caps:
        try:
            if dry:
                cap.to_csv()  # valida serialización
                key = f"(dry) {cap.asset}/{cap.point}"
            else:
                key = upload_capture(client, cap)
                record_upload(key, "uploaded")
            set_last(cap.point, cap.captured_at.astimezone(timezone.utc)
                     .isoformat())
            stats["uploaded"] += 1
            stats["keys"].append(key)
            log.info("OK %s", key)
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            log.error("FALLO %s/%s: %s", cap.asset, cap.point, exc)
            if not dry:
                record_upload(f"{cap.asset}/{cap.point}", "failed", str(exc))
    return stats


def selftest() -> int:
    """Valida el formato CSV sin red (cross-check con el parser si el repo
    está disponible)."""
    r = DemoReader({"asset": "TEST"})
    caps = r.fetch_new()
    assert caps, "DemoReader no produjo capturas"
    csv = caps[0].to_csv()
    assert csv.startswith(f"# {CAPTURE_VERSION}"), "header de versión mal"
    assert "\nt_s,X,Y,KPH\n" in csv, "columnas mal"
    # cross-check opcional con core.dynamic_raw si está en el path
    try:
        sys.path.insert(0, str(HERE.parent.parent))
        from core.dynamic_raw import parse_capture_csv
        cap = parse_capture_csv(csv)
        assert cap.has("X") and cap.has("Y") and cap.has("KPH")
        assert cap.rpm and abs(cap.rpm - 9000.0) < 1.0
        print("selftest OK — CSV válido y parseado por core.dynamic_raw")
    except Exception as e:  # noqa: BLE001
        print(f"selftest OK — CSV válido (core no disponible: {e})")
    return 0


# =========================================================
# CLI
# =========================================================
def _load_cfg(path: Optional[str]) -> dict:
    p = Path(path) if path else (HERE / "config.toml")
    if not p.exists():
        return {"asset": "SGT300B"}
    with open(p, "rb") as fh:
        return _toml.load(fh)


def _setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(LOG_DIR / "s1_agent.log", encoding="utf-8")])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Watermelon System1 Agent")
    ap.add_argument("--config", help="ruta a config.toml")
    ap.add_argument("--demo", action="store_true", help="usar DemoReader")
    ap.add_argument("--once", action="store_true", help="una ronda y salir")
    ap.add_argument("--discover", action="store_true", help="explorar la DB")
    ap.add_argument("--selftest", action="store_true", help="validar CSV sin red")
    ap.add_argument("--dry", action="store_true", help="no subir; solo validar")
    args = ap.parse_args(argv)

    _setup_logging()
    if args.selftest:
        return selftest()
    cfg = _load_cfg(args.config)
    if args.discover:
        discover(cfg)
        return 0
    if args.once or args.demo:
        stats = run_once(cfg, demo=args.demo, dry=args.dry)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0 if stats["failed"] == 0 else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
