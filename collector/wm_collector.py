"""
collector/wm_collector.py
=========================

Watermelon Collector — Tier 0 A (Ciclo 23.1).

Servicio Python liviano que corre en el PC de planta (Windows Server con
ZeroTier + Starlink), lee periódicamente del gateway Modbus TCP del Bently
Nevada 3500/92, y reenvía las lecturas al API de Watermelon System en Render.

Diseño:
    1. Lee un mapa Modbus desde JSON (ej. data/modbus_maps/tes1.json).
    2. Cada N segundos (default 10s) hace lecturas Modbus por bloques.
    3. Decodifica float32 según byte_order configurable (ABCD/CDAB/BADC/DCBA).
    4. POSTea a https://watermelon-api-bpv4.onrender.com/v1/ingest/live.
    5. Si falla la red, persiste en SQLite local y reenvía cuando vuelve.

Para correr:

    python wm_collector.py --config wm_collector.config.json

Para probar lectura UNA vez (sin POST):

    python wm_collector.py --config wm_collector.config.json --dry-run

Dependencias:
    pip install pymodbus==3.6.* requests

Config example (wm_collector.config.json):

{
    "api_url": "https://watermelon-api-bpv4.onrender.com",
    "api_key": "watermelon-collector-tes1-2026",
    "modbus_map": "C:\\\\watermelon\\\\collector\\\\modbus_maps\\\\tes1.json",
    "buffer_db": "C:\\\\watermelon\\\\collector\\\\buffer.sqlite",
    "log_dir": "C:\\\\watermelon\\\\collector\\\\logs",
    "host_label": "TES1-WINSRV"
}
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import socket
import sqlite3
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("Falta dependencia: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    from pymodbus.client import ModbusTcpClient
except ImportError:
    print("Falta dependencia: pip install 'pymodbus==3.6.*'", file=sys.stderr)
    sys.exit(1)


COLLECTOR_VERSION = "1.0.0"


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_dir: Optional[str]) -> logging.Logger:
    log = logging.getLogger("wm_collector")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    # stdout siempre
    h_stream = logging.StreamHandler(sys.stdout)
    h_stream.setFormatter(fmt)
    log.addHandler(h_stream)
    # archivo rotativo si tenemos dir
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        h_file = logging.handlers.RotatingFileHandler(
            Path(log_dir) / "wm_collector.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        h_file.setFormatter(fmt)
        log.addHandler(h_file)
    return log


# =============================================================================
# Modbus float decoding — Bently 3500/92 usa float32 IEEE-754, byte order ABCD
# (big-endian) por defecto, pero algunas configs lo invierten a CDAB.
# Aceptamos las 4 combinaciones: ABCD, CDAB, BADC, DCBA.
# =============================================================================

def decode_float32(regs: List[int], byte_order: str = "ABCD") -> Optional[float]:
    """
    Convierte 2 holding registers (16-bit cada uno) en un float32 IEEE-754.
    `regs` viene del read_holding_registers de pymodbus: [reg_high, reg_low].

    byte_order:
        ABCD = big-endian estándar (default Bently).
              high_word=AB, low_word=CD → bytes A B C D
        CDAB = word-swapped (común en PLCs Schneider).
              [reg_low, reg_high] → bytes C D A B
        BADC = byte-swapped dentro de cada word.
        DCBA = full little-endian (raro).
    """
    if regs is None or len(regs) < 2:
        return None
    try:
        hi, lo = int(regs[0]) & 0xFFFF, int(regs[1]) & 0xFFFF
        bo = (byte_order or "ABCD").upper()
        if bo == "ABCD":
            raw = struct.pack(">HH", hi, lo)
        elif bo == "CDAB":
            raw = struct.pack(">HH", lo, hi)
        elif bo == "BADC":
            raw = struct.pack("<HH", hi, lo)
        elif bo == "DCBA":
            raw = struct.pack("<HH", lo, hi)
        else:
            raw = struct.pack(">HH", hi, lo)  # fallback ABCD
        val = struct.unpack(">f", raw)[0]
        # Filtrar NaN / inf
        if val != val or val in (float("inf"), float("-inf")):
            return None
        return float(val)
    except Exception:
        return None


# =============================================================================
# Local buffer — SQLite append-only para resiliencia ante caídas de red
# =============================================================================

class Buffer:
    """Persistencia local de batches no enviados todavía."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=5.0)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS pending_batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0
                )
            """)

    def add(self, payload: Dict[str, Any]) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO pending_batches(payload, created_at) VALUES (?, ?)",
                (json.dumps(payload), datetime.now(timezone.utc).isoformat()),
            )
            return int(cur.lastrowid or 0)

    def list_pending(self, limit: int = 50) -> List[Tuple[int, Dict[str, Any]]]:
        with self._conn() as c:
            cur = c.execute(
                "SELECT id, payload FROM pending_batches ORDER BY id ASC LIMIT ?",
                (limit,),
            )
            return [(int(row[0]), json.loads(row[1])) for row in cur.fetchall()]

    def delete(self, batch_id: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM pending_batches WHERE id=?", (batch_id,))

    def bump_attempts(self, batch_id: int) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE pending_batches SET attempts=attempts+1 WHERE id=?",
                (batch_id,),
            )

    def count(self) -> int:
        with self._conn() as c:
            cur = c.execute("SELECT COUNT(1) FROM pending_batches")
            return int(cur.fetchone()[0])


# =============================================================================
# Modbus reader
# =============================================================================

@dataclass
class ModbusConfig:
    server_ip: str
    port: int = 502
    unit_id: int = 1
    registers_per_value: int = 2
    byte_order: str = "ABCD"


@dataclass
class RegisterDef:
    address: int
    variable: str
    metric: str
    unit: Optional[str]
    sensor_label: Optional[str]
    encoding: str = "float32"
    quality: str = "good"


def load_modbus_map(path: str) -> Tuple[ModbusConfig, List[RegisterDef], Dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    mb = raw.get("modbus", {})
    cfg = ModbusConfig(
        server_ip=str(mb.get("server_ip", "127.0.0.1")),
        port=int(mb.get("port", 502)),
        unit_id=int(mb.get("unit_id", 1)),
        registers_per_value=int(mb.get("registers_per_value", 2)),
        byte_order=str(mb.get("byte_order", "ABCD")),
    )
    regs = []
    for addr_str, spec in (raw.get("registers") or {}).items():
        regs.append(RegisterDef(
            address=int(addr_str),
            variable=str(spec.get("variable", "")),
            metric=str(spec.get("metric", "Direct")),
            unit=spec.get("unit"),
            sensor_label=spec.get("sensor_label"),
            encoding=str(spec.get("encoding", "float32")),
        ))
    return cfg, regs, raw


def read_all_registers(
    client: ModbusTcpClient,
    cfg: ModbusConfig,
    regs: List[RegisterDef],
    log: logging.Logger,
) -> List[Tuple[RegisterDef, Optional[float], str]]:
    """
    Lee cada register del mapa. Retorna lista de (regdef, valor o None, quality).
    Cada read es individual para no asumir bloques contiguos (3500/92 puede
    tener huecos entre channels).
    """
    out: List[Tuple[RegisterDef, Optional[float], str]] = []
    for r in regs:
        try:
            resp = client.read_holding_registers(
                address=r.address,
                count=cfg.registers_per_value,
                slave=cfg.unit_id,
            )
            if resp.isError():  # type: ignore[attr-defined]
                log.warning("Modbus error reg=%d: %s", r.address, resp)
                out.append((r, None, "comm_fail"))
                continue
            val = decode_float32(resp.registers, cfg.byte_order)  # type: ignore[arg-type]
            if val is None:
                out.append((r, None, "comm_fail"))
            else:
                out.append((r, val, "good"))
        except Exception as e:
            log.warning("Modbus exception reg=%d: %s", r.address, e)
            out.append((r, None, "comm_fail"))
    return out


# =============================================================================
# API client
# =============================================================================

@dataclass
class ApiConfig:
    base_url: str
    api_key: str
    timeout_sec: float = 15.0


def post_batch(api: ApiConfig, payload: Dict[str, Any], log: logging.Logger) -> bool:
    url = api.base_url.rstrip("/") + "/v1/ingest/live"
    headers = {
        "Authorization": f"Bearer {api.api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=api.timeout_sec)
        if resp.status_code == 200:
            j = resp.json()
            log.info("POST OK: ingested=%s received=%s", j.get("ingested"), j.get("received"))
            return True
        log.warning("POST status=%d body=%s", resp.status_code, resp.text[:300])
        return False
    except requests.RequestException as e:
        log.warning("POST failed: %s", e)
        return False


# =============================================================================
# Main loop
# =============================================================================

def build_payload(
    instance_id: str,
    samples: List[Tuple[RegisterDef, Optional[float], str]],
    captured_at: datetime,
    host_label: str,
    byte_order: str,
) -> Dict[str, Any]:
    return {
        "instance_id": instance_id,
        "captured_at": captured_at.isoformat(),
        "metadata": {
            "collector_version": COLLECTOR_VERSION,
            "host": host_label,
            "modbus_byte_order": byte_order,
        },
        "readings": [
            {
                "variable": r.variable,
                "metric": r.metric,
                "value": v,
                "unit": r.unit,
                "sensor_label": r.sensor_label,
                "register": r.address,
                "quality": q,
            }
            for (r, v, q) in samples
        ],
    }


def flush_buffer(api: ApiConfig, buf: Buffer, log: logging.Logger, max_per_cycle: int = 20) -> None:
    pending = buf.list_pending(limit=max_per_cycle)
    if not pending:
        return
    log.info("Flushing %d buffered batches", len(pending))
    for batch_id, payload in pending:
        if post_batch(api, payload, log):
            buf.delete(batch_id)
        else:
            buf.bump_attempts(batch_id)
            return  # corta el flush en el primer fallo (probable que la red siga caída)


def main_loop(config_path: str, dry_run: bool = False) -> None:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    log = setup_logging(cfg.get("log_dir"))
    log.info("=== wm_collector v%s starting ===", COLLECTOR_VERSION)
    log.info("config: %s", config_path)

    mb_cfg, regs, raw_map = load_modbus_map(cfg["modbus_map"])
    instance_id = raw_map.get("instance_id") or cfg.get("instance_id") or "unknown"
    poll = int(raw_map.get("poll_interval_seconds") or cfg.get("poll_interval_seconds", 10))

    log.info("instance_id=%s registers=%d poll=%ds", instance_id, len(regs), poll)
    log.info("Modbus target: %s:%d unit=%d byte_order=%s",
             mb_cfg.server_ip, mb_cfg.port, mb_cfg.unit_id, mb_cfg.byte_order)

    api = ApiConfig(
        base_url=cfg.get("api_url", "https://watermelon-api-bpv4.onrender.com"),
        api_key=cfg["api_key"],
        timeout_sec=float(cfg.get("api_timeout_sec", 15.0)),
    )

    buf = Buffer(cfg.get("buffer_db", "wm_collector_buffer.sqlite"))
    host_label = cfg.get("host_label") or socket.gethostname()

    client = ModbusTcpClient(host=mb_cfg.server_ip, port=mb_cfg.port, timeout=8.0)

    while True:
        cycle_start = time.time()
        captured_at = datetime.now(timezone.utc)

        # Asegurar conexión al gateway
        if not client.connect():
            log.warning("Modbus connect failed; sleeping %ds", poll)
            time.sleep(poll)
            continue

        try:
            samples = read_all_registers(client, mb_cfg, regs, log)
        finally:
            try:
                client.close()
            except Exception:
                pass

        good_count = sum(1 for (_, v, q) in samples if q == "good")
        log.info("Cycle read: %d/%d good", good_count, len(samples))

        payload = build_payload(
            instance_id=instance_id,
            samples=samples,
            captured_at=captured_at,
            host_label=host_label,
            byte_order=mb_cfg.byte_order,
        )

        if dry_run:
            log.info("DRY RUN — payload preview:")
            print(json.dumps(payload, indent=2))
            return

        # Intentar POST directo; si falla, al buffer
        if not post_batch(api, payload, log):
            buf.add(payload)
            log.warning("Buffered batch (pending count=%d)", buf.count())
        else:
            # Si el post directo anduvo, intentamos flushear backlog
            flush_buffer(api, buf, log)

        elapsed = time.time() - cycle_start
        sleep_for = max(0.0, poll - elapsed)
        time.sleep(sleep_for)


def main() -> None:
    p = argparse.ArgumentParser(description="Watermelon Collector — Tier 0 A")
    p.add_argument("--config", required=True, help="Path al wm_collector.config.json")
    p.add_argument("--dry-run", action="store_true", help="1 ciclo, imprime payload, no POST")
    args = p.parse_args()
    try:
        main_loop(args.config, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")
    except Exception as e:
        logging.getLogger("wm_collector").exception("Fatal: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
