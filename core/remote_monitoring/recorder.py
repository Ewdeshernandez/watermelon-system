"""Grabador de transitorio — guarda la forma de onda CRUDA en continuo a disco
durante un arranque/parada, para que NO se pierda ningún dato pase lo que pase
(estilo Bently *transient data recorder*). Se engancha al hook `on_block` del
AcqAgent, así persiste CADA bloque (no por refresco). Post-procesable a
Bode/Cascada/Órbita a cualquier resolución de rpm desde el registro completo.

Layout en disco (WM_PERSIST_DIR/remote_monitoring/transients/<instance>/<rec_id>/):
  · manifest.json   — fs, canales, máquina, timestamps, totales
  · data.f32        — bloques crudos float32 concatenados, cada uno C-order (canales, n)
  · index.jsonl     — un renglón por bloque: {seq, off, n, ts}   (para reconstruir)
Se hace flush tras cada bloque → crash-safe (si la app muere, queda todo hasta
el último bloque).
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def _persist_root() -> str:
    root = os.environ.get("WM_PERSIST_DIR") or os.path.join(os.path.expanduser("~"), ".watermelon")
    return os.path.join(root, "remote_monitoring", "transients")


# Retención de grabaciones transitorias locales. Por defecto 60 días; se puede
# ajustar por env (WM_TRANSIENTS_RETENTION_DAYS). Evita que el disco se llene
# solo (bug real: transients llenó /var/data a 100%).
RETENTION_DAYS = int(os.environ.get("WM_TRANSIENTS_RETENTION_DAYS", "60") or 60)


def purge_old_recordings(days: int = None, root: str = None) -> dict:
    """Borra grabaciones transitorias locales con más de `days` días (por fecha
    de modificación de la carpeta rec_*). Devuelve {'deleted','bytes'}.

    Seguro: solo toca .../remote_monitoring/transients/<instance>/rec_*; nunca
    reportes, snapshots ni configuraciones. No crashea ante errores de FS.
    Las grabaciones ya suelen estar sincronizadas al bucket en la nube, así que
    lo local es caché."""
    days = int(days if days is not None else RETENTION_DAYS)
    base = root or _persist_root()
    cutoff = time.time() - days * 86400.0
    deleted = 0
    freed = 0
    try:
        if not os.path.isdir(base):
            return {"deleted": 0, "bytes": 0}
        for inst in os.listdir(base):
            inst_dir = os.path.join(base, inst)
            if not os.path.isdir(inst_dir):
                continue
            for rid in os.listdir(inst_dir):
                if not rid.startswith("rec_"):
                    continue
                rec_dir = os.path.join(inst_dir, rid)
                try:
                    if not os.path.isdir(rec_dir):
                        continue
                    if os.path.getmtime(rec_dir) >= cutoff:
                        continue  # reciente → conservar
                    sz = 0
                    for dp, _dn, fns in os.walk(rec_dir):
                        for fn in fns:
                            try:
                                sz += os.path.getsize(os.path.join(dp, fn))
                            except Exception:
                                pass
                    shutil.rmtree(rec_dir, ignore_errors=True)
                    deleted += 1
                    freed += sz
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        pass
    return {"deleted": deleted, "bytes": freed}


def free_bytes() -> int:
    """Espacio libre en el disco de persistencia (0 si no se puede leer)."""
    p = _persist_root()
    while p and not os.path.isdir(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    try:
        return int(shutil.disk_usage(p or "/").free)
    except Exception:  # noqa: BLE001
        return 0


@dataclass
class RecorderStatus:
    rec_id: str
    n_channels: int
    fs: float
    blocks: int = 0
    samples: int = 0
    bytes: int = 0
    started_ts: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.samples / self.fs if self.fs else 0.0

    @property
    def size_mb(self) -> float:
        return self.bytes / 1e6


class TransientRecorder:
    """Graba bloques crudos a disco. Thread-safe (on_block puede venir de un hilo)."""

    def __init__(self, instance_id: str, fs: float, ch_meta: List[dict],
                 machine: str = "", rec_id: Optional[str] = None) -> None:
        self.fs = float(fs)
        self.ch_meta = list(ch_meta)
        self.n_channels = len(ch_meta)
        self.rec_id = rec_id or time.strftime("rec_%Y%m%d_%H%M%S")
        self.dir = os.path.join(_persist_root(), instance_id, self.rec_id)
        os.makedirs(self.dir, exist_ok=True)
        self._lock = threading.Lock()
        self._data = open(os.path.join(self.dir, "data.f32"), "ab")
        self._index = open(os.path.join(self.dir, "index.jsonl"), "a")
        self._offset = 0
        self._open = True
        self.status = RecorderStatus(self.rec_id, self.n_channels, self.fs, started_ts=time.time())
        self.error: Optional[str] = None
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump({"rec_id": self.rec_id, "fs": self.fs, "machine": machine,
                       "n_channels": self.n_channels, "channels": self.ch_meta,
                       "started": self.status.started_ts}, f, indent=2)

    def append(self, block: np.ndarray) -> None:
        """Persiste un bloque crudo (canales, muestras). Se llama por on_block.
        Si el disco se llena, detiene la grabación SIN romper la adquisición."""
        if not self._open:
            return
        b = np.ascontiguousarray(block, dtype=np.float32)
        if b.ndim != 2 or b.shape[0] != self.n_channels:
            return
        n = int(b.shape[1])
        try:
            with self._lock:
                self._data.write(b.tobytes())
                self._data.flush()
                self._index.write(json.dumps({"seq": self.status.blocks, "off": self._offset,
                                              "n": n, "ts": time.time()}) + "\n")
                self._index.flush()
                self._offset += n
                self.status.blocks += 1
                self.status.samples += n
                self.status.bytes += b.nbytes
        except OSError as e:                         # disco lleno / IO → cierra limpio
            self.error = f"{type(e).__name__}: {e}"
            self._open = False
            for fh in (self._data, self._index):
                try:
                    fh.close()
                except Exception:  # noqa: BLE001
                    pass

    def stop(self) -> RecorderStatus:
        with self._lock:
            if self._open:
                self._open = False
                for fh in (self._data, self._index):
                    try:
                        fh.flush(); fh.close()
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    p = os.path.join(self.dir, "manifest.json")
                    m = json.load(open(p))
                    m.update(stopped=time.time(), samples=self.status.samples,
                             blocks=self.status.blocks, duration_s=self.status.duration_s)
                    json.dump(m, open(p, "w"), indent=2)
                except Exception:  # noqa: BLE001
                    pass
        return self.status

    @property
    def open(self) -> bool:
        return self._open


def list_recordings(instance_id: str) -> List[dict]:
    """Manifiestos de las grabaciones de una instancia, más recientes primero."""
    base = os.path.join(_persist_root(), instance_id)
    out = []
    if not os.path.isdir(base):
        return out
    for rid in os.listdir(base):
        mp = os.path.join(base, rid, "manifest.json")
        if os.path.isfile(mp):
            try:
                m = json.load(open(mp))
                m["_dir"] = os.path.join(base, rid)
                out.append(m)
            except Exception:  # noqa: BLE001
                pass
    out.sort(key=lambda m: m.get("started", 0), reverse=True)
    return out


def _dir_size(rec_dir: str) -> int:
    total = 0
    for fn in os.listdir(rec_dir):
        try:
            total += os.path.getsize(os.path.join(rec_dir, fn))
        except OSError:
            pass
    return total


def is_synced(rec_dir: str) -> bool:
    return os.path.isfile(os.path.join(rec_dir, ".synced"))


def _sb_client():
    """Cliente Supabase. Dos contextos:
      1) WEB (Streamlit): reusa el auth de la app (st.secrets service_key).
      2) CAMPO (.exe headless): credenciales por variables de entorno
         WM_SUPABASE_URL + WM_SUPABASE_KEY (sin Streamlit).
    None si no hay credenciales o no hay internet."""
    try:
        from core.supabase_auth import get_admin_client
        c = get_admin_client()
        if c is not None:
            return c
    except Exception:  # noqa: BLE001
        pass
    url = os.environ.get("WM_SUPABASE_URL")
    key = os.environ.get("WM_SUPABASE_KEY")
    # Ignorar valores de EJEMPLO (placeholder) para que no pisen lo embebido.
    if url and ("TU-PROYECTO" in url or "TU_PROYECTO" in url):
        url = None
    if key and ("TU_SERVICE" in key or "TU-SERVICE" in key or "AQUI" in key):
        key = None
    # Credenciales EMBEBIDAS en el build (anon key) → el .exe de campo sube sin
    # configurar nada. El módulo lo genera el pipeline desde GitHub Secrets; si no
    # existe, no pasa nada (queda offline).
    if not (url and key):
        try:
            from core.remote_monitoring import _cloud_config as _cc
            url = url or getattr(_cc, "SUPABASE_URL", "")
            key = key or getattr(_cc, "SUPABASE_KEY", "")
        except Exception:  # noqa: BLE001
            pass
    # Limpiar espacios/saltos de línea (pegado en secrets) y normalizar el esquema
    url = (url or "").strip().strip('"').strip("'")
    key = (key or "").strip().strip('"').strip("'")
    if url and not url.startswith("http"):
        url = "https://" + url
    url = url.rstrip("/")
    if url and key:
        try:
            from supabase import create_client
            return create_client(url, key)
        except Exception:  # noqa: BLE001
            return None
    return None


_BUCKET = os.environ.get("WM_TRANSIENTS_BUCKET", "transients")


def upload_recording(rec_dir: str) -> dict:
    """Sube la grabación a Supabase (Storage + fila de metadata) y la marca
    `.synced`. Offline-first: si no hay cliente/internet devuelve {ok:False}."""
    import gzip
    client = _sb_client()
    if client is None:
        return {"ok": False, "reason": "offline"}
    try:
        manifest = json.load(open(os.path.join(rec_dir, "manifest.json")))
        rec_id = manifest["rec_id"]
        instance = os.path.basename(os.path.dirname(rec_dir))
        base = f"{instance}/{rec_id}"
        try:
            client.storage.create_bucket(_BUCKET)          # idempotente
        except Exception:  # noqa: BLE001
            pass
        store = client.storage.from_(_BUCKET)
        # gzip la onda cruda: la achica ~3× para pasar el límite de tamaño del
        # Storage y subir mucho más rápido. Se comprime en el hilo de subida, así
        # no congela la UI (el bloqueo real era subir crudo cientos de MB).
        for fn in ("manifest.json", "index.jsonl", "data.f32"):
            p = os.path.join(rec_dir, fn)
            if not os.path.isfile(p):
                continue
            raw = open(p, "rb").read()
            key = f"{base}/{fn}"
            if fn == "data.f32":
                raw = gzip.compress(raw, compresslevel=6)   # buen ratio, rápido
                key += ".gz"
            try:
                store.upload(key, raw, {"upsert": "true"})
            except Exception:  # noqa: BLE001
                store.update(key, raw)
        row = {"rec_id": rec_id, "instance_id": instance, "machine": manifest.get("machine", ""),
               "fs": manifest.get("fs"), "n_channels": manifest.get("n_channels"),
               "samples": manifest.get("samples"), "duration_s": manifest.get("duration_s"),
               "size_bytes": _dir_size(rec_dir), "started": manifest.get("started"),
               "storage_path": base}
        client.table("transient_recordings").upsert(row).execute()
        with open(os.path.join(rec_dir, ".synced"), "w") as f:
            f.write(str(time.time()))
        return {"ok": True, "path": base}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def pending_count(instance_id: str) -> int:
    return sum(1 for m in list_recordings(instance_id) if not is_synced(m["_dir"]))


def sync_pending(instance_id: str) -> Tuple[int, int]:
    """Sube todas las grabaciones no sincronizadas. Devuelve (ok, fallos)."""
    ok = fail = 0
    for m in list_recordings(instance_id):
        if is_synced(m["_dir"]):
            continue
        r = upload_recording(m["_dir"])
        ok += 1 if r.get("ok") else 0
        fail += 0 if r.get("ok") else 1
    return ok, fail


def cloud_recordings(instance_id: str, limit: int = 30) -> List[dict]:
    """Grabaciones en Supabase para esta máquina (para el aviso al especialista)."""
    client = _sb_client()
    if client is None:
        return []
    try:
        res = (client.table("transient_recordings").select("*")
               .eq("instance_id", instance_id).order("started", desc=True).limit(limit).execute())
        return res.data or []
    except Exception:  # noqa: BLE001
        return []


def cloud_recordings_all(limit: int = 60) -> List[dict]:
    """TODAS las grabaciones en Supabase (cualquier máquina). Para que el
    especialista las encuentre desde la web sin depender del nombre de máquina."""
    client = _sb_client()
    if client is None:
        return []
    try:
        res = (client.table("transient_recordings").select("*")
               .order("started", desc=True).limit(limit).execute())
        return res.data or []
    except Exception:  # noqa: BLE001
        return []


def download_recording(instance_id: str, rec_id: str) -> Optional[str]:
    """Descarga una grabación de Supabase a disco local (para reprocesar en la
    Mac). Devuelve el directorio local o None. Idempotente: si ya está local con
    onda cruda, no re-descarga."""
    import gzip
    dest = os.path.join(_persist_root(), instance_id, rec_id)
    if os.path.isfile(os.path.join(dest, "data.f32")) and os.path.isfile(
            os.path.join(dest, "manifest.json")):
        return dest
    client = _sb_client()
    if client is None:
        return None
    base = f"{instance_id}/{rec_id}"
    try:
        os.makedirs(dest, exist_ok=True)
        store = client.storage.from_(_BUCKET)
        for fn in ("manifest.json", "index.jsonl", "data.f32"):
            raw = None
            if fn == "data.f32":
                # nuevas grabaciones: crudo; viejas: .gz → probar ambos
                try:
                    raw = store.download(f"{base}/data.f32")
                except Exception:  # noqa: BLE001
                    try:
                        raw = gzip.decompress(store.download(f"{base}/data.f32.gz"))
                    except Exception:  # noqa: BLE001
                        return None      # la onda cruda es imprescindible
            else:
                try:
                    raw = store.download(f"{base}/{fn}")
                except Exception:  # noqa: BLE001
                    continue
            with open(os.path.join(dest, fn), "wb") as f:
                f.write(raw)
        with open(os.path.join(dest, ".synced"), "w") as f:
            f.write("cloud")
        return dest
    except Exception:  # noqa: BLE001
        return None


def local_usage(instance_id: str) -> Tuple[int, int]:
    """(cantidad, bytes) que ocupan las grabaciones locales de la instancia."""
    recs = list_recordings(instance_id)
    return len(recs), sum(_dir_size(m["_dir"]) for m in recs)


def delete_recording(rec_dir: str) -> int:
    """Borra una grabación local. Devuelve bytes liberados (aprox)."""
    freed = _dir_size(rec_dir) if os.path.isdir(rec_dir) else 0
    try:
        shutil.rmtree(rec_dir)
    except Exception:  # noqa: BLE001
        return 0
    return freed


def clear_recordings(instance_id: str, only_synced: bool = False) -> Tuple[int, int]:
    """Borra grabaciones locales de una instancia. only_synced=True borra solo las
    ya subidas a Supabase. Devuelve (cantidad, bytes_liberados)."""
    cnt = freed = 0
    for m in list_recordings(instance_id):
        rd = m["_dir"]
        if only_synced and not is_synced(rd):
            continue
        f = delete_recording(rd)
        if f or not os.path.isdir(rd):
            cnt += 1
            freed += f
    return cnt, freed


def prune_raw_after_upload(rec_dir: str) -> int:
    """Tras subir a la nube, borra SOLO la onda cruda local (data.f32) para
    liberar disco; deja manifest/index/.synced para seguir listando. Marca .cloud.
    Devuelve bytes liberados."""
    if not is_synced(rec_dir):
        return 0
    p = os.path.join(rec_dir, "data.f32")
    freed = 0
    if os.path.isfile(p):
        try:
            freed = os.path.getsize(p)
            os.remove(p)
            open(os.path.join(rec_dir, ".cloud"), "w").write("1")
        except Exception:  # noqa: BLE001
            return 0
    return freed


def is_raw_local(rec_dir: str) -> bool:
    return os.path.isfile(os.path.join(rec_dir, "data.f32"))


def list_all_recordings() -> List[dict]:
    """Todas las grabaciones locales (de cualquier instancia), recientes primero.
    Sirve para que la app encuentre lo que grabó el colector headless aunque el
    nombre de máquina activo no coincida."""
    base = _persist_root()
    out = []
    if not os.path.isdir(base):
        return out
    for inst in os.listdir(base):
        idir = os.path.join(base, inst)
        if not os.path.isdir(idir):
            continue
        for rid in os.listdir(idir):
            mp = os.path.join(idir, rid, "manifest.json")
            if os.path.isfile(mp):
                try:
                    m = json.load(open(mp))
                    m["_dir"] = os.path.join(idir, rid)
                    m["_instance"] = inst
                    out.append(m)
                except Exception:  # noqa: BLE001
                    pass
    out.sort(key=lambda m: m.get("started", 0), reverse=True)
    return out


def load_recording(rec_dir: str) -> Tuple[dict, np.ndarray]:
    """Reconstruye (manifest, waveform completa (canales, muestras)) del registro."""
    manifest = json.load(open(os.path.join(rec_dir, "manifest.json")))
    nch = int(manifest["n_channels"])
    raw = np.fromfile(os.path.join(rec_dir, "data.f32"), dtype=np.float32)
    idx_path = os.path.join(rec_dir, "index.jsonl")
    cols, pos = [], 0
    if os.path.isfile(idx_path):
        for line in open(idx_path):
            try:
                e = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            n = int(e["n"])
            chunk = raw[pos:pos + nch * n]
            if chunk.size == nch * n:
                cols.append(chunk.reshape(nch, n))
            pos += nch * n
    full = np.hstack(cols) if cols else np.zeros((nch, 0), dtype=np.float32)
    return manifest, full
