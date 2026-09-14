"""
core/modal/licensing.py — Licenciamiento del cliente (Fase 1)
=============================================================

Defensas del lado del CLIENTE (el .exe). La confianza vive en un TOKEN firmado
por el servidor con una llave PRIVADA que NUNCA está en el binario; el cliente
sólo trae la llave PÚBLICA para VERIFICAR (no puede forjar).

Amenazas cubiertas aquí:
  · Copia a otra máquina / VM  → `machine_fingerprint()` (binding) + `detect_vm()`.
  · Atrasar el reloj / editar la fecha local → `check_rollback()` (última hora vista,
    firmada; si el reloj va hacia atrás → sospecha) + la expiración vive en el token
    firmado, no en la fecha local.
  · Parchar el chequeo → la verificación es criptográfica (Ed25519); sin la llave
    privada del servidor no se puede emitir/forjar un token.

La activación online, la emisión de tokens y la revocación (kill-switch) las hace el
servidor (Fase 2). Aquí quedan las primitivas y la verificación local.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import time
import uuid
from typing import Any, Dict, Optional, Tuple

# Llave PÚBLICA del emisor de licencias (Ed25519, base64url). Es PÚBLICA por diseño:
# sólo VERIFICA; no puede firmar. La PRIVADA vive SOLO en el servidor (Edge Function).
# Segura de embeber en el .exe. Override por env para pruebas.
LICENSE_PUBKEY_B64 = os.environ.get("WM_LICENSE_PUBKEY", "") or \
    "6yd-Rfp0GEdlFo_hLZ3O0oQD890vc_ylecJi4TyYWzA"

_STATE_DIR = os.path.join(os.path.expanduser("~"), ".watermelon")
_STATE_FILE = os.path.join(_STATE_DIR, "wm_license.json")
_CLOCK_SKEW_S = 36 * 3600          # tolerancia de reloj (36 h) para el anti-rollback


# =====================================================================
# 1) Huella de máquina (machine binding)
# =====================================================================
def _disk_serial() -> str:
    """Serial del disco/volumen del sistema (best-effort, multiplataforma)."""
    try:
        if platform.system() == "Windows":
            import subprocess
            out = subprocess.run(["wmic", "diskdrive", "get", "SerialNumber"],
                                 capture_output=True, text=True, timeout=4).stdout
            for ln in out.splitlines()[1:]:
                s = ln.strip()
                if s:
                    return s
        else:
            # macOS/Linux: UUID del volumen raíz si está disponible
            for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
                if os.path.exists(p):
                    return open(p).read().strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _cpu_id() -> str:
    try:
        if platform.system() == "Windows":
            import subprocess
            out = subprocess.run(["wmic", "cpu", "get", "ProcessorId"],
                                 capture_output=True, text=True, timeout=4).stdout
            for ln in out.splitlines()[1:]:
                s = ln.strip()
                if s:
                    return s
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or ""


def machine_fingerprint() -> str:
    """Huella ESTABLE del equipo (hash). Combina MAC + CPU + disco + plataforma.
    Igual en cada arranque de la MISMA máquina; distinta en otra máquina/VM clonada."""
    parts = [
        str(uuid.getnode()),                 # MAC (48 bits)
        _cpu_id(),
        _disk_serial(),
        platform.machine(),
        platform.system(),
        platform.node(),                     # hostname
    ]
    raw = "|".join(p for p in parts if p)
    return hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()


# =====================================================================
# 2) Detección de máquina virtual (clonado de VMs = fraude común)
# =====================================================================
_VM_MAC_PREFIXES = ("00:05:69", "00:0c:29", "00:1c:14", "00:50:56",   # VMware
                    "08:00:27",                                        # VirtualBox
                    "00:15:5d",                                        # Hyper-V
                    "00:16:3e",                                        # Xen
                    "52:54:00")                                        # KVM/QEMU
_VM_HINTS = ("vmware", "virtualbox", "vbox", "qemu", "kvm", "hyper-v", "hyperv",
             "xen", "parallels", "bhyve", "innotek")


def detect_vm() -> Tuple[bool, str]:
    """Devuelve (es_vm, motivo). Heurística: prefijo MAC + strings de fabricante."""
    try:
        node = uuid.getnode()
        mac = ":".join(f"{(node >> (8 * i)) & 0xff:02x}" for i in reversed(range(6)))
        if any(mac.startswith(p) for p in _VM_MAC_PREFIXES):
            return True, f"MAC {mac}"
    except Exception:  # noqa: BLE001
        pass
    blob = ""
    try:
        if platform.system() == "Windows":
            import subprocess
            blob = subprocess.run(["wmic", "computersystem", "get", "Manufacturer,Model"],
                                  capture_output=True, text=True, timeout=4).stdout.lower()
        else:
            blob = (platform.platform() + " " + platform.node()).lower()
    except Exception:  # noqa: BLE001
        blob = platform.platform().lower()
    for h in _VM_HINTS:
        if h in blob:
            return True, h
    return False, ""


# =====================================================================
# 3) Estado local + anti-rollback de reloj
# =====================================================================
def _load_state() -> Dict[str, Any]:
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _save_state(st: Dict[str, Any]) -> None:
    try:
        os.makedirs(_STATE_DIR, exist_ok=True)
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(st, f)
    except Exception:  # noqa: BLE001
        pass


def check_rollback(now: Optional[float] = None) -> Tuple[bool, float]:
    """Anti-rollback: guarda la MAYOR hora vista. Si el reloj actual va MUY por
    detrás de la última vista → probable atraso manual de fecha. Devuelve
    (rollback_sospechoso, ultima_hora_vista). NO es la fuente de verdad de la
    expiración (esa está en el token del servidor), sólo detecta manipulación."""
    now = float(now if now is not None else time.time())
    st = _load_state()
    last = float(st.get("last_seen", 0.0))
    suspicious = now < (last - _CLOCK_SKEW_S)
    if now > last:                       # avanza normal → actualiza la marca
        st["last_seen"] = now
        _save_state(st)
    return suspicious, last


# =====================================================================
# 4) Verificación del token de licencia (firmado por el servidor)
# =====================================================================
def _b64d(s: str) -> bytes:
    s = s + "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode("ascii"))


def verify_license_token(token: str, expected_fingerprint: Optional[str] = None
                         ) -> Tuple[bool, str, Dict[str, Any]]:
    """Verifica un token de licencia `payload.signature` (base64url). El payload es
    JSON {account, machine_fp, exp, features, seat, iat}. Comprueba: firma Ed25519
    con la llave pública embebida, expiración, y binding de huella. Devuelve
    (válido, motivo, payload)."""
    if not token or "." not in token:
        return False, "no_token", {}
    try:
        p_b64, sig_b64 = token.split(".", 1)
        payload_bytes = _b64d(p_b64)
        signature = _b64d(sig_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return False, "malformed", {}

    if not LICENSE_PUBKEY_B64:
        # Sin llave pública configurada (build de desarrollo): no valida como producción.
        return False, "no_pubkey", payload

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        pk = Ed25519PublicKey.from_public_bytes(_b64d(LICENSE_PUBKEY_B64))
        pk.verify(signature, payload_bytes)          # lanza si la firma no cuadra
    except Exception:  # noqa: BLE001
        return False, "bad_signature", payload

    # Expiración (fuente de verdad = token del servidor). Anti-rollback local por si
    # el reloj está atrasado a propósito para que 'exp' parezca futuro.
    now = time.time()
    suspicious, last = check_rollback(now)
    ref = max(now, last)                             # usa la mayor hora conocida
    if float(payload.get("exp", 0)) < ref:
        return False, "expired", payload

    fp = expected_fingerprint or machine_fingerprint()
    if payload.get("machine_fp") and payload["machine_fp"] != fp:
        return False, "machine_mismatch", payload

    return True, ("clock_suspect" if suspicious else "ok"), payload


def local_license_status() -> Dict[str, Any]:
    """Resumen para la UI: huella, VM, token guardado y su validez."""
    st = _load_state()
    is_vm, vm_why = detect_vm()
    token = st.get("token", "")
    ok, reason, payload = verify_license_token(token) if token else (False, "no_token", {})
    return {"fingerprint": machine_fingerprint(), "is_vm": is_vm, "vm_reason": vm_why,
            "valid": ok, "reason": reason, "payload": payload,
            "exp": payload.get("exp"), "account": payload.get("account")}


def store_token(token: str) -> None:
    st = _load_state(); st["token"] = token; _save_state(st)


# =====================================================================
# 5) Activación online (login + Edge Function) y GATE de arranque
# =====================================================================
def _supabase_url() -> str:
    for src in ("WM_SUPABASE_URL", "SUPABASE_URL"):
        if os.environ.get(src):
            return os.environ[src].rstrip("/")
    try:
        from core.remote_monitoring import _cloud_config as _cc
        return str(getattr(_cc, "SUPABASE_URL", "")).rstrip("/")
    except Exception:  # noqa: BLE001
        return ""


def activate_online(email: str, password: str, endpoint: Optional[str] = None
                    ) -> Dict[str, Any]:
    """Login (Supabase) + llama a la Edge Function `activate` con la huella de esta
    máquina; guarda y verifica el token firmado. Devuelve {ok, reason, ...}."""
    try:
        from core.supabase_auth import signin_user
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"auth_unavailable: {e}"}
    r = signin_user(email, password)
    if not r.get("ok"):
        return {"ok": False, "reason": r.get("error", "signin_failed")}
    access = (r.get("session") or {}).get("access_token")
    if not access:
        return {"ok": False, "reason": "no_session"}
    url = endpoint or (_supabase_url() + "/functions/v1/activate")
    is_vm, _ = detect_vm()
    body = json.dumps({"machine_fp": machine_fingerprint(), "is_vm": is_vm}).encode()
    try:
        import urllib.request
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {access}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"activate_failed: {e}"}
    token = data.get("token")
    if not token:
        return {"ok": False, "reason": data.get("error", "no_token")}
    store_token(token)
    ok, why, payload = verify_license_token(token)
    return {"ok": ok, "reason": why, "account": payload.get("account"), "exp": payload.get("exp")}


def gate_check(grace_recheck_days: int = 14) -> Dict[str, Any]:
    """Decisión de arranque. Devuelve {allowed, reason, needs_activation, account, exp,
    recheck_soon}. La app corre si hay token válido (funciona OFFLINE hasta que expire);
    si no, exige activación. `recheck_soon` sugiere refrescar el token online."""
    st = _load_state()
    token = st.get("token", "")
    if not token:
        return {"allowed": False, "reason": "no_license", "needs_activation": True}
    ok, why, payload = verify_license_token(token)
    if ok:
        exp = float(payload.get("exp", 0))
        recheck = (exp - time.time()) < (grace_recheck_days * 86400)
        return {"allowed": True, "reason": why, "needs_activation": False,
                "account": payload.get("account"), "exp": exp, "recheck_soon": recheck}
    # inválido: expirado / otra máquina / firma mala → exige (re)activación
    return {"allowed": False, "reason": why, "needs_activation": True,
            "account": payload.get("account")}
