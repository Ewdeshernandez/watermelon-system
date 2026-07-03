"""
planta/updater.py — Auto-update checker para Watermelon Planta Edition
=========================================================================

Chequea GitHub Releases del repo Watermelon al arrancar la app y muestra
un banner si hay versión nueva. Cachea 24h para no spammear GitHub.

DISEÑO:
  • Solo LEE de GitHub API. No manda nada de la máquina del cliente.
    Cero telemetría — ni siquiera User-Agent identificable más allá de
    "WatermelonPlanta".
  • Timeout 5s. Si no hay internet (Planta es offline-first), la app
    arranca igual y nunca se entera del check.
  • Cache JSON en data/.update_check_cache.json (gitignored).
  • Comparación de versiones es lexicográfica por tupla de ints
    (v3.31.215 → (3,31,215)), tolerante a prefijo "v".
  • UI: banner sutil amarillo + link directo al .exe del release.
    NO se auto-descarga ni auto-instala — el cliente decide cuándo.
  • Opt-out: el cliente puede deshabilitar editando un flag en data/.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

GITHUB_OWNER = "Ewdeshernandez"
GITHUB_REPO = "watermelon-system"
GITHUB_RELEASES_API = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
)
CHECK_INTERVAL_HOURS = 24
TIMEOUT_SECONDS = 5
USER_AGENT = "WatermelonPlanta"

# Filename del cache local
_CACHE_FILENAME = ".update_check_cache.json"
# Flag para deshabilitar el updater (cliente pone este archivo y ya no chequea)
_OPTOUT_FILENAME = ".no_updates.flag"


# ============================================================================
# DATA CLASS
# ============================================================================

@dataclass
class UpdateInfo:
    has_update: bool = False
    current_version: str = ""
    latest_version: str = ""
    release_url: str = ""           # html_url del release (página GitHub)
    download_url: str = ""          # .exe directo si está, sino html_url
    release_notes: str = ""         # primeros 500 chars del body
    published_at: str = ""          # ISO datetime de publicación
    checked_at: str = ""            # cuándo hicimos el check (UTC ISO)
    from_cache: bool = False
    error: str = ""                 # si no vacío, hubo problema


# ============================================================================
# HELPERS
# ============================================================================

def _get_data_dir() -> Path:
    """Mismo path strategy que license_manager.py — soporta dev + bundled."""
    env_dir = os.environ.get("WATERMELON_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "data"
    return Path(__file__).parent / "data"


def _version_tuple(v: str) -> tuple:
    """'v3.31.215' → (3, 31, 215). Tolerante a prefijo 'v' y partes con texto."""
    v = (v or "").strip().lstrip("v").lstrip("V")
    out = []
    for part in v.split("."):
        # Quedarnos con dígitos consecutivos del principio
        digits = ""
        for c in part:
            if c.isdigit():
                digits += c
            else:
                break
        try:
            out.append(int(digits) if digits else 0)
        except ValueError:
            out.append(0)
    # Pad a 3 niveles para comparaciones consistentes
    while len(out) < 3:
        out.append(0)
    return tuple(out)


def _is_optout() -> bool:
    """Cliente puede crear data/.no_updates.flag para deshabilitar el check."""
    return (_get_data_dir() / _OPTOUT_FILENAME).exists()


def _read_cache() -> dict | None:
    cache_path = _get_data_dir() / _CACHE_FILENAME
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(info: UpdateInfo) -> None:
    cache_path = _get_data_dir() / _CACHE_FILENAME
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(asdict(info), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass  # no es fatal si no podemos cachear


# ============================================================================
# CORE: chequeo
# ============================================================================

def check_for_updates(
    current_version: str,
    force: bool = False,
) -> UpdateInfo:
    """
    Chequea si hay versión nueva en GitHub Releases.

    Args:
        current_version: versión actual (string tipo 'v3.31.215').
        force: si True, ignora el cache y siempre hace request a GitHub.

    Returns:
        UpdateInfo. Si `has_update=True`, hay versión nueva.
        Si `error != ""`, hubo problema (sin internet, GitHub down, etc).
        En ambos casos seguros — nunca lanza excepción.
    """
    # 1. Opt-out del cliente
    if _is_optout():
        return UpdateInfo(
            current_version=current_version,
            error="opted-out",
        )

    # 2. Cache reciente?
    if not force:
        cached = _read_cache()
        if cached:
            try:
                checked_at = datetime.fromisoformat(cached.get("checked_at", ""))
                if datetime.utcnow() - checked_at < timedelta(
                    hours=CHECK_INTERVAL_HOURS
                ):
                    cached["from_cache"] = True
                    # Recomputar has_update por si bumpeamos la versión local
                    # entre arranques pero no cambió la release de GitHub
                    cached["has_update"] = (
                        _version_tuple(cached.get("latest_version", ""))
                        > _version_tuple(current_version)
                    )
                    cached["current_version"] = current_version
                    return UpdateInfo(**{
                        k: v for k, v in cached.items()
                        if k in UpdateInfo.__dataclass_fields__
                    })
            except (ValueError, TypeError):
                pass  # cache corrupto, ignorar y refetch

    # 3. Hacer request a GitHub Releases API
    try:
        req = urllib.request.Request(
            GITHUB_RELEASES_API,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # 404 = el repo aún no tiene NINGÚN release publicado.
        # No es un error real — solo significa "no hay versión nueva todavía".
        # Lo cacheamos como "sin update" para no repegar a GitHub 24h.
        if e.code == 404:
            info = UpdateInfo(
                has_update=False,
                current_version=current_version,
                latest_version=current_version,
                checked_at=datetime.utcnow().isoformat(),
            )
            _write_cache(info)
            return info
        # 403 = rate limit de GitHub API (60 req/h sin auth).
        # 5xx = GitHub down. Ambos transitorios.
        return UpdateInfo(
            current_version=current_version,
            error=f"http_{e.code}",
            checked_at=datetime.utcnow().isoformat(),
        )
    except (urllib.error.URLError, TimeoutError,
            ConnectionError, OSError) as e:
        # Sin internet (Planta es offline-first) — silencioso
        return UpdateInfo(
            current_version=current_version,
            error=f"network: {type(e).__name__}",
            checked_at=datetime.utcnow().isoformat(),
        )
    except json.JSONDecodeError as e:
        return UpdateInfo(
            current_version=current_version,
            error=f"parse: {e}",
            checked_at=datetime.utcnow().isoformat(),
        )
    except Exception as e:  # noqa: BLE001
        return UpdateInfo(
            current_version=current_version,
            error=f"unknown: {type(e).__name__}: {e}",
            checked_at=datetime.utcnow().isoformat(),
        )

    # 4. Parsear respuesta de GitHub
    latest_tag = str(data.get("tag_name", "")).strip()
    release_url = str(data.get("html_url", ""))
    body = str(data.get("body", ""))
    published_at = str(data.get("published_at", ""))

    # 5. Buscar el installer: primero como asset de GitHub (.exe con
    #    "planta"), y si no hay assets (v3.31.417 — el pipeline sube el .exe
    #    a Supabase Storage y solo deja el LINK FIRMADO en el body del
    #    release), extraer la URL de "[DESCARGA DIRECTA](...)" del body.
    download_url = release_url
    for asset in data.get("assets", []):
        name = str(asset.get("name", ""))
        if name.lower().endswith(".exe") and "planta" in name.lower():
            download_url = str(asset.get("browser_download_url", release_url))
            break
    if download_url == release_url and body:
        import re as _re
        m = (_re.search(r"\[DESCARGA DIRECTA\]\((https?://[^)\s]+)\)", body)
             or _re.search(r"(https?://\S+?\.exe(?:\?\S+)?)", body))
        if m:
            download_url = m.group(1)

    # 6. Comparar versiones
    has_update = (
        bool(latest_tag) and
        _version_tuple(latest_tag) > _version_tuple(current_version)
    )

    info = UpdateInfo(
        has_update=has_update,
        current_version=current_version,
        latest_version=latest_tag,
        release_url=release_url,
        download_url=download_url,
        release_notes=body[:500],
        published_at=published_at,
        checked_at=datetime.utcnow().isoformat(),
        from_cache=False,
    )

    # 7. Cachear (incluso si no hay update, para no repegar a GitHub)
    _write_cache(info)

    return info


# ============================================================================
# AUTO-UPDATE (v3.31.398) — descarga el installer del release y lo corre en
# SILENCIO (Inno Setup /VERYSILENT): el cliente NO reinstala nada a mano.
# Flujo: banner → "Actualizar ahora" (o modo desatendido con
# data/.auto_update_on_start.flag) → descarga .exe → .bat que espera, corre
# el installer silencioso y relanza la app → la app se cierra sola.
# Solo aplica en el bundle congelado de Windows (PyInstaller): en dev el
# código vive en el repo y se actualiza con git.
# ============================================================================

_AUTOSTART_FLAG = ".auto_update_on_start.flag"


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def is_auto_update_enabled() -> bool:
    """Modo desatendido: si existe data/.auto_update_on_start.flag, la app
    descarga e instala sola al detectar versión nueva con internet."""
    return (_get_data_dir() / _AUTOSTART_FLAG).exists()


def can_self_update(info: "UpdateInfo") -> bool:
    """True si podemos auto-instalar: hay update, corremos como .exe
    congelado en Windows, y hay URL de installer (.exe en el path — las
    URLs firmadas de Supabase llevan ?token=... después del .exe)."""
    url = (info.download_url or "").lower().split("?", 1)[0]
    return bool(
        info.has_update and not info.error
        and _is_frozen() and os.name == "nt"
        and url.endswith(".exe")
    )


def download_installer(info: "UpdateInfo", progress_cb=None) -> Path:
    """Descarga el installer del release a data/updates/. Reanuda NO — si
    existe completo (mismo tamaño esperado desconocido → si existe se
    re-descarga para evitar corruptos). Devuelve el path local."""
    updates_dir = _get_data_dir() / "updates"
    updates_dir.mkdir(parents=True, exist_ok=True)
    fname = (info.download_url.split("?", 1)[0].rsplit("/", 1)[-1]
             or f"update_{info.latest_version}.exe")
    dest = updates_dir / fname
    req = urllib.request.Request(
        info.download_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        done = 0
        with open(dest, "wb") as fh:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress_cb and total:
                    progress_cb(min(done / total, 1.0))
    if dest.stat().st_size < 1024 * 1024:  # <1MB = seguro no es el installer
        dest.unlink(missing_ok=True)
        raise RuntimeError("Descarga incompleta del installer.")
    return dest


def apply_update_and_restart(installer_path: Path) -> None:
    """Lanza el installer en modo silencioso vía un .bat desacoplado y cierra
    la app. El .bat espera 3s (a que muera este proceso), corre Inno Setup
    /VERYSILENT (instala sobre la misma carpeta) y relanza el .exe."""
    import subprocess
    exe_path = Path(sys.executable)  # el WatermelonPlanta.exe congelado
    bat = _get_data_dir() / "updates" / "apply_update.bat"
    bat.write_text(
        "@echo off\r\n"
        "timeout /t 3 /nobreak >nul\r\n"
        f"\"{installer_path}\" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART "
        "/CLOSEAPPLICATIONS /RESTARTAPPLICATIONS=0\r\n"
        f"start \"\" \"{exe_path}\"\r\n"
        f"del \"{installer_path}\" >nul 2>&1\r\n",
        encoding="ascii", errors="replace",
    )
    DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
    subprocess.Popen(["cmd", "/c", str(bat)], creationflags=DETACHED,
                     close_fds=True)
    os._exit(0)  # el .bat toma el control: instala y relanza


def run_auto_update_ui(info: "UpdateInfo") -> None:
    """Botón 'Actualizar ahora' con barra de progreso (Streamlit). Si el modo
    desatendido está activo, arranca solo sin preguntar."""
    try:
        import streamlit as st
    except ImportError:
        return
    if not can_self_update(info):
        return

    _auto = is_auto_update_enabled()
    _clicked = st.button(
        f"⬇ Actualizar AUTOMÁTICAMENTE a {info.latest_version} "
        f"(descarga, instala y reabre sola)",
        key="auto_update_now", type="primary", use_container_width=True)
    if _clicked or (_auto and not st.session_state.get("_au_started")):
        st.session_state["_au_started"] = True
        _bar = st.progress(0.0, text="Descargando actualización…")
        try:
            dest = download_installer(
                info, progress_cb=lambda p: _bar.progress(
                    p, text=f"Descargando actualización… {int(p*100)}%"))
            _bar.progress(1.0, text="Instalando — la app se reiniciará sola "
                                    "en unos segundos…")
            apply_update_and_restart(dest)
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo auto-actualizar: {e}. Usa el botón "
                     f"'Descargar installer' y córrelo manualmente.")


# ============================================================================
# UI (Streamlit)
# ============================================================================

def render_update_banner(info: UpdateInfo) -> None:
    """
    Banner sutil que avisa de nueva versión. Solo se muestra si hay update.
    Llamar después del chip de licencia en app_planta.py:

        info = check_for_updates(_app_version)
        render_update_banner(info)
    """
    if not info.has_update or info.error:
        return

    try:
        import streamlit as st
    except ImportError:
        return

    # Banner amarillo discreto con CTA
    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg,#fffbeb 0%,#fef3c7 100%);
            border:1px solid #f59e0b;border-left:4px solid #f59e0b;
            border-radius:10px;padding:14px 18px;margin-bottom:18px;
            display:flex;align-items:center;justify-content:space-between;
            gap:16px;flex-wrap:wrap;
            box-shadow:0 2px 6px rgba(245,158,11,0.12);">
            <div style="flex:1;min-width:280px;">
                <div style="font-weight:700;color:#92400e;
                            font-size:14px;margin-bottom:2px;">
                    ✨ Nueva versión disponible:
                    <span style="font-family:monospace;
                                  background:rgba(245,158,11,0.15);
                                  padding:1px 8px;border-radius:6px;">
                        {info.latest_version}
                    </span>
                </div>
                <div style="color:#78350f;font-size:12px;opacity:0.85;">
                    Tu versión actual: <code>{info.current_version}</code>
                    &middot; publicada {info.published_at[:10] if info.published_at else ""}
                </div>
            </div>
            <a href="{info.download_url}" target="_blank"
               style="background:#f59e0b;color:white;
                      padding:8px 16px;border-radius:8px;
                      text-decoration:none;font-weight:600;font-size:13px;
                      white-space:nowrap;
                      box-shadow:0 2px 4px rgba(245,158,11,0.3);">
                📥 Descargar installer
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Release notes en expander si hay
    if info.release_notes:
        with st.expander(f"Ver novedades de {info.latest_version}",
                          expanded=False):
            st.markdown(info.release_notes)
            st.caption(f"Página completa: {info.release_url}")


def render_update_check_button(current_version: str) -> None:
    """
    Botón "Chequear updates ahora" para poner en el sidebar.
    Fuerza el bypass del cache de 24h.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    if st.button("🔄 Chequear actualizaciones", key="manual_update_check",
                  use_container_width=True):
        with st.spinner("Consultando GitHub..."):
            info = check_for_updates(current_version, force=True)
        if info.error == "opted-out":
            st.info(
                "Updates deshabilitados. Borra el archivo "
                "`data/.no_updates.flag` para reactivar."
            )
        elif info.error:
            st.warning(
                f"No se pudo chequear (sin internet?): {info.error}"
            )
        elif info.has_update:
            st.success(f"✓ Versión nueva: {info.latest_version}")
            st.rerun()
        else:
            st.success(f"✓ Estás en la última versión ({info.current_version})")


# ============================================================================
# CACHE EN SESSION STATE (para no chequear en cada rerun)
# ============================================================================

_SESSION_KEY = "_watermelon_update_info"


def get_cached_check(current_version: str) -> UpdateInfo:
    """
    Devuelve UpdateInfo desde session_state si ya chequeamos en esta sesión,
    o hace el check y lo cachea. Apropiado para llamar en cada rerun de
    Streamlit sin penalty.
    """
    try:
        import streamlit as st
    except ImportError:
        return check_for_updates(current_version)

    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = check_for_updates(current_version)
    return st.session_state[_SESSION_KEY]


# ============================================================================
# CLI DE DEBUG
# ============================================================================

if __name__ == "__main__":
    # Lee VERSION del repo para test local
    version_file = Path(__file__).parent.parent / "VERSION"
    current = version_file.read_text().strip() if version_file.exists() else "v0.0.0"

    print()
    print("=" * 60)
    print("  WATERMELON UPDATER — debug check")
    print("=" * 60)
    print(f"  Versión local:  {current}")
    print(f"  Endpoint:       {GITHUB_RELEASES_API}")
    print()

    info = check_for_updates(current, force=True)
    if info.error:
        print(f"  ✗ Error: {info.error}")
    elif info.has_update:
        print(f"  ✨ Hay update: {info.latest_version}")
        print(f"     Descarga:  {info.download_url}")
        print(f"     Publicada: {info.published_at}")
        print()
        if info.release_notes:
            print("  RELEASE NOTES (primeros 500 chars):")
            for line in info.release_notes.split("\n")[:10]:
                print(f"    {line}")
    else:
        print(f"  ✓ Estás en la última: {info.latest_version}")
