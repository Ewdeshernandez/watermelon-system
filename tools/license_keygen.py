#!/usr/bin/env python3
"""
tools/license_keygen.py — Generador de par de claves RSA inicial
=================================================================

⚠️ CORRER UNA SOLA VEZ EN LA HISTORIA DE WATERMELON PLANTA ⚠️

Esto genera el par de claves criptográficas RSA-2048 que se usan para
todo el sistema de licencias del producto comercial:

  private_key.pem  → QUEDA EN BÓVEDA DE SIGA (NUNCA al repo ni a
                     ningún cliente). Sirve para FIRMAR licencias.
                     Si se filtra, hay que regenerar TODAS las licencias
                     y rebuild Watermelon Planta con nueva public key.

  public_key.pem   → Se EMBEBE en planta/license_manager.py (hardcoded
                     como string Python). Sirve para VERIFICAR licencias
                     en runtime del cliente. Es OK que sea pública.

Uso:
    cd watermelon-system
    python tools/license_keygen.py
    # Output: tools/.keys/private_key.pem + tools/.keys/public_key.pem

Después:
1. Copiar el contenido de public_key.pem al string _EMBEDDED_PUBLIC_KEY
   en planta/license_manager.py
2. private_key.pem → mover a bóveda offline de SIGA (encriptada,
   backup en USB en caja fuerte, etc.)
3. NUNCA commitear tools/.keys/ — ya está en .gitignore
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        print("ERROR: falta el módulo 'cryptography'.")
        print("Instala con:  pip install cryptography")
        return 1

    # Directorio output (gitignored)
    keys_dir = Path(__file__).parent / ".keys"
    keys_dir.mkdir(parents=True, exist_ok=True)

    priv_path = keys_dir / "private_key.pem"
    pub_path = keys_dir / "public_key.pem"

    if priv_path.exists() or pub_path.exists():
        print(f"⚠ Ya existen claves en {keys_dir}")
        print(f"  · {priv_path}")
        print(f"  · {pub_path}")
        print()
        print("  Si generas claves nuevas, las viejas dejan de funcionar")
        print("  y TODAS las licencias ya emitidas quedan inválidas.")
        print()
        resp = input("  ¿Continuar y SOBREESCRIBIR? (escribir 'sobrescribir'): ")
        if resp.strip().lower() != "sobrescribir":
            print("Cancelado. Las claves existentes NO se modificaron.")
            return 0

    print("Generando par RSA-2048 (puede tardar 2-5 segundos)...")
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Serializar private key (formato PEM, sin password — confiamos en
    # el filesystem permissions + offline storage)
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Serializar public key
    public_key = private_key.public_key()
    pub_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    priv_path.write_bytes(priv_pem)
    pub_path.write_bytes(pub_pem)

    # Permisos restrictivos para private key (Unix)
    try:
        priv_path.chmod(0o600)
    except Exception:  # noqa: BLE001
        pass

    print()
    print("=" * 60)
    print("  ✓ PAR DE CLAVES GENERADO")
    print("=" * 60)
    print()
    print(f"  Private key:  {priv_path}")
    print(f"  Public key:   {pub_path}")
    print()
    print("  TAMAÑOS:")
    print(f"    Private:  {len(priv_pem)} bytes")
    print(f"    Public:   {len(pub_pem)} bytes")
    print()
    print("=" * 60)
    print("  PRÓXIMOS PASOS")
    print("=" * 60)
    print()
    print("  1. COPIA el contenido de public_key.pem al string")
    print("     _EMBEDDED_PUBLIC_KEY en planta/license_manager.py")
    print()
    print("  2. MUEVE private_key.pem a BÓVEDA OFFLINE de SIGA:")
    print("     - USB en caja fuerte")
    print("     - Encrypted disk image")
    print("     - 1Password / Bitwarden vault como secret")
    print("     NUNCA al repo. NUNCA a un cliente. NUNCA en email.")
    print()
    print("  3. BACKUP de private_key.pem en 2 lugares diferentes.")
    print("     Si la pierdes, hay que regenerar TODO desde cero.")
    print()
    print("  4. Verifica que tools/.keys/ está en .gitignore:")
    print("       grep 'tools/.keys' .gitignore")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
