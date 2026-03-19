import hashlib
import secrets
import getpass

def make_password_hash(password):
    salt = secrets.token_hex(16)
    iterations = 260000
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        iterations
    ).hex()
    return f"pbkdf2_sha256${iterations}${salt}${dk}"

password = getpass.getpass("Nueva clave: ")
confirm = getpass.getpass("Confirmar clave: ")

if password != confirm:
    print("Las claves no coinciden")
else:
    print("\nHASH:\n")
    print(make_password_hash(password))