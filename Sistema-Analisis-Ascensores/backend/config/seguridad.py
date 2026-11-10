
import os
import secrets

SECRET_KEY = os.getenv("AFOROVISION_SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError(
        "Falta configurar la variable de entorno "
        "AFOROVISION_SECRET_KEY."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
