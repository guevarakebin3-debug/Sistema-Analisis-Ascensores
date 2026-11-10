
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from backend.models.password_reset_token import PasswordResetToken
from backend.repositories.usuario_repository import buscar_por_correo


def generar_token_recuperacion(db: Session, correo: str):
    correo = correo.strip().lower()
    usuario = buscar_por_correo(db, correo)

    if usuario is None or not usuario.activo:
        return None

    # Invalidar solicitudes anteriores del mismo usuario.
    tokens_anteriores = (
        db.query(PasswordResetToken)
        .filter(
            PasswordResetToken.usuario_id == usuario.id,
            PasswordResetToken.utilizado.is_(False),
        )
        .all()
    )

    for token_anterior in tokens_anteriores:
        token_anterior.utilizado = True

    # Generar un token aleatorio y guardar solamente su hash.
    token = secrets.token_urlsafe(32)

    token_hash = hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()

    ahora = datetime.now(timezone.utc).replace(tzinfo=None)

    nuevo_token = PasswordResetToken(
        usuario_id=usuario.id,
        token_hash=token_hash,
        expira_en=ahora + timedelta(minutes=15),
        utilizado=False,
    )

    db.add(nuevo_token)
    db.commit()

    return token


def buscar_token_valido(db: Session, token: str):
    token_hash = hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()

    ahora = datetime.now(timezone.utc).replace(tzinfo=None)

    return (
        db.query(PasswordResetToken)
        .filter(
            PasswordResetToken.token_hash == token_hash,
            PasswordResetToken.utilizado.is_(False),
            PasswordResetToken.expira_en > ahora,
        )
        .first()
    )
