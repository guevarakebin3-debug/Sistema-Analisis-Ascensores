
import os
import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from backend.config.database import obtener_db
from backend.models.password_reset_token import PasswordResetToken
from backend.models.usuario import Usuario
from backend.services.auth_service import (
    registrar_usuario,
    autenticar_usuario,
    crear_token_acceso,
    password_hash,
)
from backend.services.email_service import enviar_correo
from backend.services.password_reset_service import (
    generar_token_recuperacion,
    buscar_token_valido,
)

router = APIRouter(
    prefix="/auth",
    tags=["Autenticación"],
)


class DatosRegistro(BaseModel):
    nombre: str = Field(min_length=2, max_length=100)
    usuario: str = Field(
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_.-]+$",
    )
    correo: EmailStr
    contrasena: str = Field(min_length=8, max_length=128)
    clave_registro: str = Field(min_length=1, max_length=128)


class DatosLogin(BaseModel):
    usuario: str = Field(min_length=1, max_length=50)
    contrasena: str = Field(min_length=1, max_length=128)


class SolicitudRecuperacion(BaseModel):
    correo: EmailStr


class RestablecerContrasena(BaseModel):
    token: str = Field(min_length=20, max_length=200)
    nueva_contrasena: str = Field(min_length=8, max_length=128)


def enviar_bienvenida(usuario):
    enviar_correo(
        destinatario=usuario.correo,
        asunto="Bienvenido a AforoVisión",
        contenido=(
            f"Hola {usuario.nombre},\n\n"
            "Tu cuenta de AforoVisión está registrada correctamente.\n\n"
            f"Tu nombre de usuario es: {usuario.usuario}\n\n"
            "Gracias por utilizar nuestro sistema."
        ),
    )


@router.post(
    "/registro",
    status_code=status.HTTP_201_CREATED,
)
def registro(
    datos: DatosRegistro,
    db: Session = Depends(obtener_db),
):
    clave_configurada = os.getenv("AFOROVISION_REGISTRO_CLAVE")

    if not clave_configurada:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El registro está deshabilitado por configuración.",
        )

    if not secrets.compare_digest(
        datos.clave_registro,
        clave_configurada,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clave de registro incorrecta.",
        )

    try:
        usuario = registrar_usuario(
            db=db,
            nombre=datos.nombre,
            nombre_usuario=datos.usuario,
            correo=str(datos.correo),
            contrasena=datos.contrasena,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )

    try:
        enviar_bienvenida(usuario)

        print(f"Correo de bienvenida enviado a {usuario.correo}")

        return {
            "mensaje": "Usuario registrado y correo enviado correctamente.",
            "usuario": usuario.usuario,
            "correo_enviado": True,
        }

    except Exception as error:
        print(
            f"Error al enviar el correo: "
            f"{type(error).__name__}: {error}"
        )

        return {
            "mensaje": (
                "La cuenta se creó, pero no se pudo enviar "
                "el correo de bienvenida."
            ),
            "usuario": usuario.usuario,
            "correo_enviado": False,
        }


@router.post("/reenviar-bienvenida")
def reenviar_bienvenida(
    datos: DatosLogin,
    db: Session = Depends(obtener_db),
):
    usuario = autenticar_usuario(
        db=db,
        nombre_usuario=datos.usuario,
        contrasena=datos.contrasena,
    )

    if usuario is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos.",
        )

    try:
        enviar_bienvenida(usuario)

        print(f"Correo de bienvenida enviado a {usuario.correo}")

        return {
            "mensaje": "Correo de bienvenida enviado correctamente.",
            "correo_enviado": True,
        }

    except Exception as error:
        print(
            f"Error al reenviar el correo: "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo enviar el correo. Revisa la terminal del backend.",
        )


@router.post("/login")
def login(
    datos: DatosLogin,
    db: Session = Depends(obtener_db),
):
    usuario = autenticar_usuario(
        db=db,
        nombre_usuario=datos.usuario,
        contrasena=datos.contrasena,
    )

    if usuario is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Error de acceso: usuario o contraseña incorrectos.",
        )

    token = crear_token_acceso(usuario)

    return {
        "mensaje": "Inicio de sesión correcto.",
        "access_token": token,
        "token_type": "bearer",
        "usuario": {
            "id": usuario.id,
            "nombre": usuario.nombre,
            "usuario": usuario.usuario,
            "correo": usuario.correo,
        },
    }


@router.post("/solicitar-recuperacion")
def solicitar_recuperacion(
    datos: SolicitudRecuperacion,
    db: Session = Depends(obtener_db),
):
    token = generar_token_recuperacion(
        db=db,
        correo=str(datos.correo),
    )

    if token:
        enlace = (
            "http://127.0.0.1:5500/"
            "Sistema-Analisis-Ascensores/frontend/pages/"
            "restablecer-contrasena.html"
            f"?token={token}"
        )

        try:
            enviar_correo(
                destinatario=str(datos.correo),
                asunto="Recuperación de contraseña - AforoVisión",
                contenido=(
                    "Hola,\n\n"
                    "Recibimos una solicitud para restablecer "
                    "tu contraseña de AforoVisión.\n\n"
                    "Abre este enlace para continuar:\n"
                    f"{enlace}\n\n"
                    "El enlace vence en 15 minutos y solo puede "
                    "utilizarse una vez.\n\n"
                    "Si no solicitaste este cambio, ignora este mensaje."
                ),
            )

        except RuntimeError as error:
            print(
                f"Error al enviar recuperación: "
                f"{type(error).__name__}: {error}"
            )

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            )

        except Exception as error:
            print(
                f"Error al enviar recuperación: "
                f"{type(error).__name__}: {error}"
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No se pudo enviar el correo de recuperación.",
            )

    return {
        "mensaje": (
            "Si el correo está registrado, recibirás instrucciones "
            "para recuperar tu contraseña."
        )
    }


@router.post("/restablecer-contrasena")
def restablecer_contrasena(
    datos: RestablecerContrasena,
    db: Session = Depends(obtener_db),
):
    registro_token = buscar_token_valido(
        db=db,
        token=datos.token,
    )

    if registro_token is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El enlace es inválido, venció o ya fue utilizado.",
        )

    usuario = (
        db.query(Usuario)
        .filter(Usuario.id == registro_token.usuario_id)
        .first()
    )

    if usuario is None or not usuario.activo:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se pudo restablecer la contraseña.",
        )

    usuario.contrasena_hash = password_hash.hash(
        datos.nueva_contrasena
    )

    registro_token.utilizado = True

    otros_tokens = (
        db.query(PasswordResetToken)
        .filter(
            PasswordResetToken.usuario_id == usuario.id,
            PasswordResetToken.utilizado.is_(False),
            PasswordResetToken.id != registro_token.id,
        )
        .all()
    )

    for otro_token in otros_tokens:
        otro_token.utilizado = True

    db.commit()

    return {
        "mensaje": "Tu contraseña se restableció correctamente."
    }
