
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.usuario import Usuario


def buscar_por_usuario(db: Session, nombre_usuario: str):
    consulta = select(Usuario).where(
        Usuario.usuario == nombre_usuario
    )
    return db.scalar(consulta)


def buscar_por_correo(db: Session, correo: str):
    consulta = select(Usuario).where(
        Usuario.correo == correo
    )
    return db.scalar(consulta)


def buscar_por_id(db: Session, usuario_id: int):
    return db.get(Usuario, usuario_id)


def crear_usuario(
    db: Session,
    nombre: str,
    nombre_usuario: str,
    correo: str,
    contrasena_hash: str
):
    nuevo_usuario = Usuario(
        nombre=nombre,
        usuario=nombre_usuario,
        correo=correo,
        contrasena_hash=contrasena_hash
    )

    db.add(nuevo_usuario)
    db.commit()
    db.refresh(nuevo_usuario)

    return nuevo_usuario
