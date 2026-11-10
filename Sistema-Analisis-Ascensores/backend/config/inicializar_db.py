
from backend.config.database import Base, engine
from backend.models.usuario import Usuario
from backend.models.password_reset_token import PasswordResetToken


def inicializar_base_datos():
    Base.metadata.create_all(bind=engine)
