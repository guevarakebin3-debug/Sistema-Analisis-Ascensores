
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


# Ruta de la raíz del proyecto.
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

# Ubicación de la base de datos.
RUTA_BASE_DATOS = (
    RAIZ_PROYECTO / "database" / "aforovision.db"
)

# Crear la carpeta database si todavía no existe.
RUTA_BASE_DATOS.parent.mkdir(
    parents=True,
    exist_ok=True
)

DATABASE_URL = f"sqlite:///{RUTA_BASE_DATOS.as_posix()}"

# Crear la conexión con SQLite.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Crear sesiones para consultar y guardar datos.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)


# Clase base para los modelos de la base de datos.
class Base(DeclarativeBase):
    pass


# Obtener una sesión y cerrarla al terminar.
def obtener_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
