
from contextlib import asynccontextmanager
from math import isfinite

from fastapi import (
    FastAPI,
    APIRouter,
    HTTPException,
    UploadFile,
    File,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.services.decision_service import evaluar_ocupacion
from backend.config.inicializar_db import inicializar_base_datos
from backend.controllers.auth_controller import router as auth_router

from backend.services.vision_service import (
    consultar_ocupacion,
    detectar_personas,
    reiniciar_seguimiento,
    reiniciar_sesion_vision,
)


# ==========================================================
# INICIALIZACIÓN
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    inicializar_base_datos()
    yield


# ==========================================================
# APLICACIÓN FASTAPI
# ==========================================================

app = FastAPI(
    title="AforoVisión - Backend Principal",
    description=(
        "Backend principal del sistema web AforoVisión. "
        "Gestiona autenticación, consultas de ocupación, "
        "detección, seguimiento y evaluación de ingreso."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# ROUTER PRINCIPAL
# ==========================================================

router = APIRouter(prefix="/api")


# ==========================================================
# MODELO DE SOLICITUD PARA DECISIONES
# ==========================================================

class DecisionRequest(BaseModel):

    superficie_actual_m2: float = Field(
        ge=0,
        description=(
            "Superficie ocupada actualmente dentro de la cabina."
        ),
    )

    superficie_entrante_m2: float = Field(
        ge=0,
        description=(
            "Superficie estimada que aporta la persona "
            "que intenta ingresar, incluyendo los objetos "
            "que transporta sin duplicarlos."
        ),
    )

    superficie_util_cabina_m2: float = Field(
        gt=0,
        description=(
            "Superficie útil configurada para la cabina."
        ),
    )

    personas_actuales: int = Field(
        ge=0,
        description=(
            "Cantidad de personas que actualmente ocupan "
            "la cabina."
        ),
    )

    personas_entrantes: int = Field(
        ge=0,
        description=(
            "Cantidad de personas que intentan ingresar."
        ),
    )

    capacidad_maxima_personas: int = Field(
        gt=0,
        description=(
            "Capacidad máxima de personas configurada."
        ),
    )

    limite_ocupacion_pct: float = Field(
        ge=0,
        le=100,
        description=(
            "Porcentaje máximo de ocupación espacial permitido."
        ),
    )

    @field_validator(
        "superficie_actual_m2",
        "superficie_entrante_m2",
        "superficie_util_cabina_m2",
        "limite_ocupacion_pct",
    )
    @classmethod
    def validar_valores_finitos(cls, valor):

        if not isfinite(valor):
            raise ValueError(
                "El valor debe ser un número finito."
            )

        return valor


# ==========================================================
# ENDPOINT INICIAL
# ==========================================================

@app.get("/")
def inicio():

    return {
        "sistema": "AforoVisión",
        "servicio": "Backend principal",
        "estado": "disponible",
        "version": "1.2.0",
    }


# ==========================================================
# ESTADO DEL BACKEND
# ==========================================================

@app.get("/api/health")
def health():

    return {
        "servicio": "backend-principal",
        "estado": "disponible",
    }


# ==========================================================
# CONSULTAR OCUPACIÓN
# ==========================================================

@router.get("/vision/ocupacion")
def obtener_ocupacion(
    camara_id: str | None = Query(default=None),
):

    if camara_id is not None and camara_id not in (
        "interior",
        "exterior",
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "La cámara debe ser interior o exterior."
            ),
        )

    return consultar_ocupacion(camara_id)


# ==========================================================
# DETECTAR PERSONAS Y OBJETOS
# ==========================================================

@router.post("/vision/detectar")
def detectar(
    camara_id: str = Query(
        default="interior",
    ),

    linea_porcentaje: float = Query(
        default=50,
        ge=0,
        le=100,
    ),

    area_cabina_m2: float = Query(
        default=3.5,
        gt=0,
    ),

    capacidad: int = Query(
        default=10,
        gt=0,
    ),

    limite_ocupacion: float = Query(
        default=80,
        ge=1,
        le=100,
    ),

    imagen: UploadFile = File(...),
):

    if camara_id not in ("interior", "exterior"):
        raise HTTPException(
            status_code=400,
            detail=(
                "La cámara indicada debe ser interior "
                "o exterior."
            ),
        )

    if not isfinite(linea_porcentaje):
        raise HTTPException(
            status_code=400,
            detail=(
                "El porcentaje de la línea debe ser válido."
            ),
        )

    if not isfinite(area_cabina_m2):
        raise HTTPException(
            status_code=400,
            detail=(
                "El área de la cabina debe ser un número válido."
            ),
        )

    if not isfinite(limite_ocupacion):
        raise HTTPException(
            status_code=400,
            detail=(
                "El límite de ocupación debe ser un número válido."
            ),
        )

    return detectar_personas(
        imagen=imagen,
        camara_id=camara_id,
        linea_porcentaje=linea_porcentaje,
        area_cabina_m2=area_cabina_m2,
        capacidad=capacidad,
        limite_ocupacion=limite_ocupacion,
    )


# ==========================================================
# REINICIAR SEGUIMIENTO DE UNA CÁMARA
# ==========================================================

@router.post("/vision/reiniciar")
def reiniciar_vision(
    camara_id: str = Query(default="interior"),
):

    if camara_id not in ("interior", "exterior"):
        raise HTTPException(
            status_code=400,
            detail=(
                "La cámara indicada debe ser interior "
                "o exterior."
            ),
        )

    return reiniciar_seguimiento(camara_id)


# ==========================================================
# REINICIAR SESIÓN COMPLETA
# ==========================================================

@router.post("/vision/reiniciar-sesion")
def reiniciar_sesion():

    return reiniciar_sesion_vision()


# ==========================================================
# EVALUAR DECISIÓN DE INGRESO
# ==========================================================

@router.post("/decision")
def decision(datos: DecisionRequest):

    try:

        resultado = evaluar_ocupacion(
            superficie_actual_m2=datos.superficie_actual_m2,
            superficie_entrante_m2=datos.superficie_entrante_m2,
            superficie_util_cabina_m2=(
                datos.superficie_util_cabina_m2
            ),
            personas_actuales=datos.personas_actuales,
            personas_entrantes=datos.personas_entrantes,
            capacidad_maxima_personas=(
                datos.capacidad_maxima_personas
            ),
            limite_ocupacion_pct=(
                datos.limite_ocupacion_pct
            ),
        )

        return resultado

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# ==========================================================
# REGISTRO DE RUTAS
# ==========================================================

app.include_router(auth_router, prefix="/api")
app.include_router(router)
