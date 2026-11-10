
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO

from tracking.tracker import (
    SeguimientoObjetos,
    ADMINISTRADOR_IDS_GLOBAL,
)
from tracking.linea_virtual import LineaVirtual
from tracking.contador_ocupacion import ContadorOcupacion


# ==========================================================
# RUTAS Y CONFIGURACIÓN
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODELO_SEGMENTACION = BASE_DIR / "yolo11n-seg.pt"
TRACKER_CONFIG = BASE_DIR / "config" / "bytetrack.yaml"

CONF_PERSONA_EXTERIOR = 0.15
CONF_INTERIOR = 0.25

IOU = 0.45
IMGSZ = 640

AREA_CABINA_DEFAULT = 3.5
CAPACIDAD_DEFAULT = 10
LIMITE_DEFAULT = 80.0

# Los cálculos experimentales utilizan la proporción de
# píxeles ocupados de la imagen como aproximación del área.
#
# IMPORTANTE:
# No representa una medición física calibrada.
# Cada cámara tiene una perspectiva diferente.

MODO_EXPERIMENTAL = True

estado_camaras = {}

LOCK_REINICIO_SESION = Lock()


# ==========================================================
# INICIALIZACIÓN DE CÁMARAS
# ==========================================================

def crear_estado_camara(camara_id):
    """
    Crea el estado independiente de cada cámara.

    Ambas cámaras comparten el administrador global de IDs,
    pero mantienen trackers independientes.
    """

    return {
        "modelo": YOLO(str(MODELO_SEGMENTACION)),

        "seguimiento": SeguimientoObjetos(
            administrador_ids=ADMINISTRADOR_IDS_GLOBAL
        ),

        "contador": ContadorOcupacion(),

        "lock": Lock(),

        "ultimo_resultado": None,

        "area_cabina_m2": AREA_CABINA_DEFAULT,

        "capacidad": CAPACIDAD_DEFAULT,

        "limite_ocupacion": LIMITE_DEFAULT,

        "linea": None,

        "linea_porcentaje": None,

        "ultimo_ancho": None,

        "ultimo_alto": None,

        "eventos_generados": 0,

        "ultimo_evento_ingreso": None,
    }


# ==========================================================
# LIFESPAN
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("======================================")
    print(" INICIANDO MICROSERVICIO AFOROVISIÓN")
    print("======================================")

    if not MODELO_SEGMENTACION.exists():
        raise RuntimeError(
            f"No se encontró el modelo: {MODELO_SEGMENTACION}"
        )

    if not TRACKER_CONFIG.exists():
        raise RuntimeError(
            f"No se encontró la configuración: {TRACKER_CONFIG}"
        )

    estado_camaras.clear()

    ADMINISTRADOR_IDS_GLOBAL.reiniciar()

    for camara_id in ("interior", "exterior"):

        estado_camaras[camara_id] = crear_estado_camara(
            camara_id
        )

        print(f"Cámara {camara_id} preparada.")

    print("Modelo YOLO11n-seg cargado.")
    print("Administrador global de IDs preparado.")
    print("Modo experimental de ocupación activado.")
    print("Microservicio disponible.")

    yield

    estado_camaras.clear()

    print("Microservicio detenido.")


# ==========================================================
# APLICACIÓN FASTAPI
# ==========================================================

app = FastAPI(
    title="AforoVisión - Microservicio de Visión",
    description=(
        "Detección, segmentación y seguimiento de personas "
        "y objetos para el análisis experimental de ocupación "
        "de ascensores."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# VALIDACIONES
# ==========================================================

def validar_camara(camara_id):

    if camara_id not in ("interior", "exterior"):
        raise HTTPException(
            status_code=400,
            detail="La cámara debe ser interior o exterior.",
        )

    if camara_id not in estado_camaras:
        raise HTTPException(
            status_code=503,
            detail="La cámara todavía no está inicializada.",
        )


def validar_parametros(
    linea_porcentaje,
    area_cabina_m2,
    capacidad,
    limite_ocupacion,
):

    if (
        not np.isfinite(linea_porcentaje)
        or not 0 <= linea_porcentaje <= 100
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "El porcentaje de línea debe estar "
                "entre 0 y 100."
            ),
        )

    if (
        not np.isfinite(area_cabina_m2)
        or area_cabina_m2 <= 0
    ):
        raise HTTPException(
            status_code=400,
            detail="El área de cabina debe ser mayor que cero.",
        )

    if capacidad < 1:
        raise HTTPException(
            status_code=400,
            detail="La capacidad debe ser mayor que cero.",
        )

    if (
        not np.isfinite(limite_ocupacion)
        or not 1 <= limite_ocupacion <= 100
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "El límite de ocupación debe estar "
                "entre 1 y 100."
            ),
        )


# ==========================================================
# FUNCIONES DE GEOMETRÍA
# ==========================================================

def calcular_x_linea(ancho, porcentaje):

    return int(ancho * porcentaje / 100)


def calcular_area_union_mascaras(
    detecciones,
    ancho,
    alto,
):
    """
    Calcula la unión de las máscaras de las detecciones.

    Las regiones superpuestas se contabilizan una sola vez.

    Devuelve:
        Área de unión en píxeles.
        Máscara binaria de la unión.
    """

    union = np.zeros(
        (alto, ancho),
        dtype=np.uint8,
    )

    for deteccion in detecciones:

        poligono = deteccion.get("mascara")

        if poligono is None:
            continue

        puntos = np.asarray(
            poligono,
            dtype=np.float32,
        )

        if puntos.ndim != 2 or len(puntos) < 3:
            continue

        if puntos.shape[1] != 2:
            continue

        puntos[:, 0] = np.clip(
            puntos[:, 0],
            0,
            ancho - 1,
        )

        puntos[:, 1] = np.clip(
            puntos[:, 1],
            0,
            alto - 1,
        )

        puntos = puntos.astype(
            np.int32
        ).reshape((-1, 1, 2))

        cv2.fillPoly(
            union,
            [puntos],
            1,
        )

    area_pixeles = int(
        np.count_nonzero(union)
    )

    return area_pixeles, union


def calcular_porcentaje_imagen(
    area_pixeles,
    ancho,
    alto,
):

    total_pixeles = ancho * alto

    if total_pixeles <= 0:
        return 0.0

    return round(
        area_pixeles / total_pixeles * 100,
        2,
    )


def calcular_superficie_experimental(
    area_pixeles,
    ancho,
    alto,
    area_cabina_m2,
):
    """
    Convierte la proporción de píxeles ocupados en una
    estimación experimental de superficie.

    Fórmula:

        porcentaje = área de máscara / píxeles de imagen

        superficie aproximada =
            porcentaje * área configurada de cabina

    Esta conversión es provisional y NO equivale a una
    medición espacial real.
    """

    porcentaje = calcular_porcentaje_imagen(
        area_pixeles,
        ancho,
        alto,
    )

    superficie = (
        porcentaje / 100
    ) * area_cabina_m2

    superficie = min(
        area_cabina_m2,
        max(0.0, superficie),
    )

    return {
        "porcentaje_aproximado": round(
            porcentaje,
            2,
        ),

        "superficie_aproximada_m2": round(
            superficie,
            4,
        ),
    }


def calcular_disponibilidad_experimental(
    superficie_ocupada,
    area_cabina,
):
    disponible = max(
        0.0,
        area_cabina - superficie_ocupada,
    )

    return round(disponible, 4)


# ==========================================================
# OCUPACIÓN ACTUAL DE LA CÁMARA INTERIOR
# ==========================================================

def obtener_ultima_ocupacion_interior():
    """
    Recupera la última medición de la cámara interior.

    La cámara exterior utiliza esta información como
    referencia para proyectar un nuevo ingreso.
    """

    if "interior" not in estado_camaras:

        return {
            "disponible": False,
            "motivo": (
                "La cámara interior no está inicializada."
            ),
        }

    camara_interior = estado_camaras["interior"]

    with camara_interior["lock"]:

        resultado = camara_interior["ultimo_resultado"]

        if resultado is None:

            return {
                "disponible": False,
                "motivo": (
                    "La cámara interior todavía no ha "
                    "procesado ninguna imagen."
                ),
            }

        return {
            "disponible": True,

            "camara_id": "interior",

            "personas_detectadas": resultado.get(
                "personas_detectadas",
                0,
            ),

            "objetos_detectados": resultado.get(
                "objetos_detectados",
                0,
            ),

            "area_union_pixeles": resultado.get(
                "area_union_pixeles",
                0,
            ),

            "porcentaje_imagen_ocupada": resultado.get(
                "porcentaje_imagen_ocupada",
                0.0,
            ),

            "superficie_ocupada_m2": resultado.get(
                "superficie_ocupada_m2"
            ),

            "superficie_disponible_m2": resultado.get(
                "superficie_disponible_m2"
            ),

            "porcentaje_ocupacion_espacial": resultado.get(
                "porcentaje_ocupacion_espacial"
            ),

            "porcentaje_ocupacion_personas": resultado.get(
                "porcentaje_ocupacion_personas",
                0.0,
            ),

            "area_cabina_m2": resultado.get(
                "area_cabina_m2",
                AREA_CABINA_DEFAULT,
            ),

            "capacidad": resultado.get(
                "capacidad",
                CAPACIDAD_DEFAULT,
            ),

            "modo_calculo": resultado.get(
                "modo_calculo",
                "experimental",
            ),

            "requiere_calibracion": True,
        }


# ==========================================================
# APORTACIÓN ESPACIAL DE UNA PERSONA Y SUS OBJETOS
# ==========================================================

def obtener_detecciones_persona(
    detecciones,
    id_persona,
):
    """
    Selecciona la persona que cruzó y sus objetos asociados.

    La asociación depende del campo transportado_por,
    generado por SeguimientoObjetos.

    No incorpora objetos de otras personas.
    """

    seleccionadas = []

    for deteccion in detecciones:

        if (
            deteccion.get("tipo") == "persona"
            and deteccion.get("id") == id_persona
        ):

            seleccionadas.append(deteccion)
            continue

        if (
            deteccion.get("tipo") == "objeto"
            and deteccion.get("transportado_por") == id_persona
        ):

            seleccionadas.append(deteccion)

    return seleccionadas


def calcular_aportacion_ingreso(
    detecciones,
    id_persona,
    ancho,
    alto,
    area_cabina_m2,
):
    """
    Calcula la unión de las máscaras de la persona que cruza
    y de sus objetos asociados.

    La proporción de imagen exterior se utiliza como
    estimación experimental de la aportación espacial.

    No es una medición física calibrada.
    """

    detecciones_persona = obtener_detecciones_persona(
        detecciones,
        id_persona,
    )

    if not detecciones_persona:

        return {
            "id_persona": id_persona,

            "detecciones_asociadas": [],

            "cantidad_objetos_asociados": 0,

            "area_union_pixeles": 0,

            "porcentaje_imagen": 0.0,

            "superficie_adicional_m2": None,

            "superficie_ingreso_m2_aprox": None,

            "calculo_disponible": False,

            "requiere_calibracion": True,

            "motivo": (
                "No se encontraron máscaras de la persona "
                "en el fotograma del cruce."
            ),
        }

    area_pixeles, _ = calcular_area_union_mascaras(
        detecciones_persona,
        ancho,
        alto,
    )

    if area_pixeles <= 0:

        return {
            "id_persona": id_persona,

            "detecciones_asociadas": detecciones_persona,

            "cantidad_objetos_asociados": 0,

            "area_union_pixeles": 0,

            "porcentaje_imagen": 0.0,

            "superficie_adicional_m2": None,

            "superficie_ingreso_m2_aprox": None,

            "calculo_disponible": False,

            "requiere_calibracion": True,

            "motivo": (
                "Las detecciones no contienen máscaras "
                "válidas para calcular la aportación."
            ),
        }

    estimacion = calcular_superficie_experimental(
        area_pixeles,
        ancho,
        alto,
        area_cabina_m2,
    )

    objetos_asociados = [
        deteccion
        for deteccion in detecciones_persona
        if deteccion.get("tipo") == "objeto"
    ]

    superficie = estimacion[
        "superficie_aproximada_m2"
    ]

    return {
        "id_persona": id_persona,

        "detecciones_asociadas": detecciones_persona,

        "cantidad_objetos_asociados": len(
            objetos_asociados
        ),

        "area_union_pixeles": area_pixeles,

        "porcentaje_imagen": (
            estimacion["porcentaje_aproximado"]
        ),

        "superficie_adicional_m2": superficie,

        "superficie_ingreso_m2_aprox": superficie,

        "calculo_disponible": True,

        "modo_calculo": "experimental",

        "requiere_calibracion": True,

        "motivo": (
            "Aportación estimada mediante la proporción "
            "de la máscara de la persona y sus objetos "
            "asociados respecto de la imagen exterior. "
            "No representa una medición física calibrada."
        ),
    }


# ==========================================================
# PROYECCIÓN DE OCUPACIÓN
# ==========================================================

def calcular_proyeccion_ingreso(
    ocupacion_interior,
    aportacion_ingreso,
    capacidad,
    limite_ocupacion,
):
    """
    Calcula la ocupación proyectada después de un nuevo ingreso.

    La decisión es experimental y utiliza los valores
    aproximados disponibles.
    """

    if not ocupacion_interior.get("disponible"):

        return {
            "evaluacion_disponible": False,

            "decision_automatica_disponible": False,

            "permitir_ingreso": None,

            "motivo_decision": (
                "Primero se necesita una medición de la "
                "cámara interior."
            ),
        }

    if not aportacion_ingreso.get("calculo_disponible"):

        return {
            "evaluacion_disponible": False,

            "decision_automatica_disponible": False,

            "permitir_ingreso": None,

            "motivo_decision": (
                "No se pudo calcular la aportación espacial "
                "del nuevo ingreso."
            ),
        }

    superficie_actual = float(
        ocupacion_interior.get(
            "superficie_ocupada_m2",
            0.0,
        ) or 0.0
    )

    superficie_ingreso = float(
        aportacion_ingreso.get(
            "superficie_ingreso_m2_aprox",
            0.0,
        ) or 0.0
    )

    area_cabina = float(
        ocupacion_interior.get(
            "area_cabina_m2",
            AREA_CABINA_DEFAULT,
        )
    )

    personas_actuales = int(
        ocupacion_interior.get(
            "personas_detectadas",
            0,
        )
    )

    personas_proyectadas = personas_actuales + 1

    superficie_proyectada = (
        superficie_actual + superficie_ingreso
    )

    porcentaje_actual = (
        superficie_actual / area_cabina
    ) * 100

    porcentaje_ingreso = (
        superficie_ingreso / area_cabina
    ) * 100

    porcentaje_proyectado = (
        superficie_proyectada / area_cabina
    ) * 100

    superficie_disponible = max(
        0.0,
        area_cabina - superficie_proyectada,
    )

    cumple_limite = (
        porcentaje_proyectado <= limite_ocupacion
    )

    cumple_capacidad = (
        personas_proyectadas <= capacidad
    )

    permitir_ingreso = (
        cumple_limite and cumple_capacidad
    )

    if permitir_ingreso:

        motivo = (
            "El ingreso cumple el límite experimental "
            "de ocupación y la capacidad configurada."
        )

    elif not cumple_capacidad:

        motivo = (
            "Ingreso no permitido: se superaría la "
            "capacidad configurada de personas."
        )

    else:

        motivo = (
            "Ingreso no permitido: la ocupación proyectada "
            "superaría el límite configurado."
        )

    return {
        "evaluacion_disponible": True,

        "decision_automatica_disponible": True,

        "modo_calculo": "experimental",

        "requiere_calibracion": True,

        "superficie_actual_m2_aprox": round(
            superficie_actual,
            4,
        ),

        "superficie_ingreso_m2_aprox": round(
            superficie_ingreso,
            4,
        ),

        "superficie_proyectada_m2_aprox": round(
            superficie_proyectada,
            4,
        ),

        "superficie_disponible_proyectada_m2_aprox": round(
            superficie_disponible,
            4,
        ),

        "porcentaje_actual_aprox": round(
            porcentaje_actual,
            2,
        ),

        "porcentaje_ingreso_aprox": round(
            porcentaje_ingreso,
            2,
        ),

        "porcentaje_proyectado_aprox": round(
            porcentaje_proyectado,
            2,
        ),

        "personas_actuales": personas_actuales,

        "personas_proyectadas": personas_proyectadas,

        "capacidad": capacidad,

        "limite_ocupacion": limite_ocupacion,

        "cumple_limite": cumple_limite,

        "cumple_capacidad": cumple_capacidad,

        "permitir_ingreso": permitir_ingreso,

        "decision": (
            "INGRESO PERMITIDO"
            if permitir_ingreso
            else "INGRESO NO PERMITIDO"
        ),

        "motivo_decision": motivo,
    }


# ==========================================================
# IDENTIFICADOR ÚNICO DE EVENTO
# ==========================================================

def crear_evento_cruce(camara, cruce):
    """
    Crea un identificador único por cruce.

    Una misma persona puede volver a intentar ingresar.
    Por eso no se deduplican eventos únicamente por persona.
    """

    camara["eventos_generados"] += 1

    return {
        "evento_id": uuid4().hex,

        "numero_evento": camara["eventos_generados"],

        "id_persona": cruce["id"],

        "direccion": cruce["direccion"],

        "tipo": cruce["tipo"],
    }


# ==========================================================
# PROCESAMIENTO DE LA LÍNEA VIRTUAL
# ==========================================================

def obtener_cruces(
    camara,
    detecciones,
):
    """
    Actualiza la presencia de personas y detecta cruces.

    Solo las personas activan la línea virtual.
    Los objetos asociados no generan cruces.
    """

    cruces = []

    linea = camara["linea"]

    if linea is None:
        return cruces

    linea.actualizar_fotograma(detecciones)

    for persona in detecciones:

        if persona.get("tipo") != "persona":
            continue

        id_persona = persona.get("id")

        if id_persona is None:
            continue

        direccion = linea.detectar_cruce(persona)

        if direccion is None:
            continue

        tipo = camara["contador"].registrar_cruce(
            direccion,
            id_persona=id_persona,
        )

        if tipo is None:
            continue

        cruce = {
            "id": id_persona,
            "direccion": direccion,
            "tipo": tipo,
        }

        cruce.update(
            crear_evento_cruce(
                camara,
                cruce,
            )
        )

        cruces.append(cruce)

    return cruces


# ==========================================================
# REINICIAR ESTADO DE UNA CÁMARA
# ==========================================================

def limpiar_estado_camara(
    camara,
    modelo_nuevo,
):
    """
    Reinicia el estado interno de una cámara.

    No modifica el administrador global de IDs.
    """

    camara["modelo"] = modelo_nuevo

    camara["seguimiento"].reiniciar()

    camara["contador"].reiniciar()

    camara["linea"] = None

    camara["linea_porcentaje"] = None

    camara["ultimo_resultado"] = None

    camara["ultimo_ancho"] = None

    camara["ultimo_alto"] = None

    camara["eventos_generados"] = 0

    camara["ultimo_evento_ingreso"] = None


# ==========================================================
# REINICIAR UNA SOLA CÁMARA
# ==========================================================

@app.post("/reiniciar")
def reiniciar_seguimiento(
    camara_id: str = "interior",
):
    """
    Reinicia solamente la cámara solicitada.

    El administrador global de IDs se conserva.
    """

    validar_camara(camara_id)

    camara = estado_camaras[camara_id]

    with camara["lock"]:

        try:

            modelo_nuevo = YOLO(
                str(MODELO_SEGMENTACION)
            )

            limpiar_estado_camara(
                camara,
                modelo_nuevo,
            )

            return {
                "estado": "reiniciado",

                "camara_id": camara_id,

                "ids_globales_reiniciados": False,

                "mensaje": (
                    "El seguimiento de esta cámara se reinició. "
                    "El administrador global de IDs se conserva."
                ),
            }

        except Exception as error:

            print(
                f"Error reiniciando cámara {camara_id}: {error}"
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "No se pudo reiniciar el seguimiento."
                ),
            )


# ==========================================================
# REINICIAR SESIÓN COMPLETA
# ==========================================================

@app.post("/reiniciar-sesion")
def reiniciar_sesion_completa():
    """
    Reinicia ambas cámaras como una sesión nueva.

    Reinicia el administrador global de IDs una sola vez.
    """

    with LOCK_REINICIO_SESION:

        validar_camara("interior")
        validar_camara("exterior")

        interior = estado_camaras["interior"]
        exterior = estado_camaras["exterior"]

        try:

            modelo_interior = YOLO(
                str(MODELO_SEGMENTACION)
            )

            modelo_exterior = YOLO(
                str(MODELO_SEGMENTACION)
            )

            with interior["lock"], exterior["lock"]:

                limpiar_estado_camara(
                    interior,
                    modelo_interior,
                )

                limpiar_estado_camara(
                    exterior,
                    modelo_exterior,
                )

                ADMINISTRADOR_IDS_GLOBAL.reiniciar()

            return {
                "estado": "reiniciado",

                "sesion": "nueva",

                "camaras": [
                    "interior",
                    "exterior",
                ],

                "ids_globales_reiniciados": True,

                "siguiente_id_global": 1,

                "mensaje": (
                    "La sesión completa se reinició. "
                    "Ambas cámaras están preparadas."
                ),
            }

        except Exception as error:

            print(
                f"Error reiniciando sesión completa: {error}"
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "No se pudo reiniciar la sesión completa."
                ),
            )


# ==========================================================
# ENDPOINT INICIAL
# ==========================================================

@app.get("/")
def inicio():

    return {
        "servicio": "AforoVisión - Microservicio de Visión",
        "version": "1.2.0",
        "estado": "disponible",
        "modelo": "yolo11n-seg.pt",
        "modo_calculo": "experimental",
    }


# ==========================================================
# ESTADO DEL MICROSERVICIO
# ==========================================================

@app.get("/health")
def health():

    return {
        "servicio": "vision",

        "estado": "disponible",

        "modelo": "yolo11n-seg.pt",

        "modo_calculo": "experimental",

        "camaras": list(estado_camaras.keys()),
    }


# ==========================================================
# DETECTAR Y SEGMENTAR IMAGEN
# ==========================================================

@app.post("/detectar")
def detectar(
    imagen: UploadFile = File(...),
    camara_id: str = "interior",
    linea_porcentaje: float = 50,
    area_cabina_m2: float = AREA_CABINA_DEFAULT,
    capacidad: int = CAPACIDAD_DEFAULT,
    limite_ocupacion: float = LIMITE_DEFAULT,
):

    validar_camara(camara_id)

    validar_parametros(
        linea_porcentaje,
        area_cabina_m2,
        capacidad,
        limite_ocupacion,
    )

    if imagen.content_type not in (
        "image/jpeg",
        "image/png",
        "image/jpg",
    ):
        raise HTTPException(
            status_code=415,
            detail="Formato no válido. Utiliza JPG o PNG.",
        )

    contenido = imagen.file.read()

    if not contenido:

        raise HTTPException(
            status_code=400,
            detail="La imagen está vacía.",
        )

    buffer = np.frombuffer(
        contenido,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        buffer,
        cv2.IMREAD_COLOR,
    )

    if frame is None:

        raise HTTPException(
            status_code=400,
            detail="No se pudo interpretar la imagen.",
        )

    alto, ancho = frame.shape[:2]

    camara = estado_camaras[camara_id]

    inicio_procesamiento = perf_counter()

    with camara["lock"]:

        try:

            confianza = (
                CONF_PERSONA_EXTERIOR
                if camara_id == "exterior"
                else CONF_INTERIOR
            )

            # ==============================================
            # ACTUALIZAR CONFIGURACIÓN
            # ==============================================

            camara["area_cabina_m2"] = area_cabina_m2
            camara["capacidad"] = capacidad
            camara["limite_ocupacion"] = limite_ocupacion

            # ==============================================
            # YOLO11N-SEG
            # ==============================================

            resultados = camara["modelo"].track(
                source=frame,
                tracker=str(TRACKER_CONFIG),
                persist=True,
                classes=None,
                conf=confianza,
                iou=IOU,
                imgsz=IMGSZ,
                verbose=False,
            )

            # ==============================================
            # PROCESAR TRACKING
            # ==============================================

            detecciones = (
                camara["seguimiento"].procesar_resultados(
                    resultados,
                    camara_id=camara_id,
                )
            )

            personas = [
                deteccion
                for deteccion in detecciones
                if deteccion.get("tipo") == "persona"
            ]

            objetos = [
                deteccion
                for deteccion in detecciones
                if deteccion.get("tipo") == "objeto"
            ]

            camara["ultimo_ancho"] = ancho
            camara["ultimo_alto"] = alto

            # ==============================================
            # CÁMARA EXTERIOR
            # ==============================================

            if camara_id == "exterior":

                x_linea = calcular_x_linea(
                    ancho,
                    linea_porcentaje,
                )

                if (
                    camara["linea"] is None
                    or camara["linea_porcentaje"]
                    != linea_porcentaje
                ):

                    camara["linea"] = LineaVirtual(
                        x_linea=x_linea,

                        margen=max(
                            4,
                            int(ancho * 0.01),
                        ),

                        fotogramas_confirmacion=2,

                        fotogramas_perdida=30,
                    )

                    camara["linea_porcentaje"] = (
                        linea_porcentaje
                    )

                else:

                    camara["linea"].x_linea = x_linea

                # ==========================================
                # CRUCES CONFIRMADOS
                # ==========================================

                cruces = obtener_cruces(
                    camara,
                    detecciones,
                )

                # ==========================================
                # OCUPACIÓN INTERIOR
                # ==========================================

                ocupacion_interior = (
                    obtener_ultima_ocupacion_interior()
                )

                # ==========================================
                # EVALUACIONES DE INGRESO
                # ==========================================

                evaluaciones_ingreso = []

                for cruce in cruces:

                    # Una salida no representa un nuevo ingreso.

                    if cruce["tipo"] != "entrada":
                        continue

                    aportacion = calcular_aportacion_ingreso(
                        detecciones=detecciones,
                        id_persona=cruce["id"],
                        ancho=ancho,
                        alto=alto,
                        area_cabina_m2=area_cabina_m2,
                    )

                    proyeccion = calcular_proyeccion_ingreso(
                        ocupacion_interior=ocupacion_interior,
                        aportacion_ingreso=aportacion,
                        capacidad=capacidad,
                        limite_ocupacion=limite_ocupacion,
                    )

                    evaluacion = {
                        **aportacion,

                        **proyeccion,

                        "evento_id": cruce["evento_id"],

                        "numero_evento": cruce["numero_evento"],

                        "id_persona": cruce["id_persona"],

                        "direccion": cruce["direccion"],

                        "tipo_evento": cruce["tipo"],

                        "ocupacion_actual_interior": (
                            ocupacion_interior
                        ),
                    }

                    evaluaciones_ingreso.append(
                        evaluacion
                    )

                    camara["ultimo_evento_ingreso"] = (
                        evaluacion
                    )

                # ==========================================
                # CONTADOR
                # ==========================================

                estado_contador = (
                    camara["contador"].obtener_estado()
                )

                tiempo_ms = round(
                    (
                        perf_counter()
                        - inicio_procesamiento
                    ) * 1000,
                    2,
                )

                # ==========================================
                # DATOS DEL ÚLTIMO INGRESO
                # ==========================================

                ultimo_ingreso = (
                    camara["ultimo_evento_ingreso"]
                )

                # ==========================================
                # RESPUESTA EXTERIOR
                # ==========================================

                respuesta = {

                    "camara_id": "exterior",

                    "ancho_imagen": ancho,

                    "alto_imagen": alto,

                    "cantidad_detectada": len(personas),

                    "personas_detectadas": len(personas),

                    "objetos_detectados": len(objetos),

                    "objetos": detecciones,

                    "cruces_detectados": cruces,

                    "evaluaciones_ingreso": (
                        evaluaciones_ingreso
                    ),

                    "ultimo_ingreso": ultimo_ingreso,

                    "linea_virtual": {
                        "x": x_linea,
                        "porcentaje": linea_porcentaje,
                    },

                    "ocupacion": estado_contador,

                    "ingresos_registrados": (
                        estado_contador.get("ingresos", 0)
                        if isinstance(estado_contador, dict)
                        else None
                    ),

                    "salidas_registradas": (
                        estado_contador.get("salidas", 0)
                        if isinstance(estado_contador, dict)
                        else None
                    ),

                    "ocupacion_actual_interior": (
                        ocupacion_interior
                    ),

                    "superficie_ingreso_m2_aprox": (
                        ultimo_ingreso.get(
                            "superficie_ingreso_m2_aprox"
                        )
                        if ultimo_ingreso
                        else None
                    ),

                    "superficie_proyectada_m2_aprox": (
                        ultimo_ingreso.get(
                            "superficie_proyectada_m2_aprox"
                        )
                        if ultimo_ingreso
                        else None
                    ),

                    "porcentaje_proyectado_aprox": (
                        ultimo_ingreso.get(
                            "porcentaje_proyectado_aprox"
                        )
                        if ultimo_ingreso
                        else None
                    ),

                    "decision_automatica_disponible": (
                        ultimo_ingreso.get(
                            "decision_automatica_disponible",
                            False,
                        )
                        if ultimo_ingreso
                        else False
                    ),

                    "permitir_ingreso": (
                        ultimo_ingreso.get(
                            "permitir_ingreso"
                        )
                        if ultimo_ingreso
                        else None
                    ),

                    "decision": (
                        ultimo_ingreso.get("decision")
                        if ultimo_ingreso
                        else None
                    ),

                    "modo_calculo": "experimental",

                    "requiere_calibracion": True,

                    "motivo_evaluacion": (
                        "La aportación y proyección son "
                        "estimaciones experimentales basadas "
                        "en máscaras de segmentación."
                    ),

                    "tiempo_procesamiento_ms": tiempo_ms,
                }

                camara["ultimo_resultado"] = respuesta

                return respuesta

            # ==============================================
            # CÁMARA INTERIOR
            # ==============================================

            area_pixeles, mascara_union = (
                calcular_area_union_mascaras(
                    detecciones,
                    ancho,
                    alto,
                )
            )

            estimacion = calcular_superficie_experimental(
                area_pixeles=area_pixeles,
                ancho=ancho,
                alto=alto,
                area_cabina_m2=area_cabina_m2,
            )

            porcentaje_imagen = (
                estimacion["porcentaje_aproximado"]
            )

            superficie_ocupada = (
                estimacion["superficie_aproximada_m2"]
            )

            superficie_disponible = (
                calcular_disponibilidad_experimental(
                    superficie_ocupada,
                    area_cabina_m2,
                )
            )

            porcentaje_ocupacion_espacial = round(
                (
                    superficie_ocupada
                    / area_cabina_m2
                ) * 100,
                2,
            )

            # Indicador separado de cantidad de personas.
            # No sustituye el cálculo espacial.

            porcentaje_ocupacion_personas = round(
                min(
                    100,
                    (len(personas) / capacidad) * 100,
                ),
                2,
            )

            tiempo_ms = round(
                (
                    perf_counter()
                    - inicio_procesamiento
                ) * 1000,
                2,
            )

            # ==============================================
            # RESPUESTA INTERIOR
            # ==============================================

            respuesta = {

                "camara_id": "interior",

                "ancho_imagen": ancho,

                "alto_imagen": alto,

                "cantidad_detectada": len(personas),

                "personas_detectadas": len(personas),

                "objetos_detectados": len(objetos),

                "objetos": detecciones,

                "area_cabina_m2": area_cabina_m2,

                "capacidad": capacidad,

                "area_union_pixeles": area_pixeles,

                "porcentaje_imagen_ocupada": (
                    porcentaje_imagen
                ),

                "superficie_ocupada_m2": (
                    superficie_ocupada
                ),

                "espacio_ocupado_m2": (
                    superficie_ocupada
                ),

                "superficie_disponible_m2": (
                    superficie_disponible
                ),

                "espacio_disponible_m2": (
                    superficie_disponible
                ),

                "porcentaje_ocupacion_espacial": (
                    porcentaje_ocupacion_espacial
                ),

                "ocupacion_actual": (
                    porcentaje_ocupacion_espacial
                ),

                "porcentaje_ocupacion": (
                    porcentaje_ocupacion_espacial
                ),

                "porcentaje_ocupacion_personas": (
                    porcentaje_ocupacion_personas
                ),

                "ocupacion": (
                    porcentaje_ocupacion_espacial
                ),

                "mascara_disponible": (
                    area_pixeles > 0
                ),

                "modo_calculo": "experimental",

                "requiere_calibracion": True,

                "motivo_calibracion": (
                    "La superficie se estima utilizando la "
                    "proporción de píxeles ocupados respecto "
                    "de la imagen y el área configurada de "
                    "la cabina. Es una aproximación experimental."
                ),

                "tiempo_procesamiento_ms": tiempo_ms,
            }

            camara["ultimo_resultado"] = respuesta

            return respuesta

        except HTTPException:
            raise

        except Exception as error:

            print(
                f"ERROR procesando cámara {camara_id}: {error}"
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Ocurrió un error durante el análisis "
                    "de la imagen."
                ),
            )


# ==========================================================
# CONSULTAR OCUPACIÓN
# ==========================================================

@app.get("/ocupacion")
def consultar_ocupacion(
    camara_id: str = "exterior",
):

    validar_camara(camara_id)

    camara = estado_camaras[camara_id]

    with camara["lock"]:

        if camara_id == "exterior":

            estado_contador = (
                camara["contador"].obtener_estado()
            )

            return {
                "camara_id": "exterior",

                **estado_contador,

                "ultimo_resultado": (
                    camara["ultimo_resultado"]
                ),

                "ultimo_ingreso": (
                    camara["ultimo_evento_ingreso"]
                ),
            }

        return {
            "camara_id": "interior",

            "ultimo_resultado": (
                camara["ultimo_resultado"]
            ),

            "area_cabina_m2": (
                camara["area_cabina_m2"]
            ),

            "capacidad": camara["capacidad"],

            "modo_calculo": "experimental",
        }
