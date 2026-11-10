
import httpx
from fastapi import HTTPException


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

VISION_URL = "http://127.0.0.1:8001"

TIMEOUT_CONSULTA = 10.0
TIMEOUT_DETECCION = 60.0
TIMEOUT_REINICIO = 30.0


# ==========================================================
# MANEJO DE ERRORES HTTP
# ==========================================================

def obtener_detalle_error(error):
    """Obtiene el detalle enviado por el microservicio."""

    try:
        datos = error.response.json()

        if isinstance(datos, dict):
            return datos.get("detail", error.response.text)

        return error.response.text

    except (ValueError, TypeError):
        return error.response.text or (
            "El microservicio devolvió un error."
        )


def obtener_respuesta_json(respuesta, mensaje):
    """Comprueba que la respuesta del microservicio sea JSON válido."""

    try:
        return respuesta.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail=mensaje,
        )


# ==========================================================
# CONSULTAR OCUPACIÓN
# ==========================================================

def consultar_ocupacion(camara_id=None):

    if camara_id is not None and camara_id not in (
        "interior",
        "exterior",
    ):
        raise HTTPException(
            status_code=400,
            detail="La cámara indicada no es válida.",
        )

    try:
        parametros = {}

        if camara_id:
            parametros["camara_id"] = camara_id

        respuesta = httpx.get(
            f"{VISION_URL}/ocupacion",
            params=parametros,
            timeout=TIMEOUT_CONSULTA,
        )

        respuesta.raise_for_status()

        return obtener_respuesta_json(
            respuesta,
            "El microservicio devolvió una respuesta de ocupación no válida.",
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="El microservicio tardó demasiado en consultar la ocupación.",
        )

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=error.response.status_code,
            detail=(
                "Error al consultar la ocupación: "
                f"{obtener_detalle_error(error)}"
            ),
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="No se pudo conectar con el microservicio de visión.",
        )


# ==========================================================
# DETECTAR PERSONAS Y OBJETOS
# ==========================================================

def detectar_personas(
    imagen,
    camara_id="interior",
    linea_porcentaje=50,
    area_cabina_m2=3.5,
    capacidad=10,
    limite_ocupacion=80.0,
):
    """
    Envía una imagen al microservicio para detectar,
    segmentar y seguir personas y objetos.

    Interior:
        Analiza los elementos presentes en la cabina.

    Exterior:
        Analiza personas, objetos asociados y cruces
        de la línea virtual de acceso.

    El límite de ocupación se transmite al microservicio
    para que la evaluación utilice la configuración actual.
    """

    if camara_id not in ("interior", "exterior"):
        raise HTTPException(
            status_code=400,
            detail="La cámara indicada no es válida.",
        )

    if imagen is None or not hasattr(imagen, "file"):
        raise HTTPException(
            status_code=400,
            detail="No se recibió una imagen válida.",
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

    try:
        area_cabina_m2 = float(area_cabina_m2)
        capacidad = int(capacidad)
        limite_ocupacion = float(limite_ocupacion)
        linea_porcentaje = float(linea_porcentaje)

    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Los parámetros de configuración deben ser numéricos.",
        )

    if area_cabina_m2 <= 0:
        raise HTTPException(
            status_code=400,
            detail="El área de la cabina debe ser mayor que cero.",
        )

    if capacidad < 1:
        raise HTTPException(
            status_code=400,
            detail="La capacidad debe ser como mínimo una persona.",
        )

    if not 1 <= limite_ocupacion <= 100:
        raise HTTPException(
            status_code=400,
            detail="El límite de ocupación debe estar entre 1 y 100.",
        )

    if not 0 <= linea_porcentaje <= 100:
        raise HTTPException(
            status_code=400,
            detail="La posición de la línea debe estar entre 0 y 100.",
        )

    try:
        imagen.file.seek(0)

        # Todos los parámetros que necesita el microservicio.
        parametros = {
            "camara_id": camara_id,
            "area_cabina_m2": area_cabina_m2,
            "capacidad": capacidad,
            "limite_ocupacion": limite_ocupacion,
        }

        if camara_id == "exterior":
            parametros["linea_porcentaje"] = linea_porcentaje

        archivos = {
            "imagen": (
                imagen.filename or "imagen.jpg",
                imagen.file,
                imagen.content_type,
            )
        }

        respuesta = httpx.post(
            f"{VISION_URL}/detectar",
            params=parametros,
            files=archivos,
            timeout=TIMEOUT_DETECCION,
        )

        respuesta.raise_for_status()

        return obtener_respuesta_json(
            respuesta,
            "El microservicio devolvió una respuesta de detección no válida.",
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="El microservicio tardó demasiado en procesar la imagen.",
        )

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=error.response.status_code,
            detail=(
                "El microservicio rechazó la solicitud: "
                f"{obtener_detalle_error(error)}"
            ),
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="No se pudo conectar con el microservicio para realizar la detección.",
        )


# ==========================================================
# REINICIAR SEGUIMIENTO DE UNA CÁMARA
# ==========================================================

def reiniciar_seguimiento(camara_id="interior"):

    if camara_id not in ("interior", "exterior"):
        raise HTTPException(
            status_code=400,
            detail="La cámara indicada no es válida.",
        )

    try:
        respuesta = httpx.post(
            f"{VISION_URL}/reiniciar",
            params={"camara_id": camara_id},
            timeout=TIMEOUT_REINICIO,
        )

        respuesta.raise_for_status()

        return obtener_respuesta_json(
            respuesta,
            "El microservicio devolvió una respuesta de reinicio no válida.",
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="El microservicio tardó demasiado en reiniciar el seguimiento.",
        )

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=error.response.status_code,
            detail=(
                "El microservicio rechazó el reinicio: "
                f"{obtener_detalle_error(error)}"
            ),
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="No se pudo conectar con el microservicio para reiniciar el seguimiento.",
        )


# ==========================================================
# REINICIAR SESIÓN COMPLETA
# ==========================================================

def reiniciar_sesion_vision():
    """
    Reinicia ambas cámaras y el administrador global de IDs
    mediante el endpoint de reinicio de sesión.
    """

    try:
        respuesta = httpx.post(
            f"{VISION_URL}/reiniciar-sesion",
            timeout=TIMEOUT_REINICIO,
        )

        respuesta.raise_for_status()

        return obtener_respuesta_json(
            respuesta,
            "El microservicio devolvió una respuesta de reinicio de sesión no válida.",
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="El microservicio tardó demasiado en reiniciar la sesión completa.",
        )

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=error.response.status_code,
            detail=(
                "El microservicio rechazó el reinicio de sesión: "
                f"{obtener_detalle_error(error)}"
            ),
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="No se pudo conectar con el microservicio para reiniciar la sesión.",
        )
