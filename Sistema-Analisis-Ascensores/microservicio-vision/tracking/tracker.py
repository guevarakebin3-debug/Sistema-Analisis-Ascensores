
import threading

import cv2
import numpy as np


class AdministradorIDsGlobales:
    """
    Genera identificadores únicos compartidos entre las cámaras.

    Los trackers de cada cámara siguen siendo independientes,
    pero todos solicitan sus identificadores a este administrador.
    """

    def __init__(self, primer_id=1):
        self.siguiente_id = primer_id
        self.lock = threading.Lock()

    def obtener_nuevo_id(self):
        with self.lock:
            identificador = self.siguiente_id
            self.siguiente_id += 1
            return identificador

    def reiniciar(self):
        """
        Reinicia el contador global.

        IMPORTANTE:
        Debe utilizarse únicamente cuando se inicia una sesión
        completamente nueva y se reinician ambas cámaras.
        """
        with self.lock:
            self.siguiente_id = 1


# Instancia compartida por todos los SeguimientoObjetos
# que utilicen este módulo dentro del mismo proceso.
ADMINISTRADOR_IDS_GLOBAL = AdministradorIDsGlobales()


class SeguimientoObjetos:
    """
    Procesa detecciones, identificadores y máscaras de YOLO11n-seg.

    CÁMARA INTERIOR:
        Detecta personas y objetos para estimar el espacio ocupado.

    CÁMARA EXTERIOR:
        Detecta personas y objetos para estimar el espacio adicional
        de la persona que intenta ingresar.

    Los trackers ByteTrack son independientes por cámara.
    Los identificadores visibles son globales y no se repiten.

    Los objetos se asocian a una persona utilizando:
        - Superposición de cajas.
        - Posición relativa.
        - Distancia normalizada.
        - Continuidad temporal de la asociación.

    La asociación es una estimación visual, no una prueba definitiva
    de que una persona transporte un objeto.
    """

    def __init__(
        self,
        area_minima_objeto=0.0005,
        margen_asociacion=0.15,
        fotogramas_perdida=30,
        administrador_ids=None,
    ):

        self.area_minima_objeto = area_minima_objeto
        self.margen_asociacion = margen_asociacion
        self.fotogramas_perdida = fotogramas_perdida

        self.numero_fotograma = 0

        # Administrador compartido entre cámaras.
        self.administrador_ids = (
            administrador_ids
            if administrador_ids is not None
            else ADMINISTRADOR_IDS_GLOBAL
        )

        # ID local de ByteTrack -> ID global.
        self.ids_globales = {}

        # Último fotograma en que apareció cada ID local.
        self.ultimos_fotogramas = {}

        # IDs globales observados durante el seguimiento.
        self.ids_detectados = set()

        # Historial de asociación objeto -> persona.
        self.asociaciones_objetos = {}

    def reiniciar(self):
        """
        Reinicia únicamente el seguimiento de esta cámara.

        No reinicia el contador global porque la otra cámara
        podría continuar utilizando identificadores.
        """

        self.numero_fotograma = 0

        self.ids_globales.clear()
        self.ultimos_fotogramas.clear()
        self.ids_detectados.clear()
        self.asociaciones_objetos.clear()

    def obtener_id_global(self, id_local):
        """
        Convierte el identificador local de ByteTrack
        en un identificador global compartido.
        """

        if id_local is None:
            return None

        id_local = int(id_local)

        if id_local not in self.ids_globales:

            id_global = self.administrador_ids.obtener_nuevo_id()

            self.ids_globales[id_local] = id_global

        return self.ids_globales[id_local]

    def limpiar_ids_inactivos(self):
        """
        Elimina identificadores locales que llevan demasiados
        fotogramas sin aparecer.

        El administrador global nunca reutiliza los IDs eliminados.
        """

        ids_inactivos = [
            id_local
            for id_local, ultimo in self.ultimos_fotogramas.items()
            if self.numero_fotograma - ultimo > self.fotogramas_perdida
        ]

        for id_local in ids_inactivos:

            id_global = self.ids_globales.pop(id_local, None)

            self.ultimos_fotogramas.pop(id_local, None)

            if id_global is not None:
                self.ids_detectados.discard(id_global)

        # Limpiar asociaciones antiguas.
        asociaciones_antiguas = [
            id_objeto
            for id_objeto, asociacion in self.asociaciones_objetos.items()
            if (
                self.numero_fotograma
                - asociacion["ultimo_fotograma"]
                > self.fotogramas_perdida
            )
        ]

        for id_objeto in asociaciones_antiguas:
            self.asociaciones_objetos.pop(id_objeto, None)

    def procesar_resultados(
        self,
        resultados,
        camara_id="interior",
    ):

        self.numero_fotograma += 1

        detecciones = []

        if not resultados:
            self.limpiar_ids_inactivos()
            return detecciones

        for resultado in resultados:

            if resultado.boxes is None:
                continue

            alto, ancho = resultado.orig_shape

            cajas = resultado.boxes
            mascaras = resultado.masks

            for indice, caja in enumerate(cajas):

                clase_id = int(caja.cls[0].item())
                nombre_clase = resultado.names[clase_id]

                es_persona = nombre_clase == "person"

                x1, y1, x2, y2 = map(
                    int,
                    caja.xyxy[0].cpu().tolist(),
                )

                x1 = max(0, min(x1, ancho))
                x2 = max(0, min(x2, ancho))
                y1 = max(0, min(y1, alto))
                y2 = max(0, min(y2, alto))

                ancho_caja = x2 - x1
                alto_caja = y2 - y1

                if ancho_caja <= 0 or alto_caja <= 0:
                    continue

                area_caja = ancho_caja * alto_caja

                proporcion_caja = (
                    area_caja / (ancho * alto)
                    if ancho > 0 and alto > 0
                    else 0
                )

                # Evitar ruido de detecciones diminutas.
                if (
                    not es_persona
                    and proporcion_caja < self.area_minima_objeto
                ):
                    continue

                id_local = None

                if caja.id is not None:
                    id_local = int(caja.id[0].item())

                # Obtener ID global compartido.
                id_global = self.obtener_id_global(id_local)

                ya_detectado = False

                if id_local is not None:

                    ya_detectado = (
                        id_local in self.ultimos_fotogramas
                    )

                    self.ultimos_fotogramas[id_local] = (
                        self.numero_fotograma
                    )

                    if id_global is not None:
                        self.ids_detectados.add(id_global)

                confianza = round(
                    float(caja.conf[0].item()),
                    3,
                )

                centro_x = (x1 + x2) // 2
                centro_y = (y1 + y2) // 2

                poligono = []
                area_mascara_px = 0.0

                # Obtener la máscara correspondiente.
                if (
                    mascaras is not None
                    and mascaras.xy is not None
                    and indice < len(mascaras.xy)
                ):

                    puntos = mascaras.xy[indice]

                    if puntos is not None and len(puntos) >= 3:

                        contorno = np.asarray(
                            puntos,
                            dtype=np.float32,
                        )

                        perimetro = cv2.arcLength(
                            contorno.reshape(-1, 1, 2),
                            True,
                        )

                        aproximacion = cv2.approxPolyDP(
                            contorno.reshape(-1, 1, 2),
                            max(1.0, 0.002 * perimetro),
                            True,
                        )

                        poligono = [
                            [
                                round(float(p[0]), 1),
                                round(float(p[1]), 1),
                            ]
                            for p in aproximacion.reshape(-1, 2)
                        ]

                        area_mascara_px = round(
                            abs(cv2.contourArea(contorno)),
                            2,
                        )

                detecciones.append({

                    # Identificador compartido entre cámaras.
                    "id": id_global,

                    # Se conserva para depuración interna.
                    "id_local": id_local,

                    # Clase interna del modelo.
                    "clase": nombre_clase,

                    "tipo": (
                        "persona"
                        if es_persona
                        else "objeto"
                    ),

                    "camara_id": camara_id,

                    "confianza": confianza,
                    "ya_detectado": ya_detectado,

                    "posicion": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "centro_x": centro_x,
                        "centro_y": centro_y,
                        "ancho": ancho_caja,
                        "alto": alto_caja,
                    },

                    "mascara": poligono,
                    "area_mascara_px": area_mascara_px,

                    "transportado_por": None,
                })

        self.limpiar_ids_inactivos()

        self.asociar_objetos_a_personas(detecciones)

        return detecciones

    def calcular_relacion_objeto_persona(
        self,
        objeto,
        persona,
    ):
        """
        Calcula la relación espacial entre un objeto y una persona.

        Una asociación se considera posible si:
            - Existe superposición suficiente, o
            - El objeto está muy próximo al cuerpo y su posición
              relativa es compatible con una asociación.

        La cercanía por sí sola no es suficiente.
        """

        caja_objeto = objeto["posicion"]
        caja_persona = persona["posicion"]

        ix1 = max(
            caja_objeto["x1"],
            caja_persona["x1"],
        )

        iy1 = max(
            caja_objeto["y1"],
            caja_persona["y1"],
        )

        ix2 = min(
            caja_objeto["x2"],
            caja_persona["x2"],
        )

        iy2 = min(
            caja_objeto["y2"],
            caja_persona["y2"],
        )

        interseccion = (
            max(0, ix2 - ix1)
            * max(0, iy2 - iy1)
        )

        area_objeto = max(
            1,
            caja_objeto["ancho"] * caja_objeto["alto"],
        )

        proporcion_superpuesta = (
            interseccion / area_objeto
        )

        centro_x = caja_objeto["centro_x"]
        centro_y = caja_objeto["centro_y"]

        alto_persona = max(1, caja_persona["alto"])
        ancho_persona = max(1, caja_persona["ancho"])

        # Región ampliada alrededor de la persona.
        margen_x = max(
            5,
            int(ancho_persona * self.margen_asociacion),
        )

        margen_y = max(
            5,
            int(alto_persona * self.margen_asociacion),
        )

        dentro_region = (
            caja_persona["x1"] - margen_x
            <= centro_x
            <= caja_persona["x2"] + margen_x
            and
            caja_persona["y1"] - margen_y
            <= centro_y
            <= caja_persona["y2"] + margen_y
        )

        distancia_x = (
            abs(centro_x - caja_persona["centro_x"])
            / ancho_persona
        )

        distancia_y = (
            abs(centro_y - caja_persona["centro_y"])
            / alto_persona
        )

        distancia_normalizada = (
            (distancia_x ** 2 + distancia_y ** 2) ** 0.5
        )

        # Una superposición clara es una señal fuerte.
        if proporcion_superpuesta >= 0.20:

            puntuacion = min(
                1.0,
                0.65 + proporcion_superpuesta,
            )

            return puntuacion

        # Si no existe superposición, exigir proximidad real.
        if not dentro_region:
            return 0.0

        # No asociar objetos situados muy lejos del cuerpo.
        if distancia_normalizada > 0.65:
            return 0.0

        # Evitar asociar objetos muy alejados verticalmente.
        diferencia_vertical = (
            abs(centro_y - caja_persona["centro_y"])
            / alto_persona
        )

        if diferencia_vertical > 0.65:
            return 0.0

        puntuacion = (
            0.35
            * (1.0 - distancia_normalizada / 0.65)
        )

        return max(0.0, puntuacion)

    def asociar_objetos_a_personas(self, detecciones):
        """
        Asocia cada objeto como máximo a una persona.

        Conserva temporalmente asociaciones anteriores cuando
        un objeto deja de detectarse durante algunos fotogramas,
        siempre que el ID continúe siendo válido.

        Los objetos lejanos permanecen independientes.
        """

        personas = [
            d for d in detecciones
            if (
                d["tipo"] == "persona"
                and d["id"] is not None
            )
        ]

        objetos = [
            d for d in detecciones
            if (
                d["tipo"] == "objeto"
                and d["id"] is not None
            )
        ]

        personas_por_id = {
            persona["id"]: persona
            for persona in personas
        }

        objetos_observados = set()

        for objeto in objetos:

            id_objeto = objeto["id"]

            objetos_observados.add(id_objeto)

            mejor_persona = None
            mejor_puntuacion = 0.0

            for persona in personas:

                puntuacion = self.calcular_relacion_objeto_persona(
                    objeto,
                    persona,
                )

                if puntuacion > mejor_puntuacion:

                    mejor_puntuacion = puntuacion
                    mejor_persona = persona

            asociacion_anterior = (
                self.asociaciones_objetos.get(id_objeto)
            )

            # Si existe una asociación anterior válida,
            # evitar cambiar de persona por pequeñas variaciones.
            if asociacion_anterior is not None:

                id_persona_anterior = (
                    asociacion_anterior["id_persona"]
                )

                persona_anterior = personas_por_id.get(
                    id_persona_anterior
                )

                if persona_anterior is not None:

                    puntuacion_anterior = (
                        self.calcular_relacion_objeto_persona(
                            objeto,
                            persona_anterior,
                        )
                    )

                    if puntuacion_anterior >= 0.10:

                        if (
                            mejor_persona is None
                            or puntuacion_anterior
                            >= mejor_puntuacion * 0.80
                        ):

                            mejor_persona = persona_anterior

                            mejor_puntuacion = (
                                puntuacion_anterior
                            )

            if (
                mejor_persona is not None
                and mejor_puntuacion >= 0.15
            ):

                objeto["transportado_por"] = (
                    mejor_persona["id"]
                )

                self.asociaciones_objetos[id_objeto] = {
                    "id_persona": mejor_persona["id"],
                    "ultimo_fotograma": self.numero_fotograma,
                }

            elif asociacion_anterior is not None:

                # Mantener la relación anterior si la persona
                # todavía no puede descartarse.
                id_persona_anterior = (
                    asociacion_anterior["id_persona"]
                )

                tiempo_transcurrido = (
                    self.numero_fotograma
                    - asociacion_anterior["ultimo_fotograma"]
                )

                if (
                    id_persona_anterior in personas_por_id
                    and tiempo_transcurrido <= 3
                ):

                    objeto["transportado_por"] = (
                        id_persona_anterior
                    )

        # Actualizar el historial de objetos no observados.
        # No se elimina inmediatamente porque el objeto puede
        # desaparecer momentáneamente y volver con el mismo ID.
        for id_objeto, asociacion in list(
            self.asociaciones_objetos.items()
        ):

            if id_objeto not in objetos_observados:

                if (
                    self.numero_fotograma
                    - asociacion["ultimo_fotograma"]
                    > self.fotogramas_perdida
                ):

                    self.asociaciones_objetos.pop(
                        id_objeto,
                        None,
                    )
