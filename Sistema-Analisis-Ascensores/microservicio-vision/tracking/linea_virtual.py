
class LineaVirtual:
    """
    Detecta cruces de personas a través de una línea vertical.

    Dirección:
        Izquierda -> derecha: entrada.
        Derecha -> izquierda: salida.

    El cruce se confirma después de observar varios fotogramas
    consecutivos al otro lado de la línea.

    Utiliza los IDs globales proporcionados por SeguimientoObjetos.
    """

    def __init__(
        self,
        x_linea: int = 320,
        margen: int = 8,
        fotogramas_confirmacion: int = 2,
        fotogramas_perdida: int = 30,
    ):

        self.x_linea = int(x_linea)
        self.margen = max(0, int(margen))

        self.fotogramas_confirmacion = max(
            1,
            int(fotogramas_confirmacion),
        )

        self.fotogramas_perdida = max(
            1,
            int(fotogramas_perdida),
        )

        # Último lado confirmado de cada persona.
        self.lados_confirmados = {}

        # Lado al que intenta cruzar cada persona.
        self.lados_pendientes = {}

        # Cantidad de fotogramas consecutivos al otro lado.
        self.confirmaciones = {}

        # Última observación de cada persona.
        self.ultimas_observaciones = {}

        # Último fotograma procesado.
        self.numero_fotograma = 0

        # Historial del último cruce por persona.
        self.ultimos_cruces = {}

    def configurar_linea(self, x_linea):
        """
        Permite modificar la posición de la línea virtual.
        Reinicia los intentos pendientes para evitar cruces
        artificiales al cambiar su posición.
        """

        self.x_linea = int(x_linea)

        self.lados_pendientes.clear()
        self.confirmaciones.clear()

    def actualizar_fotograma(self, detecciones):
        """
        Debe llamarse exactamente una vez por fotograma,
        incluso cuando no se detectan personas.

        Actualiza la presencia de los IDs y limpia los antiguos.
        """

        self.numero_fotograma += 1

        ids_presentes = set()

        for objeto in detecciones:

            if objeto.get("tipo") != "persona":
                continue

            id_persona = objeto.get("id")

            if id_persona is None:
                continue

            ids_presentes.add(id_persona)

            self.ultimas_observaciones[id_persona] = (
                self.numero_fotograma
            )

        self.limpiar_ids_inactivos(ids_presentes)

    def limpiar_ids_inactivos(self, ids_presentes=None):
        """
        Elimina el historial de personas que llevan demasiado
        tiempo sin aparecer.
        """

        ids_conocidos = (
            set(self.lados_confirmados)
            | set(self.lados_pendientes)
            | set(self.ultimas_observaciones)
            | set(self.ultimos_cruces)
        )

        for id_persona in ids_conocidos:

            if (
                ids_presentes is not None
                and id_persona in ids_presentes
            ):
                continue

            ultimo = self.ultimas_observaciones.get(
                id_persona
            )

            if ultimo is None:
                continue

            fotogramas_ausente = (
                self.numero_fotograma - ultimo
            )

            if fotogramas_ausente > self.fotogramas_perdida:

                self.lados_confirmados.pop(
                    id_persona,
                    None,
                )

                self.lados_pendientes.pop(
                    id_persona,
                    None,
                )

                self.confirmaciones.pop(
                    id_persona,
                    None,
                )

                self.ultimas_observaciones.pop(
                    id_persona,
                    None,
                )

                self.ultimos_cruces.pop(
                    id_persona,
                    None,
                )

    def determinar_lado(self, centro_x):
        """
        Determina en qué lado se encuentra la persona.

        Dentro del margen no se confirma ningún cambio.
        """

        if centro_x < self.x_linea - self.margen:
            return "izquierda"

        if centro_x > self.x_linea + self.margen:
            return "derecha"

        return None

    def detectar_cruce(self, objeto):
        """
        Detecta un cruce confirmado.

        Retorna:
            izquierda_a_derecha
            derecha_a_izquierda
            None

        Una persona solamente genera un evento cuando completa
        el cruce y alcanza el número de fotogramas requerido.
        """

        if objeto.get("tipo") != "persona":
            return None

        id_persona = objeto.get("id")

        if id_persona is None:
            return None

        posicion = objeto.get("posicion", {})

        centro_x = posicion.get("centro_x")

        if centro_x is None:
            return None

        lado_actual = self.determinar_lado(
            centro_x
        )

        if lado_actual is None:
            return None

        self.ultimas_observaciones[id_persona] = (
            self.numero_fotograma
        )

        # Primera observación:
        # se establece el lado inicial, pero no se registra
        # ningún ingreso ni salida.
        if id_persona not in self.lados_confirmados:

            self.lados_confirmados[id_persona] = (
                lado_actual
            )

            return None

        lado_anterior = self.lados_confirmados[
            id_persona
        ]

        # La persona regresó a su lado inicial.
        # Cancelar el intento de cruce.
        if lado_actual == lado_anterior:

            self.lados_pendientes.pop(
                id_persona,
                None,
            )

            self.confirmaciones.pop(
                id_persona,
                None,
            )

            return None

        # Iniciar un nuevo intento de cruce.
        if (
            self.lados_pendientes.get(id_persona)
            != lado_actual
        ):

            self.lados_pendientes[id_persona] = (
                lado_actual
            )

            self.confirmaciones[id_persona] = 1

        else:

            self.confirmaciones[id_persona] += 1

        # Esperar la confirmación.
        if (
            self.confirmaciones[id_persona]
            < self.fotogramas_confirmacion
        ):
            return None

        # Confirmar el nuevo lado.
        self.lados_confirmados[id_persona] = (
            lado_actual
        )

        self.lados_pendientes.pop(
            id_persona,
            None,
        )

        self.confirmaciones.pop(
            id_persona,
            None,
        )

        # Registrar el fotograma del cruce.
        self.ultimos_cruces[id_persona] = (
            self.numero_fotograma
        )

        # Determinar la dirección.
        if (
            lado_anterior == "izquierda"
            and lado_actual == "derecha"
        ):

            return "izquierda_a_derecha"

        if (
            lado_anterior == "derecha"
            and lado_actual == "izquierda"
        ):

            return "derecha_a_izquierda"

        return None

    def reiniciar_posiciones(self):
        """
        Reinicia los estados de la línea virtual.

        No reinicia los IDs globales: esa responsabilidad
        corresponde al administrador de identificadores.
        """

        self.lados_confirmados.clear()
        self.lados_pendientes.clear()
        self.confirmaciones.clear()
        self.ultimas_observaciones.clear()
        self.ultimos_cruces.clear()

        self.numero_fotograma = 0
