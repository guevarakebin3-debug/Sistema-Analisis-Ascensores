
class ContadorOcupacion:
    """
    Registra los cruces confirmados de la cámara exterior.

    Responsabilidades:
        - Registrar entradas.
        - Registrar salidas.
        - Conservar el ID global de cada persona.
        - Evitar duplicar eventos.
        - Mantener un historial de cruces.

    No calcula la ocupación actual de la cabina.
    """

    def __init__(self):

        self.entradas = 0
        self.salidas = 0

        # Historial de eventos confirmados.
        self.eventos = []

        # Último cruce registrado por persona.
        self.ultimos_cruces = {}

    def registrar_cruce(
        self,
        direccion,
        id_persona=None,
    ):
        """
        Registra un cruce confirmado por LineaVirtual.

        Parámetros:
            direccion:
                izquierda_a_derecha
                derecha_a_izquierda

            id_persona:
                Identificador global de la persona.

        Devuelve:
            entrada
            salida
            None
        """

        if id_persona is None:
            return None

        if direccion == "izquierda_a_derecha":

            tipo = "entrada"

        elif direccion == "derecha_a_izquierda":

            tipo = "salida"

        else:

            return None

        # Evitar registrar dos veces el mismo tipo de cruce
        # para una persona sin que antes haya ocurrido
        # el cruce contrario.
        ultimo_tipo = self.ultimos_cruces.get(id_persona)

        if ultimo_tipo == tipo:
            return None

        if tipo == "entrada":
            self.entradas += 1

        else:
            self.salidas += 1

        evento = {
            "id": id_persona,
            "direccion": direccion,
            "tipo": tipo,
            "numero_evento": len(self.eventos) + 1,
        }

        self.eventos.append(evento)

        self.ultimos_cruces[id_persona] = tipo

        return tipo

    def obtener_estado(self):
        """
        Devuelve el resumen de eventos.

        Las entradas y salidas son estadísticas de cruces,
        no una medición de la ocupación actual.
        """

        return {
            "entradas": self.entradas,
            "salidas": self.salidas,
            "total_eventos": len(self.eventos),
        }

    def obtener_eventos(self):
        """
        Devuelve una copia del historial de eventos.
        """

        return [
            evento.copy()
            for evento in self.eventos
        ]

    def obtener_ultimo_evento(self):
        """
        Devuelve el último evento registrado.
        """

        if not self.eventos:
            return None

        return self.eventos[-1].copy()

    def reiniciar(self):
        """
        Limpia los eventos de esta sesión de cámara.

        No reinicia el administrador global de IDs.
        """

        self.entradas = 0
        self.salidas = 0

        self.eventos.clear()
        self.ultimos_cruces.clear()
