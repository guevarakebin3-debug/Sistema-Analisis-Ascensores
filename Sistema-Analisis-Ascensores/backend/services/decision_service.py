
def evaluar_ocupacion(
    superficie_actual_m2: float,
    superficie_entrante_m2: float,
    superficie_util_cabina_m2: float,
    personas_actuales: int,
    personas_entrantes: int,
    capacidad_maxima_personas: int,
    limite_ocupacion_pct: float
) -> dict:

    # Validar superficies
    if superficie_util_cabina_m2 <= 0:
        raise ValueError(
            "La superficie útil debe ser mayor que cero."
        )

    if superficie_actual_m2 < 0 or superficie_entrante_m2 < 0:
        raise ValueError(
            "Las superficies no pueden ser negativas."
        )

    # Validar personas
    if personas_actuales < 0 or personas_entrantes < 0:
        raise ValueError(
            "El número de personas no puede ser negativo."
        )

    if capacidad_maxima_personas <= 0:
        raise ValueError(
            "La capacidad máxima debe ser mayor que cero."
        )

    if not 0 <= limite_ocupacion_pct <= 100:
        raise ValueError(
            "El límite debe estar entre 0 y 100."
        )

    # Calcular superficie proyectada
    superficie_proyectada = (
        superficie_actual_m2 + superficie_entrante_m2
    )

    porcentaje_actual = (
        superficie_actual_m2 / superficie_util_cabina_m2
    ) * 100

    porcentaje_entrante = (
        superficie_entrante_m2 / superficie_util_cabina_m2
    ) * 100

    porcentaje_proyectado = (
        superficie_proyectada / superficie_util_cabina_m2
    ) * 100

    # Calcular personas proyectadas
    personas_proyectadas = (
        personas_actuales + personas_entrantes
    )

    # Evaluar límite espacial
    superficie_permitida = (
        porcentaje_proyectado <= limite_ocupacion_pct
    )

    # Evaluar capacidad de personas
    personas_permitidas = (
        personas_proyectadas <= capacidad_maxima_personas
    )

    permitir_ingreso = (
        superficie_permitida and personas_permitidas
    )

    # Construir motivos
    motivos = []

    if not superficie_permitida:
        motivos.append(
            "Se supera el límite de ocupación espacial."
        )

    if not personas_permitidas:
        motivos.append(
            "Se supera la capacidad máxima de personas."
        )

    if permitir_ingreso:
        motivo = "Ingreso permitido."
    else:
        motivo = " ".join(motivos)

    return {
        "superficie_actual_m2": round(
            superficie_actual_m2, 3
        ),
        "superficie_entrante_m2": round(
            superficie_entrante_m2, 3
        ),
        "superficie_proyectada_m2": round(
            superficie_proyectada, 3
        ),
        "porcentaje_actual": round(
            porcentaje_actual, 2
        ),
        "porcentaje_entrante": round(
            porcentaje_entrante, 2
        ),
        "porcentaje_proyectado": round(
            porcentaje_proyectado, 2
        ),
        "personas_proyectadas": personas_proyectadas,
        "permitir_ingreso": permitir_ingreso,
        "motivo": motivo,
        "limite_ocupacion_pct": limite_ocupacion_pct,
        "capacidad_maxima_personas": capacidad_maxima_personas
    }
