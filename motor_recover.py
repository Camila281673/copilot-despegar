"""
motor_recover.py
================================================================================
NÚCLEO DEL PROYECTO "Recover the Long Tail" — Ejercicio Analista AI · Despegar

Este módulo concentra TODA la lógica de negocio del proyecto en un solo lugar:
    - cargar y validar el dataset
    - calcular el índice de ENGAGEMENT (0 a 100)
    - clasificar cada hotel por NIVEL de engagement (Alto / Medio / Bajo)
    - segmentar cada hotel por CAUSA RAÍZ del bajo rendimiento (5 grupos)
    - diagnosticar las causas concretas de un hotel
    - recomendar una acción + un mensaje para el hotelero (reglas, y opcional LLM)
    - estimar el impacto en reservas del segmento priorizado

¿Por qué un solo archivo compartido?
    Tanto el notebook de análisis (Colab) como el prototipo (Streamlit) IMPORTAN
    este módulo. Así, si cambia un umbral o un peso, cambia en UN solo lugar y
    todos los entregables quedan consistentes entre sí (mismos números en el
    análisis, en la presentación y en el prototipo). No se duplica la lógica.

Estilo: funciones cortas, nombres en español, comentarios que explican el PORQUÉ.
Prioridad: que se entienda y se pueda explicar, antes que ser "ingenioso".
================================================================================
"""

import pandas as pd


# ==============================================================================
# CONFIG — Todos los "números mágicos" viven acá, documentados y en un solo lugar.
# Cambiar un umbral o un peso es cambiar una línea de este bloque.
# ==============================================================================
CONFIG = {
    # --- Columnas que el dataset DEBE tener (para validar que no falte nada) ---
    "columnas_esperadas": [
        "hotel_id", "nombre_hotel", "pais", "ciudad", "categoria_estrellas",
        "tipo_conexion", "clicks_90d", "reservas_90d", "conversion_rate",
        "disponibilidad_pct", "dias_sin_actualizar_tarifas", "tiene_fotos",
        "cantidad_fotos", "rating_promedio",
    ],

    # --- Pesos del índice de engagement ---------------------------------------
    # Se eligieron PROPORCIONALES a la correlación de cada variable con las
    # reservas (conversion 0.67, fotos 0.64, disponibilidad 0.40, frescura 0.33),
    # redondeados para que sumen 1.0. No son arbitrarios: están justificados por
    # los datos. (Ver el notebook §3 y §4 para el detalle.)
    "pesos_engagement": {
        "conversion": 0.35,    # eficacia comercial: ¿convierte el tráfico que recibe?
        "fotos": 0.30,         # inversión en la ficha: fotos cargadas
        "disponibilidad": 0.20,  # inventario cargado para el próximo trimestre
        "frescura_tarifas": 0.15,  # ¿mantiene las tarifas actualizadas? (anti-abandono)
    },

    # --- Reglas de segmentación por CAUSA RAÍZ --------------------------------
    # Un hotel cae en exactamente UN segmento (reglas excluyentes y exhaustivas).
    "umbral_tarifas_abandono": 90,    # días sin actualizar para considerarlo "abandonado"
    "umbral_disponibilidad_baja": 0.30,  # < 30% de fechas cargadas = casi sin inventario
    "umbral_clicks_trafico": 40,      # clicks altos = "recibe tráfico"
    "umbral_clicks_invisible": 15,    # clicks muy bajos = "nadie lo ve"
    "umbral_disponibilidad_ok": 0.60,  # disponibilidad razonable para "estar listo"

    # --- Umbrales para el DIAGNÓSTICO de causas concretas (copilot) ------------
    "umbral_pocas_fotos": 5,          # menos de 5 fotos = ficha pobre
    "umbral_tarifas_viejas": 30,      # > 30 días sin tocar tarifas = empieza a pesar
    "umbral_disponibilidad_mejorable": 0.50,
    "umbral_rating_bajo": 3.5,

    # --- Modelo de LLM por defecto (solo se usa si hay API key) ----------------
    "modelo_llm": "claude-opus-4-8",
}

# Nombres de los segmentos (constantes para no escribirlos "a mano" en varios lados)
SEG_PRODUCTIVOS = "Productivos"
SEG_ABANDONADOS = "Abandonados"
SEG_TRAFICO = "Tráfico sin conversión"
SEG_INVISIBLES = "Invisibles"
SEG_OTROS = "Otros bajos"

# El segmento que vamos a PRIORIZAR (target). Se nombra una sola vez acá.
SEGMENTO_OBJETIVO = SEG_TRAFICO


# ==============================================================================
# 1) CARGA Y VALIDACIÓN
# ==============================================================================
def validar_dataset(df):
    """
    Revisa que el DataFrame tenga lo que esperamos. Si algo no cuadra, AVISA
    con un mensaje claro en vez de seguir y dar resultados raros más adelante.

    Devuelve una lista de advertencias (vacía si está todo bien). Las cosas
    GRAVES (faltan columnas) levantan un error; las cosas menores (un rango raro)
    solo se reportan como advertencia.
    """
    advertencias = []

    # --- Grave: que estén todas las columnas esperadas ---
    faltantes = [c for c in CONFIG["columnas_esperadas"] if c not in df.columns]
    if faltantes:
        raise ValueError(
            "Al dataset le faltan columnas esperadas: "
            + ", ".join(faltantes)
            + ". Revisá que sea el archivo correcto."
        )

    # --- Grave: que no haya valores nulos (el dataset del ejercicio no tiene) ---
    nulos = df[CONFIG["columnas_esperadas"]].isna().sum()
    columnas_con_nulos = nulos[nulos > 0]
    if len(columnas_con_nulos) > 0:
        raise ValueError(
            "Hay valores nulos en: "
            + ", ".join(columnas_con_nulos.index)
            + ". Decidir cómo tratarlos antes de seguir (no imputamos en silencio)."
        )

    # --- Menor: rangos fuera de lo esperado (solo se avisa) ---
    if not df["disponibilidad_pct"].between(0, 1).all():
        advertencias.append("disponibilidad_pct tiene valores fuera de [0, 1].")
    if not df["rating_promedio"].between(1, 5).all():
        advertencias.append("rating_promedio tiene valores fuera de [1, 5].")
    if (df["clicks_90d"] < 0).any() or (df["reservas_90d"] < 0).any():
        advertencias.append("Hay clicks o reservas negativos (no debería pasar).")

    # --- Menor: consistencia entre 'tiene_fotos' y 'cantidad_fotos' ---
    inconsistentes = (
        (df["tiene_fotos"] & (df["cantidad_fotos"] == 0))
        | (~df["tiene_fotos"] & (df["cantidad_fotos"] > 0))
    ).sum()
    if inconsistentes > 0:
        advertencias.append(
            f"{inconsistentes} hoteles tienen 'tiene_fotos' inconsistente con 'cantidad_fotos'."
        )

    return advertencias


def cargar_datos(path):
    """
    Lee el dataset (acepta .xlsx o .csv) y lo valida.
    Si el archivo no existe, avisa con un mensaje claro.
    """
    try:
        if str(path).lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en: {path}. "
            "Verificá la ruta o subí el archivo a la sesión."
        )

    validar_dataset(df)  # si algo grave está mal, corta acá
    return df


# ==============================================================================
# 2) ÍNDICE DE ENGAGEMENT
# ==============================================================================
def normalizar_0a1(serie, invertir=False):
    """
    Lleva una columna a la escala 0..1 con min-max:  (x - min) / (max - min).

    - invertir=True: devuelve 1 - resultado. Sirve para variables donde "más es
      peor" (ej: días sin actualizar tarifas: mientras más días, peor).
    - Guarda anti-división-por-cero: si todos los valores son iguales (max == min),
      no se puede normalizar, así que devolvemos 0.5 (un valor neutro) para todos.
    """
    minimo = serie.min()
    maximo = serie.max()

    if maximo == minimo:
        # Columna constante: devolvemos un valor neutro en vez de dividir por cero.
        resultado = pd.Series(0.5, index=serie.index)
    else:
        resultado = (serie - minimo) / (maximo - minimo)

    if invertir:
        resultado = 1 - resultado
    return resultado


def calcular_engagement(df):
    """
    Agrega la columna 'engagement_score' (0 a 100).

    Engagement = SALUD del hotel en la plataforma, NO sus ventas.
    (Si lo definiéramos como reservas, sería circular: todo el long tail daría 0.)

    Es un índice compuesto de 4 palancas que el hotel SÍ controla:
        - conversión   (¿convierte el tráfico que recibe?)
        - fotos        (¿invirtió en la ficha?)
        - disponibilidad (¿cargó inventario?)
        - frescura de tarifas (¿mantiene los precios actualizados?)

    Cada palanca se normaliza a 0..1 y se combina con los pesos de CONFIG.
    El resultado se lleva a 0..100 para que sea fácil de leer.

    Nota de diseño: NO incluimos clicks (es demanda/visibilidad, no una palanca
    del hotel) ni rating (reputación, difícil de mover en el corto plazo). Se
    documenta como decisión en el notebook.
    """
    df = df.copy()
    pesos = CONFIG["pesos_engagement"]

    # 1) Normalizamos cada palanca a 0..1
    n_conversion = normalizar_0a1(df["conversion_rate"])
    n_fotos = normalizar_0a1(df["cantidad_fotos"])
    n_disponibilidad = normalizar_0a1(df["disponibilidad_pct"])
    # frescura: más días sin actualizar = peor -> invertimos
    n_frescura = normalizar_0a1(df["dias_sin_actualizar_tarifas"], invertir=True)

    # 2) Combinamos con los pesos y pasamos a escala 0..100
    score = (
        pesos["conversion"] * n_conversion
        + pesos["fotos"] * n_fotos
        + pesos["disponibilidad"] * n_disponibilidad
        + pesos["frescura_tarifas"] * n_frescura
    ) * 100

    df["engagement_score"] = score.round(1)
    return df


def clasificar_nivel_engagement(df):
    """
    Agrega la columna 'nivel_engagement' con tres niveles: Alto / Medio / Bajo.

    Esto responde LITERAL a la consigna ("segmentar por nivel de engagement").
    Usamos terciles (cortamos en los percentiles 33 y 66) para partir el portfolio
    en tres tercios comparables.
    """
    df = df.copy()
    if "engagement_score" not in df.columns:
        df = calcular_engagement(df)

    # qcut parte en tercios por cantidad de hoteles (terciles)
    df["nivel_engagement"] = pd.qcut(
        df["engagement_score"],
        q=3,
        labels=["Bajo", "Medio", "Alto"],
    )
    return df


# ==============================================================================
# 3) SEGMENTACIÓN POR CAUSA RAÍZ
# ==============================================================================
def _segmento_de_fila(fila):
    """
    Devuelve el segmento (causa raíz) de UN hotel.
    Reglas excluyentes y exhaustivas: todo hotel cae en exactamente un grupo.

    Lógica (en orden):
        1. Si tiene reservas > 0           -> Productivos (ya funcionan)
        2. Si tarifas muy viejas O casi sin inventario -> Abandonados (no hay qué vender)
        3. Si recibe mucho tráfico         -> Tráfico sin conversión (los ven, no cierran)
        4. Si casi nadie lo ve pero está cargado -> Invisibles (listos, sin demanda)
        5. El resto                        -> Otros bajos (casos mixtos)
    """
    if fila["reservas_90d"] > 0:
        return SEG_PRODUCTIVOS

    if (
        fila["dias_sin_actualizar_tarifas"] > CONFIG["umbral_tarifas_abandono"]
        or fila["disponibilidad_pct"] < CONFIG["umbral_disponibilidad_baja"]
    ):
        return SEG_ABANDONADOS

    if fila["clicks_90d"] >= CONFIG["umbral_clicks_trafico"]:
        return SEG_TRAFICO

    if (
        fila["clicks_90d"] < CONFIG["umbral_clicks_invisible"]
        and fila["disponibilidad_pct"] >= CONFIG["umbral_disponibilidad_ok"]
    ):
        return SEG_INVISIBLES

    return SEG_OTROS


def segmentar(df):
    """Agrega la columna 'segmento' aplicando las reglas de causa raíz a cada hotel."""
    df = df.copy()
    df["segmento"] = df.apply(_segmento_de_fila, axis=1)
    return df


# ==============================================================================
# 4) DIAGNÓSTICO DE CAUSAS CONCRETAS (para el copilot)
# ==============================================================================
def diagnosticar_causa(fila):
    """
    Devuelve una LISTA de causas concretas que explican por qué un hotel no
    convierte. Cada causa es un dict con 'codigo', 'titulo' y 'detalle'.
    Si no detecta nada, devuelve lista vacía (no None).

    Esto alimenta la recomendación del copilot: primero entendemos QUÉ está mal,
    después sugerimos qué hacer.
    """
    causas = []

    sin_fotos = (not bool(fila["tiene_fotos"])) or (fila["cantidad_fotos"] == 0)
    if sin_fotos:
        causas.append({
            "codigo": "sin_fotos",
            "titulo": "No tiene fotos cargadas",
            "detalle": "La ficha sin fotos genera desconfianza y baja la conversión.",
        })
    elif fila["cantidad_fotos"] < CONFIG["umbral_pocas_fotos"]:
        causas.append({
            "codigo": "pocas_fotos",
            "titulo": f"Pocas fotos ({int(fila['cantidad_fotos'])})",
            "detalle": "Una galería pobre rinde menos que una ficha completa.",
        })

    if fila["dias_sin_actualizar_tarifas"] > CONFIG["umbral_tarifas_viejas"]:
        causas.append({
            "codigo": "tarifas_viejas",
            "titulo": f"Tarifas desactualizadas ({int(fila['dias_sin_actualizar_tarifas'])} días)",
            "detalle": "Precios viejos suelen quedar fuera de mercado y no cierran.",
        })

    if fila["disponibilidad_pct"] < CONFIG["umbral_disponibilidad_mejorable"]:
        causas.append({
            "codigo": "baja_disponibilidad",
            "titulo": f"Baja disponibilidad ({fila['disponibilidad_pct']:.0%})",
            "detalle": "Con pocas fechas cargadas, gran parte de las búsquedas no encuentran cupo.",
        })

    if fila["rating_promedio"] < CONFIG["umbral_rating_bajo"]:
        causas.append({
            "codigo": "rating_bajo",
            "titulo": f"Rating bajo ({fila['rating_promedio']:.1f})",
            "detalle": "Una reputación floja frena la decisión de reserva.",
        })

    return causas


# ==============================================================================
# 5) RECOMENDACIÓN (acción + mensaje al hotelero)
# ==============================================================================
# Acciones sugeridas por código de causa (lo que el Account Manager debería pedir).
_ACCIONES_POR_CAUSA = {
    "sin_fotos": "Pedir al hotel que cargue al menos 8–10 fotos profesionales.",
    "pocas_fotos": "Completar la galería hasta 8–10 fotos de calidad.",
    "tarifas_viejas": "Solicitar actualización de tarifas a valores de mercado.",
    "baja_disponibilidad": "Pedir que cargue disponibilidad para el próximo trimestre.",
    "rating_bajo": "Activar gestión de reseñas y revisar la experiencia del huésped.",
}


def recomendar(fila, cliente_api=None):
    """
    Devuelve un dict con:
        - 'causas'   : lista de causas detectadas
        - 'acciones' : lista de acciones sugeridas (una por causa)
        - 'mensaje'  : borrador de mensaje para enviarle al hotelero
        - 'fuente'   : "reglas" o "LLM (Claude)" según cómo se generó el mensaje

    Por defecto trabaja con REGLAS/plantillas: corre siempre, sin internet ni key.
    Si se pasa 'cliente_api' (cliente de Anthropic ya configurado), se usa el LLM
    para redactar un mensaje más natural. Si la llamada al LLM falla por cualquier
    motivo, se cae con elegancia a la versión por reglas (un script que SIEMPRE
    funciona). El humano siempre revisa antes de enviar.
    """
    causas = diagnosticar_causa(fila)
    acciones = [_ACCIONES_POR_CAUSA[c["codigo"]] for c in causas if c["codigo"] in _ACCIONES_POR_CAUSA]

    # --- Mensaje base por reglas (plantilla) ---
    mensaje_reglas = _mensaje_por_reglas(fila, causas, acciones)
    resultado = {
        "causas": causas,
        "acciones": acciones,
        "mensaje": mensaje_reglas,
        "fuente": "reglas",
    }

    # --- Camino opcional con LLM ---
    if cliente_api is not None:
        try:
            mensaje_llm = _mensaje_por_llm(fila, causas, acciones, cliente_api)
            if mensaje_llm:  # solo si vino algo no vacío
                resultado["mensaje"] = mensaje_llm
                resultado["fuente"] = "LLM (Claude)"
        except Exception:
            # Si el LLM falla (red, key, rate limit, etc.) nos quedamos con reglas.
            # No rompemos la app: el mensaje por plantilla ya está listo.
            pass

    return resultado


def _mensaje_por_reglas(fila, causas, acciones):
    """Arma un mensaje simple y correcto a partir de plantillas. Determinista."""
    nombre = fila["nombre_hotel"]

    if not causas:
        return (
            f"Hola, equipo de {nombre}. Vimos buen potencial en su propiedad. "
            "Coordinemos una llamada para potenciar su presencia en la plataforma."
        )

    lista_acciones = "\n".join(f"  • {a}" for a in acciones)
    return (
        f"Hola, equipo de {nombre}.\n\n"
        f"Vimos que su hotel recibe visitas en Despegar, pero todavía no se traducen "
        f"en reservas. Detectamos algunas oportunidades concretas para mejorar:\n"
        f"{lista_acciones}\n\n"
        f"Son cambios rápidos que suelen mover la aguja. ¿Coordinamos una llamada esta semana "
        f"para implementarlos juntos?\n\nSaludos,\nEquipo de Travel Partners · Despegar"
    )


def _mensaje_por_llm(fila, causas, acciones, cliente_api):
    """
    Pide al LLM que redacte el mensaje. Recibe SOLO datos saneados del hotel
    (no metemos texto arbitrario en el prompt). Devuelve el texto generado.
    """
    titulos_causas = "; ".join(c["titulo"] for c in causas) or "sin causas detectadas"
    lista_acciones = "; ".join(acciones) or "ninguna"

    # Prompt acotado: el LLM solo redacta, no decide la estrategia.
    prompt = (
        "Sos un asistente de un Account Manager de Despegar (Travel Partners). "
        "Redactá un mensaje breve, cordial y profesional en español rioplatense para el "
        "hotelero, proponiendo mejoras concretas. Máximo 120 palabras.\n\n"
        f"Hotel: {fila['nombre_hotel']} ({fila['ciudad']}, {fila['pais']}).\n"
        f"Problema: recibe tráfico pero no convierte.\n"
        f"Causas detectadas: {titulos_causas}.\n"
        f"Acciones a proponer: {lista_acciones}.\n\n"
        "Devolvé solo el texto del mensaje, sin encabezados ni explicaciones."
    )

    respuesta = cliente_api.messages.create(
        model=CONFIG["modelo_llm"],
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    # La respuesta de la API trae una lista de bloques; tomamos el texto del primero.
    return respuesta.content[0].text.strip()


# ==============================================================================
# 6) ESTIMACIÓN DE IMPACTO
# ==============================================================================
def estimar_impacto(df, factor_conservador=0.25, factor_base=0.50):
    """
    Estima cuántas reservas adicionales por trimestre se podrían recuperar si el
    segmento objetivo ('Tráfico sin conversión') convirtiera como los Productivos.

    Método (transparente y conservador):
        1. Tomamos los clicks totales del segmento objetivo.
        2. Usamos la conversión AGREGADA de los Productivos
           (reservas totales / clicks totales), que es el valor esperado correcto
           para un grupo (no el promedio de ratios, que sobreestima).
        3. Techo teórico = clicks_objetivo * conversion_productivos.
        4. Mostramos un RANGO aplicando factores de captura realistas
           (no todo el potencial se logra): conservador y base.

    Devuelve un dict con los números y los supuestos, listo para la presentación.
    """
    df = segmentar(df) if "segmento" not in df.columns else df

    productivos = df[df["segmento"] == SEG_PRODUCTIVOS]
    objetivo = df[df["segmento"] == SEGMENTO_OBJETIVO]

    # Conversión agregada de los productivos (con guarda anti-división-por-cero)
    clicks_prod = productivos["clicks_90d"].sum()
    conv_productivos = productivos["reservas_90d"].sum() / clicks_prod if clicks_prod > 0 else 0.0

    clicks_objetivo = objetivo["clicks_90d"].sum()
    techo = clicks_objetivo * conv_productivos

    base_actual = df["reservas_90d"].sum()

    return {
        "n_objetivo": len(objetivo),
        "clicks_objetivo": int(clicks_objetivo),
        "conversion_productivos": round(conv_productivos, 4),
        "techo_reservas": round(techo, 1),
        "escenario_conservador": round(techo * factor_conservador, 1),
        "escenario_base": round(techo * factor_base, 1),
        "base_actual_reservas": int(base_actual),
        # Variación % sobre la base actual del dataset, para comunicar el impacto
        "var_pct_conservador": round(techo * factor_conservador / base_actual * 100, 1) if base_actual else 0,
        "var_pct_base": round(techo * factor_base / base_actual * 100, 1) if base_actual else 0,
        "var_pct_techo": round(techo / base_actual * 100, 1) if base_actual else 0,
        "supuestos": (
            "Supone que el segmento objetivo alcanza la conversión agregada de los "
            "Productivos. El techo es teórico; aplicamos factores de captura del "
            f"{int(factor_conservador*100)}% (conservador) y {int(factor_base*100)}% (base)."
        ),
    }


# ==============================================================================
# 7) PIPELINE COMPLETO (lo que usan el notebook y la app)
# ==============================================================================
def procesar(df):
    """
    Corre todo el pipeline sobre un DataFrame ya cargado:
    engagement -> nivel -> segmento. Devuelve el DataFrame enriquecido.
    """
    df = calcular_engagement(df)
    df = clasificar_nivel_engagement(df)
    df = segmentar(df)
    return df


def procesar_desde_archivo(path):
    """Carga el archivo, lo valida y corre el pipeline. Atajo de una línea."""
    df = cargar_datos(path)
    return procesar(df)
