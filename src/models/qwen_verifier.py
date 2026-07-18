"""Capa opcional de verificacion con un LLM generativo (Qwen/Qwen3-0.6B).

Se activa solo para categorias cuya probabilidad de BETO cae en zona dudosa
cerca de su umbral (ver core.logic / MARGENES_POR_CATEGORIA). Todo el modulo
esta disenado para que un fallo aqui (sin conexion, modelo no disponible,
respuesta no parseable, etc.) nunca tumbe la API -- si algo falla, se
loggea y se retorna una senal de "sin verdicto" para que el llamador se
quede con la decision original de BETO.

Carga perezosa: si HABILITAR_QWEN=false, este modulo no descarga ni carga
nada en memoria.
"""

import json
import logging
import re
from datetime import datetime

import torch

from src.core.config import HABILITAR_QWEN, HF_TOKEN, QWEN_AUDITORIA_PATH, QWEN_MODEL_NAME

logger = logging.getLogger("moderacion.qwen")

DEFINICIONES_CATEGORIA = {
    "grosero": "contiene vocabulario vulgar o insultos directos con intención de ofender",
    "amenaza": "expresa intención explícita o implícita de causar daño a una persona o negocio",
    "inapropiado": "contiene insinuación sexual o doble sentido, sin necesidad de vulgaridad explícita",
}

# Instruccion de COMO analizar el texto, especifica por categoria. Antes
# habia una sola instruccion generica ("presta atencion a insinuaciones,
# dobles sentidos...") que solo tiene sentido para "inapropiado" -- usarla
# tambien para "amenaza" o "grosero" sesgaba a Qwen a inventar significados
# ocultos donde debia leer el texto de forma literal.
INSTRUCCIONES_ANALISIS_POR_CATEGORIA = {
    "grosero": (
        "Analiza el texto de forma literal -- fíjate únicamente en vocabulario "
        "vulgar, insultos directos o groserías explícitas dirigidas a alguien. "
        "No le busques significados ocultos a las palabras."
    ),
    "amenaza": (
        "Analiza el texto de forma LITERAL. NO busques dobles sentidos ni "
        "significados ocultos: evalúa únicamente si expresa una intención "
        "real y directa de causar daño a una persona o negocio. Un "
        'intercambio, condición o acuerdo normal (ej. "si me ayudas con X, '
        'te doy Y") NO es una amenaza, aunque mencione alguna consecuencia.'
    ),
    "inapropiado": (
        "Analiza el texto con cuidado -- presta atención a insinuaciones, "
        "dobles sentidos, o frases que sugieran algo más allá de su "
        "significado literal."
    ),
}

INSTRUCCIONES_ANALISIS_POR_CATEGORIA["grosero"] = (
    "Analiza el texto de forma literal -- fíjate ÚNICAMENTE en vocabulario "
    "vulgar, palabras soeces o insultos directos explícitos. Una amenaza o "
    "advertencia de consecuencias, SIN vocabulario vulgar, NO es grosero "
    "(esa sería otra categoría distinta)."
)

# Ejemplos few-shot ESPECIFICOS de cada categoria. Nunca se mezclan entre
# categorias -- mandarle a Qwen ejemplos de "inapropiado" mientras se
# verifica "amenaza" (el bug original) le genera razonamientos incoherentes
# porque termina anclando su respuesta a un concepto que no es el que se
# le esta preguntando.
EJEMPLOS_POR_CATEGORIA = {
    "grosero": {
        "positivos": [
            "Pésimo servicio, la señora de recepción es una grosera de mierda",
        ],
        "negativos": [
            "El servicio tardó demasiado, muy decepcionante la verdad",
        ],
    },
    "amenaza": {
        "positivos": [
            "Como no me paguen lo justo voy a ir a partirle la madre al encargado",
            "Ahí les va la fiscalía a estos perros que no pagan",
        ],
        "negativos": [
            "Voy a poner una queja formal ante PROFECO porque me pesaron mal el material",
            "Si me acompañas a bajar las cajas del segundo piso, te dejo escoger primero el material que más te sirva",
        ],
    },
    "inapropiado": {
        "positivos": [
            "¿La persona que atiende los martes es la misma que sale en la foto de perfil? Porque si es así, ya sé por qué vengo tan seguido.",
            "Buen servicio de pesaje, aunque la báscula no fue lo único que me dejó con ganas de volver mañana mismo.",
        ],
        "negativos": [
            "Ofrezco material en buen estado, disponible todo el día",
            "Vivo sola así que pueden pasar cuando gusten a recoger el material, nada más avisen antes por mensaje.",
        ],
    },
}

_JSON_BLOQUE = re.compile(r"\{.*\}", re.DOTALL)
# Plan B cuando el JSON completo no parsea (ej. una comilla suelta dentro
# de "razon" rompe el bloque entero): "confirma" es el unico campo que la
# logica de negocio realmente usa para decidir, y casi siempre se puede
# rescatar de forma aislada aunque el resto del JSON este mal formado.
_CONFIRMA_BLOQUE = re.compile(r'"confirma"\s*:\s*(true|false)', re.IGNORECASE)

# Mensajes propios (no generados por Qwen) que devuelve _parsear_respuesta
# cuando tiene que recurrir a un respaldo.
_RAZON_PLAN_B = "Razón no disponible (formato de respuesta inválido)"
_RAZON_SIN_JSON = "no se pudo interpretar la respuesta"

# Estado del modulo: se cargan una sola vez por proceso (lazy singleton).
_model = None
_tokenizer = None
_carga_intentada = False
_carga_exitosa = False


def cargar_modelo_qwen() -> None:
    """Intenta cargar el modelo/tokenizer de Qwen una sola vez por proceso.

    Es un no-op si HABILITAR_QWEN=false o si ya se intento cargar antes
    (exitosa o fallidamente -- no se reintenta en cada llamada). Pensada
    para invocarse explicitamente en el lifespan de la API al arrancar,
    pero tambien se auto-invoca de forma perezosa si nadie la llamo antes
    (ver esta_disponible()).
    """
    global _model, _tokenizer, _carga_intentada, _carga_exitosa

    if _carga_intentada:
        return
    _carga_intentada = True

    if not HABILITAR_QWEN:
        logger.info("Verificacion con Qwen deshabilitada (HABILITAR_QWEN=false); no se carga nada.")
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, token=HF_TOKEN)
        _model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _model.eval()
        _carga_exitosa = True

        logger.info("Modelo Qwen (%s) cargado correctamente.", QWEN_MODEL_NAME)

    except Exception as exc:  # noqa: BLE001 - un fallo de carga nunca debe tumbar la API
        logger.error(
            "No se pudo cargar el modelo Qwen (%s): %s. La API seguira funcionando solo con BETO.",
            QWEN_MODEL_NAME,
            exc,
        )
        _model = None
        _tokenizer = None
        _carga_exitosa = False


def esta_disponible() -> bool:
    """True si Qwen esta habilitado por config Y el modelo se pudo cargar.

    Dispara la carga perezosa si todavia no se habia intentado (red de
    seguridad para cuando nadie llamo a cargar_modelo_qwen() al arrancar,
    por ejemplo en tests que invocan la logica directamente).
    """
    if not HABILITAR_QWEN:
        return False
    if not _carga_intentada:
        cargar_modelo_qwen()
    return _carga_exitosa


def _parsear_respuesta(texto_generado: str) -> tuple:
    """Extrae {"confirma": bool, "razon": str} de la respuesta del modelo de
    forma tolerante (el modelo puede agregar texto extra alrededor del
    JSON, o generar un JSON internamente mal formado -- ej. una comilla
    suelta dentro de "razon"). Devuelve (None, "no se pudo interpretar la
    respuesta") solo si ni siquiera el campo "confirma" se puede rescatar.
    """
    # Intento 1: parseo estricto del JSON completo.
    coincidencia = _JSON_BLOQUE.search(texto_generado)
    if coincidencia is not None:
        try:
            resultado = json.loads(coincidencia.group(0))
            confirma = resultado.get("confirma")
            if isinstance(confirma, bool):
                return confirma, str(resultado.get("razon", "")).strip()
        except json.JSONDecodeError:
            pass  # seguimos al plan B, no fallamos todavia

    # Intento 2 (plan B): el JSON completo esta mal formado, pero
    # "confirma" -- el unico campo que la logica de negocio realmente usa
    # para decidir -- casi siempre se puede rescatar de forma aislada.
    coincidencia_confirma = _CONFIRMA_BLOQUE.search(texto_generado)
    if coincidencia_confirma is not None:
        confirma = coincidencia_confirma.group(1).lower() == "true"
        return confirma, _RAZON_PLAN_B

    return None, _RAZON_SIN_JSON


def _detectar_posible_inconsistencia(confirma: bool, razon: str) -> bool:
    """Heurística de AUDITORÍA (no bloqueante): compara el veredicto
    booleano contra el tono aparente del texto de la razón, buscando
    negaciones que podrían indicar que el modelo se contradijo. Solo
    para fines de logging/monitoreo -- nunca debe usarse para alterar
    el resultado real."""
    negaciones = [
        "no implica", "no es", "no representa", "no expresa",
        "no se trata de", "no constituye", "no contiene", "no sugiere",
        "no fit", "does not",
    ]
    razon_lower = razon.lower()
    tiene_negacion = any(neg in razon_lower for neg in negaciones)

    # Solo marcamos como sospechoso el caso confirma=True + razon con
    # negación fuerte (el patrón que causó el problema real detectado
    # en pruebas anteriores). El caso inverso (confirma=False + razon
    # sin negación) es demasiado ambiguo para esta heurística simple
    # y generaría muchos falsos positivos de auditoría sin valor real.
    return confirma is True and tiene_negacion


def _registrar_auditoria(categoria: str, texto: str, confirma: bool, razon: str) -> None:
    """Persiste una inconsistencia detectada en un JSONL duradero (el
    logger.warning por si solo no garantiza que quede algo revisable
    despues). Nunca debe tumbar la verificacion ni la respuesta de la API
    por un problema al escribir el archivo -- cualquier fallo aqui solo se
    loggea como advertencia."""
    try:
        QWEN_AUDITORIA_PATH.parent.mkdir(parents=True, exist_ok=True)

        evento = {
            "timestamp": datetime.now().isoformat(),
            "categoria": categoria,
            "texto": texto[:200],  # truncado por privacidad
            "confirma": confirma,
            "razon": razon,
        }

        with open(QWEN_AUDITORIA_PATH, "a", encoding="utf-8") as archivo:
            archivo.write(json.dumps(evento, ensure_ascii=False) + "\n")

    except Exception as exc:  # noqa: BLE001 - un fallo al auditar nunca debe tumbar la API
        logger.warning("No se pudo escribir el registro de auditoria de Qwen: %s", exc)


def _formatear_ejemplos(categoria: str) -> str:
    """Arma el bloque de ejemplos few-shot especifico de `categoria`. Si no
    hay ejemplos definidos para esa categoria, devuelve un bloque vacio en
    vez de fallar -- el prompt sigue siendo valido sin few-shot."""
    ejemplos = EJEMPLOS_POR_CATEGORIA.get(categoria)
    if not ejemplos:
        return ""

    lineas = [f'Ejemplos de "{categoria}" CONFIRMADOS:']
    lineas.extend(f'- "{texto}" -> confirma: true' for texto in ejemplos.get("positivos", []))
    lineas.append("")
    lineas.append(f'Ejemplos que NO son "{categoria}":')
    lineas.extend(f'- "{texto}" -> confirma: false' for texto in ejemplos.get("negativos", []))

    return "\n".join(lineas)


def _construir_prompt(texto: str, categoria: str) -> str:
    """Arma el prompt completo para verificar `texto` contra `categoria`,
    incluyendo unicamente los ejemplos few-shot y la instruccion de analisis
    de esa misma categoria (ver EJEMPLOS_POR_CATEGORIA e
    INSTRUCCIONES_ANALISIS_POR_CATEGORIA) -- nunca las de otra."""
    definicion = DEFINICIONES_CATEGORIA.get(categoria, categoria)
    bloque_ejemplos = _formatear_ejemplos(categoria)
    instruccion_analisis = INSTRUCCIONES_ANALISIS_POR_CATEGORIA.get(
        categoria, "Analiza el texto con cuidado antes de decidir."
    )

    return (
        "Eres un verificador de moderación de contenido para una app de reciclaje en México. "
        f'Un primer sistema marcó este texto como POSIBLEMENTE relacionado con la categoría "{categoria}", '
        "pero no está seguro. Tu tarea es evaluar el texto de forma independiente.\n\n"
        f"Categoría a evaluar: {categoria}\n"
        f"Definición: {definicion}\n\n"
        f"{bloque_ejemplos}\n\n"
        f'Texto a evaluar: "{texto}"\n\n'
        f"{instruccion_analisis}\n\n"
        "Responde ÚNICAMENTE con JSON válido, siguiendo este mapeo exacto:\n"
        f'- Usa "confirma": true SI Y SOLO SI el texto SÍ pertenece a la categoría "{categoria}" '
        "según la definición dada arriba.\n"
        f'- Usa "confirma": false SI el texto NO pertenece a la categoría "{categoria}".\n'
        '- El campo "razon" debe ser consistente con el valor de "confirma" -- si "confirma" '
        'es false, la razón debe explicar por qué NO aplica; si es true, debe explicar por qué '
        "SÍ aplica.\n"
        '- En el campo "razon", NO uses comillas dobles ni comillas simples dentro del texto.\n\n'
        '{"confirma": true o false, "razon": "una frase breve y específica"}'
    )


def verificar_con_qwen(texto: str, categoria: str, probabilidad: float) -> tuple:
    """Le pregunta a Qwen si `texto` realmente corresponde a `categoria`.

    Devuelve (confirma, razon):
      - confirma: True/False si Qwen dio un veredicto valido, None si no
        esta disponible o su respuesta no se pudo interpretar -- en ese
        caso el llamador debe conservar la decision original de BETO.
      - razon: explicacion breve (de Qwen, o del motivo del fallo).
    """
    if not esta_disponible():
        return None, "Qwen no disponible"

    try:
        prompt = _construir_prompt(texto, categoria)

        # DIAGNÓSTICO TEMPORAL -- deja ver exactamente qué generó el modelo
        print("=" * 60)
        print("ENTRADA DE QWEN:")
        print(prompt)
        print("=" * 60)

        mensajes = [{"role": "user", "content": prompt}]
        entrada = _tokenizer.apply_chat_template(
            mensajes,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = _tokenizer(entrada, return_tensors="pt").to(_model.device)

        with torch.no_grad():
            salida = _model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                temperature=0.1,
            )

        texto_generado = _tokenizer.decode(
            salida[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # DIAGNÓSTICO TEMPORAL -- deja ver exactamente qué generó el modelo
        print("=" * 60)
        print("SALIDA CRUDA DE QWEN:")
        print(texto_generado)
        print("=" * 60)

        confirma, razon = _parsear_respuesta(texto_generado)

        if confirma is not None and _detectar_posible_inconsistencia(confirma, razon):
            logger.warning(
                "Posible inconsistencia razonamiento/veredicto detectada "
                "(categoria=%s, texto=%r): confirma=%s pero la razon "
                "contiene lenguaje de negacion: %r",
                categoria, texto[:100], confirma, razon,
            )
            # NOTA: esto es solo auditoria -- NO se descarta el veredicto,
            # se retorna igual que siempre.
            _registrar_auditoria(categoria, texto, confirma, razon)

        return confirma, razon

    except Exception as exc:  # noqa: BLE001 - un fallo de Qwen nunca debe tumbar la API
        logger.warning("Verificacion con Qwen fallo para categoria=%s: %s", categoria, exc)
        return None, f"Verificacion con Qwen fallo: {exc}"
