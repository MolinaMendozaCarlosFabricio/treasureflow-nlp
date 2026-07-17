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

import torch

from src.core.config import HABILITAR_QWEN, HF_TOKEN, QWEN_MODEL_NAME

logger = logging.getLogger("moderacion.qwen")

DEFINICIONES_CATEGORIA = {
    "grosero": "contiene vocabulario vulgar o insultos directos con intención de ofender",
    "amenaza": "expresa intención explícita o implícita de causar daño a una persona o negocio",
    "inapropiado": "contiene insinuación sexual o doble sentido, sin necesidad de vulgaridad explícita",
}

_JSON_BLOQUE = re.compile(r"\{.*?\}", re.DOTALL)

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
    JSON). Devuelve (None, "no se pudo interpretar la respuesta") si no se
    puede extraer un veredicto valido."""
    coincidencia = _JSON_BLOQUE.search(texto_generado)
    if coincidencia is None:
        return None, "no se pudo interpretar la respuesta"

    try:
        resultado = json.loads(coincidencia.group(0))
    except json.JSONDecodeError:
        return None, "no se pudo interpretar la respuesta"

    confirma = resultado.get("confirma")
    if not isinstance(confirma, bool):
        return None, "no se pudo interpretar la respuesta"

    razon = str(resultado.get("razon", "")).strip()
    return confirma, razon


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
        definicion = DEFINICIONES_CATEGORIA.get(categoria, categoria)
        prompt = ("""Eres un verificador de moderación de contenido para una app de reciclaje en México. Un primer sistema marcó este texto como POSIBLEMENTE relacionado con la categoría "{categoria}", pero no está seguro. Tu tarea es evaluar el texto de forma independiente.

Categoría a evaluar: {categoria}
Definición: {definicion}

Ejemplos de "inapropiado" CONFIRMADOS:
- "¿La persona que atiende los martes es la misma que sale en la foto de perfil? Porque si es así, ya sé por qué vengo tan seguido." -> confirma: true
- "Buen servicio de pesaje, aunque la báscula no fue lo único que me dejó con ganas de volver mañana mismo." -> confirma: true

Ejemplos que NO son "inapropiado":
- "Ofrezco material en buen estado, disponible todo el día" -> confirma: false
- "Vivo sola así que pueden pasar cuando gusten a recoger el material, nada más avisen antes por mensaje." -> confirma: false

Texto a evaluar: "{texto}"

Analiza el texto con cuidado -- presta atención a insinuaciones,
dobles sentidos, o frases que sugieran algo más allá de su significado
literal. Responde ÚNICAMENTE con JSON:
{{"confirma": true o false, "razon": "una frase breve y ESPECÍFICA sobre qué parte del texto sustenta tu decisión"}}
"""
        ).format(categoria=categoria, definicion=definicion, texto=texto)

        mensajes = [{"role": "user", "content": prompt}]
        entrada = _tokenizer.apply_chat_template(
            mensajes,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = _tokenizer(entrada, return_tensors="pt").to(_model.device)

        with torch.no_grad():
            salida = _model.generate(
                **inputs,
                max_new_tokens=200,
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

        return _parsear_respuesta(texto_generado)

    except Exception as exc:  # noqa: BLE001 - un fallo de Qwen nunca debe tumbar la API
        logger.warning("Verificacion con Qwen fallo para categoria=%s: %s", categoria, exc)
        return None, f"Verificacion con Qwen fallo: {exc}"
