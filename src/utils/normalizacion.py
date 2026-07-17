"""Utilidades de normalizacion de texto para la API de moderacion.

Importante: el modelo BETO se entreno con el texto crudo, sin lowercase ni
eliminacion de acentos (ver tokenizacion en el notebook de entrenamiento)
-- por eso aqui solo se hacen limpiezas de espacios en blanco, nunca
cambios que alteren la distribucion de texto que vio el modelo durante el
entrenamiento.
"""

import re

_ESPACIOS_MULTIPLES = re.compile(r"\s+")


def limpiar_espacios(texto: str) -> str:
    """Colapsa espacios/saltos de linea repetidos y recorta los bordes."""
    return _ESPACIOS_MULTIPLES.sub(" ", texto).strip()


def truncar_para_log(texto: str, max_caracteres: int = 80) -> str:
    """Trunca el texto para no volcar contenido completo de usuarios en logs."""
    texto_limpio = texto.strip()
    if len(texto_limpio) <= max_caracteres:
        return texto_limpio
    return f"{texto_limpio[:max_caracteres]}... [{len(texto_limpio)} caracteres]"
