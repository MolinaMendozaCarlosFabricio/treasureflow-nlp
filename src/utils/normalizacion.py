"""Utilidades de normalizacion de texto para la API de moderacion.

Importante: el modelo BETO se entreno con el texto crudo, sin lowercase ni
eliminacion de acentos (ver tokenizacion en el notebook de entrenamiento)
-- por eso limpiar_espacios/truncar_para_log solo hacen limpiezas de
espacios en blanco, nunca cambios que alteren la distribucion de texto
que vio el modelo durante el entrenamiento.

normalizar_texto(), en cambio, es deliberadamente agresiva: existe para
neutralizar intentos de ofuscar palabras prohibidas (leetspeak, espaciado,
separadores intercalados), NO para reemplazar el texto original. Por eso
beto_classifier.py evalua ambas versiones (original y normalizada) y se
queda con la probabilidad mas alta por categoria, en vez de normalizar
siempre antes de tokenizar.
"""

import re
import unicodedata

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


# --- normalizar_texto: robustez frente a ofuscacion de palabras ------------

# Paso 3: cualquier letra repetida 3+ veces seguidas se reduce a 2
# (ej. "puuutooo" -> "puutoo").
_REPETICIONES = re.compile(r"([a-zA-Z])\1{2,}")

# Paso 4: secuencias de al menos 3 letras individuales separadas por punto,
# guion o espacio (ej. "p.u.t.o" o "p u t o"). El limite de 3+ letras (dos
# pares "letra+separador" mas una letra final) es a proposito, para NO
# tocar puntuacion normal de una oracion (una palabra real de 3+ letras
# nunca queda partida letra por letra con separadores en medio).
_SEPARADORES_INTERCALADOS = re.compile(r"\b(?:[a-zA-Z][ .\-]){2,}[a-zA-Z]\b")
_SEPARADOR_SUELTO = re.compile(r"[ .\-]")

# Paso 5: mapeo leetspeak NO ambiguo. Se aplica por token (separado por
# espacios) y solo si el token mezcla letras con alguno de estos
# caracteres -- un token puramente numerico o simbolico (fechas,
# cantidades, precios) se deja intacto.
_MAPEO_LEETSPEAK = {
    "0": "o",
    "3": "e",
    "4": "a",
    "7": "t",
    "@": "a",
    "$": "s",
}


def _quitar_diacriticos(texto: str) -> str:
    """NFKD + descarte de marcas diacriticas combinantes, para resolver
    acentos y algunos homoglifos basicos (ej. "café" -> "cafe")."""
    texto_nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(caracter for caracter in texto_nfkd if not unicodedata.combining(caracter))


def _colapsar_repeticiones(texto: str) -> str:
    return _REPETICIONES.sub(r"\1\1", texto)


def _colapsar_separadores_intercalados(texto: str) -> str:
    def _colapsar(coincidencia: re.Match) -> str:
        return _SEPARADOR_SUELTO.sub("", coincidencia.group(0))

    return _SEPARADORES_INTERCALADOS.sub(_colapsar, texto)


def _token_mezcla_letras_y_leetspeak(token: str) -> bool:
    tiene_letra = any(caracter.isalpha() for caracter in token)
    tiene_leet = any(caracter in _MAPEO_LEETSPEAK for caracter in token)
    return tiene_letra and tiene_leet


def _sustituir_leetspeak(texto: str) -> str:
    def _sustituir_token(token: str) -> str:
        if not _token_mezcla_letras_y_leetspeak(token):
            return token
        return "".join(_MAPEO_LEETSPEAK.get(caracter, caracter) for caracter in token)

    return " ".join(_sustituir_token(token) for token in texto.split())


def normalizar_texto(texto: str) -> str:
    """Normalizacion lexica para robustecer la moderacion frente a
    ofuscacion de palabras prohibidas. Aplica, en orden:

    1. NFKD + eliminacion de diacriticos (acentos, algunos homoglifos).
    2. lowercase.
    3. Colapso de letras repetidas 3+ veces seguidas -> 2.
    4. Colapso de letras individuales separadas por '.', '-' o espacio.
    5. Sustitucion leetspeak (0/3/4/7/@/$) solo en tokens que mezclan
       letras con esos caracteres.
    """
    texto = _quitar_diacriticos(texto)
    texto = texto.lower()
    texto = _colapsar_repeticiones(texto)
    texto = _colapsar_separadores_intercalados(texto)
    texto = _sustituir_leetspeak(texto)
    return texto
