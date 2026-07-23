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

# Paso 4 (ya no hay lowercase global, ver normalizar_texto): mapeo
# leetspeak NO ambiguo. Se aplica por token (separado por espacios) y solo
# si el token mezcla letras con alguno de estos caracteres -- un token
# puramente numerico o simbolico (fechas, cantidades, precios) se deja
# intacto. Los digitos/simbolos no tienen "caso" propio, asi que la
# sustitucion respeta si el token esta en MAYUSCULAS (ej. "C4BR0N" ->
# "CABRON") o no (por defecto, minuscula: "c4br0n" -> "cabron").
_MAPEO_LEETSPEAK = {
    "0": "o",
    "3": "e",
    "4": "a",
    "7": "t",
    "@": "a",
    "$": "s",
}
_MAPEO_LEETSPEAK_MAYUSCULAS = {clave: valor.upper() for clave, valor in _MAPEO_LEETSPEAK.items()}


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
        mapeo = _MAPEO_LEETSPEAK_MAYUSCULAS if token.isupper() else _MAPEO_LEETSPEAK
        return "".join(mapeo.get(caracter, caracter) for caracter in token)

    return " ".join(_sustituir_token(token) for token in texto.split())


def _tiene_digito_y_letra_mezclados(texto: str) -> bool:
    def _token_mezcla(token: str) -> bool:
        return any(caracter.isalpha() for caracter in token) and any(
            caracter.isdigit() for caracter in token
        )

    return any(_token_mezcla(token) for token in texto.split())


def texto_parece_ofuscado(texto: str) -> bool:
    """Heuristica para decidir si vale la pena pagar el costo de una segunda
    pasada por el modelo con normalizar_texto() (ver beto_classifier.py).

    Se detecto (docs/prueba_1.txt) que normalizar SIEMPRE, incluso texto
    limpio sin ninguna senal de ofuscacion, y tomar el maximo entre ambas
    versiones amplificaba falsos positivos cuando el modelo reaccionaba de
    forma inesperada a la version normalizada (ej. "Botellas de plástico"
    -> "grosero" 0.07 original vs 0.96 normalizado, sin que el texto
    tuviera nada de ofuscado). Por eso ahora solo se corre la segunda
    pasada si el texto realmente muestra alguna señal de ofuscacion
    intencional:

    - Digitos mezclados con letras dentro de la misma palabra (ej.
      "3stup1do", "pvt0s").
    - 3+ letras individuales seguidas separadas por espacio, punto o guion
      (ej. "p u t o", "p.u.t.o") -- mismo patron que usa la normalizacion
      para colapsar separadores intercalados.
    - 3+ repeticiones consecutivas del mismo caracter (ej. "puuuuuto").
    """
    return (
        _tiene_digito_y_letra_mezclados(texto)
        or bool(_SEPARADORES_INTERCALADOS.search(texto))
        or bool(_REPETICIONES.search(texto))
    )


def normalizar_texto(texto: str) -> str:
    """Normalizacion lexica para robustecer la moderacion frente a
    ofuscacion de palabras prohibidas. Aplica, en orden:

    1. NFKD + eliminacion de diacriticos (acentos, algunos homoglifos).
    2. Colapso de letras repetidas 3+ veces seguidas -> 2.
    3. Colapso de letras individuales separadas por '.', '-' o espacio.
    4. Sustitucion leetspeak (0/3/4/7/@/$) solo en tokens que mezclan
       letras con esos caracteres.

    NO fuerza lowercase (a diferencia de un diseno anterior): se detecto
    que el modelo BETO usado para clasificar es sensible a mayusculas/
    minusculas de una forma inesperada -- texto normal correctamente
    capitalizado, al forzarlo a minusculas, podia dispararle una
    probabilidad de "grosero" muchisimo mas alta sin relacion con el
    contenido real (ej. "Botellas de plástico" -> "grosero" bajo con
    mayuscula, pero >0.75 en minusculas). Ninguna de las reglas de abajo
    necesita lowercase para funcionar (todas usan clases de regex
    case-insensitive o comparan caracteres individuales), asi que
    quitarlo elimina esa amplificacion de falsos positivos sin debilitar
    la deteccion de ofuscacion.
    """
    texto = _quitar_diacriticos(texto)
    texto = _colapsar_repeticiones(texto)
    texto = _colapsar_separadores_intercalados(texto)
    texto = _sustituir_leetspeak(texto)
    return texto
