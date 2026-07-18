"""Tests unitarios de la normalizacion lexica anti-ofuscacion
(src/utils/normalizacion.py::normalizar_texto), cubriendo cada regla por
separado, combinaciones, y casos negativos (texto normal que no debe
alterarse mas alla de la eliminacion de acentos del paso 1)."""

from src.utils.normalizacion import normalizar_texto


# --- Paso 1: NFKD + eliminacion de diacriticos -----------------------------


def test_elimina_acentos():
    assert normalizar_texto("café") == "cafe"


def test_elimina_acentos_en_varias_vocales():
    assert normalizar_texto("áéíóú") == "aeiou"


def test_normaliza_ene_con_tilde():
    # NFKD descompone "ñ" en "n" + tilde combinante, que se descarta --
    # comportamiento esperado segun el diseño pedido (resolver homoglifos
    # basicos), aunque linguisticamente "ñ" no sea solo una "n" acentuada.
    assert normalizar_texto("año") == "ano"


# --- Paso 2: lowercase ------------------------------------------------------


def test_convierte_a_minusculas():
    assert normalizar_texto("HOLA") == "hola"


def test_minusculas_con_mayusculas_mezcladas():
    assert normalizar_texto("HoLa MuNdO") == "hola mundo"


# --- Paso 3: colapso de repeticiones ----------------------------------------


def test_colapsa_repeticiones_de_tres_o_mas():
    assert normalizar_texto("puuutooo") == "puutoo"


def test_colapsa_repeticiones_largas():
    assert normalizar_texto("holaaaaaa") == "holaa"


def test_no_colapsa_dobles_legitimos():
    # "carro" tiene una "rr" legitima (2 repeticiones) -- el umbral es 3+,
    # no debe tocarse.
    assert normalizar_texto("carro") == "carro"


# --- Paso 4: separadores intercalados ---------------------------------------


def test_colapsa_letras_separadas_por_puntos():
    assert normalizar_texto("p.u.t.o") == "puto"


def test_colapsa_letras_separadas_por_espacios():
    assert normalizar_texto("p u t o") == "puto"


def test_colapsa_letras_separadas_por_guiones():
    assert normalizar_texto("p-u-t-o") == "puto"


def test_colapsa_letras_separadas_dentro_de_una_oracion():
    assert normalizar_texto("eres un p u t o de verdad") == "eres un puto de verdad"


def test_no_colapsa_puntuacion_normal_de_una_oracion():
    # Ninguna palabra real de 3+ letras debe partirse letra por letra.
    assert normalizar_texto("voy a ver a mi tio") == "voy a ver a mi tio"


# --- Paso 5: leetspeak no ambiguo (solo en tokens mixtos) -------------------


def test_sustituye_leetspeak_en_token_mixto():
    assert normalizar_texto("p3nd3jo") == "pendejo"


def test_sustituye_leetspeak_cuatro_y_cero():
    assert normalizar_texto("c4br0n") == "cabron"


def test_sustituye_arroba_como_a():
    assert normalizar_texto("@lgo") == "algo"


def test_no_sustituye_numero_suelto():
    # "3" no mezcla letras con caracteres leet -- no debe tocarse.
    assert normalizar_texto("tengo 3 hijos") == "tengo 3 hijos"


def test_no_sustituye_precio_con_simbolo_de_moneda():
    # "$50" no tiene ninguna letra -- no debe tocarse.
    assert normalizar_texto("cuesta $50") == "cuesta $50"


# --- Casos negativos: texto normal que no debe alterarse -------------------


def test_texto_normal_sin_acentos_queda_igual():
    assert normalizar_texto("hola, como estas") == "hola, como estas"


def test_texto_normal_con_acentos_solo_pierde_el_acento():
    assert normalizar_texto("hola, buenos días") == "hola, buenos dias"


# --- Casos combinados -------------------------------------------------------


def test_combinado_mayusculas_acentos_y_repeticiones():
    assert normalizar_texto("PÚÚÚTOOO") == "puutoo"


def test_combinado_repeticion_y_leetspeak():
    # El colapso de repeticiones (paso 3) corre antes que el leetspeak
    # (paso 5): "joooo" -> "joo", y luego 3->e en el token completo.
    assert normalizar_texto("P3ND3JOOOO") == "pendejoo"


def test_combinado_espaciado_dentro_de_frase_ofuscada():
    assert normalizar_texto("que persona tan p u t a") == "que persona tan puta"
