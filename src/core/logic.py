"""Logica de orquestacion de la moderacion de contenido.

Corre BETO, decide si alguna categoria cayo en su zona dudosa (umbral +/-
los margenes asimetricos de esa categoria en MARGENES_POR_CATEGORIA) y en
ese caso consulta a Qwen para confirmar o descartar esa categoria
puntual, sin tocar las que ya son claras.
"""

import logging

from src.core.config import LABEL_COLUMNS, MARGENES_POR_CATEGORIA, UMBRALES_POR_CATEGORIA
from src.models.beto_classifier import ClasificadorBeto
from src.models.qwen_verifier import esta_disponible, verificar_con_qwen

logger = logging.getLogger("moderacion.logic")


def esta_en_zona_dudosa(probabilidad: float, umbral: float, categoria: str) -> bool:
    margenes = MARGENES_POR_CATEGORIA[categoria]
    limite_inferior = umbral - margenes["inferior"]
    limite_superior = umbral + margenes["superior"]
    return limite_inferior <= probabilidad <= limite_superior


async def moderar_texto(texto: str, beto: ClasificadorBeto) -> dict:
    resultado_beto = beto.predict(texto)

    categorias = {
        nombre: {
            "probabilidad": resultado_beto.probabilidades[nombre],
            "activado": resultado_beto.activaciones[nombre],
            "detectado_via_normalizacion": resultado_beto.detectado_via_normalizacion[nombre],
        }
        for nombre in LABEL_COLUMNS
    }

    detalle_verificacion = []
    verificado_por_qwen = False

    if esta_disponible():
        for nombre in LABEL_COLUMNS:
            probabilidad = resultado_beto.probabilidades[nombre]
            umbral = UMBRALES_POR_CATEGORIA[nombre]

            if not esta_en_zona_dudosa(probabilidad, umbral, nombre):
                continue

            confirma, razon = await verificar_con_qwen(texto, nombre, probabilidad)

            # Solo sobreescribimos la decision de BETO si Qwen dio un
            # veredicto valido -- si confirma es None (no disponible o
            # respuesta no interpretable), nos quedamos con la decision
            # original por umbral.
            if confirma is not None:
                categorias[nombre]["activado"] = confirma
                verificado_por_qwen = True

            detalle_verificacion.append(
                {
                    "categoria": nombre,
                    "probabilidad_beto": probabilidad,
                    "confirma": confirma,
                    "razon": razon,
                }
            )

    bloqueado = any(categorias[nombre]["activado"] for nombre in LABEL_COLUMNS)

    logger.debug(
        "BETO: bloqueado=%s tiempo_beto_ms=%.1f qwen_uso=%s",
        bloqueado,
        resultado_beto.tiempo_inferencia_ms,
        verificado_por_qwen,
    )

    return {
        "texto": texto,
        "bloqueado": bloqueado,
        "categorias": categorias,
        "verificado_por_qwen": verificado_por_qwen,
        "detalle_verificacion": detalle_verificacion,
    }
