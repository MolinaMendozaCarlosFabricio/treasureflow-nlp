"""Logica de orquestacion de la moderacion de contenido.

Corre BETO, decide si alguna categoria cayo en zona dudosa (umbral +/-
MARGEN_ZONA_DUDOSA) y en ese caso consulta a Qwen para confirmar o
descartar esa categoria puntual, sin tocar las que ya son claras.
"""

import logging

from src.core.config import LABEL_COLUMNS, MARGEN_ZONA_DUDOSA, UMBRALES_POR_CATEGORIA
from src.models.beto_classifier import ClasificadorBeto
from src.models.qwen_verifier import VerificadorQwen

logger = logging.getLogger("moderacion.logic")


def en_zona_dudosa(probabilidad: float, umbral: float) -> bool:
    return abs(probabilidad - umbral) <= MARGEN_ZONA_DUDOSA


def moderar_texto(texto: str, beto: ClasificadorBeto, qwen: VerificadorQwen) -> dict:
    resultado_beto = beto.predict(texto)

    categorias = {
        nombre: {
            "probabilidad": resultado_beto.probabilidades[nombre],
            "activado": resultado_beto.activaciones[nombre],
        }
        for nombre in LABEL_COLUMNS
    }

    detalle_verificacion = []
    verificado_por_qwen = False

    if qwen.disponible():
        for nombre in LABEL_COLUMNS:
            probabilidad = resultado_beto.probabilidades[nombre]
            umbral = UMBRALES_POR_CATEGORIA[nombre]

            if not en_zona_dudosa(probabilidad, umbral):
                continue

            veredicto = qwen.verificar(texto, nombre, probabilidad)

            # Solo sobreescribimos la decision de BETO si Qwen realmente
            # respondio -- si fallo, nos quedamos con la decision por umbral.
            if veredicto["exito"]:
                categorias[nombre]["activado"] = veredicto["confirma"]
                verificado_por_qwen = True

            detalle_verificacion.append(
                {
                    "categoria": nombre,
                    "probabilidad_beto": probabilidad,
                    "confirma": veredicto["confirma"],
                    "razon": veredicto["razon"],
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
