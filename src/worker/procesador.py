"""Logica de clasificacion de un mensaje individual del worker de
moderacion. Reutiliza moderar_texto() (src/core/logic.py) -- la misma
funcion que ya usa el endpoint POST /moderar de la API principal -- para
no duplicar la logica de umbrales/zona dudosa/Qwen aqui."""

import logging

from src.core.logic import moderar_texto
from src.models.beto_classifier import ClasificadorBeto
from src.worker.esquemas_mensajes import MensajeEntrada, MensajeSalida

logger = logging.getLogger("moderacion.worker.procesador")


async def procesar_mensaje(mensaje: MensajeEntrada, beto: ClasificadorBeto) -> MensajeSalida:
    resultado = await moderar_texto(mensaje.texto, beto)

    logger.info(
        "publicacion_id=%s campo=%s bloqueado=%s verificado_por_qwen=%s",
        mensaje.publicacion_id,
        mensaje.campo,
        resultado["bloqueado"],
        resultado["verificado_por_qwen"],
    )

    return MensajeSalida(
        publicacion_id=mensaje.publicacion_id,
        campo=mensaje.campo,
        bloqueado=resultado["bloqueado"],
        categorias=resultado["categorias"],
        verificado_por_qwen=resultado["verificado_por_qwen"],
        detalle_verificacion=resultado["detalle_verificacion"],
    )
