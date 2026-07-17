"""Endpoints de moderacion de contenido."""

import logging
import time

from fastapi import APIRouter, Request

from src.core.logic import moderar_texto
from src.models.qwen_verifier import esta_disponible
from src.schemas.moderacion_schema import (
    HealthResponse,
    ModerarTextoRequest,
    ModerarTextoResponse,
)
from src.utils.normalizacion import limpiar_espacios, truncar_para_log

logger = logging.getLogger("moderacion.routes")

router = APIRouter()


@router.post("/moderar", response_model=ModerarTextoResponse)
def moderar(request: Request, payload: ModerarTextoRequest) -> ModerarTextoResponse:
    texto_limpio = limpiar_espacios(payload.texto)

    inicio = time.perf_counter()
    resultado = moderar_texto(texto_limpio, request.app.state.beto)
    tiempo_total_ms = (time.perf_counter() - inicio) * 1000

    logger.info(
        "POST /moderar campo=%s texto=%r bloqueado=%s verificado_por_qwen=%s tiempo_total_ms=%.1f",
        payload.campo,
        truncar_para_log(texto_limpio),
        resultado["bloqueado"],
        resultado["verificado_por_qwen"],
        tiempo_total_ms,
    )

    return ModerarTextoResponse(
        texto=resultado["texto"],
        bloqueado=resultado["bloqueado"],
        categorias=resultado["categorias"],
        verificado_por_qwen=resultado["verificado_por_qwen"],
        detalle_verificacion=resultado["detalle_verificacion"],
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    beto_cargado = getattr(request.app.state, "beto", None) is not None

    return HealthResponse(
        status="ok" if beto_cargado else "modelo_no_cargado",
        modelo_cargado=beto_cargado,
        qwen_habilitado=esta_disponible(),
    )
