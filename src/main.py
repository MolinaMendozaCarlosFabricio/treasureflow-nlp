"""Punto de entrada de la API de moderacion de contenido (FastAPI).

Levantar con: uvicorn src.main:app --reload (desde la raiz del repo).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.core.config import HABILITAR_QWEN
from src.models.beto_classifier import ClasificadorBeto
from src.models.qwen_verifier import VerificadorQwen
from src.routes.moderacion import router as moderacion_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("moderacion.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cargando modelo BETO...")
    app.state.beto = ClasificadorBeto()

    app.state.qwen = VerificadorQwen()
    if HABILITAR_QWEN:
        logger.info("Verificacion con Qwen habilitada (el modelo se carga en la primera consulta que la necesite).")
    else:
        logger.info("Verificacion con Qwen deshabilitada (HABILITAR_QWEN=false).")

    yield

    logger.info("Apagando la API de moderacion.")


app = FastAPI(
    title="TreasureFlow NLP - API de moderacion de contenido",
    description=(
        "Sirve el modelo BETO fine-tuneado para moderacion multilabel "
        "(grosero / amenaza / inapropiado), con verificacion opcional via Qwen "
        "para casos en zona dudosa."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def manejar_error_validacion(request: Request, exc: RequestValidationError) -> JSONResponse:
    mensajes = [error["msg"] for error in exc.errors()]
    return JSONResponse(
        status_code=422,
        content={"detail": "Texto invalido: " + "; ".join(mensajes)},
    )


app.include_router(moderacion_router)
