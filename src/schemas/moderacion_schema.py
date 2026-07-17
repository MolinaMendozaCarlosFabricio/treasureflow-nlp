"""Esquemas Pydantic para el endpoint de moderacion de contenido."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.core.config import MAX_LONGITUD_TEXTO


class ModerarTextoRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=1,
        max_length=MAX_LONGITUD_TEXTO,
        description="Texto a evaluar. No puede estar vacio ni superar "
        f"{MAX_LONGITUD_TEXTO} caracteres.",
    )
    campo: Optional[Literal["resena", "descripcion"]] = Field(
        default=None,
        description="Solo para logs/trazabilidad, no afecta la decision.",
    )

    @field_validator("texto")
    @classmethod
    def texto_no_vacio(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("El texto no puede estar vacio ni compuesto solo por espacios.")
        return valor


class ResultadoCategoria(BaseModel):
    probabilidad: float
    activado: bool


class DetalleVerificacionQwen(BaseModel):
    categoria: str
    probabilidad_beto: float
    confirma: Optional[bool]
    razon: str


class ModerarTextoResponse(BaseModel):
    texto: str
    bloqueado: bool
    categorias: dict[str, ResultadoCategoria]
    verificado_por_qwen: bool
    detalle_verificacion: list[DetalleVerificacionQwen] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    modelo_cargado: bool
    qwen_habilitado: bool
