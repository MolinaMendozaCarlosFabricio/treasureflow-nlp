"""Esquemas Pydantic para los mensajes AMQP del worker de moderacion.

MensajeEntrada valida lo que llega en COLA_ENTRADA (moderacion.pendiente,
publicado por la API principal). MensajeSalida es lo que este worker
publica en COLA_SALIDA (moderacion.resultado) para que la API principal
actualice el estado de la publicacion/resena y dispare la notificacion
FCM -- este worker nunca toca la base de datos ni FCM directamente.

Reutiliza ResultadoCategoria y DetalleVerificacionQwen de
src.schemas.moderacion_schema en vez de redefinirlas, para no duplicar la
forma de esos sub-objetos entre la API HTTP y el worker.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.schemas.moderacion_schema import DetalleVerificacionQwen, ResultadoCategoria


class MensajeEntrada(BaseModel):
    publicacion_id: str = Field(..., min_length=1)
    texto: str = Field(..., min_length=1)
    campo: Optional[Literal["resena", "descripcion"]] = None

    @field_validator("publicacion_id")
    @classmethod
    def publicacion_id_no_vacio(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("publicacion_id no puede estar vacio ni compuesto solo por espacios.")
        return valor

    @field_validator("texto")
    @classmethod
    def texto_no_vacio(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("El texto no puede estar vacio ni compuesto solo por espacios.")
        return valor


class MensajeSalida(BaseModel):
    publicacion_id: str
    bloqueado: bool
    categorias: dict[str, ResultadoCategoria]
    verificado_por_qwen: bool
    detalle_verificacion: list[DetalleVerificacionQwen] = Field(default_factory=list)
    timestamp_procesado: str = Field(default_factory=lambda: datetime.now().isoformat())
