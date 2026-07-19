"""Tests del worker de moderacion (src/worker/).

Mockean tanto la conexion AMQP (mensajes falsos con .ack()/.nack() como
AsyncMock, exchanges falsos con .publish() como AsyncMock) como el modelo
BETO (un MagicMock con .predict() devolviendo un ResultadoBeto fijo) --
no requieren RabbitMQ real corriendo ni el modelo BETO real cargado.

Fuerzan HABILITAR_QWEN=false (via monkeypatch de src.core.logic, no del
.env local) para que estos tests sean deterministas y no dependan de
GROQ_API_KEY ni de una llamada de red real.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import src.core.logic as logic_module
import src.worker.consumidor as consumidor_module
from src.models.beto_classifier import ResultadoBeto


def _beto_mock(probabilidades: dict, activaciones: dict) -> MagicMock:
    beto = MagicMock()
    beto.predict.return_value = ResultadoBeto(
        probabilidades=probabilidades,
        activaciones=activaciones,
        detectado_via_normalizacion={nombre: False for nombre in probabilidades},
        tiempo_inferencia_ms=1.0,
    )
    return beto


class _MensajeFalso:
    """Imita lo minimo de aio_pika.abc.AbstractIncomingMessage que usa
    _manejar_mensaje: .body, .headers, y .ack()/.nack() como coroutines."""

    def __init__(self, body: bytes, headers: dict | None = None):
        self.body = body
        self.headers = headers or {}
        self.ack = AsyncMock()
        self.nack = AsyncMock()


def _exchanges_falsos():
    exchange_principal = MagicMock()
    exchange_principal.publish = AsyncMock()
    exchange_fallidos = MagicMock()
    exchange_fallidos.publish = AsyncMock()
    return exchange_principal, exchange_fallidos


def test_mensaje_bien_formado_se_procesa_y_publica_el_resultado_esperado(monkeypatch):
    monkeypatch.setattr(logic_module, "esta_disponible", lambda: False)  # sin Qwen en este test

    beto = _beto_mock(
        probabilidades={"grosero": 0.9, "amenaza": 0.1, "inapropiado": 0.05},
        activaciones={"grosero": True, "amenaza": False, "inapropiado": False},
    )

    cuerpo = json.dumps(
        {
            "publicacion_id": "pub-123",
            "texto": "Pinches putos no me pagaron, este lugar es una mierda.",
            "campo": "resena",
        }
    ).encode("utf-8")
    mensaje = _MensajeFalso(cuerpo)
    exchange_principal, exchange_fallidos = _exchanges_falsos()

    asyncio.run(
        consumidor_module._manejar_mensaje(mensaje, beto, exchange_principal, exchange_fallidos)
    )

    beto.predict.assert_called_once()
    mensaje.ack.assert_awaited_once()
    mensaje.nack.assert_not_called()
    exchange_fallidos.publish.assert_not_called()

    exchange_principal.publish.assert_awaited_once()
    _args, kwargs = exchange_principal.publish.call_args
    mensaje_publicado = _args[0]
    assert kwargs["routing_key"] == consumidor_module.COLA_SALIDA

    cuerpo_publicado = json.loads(mensaje_publicado.body.decode("utf-8"))
    assert cuerpo_publicado["publicacion_id"] == "pub-123"
    assert cuerpo_publicado["bloqueado"] is True
    assert cuerpo_publicado["categorias"]["grosero"]["activado"] is True
    assert cuerpo_publicado["verificado_por_qwen"] is False
    assert "timestamp_procesado" in cuerpo_publicado


def test_mensaje_mal_formado_se_rechaza_sin_tumbar_el_worker():
    # Falta "publicacion_id" Y "texto" esta vacio -- ambos motivos de
    # rechazo por esquema en un solo caso.
    beto = MagicMock()

    cuerpo = json.dumps({"texto": ""}).encode("utf-8")
    mensaje = _MensajeFalso(cuerpo)
    exchange_principal, exchange_fallidos = _exchanges_falsos()

    # No debe lanzar -- el worker sigue vivo para el siguiente mensaje.
    asyncio.run(
        consumidor_module._manejar_mensaje(mensaje, beto, exchange_principal, exchange_fallidos)
    )

    beto.predict.assert_not_called()
    exchange_principal.publish.assert_not_called()
    exchange_fallidos.publish.assert_not_called()
    mensaje.nack.assert_awaited_once_with(requeue=False)
    mensaje.ack.assert_not_called()


def test_excepcion_durante_clasificacion_hace_nack_sin_reencolar(monkeypatch):
    monkeypatch.setattr(logic_module, "esta_disponible", lambda: False)

    beto = MagicMock()
    beto.predict.side_effect = RuntimeError("fallo simulado de inferencia")

    cuerpo = json.dumps({"publicacion_id": "pub-456", "texto": "cualquier texto"}).encode("utf-8")
    mensaje = _MensajeFalso(cuerpo)
    exchange_principal, exchange_fallidos = _exchanges_falsos()

    asyncio.run(
        consumidor_module._manejar_mensaje(mensaje, beto, exchange_principal, exchange_fallidos)
    )

    mensaje.nack.assert_awaited_once_with(requeue=False)
    mensaje.ack.assert_not_called()
    exchange_principal.publish.assert_not_called()
    exchange_fallidos.publish.assert_not_called()


def test_mensaje_que_agoto_los_reintentos_se_envia_a_fallidos_y_hace_ack(monkeypatch):
    # Simula, via el header x-death, que este mensaje ya paso
    # MAX_REINTENTOS veces por la cola de reintento -- en vez de
    # reintentar una vez mas, debe ir directo a COLA_FALLIDOS.
    monkeypatch.setattr(logic_module, "esta_disponible", lambda: False)

    beto = MagicMock()
    beto.predict.side_effect = RuntimeError("fallo persistente")

    cuerpo = json.dumps({"publicacion_id": "pub-789", "texto": "texto"}).encode("utf-8")
    headers = {
        "x-death": [
            {"queue": consumidor_module.COLA_REINTENTO, "count": consumidor_module.MAX_REINTENTOS}
        ]
    }
    mensaje = _MensajeFalso(cuerpo, headers=headers)
    exchange_principal, exchange_fallidos = _exchanges_falsos()

    asyncio.run(
        consumidor_module._manejar_mensaje(mensaje, beto, exchange_principal, exchange_fallidos)
    )

    exchange_fallidos.publish.assert_awaited_once()
    _args, kwargs = exchange_fallidos.publish.call_args
    assert kwargs["routing_key"] == consumidor_module.COLA_FALLIDOS

    mensaje.ack.assert_awaited_once()
    mensaje.nack.assert_not_called()


def test_contar_reintentos_sin_header_x_death_es_cero():
    mensaje = _MensajeFalso(b"{}")
    assert consumidor_module._contar_reintentos(mensaje) == 0


def test_contar_reintentos_lee_el_count_de_la_cola_de_reintento():
    mensaje = _MensajeFalso(
        b"{}",
        headers={
            "x-death": [
                {"queue": "otra-cola", "count": 99},
                {"queue": consumidor_module.COLA_REINTENTO, "count": 3},
            ]
        },
    )
    assert consumidor_module._contar_reintentos(mensaje) == 3
