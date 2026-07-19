"""Consumidor AMQP (worker) que recibe publicaciones/reseñas pendientes de
moderacion desde RabbitMQ (CloudAMQP), las clasifica reutilizando la misma
logica que la API principal (BETO + Qwen via Groq si la probabilidad cae
en zona dudosa), y publica el veredicto de vuelta para que la API
principal actualice el estado en su base de datos y dispare la
notificacion FCM.

Este worker NUNCA toca una base de datos ni Firebase Cloud Messaging --
esa es responsabilidad exclusiva de la API principal. Su unica
responsabilidad es: consumir -> clasificar -> publicar el resultado ->
ack (o nack sin reencolar si algo falla).

--- Topologia de colas -----------------------------------------------------

Exchange principal NOMBRE_EXCHANGE ("moderacion"), tipo DIRECT: no
necesitamos routing por patrones (eso es lo que ofrece un exchange topic,
con comodines "."/"*"/"#") porque cada cola cuelga de un routing key
exacto y fijo -- direct es la opcion mas simple que cubre el caso de uso
tal cual esta planteado.

  - COLA_ENTRADA ("moderacion.pendiente"): la API principal publica aqui.
    Su "x-dead-letter-exchange" apunta al exchange de reintento (ver
    abajo), asi que cualquier nack sin reencolar cae ahi automaticamente.
  - COLA_SALIDA ("moderacion.resultado"): este worker publica el
    veredicto aqui; la API principal la consume (fuera del alcance de
    este repo).
  - COLA_FALLIDOS ("moderacion.fallidos"): destino final tras agotar
    MAX_REINTENTOS. Tiene su propio exchange dedicado (mismo nombre) para
    poder publicar ahi explicitamente desde el codigo, no solo via DLX.

Reintentos (patron estandar TTL + dead-letter-exchange que reencola, el
mecanismo mas simple soportado nativamente por RabbitMQ sin plugins
extra): un mensaje que falla se nackea SIN reencolar -> por el
"x-dead-letter-exchange" de COLA_ENTRADA cae al exchange de reintento ->
una cola de reintento dedicada (con "x-message-ttl") lo retiene
TTL_REINTENTO_MS -> al expirar el TTL, su propio "x-dead-letter-exchange"
lo manda de vuelta al exchange principal con el routing key de
COLA_ENTRADA -> se reprocesa. RabbitMQ agrega automaticamente un header
"x-death" en cada ciclo con, entre otros campos, "queue" y "count" -- lo
usamos para contar cuantas veces ya paso por la cola de reintento; al
llegar a MAX_REINTENTOS, en vez de reencolar una vez mas lo publicamos
explicitamente en COLA_FALLIDOS y hacemos ack del mensaje original (para
sacarlo del flujo de reintentos de forma definitiva).
"""

import asyncio
import json
import logging
import signal

import aio_pika
from aio_pika import ExchangeType, Message
from aio_pika.abc import AbstractExchange, AbstractIncomingMessage, AbstractQueue
from pydantic import ValidationError

from src.core.config import (
    AMQP_URL,
    COLA_ENTRADA,
    COLA_FALLIDOS,
    COLA_SALIDA,
    MAX_REINTENTOS,
    NOMBRE_EXCHANGE,
    PREFETCH_COUNT,
    TTL_REINTENTO_MS,
)
from src.models.beto_classifier import ClasificadorBeto
from src.worker.esquemas_mensajes import MensajeEntrada
from src.worker.procesador import procesar_mensaje

logger = logging.getLogger("moderacion.worker")

# Exchange/cola intermedios del patron de reintento -- detalle de
# implementacion, no forman parte del contrato con la API principal (que
# solo conoce COLA_ENTRADA y COLA_SALIDA).
NOMBRE_EXCHANGE_REINTENTO = f"{NOMBRE_EXCHANGE}.reintento"
COLA_REINTENTO = f"{COLA_ENTRADA}.reintento"


def _contar_reintentos(mensaje: AbstractIncomingMessage) -> int:
    """Cuenta cuantas veces este mensaje ya paso por COLA_REINTENTO,
    leyendo el header "x-death" que RabbitMQ agrega automaticamente cada
    vez que un mensaje se dead-letterea desde una cola."""
    x_death = (mensaje.headers or {}).get("x-death")
    if not x_death:
        return 0

    for entrada in x_death:
        if entrada.get("queue") == COLA_REINTENTO:
            return int(entrada.get("count", 0))
    return 0


async def _declarar_topologia(canal: aio_pika.abc.AbstractChannel):
    """Declara exchanges/colas/bindings. declare_* es idempotente (no
    falla si ya existen con los mismos argumentos), asi que se puede
    correr en cada arranque del worker sin coordinarse con nadie mas."""
    exchange_principal = await canal.declare_exchange(
        NOMBRE_EXCHANGE, ExchangeType.DIRECT, durable=True
    )
    exchange_reintento = await canal.declare_exchange(
        NOMBRE_EXCHANGE_REINTENTO, ExchangeType.DIRECT, durable=True
    )
    exchange_fallidos = await canal.declare_exchange(
        COLA_FALLIDOS, ExchangeType.DIRECT, durable=True
    )

    cola_entrada = await canal.declare_queue(
        COLA_ENTRADA,
        durable=True,
        arguments={"x-dead-letter-exchange": NOMBRE_EXCHANGE_REINTENTO},
    )
    await cola_entrada.bind(exchange_principal, routing_key=COLA_ENTRADA)

    cola_reintento = await canal.declare_queue(
        COLA_REINTENTO,
        durable=True,
        arguments={
            "x-dead-letter-exchange": NOMBRE_EXCHANGE,
            "x-dead-letter-routing-key": COLA_ENTRADA,
            "x-message-ttl": TTL_REINTENTO_MS,
        },
    )
    await cola_reintento.bind(exchange_reintento, routing_key=COLA_ENTRADA)

    cola_salida = await canal.declare_queue(COLA_SALIDA, durable=True)
    await cola_salida.bind(exchange_principal, routing_key=COLA_SALIDA)

    cola_fallidos = await canal.declare_queue(COLA_FALLIDOS, durable=True)
    await cola_fallidos.bind(exchange_fallidos, routing_key=COLA_FALLIDOS)

    return cola_entrada, exchange_principal, exchange_fallidos


async def _enviar_a_fallidos_o_reintentar(
    mensaje: AbstractIncomingMessage,
    exchange_fallidos: AbstractExchange,
    cuerpo_crudo: bytes,
    motivo: str,
) -> None:
    """Si el mensaje ya agoto MAX_REINTENTOS (contados via "x-death"), lo
    publica explicitamente en COLA_FALLIDOS y hace ack del original --
    si no, lo nackea sin reencolar para que el dead-letter-exchange de
    COLA_ENTRADA lo mande a la cola de reintento (TTL, luego vuelve solo)."""
    intentos = _contar_reintentos(mensaje)

    if intentos >= MAX_REINTENTOS:
        logger.error(
            "Mensaje agoto %s reintentos (%s), se envia a '%s': %r",
            MAX_REINTENTOS, motivo, COLA_FALLIDOS, cuerpo_crudo[:200],
        )
        await exchange_fallidos.publish(
            Message(body=cuerpo_crudo, delivery_mode=aio_pika.DeliveryMode.PERSISTENT),
            routing_key=COLA_FALLIDOS,
        )
        await mensaje.ack()  # ya quedo en COLA_FALLIDOS -- lo sacamos del flujo de reintentos
    else:
        logger.warning(
            "Reintento %s/%s (%s): %r", intentos + 1, MAX_REINTENTOS, motivo, cuerpo_crudo[:200]
        )
        await mensaje.nack(requeue=False)  # cae al dead-letter-exchange de reintento


async def _manejar_mensaje(
    mensaje: AbstractIncomingMessage,
    beto: ClasificadorBeto,
    exchange_principal: AbstractExchange,
    exchange_fallidos: AbstractExchange,
) -> None:
    cuerpo_crudo = mensaje.body

    try:
        datos = json.loads(cuerpo_crudo.decode("utf-8"))
        mensaje_entrada = MensajeEntrada.model_validate(datos)
    except (json.JSONDecodeError, ValidationError, UnicodeDecodeError) as exc:
        logger.warning("Mensaje mal formado, se rechaza: %s", exc)
        await _enviar_a_fallidos_o_reintentar(
            mensaje, exchange_fallidos, cuerpo_crudo, motivo="mensaje mal formado"
        )
        return

    try:
        resultado = await procesar_mensaje(mensaje_entrada, beto)
    except Exception as exc:  # noqa: BLE001 - un fallo de clasificacion nunca debe tumbar el worker
        logger.error(
            "Fallo al procesar publicacion_id=%s: %s", mensaje_entrada.publicacion_id, exc
        )
        await _enviar_a_fallidos_o_reintentar(
            mensaje, exchange_fallidos, cuerpo_crudo, motivo="excepcion durante clasificacion"
        )
        return

    await exchange_principal.publish(
        Message(
            body=resultado.model_dump_json().encode("utf-8"),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=COLA_SALIDA,
    )
    await mensaje.ack()


async def ejecutar_worker() -> None:
    if not AMQP_URL:
        raise RuntimeError(
            "AMQP_URL no esta definida. Configura esta variable de entorno (ver "
            ".env.example) antes de arrancar el worker de moderacion -- sin ella "
            "no hay forma de conectarse a RabbitMQ/CloudAMQP."
        )

    logger.info("Cargando modelo BETO...")
    beto = ClasificadorBeto()

    logger.info("Conectando a RabbitMQ...")
    conexion = await aio_pika.connect_robust(AMQP_URL)

    detener = asyncio.Event()

    def _solicitar_apagado(*_args) -> None:
        logger.info(
            "Señal de apagado recibida -- se termina de procesar el mensaje en "
            "curso (si hay uno) antes de cerrar la conexion."
        )
        detener.set()

    for nombre_senal in ("SIGINT", "SIGTERM"):
        senal = getattr(signal, nombre_senal, None)
        if senal is None:
            continue
        try:
            signal.signal(senal, _solicitar_apagado)
        except (ValueError, OSError):
            # Alguna señales no estan disponibles en todas las plataformas
            # (ej. SIGTERM en Windows se comporta distinto) -- no es fatal.
            pass

    async with conexion:
        canal = await conexion.channel()
        await canal.set_qos(prefetch_count=PREFETCH_COUNT)

        cola_entrada: AbstractQueue
        cola_entrada, exchange_principal, exchange_fallidos = await _declarar_topologia(canal)

        logger.info(
            "Worker listo. Escuchando '%s' (prefetch=%s, max_reintentos=%s)...",
            COLA_ENTRADA, PREFETCH_COUNT, MAX_REINTENTOS,
        )

        async with cola_entrada.iterator() as mensajes:
            async for mensaje in mensajes:
                await _manejar_mensaje(mensaje, beto, exchange_principal, exchange_fallidos)

                if detener.is_set():
                    logger.info("Apagando el worker limpiamente.")
                    break


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(ejecutar_worker())


if __name__ == "__main__":
    main()
