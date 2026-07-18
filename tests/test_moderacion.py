"""Tests de humo para la API de moderacion de contenido.

Usan el modelo real ya entrenado en model_artifacts/modelo_moderacion_final
(no se mockea) para verificar el comportamiento end-to-end del endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture(scope="module")
def client():
    # El context manager dispara el lifespan de FastAPI (carga del modelo
    # BETO una sola vez, reutilizado por todos los tests de este modulo).
    with TestClient(app) as test_client:
        yield test_client


def test_health_confirma_modelo_cargado(client):
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["modelo_cargado"] is True
    assert cuerpo["status"] == "ok"
    # No asumimos un valor fijo para qwen_habilitado: depende de HABILITAR_QWEN
    # en el .env de quien corra el test. Esa garantia especifica (que con
    # HABILITAR_QWEN=false no se toca Qwen) ya se prueba de forma determinista,
    # sin depender del .env local, en tests/test_qwen_verifier.py.
    assert isinstance(cuerpo["qwen_habilitado"], bool)


def test_texto_neutro_permite(client):
    respuesta = client.post(
        "/moderar",
        json={"texto": "Buen servicio, me atendieron muy bien, completamente recomendado."},
    )

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["bloqueado"] is False


def test_texto_con_groserias_obvias_bloquea(client):
    respuesta = client.post(
        "/moderar",
        json={"texto": "Pinches putos no me pagaron, este lugar es una mierda."},
    )

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["bloqueado"] is True
    assert cuerpo["categorias"]["grosero"]["activado"] is True


def test_texto_vacio_da_422(client):
    respuesta = client.post("/moderar", json={"texto": ""})

    assert respuesta.status_code == 422


def test_texto_muy_largo_da_422(client):
    respuesta = client.post("/moderar", json={"texto": "a" * 1001})

    assert respuesta.status_code == 422


# --- Robustez frente a ofuscacion (normalizar_texto + maximo entre texto
# original y normalizado, ver src/models/beto_classifier.py) --------------
# Antes de la normalizacion, el espaciado entre letras rompia el token que
# el tokenizer necesitaba para reconocer la palabra -- estos casos antes
# probablemente no se detectaban bien.


def test_texto_con_grosero_espaciado_letra_por_letra_bloquea(client):
    respuesta = client.post(
        "/moderar",
        json={"texto": "Eres un p u t o de mierda"},
    )

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["bloqueado"] is True
    assert cuerpo["categorias"]["grosero"]["activado"] is True
    # Prueba que la version normalizada fue la que realmente aporto la
    # señal ganadora, no una casualidad del texto original sin ofuscar.
    assert cuerpo["categorias"]["grosero"]["detectado_via_normalizacion"] is True


def test_texto_con_grosero_espaciado_en_resena_bloquea(client):
    respuesta = client.post(
        "/moderar",
        json={"texto": "Que p u t o servicio, nunca vuelvo", "campo": "resena"},
    )

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["bloqueado"] is True
    assert cuerpo["categorias"]["grosero"]["activado"] is True
    assert cuerpo["categorias"]["grosero"]["detectado_via_normalizacion"] is True


def test_texto_con_grosero_separado_por_puntos_bloquea(client):
    respuesta = client.post("/moderar", json={"texto": "p.u.t.o"})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["bloqueado"] is True
    assert cuerpo["categorias"]["grosero"]["activado"] is True
    assert cuerpo["categorias"]["grosero"]["detectado_via_normalizacion"] is True
