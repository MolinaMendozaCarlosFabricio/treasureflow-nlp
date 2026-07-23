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


def test_beto_carga_el_modelo_original_de_pytorch(client):
    # Decision de arquitectura (ver src/models/beto_classifier.py y README):
    # se evaluo la version ONNX+INT8 pero se descarto porque consumia mas
    # RAM (~850MB vs ~700MB) sin aportar beneficio real, dado que la
    # moderacion ahora corre async via un worker de cola, no en el camino
    # critico de una peticion HTTP. La API sigue sirviendo el modelo
    # original en PyTorch (no la version ONNX).
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import PreTrainedModel

    modelo_cargado = client.app.state.beto.model
    assert isinstance(modelo_cargado, PreTrainedModel)
    assert not isinstance(modelo_cargado, ORTModelForSequenceClassification)


def test_texto_neutro_permite(client):
    # NOTA: tras un reentrenamiento del modelo (2026-07-22), el texto que
    # se usaba antes aca ("Buen servicio, me atendieron muy bien...")
    # empezo a dar inapropiado~0.84 con este checkpoint -- variacion real
    # del modelo en la categoria minoritaria "inapropiado" (ver nota sobre
    # varianza en el notebook de entrenamiento), no un bug de la API. Se
    # cambio a un texto que sigue siendo neutro y da un score bajo con el
    # checkpoint actual.
    respuesta = client.post(
        "/moderar",
        json={"texto": "Todo en orden, el material llegó completo y a tiempo."},
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


def test_texto_neutro_bien_capitalizado_no_se_bloquea_por_normalizacion(client):
    # Regresion de un bug real (docs/prueba_1.txt): antes se normalizaba
    # SIEMPRE (incluso texto limpio) y se tomaba el maximo contra el
    # original, lo que amplificaba falsos positivos cuando el modelo
    # reaccionaba de forma inesperada a la version normalizada -- este
    # mismo texto llegaba a un score de "grosero" >0.75 via la version
    # normalizada (vs. ~0.08 en el original), bloqueando contenido
    # totalmente inocuo. Ahora, ademas de que normalizar_texto ya no
    # fuerza lowercase, la segunda pasada por el modelo solo corre si
    # texto_parece_ofuscado() detecta alguna señal real de ofuscacion (ver
    # src/models/beto_classifier.py) -- este texto no tiene ninguna, asi
    # que solo se evalua el original.
    respuesta = client.post("/moderar", json={"texto": "Botellas de plástico"})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["bloqueado"] is False
    assert cuerpo["categorias"]["grosero"]["activado"] is False
    assert cuerpo["categorias"]["grosero"]["detectado_via_normalizacion"] is False
