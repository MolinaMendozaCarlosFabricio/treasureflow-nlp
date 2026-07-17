"""Tests unitarios de la capa de verificacion con Qwen.

Estos tests NO descargan ni cargan el modelo real: fuerzan
HABILITAR_QWEN=False (independientemente de lo que tenga el .env local) y
verifican que en ese caso el modulo es un no-op completo, ademas de
probar el parseo tolerante de la respuesta de forma aislada.
"""

from unittest.mock import patch

import src.models.qwen_verifier as qwen_verifier


def _resetear_estado_modulo():
    qwen_verifier._model = None
    qwen_verifier._tokenizer = None
    qwen_verifier._carga_intentada = False
    qwen_verifier._carga_exitosa = False


def test_qwen_deshabilitado_no_intenta_cargar_el_modelo(monkeypatch):
    monkeypatch.setattr(qwen_verifier, "HABILITAR_QWEN", False)
    _resetear_estado_modulo()

    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_causal_lm:
        assert qwen_verifier.esta_disponible() is False
        mock_causal_lm.assert_not_called()

    _resetear_estado_modulo()


def test_verificar_con_qwen_deshabilitado_retorna_none_sin_lanzar(monkeypatch):
    monkeypatch.setattr(qwen_verifier, "HABILITAR_QWEN", False)
    _resetear_estado_modulo()

    confirma, razon = qwen_verifier.verificar_con_qwen("cualquier texto", "grosero", 0.5)

    assert confirma is None
    assert "no disponible" in razon.lower()

    _resetear_estado_modulo()


def test_parsear_respuesta_tolerante_a_texto_extra_alrededor_del_json():
    confirma, razon = qwen_verifier._parsear_respuesta(
        'Claro, aqui esta mi analisis: {"confirma": true, "razon": "es una amenaza directa"} espero que ayude.'
    )

    assert confirma is True
    assert razon == "es una amenaza directa"


def test_parsear_respuesta_sin_json_valido_retorna_none():
    confirma, razon = qwen_verifier._parsear_respuesta("esto no tiene ningun json adentro")

    assert confirma is None
    assert razon == "no se pudo interpretar la respuesta"
