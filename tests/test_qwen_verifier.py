"""Tests unitarios de la capa de verificacion con Qwen.

Estos tests NO descargan ni cargan el modelo real: fuerzan
HABILITAR_QWEN=False (independientemente de lo que tenga el .env local) y
verifican que en ese caso el modulo es un no-op completo, ademas de
probar el parseo tolerante de la respuesta de forma aislada.
"""

import json
import logging
from unittest.mock import MagicMock, patch

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


def test_parsear_respuesta_con_comilla_suelta_en_razon_usa_plan_b():
    # La comilla suelta alrededor de "cuidate mucho" rompe el parseo JSON
    # estricto (json.loads fallaria aca), pero el campo "confirma" sigue
    # siendo rescatable con la regex de respaldo.
    texto_generado = (
        '{"confirma": true, "razon": "el texto dice "cuidate mucho" de forma amenazante"}'
    )

    confirma, razon = qwen_verifier._parsear_respuesta(texto_generado)

    assert confirma is True
    assert razon == "Razón no disponible (formato de respuesta inválido)"


def test_parsear_respuesta_con_comilla_suelta_y_confirma_false_usa_plan_b():
    texto_generado = (
        '{"confirma": false, "razon": "no hay "insinuacion" real en el texto"}'
    )

    confirma, razon = qwen_verifier._parsear_respuesta(texto_generado)

    assert confirma is False
    assert razon == "Razón no disponible (formato de respuesta inválido)"


# --- Ejemplos few-shot especificos por categoria (bug: se colaban los de
# "inapropiado" sin importar que categoria se estuviera verificando) -----


def test_prompt_de_amenaza_no_incluye_ejemplos_de_inapropiado():
    prompt = qwen_verifier._construir_prompt("texto de prueba", "amenaza")

    assert "partirle la madre al encargado" in prompt
    assert "báscula" not in prompt
    assert "foto de perfil" not in prompt


def test_prompt_de_inapropiado_no_incluye_ejemplos_de_amenaza():
    prompt = qwen_verifier._construir_prompt("texto de prueba", "inapropiado")

    assert "báscula" in prompt
    assert "partirle la madre" not in prompt


def test_prompt_de_grosero_no_incluye_ejemplos_de_otras_categorias():
    prompt = qwen_verifier._construir_prompt("texto de prueba", "grosero")

    assert "grosera de mierda" in prompt
    assert "báscula" not in prompt
    assert "partirle la madre" not in prompt


def test_prompt_incluye_el_texto_y_la_categoria_evaluada():
    prompt = qwen_verifier._construir_prompt("Hola mundo", "grosero")

    assert "Hola mundo" in prompt
    assert "Categoría a evaluar: grosero" in prompt


# --- Instruccion de analisis especifica por categoria (bug: la instruccion
# de buscar "insinuaciones/dobles sentidos" -- solo valida para
# "inapropiado" -- se aplicaba tambien a "amenaza" y "grosero") ------------


def test_prompt_de_amenaza_pide_analisis_literal_sin_dobles_sentidos():
    prompt = qwen_verifier._construir_prompt("texto de prueba", "amenaza")

    assert "LITERAL" in prompt
    assert "no es una amenaza" in prompt.lower()
    # No debe llevar la instruccion de "inapropiado" (buscar insinuaciones).
    assert "insinuaciones" not in prompt.lower()


def test_prompt_de_grosero_pide_analisis_literal_de_vocabulario():
    prompt = qwen_verifier._construir_prompt("texto de prueba", "grosero")

    assert "vulgar" in prompt.lower()
    assert "insinuaciones" not in prompt.lower()
    assert "no es una amenaza" not in prompt.lower()


def test_prompt_de_inapropiado_conserva_instruccion_de_dobles_sentidos():
    prompt = qwen_verifier._construir_prompt("texto de prueba", "inapropiado")

    assert "insinuaciones" in prompt.lower()
    assert "dobles sentidos" in prompt.lower()


# --- Mapeo explicito true/false por categoria en el formato de salida
# (bug: el razonamiento en texto de Qwen podia concluir "no es amenaza" y
# aun asi el campo "confirma" quedar en true, por ambiguedad en el prompt) -


def test_prompt_mapea_confirma_true_false_a_la_categoria_evaluada():
    prompt = qwen_verifier._construir_prompt("texto de prueba", "amenaza")

    assert "SI Y SOLO SI" in prompt
    assert 'pertenece a la categoría "amenaza"' in prompt
    assert '"confirma": false SI el texto NO pertenece a la categoría "amenaza"' in prompt


def test_prompt_mapeo_usa_la_categoria_dinamicamente_no_hardcodeada():
    prompt_grosero = qwen_verifier._construir_prompt("texto de prueba", "grosero")

    assert 'pertenece a la categoría "grosero"' in prompt_grosero
    assert 'pertenece a la categoría "amenaza"' not in prompt_grosero


# --- Chequeo de auditoria: razon del modelo vs. valor de "confirma" ------
# Puramente informativo (solo logging) -- nunca debe alterar el veredicto.


def test_detectar_inconsistencia_confirma_true_con_razon_negativa():
    assert (
        qwen_verifier._detectar_posible_inconsistencia(
            True, "esto no implica ninguna amenaza"
        )
        is True
    )


def test_detectar_inconsistencia_confirma_false_con_razon_negativa_es_caso_normal():
    # Negacion legitima explicando un false correcto -- no es una
    # inconsistencia, es el patron esperado.
    assert (
        qwen_verifier._detectar_posible_inconsistencia(
            False, "esto no implica ninguna amenaza"
        )
        is False
    )


def test_verificar_con_qwen_no_altera_el_veredicto_pese_a_inconsistencia_detectada(
    tmp_path, monkeypatch, caplog
):
    # Aunque la salvaguarda detecte y loggee la inconsistencia, el
    # (confirma, razon) que retorna verificar_con_qwen debe ser exactamente
    # el mismo que devolvio _parsear_respuesta -- la auditoria no debe
    # descartar ni sobreescribir el veredicto de Qwen bajo ninguna
    # circunstancia.
    monkeypatch.setattr(qwen_verifier, "HABILITAR_QWEN", True)
    # Esta llamada tambien dispara _registrar_auditoria (escribe a disco) --
    # redirigimos a un tmp_path para no contaminar el JSONL real del repo.
    monkeypatch.setattr(qwen_verifier, "QWEN_AUDITORIA_PATH", tmp_path / "qwen_auditoria.jsonl")
    _resetear_estado_modulo()
    qwen_verifier._carga_intentada = True
    qwen_verifier._carga_exitosa = True
    qwen_verifier._tokenizer = MagicMock()
    qwen_verifier._model = MagicMock()

    resultado_inconsistente = (True, "esto no implica ninguna amenaza")
    monkeypatch.setattr(
        qwen_verifier, "_parsear_respuesta", lambda texto_generado: resultado_inconsistente
    )

    with caplog.at_level(logging.WARNING, logger="moderacion.qwen"):
        resultado = qwen_verifier.verificar_con_qwen("texto de prueba", "amenaza", 0.5)

    assert resultado == resultado_inconsistente
    assert any("inconsistencia" in registro.message.lower() for registro in caplog.records)

    _resetear_estado_modulo()


# --- Persistencia de la auditoria en JSONL (training/notebooks/artifacts/
# qwen_auditoria.jsonl) -- el logging por consola no garantiza que quede
# algo revisable despues. ---------------------------------------------------


def test_registrar_auditoria_escribe_linea_jsonl_valida(tmp_path, monkeypatch):
    ruta = tmp_path / "subdir" / "qwen_auditoria.jsonl"
    monkeypatch.setattr(qwen_verifier, "QWEN_AUDITORIA_PATH", ruta)

    qwen_verifier._registrar_auditoria(
        "amenaza", "texto de prueba", True, "esto no implica ninguna amenaza"
    )

    assert ruta.exists()  # incluye la creacion de la carpeta padre
    lineas = ruta.read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 1

    evento = json.loads(lineas[0])
    assert evento["categoria"] == "amenaza"
    assert evento["texto"] == "texto de prueba"
    assert evento["confirma"] is True
    assert evento["razon"] == "esto no implica ninguna amenaza"
    assert "timestamp" in evento


def test_registrar_auditoria_acumula_en_vez_de_sobreescribir(tmp_path, monkeypatch):
    ruta = tmp_path / "qwen_auditoria.jsonl"
    monkeypatch.setattr(qwen_verifier, "QWEN_AUDITORIA_PATH", ruta)

    qwen_verifier._registrar_auditoria("amenaza", "primer texto", True, "razon 1")
    qwen_verifier._registrar_auditoria("grosero", "segundo texto", True, "razon 2")

    lineas = ruta.read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 2
    assert json.loads(lineas[0])["texto"] == "primer texto"
    assert json.loads(lineas[1])["texto"] == "segundo texto"


def test_registrar_auditoria_trunca_texto_a_200_caracteres(tmp_path, monkeypatch):
    ruta = tmp_path / "qwen_auditoria.jsonl"
    monkeypatch.setattr(qwen_verifier, "QWEN_AUDITORIA_PATH", ruta)

    qwen_verifier._registrar_auditoria("grosero", "a" * 500, True, "razon")

    evento = json.loads(ruta.read_text(encoding="utf-8").strip())
    assert len(evento["texto"]) == 200


def test_registrar_auditoria_no_lanza_si_falla_la_escritura(monkeypatch):
    # QWEN_AUDITORIA_PATH=None hace que .parent falle dentro del try/except
    # -- simula un fallo de escritura sin depender del sistema de archivos.
    monkeypatch.setattr(qwen_verifier, "QWEN_AUDITORIA_PATH", None)

    qwen_verifier._registrar_auditoria("amenaza", "texto", True, "razon")  # no debe lanzar


def test_verificar_con_qwen_registra_auditoria_cuando_detecta_inconsistencia(
    tmp_path, monkeypatch
):
    ruta_auditoria = tmp_path / "qwen_auditoria.jsonl"
    monkeypatch.setattr(qwen_verifier, "QWEN_AUDITORIA_PATH", ruta_auditoria)
    monkeypatch.setattr(qwen_verifier, "HABILITAR_QWEN", True)
    _resetear_estado_modulo()
    qwen_verifier._carga_intentada = True
    qwen_verifier._carga_exitosa = True
    qwen_verifier._tokenizer = MagicMock()
    qwen_verifier._model = MagicMock()

    resultado_inconsistente = (True, "esto no implica ninguna amenaza")
    monkeypatch.setattr(
        qwen_verifier, "_parsear_respuesta", lambda texto_generado: resultado_inconsistente
    )

    qwen_verifier.verificar_con_qwen("texto de prueba", "amenaza", 0.5)

    assert ruta_auditoria.exists()
    evento = json.loads(ruta_auditoria.read_text(encoding="utf-8").strip())
    assert evento["categoria"] == "amenaza"
    assert evento["confirma"] is True

    _resetear_estado_modulo()
