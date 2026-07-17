"""Capa opcional de verificacion con un LLM generativo (Qwen).

Se activa solo cuando una probabilidad de BETO cae en zona dudosa cerca de
su umbral (ver core.logic). Si Qwen falla, no esta disponible, o no esta
habilitado por variable de entorno, la API debe seguir funcionando
unicamente con la decision de BETO -- por eso este modulo nunca deja
propagar una excepcion hacia arriba.
"""

import json
import logging

import torch

from src.core.config import HABILITAR_QWEN, HF_TOKEN, QWEN_MODEL_NAME

logger = logging.getLogger("moderacion.qwen")

DEFINICIONES_CATEGORIA = {
    "grosero": "lenguaje ofensivo, insultos o groserias dirigidas a alguien",
    "amenaza": "una amenaza real de dano o violencia contra una persona",
    "inapropiado": "contenido de doble sentido sexual o inapropiado para un marketplace",
}


class VerificadorQwen:
    """El modelo se carga de forma perezosa (solo en la primera verificacion
    real) para no pagar el costo de descarga/carga si nunca se usa."""

    def __init__(self):
        self.habilitado = HABILITAR_QWEN
        self._model = None
        self._tokenizer = None
        self._device = None

    def disponible(self) -> bool:
        return self.habilitado

    def _cargar_modelo(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, token=HF_TOKEN)
        self._model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, token=HF_TOKEN)
        self._model.to(self._device)
        self._model.eval()

        logger.info("Modelo Qwen (%s) cargado (device=%s)", QWEN_MODEL_NAME, self._device)

    def verificar(self, texto: str, categoria: str, probabilidad: float) -> dict:
        """Devuelve {"confirma": bool, "razon": str, "exito": bool}.

        "exito" indica si la verificacion realmente se pudo ejecutar; el
        llamador debe ignorar "confirma" y conservar la decision original de
        BETO cuando exito=False, en vez de asumir que Qwen dijo que no.
        """
        try:
            if self._model is None:
                self._cargar_modelo()

            definicion = DEFINICIONES_CATEGORIA.get(categoria, categoria)
            prompt = (
                "Eres un moderador de contenido para un marketplace de compraventa. "
                f"Evalua si el siguiente texto realmente corresponde a la categoria "
                f"'{categoria}' ({definicion}).\n\n"
                f'Texto: "{texto}"\n\n'
                "Responde UNICAMENTE con un JSON de la forma: "
                '{"confirma": true o false, "razon": "explicacion breve"}'
            )

            mensajes = [{"role": "user", "content": prompt}]
            entrada = self._tokenizer.apply_chat_template(
                mensajes, add_generation_prompt=True, tokenize=False
            )
            inputs = self._tokenizer(entrada, return_tensors="pt").to(self._device)

            with torch.no_grad():
                salida = self._model.generate(**inputs, max_new_tokens=150, do_sample=False)

            texto_generado = self._tokenizer.decode(
                salida[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            inicio_json = texto_generado.find("{")
            fin_json = texto_generado.rfind("}")
            if inicio_json == -1 or fin_json == -1:
                raise ValueError(f"Qwen no devolvio JSON valido: {texto_generado!r}")

            resultado = json.loads(texto_generado[inicio_json : fin_json + 1])
            return {
                "confirma": bool(resultado.get("confirma", False)),
                "razon": str(resultado.get("razon", "")),
                "exito": True,
            }

        except Exception as exc:  # noqa: BLE001 - un fallo de Qwen nunca debe tumbar la API
            logger.warning("Verificacion con Qwen fallo para categoria=%s: %s", categoria, exc)
            return {
                "confirma": False,
                "razon": f"Verificacion con Qwen no disponible: {exc}",
                "exito": False,
            }
