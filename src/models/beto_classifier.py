"""Wrapper del modelo BETO fine-tuneado para moderacion de contenido.

Carga el modelo ya entrenado y validado en model_artifacts/modelo_moderacion_final
(ver training/notebooks/NLP_treasureflow.ipynb) -- este modulo NUNCA entrena ni
modifica esos artefactos, solo los consume para inferencia.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.config import LABEL_COLUMNS, RUTA_MODELO_BETO, UMBRALES_POR_CATEGORIA
from src.utils.normalizacion import normalizar_texto

logger = logging.getLogger("moderacion.beto")


@dataclass
class ResultadoBeto:
    probabilidades: dict
    activaciones: dict
    detectado_via_normalizacion: dict
    tiempo_inferencia_ms: float


class ClasificadorBeto:
    """Carga el modelo BETO fine-tuneado una sola vez (pensado para vivir
    en app.state durante todo el ciclo de vida de la API) y expone predict()."""

    def __init__(self, ruta_modelo: Path = RUTA_MODELO_BETO):
        if not ruta_modelo.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo en {ruta_modelo}. Corre el notebook de "
                "entrenamiento (training/notebooks/NLP_treasureflow.ipynb) antes de "
                "levantar la API."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(ruta_modelo))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(ruta_modelo))
        self.model.to(self.device)
        self.model.eval()

        logger.info("Modelo BETO cargado desde %s (device=%s)", ruta_modelo, self.device)

    def _inferir_probabilidades(self, texto: str) -> dict:
        """Corre el modelo sobre `texto` tal cual (sin normalizar) y
        devuelve {categoria: probabilidad}. Reutilizado tanto para el
        texto original como para la version normalizada en predict()."""
        inputs = self.tokenizer(
            texto, return_tensors="pt", truncation=True, padding="max_length", max_length=128
        )
        inputs = {clave: valor.to(self.device) for clave, valor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

        return {nombre: float(probs[i]) for i, nombre in enumerate(LABEL_COLUMNS)}

    def predict(self, texto: str) -> ResultadoBeto:
        """Infiere sobre el texto original Y sobre su version normalizada
        (ver utils.normalizacion.normalizar_texto), para ser robustos a
        ofuscacion (leetspeak, espaciado, separadores intercalados). El
        resultado final por categoria es el MAXIMO de ambas probabilidades,
        nunca un promedio -- si cualquiera de las dos versiones detecta
        algo sospechoso, eso se refleja en el resultado."""
        inicio = time.perf_counter()

        probabilidades_original = self._inferir_probabilidades(texto)

        texto_normalizado = normalizar_texto(texto)
        if texto_normalizado != texto:
            probabilidades_normalizado = self._inferir_probabilidades(texto_normalizado)
        else:
            # La normalizacion no cambio nada -- evitamos una segunda
            # pasada por el modelo con el mismo texto.
            probabilidades_normalizado = probabilidades_original

        probabilidades = {}
        activaciones = {}
        detectado_via_normalizacion = {}
        for nombre in LABEL_COLUMNS:
            prob_original = probabilidades_original[nombre]
            prob_normalizado = probabilidades_normalizado[nombre]

            gano_normalizado = prob_normalizado > prob_original
            probabilidad_final = prob_normalizado if gano_normalizado else prob_original

            probabilidades[nombre] = round(probabilidad_final, 4)
            activaciones[nombre] = bool(probabilidad_final > UMBRALES_POR_CATEGORIA[nombre])
            detectado_via_normalizacion[nombre] = gano_normalizado

        tiempo_inferencia_ms = (time.perf_counter() - inicio) * 1000
        return ResultadoBeto(
            probabilidades, activaciones, detectado_via_normalizacion, tiempo_inferencia_ms
        )
