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

logger = logging.getLogger("moderacion.beto")


@dataclass
class ResultadoBeto:
    probabilidades: dict
    activaciones: dict
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

    def predict(self, texto: str) -> ResultadoBeto:
        inicio = time.perf_counter()

        inputs = self.tokenizer(
            texto, return_tensors="pt", truncation=True, padding="max_length", max_length=128
        )
        inputs = {clave: valor.to(self.device) for clave, valor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

        probabilidades = {}
        activaciones = {}
        for i, nombre in enumerate(LABEL_COLUMNS):
            probabilidades[nombre] = round(float(probs[i]), 4)
            activaciones[nombre] = bool(probs[i] > UMBRALES_POR_CATEGORIA[nombre])

        tiempo_inferencia_ms = (time.perf_counter() - inicio) * 1000
        return ResultadoBeto(probabilidades, activaciones, tiempo_inferencia_ms)
