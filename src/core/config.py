"""Configuracion central de la API: rutas, umbrales y flags de entorno.

Reutiliza el mismo patron que el notebook de entrenamiento para resolver
rutas del proyecto (pathlib relativo a este archivo, no al cwd) y para
cargar variables de entorno desde el .env de la raiz del repo via
python-dotenv.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

MODEL_ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"
RUTA_MODELO_BETO = MODEL_ARTIFACTS_DIR / "modelo_moderacion_final"

# Mismo orden usado durante el entrenamiento (ver notebook): el indice de
# cada logit del modelo corresponde a esta posicion.
LABEL_COLUMNS = ["grosero", "amenaza", "inapropiado"]

# Umbrales ya calibrados y validados con barrido sobre el set de test real.
# No se recalculan aqui -- se usan tal cual.
UMBRALES_POR_CATEGORIA = {
    "grosero": 0.5,
    "amenaza": 0.5,
    "inapropiado": 0.30,
}

# Margen alrededor del umbral de cada categoria que define su "zona dudosa"
# y dispara la verificacion opcional con Qwen. Es por categoria porque no
# todas necesitan el mismo margen de duda (ej. amenaza justifica un margen
# mas amplio dado el costo de un falso negativo).
MARGENES_POR_CATEGORIA = {
    "grosero": 0.15,
    "amenaza": 0.20,
    "inapropiado": 0.15,
}

MAX_LONGITUD_TEXTO = 1000

HABILITAR_QWEN = os.environ.get("HABILITAR_QWEN", "false").strip().lower() == "true"
QWEN_MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "Qwen/Qwen3-0.6B")

HF_TOKEN = os.environ.get("HF_TOKEN") or None
