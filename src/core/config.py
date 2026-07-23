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

# Version exportada a ONNX + cuantizada a INT8 (ver
# scripts/exportar_modelo_onnx.py) -- la que carga la API en produccion por
# defecto, para reducir RAM y tiempo de inferencia en CPU (sin GPU).
RUTA_MODELO_BETO_ONNX = MODEL_ARTIFACTS_DIR / "modelo_moderacion_final_onnx_int8"

# Registro persistente (JSONL) de posibles inconsistencias
# razonamiento/veredicto detectadas en la verificacion con Qwen -- ver
# src/models/qwen_verifier.py y training/scripts/analizar_auditoria_qwen.py.
QWEN_AUDITORIA_PATH = PROJECT_ROOT / "training" / "notebooks" / "artifacts" / "qwen_auditoria.jsonl"

# Mismo orden usado durante el entrenamiento (ver notebook): el indice de
# cada logit del modelo corresponde a esta posicion.
LABEL_COLUMNS = ["grosero", "amenaza", "inapropiado"]

# Umbrales ya calibrados y validados con barrido sobre el set de test real.
# No se recalculan aqui -- se usan tal cual.
UMBRALES_POR_CATEGORIA = {
    "grosero": 0.35,
    "amenaza": 0.35,
    "inapropiado": 0.15,
}

# Margenes alrededor del umbral de cada categoria que definen su "zona
# dudosa" y disparan la verificacion opcional con Qwen. Asimetricos a
# proposito: el costo de un falso negativo (dejar pasar algo que si era
# grave) no es igual al de un falso positivo (bloquear algo inocuo), asi
# que el margen "superior" (hacia probabilidades mas altas que el umbral,
# donde dudar de menos es mas seguro) suele ser mas ancho que el
# "inferior" -- sobre todo en amenaza/inapropiado, donde el costo de un
# falso negativo es mayor.
MARGENES_POR_CATEGORIA = {
    "grosero": {"inferior": 0.15, "superior": 0.15},  # zona: [0.20, 0.50]
    "amenaza": {"inferior": 0.15, "superior": 0.20},  # zona: [0.20, 0.55]
    "inapropiado": {"inferior": 0.05, "superior": 0.20},  # zona: [0.10, 0.35]
}

MAX_LONGITUD_TEXTO = 1000

HABILITAR_QWEN = os.environ.get("HABILITAR_QWEN", "false").strip().lower() == "true"
# Id del modelo tal como lo espera la API de Groq (compatible con el SDK
# de OpenAI) -- ya no es un repo de Hugging Face, ver src/models/qwen_verifier.py.
QWEN_MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "qwen/qwen3.6-27b")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or None

HF_TOKEN = os.environ.get("HF_TOKEN") or None

# --- Worker de moderacion (consumidor AMQP) --------------------------------
# Sin valor por defecto a proposito: el worker (src/worker/consumidor.py)
# debe fallar explicitamente al arrancar si esto no esta definido. NO se
# valida aqui (este modulo tambien lo importa la API principal, que no
# necesita AMQP para funcionar) -- la validacion vive en el worker.
AMQP_URL = os.environ.get("AMQP_URL")

NOMBRE_EXCHANGE = os.environ.get("NOMBRE_EXCHANGE", "moderacion")
COLA_ENTRADA = os.environ.get("COLA_ENTRADA", "moderacion.pendiente")
COLA_SALIDA = os.environ.get("COLA_SALIDA", "moderacion.resultado")
COLA_FALLIDOS = os.environ.get("COLA_FALLIDOS", "moderacion.fallidos")

# Cada mensaje puede tardar desde milisegundos (si solo decide BETO) hasta
# unos segundos (si se consulta a Qwen via Groq) -- un prefetch moderado
# deja algo de pipeline en vuelo sin acumular demasiados mensajes sin-ack
# en un solo worker si varios caen en zona dudosa a la vez.
PREFETCH_COUNT = int(os.environ.get("PREFETCH_COUNT", "4"))

# Politica de reintentos (ver src/worker/consumidor.py para el patron
# TTL + dead-letter-exchange completo): cuantos intentos antes de mandar
# el mensaje a COLA_FALLIDOS, y cuanto esperar entre cada reintento.
MAX_REINTENTOS = int(os.environ.get("MAX_REINTENTOS", "5"))
TTL_REINTENTO_MS = int(os.environ.get("TTL_REINTENTO_MS", "10000"))
