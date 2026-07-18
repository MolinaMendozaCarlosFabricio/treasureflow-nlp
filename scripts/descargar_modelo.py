"""Descarga el modelo de moderacion (BETO fine-tuneado) desde un
repositorio privado de Hugging Face Hub hacia model_artifacts/modelo_moderacion_final/.

Pensado para correrse como paso previo al arranque de la API en un
entorno nuevo (ej. despliegue) donde el modelo todavia no esta presente
localmente. NO reemplaza como src/models/beto_classifier.py carga el
modelo -- ese archivo sigue leyendo desde la misma ruta local de
siempre (model_artifacts/modelo_moderacion_final/); este script solo se
encarga de dejarla poblada de antemano.

Uso:
    python scripts/descargar_modelo.py

Variables de entorno (.env, via python-dotenv -- mismo patron que ya usa
el resto del proyecto para HF_TOKEN):
    HF_TOKEN       - token de Hugging Face (el repo es privado, hace falta)
    MODEL_REPO     - id del repo en el Hub
                     (default: "charlie-charl/treasureflow-moderation-beto")
    MODEL_REVISION - revision/tag a descargar (default: "main")
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

RUTA_DESTINO = PROJECT_ROOT / "model_artifacts" / "modelo_moderacion_final"

MODEL_REPO = os.environ.get("MODEL_REPO") or "charlie-charl/treasureflow-moderation-beto"
MODEL_REVISION = os.environ.get("MODEL_REVISION") or "main"
HF_TOKEN = os.environ.get("HF_TOKEN") or None


def _carpeta_ya_tiene_contenido(ruta: Path) -> bool:
    return ruta.exists() and any(ruta.iterdir())


def main() -> None:
    if _carpeta_ya_tiene_contenido(RUTA_DESTINO):
        print(f"El modelo ya existe en {RUTA_DESTINO} (tiene archivos) -- se omite la descarga.")
        return

    print(f"Descargando modelo desde el repo '{MODEL_REPO}' (revision '{MODEL_REVISION}')...")
    print(f"Destino: {RUTA_DESTINO}")

    try:
        from huggingface_hub import snapshot_download

        RUTA_DESTINO.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            repo_type="model",
            token=HF_TOKEN,
            local_dir=str(RUTA_DESTINO),
        )

    except Exception as exc:  # noqa: BLE001 - cualquier fallo de red/auth/repo debe frenar el despliegue
        print(
            f"ERROR: no se pudo descargar el modelo desde '{MODEL_REPO}' "
            f"(revision '{MODEL_REVISION}').\n"
            f"Causa: {exc}\n\n"
            "Motivos probables: sin conexion a internet, HF_TOKEN invalido o sin "
            "permisos sobre el repo (es privado), o MODEL_REPO/MODEL_REVISION mal "
            "configurados en tu .env."
        )
        sys.exit(1)

    print(f"Modelo descargado correctamente en: {RUTA_DESTINO}")


if __name__ == "__main__":
    main()
