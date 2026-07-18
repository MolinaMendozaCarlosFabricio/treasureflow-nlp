from huggingface_hub import HfApi

import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv(".env")

hf_token = os.environ.get("HF_TOKEN")
print(hf_token)
if hf_token:
    login(hf_token)
    print("Sesion de Hugging Face iniciada.")
else:
    print("HF_TOKEN no esta definido (revisa tu archivo .env). "
          "Continuo sin iniciar sesion; solo hace falta para datasets/modelos con acceso restringido.")

api = HfApi()
api.upload_folder(
    folder_path="model_artifacts/modelo_moderacion_final",
    repo_id="charlie-charl/treasureflow-moderation-beto",
    repo_type="model",
    commit_message="feat: Modelo optimizado con Optuna, umbrales calibrados. v1",
)