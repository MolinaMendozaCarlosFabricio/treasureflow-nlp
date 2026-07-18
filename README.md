# treasureflow_nlp

Fine-tuning de un modelo de tipo BETO/hate-speech en español para moderación
de contenido multilabel (`grosero` / `amenaza` / `inapropiado`), pensado para
publicaciones de un marketplace / centros de acopio.

Ver [docs/arbol_conocimiento_NLP_treasureflow.json](docs/arbol_conocimiento_NLP_treasureflow.json)
para el mapa conceptual completo del pipeline.

## Estructura del proyecto

```
treasureflow_nlp/
├── docs/
│   └── arbol_conocimiento_NLP_treasureflow.json  # mapa conceptual del pipeline
├── model_artifacts/                              # salida: checkpoints y modelo final entrenado
├── training/
│   ├── data/
│   │   ├── raw/                                  # entrada: datasets descargados/curados sin combinar
│   │   └── processed/                            # salida: dataset combinado y splits train/val/test
│   └── notebooks/
│       └── NLP_treasureflow.ipynb                # notebook de fine-tuning
├── src/                                          # API FastAPI que sirve el modelo entrenado
│   ├── main.py
│   ├── routes/
│   ├── models/
│   ├── schemas/
│   ├── core/
│   └── utils/
├── tests/
│   └── test_moderacion.py
├── requirements.txt
└── .env.example
```

## Requisitos

- Python 3.10+
- Una cuenta de [Hugging Face](https://huggingface.co/) con un token de acceso
  (para descargar datasets/modelos, algunos con acceso restringido)

## Inicializar el proyecto

1. **Crear y activar el entorno virtual** en la raíz del repo:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Instalar las dependencias**:

   ```powershell
   pip install -r requirements.txt
   ```

3. **Configurar el token de Hugging Face**: copia `.env.example` a `.env` y
   completa `HF_TOKEN` con tu token.

   ```powershell
   copy .env.example .env
   ```

4. **Registrar el kernel de Jupyter** (para poder seleccionar este entorno
   virtual al ejecutar el notebook desde VS Code u otro cliente Jupyter):

   ```powershell
   python -m ipykernel install --user --name=treasureflow_nlp --display-name="Python (treasureflow_nlp)"
   ```

5. **Abrir el notebook** `training/notebooks/NLP_treasureflow.ipynb`,
   seleccionar el kernel `Python (treasureflow_nlp)` y ejecutar las celdas en
   orden. El notebook asume que se ejecuta con `training/notebooks/` como
   directorio de trabajo (comportamiento por defecto de Jupyter/VS Code al
   abrir un `.ipynb`) para resolver las rutas del proyecto.

## Datos y modelos

- Coloca los datasets de dominio propio (centros de acopio, ejemplos
  manuales, etc.) que el notebook espera en `training/data/raw/` antes de
  correr el paso de combinación de fuentes.
- El notebook descarga y escribe ahí los datasets públicos (OffendES,
  Spanish Hate Speech Superset) y los archivos intermedios de revisión
  manual.
- El dataset final combinado y los splits `train.csv` / `validation.csv` /
  `test.csv` se guardan en `training/data/processed/`.
- Los checkpoints de entrenamiento y el modelo final (`trainer.save_model`)
  se guardan en `model_artifacts/`.

Ninguna de estas carpetas de datos ni el modelo entrenado se versionan en
git (ver `.gitignore`); solo se mantiene su estructura mediante `.gitkeep`.

## Descargar el modelo ya entrenado (entorno nuevo)

Si estás preparando un entorno nuevo (ej. un despliegue) y `model_artifacts/`
está vacío porque no vas a correr el notebook de entrenamiento ahí, puedes
bajar el modelo ya entrenado desde el repositorio privado de Hugging Face
Hub donde se publica:

```powershell
python scripts/descargar_modelo.py
```

- Descarga el contenido a `model_artifacts/modelo_moderacion_final/` — la
  misma ruta que usa `src/models/beto_classifier.py` para cargar el modelo,
  así que no hace falta ningún otro cambio para que la API lo encuentre.
- Si esa carpeta ya existe y tiene archivos, el script no vuelve a
  descargar nada (evita descargas innecesarias).
- Si la descarga falla (sin conexión, token inválido, repo no encontrado),
  imprime un mensaje explicando la causa probable y termina con código de
  error distinto de cero, para que un pipeline de despliegue detecte el
  fallo en vez de seguir con un modelo faltante.
- Variables de entorno (en tu `.env`, ver `.env.example`):
  - `HF_TOKEN` — el repo es privado, necesita el mismo token que ya usas
    para el resto del proyecto.
  - `MODEL_REPO` — id del repositorio en el Hub (tiene un valor por
    defecto razonable si no lo defines).
  - `MODEL_REVISION` — revisión/tag a descargar (default: `main`).

Este paso es previo al arranque de la API: córrelo antes de `uvicorn` la
primera vez en un entorno donde el modelo todavía no esté presente.

## Levantar la API de moderación

Una vez que existe `model_artifacts/modelo_moderacion_final/` (generado por
el notebook de entrenamiento, o descargado con `scripts/descargar_modelo.py`
como se explica arriba), se puede servir ese modelo con una API FastAPI sin
necesidad de volver a entrenar nada.

1. Con el entorno virtual activado e instalado (pasos 1-3 de
   "Inicializar el proyecto"), levanta el servidor **desde la raíz del
   repo**:

   ```powershell
   uvicorn src.main:app --reload
   ```

2. La API queda disponible en `http://127.0.0.1:8000`, con documentación
   interactiva en `http://127.0.0.1:8000/docs`.

3. Endpoints:

   - `GET /health` — confirma que el modelo BETO está cargado y la API
     lista para recibir tráfico.
   - `POST /moderar` — evalúa un texto y decide si debe bloquearse.

     ```json
     // Request
     { "texto": "texto a evaluar", "campo": "resena" }

     // Response
     {
       "texto": "texto a evaluar",
       "bloqueado": false,
       "categorias": {
         "grosero": { "probabilidad": 0.02, "activado": false },
         "amenaza": { "probabilidad": 0.01, "activado": false },
         "inapropiado": { "probabilidad": 0.05, "activado": false }
       },
       "verificado_por_qwen": false,
       "detalle_verificacion": []
     }
     ```

     Un texto vacío o de más de 1000 caracteres devuelve `422` con un
     mensaje claro en vez de procesarse.

4. **Verificación opcional con Qwen (vía Groq)**: por defecto está
   desactivada. Ya no corre un modelo local — usa la API gratuita de
   [Groq](https://console.groq.com/) (compatible con el SDK de OpenAI)
   para llamar a Qwen3-32B de forma remota. Para habilitarla:

   1. Crea una cuenta en [console.groq.com](https://console.groq.com/) y
      genera una API key (sección "API Keys").
   2. En tu `.env`, pon `HABILITAR_QWEN=true` y `GROQ_API_KEY=` con la key
      que generaste (ver `.env.example`). Si falta la key, la verificación
      se considera no disponible aunque `HABILITAR_QWEN=true`.

   Solo se activa para categorías cuya probabilidad cae en zona dudosa
   cerca de su umbral; si la llamada a Groq falla (sin conexión, timeout,
   rate limit, key inválida), la API sigue respondiendo únicamente con la
   decisión de BETO.

5. **Tests**: desde la raíz del repo, con el entorno virtual activado:

   ```powershell
   pytest tests/
   ```

   Los tests usan el modelo real de `model_artifacts/modelo_moderacion_final/`
   (no hay mocks), así que requieren que ese modelo ya exista.
