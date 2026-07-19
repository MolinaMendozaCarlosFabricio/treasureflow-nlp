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
│   ├── utils/
│   └── worker/                                    # consumidor AMQP (procesamiento async de moderacion)
│       ├── consumidor.py                          # punto de entrada + topologia RabbitMQ
│       ├── procesador.py                          # clasifica un mensaje (reutiliza core/logic.py)
│       └── esquemas_mensajes.py                   # Pydantic para los mensajes AMQP
├── scripts/                                      # utilidades de despliegue (no son parte de la API)
│   ├── descargar_modelo.py                        # baja el modelo entrenado desde HF Hub
│   ├── exportar_modelo_onnx.py                    # experimento ONNX+INT8 (evaluado y descartado, ver README)
│   ├── consumo_ram.py                             # compara RAM: original vs. ONNX INT8
│   └── iniciar_worker.py                          # arranca el worker de moderacion (AMQP)
├── tests/
│   ├── test_moderacion.py
│   └── test_worker.py
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

## Optimización ONNX + INT8: evaluada y descartada (decisión documentada)

Se evaluó exportar el modelo BETO a formato ONNX y cuantizarlo
dinámicamente a INT8 (`scripts/exportar_modelo_onnx.py`, sigue en el repo
como referencia/experimento documentado), buscando reducir RAM y tiempo de
inferencia en el despliegue de producción en CPU.

**Resultados medidos** (con `AutoQuantizationConfig.avx2`):

| Métrica | Original (PyTorch) | ONNX INT8 | Cambio |
|---|---|---|---|
| Tamaño en disco | 420.0 MB | 106.5 MB | **-74.7%** |
| Tiempo de inferencia (CPU) | ~118 ms | ~85 ms | **-28%** |
| RAM al cargar el modelo | ~703 MB | ~856 MB | **+22%** ⚠️ |

**Decisión: NO se usa la versión ONNX en producción.** La API (`src/models/beto_classifier.py`)
carga el modelo original en PyTorch desde
`model_artifacts/modelo_moderacion_final/`, tal cual. Razón: el
procesamiento de moderación corre de forma **asíncrona vía un worker
consumidor de cola**, no en el camino crítico de una petición HTTP — así
que la mejora de velocidad de ONNX (~30ms menos por inferencia) no aporta
ningún beneficio real para quien espera la respuesta. En cambio, el
aumento de RAM (~22%) sí tiene costo directo en el modelo de facturación
de Railway, que es la métrica que importa priorizar en este entorno. Con
ese balance, la versión ONNX pierde en el único eje que realmente cuenta
acá.

Si en algún momento el procesamiento vuelve a ser síncrono (ej. respuesta
en el camino crítico de una petición HTTP donde la latencia sí importe) o
cambia el modelo de facturación del hosting, vale la pena revisitar esta
decisión — `scripts/exportar_modelo_onnx.py` y `scripts/consumo_ram.py`
siguen en el repo listos para volver a generar y comparar. Para
regenerar la versión ONNX manualmente:

```powershell
python scripts/exportar_modelo_onnx.py
```

Y para comparar RAM entre ambas versiones (modelo original vs. ONNX INT8),
uno debajo del otro, en tu propio hardware:

```powershell
python scripts/consumo_ram.py
```

(Nota sobre `avx512_vnni` vs `avx2`: el script usa `avx2` porque asume
hardware de producción genérico/desconocido, no necesariamente con
soporte para la extensión Intel AVX512-VNNI. Si en el futuro se retoma
esta optimización sobre hardware conocido con soporte VNNI, se puede
ajustar en `scripts/exportar_modelo_onnx.py`.)

## Levantar la API de moderación

Una vez que existe `model_artifacts/modelo_moderacion_final/` (generado
por el notebook de entrenamiento, o descargado con
`scripts/descargar_modelo.py` como se explica arriba), se puede servir ese
modelo con una API FastAPI sin necesidad de volver a entrenar nada.

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

## Worker de moderación (consumidor AMQP)

El procesamiento de moderación corre de forma **asíncrona**, desacoplado de
cualquier petición HTTP: una API principal (fuera de este repo) publica
publicaciones/reseñas pendientes de moderar en una cola de RabbitMQ
(CloudAMQP), y este worker las consume, clasifica con BETO (y Qwen vía
Groq si aplica), y publica el veredicto de vuelta a otra cola. **Este
worker nunca toca la base de datos ni Firebase Cloud Messaging** — eso lo
hace la API principal al consumir el resultado.

```
API principal  --publica-->  moderacion.pendiente  --consume-->  este worker
(otro repo/svc)               (cola AMQP)                        (BETO + Qwen)

este worker  --publica-->  moderacion.resultado  --consume-->  API principal
                            (cola AMQP)                          (otro repo/svc)
```

Reintentos (5 intentos, luego dead-letter): si falla el procesamiento, el
mensaje se nackea sin reencolar → cae a una cola de reintento con TTL →
al expirar el TTL vuelve solo a `moderacion.pendiente` → se reprocesa. Al
agotar 5 intentos (contados vía el header AMQP `x-death`), en vez de
reintentar de nuevo se publica en `moderacion.fallidos`.

**Correrlo localmente:**

```powershell
python scripts/iniciar_worker.py
```

- Carga el modelo BETO una sola vez al arrancar (reutiliza
  `src/models/beto_classifier.py`, el mismo módulo que usa la API HTTP —
  no hay una segunda copia de esa lógica).
- Declara toda la topología de RabbitMQ (exchange, colas, dead-letter) por
  su cuenta al conectarse; no hace falta crearla a mano en CloudAMQP.
- Al recibir `SIGINT`/`SIGTERM` (`Ctrl+C`, o la señal que manda el
  orquestador al desplegar), termina de procesar el mensaje en curso antes
  de cerrar la conexión — no corta a la mitad.

**Variables de entorno** (en tu `.env`, ver `.env.example`):

- `AMQP_URL` — **requerido**, sin valor por defecto. El worker falla
  explícitamente al arrancar (con un mensaje claro) si no está definida.
  Es el connection string de tu instancia de CloudAMQP
  (`amqps://usuario:password@host.cloudamqp.com/vhost`).
- `NOMBRE_EXCHANGE` (default `moderacion`), `COLA_ENTRADA` (default
  `moderacion.pendiente`), `COLA_SALIDA` (default `moderacion.resultado`),
  `COLA_FALLIDOS` (default `moderacion.fallidos`) — nombres de la
  topología; normalmente no hace falta tocarlos.
- `PREFETCH_COUNT` (default `4`) — cuántos mensajes sin confirmar puede
  tener el worker en vuelo a la vez. Cada mensaje puede tardar desde
  milisegundos (si solo decide BETO) hasta unos segundos (si consulta a
  Qwen vía Groq); 4 da algo de concurrencia en las esperas de red de Qwen
  sin acumular demasiados mensajes sin-ack de golpe.
- `MAX_REINTENTOS` (default `5`) y `TTL_REINTENTO_MS` (default `10000`,
  10s) — la política de reintentos acordada.

**Esquema del mensaje de entrada** (`moderacion.pendiente`, lo publica la
API principal):

```json
{
  "publicacion_id": "string o UUID, requerido",
  "texto": "string, requerido, no vacío",
  "campo": "resena | descripcion (opcional, solo para trazabilidad)"
}
```

**Esquema del mensaje de salida** (`moderacion.resultado`, lo publica este
worker):

```json
{
  "publicacion_id": "el mismo ID recibido",
  "bloqueado": true,
  "categorias": {
    "grosero": { "probabilidad": 0.02, "activado": false },
    "amenaza": { "probabilidad": 0.01, "activado": false },
    "inapropiado": { "probabilidad": 0.05, "activado": false }
  },
  "verificado_por_qwen": false,
  "detalle_verificacion": [],
  "timestamp_procesado": "2026-07-18T20:23:35.867399"
}
```

**Tests**: `pytest tests/test_worker.py` — mockean la conexión AMQP y el
modelo BETO, no requieren RabbitMQ real corriendo. El mecanismo de
reintentos (TTL + dead-letter-exchange, con recuento vía el header
`x-death`) también se validó a mano contra un CloudAMQP real durante el
desarrollo de este worker.

**Nota sobre lo que NO implementa este repo**: el caso de uso del lado de
la API principal que publica en `moderacion.pendiente`, y el consumidor
de `moderacion.resultado` que actualiza la base de datos y dispara la
notificación FCM, viven en otro servicio/repo — quedan fuera del alcance
de este proyecto.
