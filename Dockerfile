FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema mínimas que suelen necesitar torch/transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Descarga el modelo y arranca el worker -- en ese orden, cada vez que
# arranca el contenedor (el script ya es idempotente: si el modelo ya
# existe en el volumen, no lo vuelve a descargar)
CMD ["sh", "-c", "python scripts/descargar_modelo.py && python scripts/iniciar_worker.py"]