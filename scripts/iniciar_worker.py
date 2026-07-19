"""Punto de entrada para correr el worker de moderacion localmente:

    python scripts/iniciar_worker.py

Delega toda la logica a src/worker/consumidor.py -- este script solo
existe para tener un comando corto y consistente con el resto de
scripts/ del proyecto (ej. scripts/descargar_modelo.py).
"""

import sys
from pathlib import Path

# Permite correr el script directamente (python scripts/iniciar_worker.py)
# sin depender de -m ni de tener el repo instalado como paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.worker.consumidor import main  # noqa: E402

if __name__ == "__main__":
    main()
