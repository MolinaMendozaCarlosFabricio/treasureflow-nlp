"""Resumen rapido del registro de auditoria de posibles inconsistencias
razonamiento/veredicto detectadas por la capa de verificacion con Qwen
(ver src/models/qwen_verifier.py::_detectar_posible_inconsistencia).

Cada vez que Qwen confirma una categoria pero su "razon" en texto suena
negativa, queda una linea en el JSONL de auditoria -- este script la lee
con pandas y muestra un resumen rapido para revisar que tan seguido pasa
esto en la practica.

Uso (desde la raiz del repo, con el venv activado):
    python training/scripts/analizar_auditoria_qwen.py
"""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RUTA_AUDITORIA = PROJECT_ROOT / "training" / "notebooks" / "artifacts" / "qwen_auditoria.jsonl"


def main() -> None:
    if not RUTA_AUDITORIA.exists():
        print(
            f"No se encontro {RUTA_AUDITORIA}.\n"
            "Todavia no se registro ninguna inconsistencia (o la API con "
            "HABILITAR_QWEN=true no se ha corrido aun)."
        )
        return

    df = pd.read_json(RUTA_AUDITORIA, lines=True)

    if df.empty:
        print(f"{RUTA_AUDITORIA} existe pero esta vacio -- no hay inconsistencias registradas.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"=== Auditoria de inconsistencias Qwen: {RUTA_AUDITORIA} ===\n")
    print(f"Total de inconsistencias registradas: {len(df)}\n")

    print("Por categoria:")
    print(df["categoria"].value_counts().to_string())

    print("\nLas 5 mas recientes:")
    recientes = df.sort_values("timestamp", ascending=False).head(5)
    for _, fila in recientes.iterrows():
        print("\n" + "-" * 60)
        print(f"timestamp: {fila['timestamp']}")
        print(f"categoria: {fila['categoria']}")
        print(f"confirma:  {fila['confirma']}")
        print(f"texto:     {fila['texto']}")
        print(f"razon:     {fila['razon']}")


if __name__ == "__main__":
    main()
