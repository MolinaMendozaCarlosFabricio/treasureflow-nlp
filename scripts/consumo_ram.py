"""Compara el consumo de RAM al cargar el modelo original (PyTorch) vs. la
version exportada a ONNX + cuantizada a INT8 (ver
scripts/exportar_modelo_onnx.py). Corre en dos procesos separados (uno
por version) para que la medicion de cada uno no arrastre la memoria ya
reservada por el otro."""

import os
import subprocess
import sys

RUTA_ORIGINAL = "model_artifacts/modelo_moderacion_final"
RUTA_ONNX_INT8 = "model_artifacts/modelo_moderacion_final_onnx_int8"


def _medir_en_subproceso(codigo: str) -> str:
    """Corre `codigo` en un interprete nuevo y devuelve su stdout -- asi
    cada medicion arranca desde un proceso limpio, sin memoria ya
    reservada por el modelo anterior."""
    resultado = subprocess.run(
        [sys.executable, "-c", codigo],
        capture_output=True,
        text=True,
        check=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    return resultado.stdout.strip()


CODIGO_ORIGINAL = f"""
import psutil, os
proceso = psutil.Process(os.getpid())
ram_antes = proceso.memory_info().rss / (1024 ** 2)

from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("{RUTA_ORIGINAL}")
tokenizer = AutoTokenizer.from_pretrained("{RUTA_ORIGINAL}")

ram_despues = proceso.memory_info().rss / (1024 ** 2)
print(f"{{ram_antes:.1f}}|{{ram_despues:.1f}}")
"""

CODIGO_ONNX_INT8 = f"""
import psutil, os
proceso = psutil.Process(os.getpid())
ram_antes = proceso.memory_info().rss / (1024 ** 2)

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
model = ORTModelForSequenceClassification.from_pretrained("{RUTA_ONNX_INT8}")
tokenizer = AutoTokenizer.from_pretrained("{RUTA_ONNX_INT8}")

ram_despues = proceso.memory_info().rss / (1024 ** 2)
print(f"{{ram_antes:.1f}}|{{ram_despues:.1f}}")
"""


def _reportar(nombre: str, salida: str) -> float:
    ram_antes_str, ram_despues_str = salida.split("|")
    ram_antes, ram_despues = float(ram_antes_str), float(ram_despues_str)
    consumo = ram_despues - ram_antes

    print(f"--- {nombre} ---")
    print(f"RAM antes de cargar el modelo:     {ram_antes:.1f} MB")
    print(f"RAM después de cargar el modelo:   {ram_despues:.1f} MB")
    print(f"RAM consumida por el modelo:       {consumo:.1f} MB")
    print()
    return consumo


def main() -> None:
    print("Midiendo modelo original (PyTorch)...\n")
    consumo_original = _reportar("Modelo original (PyTorch)", _medir_en_subproceso(CODIGO_ORIGINAL))

    print("Midiendo modelo ONNX cuantizado INT8...\n")
    consumo_onnx = _reportar(
        "Modelo ONNX cuantizado INT8", _medir_en_subproceso(CODIGO_ONNX_INT8)
    )

    print("=== Comparación ===")
    print(f"Original (PyTorch):     {consumo_original:.1f} MB")
    print(f"ONNX cuantizado INT8:   {consumo_onnx:.1f} MB")
    if consumo_original > 0:
        reduccion_pct = (1 - consumo_onnx / consumo_original) * 100
        print(f"Reducción: {reduccion_pct:.1f}%")


if __name__ == "__main__":
    main()
