"""Exporta el modelo BETO ya entrenado
(model_artifacts/modelo_moderacion_final/) a formato ONNX y le aplica
cuantizacion dinamica a INT8, para reducir el consumo de RAM y el tiempo
de inferencia en CPU (el despliegue de produccion no tiene GPU).

NO reentrena nada -- solo convierte el checkpoint de PyTorch ya
entrenado. Correr una sola vez despues de cada reentrenamiento del
modelo original (cuando cambia model_artifacts/modelo_moderacion_final/):

    python scripts/exportar_modelo_onnx.py

Genera model_artifacts/modelo_moderacion_final_onnx_int8/, que es lo que
carga src/models/beto_classifier.py por defecto.
"""

from pathlib import Path

from optimum.onnxruntime import (
    AutoQuantizationConfig,
    ORTModelForSequenceClassification,
    ORTQuantizer,
)
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUTA_MODELO_ORIGINAL = PROJECT_ROOT / "model_artifacts" / "modelo_moderacion_final"
RUTA_MODELO_ONNX_INT8 = PROJECT_ROOT / "model_artifacts" / "modelo_moderacion_final_onnx_int8"


def _tamano_carpeta_mb(ruta: Path) -> float:
    """Suma el tamano de todos los archivos dentro de la carpeta (recursivo)."""
    total_bytes = sum(archivo.stat().st_size for archivo in ruta.rglob("*") if archivo.is_file())
    return total_bytes / (1024**2)


def main() -> None:
    if not RUTA_MODELO_ORIGINAL.exists():
        raise SystemExit(
            f"No se encontro el modelo original en {RUTA_MODELO_ORIGINAL}. "
            "Corre el notebook de entrenamiento o scripts/descargar_modelo.py primero."
        )

    RUTA_MODELO_ONNX_INT8.mkdir(parents=True, exist_ok=True)

    print(f"Exportando modelo desde {RUTA_MODELO_ORIGINAL} a ONNX...")
    modelo_onnx = ORTModelForSequenceClassification.from_pretrained(
        RUTA_MODELO_ORIGINAL, export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO_ORIGINAL)

    print("Aplicando cuantizacion dinamica a INT8...")

    # NOTA sobre la config: la sugerencia original era avx512_vnni, pero esa
    # asume que el CPU de produccion soporta la extension especifica Intel
    # AVX512-VNNI. Como el hardware de despliegue es generico/desconocido y
    # no hay forma de confirmar soporte VNNI de antemano, usamos avx2 --
    # segun la propia documentacion de optimum, avx512 (no-VNNI) puede sufrir
    # saturacion en la instruccion VPMADDUBSW, y avx2 es la config dinamica
    # seria para CPUs x86-64 genericas sin ese riesgo.
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

    quantizer = ORTQuantizer.from_pretrained(modelo_onnx)
    quantizer.quantize(
        quantization_config=qconfig,
        save_dir=str(RUTA_MODELO_ONNX_INT8),
        file_suffix=None,
    )

    tokenizer.save_pretrained(str(RUTA_MODELO_ONNX_INT8))

    tamano_original_mb = _tamano_carpeta_mb(RUTA_MODELO_ORIGINAL)
    tamano_onnx_int8_mb = _tamano_carpeta_mb(RUTA_MODELO_ONNX_INT8)
    reduccion_pct = (
        (1 - tamano_onnx_int8_mb / tamano_original_mb) * 100 if tamano_original_mb else 0.0
    )

    print("\n=== Tamano en disco ===")
    print(f"Modelo original (PyTorch):   {tamano_original_mb:.1f} MB  ({RUTA_MODELO_ORIGINAL})")
    print(f"Modelo ONNX cuantizado INT8: {tamano_onnx_int8_mb:.1f} MB  ({RUTA_MODELO_ONNX_INT8})")
    print(f"Reduccion: {reduccion_pct:.1f}%")

    print(f"\nListo. Modelo optimizado guardado en: {RUTA_MODELO_ONNX_INT8}")


if __name__ == "__main__":
    main()
