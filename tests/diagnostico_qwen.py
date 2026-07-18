"""
Script de diagnóstico: prueba verificar_con_qwen con un lote de textos
variados por categoría, para medir la tasa real de contradicción entre
el veredicto (confirma) y el razonamiento (razon) generado.

Ejecutar de forma aislada, sin pasar por la API, para iterar rápido:

    python -m tests.diagnostico_qwen   (desde la raiz del repo, venv activado)

No se llama "test_*.py" a propósito -- es un script de diagnóstico con
prints, no tests de pytest (no tiene asserts), y con ese prefijo pytest
lo importaría y ejecutaría por completo (10 llamadas reales a Qwen) cada
vez que corras la suite normal.
"""

import asyncio
import re
import sys

from src.models.qwen_verifier import verificar_con_qwen

# La consola de Windows suele usar cp1252 por defecto, que no puede
# imprimir los emojis de las marcas de resultado (✅ ❌ ⚠️) -- forzamos
# UTF-8 en stdout para que corra igual sin importar la codepage activa.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# IMPORTANTE: ninguno de estos textos debe ser igual ni casi idéntico a
# los ejemplos few-shot definidos en EJEMPLOS_POR_CATEGORIA, para que la
# prueba sea válida (no queremos medir memorización, sino generalización).
CASOS_PRUEBA = [
    # (texto, categoria_a_verificar, probabilidad_simulada_de_beto)
    ("Si me ayudas a cargar el material te dejo pasar a conocer mi casa completa", "amenaza", 0.3566),
    ("Si me traes el cobre antes del viernes, te dejo escoger el mejor precio", "amenaza", 0.30),
    ("Como sigan tardando tanto voy a contar en todos lados lo mal que atienden", "amenaza", 0.35),
    ("Si quieres que te compre más barato, me tienes que dar buen trato primero", "amenaza", 0.32),
    ("No se preocupen, si no llego a tiempo les aviso con anticipación", "amenaza", 0.28),
    ("Con gusto te ayudo a cargar, nada más avísame la hora que te acomode", "inapropiado", 0.20),
    ("Te espero en mi domicilio particular, ven cuando gustes, estaré disponible", "inapropiado", 0.22),
    ("Buena atención, aunque el trato fue un poco frío para mi gusto", "inapropiado", 0.18),
    ("Excelente trato, se nota que le pone ganas al trabajo diario", "inapropiado", 0.15),
    ("Deja de molestarme con tus mensajes o vas a tener problemas serios", "grosero", 0.45),
]

def detectar_contradiccion(confirma, razon):
    """Heurística simple para detectar si el texto de la razón contradice
    el valor booleano de confirma -- misma lógica que la salvaguarda."""
    negaciones = ["no implica", "no es", "no representa", "no expresa",
                  "no se trata de", "no constituye"]
    razon_lower = razon.lower()
    tiene_negacion = any(neg in razon_lower for neg in negaciones)

    if confirma is True and tiene_negacion:
        return True
    return False


async def main():
    resultados = []
    for texto, categoria, probabilidad in CASOS_PRUEBA:
        confirma, razon = await verificar_con_qwen(texto, categoria, probabilidad)
        contradice = detectar_contradiccion(confirma, razon) if confirma is not None else False
        resultados.append({
            "texto": texto,
            "categoria": categoria,
            "confirma": confirma,
            "razon": razon,
            "contradice": contradice,
        })
        marca = "⚠️ CONTRADICCIÓN" if contradice else ("❌ NO PARSEÓ" if confirma is None else "✅")
        print(f"\n{marca}")
        print(f"Texto: {texto}")
        print(f"Categoría: {categoria} | Confirma: {confirma}")
        print(f"Razón: {razon}")

    # Resumen final
    total = len(resultados)
    n_contradicciones = sum(1 for r in resultados if r["contradice"])
    n_no_parseo = sum(1 for r in resultados if r["confirma"] is None)

    print("\n" + "=" * 60)
    print(f"RESUMEN: {total} casos probados")
    print(f"  Contradicciones detectadas: {n_contradicciones} ({n_contradicciones/total*100:.1f}%)")
    print(f"  Fallos de parseo: {n_no_parseo} ({n_no_parseo/total*100:.1f}%)")
    print(f"  Respuestas aparentemente consistentes: {total - n_contradicciones - n_no_parseo}")


if __name__ == "__main__":
    asyncio.run(main())