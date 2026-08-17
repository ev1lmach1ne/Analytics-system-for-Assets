"""
consultar_memoria.py
Consulta semántica sobre la memoria vectorial del proyecto (Qdrant local).

Uso:
  venv\\Scripts\\python.exe memoria\\consultar_memoria.py "regla de riesgo del backtest"
  venv\\Scripts\\python.exe memoria\\consultar_memoria.py "stop loss" --k 5
  venv\\Scripts\\python.exe memoria\\consultar_memoria.py "dependencias" --etiqueta estado_inicial
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from memoria.cargar_memoria_inicial import MODELO, PREFIJO_PASSAJE, QDRANT_DIR, COLECCION  # noqa: E402

PREFIJO_CONSULTA = ""   # este modelo no usa prefijos


def main():
    parser = argparse.ArgumentParser(description="Consulta semántica sobre la memoria del proyecto")
    parser.add_argument("consulta", help="Texto de la consulta")
    parser.add_argument("--dir", default=QDRANT_DIR)
    parser.add_argument("--coleccion", default=COLECCION)
    parser.add_argument("--etiqueta", default=None, help="Filtra por etiqueta (ej. estado_inicial)")
    parser.add_argument("--k", type=int, default=3, help="Número de resultados")
    args = parser.parse_args()

    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from fastembed import TextEmbedding

    cliente = QdrantClient(path=args.dir)
    if not cliente.collection_exists(args.coleccion):
        print(f"ERROR: la colección '{args.coleccion}' no existe en {args.dir}. "
              "Ejecuta primero memoria/cargar_memoria_inicial.py")
        sys.exit(1)

    embedder = TextEmbedding(model_name=MODELO)
    vector = next(embedder.embed([PREFIJO_CONSULTA + args.consulta]))

    filtro = None
    if args.etiqueta:
        filtro = Filter(must=[FieldCondition(key="etiqueta", match=MatchValue(value=args.etiqueta))])

    hits = cliente.query_points(
        collection_name=args.coleccion,
        query=vector,
        query_filter=filtro,
        limit=args.k,
        with_payload=True,
    ).points

    print(f"\nResultados para: {args.consulta}\n")
    for i, h in enumerate(hits, 1):
        p = h.payload
        print(f"[{i}] score={h.score:.4f} | {p.get('tipo')} | {p.get('origen')} | {p.get('seccion')}")
        print(f"    {p.get('contenido', '')[:300].replace(chr(10), ' ')}")
        print()


if __name__ == "__main__":
    main()
