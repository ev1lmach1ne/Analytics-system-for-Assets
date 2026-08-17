"""
rag_contexto.py
Capa de recuperación (RAG) de la memoria vectorial del proyecto para los
agentes en producción. Antes de responder o actuar, el agente ejecuta una
búsqueda semántica en Qdrant sobre las colecciones de memoria (estado_inicial,
cambio, resumen_diario...) y recibe el contexto recuperado para incorporarlo a
su prompt activo.

Uso desde la capa de orquestación (línea de comandos):
  venv\\Scripts\\python.exe memoria\\rag_contexto.py "consulta"
  venv\\Scripts\\python.exe memoria\\rag_contexto.py "stop loss" --etiquetas estado_inicial,cambio,resumen_diario --k 5

Uso como librería:
  from memoria.rag_contexto import buscar_contexto, formatear_contexto, aumentar_prompt
  hits = buscar_contexto("regla de riesgo", etiquetas=["cambio"])
  contexto = formatear_contexto(hits)
  prompt_final = aumentar_prompt(system_prompt, "consulta", etiquetas=["cambio"])
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from memoria.cargar_memoria_inicial import (COLECCION, MODELO, PREFIJO_PASSAJE,
                                            QDRANT_DIR)  # noqa: E402

PREFIJO_CONSULTA = ""   # el modelo usado no requiere prefijos

ETIQUETAS_DEFECTO = ["estado_inicial", "cambio", "resumen_diario"]


def buscar_contexto(consulta, etiquetas=None, k=4, dir_store=None, coleccion=None):
    """Búsqueda semántica en Qdrant. Devuelve lista de (score, payload)."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from fastembed import TextEmbedding

    etiquetas = etiquetas or ETIQUETAS_DEFECTO
    cliente = QdrantClient(path=dir_store or QDRANT_DIR)
    coleccion = coleccion or COLECCION
    if not cliente.collection_exists(coleccion):
        cliente.close()
        return []

    embedder = TextEmbedding(model_name=MODELO)
    vector = next(embedder.embed([PREFIJO_CONSULTA + consulta]))

    filtro = None
    if etiquetas:
        filtro = Filter(should=[
            FieldCondition(key="etiqueta", match=MatchValue(value=e))
            for e in etiquetas
        ])

    hits = cliente.query_points(
        collection_name=coleccion,
        query=vector,
        query_filter=filtro,
        limit=k,
        with_payload=True,
    ).points
    cliente.close()
    return [(h.score, h.payload) for h in hits]


def formatear_contexto(hits, max_chars=600):
    """Convierte los hits en un bloque de texto listo para insertar en un prompt."""
    if not hits:
        return "(sin resultados en la memoria)"
    bloques = []
    for i, (score, payload) in enumerate(hits, 1):
        contenido = payload.get("contenido") or payload.get("seccion") or ""
        bloques.append(
            f"[{i}] (score={score:.3f}, etiqueta={payload.get('etiqueta')}, "
            f"tipo={payload.get('tipo')}, origen={payload.get('origen')}, "
            f"seccion={payload.get('seccion') or payload.get('fecha')})\n"
            f"{contenido[:max_chars]}"
        )
    return "\n\n".join(bloques)


def aumentar_prompt(system_prompt, consulta, etiquetas=None, k=4,
                    dir_store=None, coleccion=None):
    """Devuelve el system_prompt enriquecido con el contexto recuperado."""
    hits = buscar_contexto(consulta, etiquetas=etiquetas, k=k,
                           dir_store=dir_store, coleccion=coleccion)
    contexto = formatear_contexto(hits)
    return (system_prompt
            + "\n\n=== CONTEXTO RECUPERADO DE LA MEMORIA TÉCNICA (RAG) ===\n"
            + contexto
            + "\n=== FIN DEL CONTEXTO RECUPERADO ===\n"
            + "Usa este contexto si es relevante para la consulta. "
            + "Si contradice conocimiento del sistema, prioriza el contexto.")


def main():
    parser = argparse.ArgumentParser(
        description="Recupera contexto de la memoria vectorial del proyecto (RAG)")
    parser.add_argument("consulta", help="Consulta semántica")
    parser.add_argument("--etiquetas", default=",".join(ETIQUETAS_DEFECTO),
                        help="Etiquetas a filtrar separadas por coma")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--dir", default=None, help="Carpeta del store local de Qdrant")
    parser.add_argument("--coleccion", default=None)
    parser.add_argument("--formato", choices=["texto", "json"], default="texto")
    args = parser.parse_args()

    etiquetas = [e.strip() for e in args.etiquetas.split(",") if e.strip()]
    hits = buscar_contexto(args.consulta, etiquetas=etiquetas, k=args.k,
                           dir_store=args.dir, coleccion=args.coleccion)
    if args.formato == "json":
        import json
        print(json.dumps(
            [{"score": s, "etiqueta": p.get("etiqueta"), "tipo": p.get("tipo"),
              "origen": p.get("origen"), "seccion": p.get("seccion") or p.get("fecha"),
              "contenido": (p.get("contenido") or p.get("seccion") or "")[:800]}
             for s, p in hits],
            ensure_ascii=False, indent=2))
    else:
        print(formatear_contexto(hits))


if __name__ == "__main__":
    main()
