"""
cargar_memoria_inicial.py
Carga la memoria base del proyecto en Qdrant (modo local embebido, sin servidor).

Ingiere por defecto los documentos generados en el Paso 1:
  - RESUMEN_TECNICO.md      -> un punto por sección del resumen técnico
  - MEMORIA_INICIAL.json    -> un punto por ítem (DTC / DEP / RGN)
y etiqueta todos los puntos con la etiqueta de estado proporcionada
(por defecto 'estado_inicial').

Uso:
  venv\\Scripts\\python.exe memoria\\cargar_memoria_inicial.py
  venv\\Scripts\\python.exe memoria\\cargar_memoria_inicial.py --recrear
  venv\\Scripts\\python.exe memoria\\cargar_memoria_inicial.py --docs otro.md --etiqueta memoria_v2

Nota sobre imágenes: el Paso 1 no generó imágenes. Los embeddings de imagen
requieren un modelo multimodal (CLIP) de dimensiones distintas, por lo que
necesitarían una colección aparte; aquí solo se indexa texto.
"""
import argparse
import hashlib
import json
import os
import sys
import uuid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384 dims, ~50 idiomas (español incluido)
PREFIJO_PASSAJE = ""  # este modelo no usa prefijos; vacío para compatibilidad

DOCUMENTOS_DEFECTO = [
    os.path.join(PROJECT_ROOT, "RESUMEN_TECNICO.md"),
    os.path.join(PROJECT_ROOT, "MEMORIA_INICIAL.json"),
]

QDRANT_DIR = os.path.join(PROJECT_ROOT, "memoria", "qdrant_store")
COLECCION = "memoria_proyecto"
TAMANO_LOTE = 64
MAX_CHAR_SECCION = 1800


def _id_determinista(*partes):
    """UUID v5-like determinista a partir de sha1 (los IDs de Qdrant local
    deben ser UUID o uint64)."""
    digest = hashlib.sha1("|".join(partes).encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex=digest[:32]))


def _seccionar_markdown(ruta):
    """Divide un .md por encabezados ##; trocea secciones largas por párrafos."""
    with open(ruta, encoding="utf-8") as f:
        lineas = f.read().splitlines()
    secciones = []
    actual = []
    titulo = None
    for linea in lineas:
        if linea.startswith("## "):
            if actual:
                secciones.append((titulo or "", "\n".join(actual).strip()))
            titulo = linea[3:].strip()
            actual = []
        else:
            if linea.startswith("# "):
                continue
            actual.append(linea)
    if actual:
        secciones.append((titulo or "", "\n".join(actual).strip()))

    trozos = []
    for tit, cuerpo in secciones:
        if len(cuerpo) <= MAX_CHAR_SECCION:
            trozos.append((tit, cuerpo))
            continue
        parrafo = []
        n = 0
        for p in cuerpo.split("\n\n"):
            parrafo.append(p)
            n += len(p)
            if n >= MAX_CHAR_SECCION:
                trozos.append((tit, "\n\n".join(parrafo).strip()))
                parrafo = []
                n = 0
        if parrafo:
            trozos.append((tit, "\n\n".join(parrafo).strip()))
    return [(t or os.path.basename(ruta), c) for t, c in trozos if c]


def _seccionar_json(ruta):
    """Un punto por ítem de las tres listas de MEMORIA_INICIAL.json."""
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    puntos = []
    secciones = [
        ("decisiones_tecnicas_clave", "decision", "DTC"),
        ("dependencias_criticas", "detalle", "DEP"),
        ("reglas_negocio_principales", "regla", "RGN"),
    ]
    for clave, campo_principal, _ in secciones:
        for item in datos.get(clave, []):
            partes = [f"{campo_principal}: {item.get(campo_principal, '')}"]
            for campo in ("detalle", "origen", "destino", "riesgo_si_cambia", "decision", "regla", "archivos"):
                if campo != campo_principal and item.get(campo):
                    if isinstance(item[campo], list):
                        partes.append(f"{campo}: {', '.join(item[campo])}")
                    else:
                        partes.append(f"{campo}: {item[campo]}")
            puntos.append((item.get("id", "item"), "\n".join(partes)))
    return puntos


def _cargar_documento(ruta, seccionador, tipo, etiqueta, embedder):
    """Devuelve lista de (contenido, payload) para un documento."""
    puntos = []
    for seccion, contenido in seccionador(ruta):
        if not contenido.strip():
            continue
        puntos.append((
            contenido,
            {
                "etiqueta": etiqueta,
                "tipo": tipo,
                "origen": os.path.basename(ruta),
                "seccion": seccion,
                "ruta": ruta,
                "contenido": contenido,
            },
        ))
    return puntos


def main():
    parser = argparse.ArgumentParser(description="Carga la memoria base en Qdrant local")
    parser.add_argument("--dir", default=QDRANT_DIR, help="Carpeta de almacenamiento local de Qdrant")
    parser.add_argument("--coleccion", default=COLECCION, help="Nombre de la colección")
    parser.add_argument("--etiqueta", default="estado_inicial", help="Etiqueta de estado para los puntos")
    parser.add_argument("--docs", nargs="*", default=[], help="Documentos .md/.json adicionales a indexar")
    parser.add_argument("--recrear", action="store_true", help="Borra y recrea la colección antes de cargar")
    args = parser.parse_args()

    from qdrant_client import QdrantClient
    from qdrant_client.models import (Distance, PayloadSchemaType, PointStruct,
                                      VectorParams)
    from fastembed import TextEmbedding

    os.makedirs(args.dir, exist_ok=True)
    cliente = QdrantClient(path=args.dir)

    embedder = TextEmbedding(model_name=MODELO)
    dim = len(next(embedder.embed(["dim"])))

    if args.recrear or cliente.collection_exists(args.coleccion):
        if cliente.collection_exists(args.coleccion):
            if args.recrear:
                cliente.delete_collection(args.coleccion)
                print(f"Colección '{args.coleccion}' recreada.")
            else:
                print(f"Colección '{args.coleccion}' ya existe; re-cargo por upsert (IDs deterministas).")
    if not cliente.collection_exists(args.coleccion):
        cliente.create_collection(
            collection_name=args.coleccion,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        cliente.create_payload_index(args.coleccion, "etiqueta", field_schema=PayloadSchemaType.KEYWORD)
        print(f"Colección '{args.coleccion}' creada (dim={dim}, distancia=coseno).")

    rutas = DOCUMENTOS_DEFECTO + [os.path.join(PROJECT_ROOT, d) for d in args.docs]
    registros = []
    for ruta in rutas:
        if not os.path.exists(ruta):
            print(f"AVISO: no existe '{ruta}', se omite.")
            continue
        if ruta.endswith(".md"):
            registros += _cargar_documento(ruta, _seccionar_markdown, "documento", args.etiqueta, embedder)
        elif ruta.endswith(".json"):
            registros += _cargar_documento(ruta, _seccionar_json, "json_item", args.etiqueta, embedder)
        else:
            print(f"AVISO: tipo no soportado '{ruta}', se omite (imágenes requieren colección multimodal).")

    print(f"Generando embeddings para {len(registros)} fragmentos ({MODELO})...")
    textos = [PREFIJO_PASSAJE + t for t, _ in registros]
    vectores = list(embedder.embed(textos, batch_size=32))

    puntos = [
        PointStruct(
            id=_id_determinista(args.etiqueta, payload["origen"], payload["seccion"], payload["contenido"][:80]),
            vector=v,
            payload=payload,
        )
        for v, (_, payload) in zip(vectores, registros)
    ]

    for i in range(0, len(puntos), TAMANO_LOTE):
        cliente.upsert(args.coleccion, puntos[i:i + TAMANO_LOTE])

    info = cliente.get_collection(args.coleccion)
    print(f"Cargados {len(puntos)} puntos con etiqueta '{args.etiqueta}'.")
    print(f"Colección '{args.coleccion}': {info.points_count} puntos, "
          f"dim={info.config.params.vectors.size}.")

    por_origen = {}
    for _, p in registros:
        por_origen[p["origen"]] = por_origen.get(p["origen"], 0) + 1
    for origen, n in sorted(por_origen.items()):
        print(f"  - {origen}: {n} fragmentos")


if __name__ == "__main__":
    main()
