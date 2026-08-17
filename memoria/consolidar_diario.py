"""
consolidar_diario.py
Consolidador diario de memoria técnica. Toma los resúmenes de cambio del día
(memoria/cambios/<fecha>_<sha>.json), los pasa por un prompt de síntesis diaria
y genera UN único punto consolidado con la etiqueta 'resumen_diario'.

Usos:
  - Local:  venv\\Scripts\\python.exe memoria\\consolidar_diario.py --indexar
  - CI nocturno (GitHub Actions / GitLab schedule): ver
    .github/workflows/consolidacion_diaria.yml y el job 'consolidacion-diaria'
    de .gitlab-ci.yml.

Configuración LLM (compatible OpenAI): mismas variables que
generar_resumen_cambio.py (MEMORIA_LLM_API_KEY / BASE_URL / MODEL).
Sin clave, usa una agregación heurística local (fallback).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from memoria.cargar_memoria_inicial import _id_determinista  # noqa: E402
from memoria.generar_resumen_cambio import (PROJECT_ROOT, _llamada_llm,
                                            _normalizar_json)  # noqa: E402

CAMBOS_DIR = os.path.join(PROJECT_ROOT, "memoria", "cambios")
RESUMENES_DIR = os.path.join(PROJECT_ROOT, "memoria", "resumenes_diarios")

PROMPT_DIARIO = """Eres el consolidador diario de memoria técnica del proyecto.
Recibes los resúmenes de todos los cambios técnicos registrados en la fecha {fecha}.
Sintetízalos en un ÚNICO resumen diario consolidado.

Cambios del día:
{cambios}

Genera una síntesis en formato JSON con la siguiente estructura:
- resumen_diario: Frase concisa que resume el avance técnico del día.
- temas_principales: Lista de 2 a 5 temas o áreas tocadas.
- modulos_afectados: Lista de módulos del proyecto impactados.
- decisiones_clave: Lista de las decisiones técnicas más relevantes del día.
- tags: 3 a 5 palabras clave para búsqueda semántica del día."""

SYSTEM_DIARIO = ("Devuelve SOLO JSON válido, sin texto adicional ni bloques markdown. "
                 "Claves exactas: resumen_diario (str), temas_principales (lista de str), "
                 "modulos_afectados (lista de str), decisiones_clave (lista de str), "
                 "tags (lista de 3 a 5 str).")


def _cambios_de(fecha, dir_cambios):
    """Carga los JSON de resúmenes cuyo nombre empieza por <fecha>_."""
    cambios = []
    if not os.path.isdir(dir_cambios):
        return cambios
    prefijo = f"{fecha}_"
    for nombre in sorted(os.listdir(dir_cambios)):
        if not (nombre.startswith(prefijo) and nombre.endswith(".json")):
            continue
        try:
            with open(os.path.join(dir_cambios, nombre), encoding="utf-8") as f:
                cambios.append((nombre, json.load(f)))
        except (OSError, json.JSONDecodeError) as e:
            print(f"AVISO: no se pudo leer {nombre} ({e})")
    return cambios


def _texto_para_prompt(cambios):
    bloques = []
    for nombre, datos in cambios:
        bloques.append(
            f"- {nombre}: {datos.get('resumen_ejecutivo', '')}\n"
            f"    modificaciones: "
            + ", ".join(f"{m.get('archivo')} ({m.get('motivo', '')})"
                        for m in datos.get("modificaciones_clave", [])[:4])
            + f"\n    impacto: {', '.join(datos.get('impacto_tecnico', []))}"
            + f"\n    tags: {', '.join(datos.get('tags', []))}"
        )
    return "\n".join(bloques)


def _sintetizar_con_llm(fecha, cambios):
    prompt = PROMPT_DIARIO.format(fecha=fecha, cambios=_texto_para_prompt(cambios))
    texto = _llamada_llm(prompt, system=SYSTEM_DIARIO)
    if not texto:
        return None
    try:
        datos = _normalizar_json(texto)
        for clave in ("resumen_diario", "temas_principales", "modulos_afectados",
                      "decisiones_clave", "tags"):
            if clave not in datos:
                raise ValueError(f"falta '{clave}'")
        return datos
    except Exception as e:
        print(f"AVISO: síntesis del LLM no válida ({e}); se usa la agregación local.")
        return None


def _sintetizar_fallback(fecha, cambios):
    conteo_tags = {}
    modulos = []
    decisiones = []
    for _, datos in cambios:
        for t in datos.get("tags", []):
            conteo_tags[t] = conteo_tags.get(t, 0) + 1
        for m in datos.get("impacto_tecnico", []):
            if m not in modulos:
                modulos.append(m)
        ej = datos.get("resumen_ejecutivo", "")
        if ej and ej not in decisiones:
            decisiones.append(ej)
    top_tags = [t for t, _ in sorted(conteo_tags.items(),
                                     key=lambda x: -x[1])[:5]]
    return {
        "resumen_diario": f"{len(cambios)} cambio(s) técnico(s) consolidados el {fecha}.",
        "temas_principales": top_tags or ["sin cambios"],
        "modulos_afectados": modulos or ["—"],
        "decisiones_clave": decisiones or ["sin cambios registrados"],
        "tags": top_tags or ["sin_cambios"],
    }


def _indexar_qdrant(fecha, datos, args):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        from fastembed import TextEmbedding
        from memoria.cargar_memoria_inicial import (COLECCION, MODELO, PREFIJO_PASSAJE,
                                                    QDRANT_DIR, _id_determinista)
    except ImportError as e:
        print(f"AVISO: no se pudo indexar en Qdrant ({e}).")
        return False

    coleccion = args.coleccion or COLECCION
    cliente = QdrantClient(path=args.qdrant_dir or QDRANT_DIR)
    embedder = TextEmbedding(model_name=MODELO)
    dim = len(next(embedder.embed(["d"])))
    if not cliente.collection_exists(coleccion):
        cliente.create_collection(
            collection_name=coleccion,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    contenido = (f"Resumen diario {fecha}:\n" + datos["resumen_diario"] +
                 "\nTemas: " + ", ".join(datos["temas_principales"]) +
                 "\nMódulos: " + ", ".join(datos["modulos_afectados"]))
    vector = next(embedder.embed([PREFIJO_PASSAJE + contenido]))
    punto = PointStruct(
        id=_id_determinista("resumen_diario", fecha),
        vector=vector,
        payload={
            "etiqueta": "resumen_diario",
            "tipo": "resumen_diario",
            "fecha": fecha,
            "origen": "consolidador_diario",
            "n_cambios": len(datos.get("decisiones_clave", [])),
            "contenido": contenido,
            "json": json.dumps(datos, ensure_ascii=False, indent=2),
        },
    )
    cliente.upsert(coleccion, [punto])
    cliente.close()
    print(f"Punto consolidado indexado en Qdrant (colección '{coleccion}', "
          f"etiqueta 'resumen_diario', fecha {fecha}).")
    return True


def main():
    parser = argparse.ArgumentParser(description="Consolida los cambios del día en un resumen diario")
    parser.add_argument("--dir", default=CAMBOS_DIR, help="Carpeta con los resúmenes de cambio")
    parser.add_argument("--fecha", default=None,
                        help="Fecha YYYY-MM-DD a consolidar (por defecto: ayer si se omite? sí)")
    parser.add_argument("--salida", default=None, help="Ruta del JSON resultante")
    parser.add_argument("--indexar", action="store_true", help="Indexa en la memoria vectorial local")
    parser.add_argument("--qdrant-dir", default=None)
    parser.add_argument("--coleccion", default=None)
    parser.add_argument("--sin-llm", action="store_true", help="Fuerza la agregación local")
    args = parser.parse_args()

    # Si no se indica fecha, se consolida el día anterior (típico de un cron nocturno)
    fecha = args.fecha or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    cambios = _cambios_de(fecha, args.dir)
    if not cambios:
        print(f"No hay cambios registrados para {fecha} en {args.dir}. Nada que consolidar.")
        return

    print(f"Consolidando {len(cambios)} cambio(s) de {fecha}...")
    datos = None
    if not args.sin_llm:
        datos = _sintetizar_con_llm(fecha, cambios)
    if datos is None:
        print("Generando agregación local (fallback).")
        datos = _sintetizar_fallback(fecha, cambios)

    salida = json.dumps({"etiqueta": "resumen_diario", "fecha": fecha, **datos},
                        ensure_ascii=False, indent=2)
    ruta = args.salida or os.path.join(RESUMENES_DIR, f"{fecha}.json")
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(salida)
    print(f"Resumen diario guardado en {ruta}")

    if args.indexar:
        _indexar_qdrant(fecha, datos, args)
    else:
        print("Sugerencia: usa --indexar para volcarlo también en Qdrant con etiqueta 'resumen_diario'.")


if __name__ == "__main__":
    main()
