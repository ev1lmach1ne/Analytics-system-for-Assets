"""
generar_resumen_cambio.py
Agente archivista de memoria técnica. Toma un 'git diff' y los mensajes de
commit y genera una síntesis técnica en JSON con la estructura:
  resumen_ejecutivo / modificaciones_clave / impacto_tecnico / tags

Diseñado para ejecutarse:
  - En CI/CD (GitHub Actions, GitLab CI) tras cada push o pull_request
    (ver .github/workflows/memoria_tecnica.yml y .gitlab-ci.yml).
  - Localmente sobre el diff de la rama de trabajo.

Configuración vía variables de entorno (compatible con cualquier API
OpenAI-compatible: OpenAI, OpenRouter, LM Studio, Ollama...):
  MEMORIA_LLM_API_KEY   clave de API (obligatoria para modo LLM)
  MEMORIA_LLM_BASE_URL  URL base, por defecto https://api.openai.com/v1
  MEMORIA_LLM_MODEL     modelo, por defecto gpt-4o-mini
  MEMORIA_LLM_TIMEOUT   segundos de espera, por defecto 120

Sin clave configurada genera un resumen heurístico local (modo fallback),
útil para desarrollo y como respaldo de CI.

Opcionalmente indexa el resumen en la memoria vectorial local (Qdrant) con
la etiqueta 'cambio', conectando con la memoria base 'estado_inicial'.

Uso:
  venv\\Scripts\\python.exe memoria\\generar_resumen_cambio.py --diff diff.txt --commits commits.txt
  venv\\Scripts\\python.exe memoria\\generar_resumen_cambio.py --diff diff.txt --salida resumen.json
  venv\\Scripts\\python.exe memoria\\generar_resumen_cambio.py --diff diff.txt --indexar
  git diff HEAD~1 | venv\\Scripts\\python.exe memoria\\generar_resumen_cambio.py
"""
import argparse
import json
import os
import re
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MAX_DIFF_CHARS = 40000   # tope de caracteres del diff enviado al LLM
MAX_COMMITS_CHARS = 6000

PROMPT_TEMPLATE = """Eres un agente archivista de memoria técnica. Revisa el siguiente 'git diff' y los mensajes de commit recibidos:
{contenido}

Genera una síntesis técnica en formato JSON con la siguiente estructura:
- resumen_ejecutivo: Frase concisa del cambio realizado.
- modificaciones_clave: Lista de archivos o funciones modificadas y el porqué.
- impacto_tecnico: Módulos que podrían verse afectados por este cambio.
- tags: 3 a 5 palabras clave para búsqueda semántica.
{nota_truncado}"""

SYSTEM_MSG = ("Devuelve SOLO JSON válido, sin texto adicional ni bloques markdown. "
              "El JSON debe tener exactamente estas claves: resumen_ejecutivo (str), "
              "modificaciones_clave (lista de objetos con 'archivo' y 'motivo'), "
              "impacto_tecnico (lista de strings), tags (lista de 3 a 5 strings).")

# Mapa de rutas -> módulos para el resumen heurístico local y el impacto
MODULOS_RUTAS = [
    ("core/backtest.py", "motor de backtest (core/backtest.py)"),
    ("core/strategies.py", "registro de estrategias e indicadores (core/strategies.py)"),
    ("core/optimizer.py", "optimizador de parámetros IS (core/optimizer.py)"),
    ("core/metrics.py", "métricas cuantitativas (core/metrics.py)"),
    ("core/candle_patterns.py", "detección de patrones de velas (core/candle_patterns.py)"),
    ("core/questdb_manager.py", "gestión de QuestDB (core/questdb_manager.py)"),
    ("core/config.py", "configuración global y rutas (core/config.py)"),
    ("core/codegen/", "exportador de código Pine/MQL5 (core/codegen/)"),
    ("core/data_providers/", "proveedores de datos (core/data_providers/)"),
    ("core/connectors/", "conectores externos (core/connectors/)"),
    ("core/parsing.py", "parseo de datos (core/parsing.py)"),
    ("core/rf_registry.py", "registro de tasa libre de riesgo (core/rf_registry.py)"),
    ("gui/main_window.py", "ventana principal de la GUI (gui/main_window.py)"),
    ("gui/widgets/tab_backtest.py", "pestaña de backtest (gui/widgets/tab_backtest.py)"),
    ("gui/widgets/", "pestañas y widgets de la GUI (gui/widgets/)"),
    ("gui/dialogs/", "diálogos de la GUI (gui/dialogs/)"),
    ("gui/", "interfaz gráfica PyQt6 (gui/)"),
    ("tests/", "suite de pruebas (tests/)"),
    ("library/", "scripts de utilidad (library/)"),
    ("app.py", "punto de entrada de la aplicación (app.py)"),
    ("empaquetar.py", "empaquetado PyInstaller (empaquetar.py)"),
    ("memoria/", "sistema de memoria vectorial (memoria/)"),
    (".github/", "pipelines de CI (GitHub Actions)"),
    ("requirements.txt", "dependencias del proyecto"),
]


def _leer_entrada(args):
    diff = args.diff
    if not diff and not sys.stdin.isatty():
        diff = sys.stdin.read()
    if diff and diff.strip().startswith("diff --git"):
        pass
    elif diff and not os.path.exists(diff):
        diff = None
    elif diff and os.path.exists(diff):
        with open(diff, encoding="utf-8", errors="replace") as f:
            diff = f.read()
    contenido = diff or "(sin diff disponible)"

    commits = ""
    if args.commits:
        with open(args.commits, encoding="utf-8", errors="replace") as f:
            commits = f.read()
        contenido += "\n\n--- Mensajes de commit ---\n" + commits
    return contenido


def _truncar(contenido):
    nota = ""
    if len(contenido) > MAX_DIFF_CHARS + MAX_COMMITS_CHARS:
        contenido = (contenido[:MAX_DIFF_CHARS] +
                     "\n...[diff truncado por tamaño]..." +
                     contenido[-MAX_COMMITS_CHARS:])
        nota = (f"Nota: el contenido original era muy largo y se truncó "
                f"a {MAX_DIFF_CHARS + MAX_COMMITS_CHARS} caracteres.")
    return contenido, nota


# ══════════════ resumen heurístico local (fallback sin API) ══════════════

def _archivos_del_diff(diff):
    return re.findall(r"^diff --git a/(.+?) b/", diff, re.MULTILINE)


def _modulos_de(archivos):
    modulos = []
    for archivo in archivos:
        for prefijo, nombre in MODULOS_RUTAS:
            if archivo.startswith(prefijo):
                if nombre not in modulos:
                    modulos.append(nombre)
                break
    return modulos


def _metricas_del_diff(diff):
    lineas = diff.splitlines()
    ins = sum(1 for l in lineas if l.startswith("+") and not l.startswith("+++"))
    dele = sum(1 for l in lineas if l.startswith("-") and not l.startswith("---"))
    return ins, dele


def resumen_fallback(diff):
    archivos = _archivos_del_diff(diff)
    ins, dele = _metricas_del_diff(diff)
    modulos = _modulos_de(archivos)
    modificaciones = [
        {"archivo": a, "motivo": "modificación detectada en el diff (revisar hunks)"}
        for a in archivos
    ]
    if not modificaciones:
        modificaciones = [{"archivo": "(desconocido)", "motivo": "diff vacío o sin formato unificado"}]
    impacto = modulos or ["módulos no mapeados automáticamente"]
    etiquetas = [re.sub(r"[^a-z0-9_]", "_", a.split("/")[-1].rsplit(".", 1)[0]).lower()[:24]
                 for a in archivos[:4]]
    etiquetas = [t for t in etiquetas if t][:5]
    return {
        "resumen_ejecutivo": f"Cambio en {len(archivos)} archivo(s) ({ins}+ / {dele}- líneas).",
        "modificaciones_clave": modificaciones,
        "impacto_tecnico": impacto,
        "tags": etiquetas or ["cambio_tecnico"],
    }


# ══════════════ llamada al LLM (API compatible OpenAI) ══════════════

def _llamada_llm(prompt, system=None):
    clave = os.getenv("MEMORIA_LLM_API_KEY", "") or ""
    base = (os.getenv("MEMORIA_LLM_BASE_URL", "") or "https://api.openai.com/v1").rstrip("/")
    modelo = os.getenv("MEMORIA_LLM_MODEL", "") or "gpt-4o-mini"
    timeout = int(os.getenv("MEMORIA_LLM_TIMEOUT", "120") or "120")
    if not clave:
        return None
    cuerpo = {
        "model": modelo,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system or SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
    }
    if "openai.com" in base:
        cuerpo["response_format"] = {"type": "json_object"}
    peticion = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(cuerpo).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {clave}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(peticion, timeout=timeout) as resp:
            datos = json.loads(resp.read().decode("utf-8"))
        return datos["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"AVISO: error al llamar al LLM ({e}); se usará el resumen heurístico local.")
        return None


def _normalizar_json(texto):
    texto = texto.strip()
    if texto.startswith("```"):
        texto = re.sub(r"^```(?:json)?\s*", "", texto)
        texto = re.sub(r"\s*```$", "", texto)
    return json.loads(texto)


def _validar(datos):
    if not isinstance(datos, dict):
        raise ValueError("el resultado no es un objeto JSON")
    if not isinstance(datos.get("resumen_ejecutivo"), str):
        raise ValueError("falta 'resumen_ejecutivo' (str)")
    mods = datos.get("modificaciones_clave")
    if not isinstance(mods, list):
        raise ValueError("falta 'modificaciones_clave' (lista)")
    for m in mods:
        if not isinstance(m, dict) or not m.get("archivo"):
            raise ValueError("'modificaciones_clave' debe ser lista de {archivo, motivo}")
    if not isinstance(datos.get("impacto_tecnico"), list):
        raise ValueError("falta 'impacto_tecnico' (lista)")
    tags = datos.get("tags")
    if not (isinstance(tags, list) and 3 <= len(tags) <= 5 and all(isinstance(t, str) for t in tags)):
        raise ValueError("'tags' debe ser lista de 3 a 5 strings")
    return True


def _try_llm(prompt):
    texto = _llamada_llm(prompt)
    if not texto:
        return None
    try:
        datos = _normalizar_json(texto)
        _validar(datos)
        return datos
    except Exception as e:
        print(f"AVISO: respuesta del LLM no válida ({e}); se usará el resumen heurístico local.")
        return None


# ══════════════ indexado en la memoria vectorial local ══════════════

def _indexar_qdrant(datos, args):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        from fastembed import TextEmbedding
        from memoria.cargar_memoria_inicial import (COLECCION, MODELO, PREFIJO_PASSAJE,
                                                    QDRANT_DIR, _id_determinista)
    except ImportError as e:
        print(f"AVISO: no se pudo indexar en Qdrant ({e}); el resumen solo queda en salida.")
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
    texto = json.dumps(datos, ensure_ascii=False, indent=2)
    contenido = ("Cambio técnico:\n" + datos["resumen_ejecutivo"] +
                 "\nModificaciones: " +
                 ", ".join(f"{m['archivo']} ({m.get('motivo', '')})"
                           for m in datos["modificaciones_clave"][:5]) +
                 "\nImpacto: " + ", ".join(datos["impacto_tecnico"]))
    vector = next(embedder.embed([PREFIJO_PASSAJE + contenido]))
    punto = PointStruct(
        id=_id_determinista(args.etiqueta, args.origen or "cambio",
                            args.commit_sha or "local", datos["resumen_ejecutivo"] or "s"),
        vector=vector,
        payload={
            "etiqueta": args.etiqueta,
            "tipo": "resumen_cambio",
            "origen": args.origen or "cambio_manual",
            "commit_sha": args.commit_sha or "",
            "rama": args.rama or "",
            "contenido": contenido,
            "json": texto,
        },
    )
    cliente.upsert(coleccion, [punto])
    cliente.close()
    print(f"Resumen indexado en Qdrant (colección '{coleccion}', etiqueta '{args.etiqueta}').")
    return True


def main():
    parser = argparse.ArgumentParser(description="Resumen técnico de un git diff para la memoria del proyecto")
    parser.add_argument("--diff", default=None, help="Archivo con el git diff (o stdin si se omite)")
    parser.add_argument("--commits", default=None, help="Archivo con los mensajes de commit")
    parser.add_argument("--salida", default=None, help="Ruta donde guardar el JSON (por defecto: stdout)")
    parser.add_argument("--indexar", action="store_true", help="Indexa el resumen en la memoria vectorial local")
    parser.add_argument("--qdrant-dir", default=None, help="Carpeta del store local de Qdrant (con --indexar)")
    parser.add_argument("--coleccion", default=None, help="Colección de Qdrant (con --indexar)")
    parser.add_argument("--etiqueta", default="cambio", help="Etiqueta Qdrant para los resúmenes de cambio")
    parser.add_argument("--origen", default=None, help="Origen del cambio (rama, PR...) para el payload")
    parser.add_argument("--commit-sha", default=None, help="SHA del commit para el payload")
    parser.add_argument("--rama", default=None, help="Rama del commit para el payload")
    parser.add_argument("--sin-llm", action="store_true", help="Fuerza el resumen heurístico local")
    args = parser.parse_args()

    contenido = _leer_entrada(args)
    contenido, nota = _truncar(contenido)
    prompt = PROMPT_TEMPLATE.format(
        contenido=contenido,
        nota_truncado=nota or "",
    )

    datos = None
    if not args.sin_llm:
        datos = _try_llm(prompt)
    if datos is None:
        print("Generando resumen heurístico local (fallback).")
        datos = resumen_fallback(contenido)

    salida = json.dumps(datos, ensure_ascii=False, indent=2)
    if args.salida:
        os.makedirs(os.path.dirname(os.path.abspath(args.salida)) or ".", exist_ok=True)
        with open(args.salida, "w", encoding="utf-8") as f:
            f.write(salida)
        print(f"Resumen guardado en {args.salida}")
    else:
        print(salida)

    if args.indexar:
        _indexar_qdrant(datos, args)


if __name__ == "__main__":
    main()
