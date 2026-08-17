# AGENTS.md — Memoria técnica y protocolo RAG para agentes

Este repositorio incluye una **memoria vectorial técnica** (Qdrant local en
`memoria/qdrant_store/`, regenerable) con tres etiquetas de estado:

| Etiqueta | Contenido |
|---|---|
| `estado_inicial` | Memoria base: resumen técnico + decisiones/dependencias/reglas (Paso 1) |
| `cambio` | Resumen técnico JSON de cada push/PR (generado por CI con `generar_resumen_cambio.py`) |
| `resumen_diario` | Síntesis diaria consolidada (job nocturno `consolidar_diario.py`) |

## Protocolo obligatorio antes de responder o actuar

1. **Busca primero, responde después.** Antes de responder a una pregunta
   técnica o de modificar código, ejecuta una búsqueda semántica en la memoria:

   ```
   venv\Scripts\python.exe memoria\rag_contexto.py "<tu consulta o intención>" --etiquetas estado_inicial,cambio,resumen_diario --k 5
   ```

2. **Incorporar el contexto.** Usa el bloque de contexto recuperado (score,
   etiqueta, origen, contenido) para fundamentar la respuesta. Si el contexto
   contradice tu conocimiento general del código, prioriza el contexto de
   memoria (es el estado real del proyecto).

3. **Citar la procedencia.** Al usar un dato de memoria, menciona su etiqueta
   y origen (p. ej. "según `cambio` de `core/backtest.py`") para trazabilidad.

4. **Uso como librería.** Si se integra en un orquestador, usar
   `memoria.rag_contexto.aumentar_prompt(system_prompt, consulta, etiquetas=[...])`
   para inyectar el contexto recuperado en el prompt activo del agente.

## Cuándo la búsqueda es obligatoria

- Preguntas sobre reglas de negocio del backtest (riesgo, stop, órdenes).
- Preguntas sobre decisiones técnicas, dependencias entre módulos o historial
  reciente de cambios (`cambio` / `resumen_diario`).
- Antes de refactorizar un módulo de `core/` (impacto en dependencias).

## Herramientas de memoria (scripts)

| Script | Función |
|---|---|
| `memoria/cargar_memoria_inicial.py` | Recarga la memoria base (etiqueta `estado_inicial`). Re-ejecutable (IDs deterministas). |
| `memoria/generar_resumen_cambio.py` | Resumen JSON de un diff/commits (etiqueta `cambio`), con fallback local. |
| `memoria/consolidar_diario.py` | Consolida los cambios del día en un único punto `resumen_diario` (`--indexar` para Qdrant). |
| `memoria/rag_contexto.py` | Búsqueda semántica RAG + `aumentar_prompt()`. |
| `memoria/consultar_memoria.py` | Consulta semántica simple con filtro de etiqueta. |

## Notas

- La BD vectorial (`memoria/qdrant_store/`) está en `.gitignore`: si no existe,
  regenerarla con `cargar_memoria_inicial.py` (descarga el modelo ONNX la
  primera vez). Los resúmenes JSON sí viajan en el repo (`memoria/cambios/`,
  `memoria/resumenes_diarios/`) y son la fuente para la BD.
- Variables de entorno del LLM (CI o local): `MEMORIA_LLM_API_KEY`,
  `MEMORIA_LLM_BASE_URL`, `MEMORIA_LLM_MODEL`. Sin clave, todo usa fallback
  heurístico local (no requiere red).
- Configuración actual del repo: proveedor **OpenCode Zen**
  (`MEMORIA_LLM_BASE_URL=https://opencode.ai/zen/v1`), modelo
  **`deepseek-v4-flash-free`** (gratuito, no consume créditos; el plan de
  suscripción no financia la API directa, que es pay-as-you-go). El endpoint
  está tras Cloudflare: los scripts ya envían un User-Agent de navegador.
  Modelos `*-free` pueden usar los datos para mejorar el modelo durante el
  periodo gratuito (ver docs de Zen); con créditos se puede volver al modelo
  de pago `deepseek-v4-flash`.
