# Informe de auditoría — motor de backtest, comisiones/slippage, indicadores y anualización

Fecha: 2026-07-23. Alcance: verificar que el motor de backtest ejecuta bien las
operaciones (comisiones, slippage, tamaño de posición), que los indicadores se
calculan correctamente, y corregir la anualización de métricas para que sea
consciente de la clase de activo. Metodología: lectura línea por línea de
`core/backtest.py`, `core/strategies.py` y `gui/widgets/tab_backtest.py`, más
tests numéricos nuevos que fijan (pin) el comportamiento verificado.

## 1. Motor de backtest — CORRECTO

`_simular_numba` (`core/backtest.py:39-187`) fue revisado paso a paso:

- **Sin lookahead**: una señal en la vela `i` se ejecuta al **open de la vela
  `i+1`** (vía flags `pendiente_entrada`/`pendiente_salida`), nunca en la
  misma vela que la generó.
- **Slippage** se aplica una vez en la entrada y una vez en la salida, con la
  dirección correcta en ambos sentidos:
  - Largo: entra más caro (`o*(1+slip)`), sale más barato (`precio*(1-slip)`) — peor en los dos lados, como debe ser.
  - Corto: entra más barato (`o*(1-slip)`), sale más caro (`precio*(1+slip)`) — igualmente peor en los dos lados.
  - Verificado con test nuevo `test_slippage_precios_de_llenado_exactos_largo_y_corto` (valores de llenado exactos, no solo el PnL agregado).
- **Comisión**: se cobra una sola vez por trade, sobre el importe de entrada
  y salida juntos: `(precio_in + precio_out) * unidades * comision_pct`. No hay
  doble cobro. Verificado en largo (`test_comision_reduce_pnl`, ya existía) y
  ahora también en corto (`test_comision_en_corto_cobrada_una_vez_por_ambos_lados`, nuevo).
- **Unidades/tamaño de posición**: `unidades = (capital * riesgo_pct) / dist`,
  donde `dist` es la distancia del stop en ATR (o `2×ATR` si no hay stop
  configurado, solo para dimensionar). Esto es correcto como método de
  sizing, pero con una precisión a tener en cuenta (ver §4).
- **Unidades de %/fracción**: la conversión de porcentaje (como lo escribe el
  usuario en los spinboxes) a fracción (como lo usa el motor) ocurre una
  única vez en la GUI (`/100.0`, `tab_backtest.py`). No hay doble conversión
  ni desajuste de unidades.
- **Stop/TP/salida por tiempo**: el stop se revisa contra el low/high de la
  vela (según dirección), el TP igual, y si ambos podrían saltar en la misma
  vela, gana el stop (criterio conservador). La salida por tiempo cierra al
  close de la vela `entrada + N`.

Tests nuevos que blindan todo esto: `tests/test_indicadores.py` (indicadores)
y ampliación de `tests/test_backtest.py` con 4 tests nuevos (short PnL exacto,
comisión en corto, direcciones de slippage largo/corto, riesgo realizado vs
nominal). Suite completa: **180+ tests, todos en verde**.

## 2. Indicadores — verificados, con 3 desviaciones corregidas el 2026-08-17

Se agregó `tests/test_indicadores.py` con 9 tests numéricos directos (antes
NO existía ningún test que verificara los valores de `sma`/`ema`/`rsi`/`atr`/
`bollinger` contra un cálculo independiente — solo se probaban indirectamente
a través de señales generadas). La auditoría encontró 3 desviaciones respecto
a la definición "de libro" que en su momento se dejaron fijadas por test sin
corregir (cambiarlas altera las señales de estrategias existentes). Esas tres
desviaciones se **corrigieron el 2026-08-17** (ver la actualización al final
del documento). La tabla y el impacto siguientes documentan el estado que
tenían antes de la corrección:

| Indicador | Desviación encontrada | Estándar de referencia | Archivo |
|---|---|---|---|
| RSI | Devuelve **50** cuando no hubo pérdidas en la ventana (`perdida=0` → `rs=NaN` → `fillna(50)`) | Debería ser **100** (fuerza alcista pura) | `core/strategies.py:49-55` |
| ATR | Promedio **simple (SMA)** del True Range | Suavizado de **Wilder** (usado por TradingView/MT5/la mayoría de plataformas) | `core/strategies.py:58-62` |
| Bollinger | Desviación estándar **muestral** (`ddof=1`, default de pandas) | Desviación **poblacional** (`ddof=0`) | `core/strategies.py:65-69` |

Impacto práctico: los valores de ATR y Bollinger diferían (ligeramente, en el
caso de Bollinger; más notablemente en ATR según la volatilidad reciente) de
lo que mostraría un gráfico de TradingView/MT5 con los mismos parámetros. El
RSI=50 en vez de 100 solo importaba en rachas de subida pura sin ninguna vela
roja dentro de la ventana — un caso de borde poco frecuente pero real.

> **Corregido el 2026-08-17**: las 3 desviaciones se alinearon con el estándar
> (RSI→100 sin pérdidas, ATR→suavizado de Wilder, Bollinger→ddof=0), tanto en
> el motor como en los runtimes Pine/MT5 del código generado para mantener la
> fidelidad motor↔código generado. Detalle y archivos en la actualización al
> final del documento.

## 3. Anualización — CORREGIDA por clase de activo

**Antes**: `_velas_por_anio(df)` en `gui/widgets/tab_backtest.py` asumía
siempre 24/7/365 (`525600` minutos/año) para anualizar Sharpe y CAGR/retorno
anual, sin importar si el activo era CRYPTO, STOCK, FUTURO o FOREX. Esto
**inflaba** esas métricas anualizadas en cualquier activo que no cotice 24
horas los 365 días (acciones, futuros, forex).

**Ahora**: se agregó `velas_por_anio(tipo_activo, minutos_vela)` en
`core/config.py`, que usa sesión real por clase de activo (misma convención
que ya existía en `library/scripts_utiles/analisis_descriptivo.py`):

| Clase | Minutos de sesión/día | Días de trading/año |
|---|---|---|
| CRYPTO | 1440 (24h) | 365 |
| FUTURO | 1440 | 252 |
| STOCK | 390 (6.5h) | 252 |
| FOREX | 1440 | 252 |

`gui/widgets/tab_backtest.py` ahora determina la clase de activo del CSV
cargado (`tipo_activo_de_csv`, ya existente) y se la pasa a `_velas_por_anio`.
Si no se puede determinar la clase (CSV sin metadata reconocible), cae al
supuesto 24/7/365 anterior — no rompe nada, solo mejora cuando se puede
clasificar el activo.

Validado con 4 tests nuevos en `tests/test_config_anualizacion.py`, incluidos
casos que coinciden con los valores de la tabla `FACTORES` que ya existía en
`core/config.py` (15min CRYPTO/STOCK y 4h CRYPTO coinciden exactamente; 4h
STOCK da 409.5 contra el 409 redondeado de la tabla) — buena confirmación
cruzada independiente.

**Nota técnica**: la fórmula tiene dos regímenes — intradía (usa minutos de
sesión) y diario-o-más-lento (una vela por día de trading). Mezclarlos habría
subestimado brutalmente las velas/año en velas diarias de activos con sesión
corta (STOCK 1d habría dado ~68/año en vez de 252).

## 4. Notas menores de la auditoría (informativas, sin acción)

- **El riesgo realizado en un stop-out supera ligeramente el `riesgo_pct`
  nominal** configurado, porque el tamaño de posición (`cap*riesgo/dist`) no
  descuenta la comisión ni el slippage que se cobrarán al salir. Es un matiz
  esperable de cualquier motor de backtest con fricciones, no un bug — queda
  documentado y fijado por el test
  `test_riesgo_realizado_supera_el_riesgo_nominal_por_las_friccion`.
- **Modo sin stop** (`stop_atr=0`): el motor no coloca stop real, pero
  dimensiona la posición como si arriesgara `2×ATR`. Una sola vela adversa
  puede perder bastante más que el `riesgo_pct` configurado. Comportamiento
  documentado en el propio código (`core/backtest.py:20-22`).
- **`tf_to_minutes` trata `'30s'` como 30 minutos** (bug menor, no relacionado
  con el motor de backtest en sí — se usa solo para validar reglas de
  resampleo, no para anualización). No se tocó en esta tarea.
- **La tabla `FACTORES` en `core/config.py`** (previa a este cambio) es
  código muerto — no la usa ningún módulo del proyecto — y tiene huecos
  (sin FOREX; sin `30s`/`3m`/`2h`; sin `1d` para FUTURO/STOCK). Se deja
  intacta por compatibilidad, pero la nueva `velas_por_anio()` la reemplaza
  en la práctica para el backtest y no tiene esos huecos.
- **Sortino** no existe en `core/metrics.py` ni `core/backtest.py` — solo en
  el script legacy `library/scripts_utiles/analisis_descriptivo.py`, y ahí
  con una desviación (divide la desviación a la baja entre el conteo de
  retornos negativos, no entre el total N). Fuera del alcance de esta tarea
  (no se usa en el motor de backtest de la GUI).

## Resumen

| Área | Estado |
|---|---|
| Comisiones (largo y corto) | ✅ Correcto, verificado con tests exactos |
| Slippage (dirección largo/corto) | ✅ Correcto, verificado con tests de precio exacto |
| Tamaño de posición / riesgo | ✅ Correcto (con el matiz de fricciones documentado) |
| Ejecución sin lookahead | ✅ Correcto (open de la vela siguiente) |
| Indicadores (SMA/EMA/RSI/ATR/Bollinger/cruces) | ✅ Verificados con tests numéricos; 3 desviaciones corregidas el 2026-08-17 (RSI→100, ATR Wilder, Bollinger ddof=0) |
| Anualización Sharpe/CAGR | ✅ Corregida — ahora por clase de activo, con fallback seguro |

Archivos nuevos/modificados: `tests/test_indicadores.py` (nuevo),
`tests/test_config_anualizacion.py` (nuevo), `tests/test_backtest.py`
(ampliado), `core/config.py` (+`velas_por_anio`), `gui/widgets/tab_backtest.py`
(`_velas_por_anio` ahora recibe la ruta del CSV y usa la clase de activo).

---

## Actualización posterior — stop ATR por setup (fijo / dinámico), ATR causal y gaps

Fecha: 2026-08-13. Cambios sobre `core/backtest.py`, `core/strategies.py`,
`core/optimizer.py`, `core/codegen/*`, `gui/widgets/tab_backtest.py` y tests.

### 1. Modo del Stop Loss ×ATR por setup

- Nueva clave `stop_atr_modo` en `config_por_setup` (`'fijo'` por defecto).
- **Fijo**: comportamiento histórico — el stop se ancla a la primera entrada.
- **Dinámico por promedio**: tras cada entrada, el stop se reancla al precio
  medio ponderado de la posición con el ATR del momento.
- En ambos modos, el presupuesto de riesgo (`equity al abrir × riesgo_pct`)
  es el límite absoluto: cada tramo solo consume el riesgo que queda libre, y
  el stop dinámico queda limitado por el presupuesto (`_stop_por_presupuesto`,
  que resuelve la raíz exacta de `riesgo(S) = presupuesto` por tramos de
  precio). El riesgo nominal (distancia al stop × volumen, con slippage de
  entrada ya incluido en los precios) nunca supera el presupuesto.
- El stop dinámico **no rebaja** un stop ya mejorado por Break Even o Trailing
  (`origen_stop != 0`): para largos se mantiene el nivel más alto, para cortos
  el más bajo.
- El presupuesto es **nominal**: comisión y slippage se descuentan del
  resultado pero no reservan presupuesto (decisión explícita). Un gap también
  puede producir una pérdida superior al nominal.

### 2. ATR causal

- El ATR de una entrada se toma de la última vela **cerrada** (`_atr_cerrado`:
  `atr[i-1]` al operar el open de `i`), nunca de la vela de entrada. Elimina el
  look-ahead del sizing/stop en tiempo real.
- Pendiente: el `bfill()` del ATR de estrategias sigue rellenando las primeras
  velas con un ATR calculado con datos futuros (solo afecta al warm-up).

### 3. Gaps y prioridad stop/tramo

- Si una vela abre al otro lado del stop vigente **al abrir**, el cierre se
  llena al `open` (pérdida real por hueco), no al nivel teórico del stop.
- Un tramo pendiente no se ejecuta si la vela abre atravesando el stop
  (el stop tiene prioridad absoluta sobre los tramos).
- El gap se evalúa contra el stop vigente al abrir la vela, no contra un stop
  movido por BE/trailing dentro de esa misma vela.

### 4. Cobertura

Tests nuevos en `tests/test_stop_atr_modo.py` (causalidad, ambos modos,
largos/cortos, recorte de presupuesto, gap al open sin ejecutar el tramo,
stop dinámico que respeta BE) y ajuste del escenario de pirámide en
`tests/test_backtest.py`. Suite completa: 853 tests en verde.

---

## Actualización posterior — indicadores alineados con el estándar (RSI / ATR / Bollinger)

Fecha: 2026-08-17. Se corrigen las 3 desviaciones documentadas en la §2,
tanto en el motor como en los runtimes del código generado (Pine/MT5), para
mantener la garantía de fidelidad motor↔código generado
(`tests/test_runtime_diferencial.py` exige paridad exacta).

### 1. RSI — sin pérdidas → 100 (antes 50)

- `core/strategies.py:rsi()`: cuando la ventana no tiene pérdidas, devuelve
  **100** (fuerza alcista pura) en vez de 50. El 50 neutro queda solo para la
  serie plana (ganancias y pérdidas a 0 a la vez) y para el warm-up.
- Los runtimes ya usaban el nativo (Pine `ta.rsi`, MT5 `iRSI`), así que esta
  corrección **alinea** el motor con su propio código generado.

### 2. ATR — suavizado de Wilder (antes media simple)

- `core/strategies.py:atr()`: ahora usa el suavizado de **Wilder (RMA)** con
  la misma semilla que `ta.rma` de Pine (SMA de las primeras `periodo` velas y
  recursión `atr[i] = (atr[i-1]*(periodo-1) + tr[i])/periodo`), vía el helper
  `@njit _atr_rma_numba`. Conserva el `bfill()` del warm-up.
- `core/codegen/runtime/pine_runtime.pine:zcsAtr()` → `ta.atr(periodo)`.
- `core/codegen/runtime/zcs_runtime_mt5.mqh:zcsAtr()` → recursión de Wilder
  sobre la ventana de calentamiento (`ZCS_CALENTAMIENTO`), mismo patrón que
  KAMA/SAR.

### 3. Bollinger — desviación poblacional ddof=0 (antes muestral ddof=1)

- `core/strategies.py:bollinger()`: `std(ddof=0)` (divide por n), la definición
  estándar.
- `pine_runtime.pine:zcsBbDesv()` → `ta.stdev(src, periodo, true)`.
- `zcs_runtime_mt5.mqh:zcsBbDesv()` → `MathSqrt(s / periodo)`.

### 4. Consecuencia de comportamiento

Las estrategias guardadas (`Sistemas/`, `Favoritos/`) cambian de señales y de
dimensionado de stop (el ATR fija la distancia al stop y, con ella, el tamaño
de la posición). Es el tradeoff ya advertido en la §2, asumido en esta tarea.

### 5. Cobertura

Tests actualizados: `tests/test_indicadores.py` (los 3 pines pasan a verificar
el comportamiento corregido), `tests/test_runtime_diferencial.py` (`_port_atr`
y `_port_bb_desv` reflejan el nuevo runtime), `tests/test_codegen_pine.py` (las
2 aserciones del runtime) y `tests/test_strategias_volatilidad.py` (con Wilder
el ATR decae más lento tras un pico, así que la "calma" se lee como percentil
bajo algo más tarde). Suite completa: 890 tests en verde.
