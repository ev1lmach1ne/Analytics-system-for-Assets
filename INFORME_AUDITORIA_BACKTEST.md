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

## 2. Indicadores — verificados, con 3 desviaciones documentadas (sin corregir)

Se agregó `tests/test_indicadores.py` con 9 tests numéricos directos (antes
NO existía ningún test que verificara los valores de `sma`/`ema`/`rsi`/`atr`/
`bollinger` contra un cálculo independiente — solo se probaban indirectamente
a través de señales generadas). Los indicadores calculan correctamente sus
fórmulas, **excepto** por 3 desviaciones respecto a la definición "de libro"
que ya estaban presentes en el código y que, por decisión explícita, **no se
modifican** en esta tarea (cambiarlas alteraría las señales de estrategias
existentes). Quedan fijadas por test para que cualquier cambio futuro sea
intencional:

| Indicador | Desviación encontrada | Estándar de referencia | Archivo |
|---|---|---|---|
| RSI | Devuelve **50** cuando no hubo pérdidas en la ventana (`perdida=0` → `rs=NaN` → `fillna(50)`) | Debería ser **100** (fuerza alcista pura) | `core/strategies.py:49-55` |
| ATR | Promedio **simple (SMA)** del True Range | Suavizado de **Wilder** (usado por TradingView/MT5/la mayoría de plataformas) | `core/strategies.py:58-62` |
| Bollinger | Desviación estándar **muestral** (`ddof=1`, default de pandas) | Desviación **poblacional** (`ddof=0`) | `core/strategies.py:65-69` |

Impacto práctico: los valores de ATR y Bollinger difieren (ligeramente, en el
caso de Bollinger; más notablemente en ATR según la volatilidad reciente) de
lo que mostraría un gráfico de TradingView/MT5 con los mismos parámetros. El
RSI=50 en vez de 100 solo importa en rachas de subida pura sin ninguna vela
roja dentro de la ventana — un caso de borde poco frecuente pero real.

Si en algún momento quieres alinear estos indicadores con el estándar de
mercado, es un cambio localizado (3 funciones en `core/strategies.py`), pero
hay que asumir que las estrategias ya guardadas cambiarán de señales.

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
| Indicadores (SMA/EMA/RSI/ATR/Bollinger/cruces) | ✅ Verificados con tests numéricos; 3 desviaciones documentadas y fijadas, sin corregir por decisión explícita |
| Anualización Sharpe/CAGR | ✅ Corregida — ahora por clase de activo, con fallback seguro |

Archivos nuevos/modificados: `tests/test_indicadores.py` (nuevo),
`tests/test_config_anualizacion.py` (nuevo), `tests/test_backtest.py`
(ampliado), `core/config.py` (+`velas_por_anio`), `gui/widgets/tab_backtest.py`
(`_velas_por_anio` ahora recibe la ruta del CSV y usa la clase de activo).
