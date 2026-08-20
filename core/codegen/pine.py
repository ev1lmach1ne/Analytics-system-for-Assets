"""
core/codegen/pine.py
Emisor de Pine Script v6 para TradingView.

Estructura del archivo generado: cabecera con las notas de fidelidad →
strategy() con la cuenta del backtest → guardas de activo y temporalidad →
inputs → runtime insertado → indicadores → filtros → señales → gestión.

CÓMO SE MAPEA LA EJECUCIÓN DEL MOTOR
────────────────────────────────────
El motor decide en la vela t y ejecuta al open de t+1. Pine hace eso solo si
`process_orders_on_close=false` (el valor por defecto, que aquí se escribe
explícito porque de él depende que el script opere como el backtest).

El stop y el take-profit se ponen con strategy.exit(stop=, limit=), que Pine
comprueba contra el recorrido de la vela y rellena en el nivel — igual que el
motor los comprueba contra low/high y rellena en el precio del nivel.

Lo que NO se puede reproducir: el motor dimensiona con el ATR de la última
vela cerrada antes de la entrada (atr[i-1], ver core/backtest._atr_cerrado);
aquí se usa el de la vela de la señal, que ya está cerrada, así que coinciden
salvo en el borde inicial de la serie. Va declarado en fidelidad.CAPACIDADES y
avisado en la cabecera.
"""
import os

from core.codegen import fidelidad
from core.codegen.base import Emisor
from core.strategies import ESTRATEGIAS

RUTA_RUNTIME = os.path.join(os.path.dirname(__file__), 'runtime',
                            'pine_runtime.pine')

# temporalidad del backtest -> literal de timeframe de TradingView
_TF_PINE = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
            '1h': '60', '2h': '120', '4h': '240', '6h': '360', '8h': '480',
            '12h': '720', '1d': 'D', '1w': 'W', '1M': 'M'}

_DIAS_PINE = ['dayofweek.monday', 'dayofweek.tuesday', 'dayofweek.wednesday',
              'dayofweek.thursday', 'dayofweek.friday', 'dayofweek.saturday',
              'dayofweek.sunday']


class EmisorPine(Emisor):
    clave = 'tradingview'
    nombre = 'TradingView'
    lenguaje = 'Pine Script v6'
    extension = '.pine'
    comentario = '//'

    # ══════════════ ganchos de lenguaje ══════════════

    def op_y(self, partes):
        return "(" + " and ".join(partes) + ")"

    def op_o(self, partes):
        return "(" + " or ".join(partes) + ")"

    def op_no(self, parte):
        return f"not {parte}"

    def op_cruza_arriba(self, izq, der):
        return f"zcsCruzaArriba({izq}, {der})"

    def op_cruza_abajo(self, izq, der):
        return f"zcsCruzaAbajo({izq}, {der})"

    def precio(self, campo):
        return campo

    def giro_sar(self, nodo):
        """El giro es un cambio de la tendencia del SAR entre dos velas, no
        una comparación de precio: se lee de la variable de tendencia que
        declara zcsSar."""
        tend = self.nombre_serie(nodo['sar']) + '_tend'
        signo = '1' if nodo['sentido'] > 0 else '-1'
        contrario = '-1' if nodo['sentido'] > 0 else '1'
        return f"({tend} == {signo} and {tend}[1] == {contrario})"

    def giro_supertrend(self, nodo):
        """Mismo patrón que el giro del SAR: la tendencia la expone
        zcsSupertrend en su segunda variable de retorno."""
        tend = self.nombre_serie(nodo['sar']) + '_tend'
        signo = '1' if nodo['sentido'] > 0 else '-1'
        contrario = '-1' if nodo['sentido'] > 0 else '1'
        return f"({tend} == {signo} and {tend}[1] == {contrario})"

    # ══════════════ declaración de indicadores ══════════════

    def declarar_serie(self, nombre, nodo):
        tipo = nodo['tipo']
        arg = lambda campo: self.arg_serie(nodo, campo)   # noqa: E731

        if tipo == 'SMA':
            return f"{nombre} = ta.sma(close, {arg('periodo')})"
        if tipo == 'EMA':
            return f"{nombre} = ta.ema(close, {arg('periodo')})"
        if tipo == 'RSI':
            return f"{nombre} = ta.rsi(close, {arg('periodo')})"
        if tipo == 'CCI':
            return f"{nombre} = ta.cci(close, {arg('periodo')})"
        if tipo == 'WILLR':
            return f"{nombre} = ta.wpr({arg('periodo')})"
        if tipo == 'ATR':
            return f"{nombre} = zcsAtr({arg('periodo')})"
        if tipo == 'BB_media':
            return f"{nombre} = zcsBbMedia(close, {arg('periodo')})"
        if tipo == 'BB_sup':
            return (f"{nombre} = zcsBbSup(close, {arg('periodo')}, "
                    f"{arg('desv')})")
        if tipo == 'BB_inf':
            return (f"{nombre} = zcsBbInf(close, {arg('periodo')}, "
                    f"{arg('desv')})")
        if tipo == 'KAMA':
            return (f"{nombre} = zcsKama({arg('periodo')}, {arg('rapido')}, "
                    f"{arg('lento')})")
        if tipo == 'ER':
            return f"{nombre} = zcsEr({arg('periodo')})"
        if tipo in ('STOCH_K', 'STOCH_D'):
            if tipo == 'STOCH_K':
                return (f"{nombre} = zcsStochK({arg('periodo_k')}, "
                        f"{arg('suavizado_k')})")
            return (f"{nombre} = zcsStochD({arg('periodo_k')}, "
                    f"{arg('suavizado_k')}, {arg('periodo_d')})")
        if tipo in ('DONCHIAN_SUP', 'DONCHIAN_INF'):
            # con fuente='close' el canal se forma con CIERRES por los dos
            # lados, no es un cierre medido contra un canal de máximos
            cerrado = nodo.get('fuente') == 'close'
            if tipo == 'DONCHIAN_SUP':
                src = 'close' if cerrado else 'high'
                return f"{nombre} = zcsDonchianSup({src}, {arg('periodo')})"
            src = 'close' if cerrado else 'low'
            return f"{nombre} = zcsDonchianInf({src}, {arg('periodo')})"
        if tipo == 'SAR':
            return (f"[{nombre}, {nombre}_tend] = zcsSar({arg('af_inicial')}, "
                    f"{arg('af_paso')}, {arg('af_max')})")
        if tipo == 'SUPERTREND':
            return (f"[{nombre}, {nombre}_tend] = zcsSupertrend("
                    f"{arg('periodo')}, {self.num(nodo['multiplicador'])})")
        if tipo in ('MACD_LINEA', 'MACD_SENAL', 'MACD_HIST'):
            # cada serie declara el trío completo (zcsMacd devuelve tres
            # valores): los nombres se derivan del tipo de cada una
            suf = nombre[len(tipo.lower()):]
            return (f"[macd_linea{suf}, macd_senal{suf}, macd_hist{suf}] = "
                    f"zcsMacd({arg('rapido')}, {arg('lento')}, {arg('senal')})")
        if tipo in ('ADX', 'DI_PLUS', 'DI_MINUS'):
            suf = nombre[len(tipo.lower()):]
            return (f"[adx{suf}, di_plus{suf}, di_minus{suf}] = "
                    f"zcsAdx({arg('periodo')})")
        if tipo in ('AROON_UP', 'AROON_DN'):
            funcion = 'zcsAroonUp' if tipo == 'AROON_UP' else 'zcsAroonDown'
            return f"{nombre} = {funcion}({arg('periodo')})"
        if tipo == 'CMO':
            return f"{nombre} = zcsCmo({arg('periodo')})"
        if tipo == 'TRIX':
            return f"{nombre} = zcsTrix({arg('periodo')})"
        if tipo in ('STOCHRSI', 'STOCHRSI_D'):
            fname = 'zcsStochRsiK' if tipo == 'STOCHRSI' else 'zcsStochRsiD'
            return f"{nombre} = {fname}({arg('periodo')})"
        if tipo in ('ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU_A',
                    'ICHIMOKU_SENKOU_B', 'ICHIMOKU_CHIKOU'):
            fname = {'ICHIMOKU_TENKAN': 'zcsIchimokuTenkan',
                     'ICHIMOKU_KIJUN': 'zcsIchimokuKijun',
                     'ICHIMOKU_SENKOU_A': 'zcsIchimokuSenkouA',
                     'ICHIMOKU_SENKOU_B': 'zcsIchimokuSenkouB',
                     'ICHIMOKU_CHIKOU': 'zcsIchimokuChikou'}[tipo]
            return (f"{nombre} = {fname}({arg('tenkan')}, {arg('kijun')}, "
                    f"{arg('senkou')})")
        if tipo in ('KELTNER_SUP', 'KELTNER_INF', 'KELTNER_MEDIA'):
            fname = {'KELTNER_MEDIA': 'zcsKeltnerMedia',
                     'KELTNER_SUP': 'zcsKeltnerSup',
                     'KELTNER_INF': 'zcsKeltnerInf'}[tipo]
            return (f"{nombre} = {fname}({arg('periodo')}, "
                    f"{self.num(nodo['multiplicador'])})")
        if tipo in ('TTM_SQUEEZE', 'TTM_MOMENTUM'):
            fname = 'zcsTtmSqueeze' if tipo == 'TTM_SQUEEZE' else 'zcsTtmMomentum'
            return (f"{nombre} = {fname}({arg('periodo')}, "
                    f"{self.num(nodo['mult_bb'])}, {self.num(nodo['mult_kc'])})")
        if tipo in ('VWAP_MEDIA', 'VWAP_SD', 'VWAP_SUP', 'VWAP_INF'):
            fname = {'VWAP_MEDIA': 'zcsVwapMedia', 'VWAP_SD': 'zcsVwapSd',
                     'VWAP_SUP': 'zcsVwapSup', 'VWAP_INF': 'zcsVwapInf'}[tipo]
            ancla = f'"{nodo["anclaje"]}"'
            modo_lit = f'"{nodo["modo"]}"'
            return (f"{nombre} = {fname}({ancla}, {arg('k')}, {modo_lit})")
        if tipo == 'ZIGZAG':
            return (f"[{nombre}, {nombre}_tipo] = zcsZigzag("
                    f"{self.num(nodo['desviacion'])}, "
                    f"{self.num(nodo['piernas'])})")
        if tipo == 'PCT_ATR':
            return (f"{nombre} = zcsPercentilRodante("
                    f"zcsAtr({self.num(nodo['periodo_base'])}), "
                    f"{self.num(nodo['ventana'])})")
        if tipo == 'PCT_STDEV':
            return (f"{nombre} = zcsPercentilRodante("
                    f"zcsStdevRet({self.num(nodo['periodo_base'])}), "
                    f"{self.num(nodo['ventana'])})")
        if tipo == 'HURST':
            raise ValueError(
                "Pine Script no puede calcular el Hurst por vela (ver "
                "fidelidad.CAPACIDADES): este setup no debería haber llegado "
                "al emisor con el filtro de régimen puesto.")
        raise ValueError(f"{self.nombre}: serie sin traducción: {tipo!r}")

    # ══════════════ archivos ══════════════

    def archivos_setup(self, ir_setup, ir_sistema, avisos, meta=None):
        meta = meta or {}
        nombre = self.identificador(
            f"{meta.get('sistema', 'sistema')}_S{ir_setup['indice']}")
        texto = self.construir(ir_setup, ir_sistema, avisos, meta)
        return {f"TradingView/{nombre}{self.extension}": texto}

    def archivos_comunes(self, ir_sistema, avisos, meta=None):
        return {"TradingView/INSTALAR.md": _instalar_md(meta or {})}

    # ══════════════ construcción del script ══════════════

    def construir(self, ir_setup, ir_sistema, avisos, meta):
        bloques = [
            "//@version=6",
            self.cabecera(ir_setup, ir_sistema, avisos, meta),
            self._strategy(ir_setup, ir_sistema, meta),
            self._inputs(ir_setup, meta),
            self._runtime(),
            self._guardas(meta),
            self._indicadores(ir_setup),
            self._filtros(ir_setup),
            self._senales(ir_setup),
            self._gestion(ir_setup),
            self._aviso_runtime(avisos),
        ]
        return "\n\n".join(b for b in bloques if b) + "\n"

    def _seccion(self, titulo):
        return f"// ── {titulo} " + "─" * max(0, 62 - len(titulo))

    def _strategy(self, ir_setup, ir_sistema, meta):
        cuenta = ir_sistema['cuenta']
        titulo = f"{meta.get('sistema', 'Sistema')} · {ir_setup['nombre']}"
        return "\n".join([
            self._seccion("estrategia"),
            f'strategy({self.texto(titulo)}, overlay=true,',
            f'         initial_capital={self.num(cuenta["capital_inicial"])},',
            '         commission_type=strategy.commission.percent,',
            f'         commission_value={self.num(cuenta["comision_pct"] * 100)},',
            '         default_qty_type=strategy.fixed,',
            '         calc_on_every_tick=false,',
            '         // false = la orden se rellena al OPEN de la vela',
            '         // siguiente, que es como ejecuta el motor del backtest',
            '         process_orders_on_close=false)',
        ])

    def _inputs(self, ir_setup, meta):
        """Un input por cada parámetro de la plantilla que el código usa, más
        los de riesgo y los de la guarda de activo."""
        lineas = [self._seccion("parámetros de la plantilla")]
        specs = {s['clave']: s for s in ESTRATEGIAS[ir_setup['plantilla']]['params']}
        for clave in self.params_usados(ir_setup):
            spec = specs.get(clave)
            valor = ir_setup['params'].get(clave)
            lineas.append(self._input(clave, spec, valor))

        g = ir_setup['gestion']
        lineas += ["", self._seccion("riesgo y gestión")]
        lineas.append(
            f'p_riesgo_pct = input.float({self.num(g["riesgo_pct"] * 100)}, '
            f'"Riesgo por operación (%)", minval=0.01, maxval=100.0, '
            f'step=0.05, group="Riesgo") / 100.0')
        lineas.append(
            f'p_stop_atr = input.float({self.num(g["stop_atr"])}, '
            f'"Stop (× ATR)", minval=0.0, step=0.1, group="Riesgo")')
        lineas.append(
            f'p_tp_r = input.float({self.num(g["tp_r"])}, '
            f'"Take-profit (R)  ·  0 = sin TP", minval=0.0, step=0.1, '
            f'group="Riesgo")')
        lineas.append(
            f'p_be_atr = input.float({self.num(g["be_atr"])}, '
            f'"Break-even ({"R" if g["be_unidad"] == "r" else "× ATR"})  ·  '
            f'0 = sin BE", minval=0.0, step=0.1, group="Riesgo")')
        lineas.append(
            f'p_trailing_atr = input.float({self.num(g["trailing_atr"])}, '
            f'"Trailing (× ATR)  ·  0 = sin trailing", minval=0.0, step=0.1, '
            f'group="Riesgo")')
        lineas.append(
            f'p_salida_velas = input.int({g["salida_n_velas"]}, '
            f'"Salida por tiempo (velas)  ·  0 = sin límite", minval=0, '
            f'group="Riesgo")')
        lineas.append(
            f'p_periodo_atr = input.int({g["periodo_atr"]}, '
            f'"Periodo del ATR de gestión", minval=1, group="Riesgo")')
        return "\n".join(lineas)

    def _input(self, clave, spec, valor):
        etiqueta = spec['etiqueta'] if spec else clave
        grupo = 'group="Plantilla"'
        if spec and spec['tipo'] == 'int':
            return (f'p_{clave} = input.int({int(valor)}, '
                    f'{self.texto(etiqueta)}, minval={int(spec["min"])}, '
                    f'maxval={int(spec["max"])}, {grupo})')
        if spec and spec['tipo'] == 'float':
            return (f'p_{clave} = input.float({self.num(float(valor))}, '
                    f'{self.texto(etiqueta)}, minval={self.num(float(spec["min"]))}, '
                    f'maxval={self.num(float(spec["max"]))}, step=0.1, {grupo})')
        return (f'p_{clave} = input.float({self.num(float(valor))}, '
                f'{self.texto(etiqueta)}, {grupo})')

    def _runtime(self):
        with open(RUTA_RUNTIME, encoding='utf-8') as f:
            return f.read().rstrip()

    def _guardas(self, meta):
        """Comprobación de activo y temporalidad.

        Los parámetros están ajustados a un activo y una temporalidad
        concretos; en otro sitio no significan lo mismo. La temporalidad se
        puede comprobar exacta; el símbolo no, porque el nombre del CSV del
        backtest no tiene por qué coincidir con el del bróker, así que se deja
        como input ajustable y solo avisa."""
        tf = meta.get('tf') or ''
        tf_pine = _TF_PINE.get(tf, '')
        activo = meta.get('activo') or '?'
        lineas = [
            self._seccion("guardas de activo y temporalidad"),
            f'p_simbolo_esperado = input.string({self.texto(activo)}, '
            f'"Símbolo del backtest", group="Guardas")',
            'p_permitir_otro = input.bool(false, '
            '"Permitir otro activo o temporalidad", group="Guardas")',
        ]
        if tf_pine:
            lineas += [
                f'tfEsperado = {self.texto(tf_pine)}',
                'tfCorrecto = timeframe.period == tfEsperado',
            ]
        else:
            lineas.append('tfCorrecto = true')
        # El SÍMBOLO no puede bloquear: el nombre del CSV del backtest
        # («zcmaiz») casi nunca coincide con el ticker del bróker («ZC1!»), así
        # que exigir que casen dejaría el script sin operar NUNCA, en silencio.
        # Solo avisa. La temporalidad sí bloquea: esa se puede comprobar
        # exacta, y con otra los parámetros no significan lo mismo.
        texto_tf = (f"NO SE OPERARA: este script se backtesteo en {tf} y el "
                    f"grafico esta en otra temporalidad. Activa 'Permitir "
                    f"otro activo o temporalidad' si es a proposito.")
        texto_simbolo = (f"AVISO: backtesteado en {activo}. Comprueba que este "
                         f"grafico es el mismo activo.")
        lineas += [
            'simboloCorrecto = str.contains(str.upper(syminfo.ticker), '
            'str.upper(p_simbolo_esperado))',
            '// el simbolo solo avisa; la temporalidad es la que decide',
            'puedeOperar = p_permitir_otro or tfCorrecto',
            f'avisoTf = {self.texto(texto_tf)}',
            f'avisoSimbolo = {self.texto(texto_simbolo)}',
            'hayAviso = not tfCorrecto or not simboloCorrecto',
            'textoAviso = not tfCorrecto ? avisoTf : avisoSimbolo',
            'if barstate.islast and hayAviso',
            '    colorAviso = not tfCorrecto ? color.red : color.orange',
            '    tablaAviso = table.new(position.top_right, 1, 1, '
            'bgcolor=color.new(colorAviso, 20))',
            '    table.cell(tablaAviso, 0, 0, textoAviso, '
            'text_color=color.white, text_size=size.small)',
        ]
        return "\n".join(lineas)

    def _indicadores(self, ir_setup):
        lineas = [self._seccion("indicadores")]
        lineas.append("atrGestion = zcsAtr(p_periodo_atr)")
        for nombre, nodo in self.series_declarables(ir_setup):
            lineas.append(self.declarar_serie(nombre, nodo))
        return "\n".join(lineas)

    def _filtros(self, ir_setup):
        """Los filtros solo condicionan NUEVAS entradas: las salidas nunca se
        filtran, para no dejar una posición abierta sin forma de cerrarse si
        el régimen o la sesión cambian a mitad de operación."""
        from core.codegen.ir import condiciones_de_lado
        filtros = ir_setup['filtros']
        comunes = []
        lineas = [self._seccion("filtros (solo condicionan nuevas entradas)")]

        if filtros['dias_semana'] is not None:
            dias = " or ".join(f"dayofweek == {_DIAS_PINE[d]}"
                               for d in filtros['dias_semana'] if 0 <= d < 7)
            lineas.append(f"filtroDia = {dias or 'true'}")
            comunes.append('filtroDia')

        ses = filtros['sesion']
        if ses:
            tz = ses['tz'] or 'UTC'
            rango = f"{ses['hora_inicio']:02d}00-{ses['hora_fin']:02d}00"
            lineas.append(
                f"filtroSesion = not na(time(timeframe.period, "
                f"{self.texto(rango)}, {self.texto(tz)}))")
            comunes.append('filtroSesion')

        if filtros['regimen']:
            lineas.append(
                f"filtroRegimen = {self.expr(filtros['regimen']['condicion'])}")
            comunes.append('filtroRegimen')

        if filtros['volatilidad']:
            lineas.append(
                f"filtroVolatilidad = "
                f"{self.expr(filtros['volatilidad']['condicion'])}")
            comunes.append('filtroVolatilidad')

        for lado in ('long', 'short'):
            cond = condiciones_de_lado(filtros['condiciones_entrada'], lado)
            partes = list(comunes)
            if cond is not None:
                partes.append(self.expr(cond))
            nombre = f"filtro{lado.capitalize()}"
            lineas.append(f"{nombre} = " + (" and ".join(partes) or "true"))

        for lado in ('long', 'short'):
            cond = condiciones_de_lado(filtros['condiciones_salida'], lado)
            nombre = f"permiteSalida{lado.capitalize()}"
            lineas.append(f"{nombre} = " + (self.expr(cond) if cond is not None
                                            else "true"))
        return "\n".join(lineas)

    def _senales(self, ir_setup):
        senales = ir_setup['senales']
        lineas = [self._seccion("señales de la plantilla")]
        for lado in ('entradas_long', 'entradas_short',
                     'salidas_long', 'salidas_short'):
            nombre = _VAR_SENAL[lado]
            nodo = senales[lado]
            lineas.append(f"{nombre} = "
                          + (self.expr(nodo) if nodo is not None else "false"))
        return "\n".join(lineas)

    def _gestion(self, ir_setup):
        """Sizing por riesgo, stop, TP, break-even, trailing y salida por
        tiempo. Reproduce el bloque de gestión del motor con las piezas que
        Pine ya da hechas (strategy.exit comprueba el stop contra el recorrido
        de la vela igual que el motor contra low/high)."""
        g = ir_setup['gestion']
        bloques = [
            self._seccion("gestión de la posición"),
            "// Distancia al stop. El motor dimensiona con el ATR de la",
            "// última vela cerrada antes de la entrada (atr[i-1]); aquí se usa",
            "// el de la vela de la señal, que ya está cerrada: coinciden.",
            "distRef = p_stop_atr > 0 ? p_stop_atr * atrGestion "
            ": 2.0 * atrGestion",
            "",
            "enLargo = strategy.position_size > 0",
            "enCorto = strategy.position_size < 0",
            "plano = strategy.position_size == 0",
            "",
            "var float distEntrada = na",
            "var float stopActual = na",
            "var float maxFav = na",
            "var int barraEntrada = na",
            "var bool beAplicado = false",
        ]
        tramos = g['tramos']
        if len(tramos) > 1:
            bloques += [
                "var int tramoActual = 1   // tramo que falta por añadir",
                "var bool tramoOrden = false   // ya se pidió el del vela en curso",
            ]
        bloques += [
            "",
            "// ── entradas: al open de la vela siguiente a la señal ──",
            "if plano and puedeOperar",
            "    qty = zcsUnidadesPorRiesgo(strategy.equity, p_riesgo_pct, "
            "distRef)",
            "    if qty > 0",
            f"        if {_VAR_SENAL['entradas_long']} and filtroLong",
            '            strategy.entry("L", strategy.long, qty=qty)',
            f"        else if {_VAR_SENAL['entradas_short']} and filtroShort",
            '            strategy.entry("S", strategy.short, qty=qty)',
            "",
            "// ── al abrirse la posición se fija la referencia de riesgo ──",
            "acabaDeAbrir = strategy.position_size != 0 and "
            "strategy.position_size[1] == 0",
            "if acabaDeAbrir",
            "    distEntrada := nz(distRef[1], distRef)",
            "    maxFav := strategy.position_avg_price",
            "    barraEntrada := bar_index",
            "    beAplicado := false",
        ]
        if len(tramos) > 1:
            bloques.append("    tramoActual := 1")
        bloques += [
            "    // el stop base solo existe si se pidió: sin él, BE y trailing",
            "    // lo crean desde cero cuando disparan, igual que el motor",
            "    stopActual := p_stop_atr > 0 ? (enLargo ? "
            "strategy.position_avg_price - distEntrada : "
            "strategy.position_avg_price + distEntrada) : na",
            "",
            "// ── gestión con la posición abierta ──",
            "if strategy.position_size != 0",
            "    dir = enLargo ? 1 : -1",
            "    precioIn = strategy.position_avg_price",
            "    maxFav := enLargo ? math.max(maxFav, high) "
            ": math.min(maxFav, low)",
        ]
        if len(tramos) > 1:
            bloques += self._tramos(ir_setup, tramos[1:])
        bloques += [
            "",
            "    // break-even: mueve el stop a la entrada cuando el avance a",
            "    // favor supera el umbral. La referencia se remide con el ATR",
            "    // de la vela actual, igual que el motor.",
            "    refBe = " + ("distEntrada" if g['be_unidad'] == 'r'
                             else "atrGestion"),
            "    if p_be_atr > 0 and not beAplicado and "
            "(maxFav - precioIn) * dir >= p_be_atr * refBe",
            "        stopActual := na(stopActual) ? precioIn : (enLargo ? "
            "math.max(stopActual, precioIn) : math.min(stopActual, precioIn))",
            "        beAplicado := true",
            "",
            "    // trailing: el stop sigue al extremo alcanzado y nunca afloja",
            "    if p_trailing_atr > 0",
            "        candidato = maxFav - p_trailing_atr * atrGestion * dir",
            "        stopActual := na(stopActual) ? candidato : (enLargo ? "
            "math.max(stopActual, candidato) : "
            "math.min(stopActual, candidato))",
            "",
            "    tpNivel = p_tp_r > 0 ? precioIn + p_tp_r * distEntrada * dir "
            ": na",
            "    // el stop se aplica aunque no haya stop base: BE y trailing",
            "    // pueden crearlo desde cero, igual que el motor",
            "    hayStop = p_stop_atr > 0 or p_be_atr > 0 or p_trailing_atr > 0",
            "    if enLargo",
            '        strategy.exit("SalidaL", from_entry="L", '
            'stop=hayStop ? stopActual : na, limit=tpNivel)',
            "    else",
            '        strategy.exit("SalidaS", from_entry="S", '
            'stop=hayStop ? stopActual : na, limit=tpNivel)',
            "",
            "    // salida por señal de la plantilla",
            f"    if enLargo and {_VAR_SENAL['salidas_long']} and "
            "permiteSalidaLong",
            '        strategy.close("L", comment="Señal")',
            f"    if enCorto and {_VAR_SENAL['salidas_short']} and "
            "permiteSalidaShort",
            '        strategy.close("S", comment="Señal")',
            "",
            "    // salida por tiempo",
            "    if p_salida_velas > 0 and bar_index - barraEntrada >= "
            "p_salida_velas",
            '        strategy.close_all(comment="Tiempo")',
        ]
        return "\n".join(bloques)

    def _tramos(self, ir_setup, tramos):
        """Tramos de entrada escalonada (2º en adelante).

        Cada tramo añade posición al open de la vela siguiente a su disparador,
        con un tamaño = riesgo_total × pct/100 / distancia_al_stop (con el mismo
        suelo del 25% de la distancia de referencia que usa el motor). El
        presupuesto no se suma: cada tramo arriesga su parte del riesgo total
        del setup, que es exactamente lo que prometió el backtest."""
        from core.codegen.ir import condiciones_de_lado
        lineas = ["",
                  "    // ── entrada escalonada: tramos adicionales ──",
                  "    tramoOrden := false",
                  "    // distancia al stop para dimensionar el tramo (suelo 25%)",
                  "    distTramo = math.max(na(stopActual) ? distRef : "
                  "math.abs(precioIn - stopActual), 0.25 * distRef)"]
        for k, t in enumerate(tramos, 1):
            cond = condiciones_de_lado(t['condiciones'], 'long')
            cond_s = condiciones_de_lado(t['condiciones'], 'short')
            pct = float(t['pct']) / 100.0
            nombre = f"T{k}"
            lineas += [
                "",
                f"    if tramoActual == {k} and not tramoOrden",
                "        disparaT = false",
                "        if enLargo",
                f"            disparaT := {self._tramo_trig(t, 'long')}",
                "            if disparaT and "
                + (self.expr(cond) if cond is not None else "true"),
                f"                qtyT = zcsUnidadesPorRiesgo(strategy.equity, "
                f"p_riesgo_pct * {self.num(pct)}, distTramo)",
                "                if qtyT > 0",
                f'                    strategy.order("{nombre}", '
                'strategy.long, qty=qtyT)',
                "                    tramoActual += 1",
                "                    tramoOrden := true",
                "        if enCorto and not tramoOrden",
                f"            disparaT := {self._tramo_trig(t, 'short')}",
                "            if disparaT and "
                + (self.expr(cond_s) if cond_s is not None else "true"),
                f"                qtyT = zcsUnidadesPorRiesgo(strategy.equity, "
                f"p_riesgo_pct * {self.num(pct)}, distTramo)",
                "                if qtyT > 0",
                f'                    strategy.order("{nombre}", '
                'strategy.short, qty=qtyT)',
                "                tramoActual += 1",
                "                tramoOrden := true",
            ]
        return lineas

    def _tramo_trig(self, t, lado):
        """Expresión booleana del disparador de un tramo para un lado."""
        val = float(t.get('val', 0.0))
        trig = t['trigger']
        if trig == 'senal':
            return ("entradaLong and filtroLong" if lado == 'long'
                    else "entradaShort and filtroShort")
        if trig == 'velas':
            return f"bar_index - barraEntrada >= {self.num(int(val))}"
        if trig == 'retroceso':
            return (f"low <= precioIn - {self.num(val)} * atrGestion"
                    if lado == 'long'
                    else f"high >= precioIn + {self.num(val)} * atrGestion")
        if trig == 'avance':
            return (f"high >= precioIn + {self.num(val)} * distRef"
                    if lado == 'long'
                    else f"low <= precioIn - {self.num(val)} * distRef")
        if trig == 'cond':
            return "true"   # lo decide solo la condición de abajo
        return "false"

    def _aviso_runtime(self, avisos):
        """Aviso en ejecución de lo que se ha omitido. Sobrevive a que alguien
        copie el archivo suelto, sin la carpeta ni las notas."""
        texto = fidelidad.texto_runtime(avisos)
        if not texto:
            return ""
        return "\n".join([
            self._seccion("aviso de fidelidad en ejecución"),
            "if barstate.isfirst",
            f"    log.warning({self.texto(texto)})",
        ])


_VAR_SENAL = {'entradas_long': 'entradaLong', 'entradas_short': 'entradaShort',
              'salidas_long': 'salidaLong', 'salidas_short': 'salidaShort'}


def _instalar_md(meta):
    activo = meta.get('activo', '?')
    tf = meta.get('tf', '?')
    return f"""# Instalar en TradingView

TradingView **no importa archivos**: el código se pega a mano. El `.pine` de
esta carpeta es texto plano y la extensión es solo una convención, así que se
abre con cualquier editor de texto.

1. Clic derecho en el archivo `.pine` → *Abrir con* → **Bloc de notas**
   (Windows no conoce esa extensión y no lo abrirá con doble clic).
2. `Ctrl+A` para seleccionar todo y `Ctrl+C` para copiar. Comprueba que la
   primera línea es `//@version=6`: si al pegar aparece otra cosa arriba, se
   ha quedado texto fuera y el script no compilará.
3. Abre TradingView con el gráfico en **{activo}**, temporalidad **{tf}**.
4. *Pine Editor* (pestaña de abajo del todo) → *Abrir* → *Nuevo indicador en
   blanco*. Borra lo que traiga por defecto y pega con `Ctrl+V`.
5. *Guardar*, y después *Añadir al gráfico*.
6. La pestaña *Probador de estrategias* muestra el resultado.

## Antes de comparar con el backtest

- Comprueba en las propiedades del script que la **comisión** y el **capital
  inicial** son los que quieres: se han rellenado con los del backtest.
- El **slippage** de TradingView se mide en *ticks*, no en porcentaje, así que
  no se ha podido trasladar automáticamente. Ponlo a mano en las propiedades
  del script si quieres acercarte al backtest.
- Un archivo por setup: si el sistema tenía varios, añade cada uno por
  separado. El backtest los arbitraba entre sí vela a vela y eso no se
  reproduce al cargarlos como estrategias independientes.

Lee `NOTAS_DE_FIDELIDAD.md` en la carpeta de arriba: dice exactamente en qué se
diferencia este script del backtest.
"""
