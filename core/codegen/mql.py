"""
core/codegen/mql.py
Emisor de MQL5 para MetaTrader 5.

CÓMO SE MAPEA LA EJECUCIÓN DEL MOTOR
────────────────────────────────────
El motor decide con la vela t y ejecuta al open de t+1. El EA hace lo mismo
actuando SOLO en el primer tick de cada vela nueva (zcsVelaNueva): en ese
instante la vela recién abierta es la 0 y la última cerrada —la de la señal—
es la 1. Por eso todas las condiciones se evalúan en el desplazamiento 1.

Gestionar en cada tick en vez de en cada vela haría saltar el break-even y el
trailing en momentos que el backtest nunca vio, así que no se hace.

POR QUÉ CADA SERIE SE LEE DOS VECES
───────────────────────────────────
MQL no tiene el operador `[1]` de Pine sobre una expresión: para saber si dos
series se han cruzado hay que tener sus valores en la vela de la señal y en la
anterior. El emisor declara por eso `<serie>_1` y `<serie>_2`, y los cruces se
resuelven con zcsCruzaArriba(a1, a2, b1, b2).

LO QUE NO SE PUEDE REPRODUCIR
─────────────────────────────
El motor dimensiona con el ATR de la vela en cuya apertura entra, que incluye
el máximo, el mínimo y el cierre de esa misma vela. Aquí se usa el de la
última vela cerrada, que es lo único que existe en ese momento. Va declarado
en fidelidad.CAPACIDADES y avisado en la cabecera y en el OnInit.
"""
import os
import zlib

from core.codegen import fidelidad
from core.codegen.base import Emisor
from core.strategies import ESTRATEGIAS

RUTA_RUNTIME = os.path.join(os.path.dirname(__file__), 'runtime',
                            'zcs_runtime_mt5.mqh')

# minutos por temporalidad, para la guarda de marco temporal
_TF_MINUTOS = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60,
               '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
               '1d': 1440, '1w': 10080}

_DIAS_MQL = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY',
             'SATURDAY', 'SUNDAY']

# indicadores que MetaTrader calcula igual que el motor y se resuelven con un
# handle nativo; el resto vive en zcs_runtime.mqh
_NATIVOS = {
    'SMA': ('iMA', 'MODE_SMA'), 'EMA': ('iMA', 'MODE_EMA'),
    'RSI': ('iRSI', None), 'CCI': ('iCCI', None), 'WILLR': ('iWPR', None),
    'STOCH_K': ('iStochastic', 0), 'STOCH_D': ('iStochastic', 1),
}

# impacto mínimo del setup (TradingEconomics) -> enum del calendario de MetaQuotes
_IMPACTO_MQL = {
    'bajo': 'CALENDAR_IMPACT_LOW',
    'medio': 'CALENDAR_IMPACT_MEDIUM',
    'alto': 'CALENDAR_IMPACT_HIGH',
}


class EmisorMQL5(Emisor):
    clave = 'mt5'
    nombre = 'MetaTrader 5'
    lenguaje = 'MQL5'
    extension = '.mq5'
    comentario = '//'

    def __init__(self):
        self._shift = 1

    # ══════════════ ganchos de lenguaje ══════════════

    def op_y(self, partes):
        return "(" + " && ".join(partes) + ")"

    def op_o(self, partes):
        return "(" + " || ".join(partes) + ")"

    def op_no(self, parte):
        return f"!({parte})"

    def precio(self, campo):
        return f"{campo[0]}{self._shift}"

    def nombre_serie(self, nodo):
        return f"{super().nombre_serie(nodo)}_{self._shift}"

    def comparacion(self, nodo):
        """Los cruces necesitan las dos velas a la vez, así que se renderizan
        los dos operandos en el desplazamiento de la señal y en el anterior.
        El resto de comparaciones solo mira la vela de la señal."""
        op = nodo['op']
        if op not in ('cruza arriba', 'cruza abajo'):
            return super().comparacion(nodo)
        previo = self._shift
        try:
            self._shift = 1
            izq1, der1 = self.serie(nodo['izq']), self.serie(nodo['der'])
            self._shift = 2
            izq2, der2 = self.serie(nodo['izq']), self.serie(nodo['der'])
        finally:
            self._shift = previo
        funcion = ('zcsCruzaArriba' if op == 'cruza arriba'
                   else 'zcsCruzaAbajo')
        return f"{funcion}({izq1}, {izq2}, {der1}, {der2})"

    def giro_sar(self, nodo):
        base = Emisor.nombre_serie(self, nodo['sar'])
        signo = '1' if nodo['sentido'] > 0 else '-1'
        contrario = '-1' if nodo['sentido'] > 0 else '1'
        return f"({base}_tend_1 == {signo} && {base}_tend_2 == {contrario})"

    # ══════════════ handles de indicadores nativos ══════════════

    def _nombre_handle(self, nodo):
        """%K y %D salen del MISMO handle de iStochastic (buffers 0 y 1), así
        que comparten nombre: crear dos handles idénticos desperdiciaría
        recursos y podría desincronizarlos."""
        base = Emisor.nombre_serie(self, nodo)
        if nodo['tipo'] in ('STOCH_K', 'STOCH_D'):
            base = base.replace('stoch_k', 'stoch').replace('stoch_d', 'stoch')
        return f"h_{base}"

    def _handles(self, ir_setup):
        vistos, fuera = set(), []
        for _nombre, nodo in self.series_declarables(ir_setup):
            if nodo['tipo'] not in _NATIVOS:
                continue
            handle = self._nombre_handle(nodo)
            if handle in vistos:
                continue
            vistos.add(handle)
            fuera.append((handle, nodo))
        return fuera

    def _crear_handle(self, nodo):
        tipo = nodo['tipo']
        arg = lambda campo: self.arg_serie(nodo, campo)   # noqa: E731
        if tipo in ('SMA', 'EMA'):
            modo = _NATIVOS[tipo][1]
            return (f"iMA(_Symbol, _Period, {arg('periodo')}, 0, {modo}, "
                    f"PRICE_CLOSE)")
        if tipo == 'RSI':
            return f"iRSI(_Symbol, _Period, {arg('periodo')}, PRICE_CLOSE)"
        if tipo == 'CCI':
            # el motor usa el precio típico (h+l+c)/3, que es PRICE_TYPICAL
            return f"iCCI(_Symbol, _Period, {arg('periodo')}, PRICE_TYPICAL)"
        if tipo == 'WILLR':
            return f"iWPR(_Symbol, _Period, {arg('periodo')})"
        if tipo in ('STOCH_K', 'STOCH_D'):
            return (f"iStochastic(_Symbol, _Period, {arg('periodo_k')}, "
                    f"{arg('periodo_d')}, {arg('suavizado_k')}, MODE_SMA, "
                    f"STO_LOWHIGH)")
        raise ValueError(f"{self.nombre}: sin handle nativo para {tipo!r}")

    # ══════════════ declaración de series ══════════════

    def declarar_serie(self, nombre, nodo):
        tipo = nodo['tipo']
        shift = self._shift
        arg = lambda campo: self.arg_serie(nodo, campo)   # noqa: E731

        if tipo in _NATIVOS:
            buffer = _NATIVOS[tipo][1] if tipo.startswith('STOCH') else 0
            return (f"   double {nombre} = zcsValor("
                    f"{self._nombre_handle(nodo)}, {buffer}, {shift});")
        if tipo == 'ATR':
            return f"   double {nombre} = zcsAtr({arg('periodo')}, {shift});"
        if tipo == 'BB_media':
            return (f"   double {nombre} = zcsBbMedia({arg('periodo')}, "
                    f"{shift});")
        if tipo == 'BB_sup':
            return (f"   double {nombre} = zcsBbSup({arg('periodo')}, "
                    f"{arg('desv')}, {shift});")
        if tipo == 'BB_inf':
            return (f"   double {nombre} = zcsBbInf({arg('periodo')}, "
                    f"{arg('desv')}, {shift});")
        if tipo == 'ER':
            return f"   double {nombre} = zcsEr({arg('periodo')}, {shift});"
        if tipo == 'KAMA':
            return (f"   double {nombre} = zcsKama({arg('periodo')}, "
                    f"{arg('rapido')}, {arg('lento')}, {shift});")
        if tipo in ('DONCHIAN_SUP', 'DONCHIAN_INF'):
            cerrado = 'true' if nodo.get('fuente') == 'close' else 'false'
            funcion = ('zcsDonchianSup' if tipo == 'DONCHIAN_SUP'
                       else 'zcsDonchianInf')
            return (f"   double {nombre} = {funcion}({cerrado}, "
                    f"{arg('periodo')}, {shift});")
        if tipo == 'SAR':
            base = nombre[:-2]      # sin el sufijo de desplazamiento
            return (f"   double {nombre} = 0.0; int {base}_tend_{shift} = 0;\n"
                    f"   zcsSar({arg('af_inicial')}, {arg('af_paso')}, "
                    f"{arg('af_max')}, {shift}, {nombre}, "
                    f"{base}_tend_{shift});")
        if tipo == 'PCT_ATR':
            return (f"   double {nombre} = zcsPercentilAtr("
                    f"{self.num(nodo['periodo_base'])}, "
                    f"{self.num(nodo['ventana'])}, {shift});")
        if tipo == 'PCT_STDEV':
            return (f"   double {nombre} = zcsPercentilStdev("
                    f"{self.num(nodo['periodo_base'])}, "
                    f"{self.num(nodo['ventana'])}, {shift});")
        if tipo == 'HURST':
            raise ValueError(
                "El Hurst todavía no está portado a MQL5 (ver "
                "fidelidad.CAPACIDADES): este setup no debería haber llegado "
                "al emisor con el filtro de régimen puesto.")
        raise ValueError(f"{self.nombre}: serie sin traducción: {tipo!r}")

    # ══════════════ archivos ══════════════

    def archivos_setup(self, ir_setup, ir_sistema, avisos, meta=None):
        meta = meta or {}
        slug = self.identificador(meta.get('sistema', 'sistema'))
        nombre = f"{slug}_S{ir_setup['indice']}"
        texto = self.construir(ir_setup, ir_sistema, avisos, meta)
        return {f"MT5/MQL5/Experts/{nombre}{self.extension}": texto}

    def archivos_comunes(self, ir_sistema, avisos, meta=None):
        meta = meta or {}
        slug = self.identificador(meta.get('sistema', 'sistema'))
        with open(RUTA_RUNTIME, encoding='utf-8') as f:
            runtime = f.read()
        return {f"MT5/MQL5/Include/{slug}/zcs_runtime.mqh": runtime,
                "MT5/INSTALAR.md": _instalar_md(meta, slug)}

    # ══════════════ construcción del EA ══════════════

    def construir(self, ir_setup, ir_sistema, avisos, meta):
        slug = self.identificador(meta.get('sistema', 'sistema'))
        bloques = [
            self.cabecera(ir_setup, ir_sistema, avisos, meta),
            self._propiedades(slug),
            self._inputs(ir_setup, meta),
            self._estado(ir_setup),
            self._on_init(ir_setup, avisos, meta),
            self._on_deinit(ir_setup),
            self._on_tick(ir_setup),
        ]
        return "\n\n".join(b for b in bloques if b) + "\n"

    def _seccion(self, titulo):
        return (f"//+------------------------------------------------------------------+\n"
                f"//| {titulo:<64} |\n"
                f"//+------------------------------------------------------------------+")

    def _propiedades(self, slug):
        return "\n".join([
            '#property copyright "Analytics System"',
            '#property version   "1.00"',
            '#property strict',
            '',
            f'#include <{slug}/zcs_runtime.mqh>',
        ])

    def _inputs(self, ir_setup, meta):
        lineas = [self._seccion("Parametros"), ""]
        specs = {s['clave']: s
                 for s in ESTRATEGIAS[ir_setup['plantilla']]['params']}
        for clave in self.params_usados(ir_setup):
            spec = specs.get(clave)
            valor = ir_setup['params'].get(clave)
            etiqueta = _ascii(spec['etiqueta'] if spec else clave)
            if spec and spec['tipo'] == 'int':
                lineas.append(f"input int    p_{clave} = {int(valor)}; "
                              f"// {etiqueta}")
            else:
                lineas.append(f"input double p_{clave} = "
                              f"{self.num(float(valor))}; // {etiqueta}")

        g = ir_setup['gestion']
        unidad_be = 'R' if g['be_unidad'] == 'r' else 'x ATR'
        lineas += [
            "",
            f"input double p_riesgo_pct   = {self.num(g['riesgo_pct'] * 100)}; "
            f"// Riesgo por operacion (%)",
            f"input double p_stop_atr     = {self.num(g['stop_atr'])}; "
            f"// Stop (x ATR)",
            f"input double p_tp_r         = {self.num(g['tp_r'])}; "
            f"// Take-profit (R), 0 = sin TP",
            f"input double p_be_atr       = {self.num(g['be_atr'])}; "
            f"// Break-even ({unidad_be}), 0 = sin BE",
            f"input double p_trailing_atr = {self.num(g['trailing_atr'])}; "
            f"// Trailing (x ATR), 0 = sin trailing",
            f"input int    p_salida_velas = {g['salida_n_velas']}; "
            f"// Salida por tiempo (velas), 0 = sin limite",
            f"input int    p_periodo_atr  = {g['periodo_atr']}; "
            f"// Periodo del ATR de gestion",
            "",
            f'input string p_simbolo_esperado = "{_ascii(meta.get("activo", ""))}"; '
            f'// Simbolo del backtest',
            "input bool   p_permitir_otro = false; "
            "// Permitir otro activo o temporalidad",
            f"input ulong  p_magic = {_magic(meta.get('sistema', ''), ir_setup['indice'])}; "
            f"// Magic number (uno por setup)",
        ]
        return "\n".join(lineas)

    def _estado(self, ir_setup):
        return "\n".join([
            self._seccion("Estado de la posicion"),
            "",
            "double zcsDistEntrada  = 0.0;   // distancia al stop al entrar",
            "double zcsStopActual   = 0.0;   // stop vigente (lo mueven BE y trailing)",
            "double zcsMaxFav       = 0.0;   // extremo mas favorable alcanzado",
            "int    zcsBarraEntrada = 0;",
            "bool   zcsBeAplicado   = false;",
            "double zcsDistPendiente = 0.0;  // distancia calculada en la vela de la senal",
        ] + [f"int {handle} = INVALID_HANDLE;"
             for handle, _nodo in self._handles(ir_setup)])

    def _on_init(self, ir_setup, avisos, meta):
        minutos = _TF_MINUTOS.get(meta.get('tf') or '', 0)
        lineas = [
            self._seccion("OnInit"),
            "",
            "int OnInit()",
            "{",
            "   string motivo = \"\", aviso = \"\";",
            f"   if(!zcsActivoCorrecto(p_simbolo_esperado, {minutos}, "
            f"p_permitir_otro, motivo, aviso))",
            "   {",
            '      Print("Este EA se genero para otro contexto: ", motivo,',
            '            ". Activa PermitirOtroActivo si es a proposito.");',
            "      return(INIT_FAILED);",
            "   }",
            "   // el simbolo no impide arrancar, solo avisa: el nombre del CSV",
            "   // del backtest casi nunca coincide con el del broker",
            "   if(StringLen(aviso) > 0) Print(aviso);",
            "",
        ]
        for handle, nodo in self._handles(ir_setup):
            lineas.append(f"   {handle} = {self._crear_handle(nodo)};")
            lineas.append(f"   if({handle} == INVALID_HANDLE) return(INIT_FAILED);")
        texto = fidelidad.texto_runtime(avisos)
        if texto:
            lineas += ["", f'   Print({self.texto(texto)});']
        lineas += ["   return(INIT_SUCCEEDED);", "}"]
        return "\n".join(lineas)

    def _on_deinit(self, ir_setup):
        lineas = [self._seccion("OnDeinit"), "", "void OnDeinit(const int reason)",
                  "{"]
        for handle, _nodo in self._handles(ir_setup):
            lineas.append(f"   if({handle} != INVALID_HANDLE) "
                          f"IndicatorRelease({handle});")
        if len(lineas) == 4:
            lineas.append("   // sin handles nativos que liberar")
        lineas.append("}")
        return "\n".join(lineas)

    def _on_tick(self, ir_setup):
        lineas = [self._seccion("OnTick"), "", "void OnTick()", "{",
                  "   // El motor decide con la vela cerrada y ejecuta al open de",
                  "   // la siguiente: por eso solo se actua en la vela nueva.",
                  "   if(!zcsVelaNueva()) return;", ""]
        lineas += self._lecturas(ir_setup)
        lineas += ["", *self._filtros(ir_setup)]
        lineas += ["", *self._senales(ir_setup)]
        lineas += ["", *self._gestion(ir_setup)]
        lineas.append("}")
        return "\n".join(lineas)

    def _lecturas(self, ir_setup):
        """Precios e indicadores en la vela de la señal (1) y en la anterior
        (2). La 2 solo hace falta para los cruces, pero declararla siempre
        cuesta una llamada y evita razonar sobre qué serie la necesita."""
        lineas = ["   // --- precios de la vela de la senal (1) y la previa (2)"]
        for campo in ('open', 'high', 'low', 'close'):
            inicial = campo[0]
            lineas.append(
                f"   double {inicial}1 = i{campo.capitalize()}(_Symbol, _Period, 1), "
                f"{inicial}2 = i{campo.capitalize()}(_Symbol, _Period, 2);")
        lineas.append("")
        lineas.append("   // --- indicadores")
        lineas.append("   double atrGestion = zcsAtr(p_periodo_atr, 1);")
        for shift in (1, 2):
            self._shift = shift
            # series_declarables ya devuelve el nombre con el desplazamiento
            # pegado (nombre_serie está sobrescrito en este emisor)
            for nombre, nodo in self.series_declarables(ir_setup):
                lineas.append(self.declarar_serie(nombre, nodo))
        self._shift = 1
        return lineas

    def _filtros(self, ir_setup):
        from core.codegen.ir import condiciones_de_lado
        filtros = ir_setup['filtros']
        comunes = []
        lineas = ["   // --- filtros (solo condicionan nuevas entradas)"]

        if filtros['dias_semana'] is not None:
            dias = " || ".join(f"dia == {_DIAS_MQL[d]}"
                               for d in filtros['dias_semana'] if 0 <= d < 7)
            lineas += [
                "   MqlDateTime hora; TimeToStruct(iTime(_Symbol, _Period, 1), hora);",
                "   ENUM_DAY_OF_WEEK dia = (ENUM_DAY_OF_WEEK)hora.day_of_week;",
                f"   bool filtroDia = ({dias or 'true'});",
            ]
            comunes.append('filtroDia')

        ses = filtros['sesion']
        if ses:
            lineas += [
                "   MqlDateTime hs; TimeToStruct(iTime(_Symbol, _Period, 1), hs);",
                f"   int horaVela = hs.hour;",
            ]
            ini, fin = ses['hora_inicio'], ses['hora_fin']
            if ini <= fin:
                cond = f"(horaVela >= {ini} && horaVela < {fin})"
            else:
                cond = f"(horaVela >= {ini} || horaVela < {fin})"
            lineas.append(f"   bool filtroSesion = {cond};")
            comunes.append('filtroSesion')

        n = filtros['noticias']
        if n:
            antes = int(n['minutos_antes'])
            despues = int(n['minutos_despues'])
            umbral = _IMPACTO_MQL.get(n['impacto_minimo'], 'CALENDAR_IMPACT_HIGH')
            monedas = n.get('monedas') or []
            comillas = ', '.join(f'"{m}"' for m in monedas)
            lineas += [
                "   // --- filtro de noticias (calendario de MetaQuotes;",
                "   // aproximado: su impacto/divisas no coinciden con los",
                "   // del proveedor del backtest)",
                (f"   string zcsMonedas[] = {{{comillas}}};"
                 if monedas else "   string zcsMonedas[];"),
                f"   bool filtroNoticias = !zcsHayEvento(",
                f"      TimeCurrent() - {antes} * 60,",
                # la ventana hacia delante cubre al menos el resto de la vela
                # actual (en vivo solo se evalúa al abrir cada vela)
                f"      TimeCurrent() + MathMax({despues}, "
                "PeriodSeconds(_Period) / 60) * 60,",
                f"      {umbral}, zcsMonedas, {len(monedas)});",
            ]
            comunes.append('filtroNoticias')

        if filtros['regimen']:
            lineas.append(f"   bool filtroRegimen = "
                          f"{self.expr(filtros['regimen']['condicion'])};")
            comunes.append('filtroRegimen')

        if filtros['volatilidad']:
            lineas.append(f"   bool filtroVolatilidad = "
                          f"{self.expr(filtros['volatilidad']['condicion'])};")
            comunes.append('filtroVolatilidad')

        for lado in ('long', 'short'):
            cond = condiciones_de_lado(filtros['condiciones_entrada'], lado)
            partes = list(comunes)
            if cond is not None:
                partes.append(self.expr(cond))
            lineas.append(f"   bool filtro{lado.capitalize()} = "
                          + (" && ".join(partes) or "true") + ";")
        for lado in ('long', 'short'):
            cond = condiciones_de_lado(filtros['condiciones_salida'], lado)
            lineas.append(
                f"   bool permiteSalida{lado.capitalize()} = "
                + (self.expr(cond) if cond is not None else "true") + ";")
        return lineas

    def _senales(self, ir_setup):
        senales = ir_setup['senales']
        lineas = ["   // --- senales de la plantilla"]
        for lado, var in _VAR_SENAL.items():
            nodo = senales[lado]
            valor = self.expr(nodo) if nodo is not None else "false"
            lineas.append(f"   bool {var} = {valor};")
        return lineas

    def _gestion(self, ir_setup):
        g = ir_setup['gestion']
        ref_be = 'zcsDistEntrada' if g['be_unidad'] == 'r' else 'atrGestion'
        n = (ir_setup['filtros'] or {}).get('noticias')
        cierre_noticias = []
        if n and n.get('cerrar_posiciones'):
            umbral = _IMPACTO_MQL.get(n['impacto_minimo'],
                                      'CALENDAR_IMPACT_HIGH')
            antes = int(n['minutos_antes'])
            monedas = n.get('monedas') or []
            cierre_noticias = [
                "   // cierre por noticia inminente (aproximado, calendario",
                "   // de MetaQuotes): replica el cierre forzado del backtest",
                f"   if(zcsHayEvento(TimeCurrent(), TimeCurrent() + "
                f"MathMax({antes}, PeriodSeconds(_Period) / 60) * 60,",
                f"      {umbral}, zcsMonedas, {len(monedas)}))",
                "      { zcsCerrar(p_magic); return; }",
            ]
        return [
            "   // --- gestion de la posicion",
            "   // El motor mide la distancia con el ATR de la vela en cuya",
            "   // apertura entra; en vivo ese dato aun no existe, asi que se",
            "   // usa el de la ultima vela cerrada (ver cabecera).",
            "   double distRef = (p_stop_atr > 0.0 ? p_stop_atr * atrGestion",
            "                                      : 2.0 * atrGestion);",
            "   int dir = zcsDireccion(p_magic);",
            "",
            "   if(dir == 0)",
            "   {",
            "      int nuevaDir = 0;",
            "      if(entradaLong  && filtroLong)  nuevaDir =  1;",
            "      else if(entradaShort && filtroShort) nuevaDir = -1;",
            "      if(nuevaDir != 0)",
            "      {",
            "         double lotes = zcsLotesPorRiesgo(p_riesgo_pct / 100.0, distRef);",
            "         if(lotes > 0.0)",
            "         {",
            "            double precio = (nuevaDir > 0 ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)",
            "                                          : SymbolInfoDouble(_Symbol, SYMBOL_BID));",
            "            double stop = (p_stop_atr > 0.0",
            "                           ? precio - distRef * nuevaDir : 0.0);",
            "            double tp   = (p_tp_r > 0.0",
            "                           ? precio + p_tp_r * distRef * nuevaDir : 0.0);",
            "            if(zcsAbrir(nuevaDir, lotes, stop, tp, p_magic))",
            "            {",
            "               zcsDistEntrada  = distRef;",
            "               zcsStopActual   = stop;",
            "               zcsMaxFav       = precio;",
            "               zcsBarraEntrada = Bars(_Symbol, _Period);",
            "               zcsBeAplicado   = false;",
            "            }",
            "         }",
            "      }",
            "      return;",
            "   }",
            "",
            "   double precioIn = zcsPrecioEntrada(p_magic);",
            "   zcsMaxFav = (dir > 0 ? MathMax(zcsMaxFav, h1)",
            "                        : MathMin(zcsMaxFav, l1));",
            "",
            *cierre_noticias,
            "",
            "   // break-even: mueve el stop a la entrada cuando el avance a",
            "   // favor supera el umbral. La referencia se remide con el ATR",
            "   // de la vela actual, igual que el motor.",
            f"   double refBe = {ref_be};",
            "   if(p_be_atr > 0.0 && !zcsBeAplicado &&",
            "      (zcsMaxFav - precioIn) * dir >= p_be_atr * refBe)",
            "   {",
            "      zcsStopActual = precioIn;",
            "      zcsBeAplicado = true;",
            "   }",
            "",
            "   // trailing: el stop sigue al extremo alcanzado y nunca afloja",
            "   if(p_trailing_atr > 0.0)",
            "   {",
            "      double candidato = zcsMaxFav - p_trailing_atr * atrGestion * dir;",
            "      zcsStopActual = (dir > 0 ? MathMax(zcsStopActual, candidato)",
            "                               : MathMin(zcsStopActual, candidato));",
            "   }",
            "",
            "   double tpVivo = (p_tp_r > 0.0",
            "                    ? precioIn + p_tp_r * zcsDistEntrada * dir : 0.0);",
            "   if(p_stop_atr > 0.0 || p_trailing_atr > 0.0 || p_be_atr > 0.0)",
            "      zcsMoverStop(zcsStopActual, tpVivo, p_magic);",
            "",
            "   // salida por la senal de la plantilla",
            "   if(dir > 0 && salidaLong  && permiteSalidaLong)  "
            "{ zcsCerrar(p_magic); return; }",
            "   if(dir < 0 && salidaShort && permiteSalidaShort) "
            "{ zcsCerrar(p_magic); return; }",
            "",
            "   // salida por tiempo",
            "   if(p_salida_velas > 0 &&",
            "      Bars(_Symbol, _Period) - zcsBarraEntrada >= p_salida_velas)",
            "      zcsCerrar(p_magic);",
        ]


_VAR_SENAL = {'entradas_long': 'entradaLong', 'entradas_short': 'entradaShort',
              'salidas_long': 'salidaLong', 'salidas_short': 'salidaShort'}


def _ascii(texto):
    """MetaEditor guarda en ANSI o UTF-8 según la versión, y un comentario con
    acentos puede salir en mojibake. Los textos del código generado van sin
    acentos a propósito."""
    import unicodedata
    plano = unicodedata.normalize('NFKD', str(texto))
    return "".join(c for c in plano if not unicodedata.combining(c))


def _magic(nombre_sistema, indice):
    """Magic number estable: mismo sistema y mismo setup dan siempre el mismo
    número, para que reexportar no deje al EA sin reconocer sus propias
    posiciones abiertas. Se deriva con crc32 (determinista entre ejecuciones,
    al contrario que hash())."""
    semilla = f"{nombre_sistema}#{indice}".encode('utf-8')
    return 500000 + (zlib.crc32(semilla) % 400000)


def _instalar_md(meta, slug):
    activo = meta.get('activo', '?')
    tf = meta.get('tf', '?')
    return f"""# Instalar en MetaTrader 5

1. En MetaTrader: *Archivo → Abrir carpeta de datos*.
2. Copia el contenido de la carpeta `MQL5/` de aquí dentro de la `MQL5/` que
   se te ha abierto, respetando la estructura:
   - `MQL5/Experts/{slug}_S0.mq5`  (uno por cada setup)
   - `MQL5/Include/{slug}/zcs_runtime.mqh`
3. Abre el `.mq5` en MetaEditor (F4) y pulsa *Compilar* (F7).
4. Vuelve a MetaTrader, actualiza el Navegador y arrastra el EA a un gráfico
   de **{activo}**, temporalidad **{tf}**.

## Antes de operar

- **Cuenta HEDGING** si el sistema tiene más de un setup. En netting todas las
  órdenes del mismo símbolo se funden en una sola posición y los EAs se
  pisarían entre ellos.
- El EA **se niega a arrancar** en otra temporalidad distinta a la del
  backtest. Si es a propósito, activa `PermitirOtroActivo`.
- El **magic number** es distinto para cada setup y así cada EA solo toca sus
  propias posiciones. No lo cambies si ya tienes operaciones abiertas.
- El tamaño se calcula por riesgo y se redondea al paso de lote del bróker.
  Si el riesgo pedido queda por debajo del lote mínimo, el EA **no abre** la
  operación en vez de arriesgar de más.

Lee `NOTAS_DE_FIDELIDAD.md` en la carpeta de arriba: dice exactamente en qué se
diferencia este EA del backtest.
"""
