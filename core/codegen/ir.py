"""
core/codegen/ir.py
Representación intermedia (IR) neutra de un sistema del Backtester.

Convierte los dicts de setup en un árbol de nodos sin nada específico de
ningún lenguaje, para que cada emisor (MQL4/MQL5/Pine/...) solo tenga que
saber escribir nodos y no volver a interpretar la configuración.

POR QUÉ NO SE CONSTRUYE SOBRE filas_plantilla
─────────────────────────────────────────────
filas_plantilla() se declara «la única fuente de verdad de qué hace esta
plantilla», y lo es para la tabla de condiciones de la GUI y para el
pseudocódigo. Pero está pensada para una tabla que solo ofrece un «Periodo»
por indicador, así que PIERDE parámetros que aquí son obligatorios:

  · KAMA      la fila usa las constantes del editor de reglas (rápido=2,
              lento=30); la plantilla puede correr con otras — _NOTA_KAMA.
  · Donchian  la fila forma el canal siempre con máximos/mínimos; con
              fuente='close' la plantilla usa un canal de cierres —
              _NOTA_DONCHIAN.
  · Stochastic, Williams %R, CCI, Parabolic SAR y Patrones de velas degradan
    a _fila_texto: una frase, sin celdas.
  · Custom (reglas) también degrada a texto, aunque sus reglas SÍ estén
    estructuradas dentro de params.

Un EA generado a partir de esas filas operaría distinto al backtest que el
usuario acaba de ver, que es justo lo que esta función no se puede permitir.
Por eso el IR se construye sobre los mismos parámetros que consumen los
_gen_* de core/strategies.py — los que de verdad ejecuta el motor — y la
coincidencia con filas_plantilla se vigila desde tests/test_codegen_ir.py en
las plantillas donde ambas representaciones son estructurales.

FORMA DE LOS NODOS
──────────────────
Serie (algo que tiene un valor por vela). El vocabulario de 'tipo' extiende
el de _serie_indicador (core/strategies.py) con lo que las plantillas de solo
texto necesitan:

    {'tipo': 'close'|'open'|'high'|'low'}
    {'tipo': 'valor', 'valor': float}
    {'tipo': 'SMA'|'EMA'|'RSI'|'ATR'|'CCI'|'WILLR', 'periodo': int}
    {'tipo': 'BB_sup'|'BB_inf'|'BB_media', 'periodo': int, 'desv': float}
    {'tipo': 'KAMA', 'periodo': int, 'rapido': int, 'lento': int}
    {'tipo': 'ER', 'periodo': int}
    {'tipo': 'HURST', 'periodo': int}
    {'tipo': 'STOCH_K'|'STOCH_D', 'periodo_k','suavizado_k','periodo_d'}
    {'tipo': 'DONCHIAN_SUP'|'DONCHIAN_INF', 'periodo': int, 'fuente': str}
    {'tipo': 'SAR', 'af_inicial': f, 'af_paso': f, 'af_max': f}
    {'tipo': 'ZIGZAG', 'desviacion': float, 'piernas': int}
    {'tipo': 'PCT_ATR'|'PCT_STDEV', 'ventana': int, 'periodo_base': int}

Booleano (algo que es cierto o falso por vela):

    {'op': '>'|'<'|'cruza arriba'|'cruza abajo', 'izq': serie, 'der': serie}
    {'op': 'giro_sar', 'sentido': +1|-1, 'sar': serie SAR}
    {'op': 'patron', 'nombre': str, 'sesgo': -1|0|+1, 'indice': int}
    {'op': 'Y'|'O'|'NO', 'partes': [nodo, ...]}

Todos los nodos son dicts planos y serializables: un emisor puede volcarlos a
JSON para depurar sin perder nada.
"""
from core.backtest import MECANISMOS_SALIDA
from core.candle_patterns import PATRONES_INFO, SESIONES
from core.strategies import (
    ESTRATEGIAS, PERIODO_ATR_DEFECTO, params_por_defecto,
    UMBRAL_ER_TENDENCIA, UMBRAL_ER_RUIDO,
    UMBRAL_HURST_TENDENCIA, UMBRAL_HURST_REVERSION,
    METODOS_REGIMEN_ER, METODOS_REGIMEN_HURST, ventana_regimen_defecto,
    _KAMA_RAPIDO_REGLA, _KAMA_LENTO_REGLA,
    _SAR_AF_INICIAL_REGLA, _SAR_AF_PASO_REGLA, _SAR_AF_MAX_REGLA,
    _ZIGZAG_DESVIACION_REGLA, _ZIGZAG_PIERNAS_REGLA,
    _lags_hurst_defecto, trigger_etapa,
)

LADOS = ('entradas_long', 'entradas_short', 'salidas_long', 'salidas_short')

OPS_COMPARACION = ('>', '<', 'cruza arriba', 'cruza abajo')
OPS_LOGICOS = ('Y', 'O', 'NO')
OPS_ESPECIALES = ('giro_sar', 'patron')

# Tipos de serie que ningún emisor puede resolver con el indicador estándar de
# su plataforma y que obligan a la librería de runtime portada. Se listan aquí
# (y no en cada emisor) porque la lista depende de nuestras fórmulas, no del
# lenguaje destino.
SERIES_RUNTIME = ('KAMA', 'ER', 'HURST', 'SAR', 'ZIGZAG',
                  'PCT_ATR', 'PCT_STDEV')


# ══════════════ constructores de nodo ══════════════

def serie(tipo, origen=None, **kw):
    """Nodo de serie.

    'origen' dice de qué parámetro de la plantilla salió cada número
    ({'periodo': 'rapida'}). No lo necesita el motor —los valores ya están
    interpolados— pero sí los emisores: sin él tendrían que adivinar qué
    literal corresponde a qué campo del formulario, y el código generado se
    quedaría con constantes en vez de con parámetros editables desde la
    plataforma."""
    nodo = {'tipo': tipo}
    nodo.update(kw)
    if origen:
        nodo['origen'] = dict(origen)
    return nodo


def valor(x, origen=None):
    """Constante. 'origen' es la clave del parámetro de plantilla del que
    viene, cuando viene de uno (los umbrales fijos como el 50 del RSI no
    tienen origen: no son configurables)."""
    nodo = {'tipo': 'valor', 'valor': float(x)}
    if origen:
        nodo['origen'] = origen
    return nodo


def precio(campo='close'):
    return {'tipo': campo}


def cmp(izq, op, der):
    """Comparación entre dos series. `op` en OPS_COMPARACION."""
    if op not in OPS_COMPARACION:
        raise ValueError(f"Operador de comparación desconocido: {op}")
    return {'op': op, 'izq': izq, 'der': der}


def y(*partes):
    """AND de nodos booleanos, aplanando los AND anidados y descartando los
    None (un filtro inactivo se representa con None para que quien lo
    construye no tenga que comprobarlo antes de pasarlo)."""
    return _logico('Y', partes)


def o(*partes):
    return _logico('O', partes)


def no(parte):
    return None if parte is None else {'op': 'NO', 'partes': [parte]}


def _logico(op, partes):
    planas = []
    for p in partes:
        if p is None:
            continue
        if p.get('op') == op:
            planas.extend(p['partes'])
        else:
            planas.append(p)
    if not planas:
        return None
    if len(planas) == 1:
        return planas[0]
    return {'op': op, 'partes': planas}


# ══════════════ señales de cada plantilla ══════════════

def _lados_activos(p):
    """(hace_long, hace_short) según el parámetro 'direccion'. Mismo criterio
    que _lados de core/strategies.py; las plantillas sin ese parámetro
    (Patrones de velas) operan los dos lados."""
    d = p.get('direccion', 'Ambas')
    return d in ('Long', 'Ambas'), d in ('Short', 'Ambas')


def _vacio():
    return {clave: None for clave in LADOS}


def _ir_cruce_medias(p):
    # 'tipo' (SMA/EMA) no lleva origen a propósito: no es un número que se
    # sustituya, sino qué media se calcula. Cambiarlo cambia el código.
    tipo = 'EMA' if p['tipo'] == 'EMA' else 'SMA'
    rapida = serie(tipo, periodo=int(p['rapida']), origen={'periodo': 'rapida'})
    lenta = serie(tipo, periodo=int(p['lenta']), origen={'periodo': 'lenta'})
    arriba = cmp(rapida, 'cruza arriba', lenta)
    abajo = cmp(rapida, 'cruza abajo', lenta)
    return _direccional(p, arriba, abajo, abajo, arriba)


def _ir_bollinger(p):
    per, desv = int(p['periodo']), float(p['desv'])
    org = {'periodo': 'periodo', 'desv': 'desv'}
    inf = serie('BB_inf', periodo=per, desv=desv, origen=org)
    sup = serie('BB_sup', periodo=per, desv=desv, origen=org)
    media = serie('BB_media', periodo=per, desv=desv, origen=org)
    c = precio()
    return _direccional(p,
                        cmp(c, '<', inf), cmp(c, '>', sup),
                        cmp(c, '>', media), cmp(c, '<', media))


def _ir_rsi(p):
    r = serie('RSI', periodo=int(p['periodo']), origen={'periodo': 'periodo'})
    # el 50 de la salida es la línea media del oscilador, no un parámetro:
    # va sin origen porque no hay nada que editar en el formulario
    return _direccional(p,
                        cmp(r, '<', valor(p['sobreventa'], 'sobreventa')),
                        cmp(r, '>', valor(p['sobrecompra'], 'sobrecompra')),
                        cmp(r, '>', valor(50.0)),
                        cmp(r, '<', valor(50.0)))


def _ir_stochastic(p):
    """Entrada híbrida (cruce de %K sobre %D CONFIRMADO dentro de la zona
    extrema) y salida al recuperar la línea media — ver _gen_stochastic."""
    args = dict(periodo_k=int(p['periodo_k']),
                suavizado_k=int(p['suavizado_k']),
                periodo_d=int(p['periodo_d']))
    org = {'periodo_k': 'periodo_k', 'suavizado_k': 'suavizado_k',
           'periodo_d': 'periodo_d'}
    k = serie('STOCH_K', origen=org, **args)
    d = serie('STOCH_D', origen=org, **args)
    return _direccional(
        p,
        y(cmp(k, 'cruza arriba', d),
          cmp(k, '<', valor(p['sobreventa'], 'sobreventa'))),
        y(cmp(k, 'cruza abajo', d),
          cmp(k, '>', valor(p['sobrecompra'], 'sobrecompra'))),
        cmp(k, '>', valor(50.0)),
        cmp(k, '<', valor(50.0)))


def _ir_williams(p):
    wr = serie('WILLR', periodo=int(p['periodo']), origen={'periodo': 'periodo'})
    return _direccional(p,
                        cmp(wr, '<', valor(p['sobreventa'], 'sobreventa')),
                        cmp(wr, '>', valor(p['sobrecompra'], 'sobrecompra')),
                        cmp(wr, '>', valor(-50.0)),
                        cmp(wr, '<', valor(-50.0)))


def _ir_cci(p):
    v = serie('CCI', periodo=int(p['periodo']), origen={'periodo': 'periodo'})
    return _direccional(p,
                        cmp(v, '<', valor(p['sobreventa'], 'sobreventa')),
                        cmp(v, '>', valor(p['sobrecompra'], 'sobrecompra')),
                        cmp(v, '>', valor(0.0)),
                        cmp(v, '<', valor(0.0)))


def _ir_kama(p):
    """A diferencia de la fila de la GUI, aquí sí viajan 'rapido' y 'lento':
    son los que usa la plantilla y sin ellos el KAMA emitido sería otro."""
    kama = serie('KAMA', periodo=int(p['periodo_er']),
                 rapido=int(p['rapido']), lento=int(p['lento']),
                 origen={'periodo': 'periodo_er', 'rapido': 'rapido',
                         'lento': 'lento'})
    c = precio()
    arriba = cmp(c, 'cruza arriba', kama)
    abajo = cmp(c, 'cruza abajo', kama)
    return _direccional(p, arriba, abajo, abajo, arriba)


def _ir_breakout(p):
    """La fuente define tanto qué forma el canal como qué lo rompe: con
    'close' es una ruptura de CIERRES por los dos lados, no un cierre medido
    contra un canal de máximos — ver _gen_breakout."""
    fuente = p['fuente']
    usa_extremos = fuente == 'high/low'
    src_arriba = precio('high' if usa_extremos else 'close')
    src_abajo = precio('low' if usa_extremos else 'close')
    per = int(p['periodo'])
    org = {'periodo': 'periodo'}
    sup = serie('DONCHIAN_SUP', periodo=per, fuente=fuente, origen=org)
    inf = serie('DONCHIAN_INF', periodo=per, fuente=fuente, origen=org)
    rompe_arriba = cmp(src_arriba, '>', sup)
    rompe_abajo = cmp(src_abajo, '<', inf)

    per_salida = int(p.get('periodo_salida') or 0)
    if per_salida > 0 and per_salida != per:
        org_s = {'periodo': 'periodo_salida'}
        sup_s = serie('DONCHIAN_SUP', periodo=per_salida, fuente=fuente,
                      origen=org_s)
        inf_s = serie('DONCHIAN_INF', periodo=per_salida, fuente=fuente,
                      origen=org_s)
        salida_long = cmp(src_abajo, '<', inf_s)
        salida_short = cmp(src_arriba, '>', sup_s)
    else:
        salida_long, salida_short = rompe_abajo, rompe_arriba

    return _direccional(p, rompe_arriba, rompe_abajo, salida_long, salida_short)


def _ir_sar(p):
    sar = serie('SAR', af_inicial=float(p['af_inicial']),
                af_paso=float(p['af_paso']), af_max=float(p['af_max']),
                origen={'af_inicial': 'af_inicial', 'af_paso': 'af_paso',
                        'af_max': 'af_max'})
    alcista = {'op': 'giro_sar', 'sentido': +1, 'sar': sar}
    bajista = {'op': 'giro_sar', 'sentido': -1, 'sar': sar}
    return _direccional(p, alcista, bajista, bajista, alcista)


def _ir_patrones(p):
    """Un nodo por patrón elegido, unidos por O. La dirección de cada uno la
    decide el sesgo del patrón, no un parámetro del setup, así que esta
    plantilla no pasa por _direccional. 'indice' es la posición en la lista,
    que el motor usa como setup_id para permitir riesgo por patrón."""
    s = _vacio()
    largos, cortos = [], []
    for k, nombre in enumerate(p.get('patrones') or []):
        info = PATRONES_INFO.get(nombre)
        if info is None:
            continue
        nodo = {'op': 'patron', 'nombre': nombre,
                'sesgo': int(info['dir']), 'indice': k}
        # sesgo 0 = giro que depende del contexto: detectar_patrones resuelve
        # la dirección vela a vela, así que el patrón puede entrar por los dos
        # lados y hay que ofrecerlo a ambos
        if info['dir'] >= 0:
            largos.append(nodo)
        if info['dir'] <= 0:
            cortos.append(nodo)
    s['entradas_long'] = o(*largos)
    s['entradas_short'] = o(*cortos)
    # la salida es por tiempo (salida_n_velas), no por señal — la resuelve el
    # plan de gestión, no este bloque
    return s


def _ir_custom(p):
    """Las reglas custom ya vienen estructuradas en params: cada lado es una
    lista de reglas y cada regla un AND de condiciones, así que el lado entero
    es el OR de sus reglas."""
    s = _vacio()
    reglas = p.get('reglas') or {}
    for clave in LADOS:
        nodos = []
        for regla in reglas.get(clave, []):
            conds = [_cond_a_nodo(c) for c in regla.get('condiciones', [])]
            nodo = y(*conds)
            if nodo is not None:
                nodos.append(nodo)
        s[clave] = o(*nodos)
    return s


def _cond_a_nodo(cond):
    """Condición del editor de reglas -> nodo. Las specs del editor solo
    llevan 'periodo', así que los indicadores que necesitan más números usan
    aquí las mismas constantes fijas que _serie_indicador."""
    return cmp(_spec_a_serie(cond['izq']), cond['op'],
               _spec_a_serie(cond['der']))


def _spec_a_serie(spec):
    tipo = spec['tipo']
    if tipo == 'valor':
        return valor(spec['valor'])
    if tipo in ('close', 'open', 'high', 'low'):
        return precio(tipo)
    periodo = int(spec.get('periodo', 14))
    if tipo in ('BB_sup', 'BB_inf', 'BB_media'):
        return serie(tipo, periodo=periodo, desv=float(spec.get('desv', 2.0)))
    if tipo == 'KAMA':
        return serie('KAMA', periodo=periodo,
                     rapido=_KAMA_RAPIDO_REGLA, lento=_KAMA_LENTO_REGLA)
    if tipo == 'SAR':
        return serie('SAR', af_inicial=_SAR_AF_INICIAL_REGLA,
                     af_paso=_SAR_AF_PASO_REGLA, af_max=_SAR_AF_MAX_REGLA)
    if tipo == 'ZIGZAG':
        return serie('ZIGZAG', desviacion=_ZIGZAG_DESVIACION_REGLA,
                     piernas=int(spec.get('periodo', _ZIGZAG_PIERNAS_REGLA)))
    if tipo in ('DONCHIAN_SUP', 'DONCHIAN_INF'):
        return serie(tipo, periodo=periodo, fuente='high/low')
    if tipo in ('SMA', 'EMA', 'RSI', 'ATR', 'ER'):
        return serie(tipo, periodo=periodo)
    raise ValueError(f"Indicador desconocido en el IR: {tipo}")


def _direccional(p, ent_long, ent_short, sal_long, sal_short):
    """Aplica el filtro de dirección de la plantilla: el lado apagado se
    queda en None (no hay señal), no en «nunca cierto»."""
    hace_long, hace_short = _lados_activos(p)
    return {
        'entradas_long': ent_long if hace_long else None,
        'entradas_short': ent_short if hace_short else None,
        'salidas_long': sal_long if hace_long else None,
        'salidas_short': sal_short if hace_short else None,
    }


_CONSTRUCTORES = {
    'Cruce de medias': _ir_cruce_medias,
    'Bollinger + ATR': _ir_bollinger,
    'RSI': _ir_rsi,
    'Stochastic (%K/%D)': _ir_stochastic,
    'Williams %R': _ir_williams,
    'CCI': _ir_cci,
    'KAMA': _ir_kama,
    'Breakout de canal (Donchian)': _ir_breakout,
    'Parabolic SAR': _ir_sar,
    'Patrones de velas': _ir_patrones,
    'Custom (reglas)': _ir_custom,
}


def senales_plantilla(plantilla, params=None):
    """Nodos de señal de una plantilla: {'entradas_long', 'entradas_short',
    'salidas_long', 'salidas_short'}, cada uno un nodo booleano o None si ese
    lado no opera."""
    if plantilla not in _CONSTRUCTORES:
        raise ValueError(f"Plantilla sin constructor de IR: {plantilla}")
    p = params_por_defecto(plantilla)
    p.update(params or {})
    return _CONSTRUCTORES[plantilla](p)


# ══════════════ filtros del setup ══════════════

def _ir_regimen(reg):
    """Filtro de régimen -> comparación contra un umbral ABSOLUTO.

    Los umbrales no dependen de la distribución del activo (los impone el
    Backtester vía `umbrales_er`, ver preparar_contexto), así que se traducen
    literales y no hay nada que calibrar en la plataforma destino.

    'nan_como' es el valor con el que preparar_contexto rellena el warm-up:
    0.0 en ER y 0.5 en Hurst. Importa — con ER, las velas de calentamiento
    quedan por debajo del umbral de ruido y por tanto SÍ pasan el filtro
    'er_rango'. Un emisor que las descarte cambiaría el número de trades."""
    metodo = (reg or {}).get('metodo', 'ninguno')
    if metodo == 'ninguno':
        return None
    periodo = int((reg or {}).get('periodo') or ventana_regimen_defecto(metodo))
    if metodo in METODOS_REGIMEN_ER:
        s = serie('ER', periodo=periodo)
        op, umbral = ('>', UMBRAL_ER_TENDENCIA) if metodo == 'er_tendencia' \
            else ('<', UMBRAL_ER_RUIDO)
        nan_como = 0.0
    elif metodo in METODOS_REGIMEN_HURST:
        s = serie('HURST', periodo=periodo)
        op, umbral = ('>', UMBRAL_HURST_TENDENCIA) if metodo == 'hurst_tendencia' \
            else ('<', UMBRAL_HURST_REVERSION)
        nan_como = 0.5
    else:
        raise ValueError(f"Método de régimen desconocido: {metodo}")
    ir_reg = {'metodo': metodo, 'periodo': periodo, 'serie': s,
              'op': op, 'umbral': float(umbral), 'nan_como': nan_como,
              'condicion': cmp(s, op, valor(umbral))}
    if metodo in METODOS_REGIMEN_HURST:
        lags, paso = _lags_hurst_defecto(periodo)
        ir_reg['lags'] = [int(x) for x in lags]
        ir_reg['paso'] = int(paso)
    return ir_reg


def _ir_volatilidad(vol):
    """Filtro de volatilidad -> percentil rodante del ATR (o de la desviación
    estándar de retornos log) dentro de su propia ventana.

    El indicador se calcula SIEMPRE con PERIODO_ATR_DEFECTO velas y 'ventana'
    es solo el histórico contra el que se compara: son dos cosas distintas a
    propósito (ver _volatilidad_serie). El lado 'bajo' exige además percentil
    >= 0 porque el warm-up se mapea a -1 y no debe colarse por el corte."""
    metodo = (vol or {}).get('metodo', 'ninguno')
    if metodo == 'ninguno':
        return None
    ventana = int((vol or {}).get('periodo', 100))
    umbral = float((vol or {}).get('percentil', 50.0))
    tipo = 'PCT_ATR' if metodo.startswith('atr') else 'PCT_STDEV'
    s = serie(tipo, ventana=ventana, periodo_base=PERIODO_ATR_DEFECTO)
    if metodo.endswith('alto'):
        condicion = cmp(s, '>', valor(umbral - 1e-12))
    else:
        condicion = y(cmp(s, '>', valor(-1e-12)),
                      cmp(s, '<', valor(umbral + 1e-12)))
    return {'metodo': metodo, 'ventana': ventana, 'percentil': umbral,
            'serie': s, 'lado': 'alto' if metodo.endswith('alto') else 'bajo',
            'nan_como': -1.0, 'condicion': condicion}


def _ir_sesion(ses):
    """Filtro de sesión. 'londres'/'ny' llevan huso IANA y hora LOCAL de esa
    plaza, así que su rango efectivo en UTC se mueve con el horario de verano;
    'overnight' y 'personalizada' son UTC fijo. La diferencia decide si un
    emisor puede traducirlo fiel o solo aproximarlo."""
    tipo = (ses or {}).get('tipo', 'ninguna')
    if tipo == 'ninguna':
        return None
    if tipo == 'personalizada':
        return {'tipo': tipo, 'tz': None,
                'hora_inicio': int((ses or {}).get('hora_inicio', 0)),
                'hora_fin': int((ses or {}).get('hora_fin', 0))}
    cfg = SESIONES.get(tipo)
    if cfg is None:
        return None
    if 'tz' in cfg:
        h_ini, h_fin = cfg['local']
        return {'tipo': tipo, 'tz': cfg['tz'],
                'hora_inicio': h_ini, 'hora_fin': h_fin}
    h_ini, h_fin = cfg['utc']
    return {'tipo': tipo, 'tz': None, 'hora_inicio': h_ini, 'hora_fin': h_fin}


def _ir_noticias(noticias):
    n = noticias or {}
    if not n.get('activo'):
        return None
    return {'minutos_antes': int(n.get('minutos_antes', 30)),
            'minutos_despues': int(n.get('minutos_despues', 30)),
            'impacto_minimo': n.get('impacto_minimo', 'alto'),
            'monedas': list(n['monedas']) if n.get('monedas') else None,
            'cerrar_posiciones': bool(n.get('cerrar_posiciones'))}


def _ir_condiciones(condiciones):
    """Condiciones de filtro con su dirección. Se devuelven por separado (y no
    ya unidas por AND) porque cada una solo entra en el lado al que apunta —
    ver _mascaras_condiciones_dir."""
    fuera = []
    for cond in (condiciones or []):
        fuera.append({'condicion': _cond_a_nodo(cond),
                      'direccion': cond.get('direccion', 'ambas')})
    return fuera


def filtros_setup(filtros):
    """IR de los filtros de un setup. Cada eje es None cuando está inactivo,
    de modo que `if ir['regimen']` basta para saber si hay que emitirlo."""
    f = filtros or {}
    dias = f.get('dias_semana')
    return {
        'dias_semana': sorted(int(d) for d in dias) if dias else None,
        'regimen': _ir_regimen(f.get('regimen')),
        'volatilidad': _ir_volatilidad(f.get('volatilidad')),
        'sesion': _ir_sesion(f.get('sesion')),
        'noticias': _ir_noticias(f.get('noticias')),
        'condiciones_entrada': _ir_condiciones(f.get('condiciones_entrada')),
        'condiciones_salida': _ir_condiciones(f.get('condiciones_salida')),
    }


def condiciones_de_lado(condiciones, direccion):
    """AND de las condiciones de _ir_condiciones que aplican a un lado
    ('long' o 'short'). None = sin restricción."""
    partes = [c['condicion'] for c in condiciones
              if c['direccion'] in ('ambas', direccion)]
    return y(*partes)


# ══════════════ plan de gestión ══════════════

def _ir_etapa_salida(etapa, orden):
    """Etapa de salida parcial. 'pct' es porcentaje del tamaño ORIGINAL de la
    posición, no de lo que queda abierto — la diferencia importa al emitir el
    cierre parcial."""
    trigger = trigger_etapa(etapa)
    ir = {'orden': orden, 'pct': float(etapa.get('pct', 100.0)),
          'trigger': trigger, 'gestion': _ir_gestion(etapa.get('gestion')),
          'condiciones': _ir_condiciones(etapa.get('condiciones'))}
    if trigger == 'r':
        ir['r'] = float(etapa.get('r', 0.0))
    elif trigger == 'estancamiento':
        ir['velas_max'] = int(etapa.get('velas_max', 10))
        ir['r_min'] = float(etapa.get('r_min', 1.0))
    return ir


def _ir_tramo_entrada(tramo, orden):
    """Tramo de entrada escalonada. 'pct' es porcentaje del RIESGO total del
    setup, no del capital: el tamaño de cada tramo se recalcula con su propia
    distancia al stop en el momento de entrar."""
    trigger = tramo.get('trigger', 'senal')
    return {'orden': orden, 'pct': float(tramo.get('pct', 100.0)),
            'trigger': trigger, 'val': float(tramo.get('val', 0.0)),
            'gestion': _ir_gestion(tramo.get('gestion')),
            'condiciones': _ir_condiciones(tramo.get('condiciones'))}


def _ir_gestion(gestion):
    """Gestión que dispara una etapa/tramo al ejecutarse: mover el stop a
    break-even, activar un trailing o llevarlo al precio de referencia
    anterior. Códigos según _aplicar_gestion_parcial (core/backtest.py)."""
    g = gestion or {}
    tipo = int(g.get('tipo', 0))
    nombres = {0: 'ninguna', 1: 'break_even', 2: 'trailing',
               3: 'stop_a_referencia_previa'}
    return {'tipo': tipo, 'nombre': nombres.get(tipo, 'ninguna'),
            'val': float(g.get('val', 0.0))}


def _ir_mecanismo(setup, clave):
    """Cierre parcial asociado a un mecanismo de salida (stop, TP, BE,
    trailing, tiempo). 'pct' aquí es porcentaje de lo que queda ABIERTO, y el
    mecanismo solo puede cerrar parcialmente UNA vez por posición: a partir de
    ahí cierra todo — ver la regla de disparo único en core/backtest.py."""
    mec = setup.get(clave)
    if not mec:
        return None
    pct = float(mec.get('pct', 100.0))
    if pct >= 100.0 and not mec.get('condiciones'):
        return None      # equivale a no configurarlo: no merece código
    return {'clave': clave, 'pct': pct,
            'gestion': _ir_gestion(mec.get('gestion')),
            'condiciones': _ir_condiciones(mec.get('condiciones'))}


def _salida_n_velas(setup, params):
    """Salida por tiempo en velas. Patrones de velas la toma del parámetro
    'lag_salida' de la plantilla cuando el setup no fija una propia — misma
    regla que aplica el hilo de backtest antes de llamar al motor
    (gui/widgets/tab_backtest.py). Sin esto, el código exportado de un sistema
    de patrones no cerraría nunca por tiempo."""
    n = int(setup.get('salida_n_velas', 0))
    if n == 0 and setup.get('plantilla') == 'Patrones de velas':
        n = int(params.get('lag_salida', 5))
    return n


def plan_gestion(setup, params):
    """Todo lo que no es señal: riesgo, stop, objetivos, escalonado y
    parciales. Es lo que el emisor traduce a llamadas de su runtime."""
    entrada = setup.get('entrada') or {}
    mecanismos = {}
    for clave in MECANISMOS_SALIDA:
        ir_mec = _ir_mecanismo(setup, clave)
        if ir_mec is not None:
            mecanismos[clave] = ir_mec
    return {
        'riesgo_pct': float(setup.get('riesgo_pct', 0.01)),
        'stop_atr': float(setup.get('stop_atr', 0.0)),
        'tp_r': float(setup.get('tp_r', 0.0)),
        'be_atr': float(setup.get('be_atr', 0.0)),
        'be_unidad': 'r' if setup.get('be_unidad') == 'r' else 'atr',
        'trailing_atr': float(setup.get('trailing_atr', 0.0)),
        'salida_n_velas': _salida_n_velas(setup, params),
        'periodo_atr': PERIODO_ATR_DEFECTO,
        'tramos': [_ir_tramo_entrada(t, k)
                   for k, t in enumerate(setup.get('tramos') or [])],
        'parciales': [_ir_etapa_salida(e, k)
                      for k, e in enumerate(setup.get('parciales') or [])],
        'mecanismos': mecanismos,
        'entrada': {
            'tipo': entrada.get('tipo', 'mercado'),
            'nivel_fib': float(entrada.get('nivel_fib', 0.618)),
            'zigzag_desviacion': float(entrada.get('zigzag_desviacion', 5.0)),
            'zigzag_piernas': int(entrada.get('zigzag_piernas', 10)),
            'vigencia_velas': int(setup.get('limite_vigencia_velas', 0)),
            'cancelar_avance_r': float(setup.get('limite_cancelar_avance_r', 0.0)),
        },
    }


# ══════════════ recorrido de nodos ══════════════

def recorrer_series(nodo, vistas=None):
    """Todas las series que aparecen en un árbol de nodos, sin repetir.
    La usan los emisores para declarar los handles/variables de indicador una
    sola vez, y fidelidad.py para saber qué usa de verdad el sistema."""
    if vistas is None:
        vistas = []
    if nodo is None:
        return vistas
    if 'tipo' in nodo:
        if nodo not in vistas:
            vistas.append(nodo)
        return vistas
    op = nodo.get('op')
    if op in OPS_LOGICOS:
        for parte in nodo['partes']:
            recorrer_series(parte, vistas)
    elif op == 'giro_sar':
        recorrer_series(nodo['sar'], vistas)
    elif op == 'patron':
        pass
    else:
        recorrer_series(nodo['izq'], vistas)
        recorrer_series(nodo['der'], vistas)
    return vistas


def patrones_usados(nodo, vistos=None):
    """Nombres de patrón de vela que aparecen en el árbol, en orden."""
    if vistos is None:
        vistos = []
    if nodo is None:
        return vistos
    if 'tipo' in nodo:
        return vistos
    op = nodo.get('op')
    if op == 'patron':
        if nodo['nombre'] not in vistos:
            vistos.append(nodo['nombre'])
    elif op in OPS_LOGICOS:
        for parte in nodo['partes']:
            patrones_usados(parte, vistos)
    elif op == 'giro_sar':
        pass
    return vistos


# ══════════════ API pública ══════════════

def ir_setup(setup, indice=0):
    """IR completo de un setup: identidad, señales, filtros y gestión.

    `indice` es la posición del setup en el sistema; se usa para nombrar el
    archivo generado y para derivar un identificador estable (magic number en
    MetaTrader), donde cada setup va a un archivo propio."""
    plantilla = setup['plantilla']
    if plantilla not in ESTRATEGIAS:
        raise ValueError(f"Plantilla desconocida: {plantilla}")
    p = params_por_defecto(plantilla)
    p.update(setup.get('params') or {})

    senales = senales_plantilla(plantilla, p)
    filtros = filtros_setup(setup.get('filtros'))
    gestion = plan_gestion(setup, p)

    # el modo «edge» prueba la señal desnuda: el editor ya deja los campos a 0,
    # pero se vuelve a forzar aquí para que un favorito antiguo guardado con
    # edge=True no exporte un stop que el backtest no usó
    if setup.get('edge'):
        gestion.update({'stop_atr': 0.0, 'tp_r': 0.0, 'be_atr': 0.0,
                        'trailing_atr': 0.0})

    series = []
    for clave in LADOS:
        recorrer_series(senales[clave], series)
    for eje in ('regimen', 'volatilidad'):
        if filtros[eje]:
            recorrer_series(filtros[eje]['condicion'], series)
    for clave in ('condiciones_entrada', 'condiciones_salida'):
        for c in filtros[clave]:
            recorrer_series(c['condicion'], series)

    patrones = []
    for clave in LADOS:
        patrones_usados(senales[clave], patrones)

    return {
        'indice': indice,
        'nombre': setup.get('nombre') or plantilla,
        'plantilla': plantilla,
        'params': p,
        'edge': bool(setup.get('edge')),
        'senales': senales,
        'filtros': filtros,
        'gestion': gestion,
        'series': series,
        'patrones': patrones,
    }


def ir_sistema(setups, config_global=None):
    """IR de un sistema completo: cuenta global + un bloque por setup.

    No se fusionan los setups: cada uno se exporta a su propio archivo (el
    arbitraje por vela del motor, que da la vela al primer setup que la
    reclama, no tiene equivalente al cargar estrategias independientes en una
    plataforma; el README generado lo advierte)."""
    cfg = config_global or {}
    return {
        'cuenta': {
            'capital_inicial': float(cfg.get('capital_inicial', 10000.0)),
            'comision_pct': float(cfg.get('comision_pct', 0.0)),
            'slippage_pct': float(cfg.get('slippage_pct', 0.0)),
        },
        'setups': [ir_setup(s, k) for k, s in enumerate(setups)],
    }
