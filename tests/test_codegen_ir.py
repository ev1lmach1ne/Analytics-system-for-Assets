"""
tests/test_codegen_ir.py
El IR es lo único que ven los emisores: si pierde un parámetro por el camino,
el código exportado opera distinto al backtest y no hay forma de notarlo
leyendo el .mq5 o el .pine. Estos tests fijan justo eso.
"""
from core.codegen import ir
from core.strategies import (
    ESTRATEGIAS, PERIODO_ATR_DEFECTO, params_por_defecto, filas_plantilla,
    UMBRAL_ER_TENDENCIA, UMBRAL_ER_RUIDO,
    UMBRAL_HURST_TENDENCIA, UMBRAL_HURST_REVERSION,
    _filtros_por_defecto,
)

# plantillas en las que filas_plantilla SÍ devuelve celdas (izq/op/der) y por
# tanto se puede contrastar contra el IR. El resto degrada a _fila_texto.
PLANTILLAS_ESTRUCTURALES = ('Cruce de medias', 'Bollinger + ATR', 'RSI')


def _setup(plantilla, **extra):
    s = {'nombre': f'S {plantilla}', 'plantilla': plantilla}
    s.update(extra)
    return s


def _firma(nodo):
    """(op, tipo_izq, tipo_der) de una comparación, para contrastar el IR
    contra una fila de la GUI sin depender del resto de parámetros."""
    return (nodo['op'], nodo['izq']['tipo'], nodo['der']['tipo'])


# ══════════════ cobertura ══════════════

def test_todas_las_plantillas_producen_ir():
    """Una plantilla sin constructor de IR reventaría en el momento de
    exportar, con el backtest ya corrido y el usuario esperando el archivo."""
    for plantilla in ESTRATEGIAS:
        senales = ir.senales_plantilla(plantilla)
        assert set(senales) == set(ir.LADOS), plantilla


def test_toda_plantilla_salvo_custom_vacia_opera_algun_lado():
    """Custom (reglas) sin reglas no dispara nunca — igual que _gen_custom.
    Las demás tienen que ofrecer al menos una entrada con sus defaults."""
    for plantilla in ESTRATEGIAS:
        senales = ir.senales_plantilla(plantilla)
        activos = [k for k in ir.LADOS if senales[k] is not None]
        if plantilla == 'Custom (reglas)':
            assert activos == []
        else:
            assert activos, plantilla


def test_ir_setup_funciona_con_todas_las_plantillas():
    for plantilla in ESTRATEGIAS:
        bloque = ir.ir_setup(_setup(plantilla), indice=0)
        assert bloque['plantilla'] == plantilla
        assert bloque['gestion']['periodo_atr'] == PERIODO_ATR_DEFECTO


# ══════════════ lo que filas_plantilla no puede representar ══════════════

def test_el_ir_conserva_el_suavizado_de_kama_que_la_fila_de_la_gui_pierde():
    """La fila de la GUI describe KAMA con las constantes del editor de
    reglas (rápido=2, lento=30) porque su tabla solo tiene un «Periodo». Si el
    IR se construyera sobre ella, un KAMA con otro suavizado se exportaría
    como un indicador distinto — ver _NOTA_KAMA en core/strategies.py."""
    params = {'periodo_er': 10, 'rapido': 5, 'lento': 88}
    nodo = ir.senales_plantilla('KAMA', params)['entradas_long']
    kama = nodo['der']
    assert kama['tipo'] == 'KAMA'
    assert (kama['rapido'], kama['lento']) == (5, 88)

    # la fila de la GUI, en cambio, no lleva ese dato encima
    fila = filas_plantilla('KAMA', params)[0]
    assert 'rapido' not in fila['der'] and 'lento' not in fila['der']
    assert fila['nota'], "la propia fila avisa de que no puede reflejarlo"


def test_el_ir_conserva_la_fuente_del_canal_de_donchian():
    """Con fuente='close' la plantilla forma el canal con CIERRES por los dos
    lados; la fila de la GUI siempre lo describe con máximos/mínimos
    (_NOTA_DONCHIAN). Exportar desde la fila daría rupturas distintas."""
    params = {'fuente': 'close', 'periodo': 20}
    senales = ir.senales_plantilla('Breakout de canal (Donchian)', params)
    entrada = senales['entradas_long']
    assert entrada['izq']['tipo'] == 'close'
    assert entrada['der']['fuente'] == 'close'

    params_hl = {'fuente': 'high/low', 'periodo': 20}
    entrada_hl = ir.senales_plantilla(
        'Breakout de canal (Donchian)', params_hl)['entradas_long']
    assert entrada_hl['izq']['tipo'] == 'high'
    assert entrada_hl['der']['fuente'] == 'high/low'


def test_el_ir_estructura_las_plantillas_que_la_gui_solo_describe_con_texto():
    """Stochastic, Williams %R, CCI, SAR y patrones degradan a _fila_texto en
    la GUI. Un emisor no puede traducir una frase, así que el IR tiene que
    darles estructura."""
    for plantilla in ('Stochastic (%K/%D)', 'Williams %R', 'CCI',
                      'Parabolic SAR', 'Patrones de velas'):
        assert all(f['izq'] is None for f in filas_plantilla(plantilla)), plantilla
        nodo = ir.senales_plantilla(plantilla)['entradas_long']
        assert nodo is not None and 'op' in nodo, plantilla


def test_el_stochastic_exige_cruce_y_zona_extrema():
    """La entrada es híbrida (cruce de %K sobre %D CONFIRMADO dentro de la
    zona); emitir solo el cruce dispararía muchas más entradas."""
    nodo = ir.senales_plantilla('Stochastic (%K/%D)')['entradas_long']
    assert nodo['op'] == 'Y'
    ops = sorted(p['op'] for p in nodo['partes'])
    assert ops == ['<', 'cruza arriba']


def test_el_sar_entra_por_giro_no_por_comparacion_de_precio():
    """El giro del SAR depende del estado acumulado del AF, no de comparar dos
    series: el IR lo marca con su propio operador para que el emisor sepa que
    necesita la máquina de estados, no un cruce."""
    nodo = ir.senales_plantilla('Parabolic SAR')['entradas_long']
    assert nodo['op'] == 'giro_sar' and nodo['sentido'] == +1
    assert nodo['sar']['tipo'] == 'SAR'


def test_los_patrones_se_reparten_por_su_sesgo():
    """La dirección de un patrón la decide su sesgo, no el setup. Un patrón
    bajista no puede acabar en el lado largo del código exportado."""
    senales = ir.senales_plantilla(
        'Patrones de velas', {'patrones': ['Martillo', 'Estrella Fugaz']})
    assert senales['entradas_long']['nombre'] == 'Martillo'
    assert senales['entradas_short']['nombre'] == 'Estrella Fugaz'


def test_un_patron_sin_sesgo_definido_entra_por_los_dos_lados():
    """Sesgo 0 = giro que depende del contexto: detectar_patrones resuelve la
    dirección vela a vela, así que el patrón tiene que ofrecerse a ambos."""
    senales = ir.senales_plantilla('Patrones de velas', {'patrones': ['Doji']})
    assert senales['entradas_long'] is not None
    assert senales['entradas_short'] is not None


def test_las_reglas_custom_son_or_de_reglas_y_and_de_condiciones():
    reglas = {'entradas_long': [
        {'condiciones': [
            {'izq': {'tipo': 'close'}, 'op': '>',
             'der': {'tipo': 'SMA', 'periodo': 50}},
            {'izq': {'tipo': 'RSI', 'periodo': 14}, 'op': '<',
             'der': {'tipo': 'valor', 'valor': 30.0}}]},
        {'condiciones': [
            {'izq': {'tipo': 'close'}, 'op': 'cruza arriba',
             'der': {'tipo': 'EMA', 'periodo': 20}}]}]}
    nodo = ir.senales_plantilla('Custom (reglas)', {'reglas': reglas})['entradas_long']
    assert nodo['op'] == 'O'
    assert nodo['partes'][0]['op'] == 'Y'
    assert len(nodo['partes'][0]['partes']) == 2
    assert nodo['partes'][1]['op'] == 'cruza arriba'


# ══════════════ guardia anti-divergencia con la GUI ══════════════

def test_el_ir_coincide_con_filas_plantilla_donde_ambas_son_estructurales():
    """Donde la tabla de la GUI SÍ sabe representar la plantilla, el IR tiene
    que decir lo mismo. Es la red que impide que el código exportado y lo que
    el usuario ve en pantalla se separen sin que nadie se entere."""
    for plantilla in PLANTILLAS_ESTRUCTURALES:
        for salida in (False, True):
            filas = filas_plantilla(plantilla, salida=salida)
            senales = ir.senales_plantilla(plantilla)
            for fila in filas:
                clave = ('salidas_' if salida else 'entradas_') + fila['direccion']
                nodo = senales[clave]
                assert nodo is not None, (plantilla, clave)
                assert _firma(nodo) == _firma(fila), (plantilla, clave)


# ══════════════ dirección ══════════════

def test_el_filtro_de_direccion_apaga_el_lado_entero():
    """Un lado apagado es None (no hay señal), no una condición que nunca se
    cumple: el emisor no debe generar código muerto para él."""
    senales = ir.senales_plantilla('RSI', {'direccion': 'Long'})
    assert senales['entradas_long'] is not None
    assert senales['salidas_long'] is not None
    assert senales['entradas_short'] is None
    assert senales['salidas_short'] is None


# ══════════════ filtros ══════════════

def test_los_umbrales_de_regimen_viajan_absolutos():
    """El Backtester impone umbrales fijos (no la media±σ del activo), así que
    se traducen literales y no hay nada que calibrar en destino."""
    casos = {'er_tendencia': ('ER', '>', UMBRAL_ER_TENDENCIA),
             'er_rango': ('ER', '<', UMBRAL_ER_RUIDO),
             'hurst_tendencia': ('HURST', '>', UMBRAL_HURST_TENDENCIA),
             'hurst_reversion': ('HURST', '<', UMBRAL_HURST_REVERSION)}
    for metodo, (tipo, op, umbral) in casos.items():
        f = _filtros_por_defecto()
        f['regimen'] = {'metodo': metodo, 'periodo': 400}
        reg = ir.filtros_setup(f)['regimen']
        assert (reg['serie']['tipo'], reg['op'], reg['umbral']) == (tipo, op, umbral)


def test_el_regimen_er_declara_que_el_warmup_cuenta_como_ruido():
    """preparar_contexto rellena el ER en NaN con 0.0, que queda por debajo
    del umbral de ruido: las velas de calentamiento SÍ pasan el filtro
    'er_rango'. Un emisor que las descartara haría menos trades."""
    f = _filtros_por_defecto()
    f['regimen'] = {'metodo': 'er_rango', 'periodo': 10}
    assert ir.filtros_setup(f)['regimen']['nan_como'] == 0.0

    f['regimen'] = {'metodo': 'hurst_reversion', 'periodo': 400}
    assert ir.filtros_setup(f)['regimen']['nan_como'] == 0.5


def test_la_volatilidad_separa_la_ventana_de_comparacion_del_periodo_del_atr():
    """El indicador se calcula siempre con PERIODO_ATR_DEFECTO velas y la
    ventana es solo el histórico contra el que se compara. Si el emisor
    igualara ambos, un tramo estable saldría siempre en el percentil medio."""
    f = _filtros_por_defecto()
    f['volatilidad'] = {'metodo': 'atr_percentil_alto', 'periodo': 250,
                        'percentil': 70.0}
    vol = ir.filtros_setup(f)['volatilidad']
    assert vol['serie']['tipo'] == 'PCT_ATR'
    assert vol['serie']['ventana'] == 250
    assert vol['serie']['periodo_base'] == PERIODO_ATR_DEFECTO
    assert vol['lado'] == 'alto'


def test_la_sesion_distingue_husos_propios_de_horas_utc():
    """Es la diferencia entre traducirla fiel o solo aproximarla: 'ny' lleva
    huso IANA y se mueve con el horario de verano; 'overnight' es UTC fijo."""
    f = _filtros_por_defecto()
    f['sesion'] = {'tipo': 'ny', 'hora_inicio': 0, 'hora_fin': 0}
    assert ir.filtros_setup(f)['sesion']['tz'] == 'America/New_York'

    f['sesion'] = {'tipo': 'overnight', 'hora_inicio': 0, 'hora_fin': 0}
    ses = ir.filtros_setup(f)['sesion']
    assert ses['tz'] is None and (ses['hora_inicio'], ses['hora_fin']) == (1, 9)


def test_las_condiciones_de_filtro_solo_entran_en_el_lado_al_que_apuntan():
    cond = {'izq': {'tipo': 'close'}, 'op': '>',
            'der': {'tipo': 'SMA', 'periodo': 200}, 'direccion': 'long'}
    f = _filtros_por_defecto()
    f['condiciones_entrada'] = [cond]
    conds = ir.filtros_setup(f)['condiciones_entrada']
    assert ir.condiciones_de_lado(conds, 'long') is not None
    assert ir.condiciones_de_lado(conds, 'short') is None


def test_los_filtros_inactivos_quedan_en_none():
    """Así el emisor pregunta `if ir['regimen']` y no tiene que reinterpretar
    la convención de 'ninguno'."""
    filtros = ir.filtros_setup(_filtros_por_defecto())
    for eje in ('regimen', 'volatilidad', 'sesion', 'noticias'):
        assert filtros[eje] is None, eje
    assert filtros['dias_semana'] is None


# ══════════════ plan de gestión ══════════════

def test_patrones_de_velas_hereda_la_salida_por_tiempo_del_lag():
    """El hilo de backtest deriva salida_n_velas del parámetro 'lag_salida'
    cuando el setup no fija una propia. Sin esa misma regla aquí, el código
    exportado de un sistema de patrones no cerraría nunca por tiempo."""
    bloque = ir.ir_setup(_setup('Patrones de velas',
                                params={'patrones': ['Martillo'],
                                        'lag_salida': 7}))
    assert bloque['gestion']['salida_n_velas'] == 7


def test_una_salida_por_tiempo_explicita_gana_al_lag():
    bloque = ir.ir_setup(_setup('Patrones de velas',
                                params={'patrones': ['Martillo'],
                                        'lag_salida': 7},
                                salida_n_velas=3))
    assert bloque['gestion']['salida_n_velas'] == 3


def test_el_modo_edge_anula_stop_y_objetivos():
    """Edge prueba la señal desnuda. Se fuerza en el IR para que un favorito
    antiguo guardado con edge=True no exporte un stop que el backtest no usó."""
    bloque = ir.ir_setup(_setup('RSI', edge=True, stop_atr=2.0, tp_r=3.0,
                                be_atr=1.0, trailing_atr=1.5))
    gestion = bloque['gestion']
    assert (gestion['stop_atr'], gestion['tp_r'],
            gestion['be_atr'], gestion['trailing_atr']) == (0.0, 0.0, 0.0, 0.0)


def test_los_mecanismos_al_cien_por_cien_no_generan_bloque():
    """Un mecanismo que cierra el 100% sin condiciones equivale a no
    configurarlo: emitir código para él sería ruido."""
    bloque = ir.ir_setup(_setup('RSI', salida_stop={'pct': 100.0,
                                                    'condiciones': []}))
    assert bloque['gestion']['mecanismos'] == {}

    bloque = ir.ir_setup(_setup('RSI', salida_stop={'pct': 50.0,
                                                    'condiciones': []}))
    assert bloque['gestion']['mecanismos']['salida_stop']['pct'] == 50.0


def test_los_tramos_y_parciales_conservan_su_orden():
    """El motor los recorre en secuencia: un emisor que los desordenara
    cambiaría qué porcentaje se cierra primero."""
    setup = _setup(
        'RSI',
        tramos=[{'pct': 50.0, 'trigger': 'senal'},
                {'pct': 50.0, 'trigger': 'retroceso', 'val': 1.0}],
        parciales=[{'pct': 40.0, 'trigger': 'r', 'r': 1.5},
                   {'pct': 60.0, 'trigger': 'senal'}])
    gestion = ir.ir_setup(setup)['gestion']
    assert [t['orden'] for t in gestion['tramos']] == [0, 1]
    assert gestion['tramos'][1]['trigger'] == 'retroceso'
    assert [p['orden'] for p in gestion['parciales']] == [0, 1]
    assert gestion['parciales'][0]['r'] == 1.5


def test_la_gestion_de_una_etapa_se_traduce_a_nombre_legible():
    setup = _setup('RSI', parciales=[{'pct': 50.0, 'trigger': 'r', 'r': 1.0,
                                      'gestion': {'tipo': 1, 'val': 0.5}}])
    gestion = ir.ir_setup(setup)['gestion']['parciales'][0]['gestion']
    assert gestion['nombre'] == 'break_even' and gestion['val'] == 0.5


# ══════════════ recorrido de nodos ══════════════

def test_recorrer_series_reune_los_indicadores_sin_repetir():
    """Los emisores declaran un handle por indicador: si la lista repitiera,
    el código generado crearía el mismo indicador dos veces."""
    bloque = ir.ir_setup(_setup('Cruce de medias',
                                params={'tipo': 'EMA', 'rapida': 20,
                                        'lenta': 50}))
    tipos = [(s['tipo'], s.get('periodo')) for s in bloque['series']]
    assert sorted(tipos) == [('EMA', 20), ('EMA', 50)]


def test_las_series_del_setup_incluyen_las_de_los_filtros():
    f = _filtros_por_defecto()
    f['regimen'] = {'metodo': 'er_tendencia', 'periodo': 10}
    bloque = ir.ir_setup(_setup('RSI', filtros=f))
    assert 'ER' in [s['tipo'] for s in bloque['series']]


def test_ir_sistema_numera_los_setups_y_arrastra_la_cuenta():
    """Cada setup va a su propio archivo, así que su índice tiene que ser
    estable: de él salen el nombre del fichero y el magic number."""
    sistema = ir.ir_sistema(
        [_setup('RSI'), _setup('CCI')],
        {'capital_inicial': 25000.0, 'comision_pct': 0.0003,
         'slippage_pct': 0.0002})
    assert [s['indice'] for s in sistema['setups']] == [0, 1]
    assert sistema['cuenta']['capital_inicial'] == 25000.0
    assert sistema['cuenta']['comision_pct'] == 0.0003
