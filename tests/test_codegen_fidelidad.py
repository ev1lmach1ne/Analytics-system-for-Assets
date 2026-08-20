"""
tests/test_codegen_fidelidad.py
La garantía de que nada se pierde en silencio al exportar.

Un filtro de sesión o de noticias que desaparece en la traducción convierte el
código exportado en un sistema distinto del backtesteado, y mirando el archivo
generado no se nota: lo que falta no se ve. Estos tests fijan que cada cosa
que una plataforma no sabe hacer produzca un aviso, y que un sistema limpio no
produzca ruido.
"""
import pytest

from core.codegen import fidelidad, ir
from core.strategies import _filtros_por_defecto

PLATAFORMAS = tuple(fidelidad.CAPACIDADES)


def _setup(filtros=None, **extra):
    s = {'nombre': 'RSI', 'plantilla': 'RSI', 'riesgo_pct': 0.01,
         'stop_atr': 2.0, 'filtros': filtros or _filtros_por_defecto()}
    s.update(extra)
    return s


def _sistema(setup):
    return ir.ir_sistema([setup], {'capital_inicial': 10000.0})


def _con_filtro(**cambios):
    f = _filtros_por_defecto()
    f.update(cambios)
    return _setup(f)


def _claves(avisos):
    return {a['clave'] for a in avisos}


def _nivel_de(avisos, clave):
    for aviso in avisos:
        if aviso['clave'] == clave:
            return aviso['nivel']
    return None


# Un setup que activa cada característica del catálogo. Es lo que hace posible
# el test de cobertura inversa: si alguien añade una capacidad no soportada sin
# registrar aquí cómo se dispara, el test falla en vez de dejarla sin aviso.
_ACTIVADORES = {
    'noticias': lambda: _con_filtro(
        noticias=dict(_filtros_por_defecto()['noticias'], activo=True)),
    'sesion_dst': lambda: _con_filtro(
        sesion={'tipo': 'ny', 'hora_inicio': 0, 'hora_fin': 0}),
    'sesion_utc': lambda: _con_filtro(
        sesion={'tipo': 'overnight', 'hora_inicio': 0, 'hora_fin': 0}),
    'dias_semana': lambda: _con_filtro(dias_semana=[0, 1, 2]),
    'hurst': lambda: _con_filtro(
        regimen={'metodo': 'hurst_tendencia', 'periodo': 400}),
    'tf_superior': lambda: _con_filtro(
        tf_superior={'indicador': 'SMA', 'tf': '1h', 'periodo': 200,
                     'relacion': 'ambos'}),
    'indicadores_editor_pendientes': lambda: {
        'nombre': 'custom obv', 'plantilla': 'Custom (reglas)',
        'params': {'reglas': {'entradas_long': [{'condiciones': [
            {'izq': {'tipo': 'OBV', 'periodo': 14}, 'op': '>',
             'der': {'tipo': 'valor', 'valor': 0.0}}]}]}},
        'filtros': _filtros_por_defecto()},
    'vwap': lambda: _setup(
        plantilla='VWAP',
        params={'anclaje': 'W', 'modo': 'sd', 'k': 2.0,
                'direccion': 'Ambas'}),
    'entrada_limite_fib': lambda: _setup(
        entrada={'tipo': 'limite_fib', 'nivel_fib': 0.618}),
    'relleno_open_siguiente': _setup,
    'redondeo_lotes': _setup,
    'atr_de_la_vela_de_entrada': _setup,
    'stop_atr_dinamico': lambda: _setup(stop_atr_modo='dinamico_promedio'),
    'slippage_en_ticks': _setup,
    'comision': _setup,
    'tramos_escalonados': lambda: _setup(
        tramos=[{'pct': 60.0, 'trigger': 'senal'},
                {'pct': 40.0, 'trigger': 'retroceso', 'val': 1.0}]),
    'salidas_parciales': lambda: _setup(
        parciales=[{'pct': 50.0, 'trigger': 'r', 'r': 1.5},
                   {'pct': 50.0, 'trigger': 'senal'}]),
    'mecanismos_parciales': lambda: _setup(
        salida_stop={'pct': 50.0, 'condiciones': []}),
    'patrones_velas': lambda: {
        'nombre': 'patrones', 'plantilla': 'Patrones de velas',
        'params': {'patrones': ['Martillo'], 'lag_salida': 5},
        'filtros': _filtros_por_defecto()},
    'zigzag': lambda: {
        'nombre': 'custom zz', 'plantilla': 'Custom (reglas)',
        'params': {'reglas': {'entradas_long': [{'condiciones': [
            {'izq': {'tipo': 'close'}, 'op': '>',
             'der': {'tipo': 'ZIGZAG', 'periodo': 10}}]}]}},
        'filtros': _filtros_por_defecto()},
    # cuenta_netting es de SISTEMA, no de setup: hace falta más de uno
    'cuenta_netting': None,
}


def _sistema_multi_setup():
    return ir.ir_sistema([_setup(), _setup()], {'capital_inicial': 10000.0})


# ══════════════ integridad del catálogo ══════════════

def test_toda_capacidad_declarada_existe_en_el_catalogo():
    for plataforma, capacidades in fidelidad.CAPACIDADES.items():
        for clave in capacidades:
            assert clave in fidelidad.CARACTERISTICAS, (plataforma, clave)


def test_toda_capacidad_no_exacta_explica_el_motivo_y_la_consecuencia():
    """Un aviso que solo dice «no soportado» no le sirve de nada a quien tiene
    que decidir si opera con ese código."""
    for plataforma, capacidades in fidelidad.CAPACIDADES.items():
        for clave, nivel in capacidades.items():
            if nivel == fidelidad.NIVEL_EXACTO:
                continue
            cat = fidelidad.CARACTERISTICAS[clave]
            motivo = (cat.get('motivos', {}).get(plataforma)
                      or cat.get('motivo_comun'))
            assert motivo, (plataforma, clave)
            assert cat['consecuencia'].get(nivel), (plataforma, clave, nivel)


def test_lo_pendiente_dice_que_es_del_generador_y_no_de_la_plataforma():
    """Al usuario le importa la diferencia: lo que la plataforma no puede
    hacer no va a cambiar nunca; lo que falta por implementar, sí."""
    for plataforma in PLATAFORMAS:
        for clave, nivel in fidelidad.CAPACIDADES[plataforma].items():
            if nivel == fidelidad.NIVEL_EXACTO:
                continue
            cat = fidelidad.CARACTERISTICAS[clave]
            motivo = (cat.get('motivos', {}).get(plataforma)
                      or cat.get('motivo_comun', ''))
            if fidelidad.es_pendiente(plataforma, clave):
                assert 'todavía no' in motivo, (plataforma, clave, motivo)
            else:
                assert 'todavía no' not in motivo, (plataforma, clave, motivo)


def test_el_hurst_es_exacto_en_mql5_pendiente_en_mql4_e_imposible_en_pine():
    """El Hurst se porta entero a MQL5 (zcsHurst del runtime) y ya no avisa;
    en MQL4 sigue pendiente de implementar; en Pine no cabe en el presupuesto
    de ejecución por barra y no va a caber."""
    assert fidelidad.nivel('mt5', 'hurst') == fidelidad.NIVEL_EXACTO
    assert fidelidad.es_pendiente('mt4', 'hurst')
    assert not fidelidad.es_pendiente('tradingview', 'hurst')


def test_solo_bloquean_el_setup_las_caracteristicas_que_son_la_senal():
    """Perder un filtro degrada el sistema; perder la señal lo deja sin nada
    que hacer, y entonces no se genera archivo. El ZigZag ya no bloquea: se
    porta al runtime (zcsZigzag)."""
    assert fidelidad.bloquea_setup('patrones_velas')
    assert not fidelidad.bloquea_setup('zigzag')
    for clave in ('noticias', 'hurst', 'tramos_escalonados',
                  'salidas_parciales', 'entrada_limite_fib', 'zigzag'):
        assert not fidelidad.bloquea_setup(clave), clave


def test_lo_que_no_esta_en_el_catalogo_se_considera_exacto():
    """Los indicadores portados (ER, KAMA, SAR, percentil rodante) no entran
    aquí a propósito: se reimplementan en el runtime, no se omiten."""
    for clave in ('indicador_er', 'kama', 'sar', 'percentil', 'lo_que_sea'):
        for plataforma in PLATAFORMAS:
            assert fidelidad.nivel(plataforma, clave) == fidelidad.NIVEL_EXACTO


# ══════════════ cobertura inversa ══════════════

@pytest.mark.parametrize('plataforma', PLATAFORMAS)
def test_cada_capacidad_no_exacta_tiene_un_setup_que_la_dispara(plataforma):
    """Cobertura inversa: recorre lo DECLARADO como no soportado y exige que
    exista un sistema que lo active y produzca el aviso, con el nivel exacto
    que dice el catálogo. Impide que una capacidad nueva se añada sin aviso."""
    for clave, nivel in fidelidad.CAPACIDADES[plataforma].items():
        if nivel == fidelidad.NIVEL_EXACTO:
            continue
        assert clave in _ACTIVADORES, (
            f"{clave} está declarada como '{nivel}' en {plataforma} pero "
            f"ningún setup de prueba la activa: el aviso no está cubierto")
        activador = _ACTIVADORES[clave]
        sistema = (_sistema_multi_setup() if activador is None
                   else _sistema(activador()))
        assert _nivel_de(fidelidad.analizar(sistema, plataforma), clave) == nivel, (
            plataforma, clave)


# ══════════════ ausencia de ruido ══════════════

def test_un_sistema_limpio_no_genera_ninguna_omision():
    """Si un aviso ❌ apareciera sin motivo, el usuario aprendería a
    ignorarlos y el mecanismo entero dejaría de servir."""
    for plataforma in PLATAFORMAS:
        avisos = fidelidad.analizar(_sistema(_setup()), plataforma)
        assert not fidelidad.hay_omisiones(avisos), plataforma


def test_un_sistema_limpio_en_pine_solo_avisa_de_lo_inevitable():
    """Pine rellena al open de la vela siguiente y sabe de husos horarios, así
    que de un sistema limpio solo se le escapan dos cosas, ninguna suya: que
    el motor dimensione con el ATR de la propia vela de entrada (un dato que
    en vivo aún no existe) y que TradingView mida el slippage en ticks y no
    en porcentaje."""
    avisos = fidelidad.analizar(_sistema(_setup()), 'tradingview')
    assert _claves(avisos) == {'atr_de_la_vela_de_entrada',
                               'slippage_en_ticks'}
    assert fidelidad.nivel_global(avisos) == fidelidad.NIVEL_APROXIMADO


def test_metatrader_avisa_siempre_de_lo_inevitable_pero_sin_omitir_nada():
    """Los cinco son inevitables y afectan a cualquier sistema: el EA
    reacciona al primer tick de la vela, el tamaño se redondea al paso de
    lote, el ATR de dimensionamiento va una vela por detrás del que usó el
    backtest, y el slippage y la comisión del backtest no se pueden fijar
    desde el código (los pone el mercado/probador)."""
    for plataforma in ('mt4', 'mt5'):
        avisos = fidelidad.analizar(_sistema(_setup()), plataforma)
        assert _claves(avisos) == {'relleno_open_siguiente', 'redondeo_lotes',
                                   'atr_de_la_vela_de_entrada',
                                   'slippage_en_ticks', 'comision'}
        assert fidelidad.nivel_global(avisos) == fidelidad.NIVEL_APROXIMADO


def test_el_aviso_de_netting_solo_sale_con_mas_de_un_setup():
    """Con un solo EA sobre el símbolo no hay nada con lo que fundirse: sacar
    el aviso siempre sería ruido."""
    assert 'cuenta_netting' not in _claves(
        fidelidad.analizar(_sistema(_setup()), 'mt5'))
    avisos = fidelidad.analizar(_sistema_multi_setup(), 'mt5')
    assert _nivel_de(avisos, 'cuenta_netting') == fidelidad.NIVEL_APROXIMADO


def test_el_aviso_de_sistema_no_se_atribuye_a_ningun_setup():
    avisos = fidelidad.analizar(_sistema_multi_setup(), 'mt5')
    netting = [a for a in avisos if a['clave'] == 'cuenta_netting'][0]
    assert netting['setup'] is None
    assert 'Setup' not in fidelidad.texto_aviso(netting)


# ══════════════ lo pendiente del generador ══════════════

def test_el_tramo_unico_al_cien_por_cien_no_avisa_de_escalonado():
    """Todo setup lleva un tramo implícito al 100%: avisar de eso sería
    ruido en cada exportación."""
    setup = _setup(tramos=[{'pct': 100.0, 'trigger': 'senal'}])
    for plataforma in PLATAFORMAS:
        assert 'tramos_escalonados' not in _claves(
            fidelidad.analizar(_sistema(setup), plataforma))


def test_la_entrada_escalonada_ya_no_avisa_porque_se_emite():
    """El generador ya emite los tramos (zcsTramoActual / strategy.order), así
    que la entrada escalonada dejó de ser una omisión."""
    setup = _setup(tramos=[{'pct': 60.0, 'trigger': 'senal'},
                           {'pct': 40.0, 'trigger': 'retroceso', 'val': 1.0}])
    for plataforma in PLATAFORMAS:
        assert fidelidad.nivel(plataforma,
                               'tramos_escalonados') == fidelidad.NIVEL_EXACTO
        assert 'tramos_escalonados' not in _claves(
            fidelidad.analizar(_sistema(setup), plataforma))


def test_la_etapa_de_salida_implicita_no_avisa_de_parciales():
    """etapa_salida_por_defecto (100% con la señal de la plantilla) es cómo se
    representa 'cerrar todo': el motor la trata igual que no tener etapas."""
    setup = _setup(parciales=[{'pct': 100.0, 'trigger': 'senal',
                               'condiciones': []}])
    for plataforma in PLATAFORMAS:
        assert 'salidas_parciales' not in _claves(
            fidelidad.analizar(_sistema(setup), plataforma))


def test_una_etapa_al_cincuenta_por_ciento_si_avisa():
    setup = _setup(parciales=[{'pct': 50.0, 'trigger': 'r', 'r': 1.0}])
    assert _nivel_de(fidelidad.analizar(_sistema(setup), 'mt5'),
                     'salidas_parciales') == fidelidad.NIVEL_OMITIDO


def test_un_setup_de_patrones_no_se_puede_exportar_todavia():
    """Lo que falta es la SEÑAL: generar un archivo daría un robot que no
    abre una sola operación, que es peor que no dar archivo."""
    sistema = _sistema(_ACTIVADORES['patrones_velas']())
    for plataforma in PLATAFORMAS:
        avisos = fidelidad.analizar(sistema, plataforma)
        assert fidelidad.setups_bloqueados(avisos) == [0], plataforma


def test_un_filtro_omitido_no_bloquea_el_setup():
    """Perder el filtro de noticias degrada el sistema pero deja algo que
    generar; no debe impedir la exportación."""
    sistema = _sistema(_ACTIVADORES['noticias']())
    assert fidelidad.setups_bloqueados(
        fidelidad.analizar(sistema, 'tradingview')) == []


def test_solo_se_bloquea_el_setup_culpable():
    limpio = _setup()
    limpio['nombre'] = 'RSI normal'
    sistema = ir.ir_sistema([limpio, _ACTIVADORES['patrones_velas']()], {})
    avisos = fidelidad.analizar(sistema, 'mt5')
    assert fidelidad.setups_bloqueados(avisos) == [1]


# ══════════════ los casos concretos que motivan la función ══════════════

def test_el_filtro_de_noticias_se_omite_en_mt4_y_pine_pero_no_en_mt5():
    """MQL5 tiene calendario nativo; MQL4 y Pine no tienen ninguno."""
    sistema = _sistema(_ACTIVADORES['noticias']())
    assert _nivel_de(fidelidad.analizar(sistema, 'mt4'),
                     'noticias') == fidelidad.NIVEL_OMITIDO
    assert _nivel_de(fidelidad.analizar(sistema, 'tradingview'),
                     'noticias') == fidelidad.NIVEL_OMITIDO
    assert _nivel_de(fidelidad.analizar(sistema, 'mt5'),
                     'noticias') == fidelidad.NIVEL_APROXIMADO


def test_la_sesion_con_huso_propio_es_exacta_en_pine_y_aproximada_en_metatrader():
    """Pine acepta el huso IANA y sigue el horario de verano solo; MetaTrader
    fecha las velas en hora del servidor del bróker."""
    sistema = _sistema(_ACTIVADORES['sesion_dst']())
    assert 'sesion_dst' not in _claves(fidelidad.analizar(sistema, 'tradingview'))
    for plataforma in ('mt4', 'mt5'):
        assert _nivel_de(fidelidad.analizar(sistema, plataforma),
                         'sesion_dst') == fidelidad.NIVEL_APROXIMADO


def test_el_hurst_es_exacto_en_mql5_y_se_omite_en_pine_y_mql4():
    """MQL5 ya lo porta (zcsHurst del runtime) y no avisa; en Pine no cabe en
    el presupuesto de ejecución por barra y se queda en ❌; en MQL4 sigue
    pendiente de implementar."""
    sistema = _sistema(_ACTIVADORES['hurst']())
    assert _nivel_de(fidelidad.analizar(sistema, 'mt5'),
                     'hurst') is None            # exacto: sin aviso
    avisos = fidelidad.analizar(sistema, 'mt4')
    assert _nivel_de(avisos, 'hurst') == fidelidad.NIVEL_OMITIDO
    assert [a for a in avisos if a['clave'] == 'hurst'][0]['pendiente'] is True
    avisos = fidelidad.analizar(sistema, 'tradingview')
    assert _nivel_de(avisos, 'hurst') == fidelidad.NIVEL_OMITIDO
    assert [a for a in avisos if a['clave'] == 'hurst'][0]['pendiente'] is False


def test_el_regimen_por_er_no_se_omite_en_ninguna_plataforma():
    """El ER es |movimiento neto| / |movimiento total| y sus umbrales son
    absolutos: se porta entero a las tres. Si esto empezara a avisar,
    significaría que alguien lo ha sacado de la librería de runtime."""
    sistema = _sistema(_con_filtro(
        regimen={'metodo': 'er_tendencia', 'periodo': 10}))
    for plataforma in PLATAFORMAS:
        assert 'hurst' not in _claves(fidelidad.analizar(sistema, plataforma))
        assert not fidelidad.hay_omisiones(fidelidad.analizar(sistema, plataforma))


# ══════════════ redacción ══════════════

def test_el_aviso_nombra_el_setup_el_filtro_y_la_consecuencia():
    """Es lo que el usuario lee en el diálogo antes de confirmar."""
    sistema = _sistema(_ACTIVADORES['noticias']())
    aviso = [a for a in fidelidad.analizar(sistema, 'tradingview')
             if a['clave'] == 'noticias'][0]
    texto = fidelidad.texto_aviso(aviso)
    assert 'Setup 0' in texto and 'RSI' in texto
    assert 'noticias' in texto.lower()
    assert aviso['motivo'] and aviso['consecuencia']
    assert aviso['consecuencia'] in texto


def test_los_avisos_se_ordenan_con_las_omisiones_primero():
    """Lo que rompe el sistema va antes que lo que solo lo desplaza."""
    f = _filtros_por_defecto()
    f['noticias'] = dict(f['noticias'], activo=True)
    f['sesion'] = {'tipo': 'ny', 'hora_inicio': 0, 'hora_fin': 0}
    avisos = fidelidad.analizar(_sistema(_setup(f)), 'mt4')
    niveles = [a['nivel'] for a in avisos]
    assert niveles[0] == fidelidad.NIVEL_OMITIDO
    assert niveles == sorted(
        niveles, key=lambda n: 0 if n == fidelidad.NIVEL_OMITIDO else 1)


def test_el_aviso_en_ejecucion_solo_recoge_las_omisiones():
    """El Print() del OnInit tiene que caber en una línea de log; lo
    aproximado ya está en la cabecera del archivo."""
    sistema = _sistema(_ACTIVADORES['sesion_dst']())
    assert fidelidad.texto_runtime(fidelidad.analizar(sistema, 'mt5')) == ''

    sistema = _sistema(_ACTIVADORES['noticias']())
    texto = fidelidad.texto_runtime(fidelidad.analizar(sistema, 'mt4'))
    assert 'AVISO DE FIDELIDAD' in texto and 'noticias' in texto.lower()


def test_el_aviso_en_ejecucion_es_ascii():
    """Va dentro de un Print() de MQL, que no lleva bien los acentos ni los
    emojis según la codificación con la que se compile."""
    f = _filtros_por_defecto()
    f['noticias'] = dict(f['noticias'], activo=True)
    f['regimen'] = {'metodo': 'hurst_reversion', 'periodo': 400}
    texto = fidelidad.texto_runtime(
        fidelidad.analizar(_sistema(_setup(f)), 'tradingview'))
    assert texto
    texto.encode('ascii')


def test_el_bloque_de_notas_dice_explicitamente_cuando_no_hay_nada_que_avisar():
    """Una cabecera vacía se leería como que alguien olvidó rellenarla."""
    lineas = fidelidad.bloque_notas([], 'TradingView')
    assert len(lineas) == 1 and 'sin omisiones' in lineas[0]


def test_el_bloque_de_notas_respeta_el_ancho_pedido():
    """Va dentro de un comentario de código: una línea larga rompe el
    formato del archivo generado."""
    sistema = _sistema(_ACTIVADORES['noticias']())
    lineas = fidelidad.bloque_notas(
        fidelidad.analizar(sistema, 'mt4'), 'MetaTrader 4', ancho=76)
    assert lineas and all(len(linea) <= 76 for linea in lineas)


def test_nivel_global_es_el_peor_de_la_lista():
    f = _filtros_por_defecto()
    f['noticias'] = dict(f['noticias'], activo=True)
    f['sesion'] = {'tipo': 'ny', 'hora_inicio': 0, 'hora_fin': 0}
    avisos = fidelidad.analizar(_sistema(_setup(f)), 'mt4')
    assert fidelidad.nivel_global(avisos) == fidelidad.NIVEL_OMITIDO
    assert fidelidad.hay_omisiones(avisos)


# ══════════════ varios setups ══════════════

def test_cada_setup_genera_su_propio_aviso():
    """Con varios setups hay que poder saber CUÁL es el que pierde el filtro:
    cada uno se exporta a un archivo distinto."""
    limpio = _setup()
    limpio['nombre'] = 'sin filtros'
    con_noticias = _ACTIVADORES['noticias']()
    con_noticias['nombre'] = 'con noticias'
    sistema = ir.ir_sistema([limpio, con_noticias], {})
    avisos = [a for a in fidelidad.analizar(sistema, 'tradingview')
              if a['clave'] == 'noticias']
    assert len(avisos) == 1
    assert avisos[0]['indice_setup'] == 1
    assert avisos[0]['setup'] == 'con noticias'
