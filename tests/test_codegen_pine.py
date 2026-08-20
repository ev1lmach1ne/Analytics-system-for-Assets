"""
tests/test_codegen_pine.py
El emisor de Pine Script.

No se puede compilar Pine desde aquí, así que estos tests cubren lo que sí se
puede comprobar sin TradingView: que la estructura del archivo es la correcta,
que nada del IR se queda por el camino y que el emisor ROMPE en vez de emitir
código a medias. La sintaxis real la valida el usuario pegando el script.
"""
import re

import pytest

from core.codegen import fidelidad, ir
from core.codegen.pine import EmisorPine
from core.strategies import ESTRATEGIAS, _filtros_por_defecto

META = {'sistema': 'sistema de prueba', 'activo': 'ZCMAIZ', 'tf': '1d'}

# plantillas cuya señal el núcleo todavía no emite: se bloquean antes de
# llegar al emisor (ver fidelidad.bloquea_setup)
PLANTILLAS_EMITIBLES = [p for p in ESTRATEGIAS
                        if p not in fidelidad.PLANTILLAS_SIN_EMISOR]


def _generar(plantilla='RSI', **extra):
    setup = {'nombre': 'setup de prueba', 'plantilla': plantilla,
             'riesgo_pct': 0.01, 'stop_atr': 2.0,
             'filtros': _filtros_por_defecto()}
    setup.update(extra)
    sistema = ir.ir_sistema([setup], {'capital_inicial': 10000.0,
                                      'comision_pct': 0.0003,
                                      'slippage_pct': 0.0002})
    avisos = fidelidad.analizar(sistema, 'tradingview')
    return EmisorPine().construir(sistema['setups'][0], sistema, avisos, META)


def _sin_texto(codigo):
    """Código sin comentarios ni literales de cadena. Hace falta para contar
    paréntesis: los textos de los inputs llevan paréntesis dentro
    («Take-profit (R)») y desbalancearían la cuenta."""
    sin_cadenas = re.sub(r'"(?:[^"\\]|\\.)*"', '""', codigo)
    return "\n".join(linea.split('//')[0] for linea in sin_cadenas.splitlines())


# ══════════════ estructura del archivo ══════════════

def test_la_directiva_de_version_es_la_primera_linea():
    """Pine exige //@version en la línea 1; un comentario delante hace que el
    script se compile como v1 y falle por todas partes."""
    assert _generar().splitlines()[0] == '//@version=6'


@pytest.mark.parametrize('plantilla', PLANTILLAS_EMITIBLES)
def test_los_parentesis_quedan_balanceados(plantilla):
    codigo = _sin_texto(_generar(plantilla))
    for abre, cierra in (('(', ')'), ('[', ']')):
        assert codigo.count(abre) == codigo.count(cierra), (plantilla, abre)


@pytest.mark.parametrize('plantilla', PLANTILLAS_EMITIBLES)
def test_no_quedan_marcadores_sin_sustituir(plantilla):
    codigo = _generar(plantilla)
    assert '__' not in codigo, plantilla
    assert 'None' not in codigo, plantilla
    assert '{' not in codigo and '}' not in codigo, plantilla


@pytest.mark.parametrize('plantilla', PLANTILLAS_EMITIBLES)
def test_el_archivo_declara_la_estrategia_y_su_cuenta(plantilla):
    codigo = _generar(plantilla)
    assert 'strategy(' in codigo
    assert 'initial_capital=10000.0' in codigo
    assert 'commission_value=0.03' in codigo


def test_la_orden_se_rellena_al_open_de_la_vela_siguiente():
    """Es la semántica del motor: decide en la vela t y ejecuta al open de
    t+1. Con process_orders_on_close=true el script operaría al cierre de la
    misma vela de la señal y daría otros resultados."""
    assert 'process_orders_on_close=false' in _generar()


# ══════════════ nada se pierde por el camino ══════════════

@pytest.mark.parametrize('plantilla', PLANTILLAS_EMITIBLES)
def test_toda_serie_del_ir_aparece_declarada(plantilla):
    """La garantía a nivel de emisor: si una serie del IR no deja rastro en el
    código, el script opera con menos información que el backtest."""
    setup = {'nombre': 'x', 'plantilla': plantilla, 'stop_atr': 2.0}
    bloque = ir.ir_setup(setup)
    codigo = _generar(plantilla)
    for nombre, _nodo in EmisorPine().series_declarables(bloque):
        assert nombre in codigo, (plantilla, nombre)


@pytest.mark.parametrize('plantilla', PLANTILLAS_EMITIBLES)
def test_todo_parametro_usado_sale_como_input_editable(plantilla):
    """Un parámetro que se emitiera como constante dejaría el script sin
    forma de ajustarse desde TradingView."""
    setup = {'nombre': 'x', 'plantilla': plantilla, 'stop_atr': 2.0}
    bloque = ir.ir_setup(setup)
    codigo = _generar(plantilla)
    for clave in EmisorPine().params_usados(bloque):
        assert f"p_{clave} = input." in codigo, (plantilla, clave)


def test_los_valores_del_backtest_son_los_valores_por_defecto():
    codigo = _generar('RSI', params={'periodo': 9, 'sobrecompra': 72.4})
    assert 'p_periodo = input.int(9,' in codigo
    assert 'p_sobrecompra = input.float(72.4,' in codigo


def test_las_condiciones_usan_los_inputs_no_las_constantes():
    codigo = _generar('RSI', params={'periodo': 9, 'sobreventa': 30.0})
    assert 'entradaLong = rsi_9 < p_sobreventa' in codigo
    assert 'rsi_9 = ta.rsi(close, p_periodo)' in codigo


def test_el_umbral_fijo_del_oscilador_no_se_vuelve_input():
    """El 50 del RSI es la línea media del indicador, no un ajuste del
    usuario: emitirlo como input prometería una configuración que el
    backtest no tenía."""
    codigo = _generar('RSI', params={'periodo': 9})
    assert 'salidaLong = rsi_9 > 50.0' in codigo
    assert 'p_50' not in codigo


# ══════════════ el emisor rompe en vez de callarse ══════════════

def test_un_nodo_desconocido_lanza():
    """La regla dura: nunca devolver cadena vacía. Un nodo que se pierde en
    silencio da un robot que opera de menos sin que se note leyendo el
    archivo."""
    with pytest.raises(ValueError, match='sin traducción'):
        EmisorPine().expr({'op': 'operador_inventado', 'partes': []})


def test_traducir_un_lado_sin_senal_lanza():
    """Un lado apagado por el filtro de dirección es None y hay que
    comprobarlo antes; llegar al emisor con None significa que se iba a
    emitir una condición vacía."""
    with pytest.raises(ValueError, match='None'):
        EmisorPine().expr(None)


def test_una_serie_sin_traduccion_lanza():
    with pytest.raises(ValueError, match='sin traducción'):
        EmisorPine().declarar_serie('x', {'tipo': 'INVENTADO', 'periodo': 5})


def test_el_hurst_lanza_con_un_mensaje_que_explica_por_que():
    """No debería llegar (fidelidad lo declara omitido en Pine), pero si
    llega tiene que decir por qué en vez de generar un script mudo."""
    with pytest.raises(ValueError, match='Hurst'):
        EmisorPine().declarar_serie('h', {'tipo': 'HURST', 'periodo': 400})


# ══════════════ dirección ══════════════

def test_un_lado_apagado_se_emite_como_false():
    """El filtro de dirección deja el lado sin señal: se emite `false`
    explícito para que el script no opere ese lado y se lea que es
    intencionado."""
    codigo = _generar('RSI', params={'direccion': 'Long'})
    assert 'entradaShort = false' in codigo
    assert 'salidaShort = false' in codigo
    assert 'entradaLong = rsi_' in codigo


# ══════════════ filtros ══════════════

def test_los_filtros_solo_condicionan_las_entradas():
    """Las salidas nunca se filtran: si el régimen cambiara a mitad de
    operación, una posición abierta se quedaría sin forma de cerrarse."""
    codigo = _generar()
    assert re.search(r'entradaLong and filtroLong', codigo)
    assert 'salidaLong and permiteSalidaLong' in codigo


def test_el_filtro_de_regimen_por_er_se_emite():
    f = _filtros_por_defecto()
    f['regimen'] = {'metodo': 'er_tendencia', 'periodo': 10}
    codigo = _generar('RSI', filtros=f)
    assert 'zcsEr(' in codigo
    assert 'filtroRegimen = er_10 > 0.5' in codigo


def test_el_filtro_de_sesion_usa_el_huso_horario_nativo():
    """Es lo que Pine sí sabe hacer y MetaTrader no: la sesión se sigue con
    su huso IANA, así que el horario de verano se ajusta solo."""
    f = _filtros_por_defecto()
    f['sesion'] = {'tipo': 'ny', 'hora_inicio': 0, 'hora_fin': 0}
    codigo = _generar('RSI', filtros=f)
    assert '"America/New_York"' in codigo
    assert '"0800-1700"' in codigo


def test_el_filtro_de_dias_se_emite_con_los_dias_elegidos():
    f = _filtros_por_defecto()
    f['dias_semana'] = [0, 4]
    codigo = _generar('RSI', filtros=f)
    assert 'dayofweek.monday' in codigo and 'dayofweek.friday' in codigo
    assert 'dayofweek.wednesday' not in codigo


# ══════════════ gestión ══════════════

def test_la_gestion_emite_stop_take_profit_y_sizing():
    codigo = _generar('RSI', tp_r=3.0)
    assert 'zcsUnidadesPorRiesgo(strategy.equity' in codigo
    assert 'strategy.exit(' in codigo
    assert 'p_tp_r = input.float(3.0' in codigo


def test_el_break_even_en_r_usa_la_distancia_y_en_atr_el_atr():
    """be_unidad cambia contra qué se mide el avance: R es la distancia real
    al stop de esa operación, ×ATR es la volatilidad actual."""
    assert 'refBe = distEntrada' in _generar('RSI', be_atr=1.0, be_unidad='r')
    assert 'refBe = atrGestion' in _generar('RSI', be_atr=1.0, be_unidad='atr')


def test_la_entrada_escalonada_emite_tramos():
    """Cada tramo adicional añade posición con strategy.order y un tamaño de
    riesgo × pct/100 sobre la distancia al stop."""
    tramos = [{'pct': 50.0, 'trigger': 'senal', 'val': 0.0, 'condiciones': []},
              {'pct': 50.0, 'trigger': 'retroceso', 'val': 1.0,
               'condiciones': []}]
    codigo = _generar('RSI', tramos=tramos)
    assert 'strategy.order("T1"' in codigo
    assert 'tramoActual' in codigo
    assert 'p_riesgo_pct * 0.5' in codigo
    assert 'distTramo' in codigo
    assert 'low <= precioIn - 1.0 * atrGestion' in codigo


def test_un_solo_tramo_no_genera_escalonado():
    """El tramo implícito al 100% es la entrada normal; no debe meter lógica
    de escalonado que no se configuró."""
    codigo = _generar('RSI')
    assert 'strategy.order("T' not in codigo
    assert 'tramoActual' not in codigo


def test_el_sizing_usa_el_equity_actual_no_el_capital_inicial():
    """El motor arriesga un % del equity vivo: con capital fijo el tamaño no
    crecería ni se reduciría con la cuenta."""
    assert 'strategy.equity' in _generar()


# ══════════════ guardas y avisos ══════════════

def test_el_simbolo_no_bloquea_las_entradas():
    """El nombre del CSV del backtest («zcmaiz») casi nunca coincide con el
    ticker de la plataforma («ZC1!»), así que exigir que casen dejaba el
    script sin operar NUNCA y sin decir por qué. Solo la temporalidad decide.

    Esto pasó de verdad: el primer script generado compilaba pero el probador
    de estrategias salía vacío."""
    codigo = _generar()
    assert 'puedeOperar = p_permitir_otro or tfCorrecto' in codigo
    assert 'if plano and puedeOperar' in codigo
    # simboloCorrecto existe, pero solo para el aviso visual
    assert 'simboloCorrecto' in codigo
    assert 'or activoCorrecto' not in codigo
    assert 'and simboloCorrecto' not in codigo


def test_la_temporalidad_equivocada_dice_que_no_va_a_operar():
    """Si la guarda bloquea, el motivo tiene que estar a la vista: un
    probador vacío sin explicación es indistinguible de una estrategia mala."""
    codigo = _generar()
    assert 'NO SE OPERARA' in codigo
    assert 'table.cell(' in codigo


def test_el_archivo_lleva_el_activo_y_la_temporalidad_del_backtest():
    """Los parámetros están ajustados a ese activo y esa temporalidad; en
    otro sitio no significan lo mismo."""
    codigo = _generar()
    assert 'ZCMAIZ' in codigo
    assert 'tfEsperado = "D"' in codigo
    assert 'p_permitir_otro = input.bool(false' in codigo


def test_la_cabecera_lleva_las_notas_de_fidelidad():
    assert 'NOTAS DE FIDELIDAD' in _generar()


def test_un_sistema_con_omisiones_avisa_tambien_en_ejecucion():
    """El aviso en la cabecera se pierde si alguien copia solo el cuerpo del
    script; el log.warning sobrevive."""
    f = _filtros_por_defecto()
    f['noticias'] = dict(f['noticias'], activo=True)
    codigo = _generar('RSI', filtros=f)
    assert 'log.warning(' in codigo
    assert 'AVISO DE FIDELIDAD' in codigo


def test_un_sistema_sin_omisiones_no_mete_aviso_en_ejecucion():
    """Solo las omisiones justifican gritar en el log; lo aproximado ya está
    en la cabecera."""
    assert 'log.warning(' not in _generar()


def test_el_filtro_omitido_no_deja_ningun_stub():
    """Un «// TODO: filtro de noticias» que no filtra nada se lee como si
    estuviera resuelto: es peor que su ausencia declarada."""
    f = _filtros_por_defecto()
    f['noticias'] = dict(f['noticias'], activo=True)
    codigo = _generar('RSI', filtros=f)
    assert 'TODO' not in codigo
    assert 'filtroNoticias' not in codigo


# ══════════════ runtime ══════════════

def test_el_runtime_va_insertado_en_el_script():
    """Pine no tiene includes para un script suelto: si el bloque no viaja
    dentro, el script no compila."""
    codigo = _generar()
    for funcion in ('zcsAtr(', 'zcsCruzaArriba(', 'zcsUnidadesPorRiesgo('):
        assert funcion in codigo, funcion


def test_el_atr_del_runtime_es_suavizado_de_wilder():
    """El motor y el runtime generado usan ta.atr (suavizado de Wilder del TR),
    igual que las plataformas de referencia. Antes era una media simple."""
    codigo = _generar()
    assert 'ta.atr(periodo)' in codigo
    assert 'zcsAtr(int periodo)' in codigo


def test_las_bandas_de_bollinger_usan_desviacion_poblacional():
    """pandas .std(ddof=0) divide por n; ta.stdev con biased=true divide por
    n (poblacional), que es la definición estándar de Bollinger."""
    codigo = _generar('Bollinger + ATR')
    assert 'ta.stdev(src, periodo, true)' in codigo
