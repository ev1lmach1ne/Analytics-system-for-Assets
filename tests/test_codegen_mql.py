"""
tests/test_codegen_mql.py
El emisor de MQL5.

MetaEditor no está aquí, así que estos tests cubren lo que se puede comprobar
sin él. El más importante es el de variables sin declarar: MQL es compilado y
una variable inexistente es un error de compilación, no un fallo silencioso —
y es exactamente el fallo que produjo el sufijo de desplazamiento duplicado
durante el desarrollo.
"""
import re

import pytest

from core.codegen import fidelidad, ir
from core.codegen.mql import EmisorMQL5, _magic
from core.strategies import ESTRATEGIAS, _filtros_por_defecto

META = {'sistema': 'sistema de prueba', 'activo': 'ZCMAIZ', 'tf': '1d'}

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
    avisos = fidelidad.analizar(sistema, 'mt5')
    return EmisorMQL5().construir(sistema['setups'][0], sistema, avisos, META)


def _sin_texto(codigo):
    sin_cadenas = re.sub(r'"(?:[^"\\]|\\.)*"', '""', codigo)
    return "\n".join(l.split('//')[0] for l in sin_cadenas.splitlines())


def _declaradas(codigo):
    """Todo identificador que el archivo introduce: variables locales y
    globales, inputs y los parámetros de salida de zcsSar."""
    nombres = set(re.findall(r'^\s*(?:double|int|bool|string|ulong)\s+(\w+)',
                             codigo, re.M))
    nombres |= set(re.findall(r'input\s+\w+\s+(\w+)', codigo))
    nombres |= set(re.findall(r'zcsSar\([^)]*,\s*(\w+_tend_\d)\)', codigo))
    return nombres


# ══════════════ la garantía que sustituye al compilador ══════════════

@pytest.mark.parametrize('plantilla', PLANTILLAS_EMITIBLES)
def test_no_se_usa_ninguna_variable_sin_declarar(plantilla):
    """Toda variable con sufijo de desplazamiento que aparezca en las señales
    o en la gestión tiene que estar declarada antes. Sin este test, un fallo
    de nombrado solo se descubre al abrir MetaEditor."""
    codigo = _generar(plantilla)
    declaradas = _declaradas(codigo)
    cuerpo = codigo[codigo.find('--- senales'):]
    usadas = set(re.findall(r'\b([a-z]\w*_\d)\b', _sin_texto(cuerpo)))
    assert not (usadas - declaradas), (plantilla, sorted(usadas - declaradas))


@pytest.mark.parametrize('plantilla', PLANTILLAS_EMITIBLES)
def test_llaves_y_parentesis_balanceados(plantilla):
    codigo = _sin_texto(_generar(plantilla))
    for abre, cierra in (('{', '}'), ('(', ')')):
        assert codigo.count(abre) == codigo.count(cierra), (plantilla, abre)


@pytest.mark.parametrize('plantilla', PLANTILLAS_EMITIBLES)
def test_no_quedan_marcadores_sin_sustituir(plantilla):
    codigo = _generar(plantilla)
    assert '__' not in codigo and 'None' not in codigo, plantilla


@pytest.mark.parametrize('plantilla', PLANTILLAS_EMITIBLES)
def test_el_archivo_tiene_las_tres_funciones_del_ciclo_de_vida(plantilla):
    codigo = _generar(plantilla)
    for firma in ('int OnInit()', 'void OnDeinit(const int reason)',
                  'void OnTick()'):
        assert firma in codigo, (plantilla, firma)


@pytest.mark.parametrize('plantilla', PLANTILLAS_EMITIBLES)
def test_todo_handle_creado_se_libera(plantilla):
    """Un handle sin IndicatorRelease deja el indicador colgado cada vez que
    se recarga el EA."""
    codigo = _generar(plantilla)
    creados = set(re.findall(r'^\s*(h_\w+) = i\w+\(', codigo, re.M))
    liberados = set(re.findall(r'IndicatorRelease\((h_\w+)\)', codigo))
    assert creados == liberados, plantilla


# ══════════════ semántica de ejecución ══════════════

def test_solo_se_actua_en_la_vela_nueva():
    """Gestionar en cada tick haría saltar el break-even y el trailing en
    momentos que el backtest nunca vio."""
    assert 'if(!zcsVelaNueva()) return;' in _generar()


def test_las_senales_se_leen_de_la_vela_cerrada():
    """El motor decide con la vela t y ejecuta al open de t+1: cuando abre la
    vela nueva, la de la señal es la 1."""
    codigo = _generar('RSI', params={'periodo': 9})
    assert 'double rsi_9_1 = zcsValor(h_rsi_9, 0, 1);' in codigo
    assert 'bool entradaLong = rsi_9_1 < p_sobreventa;' in codigo


def test_los_cruces_comparan_dos_velas():
    """Comparar solo el estado actual convertiría el cruce en una condición
    de posición y dispararía en cada vela mientras se mantuviera."""
    codigo = _generar('Cruce de medias', params={'rapida': 20, 'lenta': 50})
    assert 'zcsCruzaArriba(sma_20_1, sma_20_2, sma_50_1, sma_50_2)' in codigo


def test_la_serie_previa_solo_se_declara_una_vez_por_desplazamiento():
    codigo = _generar('Cruce de medias', params={'rapida': 20, 'lenta': 50})
    for nombre in ('sma_20_1', 'sma_20_2', 'sma_50_1', 'sma_50_2'):
        assert codigo.count(f"double {nombre} =") == 1, nombre


def test_el_sar_expone_su_tendencia_para_detectar_el_giro():
    """La estrategia entra por el GIRO del SAR, no por su nivel: sin la
    tendencia no se puede saber que ha girado."""
    codigo = _generar('Parabolic SAR')
    assert 'zcsSar(' in codigo
    assert re.search(r'_tend_1 == 1 && \w+_tend_2 == -1', codigo)


def test_el_estocastico_comparte_un_solo_handle_para_k_y_d():
    """%K y %D salen de los buffers 0 y 1 del mismo iStochastic: dos handles
    idénticos malgastarían recursos y podrían desincronizarse."""
    codigo = _generar('Stochastic (%K/%D)')
    assert len(set(re.findall(r'(h_\w*stoch\w*) = iStochastic', codigo))) == 1
    assert re.search(r'zcsValor\(h_\w*stoch\w*, 0, 1\)', codigo)
    assert re.search(r'zcsValor\(h_\w*stoch\w*, 1, 1\)', codigo)


def test_el_cci_usa_el_precio_tipico():
    """El motor calcula el CCI sobre (h+l+c)/3, que en MQL es PRICE_TYPICAL;
    con PRICE_CLOSE daría otro indicador."""
    assert 'iCCI(_Symbol, _Period, p_periodo, PRICE_TYPICAL)' in _generar('CCI')


# ══════════════ gestión y riesgo ══════════════

def test_el_tamano_se_calcula_por_riesgo_y_en_lotes():
    codigo = _generar()
    assert 'zcsLotesPorRiesgo(p_riesgo_pct / 100.0, distRef)' in codigo
    assert 'if(lotes > 0.0)' in codigo


def test_el_break_even_en_r_usa_la_distancia_y_en_atr_el_atr():
    assert 'double refBe = zcsDistEntrada;' in _generar(
        'RSI', be_atr=1.0, be_unidad='r')
    assert 'double refBe = atrGestion;' in _generar(
        'RSI', be_atr=1.0, be_unidad='atr')


def test_la_entrada_escalonada_emite_tramos():
    """Cada tramo adicional abre posición con su propio tamaño de riesgo y el
    estado zcsTramoActual/zcsTramoOrden para no repetir ni solapar."""
    tramos = [{'pct': 50.0, 'trigger': 'senal', 'val': 0.0, 'condiciones': []},
              {'pct': 50.0, 'trigger': 'retroceso', 'val': 1.0,
               'condiciones': []}]
    codigo = _generar('RSI', tramos=tramos)
    assert 'zcsTramoActual' in codigo and 'zcsTramoOrden' in codigo
    assert 'zcsLotesPorRiesgo(p_riesgo_pct / 100.0 * 0.5, distTramo)' in codigo
    assert '(l1 <= precioIn - 1.0 * atrGestion)' in codigo
    assert 'distTramo' in codigo


def test_un_solo_tramo_no_genera_escalonado():
    codigo = _generar('RSI')
    assert 'zcsTramoActual' not in codigo
    assert 'distTramo' not in codigo


def test_cada_setup_tiene_su_propio_magic_number():
    """Cada setup se exporta a su EA: sin magic propio se pisarían las
    posiciones entre ellos."""
    assert _magic('sistema', 0) != _magic('sistema', 1)
    assert _magic('otro', 0) != _magic('sistema', 0)


def test_el_magic_number_es_estable_entre_ejecuciones():
    """Si cambiara al reexportar, el EA dejaría de reconocer sus propias
    posiciones abiertas. Por eso se deriva con crc32 y no con hash()."""
    assert _magic('zc rsi', 0) == _magic('zc rsi', 0)
    assert 500000 <= _magic('zc rsi', 0) < 900000


# ══════════════ guardas ══════════════

def test_el_ea_se_niega_a_arrancar_en_otra_temporalidad():
    """Los parámetros están ajustados a esa temporalidad; en otra no
    significan lo mismo, y en MQL sí se puede impedir el arranque."""
    codigo = _generar()
    assert 'zcsActivoCorrecto(p_simbolo_esperado, 1440, p_permitir_otro' in codigo
    assert 'return(INIT_FAILED);' in codigo
    assert 'input bool   p_permitir_otro = false;' in codigo


def test_el_simbolo_no_impide_arrancar_el_ea():
    """El nombre del CSV del backtest («zcmaiz») casi nunca coincide con el
    del bróker («ZC», «ZCH2026»), así que exigir que casen dejaría el EA sin
    arrancar nunca. Solo avisa; la temporalidad es la que decide."""
    codigo = _generar()
    assert 'string motivo = "", aviso = "";' in codigo
    assert 'if(StringLen(aviso) > 0) Print(aviso);' in codigo


def test_la_cabecera_lleva_las_notas_de_fidelidad():
    assert 'NOTAS DE FIDELIDAD' in _generar()


def test_un_sistema_con_omisiones_avisa_al_arrancar():
    """El filtro de noticias NO sirve aquí: en MQL5 está aproximado (hay
    calendario nativo), no omitido. Las salidas parciales sí están omitidas
    mientras el generador no las emita."""
    codigo = _generar('RSI',
                      parciales=[{'pct': 50.0, 'trigger': 'r', 'r': 1.5},
                                 {'pct': 50.0, 'trigger': 'senal'}])
    assert 'AVISO DE FIDELIDAD' in codigo
    assert re.search(r'Print\("AVISO DE FIDELIDAD', codigo)


def test_un_sistema_sin_omisiones_no_mete_aviso_al_arrancar():
    """Solo las omisiones justifican gritar en el log; lo aproximado ya está
    en la cabecera del archivo."""
    assert 'AVISO DE FIDELIDAD' not in _generar()


def test_el_codigo_generado_no_lleva_acentos():
    """MetaEditor guarda en ANSI o UTF-8 según la versión y un comentario con
    acentos puede salir en mojibake."""
    codigo = _generar()
    cuerpo = codigo[codigo.find('#property'):]
    cuerpo.encode('ascii')


# ══════════════ el emisor rompe en vez de callarse ══════════════

def test_un_nodo_desconocido_lanza():
    with pytest.raises(ValueError, match='sin traducción'):
        EmisorMQL5().expr({'op': 'inventado', 'partes': []})


def test_el_hurst_ya_se_emite_porque_esta_portado():
    """El Hurst se porta al runtime (zcsHurst) y el emisor lo declara."""
    codigo = EmisorMQL5().declarar_serie('h_1', {'tipo': 'HURST',
                                                 'periodo': 400})
    assert 'zcsHurst(400, 1)' in codigo


# ══════════════ archivos ══════════════

def test_el_arbol_respeta_la_estructura_de_metatrader():
    sistema = ir.ir_sistema([{'nombre': 'x', 'plantilla': 'RSI'}], {})
    avisos = fidelidad.analizar(sistema, 'mt5')
    em = EmisorMQL5()
    rutas = set(em.archivos_setup(sistema['setups'][0], sistema, avisos, META))
    rutas |= set(em.archivos_comunes(sistema, avisos, META))
    assert any(r.startswith('MT5/MQL5/Experts/') and r.endswith('.mq5')
               for r in rutas)
    assert any(r.startswith('MT5/MQL5/Include/') and r.endswith('.mqh')
               for r in rutas)
    assert 'MT5/INSTALAR.md' in rutas


def test_el_ea_incluye_el_runtime_por_su_ruta_de_include():
    codigo = _generar()
    assert re.search(r'#include <\w+/zcs_runtime\.mqh>', codigo)


# ══════════════ filtro de noticias (calendario de MetaQuotes) ══════════════

def _generar_con_noticias(cerrar=False):
    filtros = _filtros_por_defecto()
    filtros['noticias'] = {
        'activo': True, 'minutos_antes': 30, 'minutos_despues': 30,
        'impacto_minimo': 'alto', 'monedas': ['USD'],
        'cerrar_posiciones': cerrar,
    }
    return _generar(filtros=filtros)


def test_noticias_activas_genera_filtro_de_calendario():
    codigo = _generar_con_noticias()
    assert 'zcsHayEvento' in codigo
    assert 'filtroNoticias' in codigo
    assert 'CALENDAR_IMPACT_HIGH' in codigo
    assert '"USD"' in codigo
    assert 'filtroLong = filtroNoticias' in codigo
    assert 'filtroShort = filtroNoticias' in codigo
    # la ventana hacia delante cubre al menos el resto de la vela actual
    assert 'MathMax(30, PeriodSeconds(_Period) / 60) * 60' in codigo


def test_noticias_cerrar_genera_cierre_por_evento_inminente():
    codigo = _generar_con_noticias(cerrar=True)
    assert 'zcsCerrar(p_magic)' in codigo
    assert 'zcsHayEvento(TimeCurrent(), TimeCurrent() + MathMax(30,' in codigo


def test_noticias_inactivas_no_genera_nada_de_calendario():
    codigo = _generar()
    assert 'zcsHayEvento' not in codigo
    assert 'filtroNoticias' not in codigo
    assert 'CalendarValue' not in codigo


def test_noticias_sin_monedas_declara_array_sin_inicializar():
    """MQL5 no admite `string x[] = {};` (error de compilación); con monedas
    sin resolver se declara el array dinámico sin inicializar y se pasa n=0."""
    filtros = _filtros_por_defecto()
    filtros['noticias'] = {
        'activo': True, 'minutos_antes': 30, 'minutos_despues': 30,
        'impacto_minimo': 'alto', 'monedas': None, 'cerrar_posiciones': False,
    }
    codigo = _generar(filtros=filtros)
    assert 'string zcsMonedas[];' in codigo
    assert 'string zcsMonedas[] = {};' not in codigo
    assert 'zcsMonedas, 0' in codigo
