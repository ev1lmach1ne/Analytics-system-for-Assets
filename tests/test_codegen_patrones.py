"""
tests/test_codegen_patrones.py
El catálogo neutro de patrones de vela.

El test importante es el DIFERENCIAL: cada una de las 32 fórmulas del catálogo
se evalúa sobre datos reales y se compara vela a vela con detectar_patrones,
que es lo que ejecuta el motor. Una transcripción mal copiada —un índice
cambiado, un `<=` donde iba `<`— no se ve leyendo, pero aquí falla.

Sin esta comprobación, un patrón mal transcrito produciría un robot que opera
en velas distintas a las del backtest sin que nada lo delate.
"""
import re

import numpy as np
import pytest

from core.candle_patterns import (PARAMS_DEFECTO, PATRONES_INFO,
                                  detectar_patrones, preparar_anatomia)
from core.codegen import patrones as pat

# velas de calentamiento antes de comparar: cuerpo_medio necesita 20 velas
# anteriores y la tendencia 6, así que antes de eso el motor trabaja con NaN
CALENTAMIENTO = 30


def _ohlc(n=600, semilla=7):
    """OHLC sintético con cuerpos, mechas y huecos variados, para que
    aparezcan patrones de las cinco longitudes."""
    rng = np.random.default_rng(semilla)
    cierre = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    apertura = cierre + rng.normal(0, 0.8, n)
    # algunos huecos reales, que es lo único que dispara kicker y abandoned baby
    apertura[::37] += rng.choice([-4.0, 4.0], len(apertura[::37]))
    alto = np.maximum(apertura, cierre) + np.abs(rng.normal(0, 0.6, n))
    bajo = np.minimum(apertura, cierre) - np.abs(rng.normal(0, 0.6, n))
    return apertura, alto, bajo, cierre


class _Ventana:
    """Acceso `var[k]` a los valores de una vela y sus anteriores, para poder
    evaluar la expresión del catálogo con la sintaxis de Python tal cual."""

    def __init__(self, serie, t):
        self._serie = serie
        self._t = t

    def __getitem__(self, k):
        return self._serie[self._t - k]


def _a_python(expresion):
    """Sustituye los umbrales P(clave) por su valor.

    Es lo mismo que hará cada emisor —cambiar P(clave) por el input de su
    plataforma—, solo que aquí se pone el número directamente. Sin esto,
    Python intentaría resolver `clave` como una variable."""
    return re.sub(r'\bP\((\w+)\)',
                  lambda m: repr(PARAMS_DEFECTO[m.group(1)]), expresion)


def _evaluar(expresion_py, anatomia, t):
    """Evalúa una expresión del catálogo, ya con los umbrales sustituidos, en
    la vela t.

    Se evalúa escalar, vela a vela, en vez de vectorizado: `and`/`or` de
    Python funcionan entonces igual que en el minilenguaje, sin tener que
    traducirlos a `&`/`|` y pelearse con la precedencia de operadores."""
    entorno = {nombre: _Ventana(anatomia[_CLAVE_ANATOMIA[nombre]], t)
               for nombre in pat.VARIABLES}
    entorno['max'] = max
    entorno['min'] = min
    entorno['abs'] = abs
    return bool(eval(expresion_py, {'__builtins__': {}}, entorno))  # noqa: S307


_CLAVE_ANATOMIA = {
    'o': 'o', 'h': 'h', 'l': 'l', 'c': 'c',
    'cu': 'abs_cuerpo', 'r': 'rango', 'ms': 'mecha_sup', 'mi': 'mecha_inf',
    'alc': 'alcista', 'baj': 'bajista', 'val': 'valida',
    'cm': 'cuerpo_medio', 'T': 'tendencia',
}


@pytest.fixture(scope='module')
def datos():
    o, h, l, c = _ohlc()
    return {'ohlc': (o, h, l, c),
            'anatomia': preparar_anatomia(o, h, l, c),
            'motor': detectar_patrones(o, h, l, c)}


# ══════════════ el test diferencial ══════════════

@pytest.mark.parametrize('nombre', sorted(PATRONES_INFO))
def test_la_formula_del_catalogo_coincide_con_el_motor(nombre, datos):
    """Vela a vela, la fórmula neutra tiene que marcar exactamente las mismas
    ocurrencias que detectar_patrones."""
    expresion, _direccion = pat.definicion(nombre)
    expresion_py = _a_python(expresion)
    anatomia = datos['anatomia']
    n = len(anatomia['c'])

    mias = {t for t in range(CALENTAMIENTO, n)
            if _evaluar(expresion_py, anatomia, t)}
    suyas = {int(i) for i in datos['motor'][nombre]['idx']
             if i >= CALENTAMIENTO}

    faltan = sorted(suyas - mias)[:5]
    sobran = sorted(mias - suyas)[:5]
    assert mias == suyas, (
        f"{nombre}: el catálogo no reproduce el motor. "
        f"Velas que el motor marca y el catálogo no: {faltan}. "
        f"Velas que el catálogo marca de más: {sobran}.")


def test_los_datos_de_prueba_disparan_patrones_de_verdad(datos):
    """Un dataset que no produjera ocurrencias haría pasar el test diferencial
    comparando dos conjuntos vacíos."""
    con_ocurrencias = [n for n, occ in datos['motor'].items()
                       if len(occ['idx']) > 0]
    assert len(con_ocurrencias) >= 20, sorted(datos['motor'])


# ══════════════ coherencia del catálogo ══════════════

def test_el_catalogo_cubre_los_mismos_patrones_que_el_motor():
    """Si alguien añade un patrón a candle_patterns y no aquí, un sistema que
    lo use se exportaría en silencio sin él."""
    assert pat.comprobar_catalogo()
    assert len(pat.PATRONES) == len(PATRONES_INFO) == 32


def test_el_sesgo_de_cada_patron_coincide_con_el_del_motor():
    for nombre, info in PATRONES_INFO.items():
        _expr, direccion = pat.definicion(nombre)
        if info['dir'] == 0:
            assert direccion == 'menos_T', nombre
        else:
            assert direccion == info['dir'], nombre


def test_doji_y_spinning_top_resuelven_su_direccion_en_ejecucion():
    """Su sesgo depende de la tendencia previa, así que el emisor no puede
    fijar la dirección al generar el código."""
    for nombre in ('Doji', 'Spinning Top'):
        assert pat.definicion(nombre)[1] == 'menos_T'


# ══════════════ el minilenguaje ══════════════

def test_las_subexpresiones_se_expanden_desplazadas():
    """Es lo que permite que «Three Outside Up» reutilice la envolvente de la
    vela anterior sin volver a escribirla."""
    expandida = pat.expandir('@env_alc[1]')
    assert 'val[1]' in expandida and 'val[2]' in expandida
    assert 'val[0]' not in expandida


def test_toda_expresion_del_catalogo_valida():
    for nombre in pat.PATRONES:
        pat.definicion(nombre)


def test_una_variable_inventada_se_rechaza():
    """La red de seguridad: un identificador mal escrito se colaría en el
    archivo generado y en MQL sería un error de compilación que solo vería el
    usuario."""
    with pytest.raises(ValueError, match='desconocida'):
        pat.validar('inventada[0] and val[0]')


def test_un_umbral_inventado_se_rechaza():
    with pytest.raises(ValueError, match='Umbral desconocido'):
        pat.validar('cu[0] > P(no_existe) * r[0]')


def test_un_identificador_suelto_se_rechaza():
    with pytest.raises(ValueError, match='Identificador suelto'):
        pat.validar('val[0] and cierre > 3')


def test_una_subexpresion_inventada_se_rechaza():
    with pytest.raises(ValueError, match='Subexpresión desconocida'):
        pat.validar('@no_existe[0]')


def test_se_rechaza_pasar_del_desplazamiento_maximo():
    """El emisor declara la anatomía hasta MAX_DESPLAZAMIENTO: pedir más
    velas dejaría una variable sin declarar."""
    with pytest.raises(ValueError, match='desplazamiento máximo'):
        pat.validar('val[9]')


# ══════════════ lo que el emisor necesita saber ══════════════

def test_variables_usadas_no_repite_y_va_ordenado():
    usadas = pat.variables_usadas(pat.PATRONES['Martillo'][0])
    assert usadas == sorted(set(usadas))
    assert ('T', 0) in usadas and ('cu', 0) in usadas


def test_los_patrones_de_una_vela_no_piden_velas_anteriores():
    for nombre in ('Doji', 'Martillo', 'Marubozu Alcista'):
        indices = {k for _v, k in pat.variables_usadas(pat.PATRONES[nombre][0])}
        assert indices == {0}, nombre


def test_los_de_cinco_velas_llegan_hasta_el_desplazamiento_cuatro():
    for nombre in ('Rising Three Methods', 'Falling Three Methods'):
        indices = {k for _v, k in pat.variables_usadas(pat.PATRONES[nombre][0])}
        assert max(indices) == 4, nombre


def test_el_desplazamiento_maximo_del_catalogo_completo_es_cuatro():
    """Es cuánta anatomía tiene que declarar el emisor en el peor caso."""
    assert pat.desplazamiento_maximo(list(pat.PATRONES)) == 4
    assert pat.MAX_DESPLAZAMIENTO == 4


def test_solo_se_piden_los_umbrales_que_la_formula_usa():
    umbrales = pat.umbrales_usados(pat.PATRONES['Martillo'][0])
    assert set(umbrales) == {'mecha_dominante', 'mecha_opuesta_max'}
