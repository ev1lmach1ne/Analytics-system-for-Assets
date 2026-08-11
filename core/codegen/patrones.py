"""
core/codegen/patrones.py
Catálogo neutro de los 32 patrones de vela, para que cada emisor los escriba
en su lenguaje sin volver a interpretar la geometría.

POR QUÉ UN CATÁLOGO Y NO UNA IMPLEMENTACIÓN POR PLATAFORMA
──────────────────────────────────────────────────────────
32 patrones × cada plataforma son decenas de fórmulas duplicadas que se
desincronizan a la primera corrección. Aquí la geometría se declara UNA vez, y
cada emisor solo aporta cómo se escribe un «y», un «máximo» y una variable
desplazada. Además el emisor renderiza únicamente los patrones que ese setup
usa (ir_setup['patrones']), no los 32.

EL MINILENGUAJE
───────────────
Cada patrón es una expresión booleana sobre variables de anatomía indexadas
por desplazamiento, donde [0] es la ÚLTIMA vela de la formación (la que
dispara la señal) y [1], [2]… las anteriores:

    o h l c      apertura, máximo, mínimo, cierre
    cu           |cuerpo|          r    rango (máximo - mínimo)
    ms mi        mecha superior / inferior
    alc baj      vela alcista / bajista
    val          vela válida (rango > 0)
    cm           cuerpo medio de las 20 velas ANTERIORES (excluye la actual)
    T            tendencia previa: signo de c[t-1] - c[t-6]

    P(clave)     umbral configurable (PARAMS_DEFECTO de candle_patterns)
    @nombre[k]   subexpresión de SUBEXPRESIONES, desplazada k velas
    max(a, b)  min(a, b)  abs(a)

Se valida con validar(): cualquier identificador que no esté en el vocabulario
hace saltar un error en vez de colarse como código roto en el archivo
generado.

FIDELIDAD
─────────
Las fórmulas son transcripción directa de detectar_patrones
(core/candle_patterns.py). Dos detalles que es fácil perder al portarlas:

  · `cm` promedia las velas ANTERIORES, no incluye la actual — si la
    incluyera, «cuerpo grande» se compararía consigo mismo.
  · Doji y Spinning Top tienen sesgo -T: su dirección se decide EN EJECUCIÓN
    según la tendencia previa, no al generar el código.
"""
import re

from core.candle_patterns import PARAMS_DEFECTO, PATRONES_INFO

# variables de anatomía que el emisor tiene que declarar por desplazamiento
VARIABLES = ('o', 'h', 'l', 'c', 'cu', 'r', 'ms', 'mi',
             'alc', 'baj', 'val', 'cm', 'T')

# desplazamiento máximo que puede pedir un patrón (Rising/Falling Three
# Methods miran 5 velas)
MAX_DESPLAZAMIENTO = 4

_RE_VAR = re.compile(r'\b([a-zA-Z_]\w*)\[(\d+)\]')
_RE_SUB = re.compile(r'@(\w+)\[(\d+)\]')
_RE_PARAM = re.compile(r'\bP\((\w+)\)')
_RE_IDENT = re.compile(r'\b([a-zA-Z_]\w*)\b')

_PALABRAS = {'and', 'or', 'not', 'max', 'min', 'abs', 'P'}


# ══════════════ subexpresiones reutilizadas ══════════════

# Se declaran aparte porque otros patrones las miran DESPLAZADAS: «Three
# Outside Up» es una envolvente alcista en la vela anterior más una tercera
# que confirma.
SUBEXPRESIONES = {
    'geo_martillo':
        'val[0] and cu[0] > 0 and mi[0] >= P(mecha_dominante) * cu[0] '
        'and ms[0] <= P(mecha_opuesta_max) * r[0]',
    'geo_invertido':
        'val[0] and cu[0] > 0 and ms[0] >= P(mecha_dominante) * cu[0] '
        'and mi[0] <= P(mecha_opuesta_max) * r[0]',
    'doji_geo':
        'val[0] and cu[0] <= P(doji_cuerpo_max) * r[0]',
    'geo_marubozu':
        'val[0] and cu[0] >= P(cuerpo_grande_min) * cm[0] '
        'and ms[0] <= P(marubozu_mecha_max) * r[0] '
        'and mi[0] <= P(marubozu_mecha_max) * r[0]',
    'spinning':
        'val[0] and cu[0] > P(doji_cuerpo_max) * r[0] '
        'and cu[0] <= P(cuerpo_pequeno_max) * r[0] '
        'and ms[0] >= P(spinning_mecha_min) * cu[0] '
        'and mi[0] >= P(spinning_mecha_min) * cu[0] '
        'and abs(ms[0] - mi[0]) <= P(spinning_asimetria_max) * max(ms[0], mi[0]) '
        'and T[0] != 0',
    'env_alc':
        'val[0] and val[1] and baj[1] and alc[0] '
        'and o[0] <= c[1] and c[0] >= o[1] and cu[0] > cu[1]',
    'env_baj':
        'val[0] and val[1] and alc[1] and baj[0] '
        'and o[0] >= c[1] and c[0] <= o[1] and cu[0] > cu[1]',
    'harami_alc':
        'val[0] and val[1] and baj[1] '
        'and cu[1] >= P(cuerpo_grande_min) * cm[1] '
        'and max(o[0], c[0]) <= o[1] and min(o[0], c[0]) >= c[1]',
    'harami_baj':
        'val[0] and val[1] and alc[1] '
        'and cu[1] >= P(cuerpo_grande_min) * cm[1] '
        'and max(o[0], c[0]) <= c[1] and min(o[0], c[0]) >= o[1]',
    # las tres velas de contracción de Rising/Falling Three Methods, contenidas
    # en el rango de la primera y con cuerpo menor que ella
    'contraccion':
        'h[3] <= h[4] and l[3] >= l[4] and cu[3] < cu[4] '
        'and h[2] <= h[4] and l[2] >= l[4] and cu[2] < cu[4] '
        'and h[1] <= h[4] and l[1] >= l[4] and cu[1] < cu[4]',
}


# ══════════════ los 32 patrones ══════════════

# 'dir': +1 / -1 fijos, o 'menos_T' cuando la dirección depende de la
# tendencia previa y hay que resolverla en ejecución.
PATRONES = {
    # ── 1 vela ──
    'Doji': ('@doji_geo[0] and T[0] != 0', 'menos_T'),
    'Martillo': ('@geo_martillo[0] and T[0] < 0', +1),
    'Hombre Colgado': ('@geo_martillo[0] and T[0] > 0', -1),
    'Martillo Invertido': ('@geo_invertido[0] and T[0] < 0', +1),
    'Estrella Fugaz': ('@geo_invertido[0] and T[0] > 0', -1),
    'Doji Libélula': (
        '@doji_geo[0] and mi[0] >= P(doji_direccional_mecha_min) * r[0] '
        'and ms[0] <= P(mecha_opuesta_max) * r[0]', +1),
    'Doji Lápida': (
        '@doji_geo[0] and ms[0] >= P(doji_direccional_mecha_min) * r[0] '
        'and mi[0] <= P(mecha_opuesta_max) * r[0]', -1),
    'Marubozu Alcista': ('@geo_marubozu[0] and alc[0]', +1),
    'Marubozu Bajista': ('@geo_marubozu[0] and baj[0]', -1),
    'Spinning Top': ('@spinning[0]', 'menos_T'),

    # ── 2 velas ──
    'Envolvente Alcista': ('@env_alc[0]', +1),
    'Envolvente Bajista': ('@env_baj[0]', -1),
    'Harami Alcista': ('@harami_alc[0]', +1),
    'Harami Bajista': ('@harami_baj[0]', -1),
    'Piercing Line': (
        'val[0] and val[1] and baj[1] '
        'and cu[1] >= P(cuerpo_grande_min) * cm[1] and alc[0] '
        'and o[0] < c[1] and c[0] > (o[1] + c[1]) / 2 and c[0] < o[1]', +1),
    'Dark Cloud Cover': (
        'val[0] and val[1] and alc[1] '
        'and cu[1] >= P(cuerpo_grande_min) * cm[1] and baj[0] '
        'and o[0] > c[1] and c[0] < (o[1] + c[1]) / 2 and c[0] > o[1]', -1),
    'Tweezer Top': (
        'val[0] and val[1] and abs(h[0] - h[1]) <= P(tweezer_tol) * r[0] '
        'and T[0] > 0', -1),
    'Tweezer Bottom': (
        'val[0] and val[1] and abs(l[0] - l[1]) <= P(tweezer_tol) * r[0] '
        'and T[0] < 0', +1),
    # kicker: cambio de color con hueco REAL entre cuerpos. Con datos intradía
    # sin huecos puede no aparecer nunca fuera de temporalidades altas.
    'Kicker Alcista': (
        'val[0] and val[1] and baj[1] '
        'and cu[1] >= P(cuerpo_grande_min) * cm[1] and alc[0] '
        'and cu[0] >= P(cuerpo_grande_min) * cm[0] '
        'and min(o[0], c[0]) > max(o[1], c[1])', +1),
    'Kicker Bajista': (
        'val[0] and val[1] and alc[1] '
        'and cu[1] >= P(cuerpo_grande_min) * cm[1] and baj[0] '
        'and cu[0] >= P(cuerpo_grande_min) * cm[0] '
        'and max(o[0], c[0]) < min(o[1], c[1])', -1),

    # ── 3 velas ──
    'Three Outside Up': ('val[0] and @env_alc[1] and alc[0] and c[0] > c[1]', +1),
    'Three Outside Down': ('val[0] and @env_baj[1] and baj[0] and c[0] < c[1]', -1),
    'Three Inside Up': ('val[0] and @harami_alc[1] and c[0] > o[2]', +1),
    'Three Inside Down': ('val[0] and @harami_baj[1] and c[0] < o[2]', -1),
    'Morning Star': (
        'val[0] and val[1] and val[2] and baj[2] '
        'and cu[2] >= P(cuerpo_grande_min) * cm[2] '
        'and cu[1] <= P(cuerpo_pequeno_max) * cu[2] '
        'and alc[0] and c[0] > (o[2] + c[2]) / 2', +1),
    'Evening Star': (
        'val[0] and val[1] and val[2] and alc[2] '
        'and cu[2] >= P(cuerpo_grande_min) * cm[2] '
        'and cu[1] <= P(cuerpo_pequeno_max) * cu[2] '
        'and baj[0] and c[0] < (o[2] + c[2]) / 2', -1),
    # abandoned baby: morning/evening star con hueco real a los dos lados, de
    # modo que la vela central queda aislada
    'Abandoned Baby Alcista': (
        'val[0] and val[1] and val[2] and baj[2] '
        'and cu[2] >= P(cuerpo_grande_min) * cm[2] '
        'and cu[1] <= P(doji_cuerpo_max) * r[1] and alc[0] '
        'and max(o[1], c[1]) < min(o[2], c[2]) '
        'and min(o[0], c[0]) > max(o[1], c[1])', +1),
    'Abandoned Baby Bajista': (
        'val[0] and val[1] and val[2] and alc[2] '
        'and cu[2] >= P(cuerpo_grande_min) * cm[2] '
        'and cu[1] <= P(doji_cuerpo_max) * r[1] and baj[0] '
        'and min(o[1], c[1]) > max(o[2], c[2]) '
        'and max(o[0], c[0]) < min(o[1], c[1])', -1),
    'Tres Soldados Blancos': (
        'val[0] and val[1] and val[2] and alc[0] and alc[1] and alc[2] '
        'and c[0] > c[1] and c[1] > c[2] '
        'and o[0] >= o[1] and o[0] <= c[1] '
        'and o[1] >= o[2] and o[1] <= c[2] '
        'and ms[0] <= P(soldados_mecha_max) * r[0] '
        'and ms[1] <= P(soldados_mecha_max) * r[1] '
        'and ms[2] <= P(soldados_mecha_max) * r[2]', +1),
    'Tres Cuervos Negros': (
        'val[0] and val[1] and val[2] and baj[0] and baj[1] and baj[2] '
        'and c[0] < c[1] and c[1] < c[2] '
        'and o[0] <= o[1] and o[0] >= c[1] '
        'and o[1] <= o[2] and o[1] >= c[2] '
        'and mi[0] <= P(soldados_mecha_max) * r[0] '
        'and mi[1] <= P(soldados_mecha_max) * r[1] '
        'and mi[2] <= P(soldados_mecha_max) * r[2]', -1),

    # ── 5 velas (continuación) ──
    'Rising Three Methods': (
        'val[0] and val[1] and val[2] and val[3] and val[4] and alc[4] '
        'and cu[4] >= P(cuerpo_grande_min) * cm[4] and @contraccion[0] '
        'and alc[0] and cu[0] >= P(cuerpo_grande_min) * cm[0] '
        'and c[0] > c[4]', +1),
    'Falling Three Methods': (
        'val[0] and val[1] and val[2] and val[3] and val[4] and baj[4] '
        'and cu[4] >= P(cuerpo_grande_min) * cm[4] and @contraccion[0] '
        'and baj[0] and cu[0] >= P(cuerpo_grande_min) * cm[0] '
        'and c[0] < c[4]', -1),
}


# ══════════════ expansión y validación ══════════════

def expandir(expresion, desplazamiento=0):
    """Resuelve las @subexpresiones y aplica un desplazamiento global.

    Desplazar es lo que permite que «Three Outside Up» reutilice la envolvente
    de la vela anterior sin volver a escribirla: @env_alc[1] es la misma
    fórmula con todos sus índices sumados en 1."""
    texto = expresion
    for _ in range(10):                      # las anidaciones reales son de 1
        pendiente = _RE_SUB.search(texto)
        if pendiente is None:
            break
        nombre, k = pendiente.group(1), int(pendiente.group(2))
        if nombre not in SUBEXPRESIONES:
            raise ValueError(f"Subexpresión desconocida: @{nombre}")
        interna = _desplazar(SUBEXPRESIONES[nombre], k)
        texto = texto[:pendiente.start()] + f"({interna})" + texto[pendiente.end():]
    else:
        raise ValueError(f"Subexpresiones anidadas sin fin en: {expresion!r}")
    return _desplazar(texto, desplazamiento) if desplazamiento else texto


def _desplazar(texto, k):
    if k == 0:
        return texto
    return _RE_VAR.sub(lambda m: f"{m.group(1)}[{int(m.group(2)) + k}]", texto)


def validar(expresion):
    """Comprueba que la expresión solo usa el vocabulario declarado.

    Es la red de seguridad del catálogo: una variable mal escrita se colaría
    en el archivo generado como un identificador inexistente, y en MQL eso es
    un error de compilación que solo vería el usuario."""
    texto = expandir(expresion)
    for nombre, indice in _RE_VAR.findall(texto):
        if nombre not in VARIABLES:
            raise ValueError(f"Variable de anatomía desconocida: {nombre}")
        if int(indice) > MAX_DESPLAZAMIENTO:
            raise ValueError(
                f"{nombre}[{indice}] pasa del desplazamiento máximo "
                f"({MAX_DESPLAZAMIENTO})")
    for clave in _RE_PARAM.findall(texto):
        if clave not in PARAMS_DEFECTO:
            raise ValueError(f"Umbral desconocido: P({clave})")
    # cualquier identificador suelto que no sea variable indexada, umbral o
    # palabra del minilenguaje es un error de transcripción
    sin_vars = _RE_VAR.sub('', _RE_PARAM.sub('', texto))
    for ident in _RE_IDENT.findall(sin_vars):
        if ident not in _PALABRAS:
            raise ValueError(f"Identificador suelto sin significado: {ident}")
    return texto


def variables_usadas(expresion):
    """(nombre, desplazamiento) de cada variable de anatomía que la expresión
    necesita, sin repetir. El emisor declara solo esas."""
    texto = expandir(expresion)
    vistas = []
    for nombre, indice in _RE_VAR.findall(texto):
        par = (nombre, int(indice))
        if par not in vistas:
            vistas.append(par)
    return sorted(vistas)


def umbrales_usados(expresion):
    texto = expandir(expresion)
    vistos = []
    for clave in _RE_PARAM.findall(texto):
        if clave not in vistos:
            vistos.append(clave)
    return vistos


def definicion(nombre):
    """(expresion_expandida, direccion) de un patrón. La dirección es +1/-1, o
    'menos_T' cuando depende de la tendencia previa y hay que resolverla en
    ejecución."""
    if nombre not in PATRONES:
        raise ValueError(f"Patrón desconocido: {nombre!r}")
    expresion, direccion = PATRONES[nombre]
    return validar(expresion), direccion


def desplazamiento_maximo(nombres):
    """Cuántas velas hacia atrás necesita el conjunto de patrones elegido: es
    lo que el emisor tiene que declarar de anatomía."""
    tope = 0
    for nombre in nombres:
        for _var, indice in variables_usadas(PATRONES[nombre][0]):
            tope = max(tope, indice)
    return tope


# ══════════════ coherencia con el motor ══════════════

def comprobar_catalogo():
    """Los 32 patrones del motor están aquí, con el mismo sesgo. Se llama
    desde los tests: si alguien añade un patrón a candle_patterns y no aquí,
    el sistema que lo use se exportaría en silencio sin él."""
    faltan = set(PATRONES_INFO) - set(PATRONES)
    sobran = set(PATRONES) - set(PATRONES_INFO)
    if faltan or sobran:
        raise ValueError(f"Catálogo desalineado — faltan: {sorted(faltan)}, "
                         f"sobran: {sorted(sobran)}")
    for nombre, info in PATRONES_INFO.items():
        _expresion, direccion = PATRONES[nombre]
        esperado = info['dir']
        real = 0 if direccion == 'menos_T' else direccion
        if real != esperado:
            raise ValueError(
                f"{nombre}: el catálogo dice {direccion} y el motor {esperado}")
    return True
