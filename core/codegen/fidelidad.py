"""
core/codegen/fidelidad.py
Qué sabe reproducir cada plataforma del sistema backtesteado y qué no.

POR QUÉ ESTO EXISTE
───────────────────
Un filtro que desaparece en la traducción convierte el código exportado en un
sistema DISTINTO del que el usuario acaba de ver en Resultados, y mirando el
.mq4 o el .pine no hay forma de notarlo: lo que falta no se ve. Es el mismo
problema que resuelven los `avisos` del motor (_mascara_filtros_setup, en
core/strategies.py), donde un filtro de régimen que no se pudo calcular tenía
que anunciarse porque, si no, el backtest salía idéntico a uno sin filtro.

Aquí la regla es la misma y se aplica antes de escribir nada: se recorre lo
que el sistema USA de verdad — no el catálogo entero — y se devuelve la lista
de lo que se pierde, con su consecuencia concreta.

La mayoría de indicadores NO están en esta lista porque se reimplementan: el
ER, el KAMA, el SAR, el percentil rodante de volatilidad, el ZigZag y los
patrones de vela se portan a la librería de runtime de cada plataforma. Solo
entra aquí lo que de verdad no tiene equivalente.

NIVELES
───────
  'exacto'      se reproduce fielmente; no genera aviso
  'aproximado'  se genera algo equivalente pero no idéntico  (⚠️)
  'omitido'     el código sale SIN eso  (❌)

Para 'omitido' no se emite un stub: un «// TODO: filtro de noticias» que no
filtra nada se lee como si estuviera resuelto, y es peor que su ausencia
declarada.

DOS RAZONES DISTINTAS PARA UN MISMO ❌
──────────────────────────────────────
Una característica puede estar omitida porque la PLATAFORMA no puede con ella
(el calendario económico de Pine Script no existe y no va a existir) o porque
el GENERADOR todavía no la emite (la entrada escalonada, que está en la fase
siguiente). Al usuario le importa la diferencia: la primera no va a cambiar
nunca y la segunda sí. Las pendientes llevan 'pendiente': True y su motivo lo
dice con todas las letras, en vez de dejar creer que la plataforma no llega.

CUANDO EL ❌ IMPIDE GENERAR EL ARCHIVO
──────────────────────────────────────
Casi todo lo omitido degrada a algo razonable (una entrada única en vez de
tres tramos). Pero si lo que falta es la propia SEÑAL — un setup de patrones
de vela, o unas reglas custom que miran el ZigZag — el archivo resultante
sería un EA que no opera nunca, y eso es peor que no dar archivo. Esas
características llevan 'bloquea_setup': True y exportar_sistema() se salta ese
setup diciéndolo.
"""
import unicodedata

from core.strategies import METODOS_REGIMEN_HURST

NIVEL_EXACTO = 'exacto'
NIVEL_APROXIMADO = 'aproximado'
NIVEL_OMITIDO = 'omitido'

# orden de gravedad para presentar los avisos
_ORDEN_NIVEL = {NIVEL_OMITIDO: 0, NIVEL_APROXIMADO: 1, NIVEL_EXACTO: 2}

ICONOS = {NIVEL_OMITIDO: '❌', NIVEL_APROXIMADO: '⚠️', NIVEL_EXACTO: '✅'}


# ══════════════ catálogo de características ══════════════

# 'etiqueta'   nombre de la característica tal y como la conoce el usuario
# 'motivos'    por qué esa plataforma no llega (por clave de plataforma)
# 'consecuencia' qué pasará al operar, por nivel — se redacta en concreto
#              ("hará más trades que el backtest"), no en abstracto
CARACTERISTICAS = {
    'noticias': {
        'etiqueta': 'Filtro de noticias',
        'motivos': {
            'mt4': "MQL4 no expone ninguna API de calendario económico.",
            'mt5': "MQL5 sí tiene calendario nativo (CalendarValueHistory), "
                   "pero es el de MetaQuotes: su clasificación de impacto y su "
                   "reparto por moneda no coinciden vela a vela con el "
                   "proveedor que usa el backtest.",
            'tradingview': "Pine Script no tiene acceso a ningún calendario "
                           "económico.",
        },
        'consecuencia': {
            NIVEL_OMITIDO:
                "El código se genera SIN ese filtro: operará también dentro "
                "de las ventanas de noticias que el backtest evitaba, así que "
                "hará más trades y con peor slippage que el backtest.",
            NIVEL_APROXIMADO:
                "El filtro se genera contra el calendario de la plataforma: "
                "bloqueará ventanas parecidas, pero no exactamente las mismas "
                "velas que el backtest.",
        },
    },
    'sesion_dst': {
        'etiqueta': 'Filtro de sesión con huso horario propio (Londres / NY)',
        'motivos': {
            'mt4': "MetaTrader fecha las velas en hora del SERVIDOR del bróker "
                   "y no lleva base de datos de husos horarios, así que el "
                   "cambio de horario de verano de esa plaza no se sigue solo.",
            'mt5': "MetaTrader fecha las velas en hora del SERVIDOR del bróker "
                   "y no lleva base de datos de husos horarios, así que el "
                   "cambio de horario de verano de esa plaza no se sigue solo.",
        },
        'consecuencia': {
            NIVEL_APROXIMADO:
                "La sesión se aproxima con el desfase del servidor, que hay "
                "que ajustar a mano en los parámetros del EA. En las semanas "
                "de cambio de horario las velas del borde quedarán dentro o "
                "fuera al revés que en el backtest.",
        },
    },
    'sesion_utc': {
        'etiqueta': 'Filtro de sesión por horas UTC',
        'motivos': {
            'mt4': "Las velas van en hora del servidor del bróker, no en UTC.",
            'mt5': "Las velas van en hora del servidor del bróker, no en UTC.",
        },
        'consecuencia': {
            NIVEL_APROXIMADO:
                "La franja horaria queda desplazada por el desfase del "
                "servidor (normalmente UTC+2/+3). Hay que ajustarlo en el "
                "parámetro DesfaseServidorHoras del EA.",
        },
    },
    'dias_semana': {
        'etiqueta': 'Filtro de días de la semana',
        'motivos': {
            'mt4': "El día de la vela se toma en hora del servidor, no en UTC.",
            'mt5': "El día de la vela se toma en hora del servidor, no en UTC.",
        },
        'consecuencia': {
            NIVEL_APROXIMADO:
                "Las velas cercanas a medianoche pueden caer en otro día que "
                "en el backtest, según el desfase del servidor.",
        },
    },
    'hurst': {
        'etiqueta': 'Filtro de régimen por exponente de Hurst',
        'motivos': {
            'tradingview': "El Hurst se estima por R/S sobre una ventana de "
                           "cientos de velas, con varios lags y una regresión, "
                           "y hay que rehacerlo en CADA vela. Pine Script "
                           "corta la ejecución por tiempo/iteraciones mucho "
                           "antes de eso, así que aquí no va a poder ser.",
            'mt4': "MQL4 sí puede con el cálculo (bucles nativos), pero el "
                   "generador todavía no lo emite: está en la fase siguiente.",
            'mt5': "MQL5 sí puede con el cálculo (bucles nativos), pero el "
                   "generador todavía no lo emite: está en la fase siguiente.",
        },
        'consecuencia': {
            NIVEL_OMITIDO:
                "El código se genera SIN el filtro de régimen: entrará también "
                "en los tramos que el backtest descartaba por régimen, así que "
                "hará bastantes más trades.",
        },
    },
    'relleno_open_siguiente': {
        'etiqueta': 'Ejecución al open de la vela siguiente',
        'motivos': {
            'mt4': "El EA reacciona al primer tick de la vela nueva, que ya "
                   "no es exactamente su apertura, y paga el spread real del "
                   "bróker en vez del slippage fijo del backtest.",
            'mt5': "El EA reacciona al primer tick de la vela nueva, que ya "
                   "no es exactamente su apertura, y paga el spread real del "
                   "bróker en vez del slippage fijo del backtest.",
        },
        'consecuencia': {
            NIVEL_APROXIMADO:
                "Cada entrada y cada salida tendrán un precio algo distinto "
                "al del backtest. Es la diferencia normal entre simular y "
                "operar, pero en sistemas de muchas operaciones cortas se "
                "acumula.",
        },
    },
    'atr_de_la_vela_de_entrada': {
        'etiqueta': 'ATR de dimensionamiento tomado de la vela de entrada',
        'motivo_comun': "El motor entra en la APERTURA de la vela i y "
                        "dimensiona con el ATR de la última vela CERRADA "
                        "(atr[i-1], ver core/backtest._atr_cerrado) — igual "
                        "que el código generado. Solo difiere en el borde: "
                        "cuando aún no hay vela anterior, el motor cae al ATR "
                        "de la propia vela de entrada o al fallback del 1% del "
                        "precio, mientras que el código generado espera a tener "
                        "una vela cerrada.",
        'consecuencia': {
            NIVEL_APROXIMADO:
                "Las primeras operaciones de la serie (antes de que exista un "
                "ATR válido de vela cerrada) pueden dimensionarse con una "
                "volatilidad algo distinta a la del código generado. Del resto "
                "de la serie, el tamaño y el stop coinciden.",
        },
    },
    'stop_atr_dinamico': {
        'etiqueta': 'Stop ×ATR dinámico por precio medio',
        'motivos': {
            'mt4': "El runtime del EA fija el stop de cada entrada con su "
                   "propio ATR y no reancla un stop compartido al precio medio "
                   "de la posición cuando se añaden promedios.",
            'mt5': "El runtime del EA fija el stop de cada entrada con su "
                   "propio ATR y no reancla un stop compartido al precio medio "
                   "de la posición cuando se añaden promedios.",
            'tradingview': "La estrategia generada fija el stop al entrar y no "
                           "lo reancla al precio medio cuando se añaden "
                           "promedios.",
        },
        'consecuencia': {
            NIVEL_OMITIDO:
                "El código se genera con el stop FIJO de la primera entrada: "
                "con promedios, el stop real operará distinto (más lejos o "
                "más cerca) que el del backtest.",
        },
    },
    'slippage_en_ticks': {
        'etiqueta': 'Slippage del backtest',
        'motivos': {
            'tradingview': "El backtest usa un slippage en PORCENTAJE del "
                           "precio; TradingView lo mide en ticks, y cuántos "
                           "ticks equivalen a ese porcentaje depende del "
                           "instrumento, así que no se puede rellenar solo.",
            'mt4': "El backtest suma un slippage fijo en PORCENTAJE del "
                   "precio a cada relleno; MQL4 solo admite un deslizamiento "
                   "en puntos al enviar la orden, que además no garantiza el "
                   "precio, así que el porcentaje del backtest no se puede "
                   "aplicar tal cual.",
            'mt5': "El backtest suma un slippage fijo en PORCENTAJE del "
                   "precio a cada relleno; MQL5 solo admite un deslizamiento "
                   "en puntos al enviar la orden, que además no garantiza el "
                   "precio, así que el porcentaje del backtest no se puede "
                   "aplicar tal cual.",
        },
        'consecuencia': {
            NIVEL_APROXIMADO:
                "El código se genera SIN el slippage del backtest: cada "
                "relleno ocurre al precio del mercado en ese momento (spread "
                "real del bróker y deslizamiento real), así que las entradas "
                "y salidas saldrán algo distintas a las del backtest. El "
                "valor del backtest está en la cabecera del archivo y en "
                "INSTALAR.md.",
        },
    },
    'redondeo_lotes': {
        'etiqueta': 'Tamaño de posición redondeado a lotes',
        'motivos': {
            'mt4': "El backtest dimensiona en unidades fraccionarias; "
                   "MetaTrader opera en lotes con un mínimo y un paso que "
                   "impone el bróker.",
            'mt5': "El backtest dimensiona en unidades fraccionarias; "
                   "MetaTrader opera en lotes con un mínimo y un paso que "
                   "impone el bróker.",
        },
        'consecuencia': {
            NIVEL_APROXIMADO:
                "El riesgo real de cada operación se redondea al paso de "
                "lote. En cuentas pequeñas o con lote mínimo alto el "
                "redondeo puede ser grande, y con un riesgo por debajo del "
                "lote mínimo el EA no abrirá la operación.",
        },
    },
    'comision': {
        'etiqueta': 'Comisión del backtest',
        'motivos': {
            'mt4': "La comisión no se puede fijar desde el código del EA: "
                   "en vivo la cobra el bróker según el esquema de la cuenta, "
                   "y en el probador la aplica la configuración del símbolo "
                   "y de la ficha de contrato, no el porcentaje del backtest.",
            'mt5': "La comisión no se puede fijar desde el código del EA: "
                   "en vivo la cobra el bróker según el esquema de la cuenta, "
                   "y en el probador la aplica la configuración del símbolo "
                   "y de la ficha de contrato, no el porcentaje del backtest.",
        },
        'consecuencia': {
            NIVEL_APROXIMADO:
                "El EA no impone la comisión del backtest: en vivo se cobra "
                "la del bróker (que puede no coincidir), y en el probador "
                "hay que configurarla a mano para que coincida con la del "
                "backtest. Con una comisión distinta, la curva de resultados "
                "cambiará.",
        },
    },
    'cuenta_netting': {
        'etiqueta': 'Varios setups sobre una cuenta netting',
        'motivos': {
            'mt5': "Cada setup se exporta a su propio EA con su magic "
                   "number, pero en una cuenta NETTING todas las órdenes del "
                   "mismo símbolo se funden en UNA posición, así que el "
                   "aislamiento por magic number no basta.",
        },
        'consecuencia': {
            NIVEL_APROXIMADO:
                "Hay que cargar los EAs en una cuenta HEDGING para que cada "
                "setup mantenga su posición por separado, como en el "
                "backtest. En netting se pisarán entre ellos.",
        },
    },

    # ── pendientes: el generador todavía no las emite (fase avanzada) ──
    'tramos_escalonados': {
        'etiqueta': 'Entrada escalonada por tramos',
        'motivo_comun': "El generador todavía no emite la entrada escalonada "
                        "(cada tramo recalcula su tamaño con su propia "
                        "distancia al stop y reajusta el precio medio); está "
                        "en la fase siguiente.",
        'consecuencia': {
            NIVEL_OMITIDO:
                "El código se genera con una ENTRADA ÚNICA por el total del "
                "riesgo del setup en vez de repartirla en tramos. El tamaño "
                "final se parece, pero el precio medio de entrada y el "
                "momento en que se alcanza son distintos.",
        },
    },
    'salidas_parciales': {
        'etiqueta': 'Salidas parciales por etapas',
        'pendiente': True,
        'motivo_comun': "El generador todavía no emite las etapas de salida "
                        "parcial (porcentajes sobre el tamaño original, con "
                        "disparadores por R, por condición o por "
                        "estancamiento); están en la fase siguiente.",
        'consecuencia': {
            NIVEL_OMITIDO:
                "La posición se cierra ENTERA con la primera salida en vez "
                "de por etapas. Eso cambia la curva: se pierden los cierres "
                "escalonados que recortaban las devoluciones.",
        },
    },
    'mecanismos_parciales': {
        'etiqueta': 'Cierre parcial asociado a un mecanismo de salida',
        'pendiente': True,
        'motivo_comun': "El generador todavía no emite los cierres parciales "
                        "por mecanismo ni su regla de disparo único (cada "
                        "mecanismo cierra parcialmente una sola vez por "
                        "posición y después cierra todo).",
        'consecuencia': {
            NIVEL_OMITIDO:
                "El stop, el take-profit, el break-even, el trailing y la "
                "salida por tiempo cerrarán el 100% de la posición en vez "
                "del porcentaje configurado.",
        },
    },
    'patrones_velas': {
        'etiqueta': 'Plantilla «Patrones de velas»',
        'pendiente': True,
        'bloquea_setup': True,
        'motivo_comun': "El generador todavía no emite el reconocimiento de "
                        "patrones de vela (son 32 formaciones geométricas con "
                        "su contexto de tendencia); están en la fase "
                        "siguiente.",
        'consecuencia': {
            NIVEL_OMITIDO:
                "Sin la señal no hay nada que generar, así que ESTE SETUP NO "
                "SE EXPORTA. Un archivo sin señal sería un robot que no abre "
                "una sola operación.",
        },
    },
    'zigzag': {
        'etiqueta': 'Indicador ZigZag en las reglas',
        'motivo_comun': "El generador todavía no emite el ZigZag no "
                        "repintante del Backtester, que confirma cada pivote "
                        "con retraso de media pierna.",
        'consecuencia': {
            NIVEL_OMITIDO:
                "La regla que lo usa forma parte de la señal, así que ESTE "
                "SETUP NO SE EXPORTA en vez de generar un archivo que opere "
                "por criterios distintos a los que backtesteaste.",
        },
    },
    'entrada_limite_fib': {
        'etiqueta': 'Entrada con orden límite en nivel de Fibonacci',
        'pendiente': True,
        'motivo_comun': "El generador todavía no emite la orden límite sobre "
                        "el retroceso de Fibonacci del ZigZag, ni sus cuatro "
                        "formas de cancelarla (tramo nuevo, señal contraria, "
                        "caducidad, avance a favor).",
        'consecuencia': {
            NIVEL_OMITIDO:
                "La entrada se genera A MERCADO al open de la vela "
                "siguiente, no esperando al retroceso. Entrará en sitios "
                "donde el backtest no llegó a entrar, y a peor precio.",
        },
    },
    'tf_superior': {
        'etiqueta': 'Filtro de tendencia de TF superior',
        'pendiente': True,
        'motivo_comun': "El generador todavía no resamplea la serie al "
                        "timeframe superior para calcular la tendencia de "
                        "fondo (en Pine sería request.security() y en MQL5 "
                        "un indicador sobre el TF mayor).",
        'consecuencia': {
            NIVEL_OMITIDO:
                "El código generado ignora el filtro: entrará también cuando "
                "el precio NO esté alineado con la tendencia del TF "
                "superior, por lo que generará más operaciones de las que "
                "backtesteaste.",
        },
    },
    'indicadores_editor_pendientes': {
        'etiqueta': 'Indicadores nuevos en las reglas sin port de runtime',
        'pendiente': True,
        'bloquea_setup': True,
        'motivo_comun': "El generador todavía no emite OBV ni VWAP (llegaron "
                        "al editor de reglas sin su port de runtime; requieren "
                        "volumen, que el backtest no transporta).",
        'consecuencia': {
            NIVEL_OMITIDO:
                "La regla que los usa forma parte de la señal, así que ESTE "
                "SETUP NO SE EXPORTA en vez de generar un archivo que opere "
                "por criterios distintos a los que backtesteaste.",
        },
    },
    'vwap': {
        'etiqueta': 'VWAP con anclaje y bandas',
        'motivo_comun': "El VWAP acumula hlc3·vol/vol por anclaje de "
                        "calendario y resetea en cada borde.",
        'consecuencia': {
            NIVEL_APROXIMADO:
                "En MetaTrader la alineación exacta barra/volumen del anclaje "
                "depende de la plataforma: el acumulado se detecta con "
                "iTime/iVolume y puede diferir del backtest en el momento "
                "exacto del reinicio. En TradingView es exacto (timeframe."
                "change + ta.vwap).",
        },
    },
}


# ══════════════ capacidades por plataforma ══════════════

# Solo se listan las características del catálogo de arriba. Lo que no aparece
# se considera 'exacto': los indicadores portados (ER, KAMA, SAR, percentil),
# el sizing por riesgo sobre equity y toda la gestión del núcleo (stop, TP,
# break-even, trailing, salida por tiempo).
#
# Las marcadas PENDIENTE bajarán a 'exacto' según la fase avanzada las vaya
# implementando; el test de cobertura inversa impide que ninguna se declare
# sin registrar antes un setup que la dispare.
_PENDIENTES_NUCLEO = {
    'salidas_parciales': NIVEL_OMITIDO,
    'mecanismos_parciales': NIVEL_OMITIDO,
    'patrones_velas': NIVEL_OMITIDO,
    'entrada_limite_fib': NIVEL_OMITIDO,
    'indicadores_editor_pendientes': NIVEL_OMITIDO,
}

CAPACIDADES = {
    'mt4': {
        'noticias': NIVEL_OMITIDO,
        'sesion_dst': NIVEL_APROXIMADO,
        'sesion_utc': NIVEL_APROXIMADO,
        'dias_semana': NIVEL_APROXIMADO,
        'hurst': NIVEL_OMITIDO,          # PENDIENTE (la plataforma sí puede)
        'vwap': NIVEL_APROXIMADO,
        'tf_superior': NIVEL_OMITIDO,    # PENDIENTE
        'relleno_open_siguiente': NIVEL_APROXIMADO,
        'redondeo_lotes': NIVEL_APROXIMADO,
        'atr_de_la_vela_de_entrada': NIVEL_APROXIMADO,
        'slippage_en_ticks': NIVEL_APROXIMADO,
        'comision': NIVEL_APROXIMADO,
        'stop_atr_dinamico': NIVEL_OMITIDO,
        **_PENDIENTES_NUCLEO,
    },
    'mt5': {
        'noticias': NIVEL_APROXIMADO,
        'sesion_dst': NIVEL_APROXIMADO,
        'sesion_utc': NIVEL_APROXIMADO,
        'dias_semana': NIVEL_APROXIMADO,
        'vwap': NIVEL_APROXIMADO,
        'tf_superior': NIVEL_OMITIDO,    # PENDIENTE
        'relleno_open_siguiente': NIVEL_APROXIMADO,
        'redondeo_lotes': NIVEL_APROXIMADO,
        'atr_de_la_vela_de_entrada': NIVEL_APROXIMADO,
        'slippage_en_ticks': NIVEL_APROXIMADO,
        'comision': NIVEL_APROXIMADO,
        'stop_atr_dinamico': NIVEL_OMITIDO,
        'cuenta_netting': NIVEL_APROXIMADO,
        **_PENDIENTES_NUCLEO,
    },
    'tradingview': {
        'noticias': NIVEL_OMITIDO,
        'sesion_dst': NIVEL_EXACTO,
        'sesion_utc': NIVEL_EXACTO,
        'dias_semana': NIVEL_EXACTO,
        'hurst': NIVEL_OMITIDO,          # aquí NO es pendiente: no cabe
        'tf_superior': NIVEL_OMITIDO,    # PENDIENTE (request.security)
        'relleno_open_siguiente': NIVEL_EXACTO,
        'atr_de_la_vela_de_entrada': NIVEL_APROXIMADO,
        'stop_atr_dinamico': NIVEL_OMITIDO,
        'slippage_en_ticks': NIVEL_APROXIMADO,
        **_PENDIENTES_NUCLEO,
    },
}

# El Hurst está omitido en Pine (no cabe en el presupuesto de ejecución por
# barra) y pendiente en MQL4 (la plataforma sí puede, pero el generador aún no
# lo emite para ese lenguaje). En MQL5 ya se porta entero (zcsHurst del
# runtime) y se considera exacto.
_PENDIENTE_POR_PLATAFORMA = {
    'hurst': ('mt4',),
}


def es_pendiente(plataforma, clave):
    """¿El ❌ se debe a que el generador aún no lo hace (True) o a que la
    plataforma no puede (False)? Cambia el texto del aviso, no el nivel."""
    cat = CARACTERISTICAS.get(clave, {})
    if clave in _PENDIENTE_POR_PLATAFORMA:
        return plataforma in _PENDIENTE_POR_PLATAFORMA[clave]
    return bool(cat.get('pendiente'))


# Plantillas cuyo IR usa indicadores sin port de runtime: el aviso de
# fidelidad las BLOQUEA al exportar, así que los emisores nunca las ven. La
# lista la comparten los tests de emisores (PLANTILLAS_EMITIBLES) para no
# parametrizar plantillas que jamás llegarían a emitirse.
PLANTILLAS_SIN_EMISOR = frozenset((
    'Patrones de velas',
))


def bloquea_setup(clave):
    """¿La ausencia de esta característica impide generar el archivo? Solo
    cuando lo que falta es la propia señal del setup."""
    return bool(CARACTERISTICAS.get(clave, {}).get('bloquea_setup'))


def nivel(plataforma, clave):
    """Nivel de soporte de una característica. Lo que no está en el catálogo
    de la plataforma se reproduce fielmente."""
    return CAPACIDADES.get(plataforma, {}).get(clave, NIVEL_EXACTO)


# ══════════════ análisis de un sistema ══════════════

def _caracteristicas_usadas(ir_st, cuenta):
    """Claves del catálogo que este setup activa de verdad, con el detalle
    concreto que se le enseñará al usuario ("sesión Nueva York", no
    "sesión"). `cuenta` es el dict de cuenta global del sistema, para los
    avisos que dependen de él (comisión)."""
    usadas = []
    filtros = ir_st['filtros']
    gestion = ir_st['gestion']

    if filtros['noticias']:
        n = filtros['noticias']
        usadas.append(('noticias',
                       f"±{n['minutos_antes']}/{n['minutos_despues']} min, "
                       f"impacto {n['impacto_minimo']}"))

    ses = filtros['sesion']
    if ses:
        if ses['tz']:
            usadas.append(('sesion_dst',
                           f"«{ses['tipo']}» {ses['hora_inicio']}:00-"
                           f"{ses['hora_fin']}:00 hora de {ses['tz']}"))
        else:
            usadas.append(('sesion_utc',
                           f"«{ses['tipo']}» {ses['hora_inicio']}:00-"
                           f"{ses['hora_fin']}:00 UTC"))

    if filtros['dias_semana'] is not None:
        dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        etiquetas = ", ".join(dias[d] for d in filtros['dias_semana']
                              if 0 <= d < 7)
        usadas.append(('dias_semana', etiquetas))

    reg = filtros['regimen']
    if reg and reg['metodo'] in METODOS_REGIMEN_HURST:
        usadas.append(('hurst',
                       f"método {reg['metodo']}, ventana {reg['periodo']}"))

    tfs = filtros['tf_superior']
    if tfs:
        usadas.append(('tf_superior',
                       f"{tfs.get('indicador')}({tfs.get('periodo')}) de "
                       f"{tfs.get('tf')}"))

    if gestion['entrada']['tipo'] == 'limite_fib':
        usadas.append(('entrada_limite_fib',
                       f"nivel {gestion['entrada']['nivel_fib']:g}"))

    # más de un tramo = escalonado de verdad; uno solo al 100% es el implícito
    # de cualquier setup y no hay nada que avisar
    if len(gestion['tramos']) > 1:
        reparto = " + ".join(f"{t['pct']:g}%" for t in gestion['tramos'])
        usadas.append(('tramos_escalonados', f"{len(gestion['tramos'])} tramos: {reparto}"))

    if _hay_parciales_reales(gestion['parciales']):
        etapas = gestion['parciales']
        reparto = " + ".join(f"{e['pct']:g}%" for e in etapas)
        usadas.append(('salidas_parciales', f"{len(etapas)} etapas: {reparto}"))

    if gestion['mecanismos']:
        nombres = ", ".join(sorted(gestion['mecanismos']))
        usadas.append(('mecanismos_parciales', nombres))

    if (gestion.get('stop_atr_modo') == 'dinamico_promedio'
            and gestion['stop_atr'] > 0.0):
        usadas.append(('stop_atr_dinamico',
                       f"stop {gestion['stop_atr']:g}×ATR sobre el precio medio"))

    if ir_st['patrones'] or ir_st['plantilla'] == 'Patrones de velas':
        usadas.append(('patrones_velas',
                       ", ".join(ir_st['patrones']) or '(ninguno elegido)'))

    if any(s['tipo'] == 'ZIGZAG' for s in ir_st['series']):
        usadas.append(('zigzag', ''))

    # Aroon/CMO/TRIX/StochRSI ya tienen port de runtime; OBV y VWAP siguen
    # pendientes (necesitan volumen y son de uso marginal en reglas)
    pendientes = ('OBV', 'VWAP')
    tipos_pendientes = [s['tipo'] for s in ir_st['series']
                        if s['tipo'] in pendientes]
    if tipos_pendientes:
        usadas.append(('indicadores_editor_pendientes',
                       ", ".join(sorted(set(tipos_pendientes)))))

    if any(s['tipo'] in ('VWAP_MEDIA', 'VWAP_SD', 'VWAP_SUP', 'VWAP_INF')
           for s in ir_st['series']):
        usadas.append(('vwap',
                       ", ".join(sorted({f"{s.get('anclaje', 'W')}·{s.get('k', 1.0):g}"
                                         for s in ir_st['series']
                                         if s['tipo'] in ('VWAP_MEDIA', 'VWAP_SD',
                                                          'VWAP_SUP', 'VWAP_INF')}))))

    # afectan a cualquier sistema, siempre
    usadas.append(('relleno_open_siguiente', ''))
    usadas.append(('redondeo_lotes', f"riesgo {gestion['riesgo_pct'] * 100:g}% por operación"))
    usadas.append(('atr_de_la_vela_de_entrada',
                   f"ATR({gestion['periodo_atr']})"))
    usadas.append(('slippage_en_ticks', ''))
    usadas.append(('comision', f"{cuenta.get('comision_pct', 0.0) * 100:g}% por lado"))
    return usadas


def _hay_parciales_reales(parciales):
    """¿Las etapas de salida configuran algo, o son la etapa implícita?

    Todo setup lleva por defecto UNA etapa al 100% disparada por la señal de
    la plantilla (etapa_salida_por_defecto): materializarla como etapa
    explícita es lo que permite editarla desde la GUI, pero el motor la trata
    igual que no tener ninguna. Avisar de eso sería ruido."""
    if len(parciales) > 1:
        return True
    if not parciales:
        return False
    etapa = parciales[0]
    return (etapa['pct'] < 100.0 or etapa['trigger'] != 'senal'
            or bool(etapa['condiciones']))


def analizar(ir_sistema, plataforma):
    """Avisos de fidelidad de un sistema para una plataforma.

    ir_sistema: lo que devuelve core.codegen.ir.ir_sistema()

    Devuelve una lista de dicts ordenada por gravedad:
      {'clave', 'nivel', 'icono', 'indice_setup', 'setup', 'etiqueta',
       'detalle', 'motivo', 'consecuencia'}

    Las características reproducidas fielmente NO generan aviso: un sistema
    sin filtros problemáticos devuelve lista vacía, para que la presencia de
    un aviso signifique siempre algo."""
    avisos = []
    for ir_st in ir_sistema['setups']:
        for clave, detalle in _caracteristicas_usadas(ir_st, ir_sistema['cuenta']):
            aviso = _aviso(plataforma, clave, detalle,
                           ir_st['indice'], ir_st['nombre'])
            if aviso is not None:
                avisos.append(aviso)

    # aviso de sistema, no de setup: solo tiene sentido con varios setups,
    # porque el problema es que compartan posición entre ellos
    if len(ir_sistema['setups']) > 1:
        aviso = _aviso(plataforma, 'cuenta_netting',
                       f"{len(ir_sistema['setups'])} setups", -1, None)
        if aviso is not None:
            avisos.append(aviso)

    avisos.sort(key=lambda a: (_ORDEN_NIVEL[a['nivel']], a['indice_setup'],
                               a['clave']))
    return avisos


def _aviso(plataforma, clave, detalle, indice_setup, nombre_setup):
    """Un aviso ya redactado, o None si la plataforma reproduce eso fielmente."""
    nvl = nivel(plataforma, clave)
    if nvl == NIVEL_EXACTO:
        return None
    cat = CARACTERISTICAS[clave]
    motivo = cat.get('motivos', {}).get(plataforma) or cat.get('motivo_comun', '')
    return {
        'clave': clave,
        'nivel': nvl,
        'icono': ICONOS[nvl],
        'pendiente': es_pendiente(plataforma, clave),
        'bloquea_setup': bloquea_setup(clave) and nvl == NIVEL_OMITIDO,
        'indice_setup': indice_setup,
        'setup': nombre_setup,
        'etiqueta': cat['etiqueta'],
        'detalle': detalle,
        'motivo': motivo,
        'consecuencia': cat['consecuencia'].get(nvl, ''),
    }


def setups_bloqueados(avisos):
    """Índices de los setups que NO se pueden exportar porque lo que falta es
    su propia señal. exportar_sistema() los salta en vez de escribir un
    archivo que no abriría una sola operación."""
    return sorted({a['indice_setup'] for a in avisos if a['bloquea_setup']})


def nivel_global(avisos):
    """El peor nivel de una lista de avisos, o 'exacto' si está vacía. Es lo
    que decide el distintivo de la tarjeta de plataforma y si el diálogo de
    confirmación tiene que exigir la casilla de conformidad."""
    if any(a['nivel'] == NIVEL_OMITIDO for a in avisos):
        return NIVEL_OMITIDO
    if any(a['nivel'] == NIVEL_APROXIMADO for a in avisos):
        return NIVEL_APROXIMADO
    return NIVEL_EXACTO


def hay_omisiones(avisos):
    return any(a['nivel'] == NIVEL_OMITIDO for a in avisos)


# ══════════════ redacción de los avisos ══════════════

def texto_aviso(aviso, con_setup=True):
    """Una línea legible por aviso. Se usa tal cual en el diálogo, en la
    cabecera del código generado y en NOTAS_DE_FIDELIDAD.md, para que los
    tres digan literalmente lo mismo."""
    partes = [aviso['icono'], f"{aviso['etiqueta']}"]
    if aviso['detalle']:
        partes.append(f"({aviso['detalle']})")
    cabeza = " ".join(partes)
    # los avisos de sistema (cuenta_netting) no cuelgan de ningún setup
    if con_setup and aviso['setup'] is not None:
        cabeza = f"Setup {aviso['indice_setup']} «{aviso['setup']}» — {cabeza}"
    cuerpo = " ".join(x for x in (aviso['motivo'], aviso['consecuencia']) if x)
    return f"{cabeza}: {cuerpo}" if cuerpo else cabeza


def bloque_notas(avisos, plataforma_nombre, ancho=76):
    """Lista de líneas (sin marcas de comentario) para la cabecera del archivo
    generado. El emisor les antepone el prefijo de comentario de su lenguaje."""
    if not avisos:
        return [f"Fidelidad: este código reproduce el sistema backtesteado "
                f"sin omisiones conocidas en {plataforma_nombre}."]
    lineas = [f"NOTAS DE FIDELIDAD — {plataforma_nombre}",
              "",
              "Lo que sigue NO se reproduce igual que en el backtest del",
              "Analytics System. Leerlo antes de operar con dinero real:",
              ""]
    for aviso in avisos:
        for k, linea in enumerate(_envolver(texto_aviso(aviso), ancho - 2)):
            lineas.append(("  " if k else "- ") + linea)
    return lineas


def _envolver(texto, ancho):
    """Ajuste de línea sin dependencias: textwrap parte por caracteres y aquí
    basta con partir por palabras."""
    palabras, lineas, actual = texto.split(), [], ""
    for palabra in palabras:
        if actual and len(actual) + 1 + len(palabra) > ancho:
            lineas.append(actual)
            actual = palabra
        else:
            actual = f"{actual} {palabra}" if actual else palabra
    if actual:
        lineas.append(actual)
    return lineas or [""]


def _sin_acentos(texto):
    """Pliega los acentos a ASCII. El aviso en ejecución acaba dentro de un
    Print() de MQL, y MetaEditor guarda el archivo en ANSI o en UTF-8 según la
    versión: un mensaje de emergencia no se puede permitir salir en mojibake
    justo cuando es lo único que avisa de que el código no reproduce el
    backtest."""
    descompuesto = unicodedata.normalize('NFKD', texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def texto_runtime(avisos):
    """Frase única que el código generado imprime al arrancar. Solo recoge
    las omisiones: el aviso en ejecución tiene que caber en una línea de log y
    lo aproximado ya está en la cabecera del archivo."""
    omitidos = [a for a in avisos if a['nivel'] == NIVEL_OMITIDO]
    if not omitidos:
        return ""
    partes = sorted({_sin_acentos(a['etiqueta'].lower()) for a in omitidos})
    return ("AVISO DE FIDELIDAD: este codigo se genero SIN "
            + "; ".join(partes)
            + ". No reproduce el backtest del Analytics System.")
