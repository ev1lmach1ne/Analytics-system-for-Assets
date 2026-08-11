"""
core/codegen/__init__.py
Exportación del sistema del Backtester a código de plataformas de trading.

El paquete se divide en tres capas para que añadir una plataforma nueva no
obligue a volver a interpretar la configuración de los setups:

  ir.py         los dicts de setup -> árbol de nodos neutro (sin lenguaje)
  fidelidad.py  qué sabe hacer cada plataforma y qué se pierde al traducir
  base/pine/mql los emisores, que solo saben escribir nodos en su lenguaje

Los ficheros de runtime/ NO son Python: son las librerías de indicadores
portadas a cada lenguaje, que el emisor copia junto al código generado. Viven
bajo core/ a propósito, porque empaquetar.py hace
`--add-data "<PROJECT>/core;core"` y así entran en el .exe sin tocar el spec.
"""
import os

from core.codegen import fidelidad, ir
from core.codegen.mql import EmisorMQL5
from core.codegen.pine import EmisorPine

# ══════════════ registro de plataformas ══════════════

# Se listan también las que aún no se pueden generar: la cuadrícula de la GUI
# las enseña deshabilitadas, que es la forma honesta de comunicar hasta dónde
# llega hoy la función sin dejar al usuario adivinando.
PLATAFORMAS = [
    {'clave': 'tradingview', 'nombre': 'TradingView',
     'lenguaje': 'Pine Script v6', 'emisor': EmisorPine, 'estado': 'disponible'},
    {'clave': 'mt5', 'nombre': 'MetaTrader 5',
     'lenguaje': 'MQL5', 'emisor': EmisorMQL5, 'estado': 'disponible'},
    {'clave': 'mt4', 'nombre': 'MetaTrader 4',
     'lenguaje': 'MQL4', 'emisor': None, 'estado': 'proximamente'},
    {'clave': 'ninjatrader', 'nombre': 'NinjaTrader',
     'lenguaje': 'NinjaScript (C#)', 'emisor': None, 'estado': 'proximamente'},
    {'clave': 'ctrader', 'nombre': 'cTrader',
     'lenguaje': 'C# (.NET)', 'emisor': None, 'estado': 'proximamente'},
    {'clave': 'tradestation', 'nombre': 'TradeStation',
     'lenguaje': 'EasyLanguage', 'emisor': None, 'estado': 'proximamente'},
    {'clave': 'multicharts', 'nombre': 'MultiCharts',
     'lenguaje': 'PowerLanguage', 'emisor': None, 'estado': 'proximamente'},
    {'clave': 'quantower', 'nombre': 'Quantower',
     'lenguaje': 'C# (.NET)', 'emisor': None, 'estado': 'proximamente'},
    {'clave': 'prorealtime', 'nombre': 'ProRealTime',
     'lenguaje': 'ProBuilder / ProOrder', 'emisor': None,
     'estado': 'proximamente'},
    {'clave': 'motivewave', 'nombre': 'MotiveWave',
     'lenguaje': 'Java SDK', 'emisor': None, 'estado': 'proximamente'},
]


def plataforma(clave):
    for p in PLATAFORMAS:
        if p['clave'] == clave:
            return p
    raise ValueError(f"Plataforma desconocida: {clave!r}")


def plataformas_disponibles():
    return [p['clave'] for p in PLATAFORMAS if p['estado'] == 'disponible']


# ══════════════ análisis previo (lo que consulta la GUI) ══════════════

def analizar_sistema(setups, config_global, claves):
    """Avisos de fidelidad por plataforma, ANTES de escribir nada.

    Devuelve {clave: {'avisos', 'nivel', 'bloqueados'}}. Es lo que alimenta el
    distintivo de cada tarjeta y el diálogo de confirmación: el usuario tiene
    que poder ver qué se pierde en cada plataforma antes de elegirla, no
    después de que los archivos estén en disco."""
    sistema = ir.ir_sistema(setups, config_global)
    fuera = {}
    for clave in claves:
        avisos = fidelidad.analizar(sistema, clave)
        fuera[clave] = {
            'avisos': avisos,
            'nivel': fidelidad.nivel_global(avisos),
            'bloqueados': fidelidad.setups_bloqueados(avisos),
        }
    return fuera


# ══════════════ escritura ══════════════

def codigo_de_setup(setups, config_global, clave, indice, meta=None):
    """El código de UN setup para UNA plataforma, sin tocar el disco.

    Existe por TradingView: Pine Script no se importa desde un archivo, se
    pega. Escribir un .pine para que el usuario lo abra con el bloc de notas y
    lo copie es un rodeo; con esto la GUI puede poner el código directamente
    en el portapapeles."""
    info = plataforma(clave)
    if info['emisor'] is None:
        raise ValueError(f"{info['nombre']} todavía no tiene emisor.")
    meta = dict(meta or {})
    meta.setdefault('sistema', 'sistema')
    sistema = ir.ir_sistema(setups, config_global)
    avisos = fidelidad.analizar(sistema, clave)
    if indice in fidelidad.setups_bloqueados(avisos):
        raise ValueError(
            f"El setup {indice} no se puede generar en {info['nombre']}: "
            f"falta su propia señal.")
    ir_setup = sistema['setups'][indice]
    return info['emisor']().construir(ir_setup, sistema, avisos, meta)


def exportar_sistema(setups, config_global, claves, destino, nombre,
                     meta=None):
    """Genera el código del sistema para cada plataforma pedida.

    destino: carpeta donde se crea <slug>/ con el árbol de cada plataforma.
    meta: {'sistema', 'activo', 'tf'} — el activo y la temporalidad con los
    que se corrió el backtest, que viajan a la cabecera y a las guardas.

    Devuelve {'carpeta', 'archivos', 'avisos', 'bloqueados'}.

    Un setup cuya SEÑAL no se puede emitir (patrones de vela, reglas con
    ZigZag) no genera archivo: un robot sin señal no abriría una sola
    operación, y entregar ese archivo sería peor que no entregarlo. Se
    devuelve en 'bloqueados' y se explica en las notas."""
    meta = dict(meta or {})
    meta.setdefault('sistema', nombre)
    sistema = ir.ir_sistema(setups, config_global)
    carpeta = os.path.join(destino, _slug(nombre))

    archivos = {}
    avisos_por_plataforma = {}
    bloqueados_por_plataforma = {}

    for clave in claves:
        info = plataforma(clave)
        if info['emisor'] is None:
            raise ValueError(
                f"{info['nombre']} todavía no tiene emisor: no debería poder "
                f"seleccionarse en la cuadrícula.")
        emisor = info['emisor']()
        avisos = fidelidad.analizar(sistema, clave)
        bloqueados = fidelidad.setups_bloqueados(avisos)
        avisos_por_plataforma[clave] = avisos
        bloqueados_por_plataforma[clave] = bloqueados

        archivos.update(emisor.archivos_comunes(sistema, avisos, meta))
        for ir_setup in sistema['setups']:
            if ir_setup['indice'] in bloqueados:
                continue
            archivos.update(
                emisor.archivos_setup(ir_setup, sistema, avisos, meta))

    archivos['NOTAS_DE_FIDELIDAD.md'] = _notas_md(
        sistema, meta, claves, avisos_por_plataforma,
        bloqueados_por_plataforma)

    for ruta_rel, texto in archivos.items():
        ruta = os.path.join(carpeta, ruta_rel)
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write(texto)

    return {'carpeta': carpeta, 'archivos': sorted(archivos),
            'avisos': avisos_por_plataforma,
            'bloqueados': bloqueados_por_plataforma}


# ══════════════ notas de fidelidad ══════════════

def _notas_md(sistema, meta, claves, avisos_por_plataforma, bloqueados):
    """El informe que acompaña a los archivos. Dice lo mismo que el diálogo y
    que la cabecera de cada archivo, a propósito: si los tres textos
    divergieran, el usuario no sabría a cuál creer."""
    cuenta = sistema['cuenta']
    lineas = [
        f"# Notas de fidelidad — {meta.get('sistema', 'sistema')}",
        "",
        "Este código lo ha generado el Analytics System a partir de un "
        "backtest. Aquí está, plataforma por plataforma, en qué se diferencia "
        "de lo que se probó.",
        "",
        "## El backtest",
        "",
        f"- Activo: **{meta.get('activo', '?')}**",
        f"- Temporalidad: **{meta.get('tf', '?')}**",
        f"- Capital inicial: {cuenta['capital_inicial']:,.0f}",
        f"- Comisión: {cuenta['comision_pct'] * 100:g}% por lado",
        f"- Slippage: {cuenta['slippage_pct'] * 100:g}%",
        f"- Setups: {len(sistema['setups'])}",
        "",
        "Los parámetros están ajustados a ESE activo y ESA temporalidad. En "
        "otro mercado o marco temporal no significan lo mismo, y el código "
        "generado avisa si lo cargas en otro sitio.",
        "",
    ]

    for clave in claves:
        info = plataforma(clave)
        avisos = avisos_por_plataforma[clave]
        lineas += [f"## {info['nombre']} ({info['lenguaje']})", ""]

        for indice in bloqueados[clave]:
            nombre_setup = sistema['setups'][indice]['nombre']
            lineas.append(
                f"> **El setup {indice} «{nombre_setup}» NO se ha "
                f"exportado.** Lo que falta es su propia señal, así que el "
                f"archivo habría sido un robot incapaz de abrir una sola "
                f"operación.")
            lineas.append("")

        if not avisos:
            lineas += ["Sin diferencias conocidas.", ""]
            continue

        for nivel, titulo in ((fidelidad.NIVEL_OMITIDO, "No se reproduce"),
                              (fidelidad.NIVEL_APROXIMADO, "Se aproxima")):
            del_nivel = [a for a in avisos if a['nivel'] == nivel]
            if not del_nivel:
                continue
            lineas += [f"### {fidelidad.ICONOS[nivel]} {titulo}", ""]
            for aviso in del_nivel:
                lineas.append(f"- {fidelidad.texto_aviso(aviso)}")
            lineas.append("")

    lineas += [
        "---",
        "",
        "## Antes de operar con dinero real",
        "",
        "1. Compila el archivo en su plataforma y compruébalo en su propio "
        "probador de estrategias.",
        "2. Compara el número de operaciones y la curva con las del backtest. "
        "No van a coincidir operación a operación (spread del bróker, "
        "redondeo de lotes, semántica de relleno), pero una divergencia "
        "grande señala un problema.",
        "3. Repasa la lista de arriba y decide si lo que se pierde te importa.",
    ]
    return "\n".join(lineas) + "\n"


def _slug(nombre):
    """Nombre de carpeta seguro. Mismo criterio que _slug_sistema de la GUI,
    reimplementado aquí para que el paquete no dependa de gui/."""
    import re
    return re.sub(r'[^\w\-]+', '_', str(nombre).strip().lower()).strip('_') \
        or 'sistema'
