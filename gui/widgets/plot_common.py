"""
Piezas compartidas para los gráficos matplotlib embebidos en la GUI.

Hasta ahora cada pestaña llevaba su propia copia de la paleta y de `_style_ax`
(tab_backtest, tab_comparador y tab_patrones son idénticas salvo detalles). Este
módulo las centraliza y añade `BarraGrafico`, la barra de herramientas del
canvas.

No importa nada de `gui.widgets.tab_*` a propósito: es el módulo de más abajo en
la jerarquía, para que cualquier pestaña pueda usarlo sin ciclos de import.
"""
import functools
import traceback

import numpy as np
import pandas as pd

from PyQt6.QtCore import (Qt, QSize, QPointF, QPoint, QRect, QRectF, QEvent,
                          QPropertyAnimation, QEasingCurve)
from PyQt6.QtGui import (QIcon, QPixmap, QPainter, QPen, QColor, QPolygonF,
                         QLinearGradient)
from PyQt6.QtWidgets import (QSizePolicy, QLabel, QDialog, QTabWidget, QVBoxLayout,
                             QHBoxLayout, QApplication, QWidget, QFrame, QPushButton,
                             QGraphicsDropShadowEffect, QGraphicsOpacityEffect)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.dates import date2num, AutoDateLocator, ConciseDateFormatter
from matplotlib.ticker import Locator, FuncFormatter

# ══════════════ paleta ══════════════
# Misma que la del Backtester (gui/widgets/tab_backtest.py): el Analizador se
# integra en ese lenguaje visual, no en el del PDF (que usa grises neutros).
FIG_BG = '#0d1424'
AX_FG = '#c8d6e5'
GRID_C = '#253a60'
VERDE = '#2ecc71'
ROJO = '#e74c3c'
GRIS = '#5a7a9a'
AMBAR = '#f1c40f'
AZUL = '#4fc3f7'
NARANJA = '#ff9900'
MAGENTA = '#9b59b6'

TEXTO_TENUE = '#5a7a9a'

# Colores de régimen ER, heredados del PDF para que ambas vistas coincidan.
ER_TENDENCIA = '#00d4aa'
ER_TRANSICION = '#8b949e'
ER_RUIDO = '#f85149'


def make_canvas(alto_min=260, figsize=(9, 4.2)):
    """Figura + canvas ya tematizados y expansibles."""
    fig = Figure(figsize=figsize, facecolor=FIG_BG)
    canvas = FigureCanvasQTAgg(fig)
    canvas.setStyleSheet("background-color: transparent;")
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    canvas.setMinimumHeight(alto_min)
    return fig, canvas


def style_ax(ax):
    """Aplica el tema oscuro de la app a un eje."""
    ax.set_facecolor(FIG_BG)
    ax.tick_params(colors=AX_FG, labelsize=7)
    ax.xaxis.label.set_color(AX_FG)
    ax.yaxis.label.set_color(AX_FG)
    ax.title.set_color('#e6edf3')
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(True, alpha=0.25, color=GRID_C, linewidth=0.5)


def eje_fechas(ax, x=None):
    """Etiquetas de fecha compactas y sin solapes.

    Con el formateador por defecto, un panel estrecho (los cuadrantes del
    dashboard NATR o el mini-histograma de una tarjeta de patrón, por ejemplo)
    pinta fechas completas superpuestas. ConciseDateFormatter reparte el
    año/mes/día entre el eje y el offset.

    `x` es opcional: si se pasa un array, no se hace nada cuando no es de tipo
    datetime64 (los ejes numéricos no llevan formateador de fechas). Si se
    omite, se asume que el eje ya está en unidades de fecha (date2num).
    """
    if x is not None and not np.issubdtype(np.asarray(x).dtype, np.datetime64):
        return
    loc = AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
    ax.xaxis.get_offset_text().set_color(AX_FG)


# ══════════ eje X ordinal (índice de vela) con etiquetas de fecha ══════════
# Escalones "redondos" de menor a mayor, con la duración aproximada de cada uno
# en segundos: se recorren en orden y se elige el primero que deja como mucho
# `max_ticks` marcas en la ventana visible.
_PASOS_ORDINALES = [
    ('minuto', 1), ('minuto', 5), ('minuto', 15), ('minuto', 30),
    ('hora', 1), ('hora', 3), ('hora', 6), ('hora', 12),
    ('dia', 1), ('semana', 1),
    ('mes', 1), ('mes', 3), ('mes', 6),
    ('anio', 1), ('anio', 2), ('anio', 5), ('anio', 10),
]
_SEGUNDOS_PASO = {'minuto': 60.0, 'hora': 3600.0, 'dia': 86400.0,
                  'semana': 604800.0, 'mes': 2629800.0, 'anio': 31557600.0}


def _bordes_periodo(unidad, k, t0, t1):
    """Timestamps de inicio de cada periodo de `k` `unidad` que cae en [t0, t1].

    Las unidades de calendario (mes/año) usan DateOffset porque su duración no
    es fija; las demás parten de medianoche del primer día, con lo que las
    marcas caen siempre en horas/minutos redondos sin depender de alias de
    frecuencia de pandas (que han ido cambiando entre versiones).
    """
    if unidad == 'anio':
        ini = pd.Timestamp(year=t0.year, month=1, day=1)
        return pd.date_range(ini, t1, freq=pd.DateOffset(years=k))
    if unidad == 'mes':
        ini = pd.Timestamp(year=t0.year, month=1, day=1)
        return pd.date_range(ini, t1, freq=pd.DateOffset(months=k))
    if unidad == 'semana':
        lunes = t0.normalize() - pd.Timedelta(days=int(t0.weekday()))
        return pd.date_range(lunes, t1, freq=pd.Timedelta(days=7 * k))
    if unidad == 'dia':
        return pd.date_range(t0.normalize(), t1, freq=pd.Timedelta(days=k))
    paso = (pd.Timedelta(hours=k) if unidad == 'hora'
            else pd.Timedelta(minutes=k))
    return pd.date_range(t0.normalize(), t1, freq=paso)


class _LocalizadorOrdinalFechas(Locator):
    """Marcas del eje en índices de vela, colocadas en fronteras de calendario.

    Un MaxNLocator sobre el índice repartiría las marcas a intervalos regulares
    de VELAS, con lo que las fechas que se leen debajo serían arbitrarias ("14
    mar", "2 jun"...) y cambiarían al hacer scroll. Aquí se elige primero el
    escalón de calendario y luego se busca qué vela abre cada periodo, que es
    lo que produce etiquetas estables tipo "mar", "abr", "2025".
    """

    def __init__(self, ts, max_ticks=6):
        self.ts = ts
        self.max_ticks = max_ticks
        self.unidad = 'dia'   # lo consulta el formateador para elegir patrón
        # ningún escalón por debajo del espaciado real entre velas: con 3 velas
        # diarias en pantalla, un escalón de 12 h pondría marcas en instantes
        # donde no hay ninguna vela.
        #
        # Vía TimedeltaIndex y no vía los enteros crudos del índice: `asi8`
        # devuelve el valor en la unidad del propio DatetimeIndex (ns, us...),
        # que no es la misma en todas las versiones de pandas, y dividir por
        # 1e9 a ciegas daba un espaciado 1000 veces menor del real.
        if len(ts) > 1:
            paso_barra = float(np.median((ts[1:] - ts[:-1]).total_seconds()))
        else:
            paso_barra = 0.0
        self.segundos_barra = paso_barra

    def __call__(self):
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin, vmax):
        ts = self.ts
        n = len(ts)
        if n == 0:
            return []
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        i0 = int(np.clip(np.ceil(vmin), 0, n - 1))
        i1 = int(np.clip(np.floor(vmax), 0, n - 1))
        if i1 <= i0:
            return [float(i0)]
        t0, t1 = ts[i0], ts[i1]
        span = (t1 - t0).total_seconds()
        unidad, k = _PASOS_ORDINALES[-1]
        for u, mult in _PASOS_ORDINALES:
            dur = _SEGUNDOS_PASO[u] * mult
            if dur < self.segundos_barra:
                continue
            if span / dur <= self.max_ticks:
                unidad, k = u, mult
                break
        self.unidad = unidad
        bordes = _bordes_periodo(unidad, k, t0, t1)
        if len(bordes) == 0:
            return [float(i0), float(i1)]
        # la vela que ABRE cada periodo: la primera cuyo timestamp alcanza el
        # borde. Varios bordes seguidos sin velas (un mes entero sin datos)
        # colapsan en el mismo índice, de ahí el unique. Se busca sobre el
        # propio DatetimeIndex (no sobre sus enteros crudos) para que pandas
        # concilie las unidades de los dos índices.
        idx = ts.searchsorted(bordes, side='left')
        idx = np.unique(idx[(idx >= i0) & (idx <= i1)])
        if len(idx) == 0:
            return [float(i0), float(i1)]
        return [float(i) for i in idx]


def _formateador_ordinal(ts, locator):
    """Índice de vela → etiqueta corta, promocionando la unidad en los cambios
    de periodo (el año en el cambio de año, el mes en el cambio de mes...), que
    es el mismo criterio de ConciseDateFormatter: la marca dice lo mínimo para
    situarse, y el contexto lo da la marca anterior."""

    def _fmt(valor, _pos=None):
        n = len(ts)
        i = int(round(valor))
        if n == 0 or i < 0 or i >= n:
            return ''
        t = ts[i]
        previo = ts[i - 1] if i > 0 else None
        unidad = locator.unidad
        if unidad == 'anio':
            return t.strftime('%Y')
        if unidad == 'mes':
            return t.strftime('%Y') if t.month == 1 else t.strftime('%b')
        if unidad in ('dia', 'semana'):
            if previo is None or t.year != previo.year:
                return t.strftime('%Y')
            if t.month != previo.month:
                return t.strftime('%b')
            return t.strftime('%d')
        if previo is None or t.date() != previo.date():
            return t.strftime('%d %b')
        return t.strftime('%H:%M')

    return FuncFormatter(_fmt)


def eje_fechas_ordinal(ax, ts, max_ticks=6):
    """Eje X en índice de vela (0, 1, 2...) con etiquetas de fecha.

    Alternativa a `eje_fechas` para gráficos de velas: al dibujar contra el
    índice y no contra la fecha, los tramos sin mercado (fin de semana,
    festivos, cierres de sesión) dejan de ocupar espacio y la serie se ve
    continua, como en TradingView. La contrapartida es que el eje ya no es
    lineal en tiempo, así que no sirve para leer duraciones a ojo.

    Devuelve (locator, formatter) para poder inspeccionarlos en tests.
    """
    ts = pd.DatetimeIndex(ts)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    loc = _LocalizadorOrdinalFechas(ts, max_ticks=max_ticks)
    fmt = _formateador_ordinal(ts, loc)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.get_offset_text().set_color(AX_FG)
    return loc, fmt


def ax_placeholder(ax, texto):
    """Deja el eje vacío con un mensaje centrado (sección sin datos)."""
    ax.text(0.5, 0.5, texto, ha='center', va='center',
            color=TEXTO_TENUE, fontsize=9, transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(FIG_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)


def leyenda(ax, **kwargs):
    """Leyenda con el estilo de la app (fondo del panel, texto claro)."""
    kwargs.setdefault('fontsize', 7)
    kwargs.setdefault('framealpha', 0.4)
    return ax.legend(facecolor='#141e30', edgecolor=GRID_C,
                     labelcolor=AX_FG, **kwargs)


def fmt(v, dec=2, sufijo=''):
    """Número → texto, con guion para None e ∞ para infinito."""
    if v is None:
        return '—'
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:                      # NaN
        return '—'
    if f in (float('inf'), float('-inf')):
        return '∞' if f > 0 else '-∞'
    return f"{f:,.{dec}f}{sufijo}"


def agregar_crosshair(fig, ejes, x, series, texto_fn, nombre='default',
                      colores=None, linestyle='solid', horizontal=False,
                      x_texto_fn=None):
    """Crosshair vertical sincronizado + un punto marcado sobre cada serie +
    un tooltip independiente por panel, sobre varios ejes apilados que
    comparten X.

    `series`: lista de arrays Y (uno por eje de `ejes`, mismo orden/longitud
    que `x`) — el punto marcado en cada panel se coloca en (x[idx], series[k][idx]).
    `texto_fn(idx)` devuelve una lista con un texto por eje (mismo orden que
    `ejes`) — usa `None` o `''` en la posición de un eje para no mostrar
    tooltip ahí (el crosshair y el punto marcado se siguen dibujando en todos
    los ejes igualmente, solo cambia si ese panel muestra o no su propio
    bocadillo). `colores`: lista opcional de colores (uno por eje) para el
    punto marcado — por defecto AX_FG si no se especifica; se recomienda
    pasar el mismo color que ya usa cada línea para que se lea a qué serie
    pertenece cada punto.

    `linestyle`: estilo de la línea vertical ('solid' por defecto; p.ej.
    '--' para discontinua). `horizontal=True` añade además, en cada eje,
    una línea horizontal discontinua al nivel del punto marcado (el valor
    de la serie en esa x). `x_texto_fn(idx)`: etiqueta alternativa para el
    borde inferior de cada panel (por defecto: fecha formateada, o
    "Barra nº N" cuando el eje X no son fechas).

    Usa blitting (como el arrastre del gráfico principal en tab_backtest.py)
    en vez de draw_idle() en cada movimiento: con series largas (p.ej. TF de
    1 minuto, decenas/cientos de miles de puntos) un draw() completo cuesta
    cientos de ms, y disparar uno por cada píxel que recorre el ratón hace el
    hover visiblemente lento. Aquí solo se recomponen los artistas pequeños
    del crosshair sobre un bitmap de fondo cacheado.

    El fondo se cachea al vuelo escuchando 'draw_event' (se dispara solo tras
    CUALQUIER draw() completo: el primer pintado, un resize, un zoom/pan de
    la barra de herramientas...), no con una captura explícita aquí — así se
    resincroniza automáticamente sin necesitar un handler de resize dedicado.
    Limitación aceptada (igual que en el ejemplo de blitting de la propia
    documentación de matplotlib): si un draw_event ocurre con el cursor ya
    encima del gráfico, el crosshair de ese instante queda "horneado" en el
    fondo cacheado hasta el siguiente movimiento de ratón, que lo corrige.

    El estado (cids de mpl_connect) se guarda en `fig` (no en canvas): fig
    persiste entre repintados (solo se llama fig.clear(), nunca se recrea),
    así que es el ancla correcta para desconectar el callback anterior antes
    de reconectar uno nuevo en cada redibujado — necesario porque fig.clear()
    destruye los ejes que el callback viejo capturó por closure.
    """
    x_arr = np.asarray(x)
    if not ejes or x_arr.size == 0 or len(series) != len(ejes):
        return
    canvas = fig.canvas
    cid_attr = f'_crosshair_cids_{nombre}'
    for cid in getattr(fig, cid_attr, ()):
        try:
            canvas.mpl_disconnect(cid)
        except Exception:
            pass

    es_fecha = np.issubdtype(x_arr.dtype, np.datetime64)
    x_num = date2num(x_arr) if es_fecha else x_arr.astype(float)
    series = [np.asarray(s, dtype=float) for s in series]
    colores = colores or [AX_FG] * len(ejes)

    patron_fecha = None
    if es_fecha:
        paso = x_arr[1] - x_arr[0] if len(x_arr) > 1 else np.timedelta64(0, 'D')
        patron_fecha = '%Y-%m-%d %H:%M' if paso < np.timedelta64(1, 'D') else '%Y-%m-%d'

    lineas_v = [axx.axvline(x_num[0], color=GRIS, linewidth=0.7, alpha=0.7,
                             linestyle=linestyle, zorder=98, visible=False)
                for axx in ejes]
    lineas_h = []
    if horizontal:
        lineas_h = [axx.axhline(0.0, color=c, linewidth=0.7, alpha=0.7,
                                linestyle='--', zorder=98, visible=False)
                    for axx, c in zip(ejes, colores)]
    puntos = [axx.plot([], [], marker='o', markersize=5, color=c,
                       markeredgecolor=FIG_BG, markeredgewidth=0.8,
                       zorder=99, linestyle='none', visible=False)[0]
              for axx, c in zip(ejes, colores)]
    annots = [axx.annotate("", xy=(0, 0), xytext=(12, -12), textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.4", fc=FIG_BG, ec=GRID_C, alpha=0.95),
                           color=AX_FG, fontsize=7, zorder=100, annotation_clip=False)
              for axx in ejes]
    for a in annots:
        a.set_visible(False)
    # etiqueta de fecha pegada al borde inferior de CADA panel (no un valor de
    # serie): xycoords mezcla 'data' en X (sigue al crosshair) con 'axes
    # fraction' en Y (0 = línea inferior de ese eje) — así no hace falta
    # conocer el rango Y de cada panel, distinto en los 3 (precio/equity×/dd%).
    etiquetas_x = [axx.annotate("", xy=(x_num[0], 0), xycoords=('data', 'axes fraction'),
                                xytext=(0, -4), textcoords='offset points',
                                ha='center', va='top',
                                bbox=dict(boxstyle='round,pad=0.3', fc=GRID_C, ec=AX_FG, alpha=0.95),
                                color=AX_FG, fontsize=6.5, zorder=101, annotation_clip=False)
                  for axx in ejes]
    for et in etiquetas_x:
        et.set_visible(False)

    estado = {'bg': None}

    def _capturar_fondo(_event=None):
        estado['bg'] = canvas.copy_from_bbox(fig.bbox)

    def _pintar_dinamicos():
        if estado['bg'] is None:
            # todavía no hubo ningún draw_event (arranque en frío) — recurso
            # de respaldo, se autocorrige en cuanto llegue el primero.
            canvas.draw_idle()
            return
        canvas.restore_region(estado['bg'])
        for l in lineas_v:
            l.axes.draw_artist(l)
        for l in lineas_h:
            l.axes.draw_artist(l)
        for p in puntos:
            p.axes.draw_artist(p)
        for a in annots:
            if a.get_visible():
                a.axes.draw_artist(a)
        for et in etiquetas_x:
            if et.get_visible():
                et.axes.draw_artist(et)
        canvas.blit(fig.bbox)

    def _ocultar():
        cambio = (any(a.get_visible() for a in annots)
                  or any(l.get_visible() for l in lineas_v)
                  or any(l.get_visible() for l in lineas_h)
                  or any(p.get_visible() for p in puntos)
                  or any(et.get_visible() for et in etiquetas_x))
        for l in lineas_v:
            l.set_visible(False)
        for l in lineas_h:
            l.set_visible(False)
        for p in puntos:
            p.set_visible(False)
        for a in annots:
            a.set_visible(False)
        for et in etiquetas_x:
            et.set_visible(False)
        if cambio:
            _pintar_dinamicos()

    def _on_move(event):
        if event.inaxes not in ejes or event.xdata is None:
            _ocultar()
            return
        i = int(np.clip(np.searchsorted(x_num, event.xdata), 1, len(x_num) - 1))
        if (x_num[i] - event.xdata) > (event.xdata - x_num[i - 1]):
            i -= 1
        xv = x_num[i]
        for l in lineas_v:
            l.set_xdata([xv, xv])
            l.set_visible(True)
        for l, s in zip(lineas_h, series):
            l.set_ydata([s[i], s[i]])
            l.set_visible(True)
        for p, s in zip(puntos, series):
            p.set_data([xv], [s[i]])
            p.set_visible(True)

        renderer = canvas.get_renderer()
        textos = texto_fn(i)
        for axx, s, a, texto in zip(ejes, series, annots, textos):
            if not texto:
                a.set_visible(False)
                continue
            yv = s[i]
            bb = axx.get_window_extent(renderer=renderer)
            x_disp, y_disp = axx.transData.transform((xv, yv))
            dx, ha = ((12, 'left') if (bb.x1 - x_disp) >= (x_disp - bb.x0)
                      else (-12, 'right'))
            dy, va = ((12, 'bottom') if (bb.y1 - y_disp) >= (y_disp - bb.y0)
                      else (-12, 'top'))
            a.xy = (xv, yv)
            a.set_position((dx, dy))
            a.set_ha(ha)
            a.set_va(va)
            a.set_text(texto)
            a.set_visible(True)

        if x_texto_fn is not None:
            texto_x = x_texto_fn(i)
        elif es_fecha:
            texto_x = pd.Timestamp(x_arr[i]).strftime(patron_fecha)
        else:
            texto_x = f"Barra nº {int(x_arr[i])}"
        for et in etiquetas_x:
            et.xy = (xv, 0)
            et.set_text(texto_x)
            et.set_visible(True)
        _pintar_dinamicos()

    cid1 = canvas.mpl_connect('motion_notify_event', _on_move)
    cid2 = canvas.mpl_connect('figure_leave_event', lambda e: _ocultar())
    cid3 = canvas.mpl_connect('draw_event', _capturar_fondo)
    setattr(fig, cid_attr, (cid1, cid2, cid3))


def agregar_hover_celdas(fig, ax, m, filas=None, cols=None, fmt='{:.2f}',
                         nombre='default', titulo=None):
    """Tooltip al pasar el ratón sobre cada celda de una matriz pintada con
    imshow/heatmap (las celdas quedan centradas en coordenadas enteras).

    Marca la celda con un borde fino y muestra `fila × columna → valor` en un
    bocadillo anclado a ella. `filas`/`cols`: etiquetas (por defecto, índices);
    `fmt`: formato del valor (p. ej. '{:.3f}'); `titulo`: cabecera opcional.

    Misma arquitectura de blitting que `agregar_crosshair` (fondo cacheado en
    cada draw_event, disconnect de cids previos vía `nombre` en `fig`), para
    que convivir ambos mecanismos en una misma sección sea seguro."""
    from matplotlib.patches import Rectangle
    m = np.asarray(m, dtype=float)
    if m.size == 0:
        return
    canvas = fig.canvas
    cid_attr = f'_hover_celdas_cids_{nombre}'
    for cid in getattr(fig, cid_attr, ()):
        try:
            canvas.mpl_disconnect(cid)
        except Exception:
            pass

    filas = list(filas) if filas is not None else [str(i) for i in range(m.shape[0])]
    cols = list(cols) if cols is not None else [str(j) for j in range(m.shape[1])]

    rect = ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='none', edgecolor=AX_FG,
                                  linewidth=1.0, zorder=99, visible=False))
    annot = ax.annotate("", xy=(0, 0), xytext=(12, -12), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.4", fc=FIG_BG, ec=GRID_C, alpha=0.95),
                        color=AX_FG, fontsize=7, zorder=100, annotation_clip=False)
    annot.set_visible(False)

    estado = {'bg': None}

    def _capturar_fondo(_event=None):
        estado['bg'] = canvas.copy_from_bbox(fig.bbox)

    def _pintar():
        if estado['bg'] is None:
            canvas.draw_idle()
            return
        canvas.restore_region(estado['bg'])
        if rect.get_visible():
            rect.axes.draw_artist(rect)
        if annot.get_visible():
            annot.axes.draw_artist(annot)
        canvas.blit(fig.bbox)

    def _ocultar():
        cambio = rect.get_visible() or annot.get_visible()
        rect.set_visible(False)
        annot.set_visible(False)
        if cambio:
            _pintar()

    def _on_move(event):
        if event.inaxes is not ax or event.xdata is None:
            _ocultar()
            return
        j = int(np.floor(event.xdata + 0.5))
        i = int(np.floor(event.ydata + 0.5))
        if not (0 <= i < m.shape[0] and 0 <= j < m.shape[1]):
            _ocultar()
            return
        val = m[i, j]
        if not np.isfinite(val):
            _ocultar()
            return
        rect.set_bounds(j - 0.5, i - 0.5, 1, 1)
        rect.set_visible(True)
        cabecera = f"{titulo}\n" if titulo else ""
        annot.xy = (j, i)
        annot.set_text(f"{cabecera}{filas[i]} × {cols[j]} → {fmt.format(val)}")
        annot.set_visible(True)
        _pintar()

    cid1 = canvas.mpl_connect('motion_notify_event', _on_move)
    cid2 = canvas.mpl_connect('figure_leave_event', lambda e: _ocultar())
    cid3 = canvas.mpl_connect('draw_event', _capturar_fondo)
    setattr(fig, cid_attr, (cid1, cid2, cid3))


class _OverlayAyuda(QWidget):
    """Panel de ayuda «?» rediseñado: se dibuja DENTRO de la ventana principal
    como un widget hijo pintado a mano (QPainter), sin ventana propia, sin
    translucencia a nivel de OS ni efectos de sombra. Como Windows no gestiona
    ninguna ventana aquí, es imposible que aparezcan bordes, sombras o cuadros
    negros alrededor: el widget solo pinta el panel redondeado y el resto queda
    transparente mostrando la app.

    Se cierra al pulsar fuera del panel (o sobre su icono, como toggle) y con
    Escape, vía un eventFilter global.
    """

    _ANCHO = 420
    _PADDING = 14
    _RADIO = 10

    def __init__(self, secciones, parent, icono=None):
        super().__init__(parent)
        self._icono = icono
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(self._ANCHO)
        self.setStyleSheet(
            "QWidget { background: transparent; }"
            "QLabel { color: #dbe8f5; font-size: 11px; }"
            "QLabel#tituloAyuda { color: #4fc3f7; font-size: 12px; font-weight: bold; }"
            "QFrame#sepAyuda { background-color: #253a60; max-height: 1px; border: none; }"
            "QTabWidget::pane { background-color: transparent; border: none; }"
            "QTabBar::tab { background-color: transparent; color: #5a7a9a;"
            " padding: 6px 12px; border: none; font-size: 10px; }"
            "QTabBar::tab:selected { color: #4fc3f7; font-weight: bold;"
            " border-bottom: 2px solid #4fc3f7; }")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(self._PADDING, 12, self._PADDING, self._PADDING)
        lay.setSpacing(8)
        ancho_texto = self._ANCHO - 2 * self._PADDING
        pantalla = QApplication.primaryScreen()
        altura_max = max(240, (pantalla.availableGeometry().height() - 160)
                         if pantalla is not None else 240)

        def _label(texto):
            lbl = QLabel(texto, self)
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            # aplicar la fuente del QSS antes de medir: si se mide con la
            # fuente por defecto, la altura sale corta y el texto se recorta
            lbl.ensurePolished()
            return lbl

        if len(secciones) == 1:
            titulo, texto = secciones[0]
            lbl_titulo = QLabel(titulo, self)
            lbl_titulo.setObjectName("tituloAyuda")
            lbl_titulo.ensurePolished()
            lay.addWidget(lbl_titulo)
            sep = QFrame(self)
            sep.setObjectName("sepAyuda")
            sep.setFixedHeight(1)
            lay.addWidget(sep)
            lbl_texto = _label(texto)
            lay.addWidget(lbl_texto)
            # altura completa: título + separador + texto, sin barras de scroll
            natural = (12 + lbl_titulo.height() + 8 + 1 + 8
                       + lbl_texto.heightForWidth(ancho_texto) + self._PADDING)
        else:
            tabs = QTabWidget()
            tabs.setDocumentMode(True)
            alturas = []
            for titulo, texto in secciones:
                lbl = _label(texto)
                alturas.append(lbl.heightForWidth(ancho_texto))
                tabs.addTab(lbl, titulo)
            lay.addWidget(tabs)
            tabs.ensurePolished()
            barra = tabs.tabBar().sizeHint().height() or 28
            # altura completa: barra + separación + el texto más alto
            natural = 12 + barra + 8 + max(alturas) + self._PADDING

        # el texto SIEMPRE cabe: se dimensiona a la altura completa del
        # contenido y se recorta solo contra la altura máxima disponible
        self.setFixedHeight(min(natural, altura_max))

        self._anim_fade = None
        QApplication.instance().installEventFilter(self)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        grad.setColorAt(0, QColor("#1b2c4a"))
        grad.setColorAt(1, QColor("#0d1424"))
        p.setBrush(grad)
        p.setPen(QPen(QColor("#2a4a6a"), 1))
        p.drawRoundedRect(rect, self._RADIO, self._RADIO)
        p.end()

    # ── cierre por clic fuera / Escape ──
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            pos = event.globalPosition().toPoint()
            if not self._rect_global().contains(pos) \
                    and not self._rect_icono_global().contains(pos):
                self.close()
        elif event.type() == QEvent.Type.KeyPress \
                and event.key() == Qt.Key.Key_Escape:
            self.close()
        return super().eventFilter(obj, event)

    def _rect_global(self):
        return QRect(self.mapToGlobal(self.rect().topLeft()), self.size())

    def _rect_icono_global(self):
        if self._icono is None:
            return QRect()
        i = self._icono
        return QRect(i.mapToGlobal(i.rect().topLeft()), i.size())

    def closeEvent(self, event):
        QApplication.instance().removeEventFilter(self)
        super().closeEvent(event)

    def _fade_in(self):
        efe = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(efe)
        efe.setOpacity(0.0)
        anim = QPropertyAnimation(efe, b"opacity", self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(160)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim_fade = anim
        anim.start()


# ══════════════ paneles flotantes sin marco de Windows ══════════════

def animar_entrada(widget, deslizamiento=8, duracion_ms=180):
    """Fade-in + deslizamiento sutil al mostrar un panel flotante.

    Las animaciones se guardan como atributos del propio widget para que Qt
    no las recolecte a mitad de vuelo."""
    pos_final = widget.pos()
    fade = QPropertyAnimation(widget, b"windowOpacity", widget)
    fade.setStartValue(0.0)
    fade.setEndValue(1.0)
    fade.setDuration(duracion_ms)
    fade.setEasingCurve(QEasingCurve.Type.OutCubic)
    slide = QPropertyAnimation(widget, b"pos", widget)
    slide.setStartValue(pos_final + QPoint(0, deslizamiento))
    slide.setEndValue(pos_final)
    slide.setDuration(duracion_ms)
    slide.setEasingCurve(QEasingCurve.Type.OutCubic)
    widget._anim_fade = fade
    widget._anim_slide = slide
    fade.start()
    slide.start()


def instalar_arrastre(ventana, widget):
    """Arrastrar una ventana sin marco desde `widget` (la cabecera)."""
    datos = {'offset': None}

    def _pressed(event):
        datos['offset'] = (event.globalPosition().toPoint()
                           - ventana.frameGeometry().topLeft())

    def _moved(event):
        if datos['offset'] is not None:
            ventana.move(event.globalPosition().toPoint() - datos['offset'])

    def _released(event):
        datos['offset'] = None

    widget.mousePressEvent = _pressed
    widget.mouseMoveEvent = _moved
    widget.mouseReleaseEvent = _released


class PanelFlotanteDialog(QDialog):
    """Diálogo sin marco de Windows que se anima (fade + deslizamiento) al
    mostrarse. Es la base de QuestDB y de la lista completa de trades."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._animado_entrada = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self._animado_entrada:
            self._animado_entrada = True
            animar_entrada(self)


def montar_panel_flotante(dlg, titulo, ancho, alto=None, subtitulo='',
                          boton_cerrar=True):
    """Monta en `dlg` la estructura de panel flotante: sin marco nativo,
    translúcido, con cabecera propia (marca + título + subtítulo + ✕) y
    arrastre desde la cabecera.

    Devuelve (lay_contenido, lbl_subtitulo, halo). `lay_contenido` es donde
    el llamador coloca su contenido; `lbl_subtitulo` se puede actualizar en
    vivo; `halo` es el anillo azul exterior (para animarlo si se quiere).
    """
    dlg.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
    dlg.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    dlg.setFixedWidth(ancho)
    if alto is not None:
        dlg.setFixedHeight(alto)
    dlg.setStyleSheet(
        "QDialog { background: transparent; }"
        "QLabel#tituloPanel { color: #4fc3f7; font-size: 13px; font-weight: bold; }"
        "QLabel#subPanel { color: #5a7a9a; font-size: 11px; }"
        "QLabel#textoPanel { color: #dbe8f5; font-size: 11px; }"
        "QPushButton#cerrarPanel { background: transparent; color: #7a90ad;"
        " border: none; font-size: 15px; font-weight: bold; padding: 0 7px;"
        " border-radius: 5px; }"
        "QPushButton#cerrarPanel:hover { color: #e74c3c; background-color: #2a1a1a; }")
    lay = QVBoxLayout(dlg)
    lay.setContentsMargins(20, 20, 20, 20)
    halo = QFrame(dlg)
    halo.setObjectName("haloPanel")
    halo.setStyleSheet(
        "QFrame#haloPanel { background-color: rgba(79, 195, 247, 40);"
        " border-radius: 14px; }")
    glow = QGraphicsDropShadowEffect(halo)
    glow.setBlurRadius(18)
    glow.setColor(QColor(79, 195, 247, 90))
    glow.setOffset(0, 0)
    halo.setGraphicsEffect(glow)
    lay.addWidget(halo)
    panel = QFrame(halo)
    panel.setObjectName("panelFlotante")
    panel.setStyleSheet(
        "QFrame#panelFlotante { background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        " stop:0 #1b2c4a, stop:1 #0d1424); border: 1px solid #2a4a6a;"
        " border-radius: 12px; }")
    sombra = QGraphicsDropShadowEffect(panel)
    sombra.setBlurRadius(18)
    sombra.setColor(QColor(0, 0, 0, 150))
    sombra.setOffset(0, 4)
    panel.setGraphicsEffect(sombra)
    lay_panel = QVBoxLayout(panel)
    lay_panel.setContentsMargins(14, 10, 14, 14)
    lay_panel.setSpacing(10)
    lay_halo = QVBoxLayout(halo)
    lay_halo.setContentsMargins(2, 2, 2, 2)
    lay_halo.addWidget(panel)
    # cabecera: marca + título + subtítulo + botón de cierre
    header = QWidget(panel)
    lay_head = QHBoxLayout(header)
    lay_head.setContentsMargins(2, 0, 2, 0)
    lay_head.setSpacing(8)
    punto = QLabel("●")
    punto.setStyleSheet("color: #4fc3f7; font-size: 12px;")
    lay_head.addWidget(punto)
    col = QVBoxLayout()
    col.setSpacing(0)
    col.setContentsMargins(0, 0, 0, 0)
    lbl_titulo = QLabel(titulo)
    lbl_titulo.setObjectName("tituloPanel")
    col.addWidget(lbl_titulo)
    lbl_sub = QLabel(subtitulo)
    lbl_sub.setObjectName("subPanel")
    col.addWidget(lbl_sub)
    lay_head.addLayout(col, 1)
    if boton_cerrar:
        btn_x = QPushButton("✕")
        btn_x.setObjectName("cerrarPanel")
        btn_x.setFixedSize(28, 26)
        btn_x.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_x.clicked.connect(dlg.close)
        lay_head.addWidget(btn_x)
    lay_panel.addWidget(header)
    sep = QFrame(panel)
    sep.setFixedHeight(1)
    sep.setStyleSheet("background-color: #253a60; border: none;")
    lay_panel.addWidget(sep)
    instalar_arrastre(dlg, header)
    return lay_panel, lbl_sub, halo


def panel_flotante_dialog(titulo, ancho, alto=None, subtitulo='', parent=None,
                          boton_cerrar=True):
    """Diálogo flotante sin marco de Windows listo para usar.

    Devuelve (dlg, lay_contenido, lbl_subtitulo, halo). El diálogo se anima
    solo al mostrarse (fade + deslizamiento)."""
    dlg = PanelFlotanteDialog(parent)
    lay, lbl_sub, halo = montar_panel_flotante(
        dlg, titulo, ancho, alto=alto, subtitulo=subtitulo,
        boton_cerrar=boton_cerrar)
    return dlg, lay, lbl_sub, halo


def _badge_ayuda(secciones, tooltip=None):
    """QLabel «?» en forma de badge que abre `secciones` en un popup al hacer
    clic. TODO icono de ayuda de la app pasa por aquí: si unos respondieran al
    clic y otros solo al pasar el ratón, siendo idénticos, los segundos se leen
    como rotos (era el caso de los del Backtester)."""
    icono = QLabel("?")
    icono.setFixedSize(16, 16)
    icono.setAlignment(Qt.AlignmentFlag.AlignCenter)
    icono.setStyleSheet(
        "QLabel { background-color: #253a60; color: #8fb3d9; "
        "border-radius: 8px; font-size: 10px; font-weight: bold; }"
        "QLabel:hover { background-color: #3a5a8a; color: #4fc3f7;"
        " border: 1px solid rgba(79, 195, 247, 120); }")
    icono.setCursor(Qt.CursorShape.PointingHandCursor)
    if tooltip:
        icono.setToolTip(tooltip)

    def _abrir(event):
        # Toggle: si el overlay de este icono sigue abierto, se cierra.
        prev = getattr(icono, '_overlay_ayuda', None)
        if prev is not None and prev.isVisible():
            prev.close()
            icono._overlay_ayuda = None
            return
        ventana = icono.window()
        if ventana is None:
            return
        # Referencia guardada en el propio icono: sin ella, el wrapper de
        # Python del overlay puede recolectarse antes de que Qt lo muestre.
        overlay = _OverlayAyuda(secciones, ventana, icono)
        icono._overlay_ayuda = overlay
        overlay.move(_posicion_overlay(icono, overlay))
        overlay._fade_in()
        overlay.show()
        overlay.raise_()
    icono.mousePressEvent = _abrir
    return icono


def _posicion_overlay(icono, overlay):
    """Esquina superior izquierda donde colocar el overlay, recortada al
    rectángulo de la ventana principal (el overlay es hijo de esa ventana y no
    puede salirse de ella).

    El panel se ancla a la esquina inferior izquierda del icono y crece hacia
    la derecha y abajo. Sin recorte, en el Backtester los iconos van pegados al
    borde derecho de su grupo (`_fila_ayuda`), así que el panel caería fuera de
    la ventana: se recorta al rectángulo del padre. Lo mismo por abajo: un
    icono en la parte baja de una página larga abriría su panel por debajo del
    borde inferior; ahí se prueba primero a abrirlo hacia ARRIBA del icono,
    como hace cualquier menú.
    """
    padre = overlay.parentWidget()
    if padre is None:
        return QPoint(0, 0)
    # el tamaño definitivo no está calculado hasta que se muestra: sin esto,
    # width()/height() devuelven el tamaño por defecto y el recorte se haría
    # contra medidas que no son las del panel
    overlay.adjustSize()
    destino = padre.mapFromGlobal(icono.mapToGlobal(icono.rect().bottomLeft()))
    w, h = overlay.width(), overlay.height()
    libre = padre.rect()

    x = destino.x()
    if x + w > libre.right():
        x = libre.right() - w
    x = max(x, libre.left())

    y = destino.y()
    if y + h > libre.bottom():
        arriba = padre.mapFromGlobal(
            icono.mapToGlobal(icono.rect().topLeft())).y() - h
        if arriba >= libre.top():
            y = arriba
    # recorte final por si tampoco cabe arriba (panel más alto que el padre,
    # o icono fuera del área visible): siempre dentro, aunque tape al icono
    y = max(libre.top(), min(y, libre.bottom() - h))
    return QPoint(x, y)


def icono_ayuda(logica, significado, uso, resultados):
    """Badge «?» cuyo popup trae 4 pestañas (Lógica/Significado/Uso/
    Resultados) — para explicaciones largas que conviene leer con calma."""
    return _badge_ayuda([('Lógica', logica), ('Significado', significado),
                         ('Uso', uso), ('Resultados', resultados)])


def icono_ayuda_texto(texto):
    """Badge «?» de una sola explicación: el mismo texto en el tooltip al pasar
    el ratón y en el popup al hacer clic, para que el gesto que haga el usuario
    funcione sea cual sea."""
    return _badge_ayuda([('Ayuda', texto)], tooltip=texto)


def no_crash(fn):
    """Blinda un slot: una excepción sin capturar dentro de un slot de PyQt6
    aborta el proceso entero (qFatal), sin traza útil. Se registra el error y
    se sigue."""
    @functools.wraps(fn)
    def envoltorio(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            print(f"[ERROR] excepción en {fn.__name__}:", flush=True)
            traceback.print_exc()
            return None
    return envoltorio


# ══════════════ barra de herramientas del canvas ══════════════
# NavigationToolbar2QT trae el motor de zoom/pan/guardar ya resuelto, pero sus
# iconos PNG por defecto (grises, estilo genérico) desentonan con el resto de la
# app. Se conserva el motor y se sustituye solo el aspecto: iconos vectoriales
# dibujados con QPainter, igual que gui/dialogs/tutorial_icons.py y el icono
# "home" de gui/widgets/file_explorer.py.

_ESTILO_BARRA = """
QToolBar { background: transparent; border: none; padding: 0px; spacing: 2px; }
QToolButton {
    background: transparent; border: none;
    padding: 4px; border-radius: 4px;
}
QToolButton:hover { background-color: #1a2a45; }
QToolButton:pressed { background-color: #253a60; }
QToolButton:checked { background-color: #2a4a6a; }
QLabel { color: #5a7a9a; font-size: 10px; }
"""


def _pt(tam, fx, fy):
    return QPointF(tam * fx, tam * fy)


def _poly(tam, *fracciones):
    return QPolygonF([_pt(tam, fx, fy) for fx, fy in fracciones])


def _dibujar_inicio(p, tam):
    """Casa: volver a la vista completa."""
    p.drawPolyline(_poly(tam, (0.16, 0.50), (0.50, 0.20), (0.84, 0.50)))
    p.drawPolyline(_poly(tam, (0.27, 0.44), (0.27, 0.80), (0.73, 0.80), (0.73, 0.44)))
    p.drawPolyline(_poly(tam, (0.42, 0.80), (0.42, 0.60), (0.58, 0.60), (0.58, 0.80)))


def _dibujar_pan(p, tam):
    """Cruz de cuatro flechas: desplazar."""
    p.drawLine(_pt(tam, 0.50, 0.14), _pt(tam, 0.50, 0.86))
    p.drawLine(_pt(tam, 0.14, 0.50), _pt(tam, 0.86, 0.50))
    for a, b, c in (((0.50, 0.14), (0.40, 0.26), (0.60, 0.26)),
                    ((0.50, 0.86), (0.40, 0.74), (0.60, 0.74)),
                    ((0.14, 0.50), (0.26, 0.40), (0.26, 0.60)),
                    ((0.86, 0.50), (0.74, 0.40), (0.74, 0.60))):
        p.drawPolyline(_poly(tam, b, a, c))


def _dibujar_zoom(p, tam):
    """Lupa con marco de selección: zoom de rectángulo."""
    r = tam * 0.24
    p.drawEllipse(_pt(tam, 0.44, 0.42), r, r)
    p.drawLine(_pt(tam, 0.62, 0.60), _pt(tam, 0.84, 0.82))
    p.drawLine(_pt(tam, 0.32, 0.42), _pt(tam, 0.56, 0.42))
    p.drawLine(_pt(tam, 0.44, 0.30), _pt(tam, 0.44, 0.54))


def _dibujar_guardar(p, tam):
    """Flecha hacia una bandeja: guardar PNG."""
    p.drawLine(_pt(tam, 0.50, 0.16), _pt(tam, 0.50, 0.58))
    p.drawPolyline(_poly(tam, (0.34, 0.44), (0.50, 0.60), (0.66, 0.44)))
    p.drawPolyline(_poly(tam, (0.20, 0.66), (0.20, 0.82), (0.80, 0.82), (0.80, 0.66)))


def _icono(dibujante, color=AZUL, tam=18):
    """Pixmap transparente con el trazo dibujado a mano → QIcon."""
    pix = QPixmap(tam, tam)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(max(1.2, tam * 0.085))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    dibujante(p, tam)
    p.end()
    return QIcon(pix)


_DIBUJANTES = {
    'Home': _dibujar_inicio,
    'Pan': _dibujar_pan,
    'Zoom': _dibujar_zoom,
    'Save': _dibujar_guardar,
}


class BarraGrafico(NavigationToolbar2QT):
    """Barra de zoom/pan/guardar con los iconos y el tema de la aplicación.

    Se queda solo con las cuatro acciones útiles aquí: se descartan `Subplots`
    (abre un diálogo de márgenes que no aporta nada en un panel embebido),
    `Customize` y el histórico Back/Forward, que sin ellas cabe en una fila
    junto al título de la sección.
    """

    toolitems = [t for t in NavigationToolbar2QT.toolitems
                 if t[0] in ('Home', 'Pan', 'Zoom', 'Save')]

    def __init__(self, canvas, parent=None, mostrar_coords=True):
        super().__init__(canvas, parent)
        self.setStyleSheet(_ESTILO_BARRA)
        self.setIconSize(QSize(18, 18))
        self.setFloatable(False)
        self.setMovable(False)

        # Los QAction se crean en el __init__ del padre a partir de toolitems,
        # en el mismo orden; se localizan por su texto para no depender de la
        # posición dentro de la barra.
        for accion in self.actions():
            dibujante = _DIBUJANTES.get(accion.text())
            if dibujante is not None:
                accion.setIcon(_icono(dibujante))

        # La etiqueta de coordenadas la crea el padre; en paneles estrechos
        # empuja los botones, así que es opcional.
        if not mostrar_coords and getattr(self, 'locLabel', None) is not None:
            self.locLabel.setVisible(False)
