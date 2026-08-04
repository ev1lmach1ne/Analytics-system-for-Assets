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

from PyQt6.QtCore import Qt, QSize, QPointF
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QPen, QColor, QPolygonF
from PyQt6.QtWidgets import (QSizePolicy, QLabel, QDialog, QTabWidget,
                             QVBoxLayout)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.dates import date2num, AutoDateLocator, ConciseDateFormatter

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
                      colores=None):
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
                             zorder=98, visible=False) for axx in ejes]
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
                  or any(p.get_visible() for p in puntos)
                  or any(et.get_visible() for et in etiquetas_x))
        for l in lineas_v:
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

        texto_x = (pd.Timestamp(x_arr[i]).strftime(patron_fecha) if es_fecha
                  else f"Barra nº {int(x_arr[i])}")
        for et in etiquetas_x:
            et.xy = (xv, 0)
            et.set_text(texto_x)
            et.set_visible(True)
        _pintar_dinamicos()

    cid1 = canvas.mpl_connect('motion_notify_event', _on_move)
    cid2 = canvas.mpl_connect('figure_leave_event', lambda e: _ocultar())
    cid3 = canvas.mpl_connect('draw_event', _capturar_fondo)
    setattr(fig, cid_attr, (cid1, cid2, cid3))


class _PopupAyuda(QDialog):
    """Panel flotante con 4 pestañas (Lógica/Significado/Uso/Resultados);
    se cierra solo al perder el foco, igual que los editores de condiciones
    del Backtester (gui/widgets/tab_backtest.py, Qt.WindowType.Popup)."""

    def __init__(self, secciones, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setFixedWidth(360)
        self.setStyleSheet(
            "QDialog { background-color: #0d1424; border: 2px solid #253a60;"
            " border-radius: 6px; }"
            "QTabWidget::pane { background-color: #0d1424; border: 1px solid #253a60; }"
            "QTabBar::tab { background-color: #141e30; color: #5a7a9a; padding: 6px 12px;"
            " border: none; border-right: 1px solid #253a60; font-size: 10px; }"
            "QTabBar::tab:selected { background-color: #0d1424; color: #4fc3f7;"
            " font-weight: bold; }"
            "QLabel { color: #c8d6e5; font-size: 11px; padding: 8px; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        # Ancho de texto real dentro de la pestaña (fijo=360 menos bordes y
        # el padding de 8px del QLabel a cada lado).
        ancho_texto = 360 - 2 - 32
        alturas = []
        for titulo, texto in secciones:
            lbl = QLabel(texto)
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            lbl.setFixedWidth(ancho_texto)
            alturas.append(lbl.heightForWidth(ancho_texto))
            tabs.addTab(lbl, titulo)
        # Sin QScrollArea: cada pestaña muestra el texto completo. La altura
        # del panel de pestañas se fija de una vez al máximo que necesita
        # cualquiera de los 4 textos (+ padding del QLabel y la barra de
        # pestañas), para que cambiar de pestaña no redimensione el popup en
        # caliente — QTabWidget solo calcula bien el heightForWidth de la
        # página VISIBLE, así que recalcular al vuelo al cambiar de pestaña
        # queda un ciclo de layout por detrás y corta el texto.
        tabs.setFixedHeight(max(alturas) + 32 + 34)
        lay.addWidget(tabs)


def icono_ayuda(logica, significado, uso, resultados):
    """QLabel «?» en forma de badge; un clic abre un popup con 4 pestañas
    (Lógica/Significado/Uso/Resultados) en vez de un tooltip al pasar el
    ratón — para explicaciones largas que conviene poder leer con calma."""
    icono = QLabel("?")
    icono.setFixedSize(16, 16)
    icono.setAlignment(Qt.AlignmentFlag.AlignCenter)
    icono.setStyleSheet(
        "QLabel { background-color: #253a60; color: #8fb3d9; "
        "border-radius: 8px; font-size: 10px; font-weight: bold; }")
    icono.setCursor(Qt.CursorShape.PointingHandCursor)
    secciones = [('Lógica', logica), ('Significado', significado),
                 ('Uso', uso), ('Resultados', resultados)]

    def _abrir(event):
        # Referencia guardada en el propio icono: sin ella, el wrapper de
        # Python del popup puede recolectarse antes de que Qt lo muestre,
        # pese a llevar `icono` como parent en C++.
        icono._popup = _PopupAyuda(secciones, icono)
        icono._popup.move(icono.mapToGlobal(icono.rect().bottomLeft()))
        icono._popup.show()
    icono.mousePressEvent = _abrir
    return icono


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
