"""
Fila de tarjetas KPI de la sub-pestaña «Métricas» del Analizador.

Las categorías de métricas de debajo siguen siendo la lista completa y exacta
del informe; estas tarjetas son el resumen de un vistazo.

El juego está orientado a **caracterizar el activo**, no a evaluar una
estrategia: además del rendimiento y el riesgo clásicos, resume la forma de la
distribución (normalidad), la dependencia temporal por escala de calendario
(ACF/PACF diaria, semanal, mensual, trimestral) y el clustering de volatilidad,
más el régimen de la Ventana seleccionada (ER, Hurst).

Los valores llegan del bundle de datos del análisis (clave `kpi`), ya como
números crudos — no del metrics.json, que contiene strings formateados con
códigos ANSI pensados para el volcado de terminal.
"""
import inspect
import re

from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QPolygonF
from PyQt6.QtWidgets import QWidget, QFrame, QGridLayout, QVBoxLayout, QLabel, QSizePolicy

from core.config import velas_a_tiempo_legible
from gui.widgets.plot_common import (VERDE, ROJO, AMBAR, AZUL, GRIS,
                                     ER_TENDENCIA, ER_TRANSICION, ER_RUIDO, AX_FG)

ESTILO = """
QWidget#tarjetasKpi { background: transparent; }
QFrame#tarjeta {
    background-color: #141e30; border: 1px solid #253a60;
    border-radius: 6px;
}
QLabel#kpiTitulo { color: #8fb3d9; font-size: 10px; font-weight: bold; }
QLabel#kpiValor { font-size: 19px; font-weight: bold; }
QLabel#kpiValorTexto { font-size: 13px; font-weight: bold; }
QLabel#kpiSub { color: #5a7a9a; font-size: 9px; }
"""

_ANCHO_TARJETA = 150


# ══════════════ formateo ══════════════
def _pct(v, dec=2):
    return f"{v * 100:.{dec}f}%"


def _num(v, dec=2):
    return f"{v:.{dec}f}"


def _escala(v, bueno, regular, invertido=False):
    """Color por umbral. `invertido` para métricas donde menos es mejor."""
    if invertido:
        if v <= bueno:
            return VERDE
        return AMBAR if v <= regular else ROJO
    if v >= bueno:
        return VERDE
    return AMBAR if v >= regular else ROJO


_UNIDAD_RE = re.compile(r'\d+\s*(?:años?|[dhm])\b')


def _compactar_duracion(texto):
    """«364d 22h 0m» → «364d 22h»; «1 año 0d 22h 0m» → «1 año 22h».

    El motor formatea la recuperación con todas las unidades para la tabla del
    PDF; en una tarjeta de 150 px sobran las dos menos significativas y las que
    valen cero.
    """
    if not texto:
        return texto
    partes = [p.strip() for p in _UNIDAD_RE.findall(texto)]
    partes = [p for p in partes if not re.match(r'^0\s*(?:años?|[dhm])$', p)]
    return ' '.join(partes[:2]) if partes else texto


# ══════════════ subtítulos y tooltips dinámicos ══════════════
def _sub_max_dd(ctx):
    g = ctx['global']
    velas = g.get('dd_recuperacion_velas') or 0
    recuperado = g.get('dd_recuperado')
    if not velas or recuperado is None:
        return "sin drawdown"
    texto = _compactar_duracion(g.get('dd_recuperacion_texto'))
    if not texto:
        # Archivo de TICKS: no hay fechas, solo el conteo de velas.
        legible = velas_a_tiempo_legible(velas, (ctx['meta'] or {}).get('tf'))
        texto = legible if legible != "—" else f"{velas:,} velas".replace(',', '.')
    return f"recuperado en {texto}" if recuperado else f"sin recuperar ({texto})"


def _tip_max_dd(ctx):
    g = ctx['global']
    base = "Peor caída acumulada de pico a valle sufrida por el activo."
    velas = g.get('dd_recuperacion_velas') or 0
    if not velas:
        return base
    rango = g.get('dd_recuperacion_rango')
    detalle = g.get('dd_recuperacion_texto') or "—"
    linea = f"\n\nRecuperación: {detalle}"
    if rango:
        linea += f"\n{rango}"
    linea += f"\n{velas:,} velas".replace(',', '.')
    if g.get('dd_recuperado') is False:
        linea += "\n\nEl activo seguía en drawdown al final de los datos."
    return base + linea


def _sub_normalidad(ctx):
    p = ctx['global'].get('jb_p_value')
    return "Jarque-Bera p = —" if p is None else f"Jarque-Bera p = {p:.4f}"


def _sub_clustering(ctx):
    p = ctx['global'].get('clustering_lb_p')
    return "Ljung-Box p = —" if p is None else f"Ljung-Box p = {p:.4f}"


_ESCALA_LABEL = {'dia': 'diario', 'semanal': 'semanal',
                 'mensual': 'mensual', 'trimestral': 'trimestral'}


def _sub_dependencia(escala):
    def _sub(ctx):
        v = ctx['global'].get(f'dependencia_{escala}_pacf1')
        return "PACF(1): —" if v is None else f"PACF(1): {v:+.4f}"
    return _sub


def _tip_dependencia(escala):
    etiqueta = _ESCALA_LABEL[escala]

    def _tip(ctx):
        base = (f"Autocorrelación parcial (PACF lag 1) del retorno {etiqueta}. "
                "INERCIA: el retorno tiende a repetir signo/tamaño de un periodo "
                "al siguiente. REVERSIÓN: tiende a invertirse. RUIDO: sin "
                "dependencia significativa.")
        umbral = ctx['global'].get(f'dependencia_{escala}_umbral')
        if umbral is not None:
            base += f"\n\nUmbral de significancia: ±{umbral:.4f}"
        return base
    return _tip


def _sub_hurst(ctx):
    h = (ctx['horizonte'] or {}).get('hurst_medio')
    if h is None:
        return "exponente de Hurst"
    if h > 0.55:
        return "persistente (tendencia)"
    if h < 0.45:
        return "antipersistente (reversión)"
    return "próximo a paseo aleatorio"


# ══════════════ colorizadores ══════════════
def _color_hurst(v):
    # El régimen no es «mejor» ni «peor»: el color identifica cuál es.
    if v > 0.55:
        return VERDE
    if v < 0.45:
        return AZUL
    return GRIS


def _color_clustering(presente):
    return AMBAR if presente else GRIS


def _color_normalidad(normal):
    # Una serie financiera NO normal (colas gruesas) es lo esperable, no un
    # defecto; SÍ normal es el caso raro y reseñable.
    return AZUL if normal else GRIS


_COLOR_DEPENDENCIA = {'INERCIA': VERDE, 'REVERSIÓN': AZUL}


def _color_dependencia(estado):
    # Mismo esquema de 3 vías que _color_hurst: persistencia (INERCIA) verde,
    # reversión azul, sin dependencia (RUIDO) gris neutro.
    return _COLOR_DEPENDENCIA.get(estado, GRIS)


# ══════════════ definición de las tarjetas ══════════════
# `fuente`: 'global' o 'horizonte' (estas últimas cambian con la Ventana).
# `sub` y `tooltip` admiten texto fijo o un callable que recibe el contexto.
# `compacto` usa una fuente menor, para valores que son un veredicto y no un número.
_KPIS = [
    dict(clave='cagr', titulo='CAGR', sub='anualizado', fuente='global',
         formato=_pct, color=lambda v: _escala(v, 0.10, 0.0),
         tooltip="Retorno compuesto anualizado del activo en el periodo analizado."),

    dict(clave='vol_anual', titulo='Volatilidad', sub='anualizada', fuente='global',
         formato=_pct, color=lambda v: _escala(v, 0.20, 0.40, invertido=True),
         tooltip="Desviación estándar de los retornos, anualizada."),

    dict(clave='sharpe', titulo='Sharpe', sub='vs. T-Bill 3m', fuente='global',
         formato=_num, color=lambda v: _escala(v, 1.0, 0.0),
         tooltip="Exceso de retorno por unidad de volatilidad. Es la métrica que "
                 "permite comparar este activo con otros."),

    dict(clave='max_dd', titulo='Max drawdown', sub=_sub_max_dd, fuente='global',
         formato=_pct, color=lambda v: _escala(abs(v), 0.20, 0.40, invertido=True),
         tooltip=_tip_max_dd),

    dict(clave='var95', titulo='VaR 95%', sub='por vela', fuente='global',
         formato=_pct, color=lambda v: _escala(abs(v), 0.01, 0.03, invertido=True),
         tooltip="Pérdida que solo se supera el 5% de las velas (percentil 5)."),

    dict(clave='cvar95', titulo='CVaR 95%', sub='media de la cola', fuente='global',
         formato=_pct, color=lambda v: _escala(abs(v), 0.02, 0.05, invertido=True),
         tooltip="Expected Shortfall: pérdida MEDIA de las velas que caen en el "
                 "peor 5%. Siempre es mayor (en magnitud) que el VaR 95%; cuanto "
                 "más se aleje de él, más gordas son las colas del activo."),

    dict(clave='es_normal', titulo='Distribución normal', sub=_sub_normalidad,
         fuente='global', formato=lambda v: "SÍ" if v else "NO",
         color=_color_normalidad,
         tooltip="Test de Jarque-Bera sobre los retornos: contrasta si la "
                 "distribución es compatible con una Normal. La mayoría de "
                 "series financieras NO lo son (colas más gruesas que la "
                 "campana de Gauss)."),

    dict(clave='clustering_presente', titulo='Clustering vol.', sub=_sub_clustering,
         fuente='global', formato=lambda v: "SÍ" if v else "NO",
         color=_color_clustering,
         tooltip="Efecto ARCH: si la volatilidad se agrupa en rachas, el tamaño "
                 "de posición ajustado por volatilidad tiene sentido en este activo."),

    dict(clave='dependencia_dia_estado', titulo='Dependencia diaria',
         sub=_sub_dependencia('dia'), fuente='global', formato=lambda v: v,
         color=_color_dependencia, compacto=True, tooltip=_tip_dependencia('dia')),

    dict(clave='dependencia_semanal_estado', titulo='Dependencia semanal',
         sub=_sub_dependencia('semanal'), fuente='global', formato=lambda v: v,
         color=_color_dependencia, compacto=True, tooltip=_tip_dependencia('semanal')),

    dict(clave='dependencia_mensual_estado', titulo='Dependencia mensual',
         sub=_sub_dependencia('mensual'), fuente='global', formato=lambda v: v,
         color=_color_dependencia, compacto=True, tooltip=_tip_dependencia('mensual')),

    dict(clave='dependencia_trimestral_estado', titulo='Dependencia trimestral',
         sub=_sub_dependencia('trimestral'), fuente='global', formato=lambda v: v,
         color=_color_dependencia, compacto=True, tooltip=_tip_dependencia('trimestral')),

    dict(clave='er_medio', titulo='ER medio', sub='efficiency ratio', fuente='horizonte',
         formato=lambda v: _num(v, 3), color=lambda v: _escala(v, 0.50, 0.30),
         tooltip="Efficiency Ratio medio de la Ventana: cuánto del recorrido total "
                 "es movimiento direccional y cuánto es ida y vuelta."),

    dict(clave='hurst_medio', titulo='Hurst medio', sub=_sub_hurst, fuente='horizonte',
         formato=lambda v: _num(v, 3), color=_color_hurst,
         tooltip="Exponente de Hurst: >0.5 indica persistencia (tendencia), <0.5 "
                 "reversión a la media, ≈0.5 paseo aleatorio."),
]


def _con_contexto(fn):
    """Normaliza un formateador/colorizador a la forma `fn(valor, ctx)`.

    Casi todos solo necesitan el valor; unos pocos (Half-Life) necesitan además
    el contexto para leer el timeframe. Se decide contando los parámetros
    OBLIGATORIOS — mirar el total confundiría a `_pct(v, dec=2)` con uno que
    pide contexto y le pasaría el dict como número de decimales.
    """
    try:
        parametros = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return lambda v, ctx: fn(v)
    obligatorios = sum(
        1 for p in parametros
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                       inspect.Parameter.POSITIONAL_OR_KEYWORD))
    return fn if obligatorios >= 2 else (lambda v, ctx: fn(v))


for _spec in _KPIS:
    _spec['formato'] = _con_contexto(_spec['formato'])
    _spec['color'] = _con_contexto(_spec['color'])


class _GaugeLineal(QWidget):
    """Línea con 2 umbrales / 3 zonas de color y una flecha señalando la
    posición actual en el dominio [0, escala_max]. Reutilizado por las
    tarjetas ER medio y Max drawdown, cada una con sus propios colores,
    umbrales y formato de la etiqueta."""

    def __init__(self, posicion, texto_valor, umbral1, umbral2, colores,
                 escala_max=1.0, etiquetas=None, parent=None):
        super().__init__(parent)
        self._pos = posicion          # ya en el dominio [0, escala_max], o None
        self._texto = texto_valor     # texto junto a la flecha (ya formateado)
        self._u1, self._u2 = umbral1, umbral2
        self._colores = colores       # (zona_baja, zona_media, zona_alta)
        self._escala_max = escala_max
        self._etiquetas = etiquetas or (
            '0', f'{umbral1:g}', f'{umbral2:g}', f'{escala_max:g}')
        self.setFixedSize(260, 56)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        margen, y = 14, 32
        x0, x1 = margen, self.width() - margen
        ancho = x1 - x0
        c_baja, c_media, c_alta = self._colores

        def a_x(v):
            return x0 + max(0.0, min(1.0, v / self._escala_max)) * ancho

        # línea por zonas de color
        for ini, fin, color in ((0.0, self._u1, c_baja),
                                 (self._u1, self._u2, c_media),
                                 (self._u2, self._escala_max, c_alta)):
            pen = QPen(QColor(color), 5)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            p.drawLine(int(a_x(ini)), y, int(a_x(fin)), y)

        # marcas y etiquetas: extremos + los 2 umbrales
        p.setPen(QPen(QColor(AX_FG), 1))
        fuente = p.font()
        fuente.setPointSize(7)
        p.setFont(fuente)
        for v, etiqueta in zip((0.0, self._u1, self._u2, self._escala_max),
                               self._etiquetas):
            xv = a_x(v)
            p.drawLine(int(xv), y - 5, int(xv), y + 5)
            p.drawText(int(xv) - 20, y + 8, 40, 14,
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, etiqueta)

        # flecha + valor, apuntando hacia la línea desde arriba
        if self._pos is not None:
            xv = a_x(self._pos)
            color = (c_baja if self._pos < self._u1 else
                     c_alta if self._pos > self._u2 else c_media)
            p.setBrush(QColor(color))
            p.setPen(Qt.PenStyle.NoPen)
            flecha = QPolygonF([QPointF(xv, y - 8), QPointF(xv - 5, y - 17),
                                QPointF(xv + 5, y - 17)])
            p.drawPolygon(flecha)
            p.setPen(QPen(QColor(color)))
            fuente_val = p.font()
            fuente_val.setBold(True)
            p.setFont(fuente_val)
            p.drawText(int(xv) - 30, y - 32, 60, 14,
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                       self._texto)


def _gauge_er(valor):
    texto = '' if valor is None else f'{valor:.3f}'
    return _GaugeLineal(valor, texto, 0.30, 0.50,
                        (ER_RUIDO, ER_TRANSICION, ER_TENDENCIA),
                        etiquetas=('0', '0.30', '0.50', '1'))


def _gauge_max_dd(valor):
    # Dominio visual acotado a 60%: por debajo de eso hay resolución fina
    # en el tramo relevante (0-40%); un drawdown peor simplemente clava la
    # flecha en el borde derecho (sigue en rojo, con la cifra real en el
    # texto) en vez de desperdiciar la mitad de la barra en valores >60%
    # que casi nunca ocurren.
    texto = '' if valor is None else f'{valor * 100:+.2f}%'
    pos = None if valor is None else abs(valor)
    return _GaugeLineal(pos, texto, 0.20, 0.40, (VERDE, AMBAR, ROJO),
                        escala_max=0.60, etiquetas=('0%', '20%', '40%', '≥60%'))


# Gauges definidos aquí (después de _KPIS, que ya está construida arriba)
# porque referencian los propios umbrales/colores de cada tarjeta; se
# asignan a posteriori por clave, mismo patrón que _con_contexto arriba.
_GAUGES = {'er_medio': _gauge_er, 'max_dd': _gauge_max_dd}
for _spec in _KPIS:
    _spec['gauge'] = _GAUGES.get(_spec['clave'])


class _Tarjeta(QFrame):
    """Tarjeta de un KPI: título, valor grande coloreado y subtítulo.

    `gauge_fn` (solo ER medio y Max drawdown): callable `valor_crudo ->
    QWidget` que construye un gauge visual. Cuando se da, en vez del
    tooltip nativo de Qt (que no puede alojar un widget pintado a medida)
    la tarjeta gestiona a mano un popup de hover propio con el texto de
    siempre + el gauge debajo — mismo retardo y comportamiento que un
    tooltip normal, pero con contenido gráfico."""

    def __init__(self, titulo, subtitulo, tooltip, compacto=False, gauge_fn=None,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("tarjeta")
        self.setFixedWidth(_ANCHO_TARJETA)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._sub_por_defecto = subtitulo
        self._gauge_fn = gauge_fn
        self._texto_ayuda = tooltip
        self._valor_crudo = None
        self._popup_hover = None

        if self._gauge_fn is not None:
            self._timer_hover = QTimer(self)
            self._timer_hover.setSingleShot(True)
            self._timer_hover.setInterval(350)
            self._timer_hover.timeout.connect(self._mostrar_popup_hover)
        else:
            self.setToolTip(tooltip)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(1)

        self.lbl_titulo = QLabel(titulo)
        self.lbl_titulo.setObjectName("kpiTitulo")
        self.lbl_valor = QLabel("—")
        self.lbl_valor.setObjectName("kpiValorTexto" if compacto else "kpiValor")
        self.lbl_sub = QLabel(subtitulo)
        self.lbl_sub.setObjectName("kpiSub")

        lay.addWidget(self.lbl_titulo)
        lay.addWidget(self.lbl_valor)
        lay.addWidget(self.lbl_sub)
        self._pintar_borde(GRIS)

    def _pintar_borde(self, color):
        # Franja de color a la izquierda, igual que las tarjetas del Backtester.
        self.setStyleSheet(
            f"QFrame#tarjeta {{ background-color: #141e30; border: 1px solid #253a60;"
            f" border-left: 4px solid {color}; border-radius: 6px; }}")

    def set_valor(self, texto, color, crudo=None):
        self.lbl_valor.setText(texto)
        self.lbl_valor.setStyleSheet(f"color: {color};")
        self._pintar_borde(color)
        if self._gauge_fn is not None:
            self._valor_crudo = crudo

    def set_subtitulo(self, texto):
        self.lbl_sub.setText(texto or self._sub_por_defecto)

    def set_tooltip(self, texto):
        if self._gauge_fn is not None:
            self._texto_ayuda = texto
        else:
            self.setToolTip(texto)

    def vaciar(self):
        self.lbl_valor.setText("—")
        self.lbl_valor.setStyleSheet(f"color: {GRIS};")
        self.lbl_sub.setText(self._sub_por_defecto)
        self._pintar_borde(GRIS)
        if self._gauge_fn is not None:
            self._valor_crudo = None

    # ── popup de hover con gauge (solo tarjetas con gauge_fn) ──
    def enterEvent(self, event):
        if self._gauge_fn is not None:
            self._timer_hover.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._gauge_fn is not None:
            self._timer_hover.stop()
            if self._popup_hover is not None:
                self._popup_hover.close()
                self._popup_hover = None
        super().leaveEvent(event)

    def _mostrar_popup_hover(self):
        popup = QWidget(self, Qt.WindowType.ToolTip)
        popup.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        popup.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        popup.setStyleSheet(
            "QWidget { background-color: #0d1424; border: 2px solid #253a60;"
            " border-radius: 6px; }"
            "QLabel { color: #c8d6e5; font-size: 11px; }")
        lay = QVBoxLayout(popup)
        lay.setContentsMargins(10, 10, 10, 8)
        lay.setSpacing(6)
        lbl = QLabel(self._texto_ayuda)
        lbl.setWordWrap(True)
        lbl.setFixedWidth(260)
        lay.addWidget(lbl)
        lay.addWidget(self._gauge_fn(self._valor_crudo))
        popup.move(self.mapToGlobal(self.rect().bottomLeft()))
        popup.show()
        self._popup_hover = popup


class TarjetasKPI(QWidget):
    """Rejilla de tarjetas que se reajusta al ancho disponible.

    Si el análisis no trae KPIs (informes generados antes de esta versión), el
    widget se oculta entero y la pestaña muestra solo las categorías de siempre.
    Las claves que falten en bundles más antiguos dejan su tarjeta en «—» sin
    afectar al resto de la fila.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tarjetasKpi")
        self.setStyleSheet(ESTILO)
        self._kpi = None
        self._meta = None
        self._horizonte = 'General'
        self._columnas = 0

        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 4)
        self._grid.setHorizontalSpacing(8)
        self._grid.setVerticalSpacing(8)

        self._tarjetas = {}
        for spec in _KPIS:
            sub = spec['sub']
            self._tarjetas[spec['clave']] = _Tarjeta(
                spec['titulo'], sub if isinstance(sub, str) else '',
                spec['tooltip'] if isinstance(spec['tooltip'], str) else '',
                compacto=spec.get('compacto', False),
                gauge_fn=spec.get('gauge'))
        self._redistribuir(4)

    # ── API ──
    def cargar(self, bundle, horizonte=None):
        bundle = bundle or {}
        self._kpi = bundle.get('kpi')
        self._meta = bundle.get('_meta') or {}
        if horizonte:
            self._horizonte = horizonte
        self.setVisible(self._kpi is not None)
        self._refrescar()

    def set_horizonte(self, horizonte):
        self._horizonte = horizonte
        self._refrescar()

    # ── interno ──
    def _refrescar(self):
        if not self._kpi:
            for t in self._tarjetas.values():
                t.vaciar()
            return

        ctx = {
            'global': self._kpi.get('global') or {},
            'horizonte': (self._kpi.get('por_horizonte') or {}).get(self._horizonte) or {},
            'meta': self._meta or {},
        }

        for spec in _KPIS:
            tarjeta = self._tarjetas[spec['clave']]
            origen = ctx['horizonte'] if spec['fuente'] == 'horizonte' else ctx['global']
            valor = origen.get(spec['clave'])
            if valor is None:
                tarjeta.vaciar()
                continue
            try:
                tarjeta.set_valor(spec['formato'](valor, ctx),
                                  spec['color'](valor, ctx), crudo=valor)
                if not isinstance(spec['sub'], str):
                    tarjeta.set_subtitulo(spec['sub'](ctx))
                if not isinstance(spec['tooltip'], str):
                    tarjeta.set_tooltip(spec['tooltip'](ctx))
            except (TypeError, ValueError, ZeroDivisionError, KeyError):
                tarjeta.vaciar()

    def _redistribuir(self, columnas):
        if columnas == self._columnas or columnas < 1:
            return
        self._columnas = columnas
        while self._grid.count():
            self._grid.takeAt(0)
        for i, spec in enumerate(_KPIS):
            self._grid.addWidget(self._tarjetas[spec['clave']], i // columnas,
                                 i % columnas, Qt.AlignmentFlag.AlignLeft)
        self._grid.setColumnStretch(columnas, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        ancho = max(self.width(), _ANCHO_TARJETA)
        self._redistribuir(max(1, min(len(_KPIS), ancho // (_ANCHO_TARJETA + 8))))


