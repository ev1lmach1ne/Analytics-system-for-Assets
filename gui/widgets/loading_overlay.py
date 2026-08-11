"""Overlay de carga: tapa toda la ventana con un fondo oscurecido y un
spinner animado + texto (+ barra de progreso opcional), para que el usuario
sepa que la app todavía está preparándose y no pueda clicar pestañas a medio
cargar.

Uso (desde la ventana principal):
    overlay = LoadingOverlay(ventana)
    overlay.begin("Preparando Backtester…")
    ...
    overlay.end()

Con barra de progreso:
    overlay.begin("Ejecutando backtest…", con_barra=True)
    overlay.set_progreso(i, total)   # total>0: determinada; 0: indeterminada
    ...
    overlay.end()

El overlay se redimensiona solo cuando la ventana cambia de tamaño. El
fade in/out replica el del splash de arranque (300ms in / 400ms out).
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QProgressBar,
                             QGraphicsOpacityEffect)
from PyQt6.QtCore import (Qt, QTimer, QRectF, QEvent, QPropertyAnimation,
                          QEasingCurve)

_FADE_IN_MS = 300
_FADE_OUT_MS = 400


class _Spinner(QWidget):
    """Arco rotatorio pintado a mano; no depende de recursos externos."""

    def __init__(self, size=56, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angulo = 0
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._avanzar)

    def start(self):
        if not self._timer.isActive():
            self._timer.start()

    def stop(self):
        self._timer.stop()

    def _avanzar(self):
        self._angulo = (self._angulo + 12) % 360
        self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QColor, QPen
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        rect = QRectF((self.width() - side) / 2.0,
                      (self.height() - side) / 2.0,
                      side, side).adjusted(3, 3, -3, -3)

        trazo = QPen(QColor('#1a2a45'), 4)
        trazo.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(trazo)
        painter.drawArc(rect, 0, 360 * 16)

        arco = QPen(QColor('#4fc3f7'), 4)
        arco.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(arco)
        painter.drawArc(rect, -self._angulo * 16, 100 * 16)
        painter.end()


class LoadingOverlay(QWidget):
    """Tapa completa de la ventana con fondo oscuro + spinner + texto +
    barra de progreso opcional, con fade in/out."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Fondo totalmente opaco: la UI que hay debajo (pestañas a medio
        # construir, placeholders) no debe verse a través del overlay.
        self.setStyleSheet("background-color: #0d1424;")
        self._padre = parent
        if parent is not None:
            parent.installEventFilter(self)

        self._efecto = QGraphicsOpacityEffect(self)
        self._efecto.setOpacity(1.0)
        self.setGraphicsEffect(self._efecto)
        self._anim = QPropertyAnimation(self._efecto, b"opacity", self)

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(18)

        self._spinner = _Spinner(56, self)
        lay.addWidget(self._spinner, alignment=Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel("Cargando…", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "color: #c8d6e5; font-size: 15px; background: transparent;")
        lay.addWidget(self._label, alignment=Qt.AlignmentFlag.AlignCenter)

        self._barra = QProgressBar(self)
        self._barra.setRange(0, 0)
        self._barra.setFixedWidth(320)
        self._barra.setFixedHeight(8)
        self._barra.setTextVisible(False)
        self._barra.setStyleSheet(
            "QProgressBar { background-color: #1a2a45; border: none;"
            " border-radius: 4px; }"
            "QProgressBar::chunk { background-color: #4fc3f7; border-radius: 4px; }")
        self._barra.hide()
        lay.addWidget(self._barra, alignment=Qt.AlignmentFlag.AlignCenter)

        self.hide()

    def eventFilter(self, obj, event):
        if obj is self._padre and event.type() == QEvent.Type.Resize:
            self._ajustar_geometria()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._ajustar_geometria()

    def _ajustar_geometria(self):
        if self._padre is not None:
            self.setGeometry(self._padre.rect())

    def begin(self, texto="Cargando…", con_barra=False):
        self._ajustar_geometria()
        self.set_text(texto)
        self._spinner.start()
        if con_barra:
            self._barra.setRange(0, 0)
            self._barra.setValue(0)
            self._barra.show()
        else:
            self._barra.hide()
        # Si hay un fade out en vuelo, interrumpirlo y arrancar desde visible.
        self._anim.stop()
        try:
            self._anim.finished.disconnect(self._finalizar_ocultar)
        except TypeError:
            pass
        self._efecto.setOpacity(0.0)
        self.show()
        self.raise_()
        self._anim.setDuration(_FADE_IN_MS)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()

    def set_text(self, texto):
        self._label.setText(texto)

    def set_progreso(self, i, total):
        total = max(int(total), 0)
        i = max(0, min(int(i), total)) if total > 0 else 0
        if total > 0:
            self._barra.setRange(0, total)
            self._barra.setValue(i)
        else:
            self._barra.setRange(0, 0)

    def end(self):
        self._anim.stop()
        self._anim.setDuration(_FADE_OUT_MS)
        self._anim.setStartValue(self._efecto.opacity())
        self._anim.setEndValue(0.0)
        self._anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._anim.finished.connect(self._finalizar_ocultar)
        self._anim.start()

    def apagar(self):
        """Cierre INMEDIATO (sin animación), para el teardown de la app: una
        animación de fade en vuelo al destruir el árbol corrompe el montón."""
        self._anim.stop()
        try:
            self._anim.finished.disconnect(self._finalizar_ocultar)
        except TypeError:
            pass
        self._spinner.stop()
        self.hide()
        self._efecto.setOpacity(1.0)

    def _finalizar_ocultar(self):
        try:
            self._anim.finished.disconnect(self._finalizar_ocultar)
        except TypeError:
            pass
        self._spinner.stop()
        self.hide()
        self._efecto.setOpacity(1.0)


def overlay_ventana(widget):
    """Overlay de la ventana principal (el de MainWindow), o None si el
    widget aún no está dentro de una ventana que lo tenga."""
    ventana = widget.window() if widget is not None else None
    if ventana is not None:
        return getattr(ventana, 'overlay', None)
    return None


def mostrar_overlay_tab(widget, texto, con_barra=False):
    """Overlay de carga DENTRO de una pestaña (solo cubre su área, no bloquea
    el resto de la ventana). Se crea perezosamente y se reutiliza."""
    overlay = getattr(widget, '_overlay_carga', None)
    if overlay is None:
        overlay = LoadingOverlay(widget)
        widget._overlay_carga = overlay
    overlay.begin(texto, con_barra=con_barra)
    return overlay


def ocultar_overlay_tab(widget):
    overlay = getattr(widget, '_overlay_carga', None)
    if overlay is not None:
        overlay.end()


def apagar_overlay_tab(widget):
    """Cierre inmediato (sin fade) del overlay de la pestaña — se usa al
    apagar la app, donde una animación en vuelo corrompe el montón."""
    overlay = getattr(widget, '_overlay_carga', None)
    if overlay is not None:
        overlay.apagar()
