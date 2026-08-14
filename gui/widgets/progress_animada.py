"""QProgressBar con relleno degradado y rayas diagonales animadas.

Qt no anima los QSS: este widget pinta el fondo, el relleno y las rayas a
mano y usa un único QTimer compartido a nivel de módulo (no uno por barra,
importante en páginas con muchas mini-barras). Solo anima mientras la barra
está visible y «activa»: porcentaje intermedio (0 < valor < máximo) o modo
indeterminado (máximo == 0). Al llegar al 100 %, al ocultarse o en reposo el
timer se desentiende de ella (el timer se para solo cuando no queda ninguna
barra activa).

Modo indeterminado: un bloque de ~35 % del ancho se desliza de lado a lado
con las mismas rayas. El texto (porcentaje) se dibuja centrado cuando está
activo `textVisible()`, arreglando el desalineado del texto por defecto.
"""

import time
import weakref

from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QColor, QPainter, QLinearGradient, QPainterPath, QPolygonF
from PyQt6.QtWidgets import QProgressBar

_BG = QColor('#1a2a45')
_CHUNK_ALTA = QColor('#6dd5fa')
_CHUNK_BAJA = QColor('#4fc3f7')
_TEXTO = QColor('#c8d6e5')
_RAYA = QColor(255, 255, 255, 26)

_VEL = 45.0              # px/s de avance de las rayas
_PERIODO_BLOQUE = 1.6    # s por ida y vuelta del bloque indeterminado

_timer = None
_t0 = 0.0
_barras = weakref.WeakSet()


def _latido():
    if not _barras:
        _timer.stop()
        return
    ahora = time.monotonic() - _t0
    for barra in list(_barras):
        barra._avanzar(ahora)


def _asegurar_timer():
    global _timer, _t0
    if _timer is None:
        _timer = QTimer()
        _timer.setInterval(33)
        _timer.timeout.connect(_latido)
    if not _timer.isActive():
        _t0 = time.monotonic()
        _timer.start()


class ProgressBarAnimada(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tiempo = 0.0
        self._activa = False

    # ── ciclo de actividad ──

    def _actividad(self):
        if not self.isVisible():
            return False
        if self.maximum() == 0:
            return True
        return 0 < self.value() < self.maximum()

    def _sincronizar(self):
        activa = self._actividad()
        if activa and self not in _barras:
            _barras.add(self)
            _asegurar_timer()
        elif not activa:
            _barras.discard(self)
        self._activa = activa

    def _avanzar(self, ahora):
        self._tiempo = ahora
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self._sincronizar()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._sincronizar()

    def setValue(self, value):
        super().setValue(value)
        self._sincronizar()

    def setRange(self, minimum, maximum):
        super().setRange(minimum, maximum)
        self._sincronizar()

    # ── pintado ──

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        r = max(2.0, min(h / 2.0, h * 0.45))
        rect = QRectF(0, 0, w, h)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_BG)
        p.drawRoundedRect(rect, r, r)

        indet = self.maximum() == 0
        if indet:
            bloque = w * 0.35
            recorrido = max(1.0, w - bloque)
            t = (self._tiempo / _PERIODO_BLOQUE) % 2.0
            pos = (1.0 - abs(t - 1.0)) * recorrido
            zona = QRectF(pos, 0, bloque, h)
        else:
            rango = max(1, self.maximum() - self.minimum())
            frac = (self.value() - self.minimum()) / rango
            zona = QRectF(0, 0, w * frac, h)

        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(rect, r, r)
        p.setClipPath(clip)

        if zona.width() >= 2:
            grad = QLinearGradient(0, 0, 0, h)
            grad.setColorAt(0.0, QColor(255, 255, 255, 70))
            grad.setColorAt(0.45, _CHUNK_ALTA)
            grad.setColorAt(0.8, _CHUNK_BAJA)
            grad.setColorAt(1.0, QColor(0, 0, 0, 40))
            p.setBrush(grad)
            p.drawRect(zona)

        if self._activa:
            self._pintar_rayas(p, QRectF(0, 0, w, h), h)

        p.restore()

        if self.isTextVisible():
            texto = self.text()
            if texto:
                p.setPen(_TEXTO)
                fuente = self.font()
                fuente.setPixelSize(max(7, min(h - 4, 12)))
                fuente.setBold(True)
                p.setFont(fuente)
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, texto)

        p.end()

    def _pintar_rayas(self, p, zona, h):
        grueso = max(5.0, h * 0.55)
        paso = grueso * 2
        desliz = (self._tiempo * _VEL) % paso
        inclin = h * 0.9
        x = zona.left() - paso + desliz
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_RAYA)
        while x < zona.right():
            p.drawPolygon(QPolygonF([
                QPointF(x, zona.bottom()),
                QPointF(x + inclin, zona.top()),
                QPointF(x + inclin + grueso, zona.top()),
                QPointF(x + grueso, zona.bottom()),
            ]))
            x += paso
