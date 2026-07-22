"""
Iconos dibujados a mano con QPainter para el asistente "Cómo funciona"
(gui/dialogs/tutorial_dialog.py) — mismo patrón que el icono "home" de
gui/widgets/file_explorer.py: sin emoji, sin archivos de imagen externos,
solo contorno fino sobre un QPixmap transparente.
"""
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QPen, QBrush, QColor, QPolygonF


def _pt(tam, fx, fy):
    return QPointF(tam * fx, tam * fy)


def _poly(tam, *fracciones):
    return QPolygonF([_pt(tam, fx, fy) for fx, fy in fracciones])


def dibujar_vision_general(p, tam):
    p.drawLine(_pt(tam, 0.16, 0.5), _pt(tam, 0.78, 0.5))
    r = tam * 0.05
    p.drawEllipse(_pt(tam, 0.16, 0.5), r, r)
    p.drawEllipse(_pt(tam, 0.5, 0.5), r, r)
    p.drawLine(_pt(tam, 0.69, 0.41), _pt(tam, 0.78, 0.5))
    p.drawLine(_pt(tam, 0.69, 0.59), _pt(tam, 0.78, 0.5))


def dibujar_descargar(p, tam):
    p.drawLine(_pt(tam, 0.5, 0.16), _pt(tam, 0.5, 0.58))
    p.drawLine(_pt(tam, 0.36, 0.44), _pt(tam, 0.5, 0.58))
    p.drawLine(_pt(tam, 0.64, 0.44), _pt(tam, 0.5, 0.58))
    p.drawLine(_pt(tam, 0.22, 0.80), _pt(tam, 0.78, 0.80))


def dibujar_importar(p, tam):
    p.drawLine(_pt(tam, 0.5, 0.14), _pt(tam, 0.5, 0.46))
    p.drawLine(_pt(tam, 0.38, 0.34), _pt(tam, 0.5, 0.46))
    p.drawLine(_pt(tam, 0.62, 0.34), _pt(tam, 0.5, 0.46))
    p.drawLine(_pt(tam, 0.22, 0.52), _pt(tam, 0.22, 0.82))
    p.drawLine(_pt(tam, 0.78, 0.52), _pt(tam, 0.78, 0.82))
    p.drawLine(_pt(tam, 0.22, 0.82), _pt(tam, 0.78, 0.82))


def dibujar_limpiador(p, tam):
    p.drawPolygon(_poly(tam, (0.16, 0.20), (0.84, 0.20), (0.60, 0.55), (0.40, 0.55)))
    p.drawLine(_pt(tam, 0.5, 0.55), _pt(tam, 0.5, 0.84))


def dibujar_metodos(p, tam):
    p.drawLine(_pt(tam, 0.16, 0.82), _pt(tam, 0.84, 0.82))
    p.drawLine(_pt(tam, 0.30, 0.82), _pt(tam, 0.30, 0.55))
    p.drawLine(_pt(tam, 0.5, 0.82), _pt(tam, 0.5, 0.34))
    p.drawLine(_pt(tam, 0.70, 0.82), _pt(tam, 0.70, 0.46))


def dibujar_empieza(p, tam):
    p.drawPolygon(_poly(tam, (0.34, 0.22), (0.34, 0.78), (0.76, 0.5)))


def dibujar_analizador(p, tam):
    p.drawLine(_pt(tam, 0.16, 0.20), _pt(tam, 0.16, 0.82))
    p.drawLine(_pt(tam, 0.16, 0.82), _pt(tam, 0.84, 0.82))
    linea = _poly(tam, (0.22, 0.72), (0.42, 0.52), (0.58, 0.62), (0.78, 0.30))
    p.drawPolyline(linea)
    r = tam * 0.045
    p.drawEllipse(_pt(tam, 0.78, 0.30), r, r)


def dibujar_comparador(p, tam):
    p.drawPolygon(_poly(tam, (0.42, 0.76), (0.58, 0.76), (0.5, 0.58)))
    p.drawLine(_pt(tam, 0.5, 0.58), _pt(tam, 0.5, 0.28))
    p.drawLine(_pt(tam, 0.20, 0.28), _pt(tam, 0.80, 0.28))
    p.drawLine(_pt(tam, 0.20, 0.28), _pt(tam, 0.20, 0.40))
    p.drawLine(_pt(tam, 0.80, 0.28), _pt(tam, 0.80, 0.40))
    r = tam * 0.06
    p.drawEllipse(_pt(tam, 0.20, 0.46), r, r)
    p.drawEllipse(_pt(tam, 0.80, 0.46), r, r)


def dibujar_backtester(p, tam):
    p.drawPolygon(_poly(tam, (0.42, 0.18), (0.58, 0.18), (0.72, 0.80), (0.28, 0.80)))
    p.drawLine(_pt(tam, 0.35, 0.58), _pt(tam, 0.65, 0.58))


def _icono(dibujar, tam=64, color='#4fc3f7', grosor=2):
    pix = QPixmap(tam, tam)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(color), grosor))
    p.setBrush(QBrush())
    dibujar(p, tam)
    p.end()
    return QIcon(pix)


def icono_pixmap(dibujar, tam=32, color='#4fc3f7', grosor=2):
    return _icono(dibujar, tam, color, grosor).pixmap(tam, tam)


def icono_tab(dibujar, tam=18, color='#4fc3f7', grosor=2):
    return _icono(dibujar, tam, color, grosor)
