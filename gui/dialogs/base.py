"""
gui/dialogs/base.py
Base de diálogos sin marco de Windows con cabecera propia (título + ✕) y
arrastre, en la misma línea visual que los paneles flotantes de la app
(plot_common.PanelFlotanteDialog / montar_panel_flotante). Sirve para que
ningún diálogo necesite la barra de título nativa del sistema.
"""
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QWidget)
from PyQt6.QtWidgets import QGraphicsDropShadowEffect
from PyQt6.QtGui import QColor

_PANEL_QSS = """
QDialog { background: transparent; }
QLabel { color: #dbe8f5; font-size: 12px; }
QLabel#dlgTitulo { color: #4fc3f7; font-size: 13px; font-weight: bold; }
QLabel#dlgSubtitulo { color: #5a7a9a; font-size: 11px; }
QLabel#dlgTexto { color: #dbe8f5; font-size: 12px; }
QFrame#dlgSep { background-color: #253a60; max-height: 1px; border: none; }
QCheckBox { color: #c8d6e5; font-size: 12px; }
QCheckBox:disabled { color: #3a5a7a; }
QRadioButton { color: #c8d6e5; font-size: 12px; }
QGroupBox { color: #8fb3d9; font-size: 12px; border: 1px solid #253a60;
    border-radius: 4px; margin-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background-color: #2a4a6a; color: #4fc3f7; border: none;
    padding: 8px 18px; border-radius: 4px; font-size: 12px; font-weight: bold; }
QPushButton:hover { background-color: #3a5a8a; }
QPushButton#dlgCerrar { background: transparent; color: #7a90ad; border: none;
    font-size: 15px; font-weight: bold; padding: 0 7px; border-radius: 5px; }
QPushButton#dlgCerrar:hover { color: #e74c3c; background-color: #2a1a1a; }
QPushButton#accionOk { background-color: #0f2a1a; color: #2ecc71; }
QPushButton#accionOk:hover { background-color: #1a3a2a; }
QPushButton#accionPeligro { background-color: #3a1a1a; color: #e74c3c; }
QPushButton#accionPeligro:hover { background-color: #4a2525; }
QPushButton#accionNeutra { background-color: #222a3a; color: #5a7a9a; }
QPushButton#accionNeutra:hover { background-color: #2a3a4a; }
QLineEdit { background-color: #1a2a45; color: #c8d6e5; border: 1px solid #253a60;
    border-radius: 4px; padding: 8px 10px; font-size: 12px; }
QLineEdit:focus { border: 1px solid #3a5a8a; }
"""


class DialogoBase(QDialog):
    """Diálogo sin marco con cabecera propia (título + subtítulo + ✕),
    arrastre y animación de entrada.

    Los llamadores construyen su contenido en `self.contenido` (QVBoxLayout)
    o, si prefieren la función de conveniencia, en el layout que devuelve
    `crear_dialogo`."""

    def __init__(self, titulo, parent=None, subtitulo='', ancho=None,
                 alto=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle(titulo)
        self.setStyleSheet(_PANEL_QSS)
        # Margen interior que deja sitio a la sombra y a las esquinas
        # redondeadas DENTRO de la ventana: si el panel llena la ventana
        # exactamente (margen 0), la sombra sobresale por el borde inferior
        # y Windows la recorta, cortando la curva de las esquinas de abajo
        # (se ve una arista/pico en vez de la curva suave). Con margen, el
        # render es idéntico al de montar_panel_flotante (QuestDB), que va
        # con 20px y se ve bien.
        _MARGEN_SOMBRA = 20
        if ancho is not None:
            self.setFixedWidth(ancho + 2 * _MARGEN_SOMBRA)
        if alto is not None:
            self.setFixedHeight(alto + 2 * _MARGEN_SOMBRA)

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(_MARGEN_SOMBRA, _MARGEN_SOMBRA,
                                _MARGEN_SOMBRA, _MARGEN_SOMBRA)
        raiz.setSpacing(0)

        panel = QFrame(self)
        panel.setObjectName("dlgPanel")
        panel.setStyleSheet(
            "QFrame#dlgPanel { background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            " stop:0 #1b2c4a, stop:1 #0d1424); border: 1px solid #2a4a6a;"
            " border-radius: 12px; }")
        sombra = QGraphicsDropShadowEffect(panel)
        sombra.setBlurRadius(18)
        sombra.setColor(QColor(0, 0, 0, 150))
        sombra.setOffset(0, 4)
        panel.setGraphicsEffect(sombra)
        raiz.addWidget(panel)

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 10, 14, 14)
        lay.setSpacing(10)

        cab = QWidget(panel)
        lay_cab = QHBoxLayout(cab)
        lay_cab.setContentsMargins(2, 0, 2, 0)
        lay_cab.setSpacing(8)
        punto = QLabel("●")
        punto.setStyleSheet("color: #4fc3f7; font-size: 12px;")
        lay_cab.addWidget(punto)
        col = QVBoxLayout()
        col.setSpacing(0)
        col.setContentsMargins(0, 0, 0, 0)
        lbl_titulo = QLabel(titulo)
        lbl_titulo.setObjectName("dlgTitulo")
        col.addWidget(lbl_titulo)
        self.lbl_sub = QLabel(subtitulo)
        self.lbl_sub.setObjectName("dlgSubtitulo")
        col.addWidget(self.lbl_sub)
        lay_cab.addLayout(col, 1)
        btn_x = QPushButton("✕")
        btn_x.setObjectName("dlgCerrar")
        btn_x.setFixedSize(28, 26)
        btn_x.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_x.clicked.connect(self.close)
        lay_cab.addWidget(btn_x)
        lay.addWidget(cab)

        sep = QFrame(panel)
        sep.setObjectName("dlgSep")
        sep.setFixedHeight(1)
        lay.addWidget(sep)

        self.contenido = QVBoxLayout()
        self.contenido.setContentsMargins(2, 0, 2, 0)
        self.contenido.setSpacing(8)
        lay.addLayout(self.contenido, 1)

        _instalar_arrastre(self, cab)
        _animar_entrada(self)


def crear_dialogo(titulo, parent=None, subtitulo='', ancho=None, alto=None):
    """Crea un diálogo sin marco listo para rellenar.

    Devuelve (dlg, lay_contenido, lbl_subtitulo). `lay_contenido` es un
    QVBoxLayout donde el llamador coloca sus widgets; `lbl_subtitulo` puede
    actualizarse en vivo. El diálogo se anima solo (fade + deslizamiento) al
    mostrarse."""
    dlg = DialogoBase(titulo, parent=parent, subtitulo=subtitulo,
                      ancho=ancho, alto=alto)
    return dlg, dlg.contenido, dlg.lbl_sub


def _instalar_arrastre(ventana, widget):
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


def _animar_entrada(widget):
    """Fade-in + deslizamiento sutil al mostrar el diálogo (misma animación
    que los paneles flotantes de plot_common.animar_entrada)."""
    from PyQt6.QtCore import QPropertyAnimation, QEasingCurve

    def _animar(event):
        pos_final = widget.pos()
        fade = QPropertyAnimation(widget, b"windowOpacity", widget)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setDuration(180)
        fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        slide = QPropertyAnimation(widget, b"pos", widget)
        slide.setStartValue(pos_final + QPoint(0, 8))
        slide.setEndValue(pos_final)
        slide.setDuration(180)
        slide.setEasingCurve(QEasingCurve.Type.OutCubic)
        widget._anim_fade = fade
        widget._anim_slide = slide
        fade.start()
        slide.start()

    widget._animado_entrada = False
    evento_original = widget.showEvent

    def _on_show(event):
        if not widget._animado_entrada:
            widget._animado_entrada = True
            _animar(event)
        if evento_original is not None:
            return evento_original(event)
        return None

    widget.showEvent = _on_show