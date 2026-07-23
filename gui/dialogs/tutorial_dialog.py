import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QWidget, QStackedWidget, QFrame,
                              QTabBar, QGraphicsOpacityEffect)
from PyQt6.QtCore import (Qt, QPoint, QPropertyAnimation, QParallelAnimationGroup,
                           QEasingCurve, QSize)
from PyQt6.QtGui import QIcon

from core.config import PROJECT_ROOT
from gui.dialogs.tutorial_icons import (
    icono_pixmap, icono_tab,
    dibujar_vision_general, dibujar_descargar, dibujar_importar,
    dibujar_limpiador, dibujar_metodos, dibujar_empieza,
    dibujar_analizador, dibujar_comparador, dibujar_backtester,
)

# Misma paleta que el resto de la app (HeaderBar/StatusBar/QTabBar en
# gui/main_window.py y los diálogos existentes) — un único acento cian,
# nada de colores saturados por página.
_SIDEBAR_BG = "#0d1424"
_PANEL_BG = "#141e30"
_BORDE = "#253a60"
_ACENTO = "#4fc3f7"
_TEXTO_APAGADO = "#5a7a9a"
_TEXTO_HOVER = "#7aaccc"
_CUERPO = "#8aa2ba"
_TARJETA_BG = "#1a2a45"

STYLE = f"""
QDialog {{ background-color: {_PANEL_BG}; }}
QLabel {{ color: #c8d6e5; }}
QFrame#barraSuperior {{ background-color: {_SIDEBAR_BG}; }}
QTabBar {{ background-color: {_SIDEBAR_BG}; }}
QTabBar::tab {{
    background-color: {_TARJETA_BG}; color: {_TEXTO_APAGADO};
    padding: 10px 8px; border: none; border-right: 1px solid {_SIDEBAR_BG};
    font-size: 11px;
}}
QTabBar::tab:selected {{ background-color: {_PANEL_BG}; color: {_ACENTO}; font-weight: bold; }}
QTabBar::tab:hover:!selected {{ background-color: #1e3050; color: {_TEXTO_HOVER}; }}
QPushButton {{
    background-color: #2a4a6a; color: {_ACENTO}; border: none;
    padding: 10px 22px; border-radius: 4px; font-size: 12px; font-weight: bold;
}}
QPushButton:hover {{ background-color: #3a5a8a; }}
QPushButton:pressed {{ padding-top: 12px; padding-bottom: 8px; }}
QPushButton:disabled {{ background-color: #1a2a45; color: #3a5a7a; }}
QPushButton#accept {{ background-color: #0f2a1a; color: #2ecc71; }}
QPushButton#accept:hover {{ background-color: #1a3a2a; }}
QPushButton#saltar {{ background-color: transparent; color: #3a5a7a; padding: 10px 8px; }}
QPushButton#saltar:hover {{ background-color: transparent; color: {_TEXTO_HOVER}; }}
QPushButton#cerrar {{ background: transparent; color: #7a9aba; border: none; font-size: 13px; border-radius: 0; padding: 0; }}
QPushButton#cerrar:hover {{ background-color: #c0392b; color: white; }}
"""

_DURACION_MS = 260
_DESPLAZAMIENTO_PX = 46

# (dibujar_xxx, nombre corto para la pestaña, título de la tarjeta, cuerpo).
# El cuerpo de "Métodos de análisis" se arma aparte con tres bloques, ver
# _PaginaMetodos (texto=None marca esa página).
_PASOS = [
    (dibujar_vision_general, "Visión general", "Un sistema, cuatro fases",
     "Los datos avanzan en una sola dirección: se descargan, se importan, "
     "se limpian. A partir de ahí, decides tú qué hacer con ellos."),
    (dibujar_descargar, "Descargar", "El punto de entrada",
     "Elige un proveedor y un activo, o carga tus propios CSV. Todo queda "
     "ordenado por activo y temporalidad en tu carpeta de datos."),
    (dibujar_importar, "Importar", "De archivo a sistema",
     "Los CSV pasan a formar parte del sistema. Si hace falta, la base de "
     "datos local se prepara sola en segundo plano — no hay nada que "
     "instalar a mano."),
    (dibujar_limpiador, "Limpiador", "Datos en los que confiar",
     "Se revisan huecos, duplicados y valores fuera de rango. Lo que sale "
     "de aquí es lo único que el resto del programa usa para analizar."),
    (dibujar_metodos, "Métodos de análisis", "Tres formas de mirar los mismos datos", None),
    (dibujar_empieza, "Empieza ya", "Vuelve cuando quieras",
     "Pulsa «?» en la barra superior para repetir este recorrido cuando lo "
     "necesites."),
]

_METODOS = [
    (dibujar_analizador, "Analizador", "estudia patrones y estadísticas de un activo."),
    (dibujar_comparador, "Comparador", "contrasta varios activos entre sí."),
    (dibujar_backtester, "Backtester", "prueba y optimiza sistemas de entrada/salida sobre el histórico."),
]


def _tarjeta_base(dibujar, titulo):
    contenedor = QWidget()
    lay = QVBoxLayout(contenedor)
    lay.setContentsMargins(28, 24, 28, 24)
    lay.setSpacing(12)
    lay.addStretch()

    insignia = QLabel()
    insignia.setFixedSize(64, 64)
    insignia.setAlignment(Qt.AlignmentFlag.AlignCenter)
    insignia.setPixmap(icono_pixmap(dibujar, tam=30))
    insignia.setStyleSheet(
        f"background-color: {_TARJETA_BG}; "
        f"border: 2px solid {_BORDE}; border-radius: 32px;")
    lay.addWidget(insignia, 0, Qt.AlignmentFlag.AlignHCenter)

    lbl_titulo = QLabel(titulo)
    lbl_titulo.setStyleSheet(f"color: {_ACENTO}; font-size: 19px; font-weight: bold;")
    lbl_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(lbl_titulo)

    return contenedor, lay


class _PaginaPaso(QWidget):
    def __init__(self, dibujar, titulo, cuerpo, parent=None):
        super().__init__(parent)
        contenedor, lay = _tarjeta_base(dibujar, titulo)
        lay_padre = QVBoxLayout(self)
        lay_padre.setContentsMargins(0, 0, 0, 0)
        lay_padre.addWidget(contenedor)

        desc = QLabel(cuerpo)
        desc.setStyleSheet(f"color: {_CUERPO}; font-size: 12px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        lay.addWidget(desc)
        lay.addStretch()


class _PaginaMetodos(QWidget):
    def __init__(self, dibujar, titulo, parent=None):
        super().__init__(parent)
        contenedor, lay = _tarjeta_base(dibujar, titulo)
        lay_padre = QVBoxLayout(self)
        lay_padre.setContentsMargins(0, 0, 0, 0)
        lay_padre.addWidget(contenedor)

        intro = QLabel("No hace falta pasar por los tres — elige el que responda a tu pregunta:")
        intro.setStyleSheet(f"color: {_CUERPO}; font-size: 12px;")
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro.setWordWrap(True)
        lay.addWidget(intro)

        fila = QHBoxLayout()
        fila.setSpacing(10)
        for m_dibujar, m_nombre, m_desc in _METODOS:
            tarjeta = QFrame()
            tarjeta.setStyleSheet(
                f"background-color: {_TARJETA_BG}; border: 1px solid {_BORDE};")
            tlay = QVBoxLayout(tarjeta)
            tlay.setContentsMargins(12, 12, 12, 12)
            tlay.setSpacing(4)

            cab_lay = QHBoxLayout()
            cab_lay.setSpacing(6)
            cab_icono = QLabel()
            cab_icono.setPixmap(icono_pixmap(m_dibujar, tam=16))
            cab_lay.addWidget(cab_icono)
            cab_texto = QLabel(m_nombre)
            cab_texto.setStyleSheet(f"color: {_ACENTO}; font-size: 13px; font-weight: bold; background: transparent; border: none;")
            cab_lay.addWidget(cab_texto)
            cab_lay.addStretch()
            tlay.addLayout(cab_lay)

            cuerpo = QLabel(m_desc)
            cuerpo.setStyleSheet(f"color: {_CUERPO}; font-size: 11px; background: transparent; border: none;")
            cuerpo.setWordWrap(True)
            tlay.addWidget(cuerpo)

            fila.addWidget(tarjeta)
        lay.addLayout(fila)
        lay.addStretch()


class _BarraSuperior(QFrame):
    """Barra superior propia (sin marco nativo de Windows): título +
    cierre, y arrastre de ventana con el ratón — mismo patrón que
    HeaderBar en gui/main_window.py."""

    def __init__(self, dialogo, parent=None):
        super().__init__(parent)
        self.setObjectName("barraSuperior")
        self.setFixedHeight(34)
        self._dialogo = dialogo
        self._drag_pos = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 0, 0)
        lay.setSpacing(6)

        icono = QLabel()
        icono.setPixmap(icono_pixmap(dibujar_vision_general, tam=15))
        lay.addWidget(icono)

        titulo = QLabel("Cómo funciona")
        titulo.setStyleSheet(
            f"color: {_ACENTO}; font-size: 12px; font-weight: bold; "
            "letter-spacing: 1px; background: transparent;")
        lay.addWidget(titulo)
        lay.addStretch()

        btn_cerrar = QPushButton("✕")
        btn_cerrar.setObjectName("cerrar")
        btn_cerrar.setFixedSize(44, 34)
        btn_cerrar.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cerrar.clicked.connect(dialogo.reject)
        lay.addWidget(btn_cerrar)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self._dialogo.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self._dialogo.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None


class TutorialDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analytics System · Cómo funciona")
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.resize(720, 500)
        self.setStyleSheet(STYLE)

        icon_path = os.path.join(PROJECT_ROOT, "icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self._anim_group = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(_BarraSuperior(self))

        contenido = QVBoxLayout()
        contenido.setContentsMargins(0, 0, 0, 0)
        contenido.setSpacing(0)
        root.addLayout(contenido, 1)

        # ── Fila de pestañas (lateral, una por paso) ──
        self.tabbar = QTabBar()
        self.tabbar.setExpanding(True)
        self.tabbar.setDrawBase(False)
        self.tabbar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tabbar.setIconSize(QSize(18, 18))
        for dibujar, nombre, _, _ in _PASOS:
            self.tabbar.addTab(icono_tab(dibujar), nombre)
        self.tabbar.currentChanged.connect(self._on_tab_changed)
        contenido.addWidget(self.tabbar)

        # ── Contenido de la fase activa ──
        self.stack = QStackedWidget()
        for dibujar, _, titulo, texto in _PASOS:
            if texto is None:
                pagina = _PaginaMetodos(dibujar, titulo)
            else:
                pagina = _PaginaPaso(dibujar, titulo, texto)
            self.stack.addWidget(pagina)
        contenido.addWidget(self.stack, 1)

        # ── Botonera inferior ──
        pie = QHBoxLayout()
        pie.setContentsMargins(20, 14, 20, 18)
        contenido.addLayout(pie)

        self.btn_saltar = QPushButton("Saltar")
        self.btn_saltar.setObjectName("saltar")
        self.btn_saltar.clicked.connect(self.accept)
        pie.addWidget(self.btn_saltar)

        pie.addStretch()

        self.btn_atras = QPushButton(" Atrás")
        self.btn_atras.clicked.connect(self._retroceder)
        pie.addWidget(self.btn_atras)

        self.btn_siguiente = QPushButton(" Siguiente")
        self.btn_siguiente.setObjectName("accept")
        self.btn_siguiente.clicked.connect(self._avanzar)
        pie.addWidget(self.btn_siguiente)

        self._actualizar_pagina()

    def _actualizar_pagina(self):
        idx = self.tabbar.currentIndex()
        self.btn_atras.setEnabled(idx > 0)
        es_ultima = idx == len(_PASOS) - 1
        self.btn_siguiente.setText(" Finalizar" if es_ultima else " Siguiente")
        self.btn_saltar.setVisible(not es_ultima)

    def _on_tab_changed(self, idx):
        idx_anterior = self.stack.currentIndex()
        if idx == idx_anterior:
            return
        direccion = 1 if idx > idx_anterior else -1
        self.stack.setCurrentIndex(idx)
        self._actualizar_pagina()
        self._animar_entrada(direccion)

    def _animar_entrada(self, direccion):
        """Desliza la página nueva desde la derecha (avanzar) o la
        izquierda (retroceder) con fundido de opacidad — sólo anima la
        página entrante, no la saliente."""
        if self._anim_group is not None:
            self._anim_group.stop()

        widget = self.stack.currentWidget()
        pos_final = widget.pos()
        pos_inicio = pos_final + QPoint(_DESPLAZAMIENTO_PX * direccion, 0)

        efecto = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(efecto)

        anim_pos = QPropertyAnimation(widget, b"pos", self)
        anim_pos.setDuration(_DURACION_MS)
        anim_pos.setStartValue(pos_inicio)
        anim_pos.setEndValue(pos_final)
        anim_pos.setEasingCurve(QEasingCurve.Type.OutCubic)

        anim_op = QPropertyAnimation(efecto, b"opacity", self)
        anim_op.setDuration(_DURACION_MS)
        anim_op.setStartValue(0.0)
        anim_op.setEndValue(1.0)

        grupo = QParallelAnimationGroup(self)
        grupo.addAnimation(anim_pos)
        grupo.addAnimation(anim_op)
        grupo.start()
        self._anim_group = grupo

    def _avanzar(self):
        idx = self.tabbar.currentIndex()
        if idx == len(_PASOS) - 1:
            self.accept()
            return
        self.tabbar.setCurrentIndex(idx + 1)

    def _retroceder(self):
        idx = self.tabbar.currentIndex()
        if idx > 0:
            self.tabbar.setCurrentIndex(idx - 1)
