from PyQt6.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QLabel, QPushButton, QHBoxLayout, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QFont, QPixmap
from gui.widgets.tab_descargar import TabDescargar
from gui.widgets.tab_importar import TabImportar
from gui.widgets.tab_analisis import TabAnalisis
from gui.widgets.tab_limpiados import TabLimpiados

STYLE = """
QMainWindow, QWidget { background-color: #111828; color: #c8d6e5; }
QTabWidget::pane { background-color: #141e30; border: 1px solid #253a60; border-top: none; }
QTabBar { background-color: #0d1424; border: none; }
QTabBar::tab { background-color: #1a2a45; color: #5a7a9a; padding: 12px 30px;
              border: none; border-right: 1px solid #253a60; font-size: 12px; }
QTabBar::tab:selected { background-color: #141e30; color: #4fc3f7; font-weight: bold; }
QTabBar::tab:hover:!selected { background-color: #1e3050; color: #7aaccc; }
QTabBar::tab:pressed { background-color: #0d1a30; }
QScrollBar:vertical { background: #1a2a45; width: 8px; }
QScrollBar::handle:vertical { background: #2a4a6a; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QFrame#header { background-color: #0d1424; border-bottom: 1px solid #1a2a45; }
QFrame#status { background-color: #0d1424; border-top: 1px solid #1a2a45; padding: 4px 14px; }
"""

class TabPlaceholder(QWidget):
    def __init__(self, title, description="", icon="⚡", parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #141e30;")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)

        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 48px; color: #2a4a6a;")

        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #4a8aba; font-size: 22px; font-weight: bold;")

        desc = QLabel(description)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: #3a5a7a; font-size: 12px;")

        hint = QLabel("(proximamente)")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #1a2a45; font-size: 10px; font-style: italic;")

        layout.addStretch()
        layout.addWidget(icon_label)
        layout.addWidget(label)
        layout.addWidget(desc)
        layout.addWidget(hint)
        layout.addStretch()


class HeaderBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("header")
        self._parent = parent
        self._drag_pos = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 0, 0)
        layout.setSpacing(0)

        title = QLabel("analytics system")
        title.setStyleSheet("color: #4fc3f7; font-size: 14px; font-weight: bold; letter-spacing: 3px;")

        subtitle = QLabel("for assets")
        subtitle.setStyleSheet("color: #3a5a7a; font-size: 11px;")
        subtitle.setContentsMargins(6, 0, 0, 0)

        version = QLabel("v0.5.5")
        version.setStyleSheet("color: #2a4a6a; font-size: 10px; padding-left: 8px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(version)
        layout.addStretch()

        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setStyleSheet("background-color: #1a2a45; max-width: 1px;")
        sep.setFixedHeight(24)
        layout.addWidget(sep)

        btn_frame = QFrame()
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(0)

        self.btn_min = QPushButton("─")
        self.btn_max = QPushButton("□")
        self.btn_close = QPushButton("✕")

        for btn in (self.btn_min, self.btn_max, self.btn_close):
            btn.setFixedSize(46, 32)
            btn.setCursor(Qt.CursorShape.ArrowCursor)
            btn_layout.addWidget(btn)

        self.btn_min.setStyleSheet("""
            QPushButton { background: transparent; color: #7a9aba; border: none; font-size: 14px; border-radius: 0; }
            QPushButton:hover { background-color: #1a2a45; color: #c8d6e5; }
        """)
        self.btn_max.setStyleSheet("""
            QPushButton { background: transparent; color: #7a9aba; border: none; font-size: 12px; border-radius: 0; }
            QPushButton:hover { background-color: #1a2a45; color: #c8d6e5; }
        """)
        self.btn_close.setStyleSheet("""
            QPushButton { background: transparent; color: #7a9aba; border: none; font-size: 13px; border-radius: 0; }
            QPushButton:hover { background-color: #c0392b; color: white; }
        """)

        self.btn_min.clicked.connect(self._on_minimize)
        self.btn_max.clicked.connect(self._on_maximize)
        self.btn_close.clicked.connect(self._on_close)

        layout.addWidget(btn_frame)

    def _on_minimize(self):
        if self._parent:
            self._parent.showMinimized()

    def _on_maximize(self):
        if self._parent:
            if self._parent.isMaximized():
                self._parent.showNormal()
                self.btn_max.setText("□")
            else:
                self._parent.showMaximized()
                self.btn_max.setText("❐")

    def _on_close(self):
        if self._parent:
            self._parent.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._parent:
            self._drag_pos = event.globalPosition().toPoint()
            self._parent._drag_offset = self._drag_pos - self._parent.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._parent and self._drag_pos:
            self._parent.move(event.globalPosition().toPoint() - self._parent._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None
            event.accept()


class StatusBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("status")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)

        self.label = QLabel("Listo")
        self.label.setStyleSheet("color: #3a5a7a; font-size: 11px;")
        layout.addWidget(self.label)
        layout.addStretch()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analytics System for Assets")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(STYLE)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self._drag_offset = QPoint()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(1, 0, 1, 1)
        layout.setSpacing(0)

        self.header = HeaderBar(self)
        layout.addWidget(self.header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.tab_descargar = TabDescargar()
        self.tab_importar = TabImportar()
        self.tab_limpiados = TabLimpiados()
        self.tab_analisis = TabAnalisis()

        self.tab_limpiados.set_analisis_tab(self.tab_analisis)

        self.tabs.addTab(self.tab_descargar, u"   Descargar   ")
        self.tabs.addTab(self.tab_importar, u"   Importar   ")
        self.tabs.addTab(self.tab_limpiados, u"   Limpiados   ")
        self.tabs.addTab(self.tab_analisis, u"   Analizador   ")

        # Refresh file explorer and table mode when import completes
        self.tab_importar.import_completed.connect(self.tab_limpiados.explorer.refresh)
        self.tab_importar.import_completed.connect(self.tab_limpiados._scan_assets)

        # Analysis completed → load results in Analisis tab and switch to it
        self.tab_limpiados.analysis_completed.connect(self.tab_analisis.load_results)
        self.tab_limpiados.analysis_completed.connect(
            lambda *a: self.tabs.setCurrentIndex(3)
        )

        # File selected in Limpiados -> preview horizon disabling by TF
        self.tab_limpiados.file_selected.connect(self.tab_analisis.preview_horizon_for)

        # Refresh explorer when switching to limpiados tab
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tabs, 1)
        layout.addWidget(StatusBar())

    def _on_tab_changed(self, index):
        if index == 2:
            self.tab_limpiados.explorer.refresh()
            self.tab_limpiados._scan_assets()

    def set_status(self, text):
        sb = self.findChild(StatusBar)
        if sb:
            sb.label.setText(text)
