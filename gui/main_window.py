import time
from PyQt6.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QLabel, QPushButton, QHBoxLayout, QFrame, QSizePolicy,
                             QApplication)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap
from gui.widgets.tab_descargar import TabDescargar
from gui.widgets.tab_importar import TabImportar
from gui.widgets.tab_analisis import TabAnalisis
from gui.widgets.tab_limpiados import TabLimpiados
from gui.widgets.tab_comparador import TabComparador
from gui.widgets.tab_backtest import TabBacktest
from gui.widgets.loading_overlay import LoadingOverlay

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

# Si la precarga supera este tiempo, la ventana queda interactiva y el resto de
# pestañas se construyen en su primera visita (con overlay breve).
PRECARGA_TOPE_MS = 3500


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

        version = QLabel("v0.6.8")
        version.setStyleSheet("color: #2a4a6a; font-size: 10px; padding-left: 8px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(version)
        layout.addStretch()

        self.btn_help = QPushButton("?")
        self.btn_help.setFixedSize(36, 32)
        self.btn_help.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_help.setToolTip("Cómo funciona")
        self.btn_help.setStyleSheet("""
            QPushButton { background: transparent; color: #7a9aba; border: none; font-size: 16px; border-radius: 0; }
            QPushButton:hover { background-color: #1a2a45; color: #4fc3f7; }
        """)
        layout.addWidget(self.btn_help)

        self.btn_settings = QPushButton("⚙")
        self.btn_settings.setFixedSize(36, 32)
        self.btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_settings.setToolTip("Ajustes")
        self.btn_settings.setStyleSheet("""
            QPushButton { background: transparent; color: #7a9aba; border: none; font-size: 16px; border-radius: 0; }
            QPushButton:hover { background-color: #1a2a45; color: #4fc3f7; }
        """)
        layout.addWidget(self.btn_settings)

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
        self.setWindowTitle("Analytics System for Assets v0.6.8")
        # 1400x900 como tamaño deseado, recortado al área visible de la
        # pantalla (en portátiles pequeños 1400x900 no cabe) y centrado.
        disp = QApplication.primaryScreen().availableGeometry()
        ancho = min(1400, int(disp.width() * 0.92))
        alto = min(900, int(disp.height() * 0.92))
        x = disp.x() + (disp.width() - ancho) // 2
        y = disp.y() + (disp.height() - alto) // 2
        self.setGeometry(x, y, ancho, alto)
        self.setStyleSheet(STYLE)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self._drag_offset = QPoint()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(1, 0, 1, 1)
        layout.setSpacing(0)

        self.header = HeaderBar(self)
        self.header.btn_settings.clicked.connect(self._on_settings)
        self.header.btn_help.clicked.connect(self._on_ayuda)
        layout.addWidget(self.header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(StatusBar())

        # Overlay de carga: tapa la ventana mientras algo se está preparando.
        self.overlay = LoadingOverlay(self)

        # Pestañas: la activa (Descargar) se construye ya; el resto se crean
        # bajo demanda (precarga acotada tras el arranque + primera visita),
        # para que la ventana no se congele varios segundos al abrirse.
        self._tab_titles = {
            0: u"   Descargar   ",
            1: u"   Importar   ",
            2: u"   Limpiador   ",
            3: u"   Analizador   ",
            4: u"   Comparador   ",
            5: u"   Backtester   ",
        }
        self._tab_labels = {
            1: "Importar",
            2: "Limpiador",
            3: "Analizador",
            4: "Comparador",
            5: "Backtester",
        }
        self._tab_classes = {
            1: TabImportar,
            2: TabLimpiados,
            3: TabAnalisis,
            4: TabComparador,
            5: TabBacktest,
        }
        self._real_tabs = {}
        self._placeholders = {}
        self._building = False
        self._tab_primera_pintada = set()
        self._preload_pendiente = []
        self._preload_inicio = None
        self._preload_tope = PRECARGA_TOPE_MS / 1000.0
        self._preload_done_cb = None

        self.tab_descargar = TabDescargar()
        self._real_tabs[0] = self.tab_descargar
        self.tabs.addTab(self.tab_descargar, self._tab_titles[0])
        for idx in range(1, 6):
            ph = TabPlaceholder(self._tab_labels[idx])
            self._placeholders[idx] = ph
            self.tabs.addTab(ph, self._tab_titles[idx])

        self.tabs.currentChanged.connect(self._on_tab_changed)

    # ── Construcción perezosa de pestañas ──────────────────────────────

    def _wire_tab(self, idx, widget):
        if idx == 1:
            widget.import_completed.connect(self._on_import_completed)
        elif idx == 2:
            widget.analysis_completed.connect(self._on_analysis_completed)
            widget.file_selected.connect(self._on_file_selected)
            widget.set_analisis_tab(self._get_tab_analisis)
        elif idx == 5:
            # overlay de carga de la ventana: lo usa el constructor del
            # backtest al seleccionar CSV, ejecutar backtest u optimizar
            widget._set_overlay(self.overlay)

    def _get_tab_analisis(self):
        return self._get_tab(3)

    def _get_tab(self, idx):
        real = self._real_tabs.get(idx)
        if real is not None:
            return real
        self._build_tab(idx)
        return self._real_tabs.get(idx)

    def _build_tab(self, idx, mantener_overlay=False):
        if self._building:
            return
        if idx in self._real_tabs or idx not in self._placeholders:
            return
        cls = self._tab_classes.get(idx)
        if cls is None:
            return
        ph = self._placeholders[idx]
        self._building = True
        inicio_overlay = False
        try:
            inicio_overlay = not self.overlay.isVisible()
            if inicio_overlay:
                self.overlay.begin(f"Preparando {self._tab_labels[idx]}…")
            elif mantener_overlay:
                self.overlay.set_text(f"Preparando {self._tab_labels[idx]}…")
            QApplication.processEvents()

            widget = cls()
            self._real_tabs[idx] = widget
            self._wire_tab(idx, widget)

            tab_index = self.tabs.indexOf(ph)
            era_actual = self.tabs.currentIndex() == tab_index
            if tab_index < 0:
                tab_index = min(idx, self.tabs.count())
                self.tabs.insertTab(tab_index, widget, self._tab_titles[idx])
            else:
                self.tabs.removeTab(tab_index)
                self.tabs.insertTab(min(tab_index, self.tabs.count()),
                                    widget, self._tab_titles[idx])
                if era_actual:
                    self.tabs.setCurrentIndex(tab_index)
            del self._placeholders[idx]
            ph.deleteLater()
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            self._building = False
            # Fuerza el layout/pintado del tab mientras el overlay aún cubre la
            # ventana: así la primera pintada pesada (p. ej. Backtester) no se
            # ve como un parpadeo/tirona al destaparla.
            QApplication.processEvents()
            try:
                if not mantener_overlay and inicio_overlay:
                    self.overlay.end()
            except Exception:
                pass

    # ── Precarga acotada tras el arranque ──────────────────────────────

    def precargar_tabs(self, tope_ms=PRECARGA_TOPE_MS, done_cb=None):
        """Construye las pestañas pendientes bajo el overlay. Si el tiempo
        supera tope_ms, deja el resto para su primera visita (pestaña a
        pestaña). done_cb se llama al terminar (en el hilo principal)."""
        self._preload_tope = max(0.0, float(tope_ms)) / 1000.0
        self._preload_done_cb = done_cb
        self._preload_inicio = time.monotonic()
        self._preload_pendiente = [i for i in range(1, 6) if i in self._placeholders]
        if not self._preload_pendiente:
            self._preload_terminado()
            return
        self.overlay.begin("Preparando interfaz…")
        QTimer.singleShot(0, self._preload_next)

    def _preload_next(self):
        try:
            # El bombeo de eventos dentro de un constructor puede disparar este
            # timer mientras _build_tab sigue en marcha: re-encolar y reintentar
            # en vez de morir (si no, la precarga quedaría a medias).
            if self._building:
                QTimer.singleShot(0, self._preload_next)
                return
            if not self._preload_pendiente:
                self._preload_terminado()
                return
            if time.monotonic() - self._preload_inicio > self._preload_tope:
                self._preload_terminado()
                return
            idx = self._preload_pendiente.pop(0)
            self._build_tab(idx, mantener_overlay=True)
            QTimer.singleShot(0, self._preload_next)
        except Exception:
            import traceback
            traceback.print_exc()
            self._preload_terminado()

    def _preload_terminado(self):
        self._preload_pendiente = []
        try:
            self.overlay.end()
        except Exception:
            pass
        if self._preload_done_cb:
            cb = self._preload_done_cb
            self._preload_done_cb = None
            try:
                cb()
            except Exception:
                pass

    # ── Cambio de pestaña ──────────────────────────────────────────────

    def _on_tab_changed(self, index):
        if self._building:
            return
        self._get_tab(index)
        QTimer.singleShot(0, lambda: self._refresh_tab(index))

    def _refresh_tab(self, index):
        if self._building:
            return
        try:
            if index == 2:
                tab = self._get_tab(2)
                self.overlay.begin("Actualizando Limpiador…")
                tab.explorer.refresh()
                tab._scan_assets_async(done_cb=self.overlay.end)
            elif index == 4:
                tab = self._get_tab(4)
                self.overlay.begin("Actualizando Comparador…")
                tab.refresh_available_async(done_cb=self.overlay.end)
            elif index == 5:
                tab = self._get_tab(5)
                # La primera vez que se muestra, su pintado completo tarda:
                # se cubre con el overlay y se destapa un instante después.
                if index not in self._tab_primera_pintada:
                    self._tab_primera_pintada.add(index)
                    self.overlay.begin("Preparando Backtester…")
                    QTimer.singleShot(120, self.overlay.end)
                tab.refresh_available()
        except Exception:
            self.overlay.end()
            import traceback
            traceback.print_exc()

    # ── Señales cruzadas entre pestañas (resueltas bajo demanda) ──────

    def _on_import_completed(self):
        try:
            tab = self._get_tab(2)
            tab.explorer.refresh()
            tab._scan_assets_async()
        except Exception:
            import traceback
            traceback.print_exc()

    def _on_analysis_completed(self, *args):
        try:
            self._get_tab(3).load_results(*args)
        except Exception:
            import traceback
            traceback.print_exc()
        self.tabs.setCurrentIndex(3)

    def _on_file_selected(self, *args):
        try:
            self._get_tab(3).preview_horizon_for(*args)
        except Exception:
            import traceback
            traceback.print_exc()

    # ── Menús y ajustes ────────────────────────────────────────────────

    def set_status(self, text):
        sb = self.findChild(StatusBar)
        if sb:
            sb.label.setText(text)

    def _on_ayuda(self):
        from gui.dialogs.tutorial_dialog import TutorialDialog
        TutorialDialog(self).exec()

    def _on_settings(self):
        from gui.dialogs.settings_dialog import SettingsDialog
        from core.config import get_base_data
        before = get_base_data()
        dlg = SettingsDialog(self)
        dlg.exec()
        after = get_base_data()
        if after != before:
            for idx in (0, 1, 2):
                tab = self._real_tabs.get(idx)
                if tab is not None and hasattr(tab, 'update_base_data'):
                    tab.update_base_data(after)
            self.set_status(f"Ruta de datos: {after}")

    def closeEvent(self, event):
        # apaga la QuestDB local solo si la arrancamos nosotros en esta
        # sesión — nunca toca una que el usuario ya tuviera corriendo
        from core.questdb_manager import detener_bundled
        try:
            detener_bundled()
        except Exception:
            pass
        # Antes de destruir la ventana hay que apagar los hilos de fondo de
        # TODAS las pestañas (cargas del Comparador, detección de días del
        # Backtester, escaneos...): Qt aborta el proceso si destruye un
        # QThread aún vivo, y el Comparador además corrompe el montón si se
        # destruye con sus activos cargados.
        for tab in list(self._real_tabs.values()):
            apagar = getattr(tab, '_apagar_hilos', None)
            if callable(apagar):
                try:
                    apagar()
                except Exception:
                    pass
        super().closeEvent(event)
