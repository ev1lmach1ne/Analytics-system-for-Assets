import os
import re
import json

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QComboBox, QCompleter,
                             QFrame, QTableWidget, QTableWidgetItem,
                             QHeaderView, QLineEdit, QSizePolicy, QButtonGroup,
                             QSplitter, QMessageBox, QInputDialog, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QColor

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from gui.widgets.asset_selector import scan_limpiados, RE_LIMPIADO, RE_LIMPIO
from gui.widgets.file_explorer import FileExplorer

STYLE_COMPARADOR = """
QWidget { background-color: #141e30; }
QPushButton {
    background-color: #2a4a6a; color: #4fc3f7; border: none;
    padding: 6px 12px; border-radius: 4px; font-size: 12px; font-weight: bold;
}
QPushButton:hover { background-color: #3a5a8a; }
QPushButton:pressed { padding-top: 8px; padding-bottom: 4px; }
QPushButton:disabled { background-color: #1a2a45; color: #3a5a7a; }
QPushButton#tf {
    background-color: #1a2a45; color: #5a7a9a; padding: 6px 14px;
    font-size: 11px; min-width: 40px;
}
QPushButton#tf:checked {
    background-color: #2a4a6a; color: #4fc3f7; font-weight: bold;
}
QComboBox {
    background-color: #1a2a45; color: #c8d6e5; border: none;
    padding: 6px 10px; border-radius: 4px; font-size: 12px; min-width: 200px;
    combobox-popup: 0;
}
QComboBox::drop-down { border: none; background: transparent; width: 22px; }
QComboBox::down-arrow { border: none; }
QComboBox QAbstractItemView {
    background-color: #1a2a45; color: #c8d6e5; selection-background-color: #2a4a6a;
    border: 1px solid #253a60; outline: none; margin: 0px;
}
QLineEdit {
    background-color: #1a2a45; color: #c8d6e5; border: none;
    padding: 6px 10px; border-radius: 4px; font-size: 11px;
}
QTableWidget {
    background-color: #0d1424; color: #aabbcc; border: 1px solid #253a60;
    gridline-color: #1a2a45; font-size: 11px;
}
QTableWidget::item { padding: 4px 8px; }
QHeaderView::section {
    background-color: #1a2a45; color: #4fc3f7; border: none;
    padding: 6px 8px; font-weight: bold; font-size: 11px;
}
QFrame#panel { background-color: #0d1424; border: 1px solid #253a60; border-radius: 4px; }
QLabel#panelTitle { color: #4fc3f7; font-size: 12px; font-weight: bold; background: transparent; }
QLabel#tfLabel { color: #5a7a9a; font-size: 11px; background: transparent; }
QFrame#chip { background-color: #1a2a45; border: 1px solid #253a60; border-radius: 10px; }
QLabel#chipName { color: #c8d6e5; font-size: 11px; background: transparent; }
QPushButton#chipDel {
    background: transparent; color: #e74c3c; border: none;
    padding: 0px 4px; font-size: 12px; font-weight: bold; min-width: 0;
}
QPushButton#chipDel:hover { color: #ff6b5b; }
QCheckBox { color: #5a7a9a; font-size: 11px; background: transparent; spacing: 5px; }
QCheckBox::indicator {
    width: 13px; height: 13px; border: 1px solid #253a60;
    border-radius: 3px; background-color: #1a2a45;
}
QCheckBox::indicator:checked { background-color: #2a4a6a; border-color: #4fc3f7; }
"""

# El popup del QCompleter es una QListView independiente que NO hereda el
# estilo del QComboBox (por eso salia con el look blanco por defecto de
# Windows): hay que estilarla explicitamente, igual que las listas de las
# demas pestañas (tab_descargar).
STYLE_COMPLETER_POPUP = """
QListView {
    background-color: #1a2a45; color: #c8d6e5; border: 1px solid #253a60;
    outline: none; font-size: 12px;
}
QListView::item { padding: 6px 10px; border-bottom: 1px solid #182030; }
QListView::item:selected { background-color: #2a4a6a; color: #4fc3f7; }
QListView::item:hover { background-color: #223755; }
QScrollBar:vertical { background: #1a2a45; width: 8px; }
QScrollBar::handle:vertical { background: #2a4a6a; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""

_ANSI_RE = re.compile(r'\033\[[\d;]+m')

# Paleta ciclica de colores distinguibles sobre fondo oscuro (uno por activo)
PALETTE = ['#4fc3f7', '#ff9900', '#2ecc71', '#e74c3c', '#9b59b6',
           '#f1c40f', '#1abc9c', '#e67e22', '#3498db', '#fd79a8']

# Estilo oscuro comun para todos los ejes matplotlib del dashboard
FIG_BG = '#0d1424'
AX_FG = '#c8d6e5'
GRID_C = '#253a60'

MODOS_NORMALIZADO = [
    'Intersección Temporal',
    'Día de Vida (Age Alignment)',
    'Histórico Completo (dispersión)',
]

# Temporalidad comun del dashboard: todas las series se convierten a este TF
# para que activos de TF distinto (1min vs 1h vs 1d) sean comparables entre si.
TF_OPCIONES = [('1h', '1h'), ('4h', '4h'), ('1d', '1D'), ('1w', '1W')]
TF_DEFECTO = '1d'

# Maximo de puntos dibujados por linea: por encima, el canvas Qt va a tirones
# al hacer zoom/redibujar (las series 1h de varios años superan 50k puntos).
MAX_PUNTOS_PLOT = 4000


def _no_crash(fn):
    """En PyQt6 una excepcion no capturada dentro de un slot ABORTA el proceso
    entero (qFatal), no solo imprime el traceback: cualquier metodo de refresco
    conectado a señales debe ser a prueba de excepciones."""
    def wrapper(self, *a, **kw):
        try:
            return fn(self, *a, **kw)
        except Exception as e:
            print(f"[Comparador] Error en {fn.__name__}: {e}")
    wrapper.__name__ = fn.__name__
    return wrapper


def parse_metric_number(text):
    """Extrae el valor numerico de un string de metrica del metrics.json
    ('14.94%', '\\x1b[91m0.2994\\x1b[0m', 'N/A (TICKS)') -> float o None."""
    if text is None:
        return None
    s = _ANSI_RE.sub('', str(text)).strip()
    m = re.search(r'-?\d+(?:[.,]\d+)?', s.replace(',', ''))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def extraer_metricas_tabla(metrics_json):
    """Devuelve dict {'cagr':%, 'vol':%, 'sharpe':x, 'calmar':x, 'max_dd':%} (None si falta)
    a partir del dict del .analysis.metrics.json."""
    out = {'cagr': None, 'vol': None, 'sharpe': None, 'calmar': None, 'max_dd': None}
    for cat, items in metrics_json.items():
        if cat == '_paginas' or not isinstance(items, dict):
            continue
        for k, v in items.items():
            if k == 'Retorno anualizado (CAGR)':
                out['cagr'] = parse_metric_number(v)
            elif k == 'Volatilidad anualizada':
                out['vol'] = parse_metric_number(v)
            elif k.startswith('Ratio Sharpe'):
                out['sharpe'] = parse_metric_number(v)
            elif k == 'Calmar Ratio':
                out['calmar'] = parse_metric_number(v)
            elif k == 'Max Drawdown Histórico':
                out['max_dd'] = parse_metric_number(v)
    return out


def calcular_stats_tf(close_tf):
    """CAGR y volatilidad anualizada (en %) desde una serie de cierres en el TF
    comun del dashboard. La anualizacion usa los periodos de MERCADO reales por
    año (velas presentes / años transcurridos), como en el analizador: excluye
    automaticamente fines de semana y horario cerrado porque esas velas no
    existen en los datos limpiados."""
    close_tf = close_tf.dropna()
    if len(close_tf) < 3:
        return None, None, None
    ret = np.log(close_tf / close_tf.shift(1)).dropna()
    dias_calendario = (close_tf.index[-1] - close_tf.index[0]).days
    if dias_calendario <= 0:
        return None, None, None
    anios = dias_calendario / 365.25
    periodos_anio = len(close_tf) / anios
    cagr = (close_tf.iloc[-1] / close_tf.iloc[0]) ** (1.0 / anios) - 1.0
    vol_anual = ret.std() * np.sqrt(periodos_anio)
    return cagr * 100.0, vol_anual * 100.0, ret


def _downsample(serie):
    """Reduce una serie a <=MAX_PUNTOS_PLOT puntos para dibujarla fluida."""
    if len(serie) <= MAX_PUNTOS_PLOT:
        return serie
    paso = int(np.ceil(len(serie) / MAX_PUNTOS_PLOT))
    return serie.iloc[::paso]


class _AssetLoadThread(QThread):
    """Carga el CSV limpiado (solo timestamp+close) y lo resamplea a una BASE
    HORARIA en segundo plano. La base 1h reduce ~60x una serie de 1min: los
    cambios de TF del dashboard resamplean esa base al vuelo sin releer el CSV.
    Si el TF nativo del activo es mas grueso que 1h (ej. 1d), la base conserva
    sus puntos originales tal cual (resample last+dropna no inventa velas).
    """
    loaded = pyqtSignal(str, object)   # path, payload dict

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        try:
            try:
                # pyarrow es mucho mas rapido y retiene menos el GIL que el
                # motor C (menos tirones de UI durante la carga de CSVs de 1min)
                df = pd.read_csv(self._path, usecols=['timestamp', 'close'],
                                 engine='pyarrow')
            except (ImportError, ValueError):
                df = pd.read_csv(self._path, usecols=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'close']).set_index('timestamp').sort_index()
            base_1h = df['close'].resample('1h').last().dropna()
            self.loaded.emit(self._path, {'base_1h': base_1h})
        except Exception as e:
            self.loaded.emit(self._path, {'error': str(e)})


class AssetChip(QFrame):
    removed = pyqtSignal(str)  # path

    def __init__(self, path, label, color, parent=None):
        super().__init__(parent)
        self.setObjectName("chip")
        self._path = path
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 3, 4, 3)
        lay.setSpacing(6)

        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background-color: {color}; border-radius: 5px;")
        lay.addWidget(dot)

        name = QLabel(label)
        name.setObjectName("chipName")
        lay.addWidget(name)

        btn = QPushButton("✕")
        btn.setObjectName("chipDel")
        btn.setFixedWidth(20)
        btn.clicked.connect(lambda: self.removed.emit(self._path))
        lay.addWidget(btn)


class _Panel(QFrame):
    """Marco visual comun de cada cuadrante del dashboard."""

    def __init__(self, titulo, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.vlay = QVBoxLayout(self)
        self.vlay.setContentsMargins(10, 8, 10, 8)
        self.vlay.setSpacing(6)
        self.header = QHBoxLayout()
        self.title = QLabel(titulo)
        self.title.setObjectName("panelTitle")
        self.header.addWidget(self.title)
        self.header.addStretch()
        self.vlay.addLayout(self.header)


def _make_canvas():
    fig = Figure(figsize=(4, 3), facecolor=FIG_BG)
    canvas = FigureCanvasQTAgg(fig)
    canvas.setStyleSheet("background-color: transparent;")
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    canvas.setMinimumHeight(180)
    return fig, canvas


def _style_ax(ax):
    ax.set_facecolor(FIG_BG)
    ax.tick_params(colors=AX_FG, labelsize=7)
    ax.xaxis.label.set_color(AX_FG)
    ax.yaxis.label.set_color(AX_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(True, alpha=0.25, color=GRID_C, linewidth=0.5)


def _ax_placeholder(ax, texto):
    ax.text(0.5, 0.5, texto, ha='center', va='center',
            color='#5a7a9a', fontsize=9, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


class TabComparador(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE_COMPARADOR)

        # Estado: orden de adicion define el color de cada activo
        self._assets = {}      # path -> {'label','color','metrics','base' (Series 1h o None),'tf_cache','chip'}
        self._orden = []       # paths en orden de adicion
        self._disponibles = [] # [(path, label, ticker, tf)] analizados no añadidos
        self._threads = {}     # path -> _AssetLoadThread vivo (hasta finished)
        self._tf = TF_DEFECTO
        self._last_splitter_sizes = [280, 800]
        self._listas = []  # listas preguardadas: [{'nombre','tf','activos'}]

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(8)

        titulo = QLabel("Analizador Multi-Activo Comparativo")
        titulo.setStyleSheet("color: #4fc3f7; font-size: 16px; font-weight: bold; background: transparent;")
        root.addWidget(titulo)

        # ── Barra de seleccion: combo + añadir + TF comun + Rf + chips ──
        barra = QHBoxLayout()
        barra.setSpacing(8)

        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setPlaceholderText("Buscar activo analizado...")
        barra.addWidget(self.combo)

        self.btn_add = QPushButton("+ Añadir")
        self.btn_add.clicked.connect(self._on_add_clicked)
        barra.addWidget(self.btn_add)

        lbl_tf = QLabel("TF común:")
        lbl_tf.setObjectName("tfLabel")
        barra.addWidget(lbl_tf)

        self._tf_group = QButtonGroup(self)
        self._tf_group.setExclusive(True)
        self._tf_buttons = {}
        for i, (tf_label, _regla) in enumerate(TF_OPCIONES):
            btn = QPushButton(tf_label)
            btn.setObjectName("tf")
            btn.setCheckable(True)
            if tf_label == TF_DEFECTO:
                btn.setChecked(True)
            self._tf_group.addButton(btn, i)
            self._tf_buttons[tf_label] = btn
            barra.addWidget(btn)
        self._tf_group.idClicked.connect(self._on_tf_changed)

        lbl_rf = QLabel("Rf %:")
        lbl_rf.setObjectName("tfLabel")
        barra.addWidget(lbl_rf)
        self.rf_input = QLineEdit("4.33")
        self.rf_input.setFixedWidth(82)
        # Enter en el campo tambien aplica (returnPressed); editingFinished
        # cubre ademas el clic fuera del campo
        self.rf_input.editingFinished.connect(self._refresh_scatter)
        # Tick DENTRO del campo (accion trailing): aplicar sin ocupar espacio
        # extra en la barra
        from PyQt6.QtGui import QPixmap, QPainter, QPen, QIcon
        from PyQt6.QtCore import QPoint
        _pix = QPixmap(14, 14)
        _pix.fill(Qt.GlobalColor.transparent)
        _painter = QPainter(_pix)
        _painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _painter.setPen(QPen(QColor('#4fc3f7'), 2))
        _painter.drawPolyline([QPoint(2, 8), QPoint(6, 11), QPoint(12, 3)])
        _painter.end()
        self._rf_action = self.rf_input.addAction(
            QIcon(_pix), QLineEdit.ActionPosition.TrailingPosition)
        self._rf_action.setToolTip("Aplicar Rf")
        self._rf_action.triggered.connect(lambda: self._refresh_scatter())
        barra.addWidget(self.rf_input)

        barra.addStretch()

        lbl_listas = QLabel("Lista:")
        lbl_listas.setObjectName("tfLabel")
        barra.addWidget(lbl_listas)

        self.lista_combo = QComboBox()
        self.lista_combo.addItem("(lista actual)")
        self.lista_combo.setMinimumWidth(160)
        self.lista_combo.currentIndexChanged.connect(self._on_lista_combo_changed)
        barra.addWidget(self.lista_combo)

        self.btn_guardar_lista = QPushButton("💾 Guardar")
        self.btn_guardar_lista.clicked.connect(self._guardar_lista_actual)
        barra.addWidget(self.btn_guardar_lista)

        self.btn_eliminar_lista = QPushButton("🗑 Eliminar")
        self.btn_eliminar_lista.setObjectName("danger")
        self.btn_eliminar_lista.clicked.connect(self._eliminar_lista_actual)
        barra.addWidget(self.btn_eliminar_lista)

        self.btn_toggle_explorer = QPushButton("📁 Explorador")
        self.btn_toggle_explorer.setCheckable(True)
        self.btn_toggle_explorer.setChecked(True)
        self.btn_toggle_explorer.setObjectName("tf")
        self.btn_toggle_explorer.clicked.connect(self._on_toggle_explorer)
        barra.addWidget(self.btn_toggle_explorer)

        root.addLayout(barra)

        self.chips_lay = QHBoxLayout()
        self.chips_lay.setSpacing(6)
        self.chips_lay.addStretch()
        root.addLayout(self.chips_lay)

        # ── Grid 2x2 de paneles ──
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # 📊 Tabla comparativa
        self.panel_tabla = _Panel("📊 TABLA COMPARATIVA DE MÉTRICAS")
        self.tabla = QTableWidget(0, 6)
        self.tabla.setHorizontalHeaderLabels(['Activo', 'CAGR', 'Vol Anual', 'Max DD', 'Sharpe', 'Calmar'])
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabla.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.panel_tabla.vlay.addWidget(self.tabla, 1)
        grid.addWidget(self.panel_tabla, 0, 0)

        # 📈 Retorno normalizado (con desplegable de modo)
        self.panel_norm = _Panel("📈 GRÁFICO DE RETORNO NORMALIZADO")
        self.modo_combo = QComboBox()
        self.modo_combo.setMinimumWidth(220)
        for m in MODOS_NORMALIZADO:
            self.modo_combo.addItem(m)
        self.modo_combo.currentIndexChanged.connect(lambda _i: self._refresh_normalizado())
        self.panel_norm.header.addWidget(self.modo_combo)
        self.fig_norm, self.canvas_norm = _make_canvas()
        self.panel_norm.vlay.addWidget(self.canvas_norm, 1)
        grid.addWidget(self.panel_norm, 0, 1)

        # 🧬 Matriz de correlacion
        self.panel_corr = _Panel("🧬 MATRIZ DE CORRELACIÓN (Heatmap)")
        self.fig_corr, self.canvas_corr = _make_canvas()
        self.panel_corr.vlay.addWidget(self.canvas_corr, 1)
        grid.addWidget(self.panel_corr, 1, 0)

        # 🎯 Dispersion riesgo-retorno
        self.panel_scatter = _Panel("🎯 DISPERSIÓN RIESGO-RETORNO")
        lbl_bench = QLabel("Benchmark:")
        lbl_bench.setObjectName("tfLabel")
        self.panel_scatter.header.addWidget(lbl_bench)
        self.benchmark_combo = QComboBox()
        self.benchmark_combo.setMinimumWidth(140)
        self.benchmark_combo.addItem("(medianas)", None)
        # lambda sin usar el arg de la señal: evita el problema de aridad con
        # los slots decorados por @_no_crash (ver nota en _on_add_clicked)
        self.benchmark_combo.currentIndexChanged.connect(lambda _i: self._refresh_scatter())
        self.panel_scatter.header.addWidget(self.benchmark_combo)
        self.frontera_check = QCheckBox("Frontera eficiente")
        self.frontera_check.toggled.connect(lambda _v: self._refresh_scatter())
        self.panel_scatter.header.addWidget(self.frontera_check)
        self.fig_scatter, self.canvas_scatter = _make_canvas()
        self.panel_scatter.vlay.addWidget(self.canvas_scatter, 1)
        grid.addWidget(self.panel_scatter, 1, 1)

        # ── Splitter: Explorer (izquierda) + Dashboard (derecha) ──
        from core.config import LIMPIADOS_DIR
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # Panel izquierdo: explorador de activos limpiados
        self._left_panel = QWidget()
        left_lay = QVBoxLayout(self._left_panel)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        search_input = QLineEdit()
        search_input.setPlaceholderText("Buscar activo...")
        left_lay.addWidget(search_input)

        self.explorer = FileExplorer(LIMPIADOS_DIR, mode='csv')
        self.explorer.file_chosen.connect(self._on_explorer_file_chosen)
        search_input.textChanged.connect(self.explorer.set_search_text)
        left_lay.addWidget(self.explorer, 1)

        self._splitter.addWidget(self._left_panel)

        # Panel derecho: dashboard 2x2
        right_panel = QWidget()
        right_lay = QVBoxLayout(right_panel)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)
        right_lay.addLayout(grid, 1)

        self._splitter.addWidget(right_panel)
        self._splitter.setSizes([280, 800])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)

        root.addWidget(self._splitter, 1)

        self.refresh_available()
        self._restore_config()
        self._refresh_all()

    # ══════════════ toggle explorador ══════════════

    @_no_crash
    def _on_toggle_explorer(self, checked):
        """Muestra/oculta el panel del explorador para ampliar los gráficos."""
        if checked:
            self._splitter.setSizes(self._last_splitter_sizes or [280, 800])
            self._left_panel.setVisible(True)
            self.btn_toggle_explorer.setText("📁 Explorador")
        else:
            self._last_splitter_sizes = self._splitter.sizes()
            self._left_panel.setVisible(False)
        # Forzar redibujado de matplotlib tras el cambio de tamaño del splitter
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(50, self._refresh_charts)

    def _config_path(self):
        from core.config import LIMPIADOS_DIR
        return os.path.join(LIMPIADOS_DIR, 'comparador_config.json')

    @_no_crash
    def refresh_available(self):
        """Rellena el combo con los activos analizados que aun no estan añadidos."""
        self._disponibles = []
        for path, label, ticker, tf in scan_limpiados():
            if path in self._assets:
                continue
            if os.path.exists(path + '.analysis.metrics.json'):
                self._disponibles.append((path, label, ticker, tf))
        if hasattr(self, 'explorer'):
            self.explorer.refresh()
        self.combo.blockSignals(True)
        self.combo.clear()
        for path, label, ticker, tf in self._disponibles:
            self.combo.addItem(label, path)
        self.combo.blockSignals(False)
        completer = QCompleter([d[1] for d in self._disponibles], self.combo)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.popup().setStyleSheet(STYLE_COMPLETER_POPUP)
        self.combo.setCompleter(completer)
        # El desplegable propio del combo tambien es una vista aparte
        if self.combo.view() is not None:
            self.combo.view().setStyleSheet(STYLE_COMPLETER_POPUP)
        if not self._disponibles:
            self.combo.setEditText("")
            self.combo.setPlaceholderText("No hay activos analizados disponibles")

    @_no_crash
    def _on_add_clicked(self, _checked=False):
        # _checked: la señal clicked(bool) SIEMPRE llega hasta aqui porque el
        # wrapper de @_no_crash usa *args y PyQt no puede inferir la aridad
        # real — sin este parametro el TypeError se tragaba en silencio y el
        # boton parecia muerto.
        idx = self.combo.currentIndex()
        if idx < 0 or idx >= len(self._disponibles):
            # El usuario pudo teclear el label a mano: buscar por texto
            texto = self.combo.currentText().strip().lower()
            match = [d for d in self._disponibles if d[1] == texto]
            if not match:
                return
            path, label = match[0][0], match[0][1]
        else:
            path, label = self._disponibles[idx][0], self._disponibles[idx][1]
        self.add_asset(path, label)

    @_no_crash
    def _on_explorer_file_chosen(self, path):
        """Manejador del FileExplorer: valida que el CSV tenga análisis
       	previo (.analysis.metrics.json) y lo añade al comparador."""
        if not os.path.exists(path):
            return
        if not os.path.exists(path + '.analysis.metrics.json'):
            QMessageBox.warning(
                self, "Activo sin analizar",
                "Este archivo no tiene resultados de análisis guardados\n"
                "(falta el .analysis.metrics.json).\n\n"
                "Analízalo primero desde la pestaña Limpiados."
            )
            return
        fname = os.path.basename(path)
        m = RE_LIMPIADO.match(fname) or RE_LIMPIO.match(fname)
        if m:
            label = f"{m.group(1).lower()} {m.group(2).lower()}"
        else:
            label = os.path.splitext(fname)[0]
        self.add_asset(path, label)

    @_no_crash
    def add_asset(self, path, label):
        if path in self._assets or not os.path.exists(path):
            return
        try:
            with open(path + '.analysis.metrics.json', encoding='utf-8') as f:
                metrics = extraer_metricas_tabla(json.load(f))
        except Exception:
            metrics = {'cagr': None, 'vol': None, 'sharpe': None, 'calmar': None}

        color = PALETTE[len(self._orden) % len(PALETTE)]
        chip = AssetChip(path, label, color)
        chip.removed.connect(self.remove_asset)
        # insertar antes del stretch final
        self.chips_lay.insertWidget(self.chips_lay.count() - 1, chip)

        self._assets[path] = {'label': label, 'color': color, 'metrics': metrics,
                              'base': None, 'error': None, 'tf_cache': {}, 'chip': chip}
        self._orden.append(path)

        # Carga de la serie en segundo plano. IMPORTANTE: el hilo se parenta a
        # este widget (ownership C++) y su entrada en _threads solo se retira
        # en _on_thread_finished — si la ultima referencia Python de un QThread
        # sin padre muere mientras el hilo aun corre, PyQt destruye el objeto
        # C++ y el proceso aborta ("QThread: Destroyed while thread is still
        # running"), que es un crash de app entera, no una excepcion.
        th = _AssetLoadThread(path, parent=self)
        th.loaded.connect(self._on_series_loaded)
        th.finished.connect(lambda p=path, t=th: self._on_thread_finished(p, t))
        self._threads[path] = th
        th.start()

        self.refresh_available()
        self._refresh_benchmark_combo()
        self._save_config()
        self._refresh_all()

    @_no_crash
    def remove_asset(self, path):
        info = self._assets.pop(path, None)
        if info is None:
            return
        self._orden.remove(path)
        info['chip'].setParent(None)
        info['chip'].deleteLater()
        # NO tocar self._threads aqui: si hay una carga en vuelo se deja
        # terminar (su resultado se descarta en _on_series_loaded porque el
        # path ya no esta en _assets). Destruir un QThread corriendo aborta
        # el proceso entero.
        for i, p in enumerate(self._orden):
            self._assets[p]['color'] = PALETTE[i % len(PALETTE)]
        self._rebuild_chips()
        self.refresh_available()
        self._refresh_benchmark_combo()
        self._save_config()
        self._refresh_all()

    def _refresh_benchmark_combo(self):
        """Reconstruye los items del combo de benchmark (uno por activo
        añadido) preservando la seleccion actual por path si sigue viva."""
        actual = self.benchmark_combo.currentData()
        self.benchmark_combo.blockSignals(True)
        self.benchmark_combo.clear()
        self.benchmark_combo.addItem("(medianas)", None)
        for p in self._orden:
            self.benchmark_combo.addItem(self._assets[p]['label'], p)
        if actual is not None:
            idx = self.benchmark_combo.findData(actual)
            self.benchmark_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.benchmark_combo.blockSignals(False)

    def _rebuild_chips(self):
        while self.chips_lay.count() > 1:
            item = self.chips_lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        for p in self._orden:
            info = self._assets[p]
            chip = AssetChip(p, info['label'], info['color'])
            chip.removed.connect(self.remove_asset)
            info['chip'] = chip
            self.chips_lay.insertWidget(self.chips_lay.count() - 1, chip)

    @_no_crash
    def _on_series_loaded(self, path, payload):
        if path not in self._assets:
            return  # el activo se quito mientras cargaba
        if payload and 'error' not in payload and payload.get('base_1h') is not None:
            self._assets[path]['base'] = payload['base_1h']
            self._assets[path]['tf_cache'] = {}
        else:
            self._assets[path]['error'] = (payload or {}).get('error', 'sin datos')
        self._refresh_charts()

    def _on_thread_finished(self, path, th):
        # El hilo ya termino de verdad: ahora si es seguro soltar la referencia
        # y programar la destruccion del objeto Qt.
        if self._threads.get(path) is th:
            self._threads.pop(path, None)
        th.deleteLater()

    # ══════════════ listas preguardadas ══════════════

    def _cargar_lista(self, nombre_lista):
        lista = next((l for l in self._listas if l['nombre'] == nombre_lista), None)
        if not lista:
            return
        for path in list(self._orden):
            self.remove_asset(path)
        tf = lista.get('tf')
        if tf in self._tf_buttons:
            self._tf = tf
            self._tf_buttons[tf].setChecked(True)
        for a in lista.get('activos', []):
            if os.path.exists(a.get('path', '')):
                self.add_asset(a['path'], a.get('label', os.path.basename(a['path'])))

    def _guardar_lista_actual(self):
        if not self._orden:
            return
        name, ok = QInputDialog.getText(self, "Guardar lista", "Nombre de la lista:")
        if not ok or not name.strip():
            return
        name = name.strip()
        self._listas = [l for l in self._listas if l['nombre'] != name]
        self._listas.append({
            'nombre': name,
            'tf': self._tf,
            'activos': [{'path': p, 'label': self._assets[p]['label']} for p in self._orden]
        })
        self._refresh_listas_combo()
        self.lista_combo.setCurrentText(name)
        self._save_config()

    def _eliminar_lista_actual(self):
        idx = self.lista_combo.currentIndex()
        if idx <= 0:
            return
        name = self.lista_combo.currentText()
        reply = QMessageBox.question(self, "Eliminar lista",
            f"¿Eliminar la lista '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._listas = [l for l in self._listas if l['nombre'] != name]
        self._refresh_listas_combo()
        self._save_config()

    def _on_lista_combo_changed(self, index):
        if index <= 0:
            return
        name = self.lista_combo.currentText()
        self._cargar_lista(name)

    def _refresh_listas_combo(self):
        self.lista_combo.blockSignals(True)
        current = self.lista_combo.currentText()
        self.lista_combo.clear()
        self.lista_combo.addItem("(lista actual)")
        for l in self._listas:
            self.lista_combo.addItem(l['nombre'])
        idx = self.lista_combo.findText(current)
        if idx >= 0:
            self.lista_combo.setCurrentIndex(idx)
        self.lista_combo.blockSignals(False)

    def _save_config(self):
        try:
            with open(self._config_path(), 'w', encoding='utf-8') as f:
                json.dump({
                    'activos': [{'path': p, 'label': self._assets[p]['label']} for p in self._orden],
                    'tf': self._tf,
                    'listas': self._listas,
                    'benchmark': self.benchmark_combo.currentData(),
                    'frontera': self.frontera_check.isChecked(),
                }, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _restore_config(self):
        try:
            with open(self._config_path(), encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            return
        self._listas = cfg.get('listas', [])
        if self._listas:
            self._refresh_listas_combo()
        tf = cfg.get('tf')
        if tf in self._tf_buttons:
            self._tf = tf
            self._tf_buttons[tf].setChecked(True)
        for a in cfg.get('activos', []):
            if os.path.exists(a.get('path', '')):
                self.add_asset(a['path'], a.get('label', os.path.basename(a['path'])))
        bench = cfg.get('benchmark')
        if bench and bench in self._assets:
            idx = self.benchmark_combo.findData(bench)
            if idx >= 0:
                self.benchmark_combo.setCurrentIndex(idx)
        if cfg.get('frontera'):
            self.frontera_check.setChecked(True)

    # ══════════════ TF comun ══════════════

    @_no_crash
    def _on_tf_changed(self, btn_id):
        tf = TF_OPCIONES[btn_id][0]
        if tf == self._tf:
            return
        self._tf = tf
        self._save_config()
        self._refresh_charts()

    def _datos_tf(self, path):
        """Serie + stats del activo en el TF comun actual, cacheado por TF.
        Devuelve dict {'close','ret','cagr','vol'} o None si aun no hay base."""
        info = self._assets[path]
        if info['base'] is None:
            return None
        cache = info['tf_cache'].get(self._tf)
        if cache is not None:
            return cache
        regla = dict(TF_OPCIONES)[self._tf]
        close_tf = info['base'].resample(regla).last().dropna()
        cagr, vol, ret = calcular_stats_tf(close_tf)
        if cagr is None:
            cache = {'close': close_tf, 'ret': None, 'cagr': None, 'vol': None}
        else:
            cache = {'close': close_tf, 'ret': ret, 'cagr': cagr, 'vol': vol}
        info['tf_cache'][self._tf] = cache
        return cache

    def _series_listas(self):
        """[(path, info, datos_tf)] de activos con serie valida en el TF actual."""
        out = []
        for p in self._orden:
            if self._assets[p]['error'] is not None:
                continue
            datos = self._datos_tf(p)
            if datos is not None and datos['cagr'] is not None:
                out.append((p, self._assets[p], datos))
        return out

    # ══════════════ refresco de paneles ══════════════

    def _refresh_all(self):
        self._refresh_tabla()
        self._refresh_charts()

    def _refresh_charts(self):
        self._refresh_normalizado()
        self._refresh_corr()
        self._refresh_scatter()

    @_no_crash
    def _refresh_tabla(self):
        self.tabla.setRowCount(len(self._orden))
        for row, path in enumerate(self._orden):
            info = self._assets[path]
            m = info['metrics']

            item_n = QTableWidgetItem(info['label'])
            item_n.setForeground(QColor(info['color']))
            self.tabla.setItem(row, 0, item_n)

            def _fmt(v, suf=''):
                return f"{v:.2f}{suf}" if v is not None else "N/A"

            celdas = [
                (_fmt(m['cagr'], '%'),
                 '#27ae60' if (m['cagr'] or 0) > 0 else '#e74c3c'),
                (_fmt(m['vol'], '%'), '#c8d6e5'),
                (_fmt(m['max_dd'], '%'),
                 '#e74c3c' if m['max_dd'] is not None else '#5a7a9a'),
                (_fmt(m['sharpe']),
                 '#27ae60' if (m['sharpe'] or 0) > 1 else '#f1c40f' if (m['sharpe'] or 0) > 0.5 else '#e74c3c'),
                (_fmt(m['calmar']),
                 '#27ae60' if (m['calmar'] or 0) > 1 else '#f1c40f' if (m['calmar'] or 0) > 0.5 else '#e74c3c'),
            ]
            for col, (texto, color) in enumerate(celdas, start=1):
                item = QTableWidgetItem(texto)
                item.setForeground(QColor('#5a7a9a') if texto == 'N/A' else QColor(color))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tabla.setItem(row, col, item)

    def _serie_display(self, datos):
        """Serie de cierres SOLO PARA DIBUJAR el grafico normalizado, con una
        resolucion fija independiente del TF comun de analisis: semanal (el
        aspecto suave de 1w) o diaria si el historico es corto y lo semanal
        quedaria a trompicones. El TF comun sigue mandando en correlacion,
        dispersion y stats — aqui solo se decide cuantos puntos se pintan."""
        s = datos['close']
        semanal = s.resample('1W').last().dropna()
        if len(semanal) >= 60:
            return semanal
        diaria = s.resample('1D').last().dropna()
        return diaria if len(diaria) >= 2 else s

    @_no_crash
    def _refresh_normalizado(self):
        self.fig_norm.clear()
        ax = self.fig_norm.add_subplot(111)
        _style_ax(ax)
        listos = self._series_listas()
        cargando = sum(1 for p in self._orden
                       if self._assets[p]['base'] is None and self._assets[p]['error'] is None)

        if not listos:
            _ax_placeholder(ax, "Cargando series..." if cargando else "Añade activos para comparar")
            self.canvas_norm.draw_idle()
            return

        modo = self.modo_combo.currentIndex()
        if modo == 0:
            # Interseccion temporal: base 100 en el inicio del activo mas joven
            t0 = max(datos['close'].index[0] for _p, _info, datos in listos)
            for _p, info, datos in listos:
                s = self._serie_display(datos)
                s = s[s.index >= t0]
                if len(s) < 2:
                    continue
                s = _downsample(s / s.iloc[0] * 100.0)
                ax.plot(s.index, s.values, color=info['color'],
                        linewidth=1.2, label=info['label'])
            ax.set_ylabel('Base 100 (T0 común)', fontsize=8)
        elif modo == 1:
            # Alineacion por dia de vida: eje X = dias desde el lanzamiento
            for _p, info, datos in listos:
                disp = self._serie_display(datos)
                s = _downsample(disp / disp.iloc[0] * 100.0)
                dias = (s.index - disp.index[0]).days
                ax.plot(dias, s.values, color=info['color'],
                        linewidth=1.2, label=info['label'])
            ax.set_xlabel('Días desde el lanzamiento', fontsize=8)
            ax.set_ylabel('Base 100 (día 1 de cada activo)', fontsize=8)
        else:
            # Historico completo: punto CAGR/vol por activo con TODO su historial
            for _p, info, datos in listos:
                ax.scatter(datos['vol'], datos['cagr'],
                           color=info['color'], s=60, zorder=3, label=info['label'])
                ax.annotate(info['label'], (datos['vol'], datos['cagr']),
                            textcoords='offset points', xytext=(6, 4),
                            color=info['color'], fontsize=7)
            ax.set_xlabel('Volatilidad anualizada (%)', fontsize=8)
            ax.set_ylabel('CAGR (%)', fontsize=8)

        if modo in (0, 1):
            ax.axhline(100, color='#5a7a9a', linewidth=0.7, linestyle='--', alpha=0.6)
            # Escala log automatica: si un activo multiplico >20x su base, la
            # escala lineal aplasta las demas curvas contra el eje X.
            # np.max (y no .max()): axhline devuelve ydata como lista Python.
            y_max = max((float(np.max(line.get_ydata())) for line in ax.lines
                         if len(line.get_ydata()) > 0), default=100.0)
            if y_max > 2000:
                ax.set_yscale('log')
                ax.set_ylabel(ax.get_ylabel() + ' [escala log]', fontsize=8)
        ax.legend(fontsize=7, framealpha=0.2, labelcolor=AX_FG, facecolor=FIG_BG)
        if cargando:
            ax.set_title(f"({cargando} serie(s) cargando...)", color='#5a7a9a', fontsize=8)
        try:
            self.fig_norm.tight_layout()
        except Exception:
            pass
        self.canvas_norm.draw_idle()

    @_no_crash
    def _refresh_corr(self):
        self.fig_corr.clear()
        ax = self.fig_corr.add_subplot(111)
        ax.set_facecolor(FIG_BG)
        listos = self._series_listas()
        if len(listos) < 2:
            _ax_placeholder(ax, "Se necesitan al menos 2 activos con serie cargada")
            self.canvas_corr.draw_idle()
            return

        # Correlacion de Pearson pairwise sobre retornos log del TF comun,
        # alineados por timestamp; min_periods evita correlaciones espurias
        # de pares con poco solape.
        retornos = pd.DataFrame({info['label']: datos['ret']
                                 for _p, info, datos in listos})
        corr = retornos.corr(min_periods=30)

        n = len(corr)
        data = corr.values.astype(float)
        masked = np.ma.masked_invalid(data)
        import matplotlib
        cmap = matplotlib.colormaps['RdYlGn'].copy()
        cmap.set_bad('#3a4a5a')
        im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=30, ha='right', fontsize=7, color=AX_FG)
        ax.set_yticklabels(corr.columns, fontsize=7, color=AX_FG)
        ax.tick_params(colors=AX_FG, length=0)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)

        for i in range(n):
            for j in range(n):
                v = data[i, j]
                if np.isnan(v):
                    ax.text(j, i, '—', ha='center', va='center', color='#8899aa', fontsize=8)
                else:
                    ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                            color='#111111', fontsize=8, fontweight='bold')

        cbar = self.fig_corr.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=AX_FG, labelsize=6)
        cbar.outline.set_edgecolor(GRID_C)
        try:
            self.fig_corr.tight_layout()
        except Exception:
            pass
        self.canvas_corr.draw_idle()

    def _rf_actual(self):
        try:
            return float(self.rf_input.text().replace(',', '.'))
        except ValueError:
            return 4.33

    _N_CARTERAS_MC = 20000

    def _calcular_frontera(self, listos):
        """Nube Monte Carlo de carteras long-only (sin cortos) con los activos
        cargados: pesos aleatorios Dirichlet, retorno/vol anualizados desde los
        retornos ARITMETICOS del TF comun sobre el SOLAPE de fechas comun a
        todos (una covarianza pairwise podria no ser positiva-definida).
        Devuelve dict {'vols','rets','pesos','labels'} en las mismas unidades
        (%) que los puntos del scatter, o None si no hay solape suficiente.
        Cacheado por (tf, activos): la nube no depende de Rf."""
        labels = [info['label'] for _p, info, _d in listos]
        clave = (self._tf, tuple(p for p, _i, _d in listos))
        if getattr(self, '_frontera_cache_key', None) == clave:
            return self._frontera_cache
        # retornos log alineados -> aritmeticos en el solape comun
        ret_log = pd.DataFrame({info['label']: datos['ret']
                                for _p, info, datos in listos}).dropna()
        if len(ret_log) < 30:
            # Identificar quien limita el solape para poder explicarlo en el
            # grafico: el activo que empieza mas tarde y el que termina antes.
            inicios = {info['label']: datos['ret'].index[0] for _p, info, datos in listos}
            fines = {info['label']: datos['ret'].index[-1] for _p, info, datos in listos}
            l_ini = max(inicios, key=inicios.get)
            l_fin = min(fines, key=fines.get)
            self._frontera_motivo = (
                f"Frontera no disponible: solo {len(ret_log)} velas comunes a todos "
                f"los activos (mínimo 30).\n"
                f"'{l_ini}' empieza el {inicios[l_ini].date()} y "
                f"'{l_fin}' termina el {fines[l_fin].date()}.")
            self._frontera_cache_key = clave
            self._frontera_cache = None
            return None
        self._frontera_motivo = None
        ret_arit = np.expm1(ret_log.values)
        anios = max((ret_log.index[-1] - ret_log.index[0]).days, 1) / 365.25
        ppy = len(ret_log) / anios  # periodos de mercado reales por año
        mu = ret_arit.mean(axis=0) * ppy          # retorno anual esperado
        cov = np.cov(ret_arit, rowvar=False) * ppy
        cov = np.atleast_2d(cov)

        rng = np.random.default_rng(42)  # semilla fija: nube estable entre refrescos
        w = rng.dirichlet(np.ones(len(listos)), size=self._N_CARTERAS_MC)
        rets = w @ mu * 100.0
        vols = np.sqrt(np.einsum('ij,jk,ik->i', w, cov, w)) * 100.0

        self._frontera_cache_key = clave
        self._frontera_cache = {'vols': vols, 'rets': rets, 'pesos': w, 'labels': labels}
        return self._frontera_cache

    @_no_crash
    def _refresh_scatter(self):
        self.fig_scatter.clear()
        ax = self.fig_scatter.add_subplot(111)
        _style_ax(ax)
        listos = self._series_listas()
        if not listos:
            _ax_placeholder(ax, "Añade activos para comparar")
            self.canvas_scatter.draw_idle()
            return

        vols = [datos['vol'] for _p, _info, datos in listos]
        cagrs = [datos['cagr'] for _p, _info, datos in listos]
        rf = self._rf_actual()

        bench_path = self.benchmark_combo.currentData()

        # ── Frontera eficiente de Markowitz (opcional) ──
        frontera = None
        if self.frontera_check.isChecked():
            aviso = None
            if len(listos) < 2:
                aviso = "Frontera no disponible: se necesitan al menos 2 activos cargados."
            else:
                frontera = self._calcular_frontera(listos)
                if frontera is None:
                    aviso = getattr(self, '_frontera_motivo', None) or \
                        "Frontera no disponible: sin solape temporal suficiente."
            if aviso:
                ax.text(0.5, 0.90, aviso, transform=ax.transAxes,
                        color='#f39c12', fontsize=7, ha='center', va='top',
                        fontweight='bold', alpha=0.9)
        if frontera is not None:
            f_vols, f_rets, f_w = frontera['vols'], frontera['rets'], frontera['pesos']
            # Nube de carteras posibles (long-only), tenue, detras de todo
            ax.scatter(f_vols, f_rets, s=2, color='#5a7a9a', alpha=0.06,
                       zorder=1, rasterized=True, linewidths=0)
            # Cartera tangente: maximo Sharpe (retorno-Rf)/vol de la nube
            with np.errstate(divide='ignore', invalid='ignore'):
                sharpes = np.where(f_vols > 0, (f_rets - rf) / f_vols, -np.inf)
            i_tan = int(np.argmax(sharpes))
            v_tan, r_tan = float(f_vols[i_tan]), float(f_rets[i_tan])
            ax.scatter(v_tan, r_tan, marker='*', s=200, color='#ffd54f',
                       edgecolors='#111111', linewidths=0.6, zorder=6)
            # CML: recta desde (0, Rf) pasando por la cartera tangente
            if v_tan > 0:
                slope = (r_tan - rf) / v_tan
                x_cml = np.linspace(0, max(f_vols.max(), max(vols)) * 1.1, 50)
                ax.plot(x_cml, rf + slope * x_cml,
                        color='#ffd54f', linewidth=0.9, linestyle='-.',
                        alpha=0.7, zorder=2)
                # Etiqueta CML FUERA del area del grafico (como Rf): x en coords
                # de ejes (>1, fuera a la derecha), y en coords de datos pegado al
                # extremo de la linea. No afecta a margenes del axes.
                import matplotlib.transforms as mtransforms
                trans_cml = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
                y_cml_end = rf + slope * x_cml[-1]
                ax.text(1.01, y_cml_end, 'CML', transform=trans_cml, clip_on=False,
                        color='#ffd54f', fontsize=6.5, alpha=0.8, fontweight='bold',
                        ha='left', va='center')
            # Composicion de la cartera tangente (pesos >3%), junto a la estrella
            pesos_txt = "  ".join(
                f"{lbl} {w_i:.0%}" for lbl, w_i in
                sorted(zip(frontera['labels'], f_w[i_tan]), key=lambda t: -t[1])
                if w_i >= 0.03)
            ax.annotate(f"Tangente: {pesos_txt}", (v_tan, r_tan),
                        textcoords='offset points', xytext=(8, -12),
                        color='#ffd54f', fontsize=6.5, alpha=0.9)

        for p, info, datos in listos:
            es_bench = (p == bench_path)
            ax.scatter(datos['vol'], datos['cagr'], color=info['color'],
                       s=110 if es_bench else 70, zorder=5 if es_bench else 4,
                       edgecolors='#ffffff' if es_bench else '#111111',
                       linewidths=1.5 if es_bench else 1.0)
            ax.annotate(info['label'] + (' (benchmark)' if es_bench else ''),
                        (datos['vol'], datos['cagr']),
                        textcoords='offset points', xytext=(7, 5),
                        color=info['color'], fontsize=7,
                        fontweight='bold' if es_bench else 'normal')

        # Cuadrantes: la cruceta se ancla en el benchmark elegido (su punto
        # (vol, cagr) en el TF actual pasa a ser el "cero" relativo de las 4
        # zonas); sin benchmark, o si aun esta cargando, en las medianas de
        # los activos cargados como hasta ahora.
        bench_datos = next((d for p, _i, d in listos if p == bench_path), None)
        if bench_datos is not None:
            x_med, y_med = float(bench_datos['vol']), float(bench_datos['cagr'])
        else:
            x_med = float(np.median(vols))
            y_med = float(np.median(cagrs))
        ax.axvline(x_med, color='#5a7a9a', linewidth=0.7, linestyle=':', alpha=0.7)
        ax.axhline(y_med, color='#5a7a9a', linewidth=0.7, linestyle=':', alpha=0.7)

        # Margenes para que las etiquetas de cuadrante no queden pegadas
        x_min, x_max = min(vols + [0]), max(vols)
        y_min, y_max = min(cagrs + [rf]), max(cagrs)
        if frontera is not None:
            # que la nube de carteras no quede recortada por los ejes
            x_min = min(x_min, float(frontera['vols'].min()))
            x_max = max(x_max, float(frontera['vols'].max()))
            y_min = min(y_min, float(frontera['rets'].min()))
            y_max = max(y_max, float(frontera['rets'].max()))
        x_pad = (x_max - x_min) * 0.15 or 1.0
        y_pad = (y_max - y_min) * 0.15 or 1.0
        ax.set_xlim(max(0, x_min - x_pad * 0.3), x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        etiquetas = [
            (0.02, 0.97, 'HIGH ALFA', '#2ecc71', 'left', 'top'),
            (0.98, 0.97, 'ALTA BETA', '#f1c40f', 'right', 'top'),
            (0.02, 0.03, 'ACTIVOS DEFENSIVOS', '#3498db', 'left', 'bottom'),
            (0.98, 0.03, 'RIESGO ASIMÉTRICO NEGATIVO', '#e74c3c', 'right', 'bottom'),
        ]
        for x, y, txt, c, ha, va in etiquetas:
            ax.text(x, y, txt, transform=ax.transAxes, color=c, fontsize=7,
                    alpha=0.65, ha=ha, va=va, fontweight='bold')

        # Linea fija de la tasa libre de riesgo: todo activo por debajo de
        # ella rinde menos que el T-Bill asumiendo ademas riesgo.
        ax.axhline(rf, color='#ffffff', linewidth=0.9, linestyle='--', alpha=0.55)
        # Etiqueta FUERA del area del grafico, pegada a la linea por la derecha
        # (x en coords de ejes >1, y en coords de datos, sin recorte)
        import matplotlib.transforms as mtransforms
        trans_rf = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.01, rf, f'Rf\n{rf:.3f}%', transform=trans_rf, clip_on=False,
                color='#ffffff', fontsize=6.5, alpha=0.8, va='center', ha='left')

        ax.set_xlabel('Volatilidad anualizada (%) → riesgo', fontsize=8)
        ax.set_ylabel('CAGR (%) → retorno', fontsize=8)
        try:
            self.fig_scatter.tight_layout()
            # dejar hueco a la derecha para la etiqueta Rf externa
            self.fig_scatter.subplots_adjust(right=0.90)
        except Exception:
            pass
        self.canvas_scatter.draw_idle()
