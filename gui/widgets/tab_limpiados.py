import os, json, re
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QLineEdit, QFrame, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMessageBox, QAbstractItemView, QSplitter,
                             QComboBox, QStackedWidget, QSizePolicy,
                             QApplication)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from gui.widgets.file_explorer import FileExplorer
from gui.widgets.console_widget import ConsoleWidget
from gui.widgets.rango_dialog import RangoAnalisisDialog
from gui.widgets.bombear import bombear_eventos
from gui.widgets import STYLE_ETIQUETA_SIN_CAJA
from gui.widgets.progress_animada import ProgressBarAnimada

from core.config import get_base_data, SCRIPTS_DIR, TF_PATTERN
from core.rf_registry import leer_rf as leer_rf_registro

_CHEVRON_SVG = os.path.join(os.path.dirname(__file__), '..', 'assets',
                            'chevron-down.svg').replace('\\', '/')

STYLE_LIMPIADOS = """
QWidget { background-color: #141e30; }
QPushButton {
    background-color: #2a4a6a; color: #4fc3f7; border: none;
    padding: 6px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;
}
QPushButton:hover { background-color: #3a5a8a; }
QPushButton:pressed { padding-top: 10px; padding-bottom: 6px; }
QPushButton:disabled { background-color: #1a2a45; color: #3a5a7a; }
QPushButton#danger { background-color: #3a1a1a; color: #e74c3c; }
QPushButton#danger:hover { background-color: #4a2525; }
QPushButton#danger:pressed { padding-top: 10px; padding-bottom: 6px; }
QPushButton#analyze { background-color: #0f2a1a; color: #2ecc71; }
QPushButton#analyze:hover { background-color: #1a3a2a; }
QPushButton#analyze:pressed { padding-top: 10px; padding-bottom: 6px; }
QPushButton#mode { background-color: #1a2a45; color: #7aaccc; }
QPushButton#mode:hover { background-color: #2a3a55; }
QPushButton#mode:pressed { padding-top: 10px; padding-bottom: 6px; }
QTableWidget {
    background-color: #0d1424; color: #aabbcc; border: 1px solid #253a60;
    gridline-color: #1a2a45; font-size: 11px;
}
QTableWidget::item { padding: 4px 8px; }
QHeaderView::section {
    background-color: #1a2a45; color: #4fc3f7; border: none;
    padding: 6px 8px; font-weight: bold; font-size: 11px;
}
QComboBox {
    background-color: #111828; font-size: 12px; min-width: 140px;
    combobox-popup: 0;
}
QComboBox::drop-down { border: none; background: transparent; width: 22px; }
QComboBox::down-arrow {
    image: url("__CHEVRON__");
    width: 12px; height: 8px;
}
QComboBox::down-arrow:on { }
QComboBox QAbstractItemView {
    background-color: #1a2a45; color: #c8d6e5;
    selection-background-color: #2a4a6a; selection-color: #4fc3f7;
    border: 1px solid #253a60; outline: none; margin: 0px;
}
QComboBox QAbstractItemView::item {
    padding: 6px 10px; border-bottom: 1px solid #182030;
}
QComboBox QAbstractItemView::item:selected {
    background-color: #2a4a6a; color: #4fc3f7;
}
QComboBox QAbstractItemView::item:hover {
    background-color: #223755;
}
QLineEdit {
    background-color: #1a2a45; color: #c8d6e5; border: none;
    padding: 6px 10px; border-radius: 4px; font-size: 11px;
}
QFrame#sep { background-color: #253a60; max-height: 1px; }
"""

# Anchos por defecto de los botones de la botonera inferior (se usan solo
# antes del primer layout; luego se calculan hasta el inicio de la consola).
_ANCHO_BTN_ANALIZAR = 220
_ANCHO_BTN_REANALIZAR = 200


class _ScanAssetsThread(QThread):
    """Escaneo de Limpiados/ en segundo plano (nombres de activos ordenados)."""

    terminado = pyqtSignal(list)

    def __init__(self, directorio, parent=None):
        super().__init__(parent)
        self._directorio = directorio

    def run(self):
        assets = set()
        if os.path.isdir(self._directorio):
            for root, dirs, files in os.walk(self._directorio):
                if 'Informes' in root:
                    continue
                if any(f.endswith('_limpiado.csv') or f.endswith('_limpio.csv')
                       for f in files):
                    assets.add(os.path.basename(root))
        self.terminado.emit(sorted(assets))


def _persistir_meta_rf(csv_path, nombre, tf, activo, rf):
    """Crea o actualiza <csv>.meta.json con la identidad y el Rf usado.

    Lo llama el análisis: aunque un archivo sea legacy (sin meta o con meta
    antigua), la primera vez que se analiza con un Rf, ese Rf queda guardado
    y la próxima selección rellena el campo del Limpiador."""
    if not csv_path:
        return
    meta_path = csv_path + '.meta.json'
    try:
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f) or {}
        meta.update({
            'nombre': nombre or '',
            'tf': tf or '1h',
            'activo': activo or 'FUTURO',
            'rf_rate': rf,
        })
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

class TabLimpiados(QWidget):
    analysis_completed = pyqtSignal(str, str, str, str, str)
    file_selected = pyqtSignal(str, str)

    def set_analisis_tab(self, tab):
        """Acepta el widget de Análisis o un callable que lo devuelva (la
        pestaña se construye bajo demanda; puede no existir aún)."""
        self._analisis_tab_ref = tab

    @property
    def _analisis_tab(self):
        ref = getattr(self, '_analisis_tab_ref', None)
        return ref() if callable(ref) else ref

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE_LIMPIADOS.replace('__CHEVRON__', _CHEVRON_SVG))
        self._selected_path = None
        self._selected_nombre = None
        self._selected_tf = '1h'
        self._selected_activo = 'FUTURO'
        self._cached_pdf = None
        self._cached_metrics = None
        self._analisis_tab_ref = None
        self._grid_mode = True
        self._rf_rate = 0.0
        self._scan_thread = None
        self._scan_done_cb = None
        self._base_data = get_base_data()
        # No usar la constante LIMPIADOS_DIR: se copia al importar el módulo
        # y en el primer arranque eso ocurre antes de que el usuario elija
        # la carpeta base en el FirstLaunchDialog.
        self._limpiados_dir = os.path.join(self._base_data, "Limpiados")
        self._rango_inicio = None
        self._rango_fin = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── Toolbar ──
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.btn_mode = QPushButton("Vista")
        self.btn_mode.setObjectName("mode")
        self.btn_mode.clicked.connect(self._toggle_mode)
        toolbar.addWidget(self.btn_mode)

        self.btn_preview = QPushButton(" Vista Previa CSV")
        self.btn_preview.clicked.connect(self._preview)
        self.btn_preview.setEnabled(False)

        self.btn_delete = QPushButton(" Eliminar")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.clicked.connect(self._delete)
        self.btn_delete.setEnabled(False)

        toolbar.addWidget(self.btn_preview)
        toolbar.addWidget(self.btn_delete)
        toolbar.addStretch()

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #4fc3f7; font-size: 11px; font-weight: bold;")
        toolbar.addWidget(self.lbl_info)

        layout.addLayout(toolbar)

        sep = QFrame()
        sep.setObjectName("sep")
        layout.addWidget(sep)

        # ── Buscador ──
        search_layout = QHBoxLayout()
        search_layout.setSpacing(8)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Buscar archivo o activo...")
        self.search_input.setMaximumWidth(300)
        self.search_input.textChanged.connect(self._filter_files)
        search_layout.addWidget(self.search_input)
        search_layout.addStretch()
        layout.addLayout(search_layout)

        # ── Splitter ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Stack: page 0 = FileExplorer (grid), page 1 = TableModeWidget
        self.mode_stack = QStackedWidget()

        # Page 0: FileExplorer
        self.explorer = FileExplorer(
            self._limpiados_dir, mode='csv',
            empty_hint="Aún no hay archivos importados.\n"
                       "Usa las pestañas Descargar o Importar para empezar.")
        self.explorer.file_chosen.connect(self._on_file_click)
        self.mode_stack.addWidget(self.explorer)

        # Page 1: Table mode (combo + asset TF table)
        self.table_mode_widget = QWidget()
        tm_layout = QVBoxLayout(self.table_mode_widget)
        tm_layout.setContentsMargins(0, 0, 0, 0)
        tm_layout.setSpacing(6)

        self.asset_combo = QComboBox()
        self.asset_combo.setToolTip("Seleccionar activo")
        self.asset_combo.currentIndexChanged.connect(self._on_asset_combo_changed)
        tm_layout.addWidget(self.asset_combo)

        self.asset_table = QTableWidget()
        self.asset_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.asset_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.asset_table.setAlternatingRowColors(True)
        self.asset_table.setStyleSheet("alternate-background-color: #111828;")
        self.asset_table.setColumnCount(4)
        self.asset_table.setHorizontalHeaderLabels(['TF', 'Periodo', 'Tipo', ''])
        self.asset_table.setColumnHidden(3, True)
        self.asset_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.asset_table.itemSelectionChanged.connect(self._on_tf_row_selected)
        tm_layout.addWidget(self.asset_table, 1)

        self.mode_stack.addWidget(self.table_mode_widget)

        splitter.addWidget(self.mode_stack)

        self.preview_table = QTableWidget()
        self.preview_table.setVisible(False)
        self.preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preview_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setStyleSheet("alternate-background-color: #111828;")
        splitter.addWidget(self.preview_table)

        self.console = ConsoleWidget()
        self.console.finished.connect(self._on_console_finished)
        self.console.progress.connect(self._on_progress)
        splitter.addWidget(self.console)

        splitter.setSizes([250, 300, 300])
        splitter.splitterMoved.connect(self._ajustar_boton_analizar)
        layout.addWidget(splitter, 1)
        bombear_eventos()

        # ── Analyze buttons ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_analyze = QPushButton("Analizar")
        self.btn_analyze.setObjectName("analyze")
        self.btn_analyze.clicked.connect(self._run_analysis)
        self.btn_analyze.setEnabled(False)
        btn_row.addWidget(self.btn_analyze)
        self.btn_reanalyze = QPushButton("Re-analizar")
        self.btn_reanalyze.setObjectName("danger")
        self.btn_reanalyze.clicked.connect(self._reanalyze)
        self.btn_reanalyze.setEnabled(False)
        self.btn_reanalyze.setVisible(False)
        btn_row.addWidget(self.btn_reanalyze)
        self.rf_input = QLineEdit()
        self.rf_input.setPlaceholderText("Rf %")
        self.rf_input.setText("0")
        self.rf_input.setToolTip("Tasa libre de riesgo anual (%)")
        self.rf_input.setMaximumWidth(55)
        lbl_rf = QLabel("Risk Free")
        lbl_rf.setStyleSheet(STYLE_ETIQUETA_SIN_CAJA)
        lbl_rf.setFixedWidth(lbl_rf.sizeHint().width() + 4)
        btn_row.addWidget(lbl_rf)
        btn_row.addWidget(self.rf_input)
        self.progress = ProgressBarAnimada()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(24)
        self.progress.setToolTip("Progreso del análisis")
        self.progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_row.addWidget(self.progress)
        layout.addLayout(btn_row)

        self._ajustar_boton_analizar()

        # Initial scan for table mode (en segundo plano para no bloquear el arranque)
        self._scan_assets_async()

    def _ancho_botones_consola(self):
        """Anchos de «Analizar»/«Re-analizar» para que el borde derecho de
        «Re-analizar» coincida con el inicio de la consola (el splitter la
        sitúa justo encima). Si aún no hay geometría válida, se usan los
        anchos por defecto."""
        consola_x = self.console.geometry().x()
        if consola_x <= 12:
            return _ANCHO_BTN_ANALIZAR, _ANCHO_BTN_REANALIZAR
        total = consola_x - 12 - 8
        ancho_ra = max(140, total * 10 // 21)
        ancho_a = max(160, total - ancho_ra)
        return ancho_a, ancho_ra

    def _ajustar_boton_analizar(self):
        """Con «Re-analizar» oculto, «Analizar» se ensancha hasta ocupar su
        hueco, de modo que «Risk Free» (y el campo Rf) queden siempre en la
        misma posición, haya o no análisis previo cacheado."""
        ancho_a, ancho_ra = self._ancho_botones_consola()
        if self.btn_reanalyze.isVisible():
            self.btn_analyze.setFixedWidth(ancho_a)
            self.btn_reanalyze.setFixedWidth(ancho_ra)
        else:
            self.btn_analyze.setFixedWidth(ancho_a + 8 + ancho_ra)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'btn_analyze'):
            self._ajustar_boton_analizar()

    # ── Search filter ──

    def _filter_files(self, text: str):
        if self._grid_mode:
            self.explorer.set_search_text(text)
        else:
            self._filter_table(text)

    def _filter_table(self, text: str):
        text = text.strip().lower()
        for row in range(self.asset_table.rowCount()):
            match = True
            if text:
                tf_item = self.asset_table.item(row, 0)
                per_item = self.asset_table.item(row, 1)
                tipo_item = self.asset_table.item(row, 2)
                row_text = ' '.join([
                    (tf_item.text() if tf_item else ''),
                    (per_item.text() if per_item else ''),
                    (tipo_item.text() if tipo_item else ''),
                ]).lower()
                match = text in row_text
            self.asset_table.setRowHidden(row, not match)

    # ── Mode toggle ──

    def _toggle_mode(self):
        self._grid_mode = not self._grid_mode
        self.mode_stack.setCurrentIndex(0 if self._grid_mode else 1)
        self.btn_mode.setText("Tabla" if self._grid_mode else "Grid")
        if not self._grid_mode:
            self._scan_assets()

    # ── Table mode: asset scanning ──

    def update_base_data(self, path):
        self._base_data = path
        self._limpiados_dir = os.path.join(path, "Limpiados")
        old = self.explorer
        new = FileExplorer(self._limpiados_dir, mode='csv')
        new.file_chosen.connect(self._on_file_click)
        idx = self.mode_stack.indexOf(old)
        self.mode_stack.removeWidget(old)
        self.mode_stack.insertWidget(idx, new)
        old.setParent(None)
        old.deleteLater()
        self.explorer = new
        self._scan_assets()

    def _scan_assets(self):
        """Escaneo síncrono de activos (para llamadas puntuales)."""
        self._poblar_activos(self._colectar_activos())

    def _colectar_activos(self):
        assets = set()
        if os.path.isdir(self._limpiados_dir):
            for root, dirs, files in os.walk(self._limpiados_dir):
                if 'Informes' in root:
                    continue
                if any(f.endswith('_limpiado.csv') or f.endswith('_limpio.csv')
                       for f in files):
                    assets.add(os.path.basename(root))
        return sorted(assets)

    def _poblar_activos(self, assets):
        self.asset_combo.blockSignals(True)
        self.asset_combo.clear()
        self.asset_combo.addItem("-- Seleccionar activo --")
        for a in assets:
            self.asset_combo.addItem(a)
        self.asset_combo.blockSignals(False)
        self.asset_table.setRowCount(0)

    def _scan_assets_async(self, done_cb=None):
        """Escaneo en segundo plano; al terminar rellena el combo. done_cb se
        llama en el hilo principal (p. ej. para ocultar el overlay)."""
        self._detener_scan()
        self._scan_done_cb = done_cb
        th = _ScanAssetsThread(self._limpiados_dir, parent=self)
        th.terminado.connect(self._on_scan_terminado)
        self._scan_thread = th
        th.start()

    def _detener_scan(self):
        th = getattr(self, '_scan_thread', None)
        if th is not None:
            try:
                th.requestInterruption()
                th.wait(150)
            except Exception:
                pass
            self._scan_thread = None

    def _on_scan_terminado(self, assets):
        self._poblar_activos(assets)
        self._scan_thread = None
        cb = self._scan_done_cb
        self._scan_done_cb = None
        if cb:
            try:
                cb()
            except Exception:
                pass

    def _apagar_hilos(self):
        """Espera a que termine el escaneo en vuelo antes de cerrar (si no,
        Qt destruiría un QThread vivo → abort del proceso). wait() bloqueante:
        bombear eventos en mitad del cierre corrompe el montón."""
        th = getattr(self, '_scan_thread', None)
        if th is not None:
            # desconectar: una señal encolada entregada en el teardown aborta
            try:
                th.terminado.disconnect()
            except (TypeError, RuntimeError):
                pass
            try:
                th.requestInterruption()
                th.wait()
            except Exception:
                pass
            self._scan_thread = None

    def _encontrar_carpeta_activo(self, nombre):
        """Localiza la carpeta de un activo dentro de Limpiados, sin asumir
        profundidad fija: puede estar directo bajo Limpiados/ (estructura
        antigua, o migracion aun no ejecutada) o dentro de una carpeta de
        categoria como Limpiados/FOREX/."""
        directo = os.path.join(self._limpiados_dir, nombre)
        if os.path.isdir(directo):
            return directo
        if os.path.isdir(self._limpiados_dir):
            for root, dirs, _files in os.walk(self._limpiados_dir):
                if 'Informes' in root:
                    continue
                if os.path.basename(root) == nombre:
                    return root
        return None

    def _on_asset_combo_changed(self, idx):
        if idx <= 0:
            self.asset_table.setRowCount(0)
            return
        nombre = self.asset_combo.currentText()
        asset_dir = self._encontrar_carpeta_activo(nombre)
        files = []
        if asset_dir and os.path.isdir(asset_dir):
            for f in os.listdir(asset_dir):
                if f.endswith('_limpiado.csv') or f.endswith('_limpio.csv'):
                    files.append(os.path.join(asset_dir, f))
        self.asset_table.setRowCount(len(files))
        for i, fp in enumerate(files):
            fname = os.path.basename(fp)
            m = TF_PATTERN.search(fname)
            tf = m.group(1) or m.group(2) if m else '?'
            periodo = self._read_period(fp)
            tipo = self._read_tipo(fp)
            self.asset_table.setItem(i, 0, QTableWidgetItem(tf))
            self.asset_table.setItem(i, 1, QTableWidgetItem(periodo))
            self.asset_table.setItem(i, 2, QTableWidgetItem(tipo))
            self.asset_table.setItem(i, 3, QTableWidgetItem(fp))

    def _read_period(self, path):
        try:
            first = pd.read_csv(path, nrows=1, parse_dates=['timestamp'], usecols=['timestamp'])
            t0 = first['timestamp'].iloc[0]
            with open(path, 'rb') as f:
                f.seek(-2000, 2)
                last_line = f.read().decode('utf-8', errors='replace').strip().split('\n')[-1]
            t1 = pd.to_datetime(last_line.split(',')[0])
            return f"{t0.strftime('%Y-%m-%d')} \u2192 {t1.strftime('%Y-%m-%d')}"
        except Exception:
            pass
        return "?"

    def _read_tipo(self, path):
        meta_path = path + '.meta.json'
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    return json.load(f).get('activo', '?')
            except Exception:
                pass
        return '?'

    def _on_tf_row_selected(self):
        rows = self.asset_table.selectedItems()
        if not rows:
            return
        row = rows[0].row()
        path_item = self.asset_table.item(row, 3)
        if path_item:
            self._select_file(path_item.text())

    # ── File selection (shared by both modes) ──

    def _on_file_click(self, path):
        self._select_file(path)

    def _select_file(self, path):
        self._selected_path = path
        self._cached_pdf = None
        self._cached_metrics = None
        self._read_meta(path)
        self.btn_preview.setEnabled(True)
        self.btn_delete.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.preview_table.setVisible(False)

        pdf_cache = path + '.analysis.pdf'
        metrics_cache = path + '.analysis.metrics.json'
        if os.path.exists(pdf_cache) and os.path.exists(metrics_cache):
            self._cached_pdf = pdf_cache
            self._cached_metrics = metrics_cache
            self.btn_reanalyze.setVisible(True)
            self.btn_reanalyze.setEnabled(True)
        else:
            self.btn_reanalyze.setVisible(False)
            self.btn_reanalyze.setEnabled(False)
        self._ajustar_boton_analizar()

    def _read_meta(self, path):
        meta_path = path + '.meta.json'
        nombre = None
        tf = None
        activo = 'FUTURO'

        rf_puesto = False
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                nombre = meta.get('nombre')
                tf = meta.get('tf')
                activo = meta.get('activo', 'FUTURO')
                # Solo se pisa el campo si el meta declara explícitamente el
                # rf_rate (aunque sea 0.0): un meta antiguo sin la clave no
                # debe sobrescribir lo que el usuario tuviera escrito.
                if 'rf_rate' in meta:
                    rf = meta['rf_rate']
                    self._rf_rate = rf
                    self.rf_input.setText(str(rf))
                    rf_puesto = True
            except Exception:
                pass
        if not rf_puesto:
            # Plan B: el registro por archivo que mantiene Importar (cubre
            # archivos legacy y limpiezas que no llegaron a escribir meta).
            rf = leer_rf_registro(self._limpiados_dir, path)
            if rf is not None:
                self._rf_rate = rf
                self.rf_input.setText(str(rf))

        if not nombre:
            nombre = os.path.basename(os.path.dirname(path))
        if not tf:
            m = TF_PATTERN.search(os.path.basename(path))
            if m:
                tf = m.group(1) or m.group(2) or '1h'
            else:
                tf = '1h'

        self._selected_nombre = nombre
        self._selected_tf = tf
        self._selected_activo = activo
        self.lbl_info.setText(f"{nombre} {tf} ({activo})")
        print(f"[DEBUG tab_limpiados] file_selected emit -> nombre={nombre!r} tf={tf!r}", flush=True)
        self.file_selected.emit(nombre or '', tf or '')

    # ── Preview, Delete ──

    def _preview(self):
        if self.preview_table.isVisible():
            self.preview_table.setVisible(False)
            return
        if not self._selected_path or not os.path.exists(self._selected_path):
            return
        try:
            df = pd.read_csv(self._selected_path, nrows=20)
            self.preview_table.setVisible(True)
            self.preview_table.setColumnCount(len(df.columns))
            self.preview_table.setHorizontalHeaderLabels(df.columns)
            self.preview_table.setRowCount(len(df))
            for i, (_, row) in enumerate(df.iterrows()):
                for j, val in enumerate(row):
                    item = QTableWidgetItem(str(val))
                    item.setForeground(Qt.GlobalColor.lightGray)
                    self.preview_table.setItem(i, j, item)
            self.preview_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents
            )
            self.preview_table.horizontalHeader().setStretchLastSection(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo leer el CSV:\n{e}")

    def _delete(self):
        if not self._selected_path or not os.path.exists(self._selected_path):
            return
        reply = QMessageBox.question(
            self, "Confirmar eliminacion",
            f"Eliminar {os.path.basename(self._selected_path)}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(self._selected_path)
                meta_path = self._selected_path + '.meta.json'
                if os.path.exists(meta_path):
                    os.remove(meta_path)
                pdf_cache = self._selected_path + '.analysis.pdf'
                metrics_cache = self._selected_path + '.analysis.metrics.json'
                for f in [pdf_cache, metrics_cache]:
                    if os.path.exists(f):
                        os.remove(f)
                self._selected_path = None
                self._selected_nombre = None
                self._cached_pdf = None
                self._cached_metrics = None
                self.btn_preview.setEnabled(False)
                self.btn_delete.setEnabled(False)
                self.btn_analyze.setEnabled(False)
                self.btn_reanalyze.setVisible(False)
                self.btn_reanalyze.setEnabled(False)
                self._ajustar_boton_analizar()
                self.preview_table.setVisible(False)
                self.lbl_info.setText("")
                self.explorer.set_root_path(self._limpiados_dir)
                self._scan_assets()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo eliminar:\n{e}")

    # ── Analysis ──

    def _run_analysis(self):
        if not self._selected_path:
            return

        # Si ya hay un analisis cacheado (todo el historico) listo, "Analizar"
        # va directo a metricas sin preguntar rango — el detector de rango
        # solo tiene sentido para volver a analizar (primera vez o
        # "Re-analizar", que limpia la cache antes de llamar aqui).
        if self._cached_pdf and self._cached_metrics:
            self._rango_inicio = None
            self._rango_fin = None
            self.analysis_completed.emit(
                self._cached_pdf,
                self._cached_metrics,
                self._selected_nombre or '',
                self._selected_tf or '',
                self._selected_path or ''
            )
            return

        dialog = RangoAnalisisDialog(self._selected_path, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        rango_inicio, rango_fin = dialog.get_rango()
        self._rango_inicio = rango_inicio
        self._rango_fin = rango_fin

        self.btn_analyze.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_delete.setEnabled(False)
        self.btn_reanalyze.setVisible(False)
        self.btn_reanalyze.setEnabled(False)
        self._ajustar_boton_analizar()
        self.progress.setValue(0)

        horizon = self._analisis_tab.current_horizon if self._analisis_tab else 'General'
        try:
            rf_val = float(self.rf_input.text().strip() or '4.5')
        except ValueError:
            rf_val = 4.5
        # Persistir el Rf (y la identidad) en el .meta.json del CSV limpio:
        # el campo del Limpiador se rellena en la próxima selección y el
        # script de análisis lo usa aunque sesion_config.json se sobrescriba.
        _persistir_meta_rf(
            self._selected_path, self._selected_nombre, self._selected_tf,
            self._selected_activo, rf_val)
        cfg = {
            'nombre': self._selected_nombre or '',
            'tf': self._selected_tf or '1h',
            'activo': self._selected_activo or 'FUTURO',
            'input_path': self._selected_path,
            'horizonte': horizon,
            'rf_rate': rf_val,
        }
        config_path = os.path.join(self._base_data, "sesion_config.json")
        with open(config_path, 'w') as f:
            json.dump(cfg, f)

        metrics_path = os.path.join(self._base_data, "_gui_metrics.json")
        pdf_path_file = os.path.join(self._base_data, "_gui_pdf_path.json")
        # Datos crudos de cada gráfico, para que el Analizador los dibuje
        # nativamente en vez de mostrar la imagen del PDF.
        plotdata_path = os.path.join(self._base_data, "_gui_plotdata.pkl")

        env = {
            'CONFIG_PATH': config_path,
            'MPLBACKEND': 'Agg',
            'PYTHONIOENCODING': 'utf-8',
            'GUI_METRICS_OUTPUT': metrics_path,
            'GUI_PDF_OUTPUT': pdf_path_file,
            'GUI_PLOTDATA_OUTPUT': plotdata_path,
            'USERPROFILE': os.environ.get('USERPROFILE', ''),
        }
        if rango_inicio is not None:
            env['GUI_RANGO_INICIO'] = rango_inicio.strftime('%Y-%m-%d')
            env['GUI_RANGO_FIN'] = rango_fin.strftime('%Y-%m-%d')

        self._metrics_path = metrics_path
        self._pdf_path_file = pdf_path_file
        self._plotdata_path = plotdata_path

        # Liberar el PDF actualmente cargado en el visor ANTES de lanzar el análisis:
        # si el nuevo informe reutiliza el mismo nombre de archivo (mismo activo/tf/
        # rango), QPdfDocument sigue reteniendo el handle del PDF anterior durante
        # toda la corrida y el script no puede sobrescribirlo (PermissionError).
        if self._analisis_tab is not None:
            self._analisis_tab.graphs_viewer.load(None)

        self.console.run(
            os.path.join(SCRIPTS_DIR, "analisis_descriptivo.py"),
            env=env
        )

    def _reanalyze(self):
        self._cached_pdf = None
        self._cached_metrics = None
        self.btn_reanalyze.setVisible(False)
        self.btn_reanalyze.setEnabled(False)
        self._ajustar_boton_analizar()
        self._run_analysis()

    def _on_console_finished(self, exit_code):
        self.progress.setValue(100)
        self.btn_analyze.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.btn_delete.setEnabled(True)

        if exit_code != 0:
            return

        pdf_path = None
        if os.path.exists(self._pdf_path_file):
            try:
                with open(self._pdf_path_file) as f:
                    data = json.load(f)
                    pdf_path = data.get('pdf_path')
            except Exception as e:
                import traceback
                traceback.print_exc()

        metrics_path = self._metrics_path if os.path.exists(self._metrics_path) else None
        plotdata_path = getattr(self, '_plotdata_path', None)
        plotdata_path = plotdata_path if (plotdata_path and os.path.exists(plotdata_path)) else None

        # Rutas que se emiten a TabAnalisis. Por defecto las temporales del
        # propio proceso (GUI_PDF_OUTPUT/GUI_METRICS_OUTPUT), pero se
        # sustituyen por el sidecar en cuanto exista: TabAnalisis._ruta_bundle()
        # deriva la ruta del .plotdata.pkl a partir del PDF que recibe (mismo
        # prefijo + '.plotdata.pkl'), y ese emparejamiento solo existe junto al
        # sidecar (`<csv>.analysis.<rango>.pdf` / `.plotdata.pkl`) — nunca junto
        # a la ruta temporal, que vive en una carpeta distinta con otro nombre.
        # Sin este cambio la pestaña Gráficos caía siempre al visor de PDF
        # antiguo con el aviso de "versión anterior", aunque el bundle se
        # acabara de generar en el mismo análisis.
        pdf_emitir = pdf_path
        metrics_emitir = metrics_path

        # Cache to sidecar files
        rango_suffix = None
        if pdf_path and self._selected_path:
            import shutil

            m = re.search(r'_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.pdf$',
                          os.path.basename(pdf_path))
            rango_suffix = f"{m.group(1)}_to_{m.group(2)}" if m else None

            try:
                if rango_suffix:
                    pdf_cache_h = self._selected_path + f'.analysis.{rango_suffix}.pdf'
                    metrics_cache_h = self._selected_path + f'.analysis.{rango_suffix}.metrics.json'
                    plot_cache_h = self._selected_path + f'.analysis.{rango_suffix}.plotdata.pkl'
                    shutil.copy2(pdf_path, pdf_cache_h)
                    if metrics_path:
                        shutil.copy2(metrics_path, metrics_cache_h)
                    if plotdata_path:
                        shutil.copy2(plotdata_path, plot_cache_h)

                if self._rango_inicio is None:
                    pdf_cache = self._selected_path + '.analysis.pdf'
                    metrics_cache = self._selected_path + '.analysis.metrics.json'
                    plot_cache = self._selected_path + '.analysis.plotdata.pkl'
                    shutil.copy2(pdf_path, pdf_cache)
                    if metrics_path:
                        shutil.copy2(metrics_path, metrics_cache)
                    if plotdata_path:
                        shutil.copy2(plotdata_path, plot_cache)
                    self._cached_pdf = pdf_cache
                    self._cached_metrics = metrics_cache
                    self.btn_reanalyze.setVisible(True)
                    self.btn_reanalyze.setEnabled(True)
                    self._ajustar_boton_analizar()
            except Exception as e:
                import traceback
                traceback.print_exc()

        # El sidecar por rango es el mismo que lista periodo_combo en
        # TabAnalisis (_poblar_periodo_combo busca '.analysis.*_to_*.pdf'):
        # emitirlo aquí también hace que la selección inicial del combo
        # coincida con el análisis recién hecho. Se comprueba con
        # os.path.exists (no si el copy "declaró éxito") para no emitir un
        # sidecar que haya fallado a mitad de copiar.
        if rango_suffix and self._selected_path:
            pdf_cache_h = self._selected_path + f'.analysis.{rango_suffix}.pdf'
            metrics_cache_h = self._selected_path + f'.analysis.{rango_suffix}.metrics.json'
            if os.path.exists(pdf_cache_h):
                pdf_emitir = pdf_cache_h
                metrics_emitir = metrics_cache_h if os.path.exists(metrics_cache_h) else None

        self.analysis_completed.emit(
            pdf_emitir or '',
            metrics_emitir or '',
            self._selected_nombre or '',
            self._selected_tf or '',
            self._selected_path or ''
        )

    def _on_progress(self, pct):
        self.progress.setValue(pct)
