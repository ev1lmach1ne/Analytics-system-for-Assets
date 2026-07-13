import os, json, re
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QLineEdit, QFrame, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMessageBox, QAbstractItemView, QSplitter,
                              QComboBox, QStackedWidget, QProgressBar, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from gui.widgets.file_explorer import FileExplorer
from gui.widgets.console_widget import ConsoleWidget
from gui.widgets.rango_dialog import RangoAnalisisDialog

from core.config import BASE_DATA, LIMPIADOS_DIR, SCRIPTS_DIR, TF_PATTERN

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
QProgressBar {
    background-color: #1a2a45; border: none;
    border-radius: 7px; height: 16px; max-width: 120px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 rgba(255,255,255,0.30),
                                stop:0.4 #6dd5fa,
                                stop:0.7 #4fc3f7,
                                stop:1 rgba(0,0,0,0.2));
    border-radius: 6px;
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
QComboBox {
    background-color: #1a2a45; color: #c8d6e5; border: none;
    padding: 6px 10px; border-radius: 4px; font-size: 12px; min-width: 140px;
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
QFrame#sep { background-color: #253a60; max-height: 1px; }
"""

class TabLimpiados(QWidget):
    analysis_completed = pyqtSignal(str, str, str, str, str)
    file_selected = pyqtSignal(str, str)

    def set_analisis_tab(self, tab):
        self._analisis_tab = tab

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE_LIMPIADOS)
        self._selected_path = None
        self._selected_nombre = None
        self._selected_tf = '1h'
        self._selected_activo = 'FUTURO'
        self._cached_pdf = None
        self._cached_metrics = None
        self._analisis_tab = None
        self._grid_mode = True
        self._rf_rate = 0.0
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
        self.explorer = FileExplorer(LIMPIADOS_DIR, mode='csv')
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
        layout.addWidget(splitter, 1)

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
        lbl_rf = QLabel("Rf:")
        lbl_rf.setFixedWidth(16)
        btn_row.addWidget(lbl_rf)
        btn_row.addWidget(self.rf_input)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setToolTip("Progreso del análisis")
        self.progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_row.addWidget(self.progress)
        layout.addLayout(btn_row)

        # Initial scan for table mode
        self._scan_assets()

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

    def _scan_assets(self):
        assets = set()
        if os.path.isdir(LIMPIADOS_DIR):
            for entry in os.listdir(LIMPIADOS_DIR):
                sub = os.path.join(LIMPIADOS_DIR, entry)
                if os.path.isdir(sub):
                    for f in os.listdir(sub):
                        if f.endswith('_limpiado.csv') or f.endswith('_limpio.csv'):
                            assets.add(entry)
                            break
        self.asset_combo.blockSignals(True)
        self.asset_combo.clear()
        self.asset_combo.addItem("-- Seleccionar activo --")
        for a in sorted(assets):
            self.asset_combo.addItem(a)
        self.asset_combo.blockSignals(False)
        self.asset_table.setRowCount(0)

    def _on_asset_combo_changed(self, idx):
        if idx <= 0:
            self.asset_table.setRowCount(0)
            return
        nombre = self.asset_combo.currentText()
        asset_dir = os.path.join(LIMPIADOS_DIR, nombre)
        files = []
        if os.path.isdir(asset_dir):
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

    def _read_meta(self, path):
        meta_path = path + '.meta.json'
        nombre = None
        tf = None
        activo = 'FUTURO'

        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                nombre = meta.get('nombre')
                tf = meta.get('tf')
                activo = meta.get('activo', 'FUTURO')
                rf = meta.get('rf_rate', 0.0)
                self._rf_rate = rf
                if rf:
                    self.rf_input.setText(str(rf))
            except Exception:
                self._rf_rate = 0.0

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
                self.preview_table.setVisible(False)
                self.lbl_info.setText("")
                self.explorer.set_root_path(LIMPIADOS_DIR)
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
        self.progress.setValue(0)

        # Auto-save .meta.json for legacy files
        meta_path = self._selected_path + '.meta.json'
        if not os.path.exists(meta_path):
            try:
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'nombre': self._selected_nombre or '',
                        'tf': self._selected_tf or '1h',
                        'activo': self._selected_activo or 'FUTURO',
                        'rf_rate': 0.0,
                    }, f, indent=2)
            except Exception:
                pass

        horizon = self._analisis_tab.current_horizon if self._analisis_tab else 'General'
        try:
            rf_val = float(self.rf_input.text().strip() or '4.5')
        except ValueError:
            rf_val = 4.5
        cfg = {
            'nombre': self._selected_nombre or '',
            'tf': self._selected_tf or '1h',
            'activo': self._selected_activo or 'FUTURO',
            'input_path': self._selected_path,
            'horizonte': horizon,
            'rf_rate': rf_val,
        }
        config_path = os.path.join(BASE_DATA, "sesion_config.json")
        with open(config_path, 'w') as f:
            json.dump(cfg, f)

        metrics_path = os.path.join(BASE_DATA, "_gui_metrics.json")
        pdf_path_file = os.path.join(BASE_DATA, "_gui_pdf_path.json")

        env = {
            'CONFIG_PATH': config_path,
            'MPLBACKEND': 'Agg',
            'PYTHONIOENCODING': 'utf-8',
            'GUI_METRICS_OUTPUT': metrics_path,
            'GUI_PDF_OUTPUT': pdf_path_file,
            'USERPROFILE': os.environ.get('USERPROFILE', ''),
        }
        if rango_inicio is not None:
            env['GUI_RANGO_INICIO'] = rango_inicio.strftime('%Y-%m-%d')
            env['GUI_RANGO_FIN'] = rango_fin.strftime('%Y-%m-%d')

        self._metrics_path = metrics_path
        self._pdf_path_file = pdf_path_file

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

        # Cache to sidecar files
        if pdf_path and self._selected_path:
            import shutil

            m = re.search(r'_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.pdf$',
                          os.path.basename(pdf_path))
            rango_suffix = f"{m.group(1)}_to_{m.group(2)}" if m else None

            try:
                if rango_suffix:
                    pdf_cache_h = self._selected_path + f'.analysis.{rango_suffix}.pdf'
                    metrics_cache_h = self._selected_path + f'.analysis.{rango_suffix}.metrics.json'
                    shutil.copy2(pdf_path, pdf_cache_h)
                    if metrics_path:
                        shutil.copy2(metrics_path, metrics_cache_h)

                if self._rango_inicio is None:
                    pdf_cache = self._selected_path + '.analysis.pdf'
                    metrics_cache = self._selected_path + '.analysis.metrics.json'
                    shutil.copy2(pdf_path, pdf_cache)
                    if metrics_path:
                        shutil.copy2(metrics_path, metrics_cache)
                    self._cached_pdf = pdf_cache
                    self._cached_metrics = metrics_cache
                    self.btn_reanalyze.setVisible(True)
                    self.btn_reanalyze.setEnabled(True)
            except Exception as e:
                import traceback
                traceback.print_exc()

        self.analysis_completed.emit(
            pdf_path or '',
            metrics_path or '',
            self._selected_nombre or '',
            self._selected_tf or '',
            self._selected_path or ''
        )

    def _on_progress(self, pct):
        self.progress.setValue(pct)
