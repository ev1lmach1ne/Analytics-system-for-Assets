import os, json, re, shutil
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSplitter, QFileDialog, QMessageBox,
                             QLineEdit, QComboBox, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from gui.widgets.file_explorer import FileExplorer
from gui.widgets.console_widget import ConsoleWidget
from core.config import BASE_DATA, LIMPIADOS_DIR, TF_LABELS, TIPO_LABELS, TIPO_MAP, SCRIPTS_DIR

STYLE_IMPORT = """
QWidget { background-color: #141e30; }
QPushButton {
    background-color: #2a4a6a; color: #4fc3f7; border: none;
    padding: 8px 18px; border-radius: 4px; font-size: 12px; font-weight: bold;
}
QPushButton:hover { background-color: #3a5a8a; }
QPushButton:pressed { padding-top: 10px; padding-bottom: 6px; }
QPushButton:disabled { background-color: #1a2a45; color: #3a5a7a; }
QPushButton#danger { background-color: #3a1a1a; color: #e74c3c; }
QPushButton#danger:hover { background-color: #4a2525; }
QPushButton#danger:pressed { padding-top: 10px; padding-bottom: 6px; }
QPushButton#success { background-color: #0f2a1a; color: #2ecc71; }
QPushButton#success:hover { background-color: #1a3a2a; }
QPushButton#success:pressed { padding-top: 10px; padding-bottom: 6px; }
QLabel#path { color: #3a5a7a; font-size: 10px; padding: 2px 0; }
QLabel#title { color: #4fc3f7; font-size: 13px; font-weight: bold; }
QLineEdit, QComboBox {
    background-color: #1a2a45; color: #c8d6e5; border: none;
    padding: 6px 10px; border-radius: 4px; font-size: 11px; min-width: 90px;
}
QComboBox { combobox-popup: 0; }
QComboBox::drop-down { border: none; background: transparent; width: 20px; }
QComboBox::down-arrow { border: none; }
QComboBox QAbstractItemView {
    background-color: #1a2a45; color: #c8d6e5; selection-background-color: #2a4a6a;
    border: 1px solid #253a60; outline: none; margin: 0px;
}
QFrame#sep { background-color: #253a60; max-height: 1px; }
"""

class TabImportar(QWidget):
    import_completed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE_IMPORT)
        self._selected_file = None
        self._running_clean = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # ── Toolbar ──
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.btn_select = QPushButton(" Seleccionar CSV")
        self.btn_select.clicked.connect(self._select_file)

        toolbar.addWidget(self.btn_select)

        self.nombre_input = QLineEdit()
        self.nombre_input.setPlaceholderText("nombre")
        self.nombre_input.setToolTip("Nombre del activo (ej: xauusd)")
        self.nombre_input.setMaximumWidth(120)
        toolbar.addWidget(QLabel("Nombre:"))
        toolbar.addWidget(self.nombre_input)

        self.tf_input = QComboBox()
        self.tf_input.addItems(TF_LABELS)
        self.tf_input.setCurrentText("1h")
        self.tf_input.setToolTip("Timeframe")
        toolbar.addWidget(QLabel("TF:"))
        toolbar.addWidget(self.tf_input)

        self.tipo_input = QComboBox()
        self.tipo_input.addItems(TIPO_LABELS)
        self.tipo_input.setToolTip("Tipo de activo")
        toolbar.addWidget(QLabel("Tipo:"))
        toolbar.addWidget(self.tipo_input)

        self.rf_input = QLineEdit()
        self.rf_input.setPlaceholderText("4.5")
        self.rf_input.setText("0")
        self.rf_input.setToolTip("Tasa libre de riesgo anual (%)")
        self.rf_input.setMaximumWidth(55)
        toolbar.addWidget(QLabel("Rf(%):"))
        toolbar.addWidget(self.rf_input)

        self.btn_config = QPushButton(" Limpiar")
        self.btn_config.setObjectName("success")
        self.btn_config.setObjectName("success")
        self.btn_config.clicked.connect(self._run_import)
        self.btn_config.setEnabled(False)

        self.btn_stop = QPushButton(" Detener")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self._stop_import)
        self.btn_stop.setEnabled(False)

        self.btn_delete = QPushButton(" Eliminar")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.setToolTip("Eliminar el archivo seleccionado")
        self.btn_delete.clicked.connect(self._delete_file)
        self.btn_delete.setEnabled(False)

        toolbar.addWidget(self.btn_config)
        toolbar.addWidget(self.btn_stop)
        toolbar.addWidget(self.btn_delete)
        toolbar.addStretch()

        self.path_label = QLabel("Ningun archivo seleccionado")
        self.path_label.setObjectName("path")
        toolbar.addWidget(self.path_label)

        layout.addLayout(toolbar)

        sep = QFrame()
        sep.setObjectName("sep")
        layout.addWidget(sep)

        # ── Buscador ──
        search_layout = QHBoxLayout()
        search_layout.setSpacing(8)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Buscar archivo...")
        self.search_input.setMaximumWidth(300)
        self.search_input.textChanged.connect(self._filter_files)
        search_layout.addWidget(self.search_input)
        search_layout.addStretch()
        layout.addLayout(search_layout)

        # ── Splitter: FileExplorer (left) + Console (right) ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.explorer = FileExplorer(BASE_DATA, mode='exclude')
        self.explorer.file_chosen.connect(self._on_file_explorer_click)
        splitter.addWidget(self.explorer)

        self.console = ConsoleWidget()
        self.console.finished.connect(self._on_import_finished)
        splitter.addWidget(self.console)

        splitter.setSizes([350, 600])
        layout.addWidget(splitter, 1)

    def _on_file_explorer_click(self, path):
        self._selected_file = path
        self.path_label.setText(os.path.basename(path))
        self.btn_config.setEnabled(True)
        self.btn_delete.setEnabled(True)

    def _filter_files(self, text: str):
        self.explorer.set_search_text(text)

    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo CSV/TXT",
            BASE_DATA, "CSV (*.csv);;TXT (*.txt);;Todos (*)"
        )
        if path:
            self._selected_file = path
            self.path_label.setText(os.path.basename(path))
            self.btn_config.setEnabled(True)
            self.btn_delete.setEnabled(True)

    def _run_import(self):
        if not self._selected_file:
            return

        # Solo letras y numeros: el nombre se usa como identificador de tabla
        # SQL y los scripts separan activo/tf con '_', asi que guiones,
        # espacios y '_' se eliminan (msci-acwi -> msciacwi).
        nombre = re.sub(r'[^a-z0-9]', '', self.nombre_input.text().strip().lower())
        if not nombre:
            QMessageBox.warning(self, "Error", "El nombre del activo no puede estar vacio.")
            self.nombre_input.setFocus()
            return

        try:
            rf_val = float(self.rf_input.text().strip() or '4.5')
        except ValueError:
            rf_val = 4.5
        cfg = {
            'nombre': nombre,
            'tf': self.tf_input.currentText(),
            'activo': TIPO_MAP[self.tipo_input.currentText()],
            'rf_rate': rf_val,
        }

        self._config = cfg
        self.btn_config.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_delete.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.path_label.setText(f"Limpiando {cfg['nombre']} ({cfg['tf']})...")

        src = self._selected_file
        dst_dir = os.path.join(BASE_DATA, cfg['nombre'])
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        if src != dst:
            try:
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.copy2(src, dst)
            except Exception as e:
                try:
                    shutil.copyfile(src, dst)
                except Exception as e2:
                    QMessageBox.critical(self, "Error", f"No se pudo copiar el archivo:\n{e}\n\n{e2}")
                    self._reset_ui()
                    return

        config_path = os.path.join(BASE_DATA, "sesion_config.json")
        with open(config_path, 'w') as f:
            json.dump(cfg, f)

        env = {
            'CSV_INPUT': dst,
            'CONFIG_SKIP': '1',
            'CONFIG_NOMBRE': cfg['nombre'],
            'CONFIG_TF': cfg['tf'],
            'CONFIG_ACTIVO': cfg['activo'],
            'MPLBACKEND': 'Agg',
            'PYTHONIOENCODING': 'utf-8',
        }

        preparar_py = os.path.join(SCRIPTS_DIR, "preparar_datos.py")
        self.console.run(preparar_py, env=env)

    def _delete_file(self):
        if not self._selected_file or not os.path.exists(self._selected_file):
            return
        reply = QMessageBox.question(
            self, "Confirmar eliminacion",
            f"Eliminar {os.path.basename(self._selected_file)}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(self._selected_file)
                # Si la carpeta del activo queda vacia, se elimina tambien
                parent = os.path.dirname(self._selected_file)
                if os.path.normpath(parent) != os.path.normpath(BASE_DATA) and not os.listdir(parent):
                    os.rmdir(parent)
                self._selected_file = None
                self.path_label.setText("Ningun archivo seleccionado")
                self.btn_config.setEnabled(False)
                self.btn_delete.setEnabled(False)
                self.explorer.refresh()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo eliminar:\n{e}")

    def _stop_import(self):
        self.console.stop()
        self._reset_ui()

    def _on_import_finished(self, exit_code):
        if exit_code == 0 and self._config and not self._running_clean:
            self._running_clean = True
            limpieza_py = os.path.join(SCRIPTS_DIR, "limpieza_datos_er.py")
            env = {
                'MPLBACKEND': 'Agg',
                'PYTHONIOENCODING': 'utf-8',
            }
            self.console.run(limpieza_py, env=env, clear=False)
        elif exit_code == 0 and self._running_clean:
            self._running_clean = False
            self._reset_ui()
            self._save_meta()
            self.import_completed.emit()
        else:
            self._running_clean = False
            self._reset_ui()
            self.import_completed.emit()

    def _save_meta(self):
        cfg = self._config
        if not cfg:
            return
        cleaned_path = os.path.join(
            LIMPIADOS_DIR, cfg['nombre'],
            f"{cfg['nombre']}_{cfg['tf']}_limpiado.csv"
        )
        meta = {
            'nombre': cfg['nombre'],
            'tf': cfg['tf'],
            'activo': cfg['activo'],
            'rf_rate': cfg.get('rf_rate', 0.0),
        }
        meta_path = cleaned_path + '.meta.json'
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    def _reset_ui(self):
        self.btn_config.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_delete.setEnabled(bool(self._selected_file))
        if self._selected_file:
            self.path_label.setText(os.path.basename(self._selected_file))
