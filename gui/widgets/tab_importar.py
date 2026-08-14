import os, json, re, shutil
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSplitter, QFileDialog, QMessageBox,
                             QLineEdit, QComboBox, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from gui.widgets.file_explorer import FileExplorer
from gui.widgets.console_widget import ConsoleWidget
from gui.widgets.bombear import bombear_eventos
from gui.widgets import STYLE_ETIQUETA_SIN_CAJA
from gui.questdb_bootstrap import mostrar_bootstrap_questdb
from core.config import (get_base_data, LIMPIADOS_DIR, TF_LABELS, TIPO_LABELS,
                          TIPO_MAP, SCRIPTS_DIR, CATEGORIAS_DESCARGA_CONOCIDAS)
from core.rf_registry import guardar_rf as _guardar_rf_registro

# Categoría de descarga -> etiqueta que se preselecciona en «Tipo». El
# desplegable es más grueso que las categorías, así que varias caen en la
# misma opción: 'Stock' agrupa acciones, ETFs e índices (mismo criterio que
# usa reorganizar_limpiados.py), y metales/materias primas van a
# 'Futuro/Cfd', que es como se operan. 'OTROS' no está: si la ruta no dice
# nada, no se toca lo que el usuario tuviera puesto.
_TIPO_SUGERIDO = {
    'CRYPTO': 'Crypto',
    'FOREX': 'Forex',
    'ACCION': 'Stock',
    'ETF': 'Stock',
    'INDICE': 'Stock',
    'METAL-ETF': 'Stock',
    'FUTURO': 'Futuro/Cfd',
    'COMMODITIES': 'Futuro/Cfd',
    'METAL': 'Futuro/Cfd',
}


def _guardar_rf_sidecar(csv_path, rf):
    """Sidecar <csv>.import_info con el Rf usado, junto al archivo de origen.

    Se escribe AL INICIAR la importación, no al terminar: así, aunque la
    limpieza falle o se corte, al volver a seleccionar ese archivo en Importar
    el campo Rf se restaura con el valor que se intentó usar."""
    if rf is None or not csv_path:
        return
    try:
        with open(csv_path + '.import_info', 'w', encoding='utf-8') as f:
            json.dump({'rf_rate': rf}, f)
    except Exception:
        pass

_CHEVRON_SVG = os.path.join(os.path.dirname(__file__), '..', 'assets',
                            'chevron-down.svg').replace('\\', '/')

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
QLineEdit {
    background-color: #1a2a45; color: #c8d6e5; border: none;
    padding: 6px 10px; border-radius: 4px; font-size: 11px; min-width: 90px;
}
QComboBox {
    background-color: #111828; font-size: 11px; min-width: 90px;
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
QFrame#sep { background-color: #253a60; max-height: 1px; }
"""

class TabImportar(QWidget):
    import_completed = pyqtSignal()
    import_failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE_IMPORT.replace('__CHEVRON__', _CHEVRON_SVG))
        self._selected_file = None
        self._running_clean = False
        self._base_data = get_base_data()

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
        lbl_nombre = QLabel("Nombre:")
        lbl_nombre.setStyleSheet(STYLE_ETIQUETA_SIN_CAJA)
        toolbar.addWidget(lbl_nombre)
        toolbar.addWidget(self.nombre_input)

        self.tf_input = QComboBox()
        self.tf_input.addItems(TF_LABELS)
        self.tf_input.setCurrentText("1h")
        self.tf_input.setToolTip("Timeframe")
        lbl_tf = QLabel("TF:")
        lbl_tf.setStyleSheet(STYLE_ETIQUETA_SIN_CAJA)
        toolbar.addWidget(lbl_tf)
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
        lbl_rf = QLabel("Rf(%):")
        lbl_rf.setStyleSheet(STYLE_ETIQUETA_SIN_CAJA)
        toolbar.addWidget(lbl_rf)
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
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.explorer = FileExplorer(self._base_data, mode='exclude')
        self.explorer.file_chosen.connect(self._on_file_explorer_click)
        self.splitter.addWidget(self.explorer)

        self.console = ConsoleWidget()
        self.console.finished.connect(self._on_import_finished)
        self.splitter.addWidget(self.console)

        self.splitter.setSizes([350, 600])
        layout.addWidget(self.splitter, 1)
        bombear_eventos()

    def update_base_data(self, path):
        self._base_data = path
        old = self.explorer
        new = FileExplorer(path, mode='exclude')
        new.file_chosen.connect(self._on_file_explorer_click)
        idx = self.splitter.indexOf(old)
        self.splitter.replaceWidget(idx, new)
        old.setParent(None)
        old.deleteLater()
        self.explorer = new

    def _on_file_explorer_click(self, path):
        self._marcar_seleccion(path)

    def _filter_files(self, text: str):
        self.explorer.set_search_text(text)

    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo CSV/TXT",
            self._base_data, "CSV (*.csv);;TXT (*.txt);;Todos (*)"
        )
        if path:
            self._marcar_seleccion(path)

    def _marcar_seleccion(self, path):
        """Registra el archivo elegido y preselecciona «Tipo» según la
        categoría que se deduzca de su ruta.

        El desplegable arranca en «Futuro/Cfd» y es fácil olvidarse de
        cambiarlo, lo que acababa clasificando mal el activo (un crypto
        descargado en ccxt/CRYPTO/... terminaba fuera de Crypto). Es solo
        una propuesta: si la ruta no dice nada se deja como estaba, y el
        usuario siempre puede cambiarlo antes de importar."""
        self._selected_file = path
        self.path_label.setText(os.path.basename(path))
        self.btn_config.setEnabled(True)
        self.btn_delete.setEnabled(True)
        # Restaurar rf_rate del sidecar si existe
        info_path = path + '.import_info'
        if os.path.exists(info_path):
            try:
                with open(info_path) as f:
                    info = json.load(f)
                rf = info.get('rf_rate')
                if rf is not None:
                    self.rf_input.setText(str(rf))
            except Exception:
                pass
        etiqueta = _TIPO_SUGERIDO.get(self._detectar_categoria(path))
        if etiqueta:
            self.tipo_input.setCurrentText(etiqueta)

    def _detectar_categoria(self, src_path):
        """Intenta extraer la categoria de descarga (FOREX/METAL/CRYPTO/...)
        del path de origen, que si viene de Descargar ya trae la forma
        <proveedor>/<categoria>/<activo>/archivo.csv. Si no se puede
        determinar (archivo elegido a mano fuera de esa estructura),
        devuelve 'OTROS'."""
        try:
            relativo = os.path.relpath(src_path, self._base_data)
        except ValueError:
            return 'OTROS'
        for parte in os.path.normpath(relativo).split(os.sep):
            if parte.upper() in CATEGORIAS_DESCARGA_CONOCIDAS:
                return parte.upper()
        return 'OTROS'

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
            'categoria': self._detectar_categoria(self._selected_file),
            'rf_rate': rf_val,
        }

        self._config = cfg
        # Recordar el Rf junto al archivo de origen ANTES de lanzar el flujo:
        # si la limpieza falla o se corta, al re-seleccionar el archivo el Rf
        # se restaura igualmente (ver _marcar_seleccion).
        _guardar_rf_sidecar(self._selected_file, rf_val)
        # Y en el registro por archivo limpio, para que el Limpiador lo
        # recupere aunque el meta.json del CSV no exista todavía.
        try:
            cleaned_path = os.path.join(
                LIMPIADOS_DIR, cfg.get('categoria', 'OTROS'), cfg['nombre'],
                f"{cfg['nombre']}_{cfg['tf']}_limpiado.csv")
            _guardar_rf_registro(LIMPIADOS_DIR, cleaned_path, rf_val)
        except Exception:
            pass
        self.btn_config.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_delete.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.path_label.setText(f"Limpiando {cfg['nombre']} ({cfg['tf']})...")

        src = self._selected_file
        dst_dir = os.path.join(self._base_data, cfg['nombre'])
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

        config_path = os.path.join(self._base_data, "sesion_config.json")
        with open(config_path, 'w') as f:
            json.dump(cfg, f)

        env = {
            'CSV_INPUT': dst,
            'CONFIG_SKIP': '1',
            'CONFIG_NOMBRE': cfg['nombre'],
            'CONFIG_TF': cfg['tf'],
            'CONFIG_ACTIVO': cfg['activo'],
            # preparar_datos.py reescribe sesion_config.json y debe conservar
            # la categoria deducida de la ruta de origen: el limpiador la usa
            # para guardar el CSV limpio en su carpeta (FOREX, CRYPTO...).
            'CONFIG_CATEGORIA': cfg['categoria'],
            'MPLBACKEND': 'Agg',
            'PYTHONIOENCODING': 'utf-8',
        }

        # "Preparar" sube el resultado a QuestDB (con fallo silencioso) y
        # "Limpiar" (siguiente paso, _on_import_finished) SÍ la necesita para
        # funcionar — se prepara aquí (descarga/arranca sola si hace falta,
        # instantáneo si ya estaba disponible) para no encontrarnos el fallo
        # más tarde, a mitad del flujo.
        if not mostrar_bootstrap_questdb(self):
            self._reset_ui()
            return

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
                if os.path.normpath(parent) != os.path.normpath(self._base_data) and not os.listdir(parent):
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
            fase = 'limpieza' if self._running_clean else 'preparación'
            self._running_clean = False
            self._reset_ui()
            self.import_failed.emit(fase)

    def _save_meta(self):
        cfg = self._config
        if not cfg:
            return
        cleaned_path = os.path.join(
            LIMPIADOS_DIR, cfg.get('categoria', 'OTROS'), cfg['nombre'],
            f"{cfg['nombre']}_{cfg['tf']}_limpiado.csv"
        )
        meta = {
            'nombre': cfg['nombre'],
            'tf': cfg['tf'],
            'activo': cfg['activo'],
            # La categoría deducida de la ruta de origen es MÁS fiable que
            # 'activo' (que sale de un desplegable que empieza en Futuro/Cfd
            # y el usuario puede no tocar), así que se guarda para que
            # reorganizar_limpiados.py la prefiera al clasificar la carpeta.
            'categoria': cfg.get('categoria', 'OTROS'),
            'rf_rate': cfg.get('rf_rate', 0.0),
        }
        meta_path = cleaned_path + '.meta.json'
        try:
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass
        _guardar_rf_registro(LIMPIADOS_DIR, cleaned_path,
                             cfg.get('rf_rate', 0.0))

    def _reset_ui(self):
        self.btn_config.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_delete.setEnabled(bool(self._selected_file))
        if self._selected_file:
            self.path_label.setText(os.path.basename(self._selected_file))
