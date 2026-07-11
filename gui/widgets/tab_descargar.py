"""
Pestana Descargar - permite descargar datos de mercado de varios providers.

Estructura:
  - Dropdown con la lista de providers disponibles (Dukascopy...)
  - Boton Actualizar para refrescar el catalogo desde la API del provider
  - Buscador de activos
  - Lista (QListWidget) con los activos del catalogo (uno a la vez)
  - Botones de TF (1m, 5m, 15m, 1h, 4h, 1d)
  - Indicacion del rango disponible para el activo seleccionado
  - Botones Descargar / Detener
  - Consola (ConsoleWidget) que ejecuta descargar_datos.py como QProcess
"""
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QLineEdit, QComboBox, QListWidget,
                              QListWidgetItem, QFrame, QGroupBox, QButtonGroup,
                              QRadioButton, QProgressBar, QDateEdit, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal, QDate
from gui.widgets.console_widget import ConsoleWidget
from core.config import SCRIPTS_DIR
from core.data_providers.dukascopy_provider import DukascopyProvider
from core.data_providers.yfinance_provider import YFinanceProvider
from core.data_providers.ccxt_provider import CCXTProvider
from core.data_providers.hyperliquid_provider import HyperliquidProvider
from core.connectors import load_connectors, provider_class_for_type

STYLE_DOWNLOAD = """
QWidget { background-color: #141e30; }
QPushButton {
    background-color: #2a4a6a; color: #4fc3f7; border: none;
    padding: 8px 14px; border-radius: 4px; font-size: 12px; font-weight: bold;
}
QPushButton:hover { background-color: #3a5a8a; }
QPushButton:pressed { padding-top: 10px; padding-bottom: 6px; }
QPushButton:disabled { background-color: #1a2a45; color: #3a5a7a; }
QPushButton#danger { background-color: #3a1a1a; color: #e74c3c; }
QPushButton#danger:hover { background-color: #4a2525; }
QPushButton#success { background-color: #0f2a1a; color: #2ecc71; }
QPushButton#success:hover { background-color: #1a3a2a; }
QPushButton#tf {
    background-color: #1a2a45; color: #5a7a9a; padding: 6px 14px;
    font-size: 11px; min-width: 40px;
}
QPushButton#tf:checked {
    background-color: #2a4a6a; color: #4fc3f7; font-weight: bold;
}
QLabel#title { color: #4fc3f7; font-size: 13px; font-weight: bold; }
QLabel#path { color: #3a5a7a; font-size: 10px; padding: 2px 0; }
QLabel#range { color: #BA7517; font-size: 11px; padding: 4px 0; }
QLineEdit, QComboBox {
    background-color: #1a2a45; color: #c8d6e5; border: none;
    padding: 6px 10px; border-radius: 4px; font-size: 11px; min-width: 90px;
}
QComboBox::drop-down { border: none; background: transparent; width: 20px; }
QComboBox::down-arrow { border: none; }
QComboBox {
    combobox-popup: 0;
}
QComboBox QAbstractItemView {
    background-color: #1a2a45; color: #c8d6e5; selection-background-color: #2a4a6a;
    border: 1px solid #253a60; outline: none; margin: 0px;
}
QListWidget {
    background-color: #0d1424; color: #c8d6e5; border: 1px solid #253a60;
    border-radius: 4px; font-size: 11px; outline: none;
}
QListWidget::item { padding: 6px 10px; border-bottom: 1px solid #182030; }
QListWidget::item:selected { background-color: #2a4a6a; color: #4fc3f7; }
QListWidget::item:hover { background-color: #1a2a45; }
QFrame#sep { background-color: #253a60; max-height: 1px; }
QProgressBar {
    background-color: #1a2a45; border: none; border-radius: 4px;
    text-align: center; color: #c8d6e5; font-size: 10px; max-height: 18px;
}
QProgressBar::chunk { background-color: #4fc3f7; border-radius: 4px; }
"""

TF_LABELS = ['1m', '5m', '15m', '1h', '4h', '1d']

_SEPARATOR_SYMBOL = '__SEPARATOR__'

# Providers disponibles para el dropdown
_PROVIDERS = {
    'Dukascopy': DukascopyProvider,
    'yfinance': YFinanceProvider,
    'ccxt (Binance)': CCXTProvider,
    'Hyperliquid': HyperliquidProvider,
}


class TabDescargar(QWidget):
    download_completed = pyqtSignal(str)  # emite ruta del CSV

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE_DOWNLOAD)
        self._running = False
        self._selected_symbol = None
        self._catalog = []
        self._connector_configs = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── Toolbar superior: provider + buscador ──
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        toolbar.addWidget(QLabel("Fuente:"))
        self.provider_combo = QComboBox()
        self._rebuild_provider_combo()
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        toolbar.addWidget(self.provider_combo)

        self.btn_connector = QPushButton("+")
        self.btn_connector.setFixedSize(24, 24)
        self.btn_connector.setStyleSheet("""
            QPushButton {
                background-color: #1a4a2a; color: #2ecc71;
                border: none; border-radius: 4px;
                font-size: 18px; padding: 2px 0px 0px 0px; margin: 0px;
            }
            QPushButton:hover { background-color: #2a5a3a; }
            QPushButton:pressed { padding: 4px 0px 0px 1px; }
        """)
        self.btn_connector.clicked.connect(self._on_connector_clicked)
        toolbar.addWidget(self.btn_connector)

        self.btn_refresh = QPushButton("  Actualizar catalogo")
        self.btn_refresh.clicked.connect(self._refresh_catalog)
        toolbar.addWidget(self.btn_refresh)

        toolbar.addStretch()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Buscar activo...")
        self.search_input.setMaximumWidth(200)
        self.search_input.textChanged.connect(self._filter_list)
        toolbar.addWidget(self.search_input)

        layout.addLayout(toolbar)

        # ── Catalogo de activos + panel derecho ──
        content = QHBoxLayout()
        content.setSpacing(8)

        # Lista de activos
        self.asset_list = QListWidget()
        self.asset_list.setMaximumWidth(350)
        self.asset_list.itemSelectionChanged.connect(self._on_asset_selected)
        content.addWidget(self.asset_list, 1)

        # Panel derecho: TF + rango + botones
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # Rango disponible
        range_group = QGroupBox("Informacion del activo")
        range_layout = QVBoxLayout(range_group)
        self.label_symbol = QLabel("Selecciona un activo de la lista")
        self.label_symbol.setObjectName("title")
        self.label_range = QLabel("")
        self.label_range.setObjectName("range")
        range_layout.addWidget(self.label_symbol)
        range_layout.addWidget(self.label_range)
        right_panel.addWidget(range_group)

        # Date range
        date_group = QGroupBox("Rango de fechas")
        date_layout = QVBoxLayout(date_group)
        self.full_history_cb = QCheckBox("Historial Completo")
        self.full_history_cb.setChecked(False)
        self.full_history_cb.toggled.connect(self._on_full_history_toggled)
        date_layout.addWidget(self.full_history_cb)

        date_picker_row = QHBoxLayout()
        date_picker_row.setSpacing(6)
        date_picker_row.addWidget(QLabel("Inicio:"))
        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_start.setDate(QDate.currentDate().addYears(-3))
        self.date_start.setDisplayFormat("dd/MM/yyyy")
        date_picker_row.addWidget(self.date_start)
        date_picker_row.addWidget(QLabel("Fin:"))
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        self.date_end.setDate(QDate.currentDate())
        self.date_end.setDisplayFormat("dd/MM/yyyy")
        date_picker_row.addWidget(self.date_end)
        date_layout.addLayout(date_picker_row)
        right_panel.addWidget(date_group)

        # TF buttons
        tf_group = QGroupBox("Temporalidad")
        tf_layout = QHBoxLayout(tf_group)
        self._tf_button_group = QButtonGroup(self)
        self._tf_buttons = []
        for i, tf in enumerate(TF_LABELS):
            btn = QPushButton(tf)
            btn.setObjectName("tf")
            btn.setCheckable(True)
            if tf == '1h':
                btn.setChecked(True)
            self._tf_button_group.addButton(btn, i)
            tf_layout.addWidget(btn)
            self._tf_buttons.append(btn)
        right_panel.addWidget(tf_group)

        # Botones de accion
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        self.btn_download = QPushButton("  Descargar")
        self.btn_download.setObjectName("success")
        self.btn_download.clicked.connect(self._start_download)
        self.btn_download.setEnabled(False)
        self.btn_stop = QPushButton("  Detener")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self._stop_download)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_download)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addStretch()
        right_panel.addLayout(btn_layout)

        # Barra de progreso
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setVisible(False)
        right_panel.addWidget(self.progress)

        right_panel.addStretch()

        content_layout_widget = QWidget()
        content_layout_widget.setLayout(right_panel)
        content.addWidget(content_layout_widget, 1)

        layout.addLayout(content, 1)

        # ── Consola ──
        sep = QFrame()
        sep.setObjectName("sep")
        layout.addWidget(sep)

        self.console = ConsoleWidget()
        self.console.finished.connect(self._on_download_finished)
        self.console.progress.connect(self._on_progress)
        layout.addWidget(self.console, 1)

        # Cargar catalogo inicial
        self._load_catalog()

    # --- Catalogo -------------------------------------------------------------

    def _rebuild_provider_combo(self):
        self.provider_combo.blockSignals(True)
        current = self.provider_combo.currentText()
        self.provider_combo.clear()
        for name in _PROVIDERS:
            self.provider_combo.addItem(name)
        self._connector_configs = {}
        for c in load_connectors():
            name = c.get('name', '')
            self.provider_combo.addItem(name)
            self._connector_configs[name] = c
        idx = self.provider_combo.findText(current)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)
        self.provider_combo.blockSignals(False)

    def _current_provider(self):
        name = self.provider_combo.currentText()
        if name in _PROVIDERS:
            return _PROVIDERS.get(name)
        config = self._connector_configs.get(name)
        if config:
            return provider_class_for_type(config.get('type', ''))
        return None

    def _connector_env(self) -> dict:
        name = self.provider_combo.currentText()
        config = self._connector_configs.get(name)
        if not config:
            return {}
        return {
            'OANDA_API_KEY': config.get('api_key', ''),
            'OANDA_ACCOUNT_ID': config.get('account_id', ''),
            'OANDA_ENVIRONMENT': config.get('environment', 'practice'),
        }

    def _on_connector_clicked(self):
        from gui.widgets.connector_dialog import ConnectorDialog
        dlg = ConnectorDialog(parent=self)
        if dlg.exec() == ConnectorDialog.DialogCode.Accepted:
            self._rebuild_provider_combo()
            # Auto-seleccionar el conector recien creado
            if dlg._config:
                idx = self.provider_combo.findText(dlg._config.get('name', ''))
                if idx >= 0:
                    self.provider_combo.setCurrentIndex(idx)

    def _load_catalog(self):
        """Carga el catalogo del provider seleccionado."""
        provider = self._current_provider()
        if not provider:
            return
        self._catalog = provider.get_catalog()
        self._populate_list()

    def _refresh_catalog(self):
        """Refresca el catalogo forzando fetch desde la API."""
        self.btn_refresh.setEnabled(False)
        self.btn_refresh.setText("  Actualizando...")
        from PyQt6.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        provider = self._current_provider()
        if not provider:
            self.btn_refresh.setEnabled(True)
            self.btn_refresh.setText("  Actualizar catalogo")
            return
        if hasattr(provider, 'refresh_catalog'):
            self._catalog = provider.refresh_catalog()
        else:
            self._catalog = provider.get_catalog()
        self._populate_list()
        self.btn_refresh.setText("  Actualizar catalogo")
        self.btn_refresh.setEnabled(True)

    def _on_provider_changed(self, _name):
        self._load_catalog()

    def _populate_list(self):
        """Llena la QListWidget con los activos del catalogo filtrados."""
        search = self.search_input.text().strip().lower()
        self.asset_list.clear()
        show_separator = not search
        sep_added = False
        for asset in self._catalog:
            if asset.symbol == _SEPARATOR_SYMBOL:
                if show_separator and not sep_added:
                    item = QListWidgetItem(asset.name)
                    item.setFlags(Qt.ItemFlag.NoItemFlags)
                    self.asset_list.addItem(item)
                    sep_added = True
                continue
            symbol_lower = asset.symbol.lower()
            name_lower = asset.name.lower()
            if search and search not in symbol_lower and search not in name_lower:
                continue
            text = f"  {asset.symbol}    {asset.name}"
            if asset.category:
                text += f"   [{asset.category}]"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, asset)
            self.asset_list.addItem(item)

    def _filter_list(self):
        self._populate_list()

    # --- Seleccion de activo -------------------------------------------------

    def _on_asset_selected(self):
        items = self.asset_list.selectedItems()
        if not items:
            self._selected_symbol = None
            self.btn_download.setEnabled(False)
            self.label_symbol.setText("Selecciona un activo de la lista")
            self.label_range.setText("")
            return
        asset = items[0].data(Qt.ItemDataRole.UserRole)
        self._selected_symbol = asset
        self.btn_download.setEnabled(True)
        self.label_symbol.setText(f"{asset.symbol}  -  {asset.name}")
        if asset.max_history_start:
            self.label_range.setText(
                f"Historial disponible desde: {asset.max_history_start}\n"
                f"(Se descargara toda la historia disponible)"
            )
        else:
            self.label_range.setText("Historial disponible: desconocido")

    def _on_full_history_toggled(self, checked):
        self.date_start.setEnabled(not checked)
        self.date_end.setEnabled(not checked)

    # --- Descarga ------------------------------------------------------------

    def _selected_tf(self):
        for i, btn in enumerate(self._tf_buttons):
            if btn.isChecked():
                return TF_LABELS[i]
        return '1h'

    def _start_download(self):
        if not self._selected_symbol:
            return
        symbol = self._selected_symbol.symbol
        tf = self._selected_tf()

        provider_name = self.provider_combo.currentText()
        provider_key = 'dukascopy'
        if 'yfinance' in provider_name.lower():
            provider_key = 'yfinance'
        elif 'ccxt' in provider_name.lower():
            provider_key = 'ccxt'
        elif 'hyperliquid' in provider_name.lower():
            provider_key = 'hyperliquid'
        elif provider_name in self._connector_configs:
            config = self._connector_configs[provider_name]
            provider_key = config.get('type', 'oanda')

        env = {
            'DOWNLOAD_PROVIDER': provider_key,
            'DOWNLOAD_SYMBOL': symbol,
            'DOWNLOAD_TF': tf,
            'MPLBACKEND': 'Agg',
            'PYTHONIOENCODING': 'utf-8',
        }
        env.update(self._connector_env())
        if not self.full_history_cb.isChecked():
            env['DOWNLOAD_START'] = self.date_start.date().toString('yyyy-MM-dd')
            env['DOWNLOAD_END'] = self.date_end.date().toString('yyyy-MM-dd')
        import os as _os
        env['PATH'] = _os.environ.get('PATH', '')

        script_path = os.path.join(SCRIPTS_DIR, 'descargar_datos.py')
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.btn_download.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._running = True
        self.console.run(script_path, env=env)

    def _stop_download(self):
        self.console.stop()
        self.btn_stop.setEnabled(False)
        self._running = False

    def _on_progress(self, pct):
        self.progress.setValue(pct)

    def _on_download_finished(self, exit_code):
        self._running = False
        self.btn_download.setEnabled(self._selected_symbol is not None)
        self.btn_stop.setEnabled(False)
        self.progress.setVisible(False)
        if exit_code == 0:
            # Buscar el CSV generado en el output de la consola
            text = self.console.full_output
            # Parsear la ruta del CSV del output
            for line in text.splitlines():
                line = line.strip()
                if line.endswith('.csv') and os.path.exists(line):
                    self.download_completed.emit(line)
                    break