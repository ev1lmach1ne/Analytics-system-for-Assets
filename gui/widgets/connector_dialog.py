"""Dialogo para anadir o editar conectores de datos (OANDA, etc.)."""

import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QLineEdit, QComboBox, QPushButton, QMessageBox,
                              QFormLayout, QGroupBox, QFrame)
from PyQt6.QtCore import Qt
from core.connectors import load_connectors, save_connector, delete_connector

STYLE = """
QDialog {
    background-color: #141e30; color: #c8d6e5;
}
QLabel {
    color: #c8d6e5; font-size: 11px;
}
QLineEdit, QComboBox {
    background-color: #1a2a45; color: #c8d6e5; border: none;
    padding: 6px 10px; border-radius: 4px; font-size: 11px;
}
QComboBox::drop-down { border: none; background: transparent; width: 20px; }
QComboBox::down-arrow { border: none; }
QComboBox QAbstractItemView {
    background-color: #1a2a45; color: #c8d6e5; selection-background-color: #2a4a6a;
    border: 1px solid #253a60; outline: none;
}
QPushButton {
    background-color: #2a4a6a; color: #4fc3f7; border: none;
    padding: 8px 14px; border-radius: 4px; font-size: 11px; font-weight: bold;
}
QPushButton:hover { background-color: #3a5a8a; }
QPushButton#test {
    background-color: #0f2a1a; color: #2ecc71;
}
QPushButton#test:hover { background-color: #1a3a2a; }
QPushButton#delete {
    background-color: #3a1a1a; color: #e74c3c;
}
QPushButton#delete:hover { background-color: #4a2525; }
QPushButton#ok {
    background-color: #1a4a2a; color: #2ecc71;
}
QPushButton#ok:hover { background-color: #2a5a3a; }
QFrame#sep { background-color: #253a60; max-height: 1px; }
"""


class ConnectorDialog(QDialog):
    def __init__(self, edit_name=None, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE)
        self.setWindowTitle("Conector de datos" if not edit_name else f"Editar: {edit_name}")
        self.setMinimumWidth(420)

        self._edit_name = edit_name
        self._config = None
        if edit_name:
            for c in load_connectors():
                if c.get('name') == edit_name:
                    self._config = c
                    break

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Conector de datos")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #4fc3f7;")
        layout.addWidget(title)
        sep = QFrame()
        sep.setObjectName("sep")
        layout.addWidget(sep)

        form = QFormLayout()
        form.setSpacing(8)

        self.type_combo = QComboBox()
        self.type_combo.addItem("OANDA")
        self.type_combo.setEnabled(self._config is None)
        form.addRow("Tipo:", self.type_combo)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Ej: OANDA Demo")
        if self._config:
            self.name_edit.setText(self._config.get('name', ''))
        else:
            existing = {c.get('name') for c in load_connectors()}
            base = "OANDA Demo"
            name = base
            i = 2
            while name in existing:
                name = f"{base} {i}"
                i += 1
            self.name_edit.setText(name)
        form.addRow("Nombre:", self.name_edit)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("OANDA API Key")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        if self._config:
            self.api_key_edit.setText(self._config.get('api_key', ''))
        form.addRow("API Key:", self.api_key_edit)

        self.account_edit = QLineEdit()
        self.account_edit.setPlaceholderText("101-xxx-xxxxxxx-xxx")
        if self._config:
            self.account_edit.setText(self._config.get('account_id', ''))
        form.addRow("Account ID:", self.account_edit)

        self.env_combo = QComboBox()
        self.env_combo.addItems(["practice", "live"])
        if self._config:
            idx = self.env_combo.findText(self._config.get('environment', 'practice'))
            if idx >= 0:
                self.env_combo.setCurrentIndex(idx)
        form.addRow("Entorno:", self.env_combo)

        layout.addLayout(form)

        # Test connection
        self.btn_test = QPushButton("  Test Connection")
        self.btn_test.setObjectName("test")
        self.btn_test.clicked.connect(self._test_connection)
        layout.addWidget(self.btn_test)

        sep2 = QFrame()
        sep2.setObjectName("sep")
        layout.addWidget(sep2)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        if self._config:
            self.btn_delete = QPushButton("  Eliminar")
            self.btn_delete.setObjectName("delete")
            self.btn_delete.clicked.connect(self._delete)
            btn_row.addWidget(self.btn_delete)

        btn_row.addStretch()

        self.btn_cancel = QPushButton("  Cancelar")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton("  Guardar")
        self.btn_ok.setObjectName("ok")
        self.btn_ok.clicked.connect(self._save)
        btn_row.addWidget(self.btn_ok)

        layout.addLayout(btn_row)

    def _test_connection(self):
        api_key = self.api_key_edit.text().strip()
        account_id = self.account_edit.text().strip()
        env = self.env_combo.currentText()

        if not api_key:
            QMessageBox.warning(self, "Error", "API Key requerida")
            return
        if not account_id:
            QMessageBox.warning(self, "Error", "Account ID requerido")
            return

        self.btn_test.setEnabled(False)
        self.btn_test.setText("  Probando...")
        from PyQt6.QtCore import QCoreApplication
        QCoreApplication.processEvents()

        try:
            import oandapyV20
            import oandapyV20.endpoints.accounts as accounts
            base = 'https://api-fxpractice.oanda.com' if env == 'practice' else 'https://api-fxtrade.oanda.com'
            client = oandapyV20.API(access_token=api_key, environment=env)
            client._base_url = base
            r = accounts.AccountSummary(account_id)
            client.request(r)
            alias = r.response.get('account', {}).get('alias', 'Unknown')
            balance = r.response.get('account', {}).get('balance', '?')
            QMessageBox.information(
                self, "Conexion OK",
                f"Conectado a OANDA {env}\nCuenta: {alias}\nBalance: ${balance}"
            )
        except ImportError:
            QMessageBox.critical(
                self, "Error",
                "oandapyV20 no instalado. Ejecuta: pip install oandapyV20"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error de conexion", str(e))
        finally:
            self.btn_test.setEnabled(True)
            self.btn_test.setText("  Test Connection")

    def _save(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Nombre requerido")
            return

        existing = load_connectors()
        for c in existing:
            if c.get('name') == name and name != self._edit_name:
                QMessageBox.warning(self, "Error",
                                    f"Ya existe un conector '{name}'")
                return

        self._config = {
            'type': self.type_combo.currentText().lower(),
            'name': name,
            'api_key': self.api_key_edit.text().strip(),
            'account_id': self.account_edit.text().strip(),
            'environment': self.env_combo.currentText(),
        }
        save_connector(self._config)
        self.accept()

    def _delete(self):
        r = QMessageBox.question(
            self, "Confirmar",
            f"Eliminar conector '{self._edit_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if r == QMessageBox.StandardButton.Yes:
            delete_connector(self._edit_name)
            self.accept()
